"""Bounded real-feature pilot and honest wall-time sizing report (GPU hold required)."""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from .data import load_exclusion_ids


def feature_inventory(root, exclude_ids=None):
    excluded = load_exclusion_ids(exclude_ids)
    tokens = sequences = 0
    for path in Path(root).glob("*.pt"):
        # mmap avoids reading tens of GB just to count tokens.
        row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if row["id"] not in excluded and row["split"] == "train":
            sequences += 1
            tokens += len(row["input_ids"]) - row["response_start"]
    return sequences, tokens


def size_run(
    base,
    measurements,
    corpus_sequences,
    hours=38.0,
    min_epochs=1.5,
    capture_seconds=0.0,
):
    """Compare candidates using measured *steady-state* seconds per sequence.

    Includes measured eval/checkpoint overhead when present in elapsed deltas.
    Subtract a separately measured extraction estimate before allocating training
    time. Shared-GPU queue delays are outside the GPU-hour budget.
    """
    options = []
    for measurement in measurements:
        steps = measurement["metrics"]
        steady = steps[max(1, len(steps) // 5) :]
        if len(steady) < 5:
            continue
        seconds = statistics.median(x["seconds_per_step"] for x in steady)
        if (
            all("elapsed_seconds" in x for x in steady)
            and steady[-1]["step"] > steady[0]["step"]
        ):
            seconds = max(
                seconds,
                (steady[-1]["elapsed_seconds"] - steady[0]["elapsed_seconds"])
                / (steady[-1]["step"] - steady[0]["step"]),
            )
        cfg = measurement["config"]
        validation = [
            sum(row["selection_tau"][str(b)]["tau"] for b in (6, 8)) / 2
            for row in measurement.get("eval", [])
            if all(str(b) in row.get("selection_tau", {}) for b in (6, 8))
        ]
        sequences_per_step = (
            cfg["gradient_accumulation"]
            if cfg.get("packed")
            else cfg["batch_size"]
            * cfg["gradient_accumulation"]
            / cfg["anchors_per_sequence"]
        )
        seconds_per_epoch = corpus_sequences / sequences_per_step * seconds
        # Reserve 10% for checkpoint/eval variance beyond measured step times.
        epochs = max(0.0, hours * 3600 - capture_seconds) / (seconds_per_epoch * 1.1)
        options.append(
            {
                "layers": cfg["model"]["num_hidden_layers"],
                "anchors_per_sequence": cfg["anchors_per_sequence"],
                "seconds_per_step": seconds,
                "seconds_per_epoch": seconds_per_epoch,
                "sequences_per_step": sequences_per_step,
                "epochs_in_budget": epochs,
                "feasible": epochs >= min_epochs,
                "validation_tau": max(validation) if validation else None,
                "config": cfg,
                "peak_memory_bytes": max(x["peak_memory_bytes"] for x in steps),
                "peak_reserved_memory_bytes": max(
                    x.get("peak_reserved_memory_bytes", x["peak_memory_bytes"])
                    for x in steps
                ),
            }
        )
    feasible = [x for x in options if x["feasible"]]
    if not feasible:
        return {"status": "budget_not_met", "options": options, "config": None}
    # Compare the measured serving-block objective among candidates that fit.
    # Synthetic/timing-only callers can omit eval, retaining the depth prior.
    chosen = max(
        feasible,
        key=lambda x: (
            x["validation_tau"] if x["validation_tau"] is not None else -1,
            x["layers"],
            x["anchors_per_sequence"],
        ),
    )
    cfg = chosen["config"].copy()
    epochs = max(min_epochs, chosen["epochs_in_budget"])
    cfg["total_steps"] = math.ceil(
        corpus_sequences * epochs / chosen["sequences_per_step"]
    )
    cfg["output"] = base["output"]
    cfg["sizing_status"] = "measured_pilot"
    cfg["epochs"] = epochs
    cfg["eval_every"] = base.get("eval_every", 250)
    cfg["checkpoint_every"] = base.get("checkpoint_every", 100)
    return {
        "status": "sized",
        "hours": hours,
        "capture_seconds": capture_seconds,
        "training_budget_seconds": max(0.0, hours * 3600 - capture_seconds),
        "corpus_sequences": corpus_sequences,
        "epochs": epochs,
        "chosen": {k: v for k, v in chosen.items() if k != "config"},
        "options": options,
        "config": cfg,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--corpus-sequences", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--min-tokens", type=int, default=5_000_000)
    p.add_argument(
        "--capture-seconds",
        type=float,
        default=0,
        help="projected full-corpus extraction seconds from measured prefill throughput",
    )
    p.add_argument("--max-minutes-per-candidate", type=float, default=12)
    p.add_argument("--candidates", default="6:64,5:64,6:512,5:512")
    args = p.parse_args()
    base = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    count, tokens = feature_inventory(base["features"], base.get("exclude_ids"))
    if tokens < args.min_tokens:
        raise RuntimeError(
            f"pilot needs >= {args.min_tokens} response tokens, have {tokens}"
        )
    measurements = []
    for candidate in args.candidates.split(","):
        layers, anchors = map(int, candidate.split(":"))
        cfg = json.loads(json.dumps(base))
        cfg["model"]["num_hidden_layers"] = layers
        cfg["anchors_per_sequence"] = anchors
        cfg["total_steps"] = args.steps
        cfg["output"] = str(out / f"L{layers}-A{anchors}")
        cfg["eval_every"] = max(10, args.steps // 4)
        cfg["eval_anchors"] = base.get("eval_anchors", 64)
        cfg["eval_before_training"] = True
        cfg["eval_at_end"] = True
        cfg["eval_monte_carlo"] = True
        cfg["checkpoint_every"] = args.steps
        cfg["packed"] = True
        cfg["compile"] = False
        path = out / f"L{layers}-A{anchors}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dspine.train",
                "--config",
                str(path),
                "--max-minutes",
                str(args.max_minutes_per_candidate),
            ],
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        records = [
            json.loads(line)
            for line in (Path(cfg["output"]) / "metrics.jsonl").read_text().splitlines()
        ]
        metrics = [x for x in records if "seconds_per_step" in x]
        measurements.append(
            {
                "config": cfg,
                "metrics": metrics,
                "eval": [x for x in records if "offline_tau" in x],
            }
        )
        (out / "pilot-measurements.json").write_text(
            json.dumps(measurements, indent=2) + "\n"
        )
    report = size_run(
        base, measurements, args.corpus_sequences, capture_seconds=args.capture_seconds
    )
    (out / "sizing.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["config"]:
        (out / "full-run.json").write_text(
            json.dumps(report["config"], indent=2) + "\n"
        )
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("config", "options")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
