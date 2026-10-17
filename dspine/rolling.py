"""Bounded extraction/training windows with optimizer/RNG continuity.

Plan on CPU before taking a GPU hold. Execute the plan in disconnect-safe,
<=165-minute ai-job chunks. Source generations and window manifests are retained;
only a completed, checkpointed training window's owned feature files are pruned.
"""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .data import (
    FEATURE_VERSION,
    generation_paths,
    load_exclusion_ids,
    records,
    split_for_id,
)

STOP = False
CHILD = None


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def estimated_feature_bytes(row, model):
    retained = min(len(row["prompt_ids"]), model.get("context_window", 4096)) + len(
        row["output_ids"]
    )
    layers = len(model["target_layer_ids"])
    hidden = model["target_hidden_size"]
    # Reserve 16 extra sparse entries for cutoff ties. Actual writes still enforce
    # the byte limit; the real pilot establishes whether a wider margin is needed.
    return retained * (layers * hidden + 2 * layers + 36 * 8 + 8) + 16384


def feature_bytes(root, exclude=()):
    root = Path(root)
    excluded = [Path(p).resolve() for p in exclude]
    return sum(
        p.stat().st_size
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".pt", ".partial")
        and not any(p.resolve().is_relative_to(x) for x in excluded)
    )


def source_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepare_plan(
    base,
    generations,
    root,
    window_bytes=400_000_000_000,
    validation_records=128,
    validation_bytes=40_000_000_000,
    max_feature_bytes=450_000_000_000,
):
    """Disk-backed deterministic prompt shuffle; RAM is bounded to one window."""
    base = json.loads(json.dumps(base))
    source_hash = source_fingerprint(base)
    excluded = load_exclusion_ids(base.get("exclude_ids"))
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["source_config_sha256"] != source_hash:
            raise ValueError("rolling resume requires the unchanged source run config")
        if plan["excluded_ids"] != sorted(excluded):
            raise ValueError(
                "exclusions changed; rebuild the rolling plan before training"
            )
        return plan
    feature_root = Path(
        base.get("feature_budget_root", Path(base["features"]).parent)
    ).resolve()
    train_root = feature_root / "rolling-train"
    external_validation = base.get("eval_generations")
    if base.get("eval_features") and not external_validation:
        raise ValueError(
            "eval_generations is required to regenerate an external validation cache"
        )
    val_root = (
        Path(base["eval_features"]).resolve().parent
        if external_validation
        else feature_root / "rolling-validation"
    )
    heldout = []
    if external_validation:
        if not val_root.is_relative_to(feature_root):
            raise ValueError(
                "external validation features must be inside the feature budget root"
            )
        heldout = [
            row
            for row in records(generation_paths(external_validation), excluded)
            if row["finish_reason"] != "abort" and len(row["output_ids"]) >= 2
        ]
        heldout.sort(
            key=lambda row: hashlib.sha256(
                f"{base.get('seed', 42)}:{row['id']}".encode()
            ).digest()
        )
        heldout = heldout[:validation_records]
        if any(
            row["model_rev"] != base["target_revision"]
            or row["sampling"] != {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
            for row in heldout
        ):
            raise ValueError(
                "held-out generation revision/sampler differs from the run config"
            )
    heldout_ids = {row["id"] for row in heldout}
    if any(
        (p / "shards").exists() and any((p / "shards").glob("*.pt"))
        for p in ((train_root,) if external_validation else (train_root, val_root))
    ):
        raise ValueError("unplanned rolling feature directories already contain shards")
    baseline = feature_bytes(feature_root, exclude=(train_root, val_root))
    safety = min(4_000_000_000, max_feature_bytes // 100)
    allocation = min(
        window_bytes, max_feature_bytes - baseline - validation_bytes - safety
    )
    if allocation <= 0:
        raise BufferError("existing features leave no rolling-window budget")
    connection = sqlite3.connect(root / "corpus.build.sqlite3")
    connection.execute("DROP TABLE IF EXISTS records")
    connection.execute(
        "CREATE TABLE records(id TEXT PRIMARY KEY, ordering TEXT, split TEXT, bytes INTEGER, body TEXT)"
    )
    paths = [path for path in generation_paths(generations)
             if not any(part.startswith(prefix) for part in path.parts
                        for prefix in base.get("deferred_platform_prefixes", []))]
    proposal = base.get("proposal_vocabulary", {})
    fit_proposal = proposal.get("fit_from_frozen_generations", False)
    proposal_counts = defaultdict(Counter)
    for row in records(paths, excluded):
        if (
            row["id"] in heldout_ids
            or row["finish_reason"] == "abort"
            or len(row["output_ids"]) < 2
        ):
            continue
        if (
            row["sampling"] != {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
            or row["model_rev"] != base["target_revision"]
        ):
            raise ValueError("generation revision/sampler differs from the run config")
        order = hashlib.sha256(
            f"{base.get('seed', 42)}:{row['id']}".encode()
        ).hexdigest()
        split = split_for_id(row["id"])
        inserted = connection.execute(
            "INSERT OR IGNORE INTO records VALUES(?,?,?,?,?)",
            (
                row["id"],
                order,
                split,
                estimated_feature_bytes(row, base["model"]),
                json.dumps(row, separators=(",", ":")),
            ),
        )
        if fit_proposal and inserted.rowcount and split == "train":
            proposal_counts[row["domain"]].update(row["output_ids"])
    connection.commit()
    proposal_receipt = None
    if fit_proposal:
        from .hot_vocab import expand_from_counts, load_hot_ids

        vocab = base["model"]["vocab_size"]
        initial = load_hot_ids(proposal.get("base_path") or proposal["path"], vocab)
        fitted, masses = expand_from_counts(proposal_counts, initial, vocab,
                                            proposal.get("minimum_domain_coverage", .9995))
        digest = hashlib.sha256(fitted.numpy().tobytes()).hexdigest()
        fitted_path = root / f"proposal-vocabulary-{digest[:16]}.json"
        atomic_json(fitted_path, fitted.tolist())
        base["proposal_vocabulary"] = {**proposal, "path": str(fitted_path),
            "sorted_ids_sha256": digest, "fit_from_frozen_generations": False}
        proposal_receipt = dict(vocab_size=len(fitted), training_label_coverage=masses,
            sorted_ids_sha256=digest, source="deduplicated_frozen_training_labels_only")

    def manifest(name, rows):
        directory = root / "manifests" / name
        directory.mkdir(parents=True, exist_ok=True)
        ids = []
        domain_records = {}
        estimate = prefill_tokens = response_tokens = 0
        with (directory / "records.jsonl").open("w") as stream:
            for record_id, size, body in rows:
                stream.write(body + "\n")
                ids.append(record_id)
                estimate += size
                record = json.loads(body)
                domain = record["domain"]
                domain_records[domain] = domain_records.get(domain, 0) + 1
                prefill_tokens += len(record["prompt_ids"]) + len(record["output_ids"])
                response_tokens += len(record["output_ids"])
        return {
            "name": name,
            "input": str(directory),
            "ids": ids,
            "estimated_bytes": estimate,
            "prefill_tokens": prefill_tokens,
            "response_tokens": response_tokens,
            "domain_records": domain_records,
        }

    val = manifest(
        "validation",
        [
            (
                row["id"],
                estimated_feature_bytes(row, base["model"]),
                json.dumps({**row, "split": "validation"}, separators=(",", ":")),
            )
            for row in heldout
        ]
        if external_validation
        else connection.execute(
            'SELECT id,bytes,body FROM records WHERE split="validation" ORDER BY ordering LIMIT ?',
            (validation_records,),
        ),
    )
    if not val["ids"] or val["estimated_bytes"] > validation_bytes:
        raise ValueError("validation records are missing or exceed their cache budget")
    windows = []
    current = []
    size = 0
    for item in connection.execute(
        '''SELECT id,bytes,body FROM (
             SELECT *, ROW_NUMBER() OVER (
               PARTITION BY json_extract(body,'$.domain') ORDER BY ordering
             ) * 1.0 / COUNT(*) OVER (
               PARTITION BY json_extract(body,'$.domain')
             ) AS domain_progress FROM records WHERE split="train"
           ) ORDER BY domain_progress,ordering'''
    ):
        if item[1] > allocation:
            raise BufferError("one feature record exceeds the rolling window budget")
        if current and size + item[1] > allocation:
            windows.append(manifest(f"window-{len(windows):05d}", current))
            current = []
            size = 0
        current.append(item)
        size += item[1]
    if current:
        windows.append(manifest(f"window-{len(windows):05d}", current))
    connection.close()
    if not windows:
        raise ValueError("no eligible training generations")
    epochs = base.get("epochs", 1.5)
    if epochs < 1.5:
        raise ValueError("full rolling run must cover at least 1.5 epochs")
    per_step = (
        base.get("gradient_accumulation", 4)
        if base.get("packed", True)
        else base["batch_size"]
        * base["gradient_accumulation"]
        / base["anchors_per_sequence"]
    )
    sizing_receipt = None
    if base.get("budget_sizing") and not base.get("append_open", False):
        measured = base["budget_sizing"]
        seconds_per_step = measured["seconds_per_step"]
        if proposal_receipt and measured.get("proposal_vocab_size"):
            # New corpus languages can widen the frozen map after benchmarking.
            # Scaling the entire step cost is conservative; only head cost grows.
            seconds_per_step *= max(1., proposal_receipt["vocab_size"] / measured["proposal_vocab_size"])
        rate = measured["capture_tokens_per_second"]
        if min(seconds_per_step, rate) <= 0:
            raise ValueError("budget sizing requires positive measured rates")
        sequences = sum(len(w["ids"]) for w in windows)
        tokens = sum(w["prefill_tokens"] for w in windows) + val["prefill_tokens"]
        # A hold can end partway through a capture window. Budget its restart
        # explicitly instead of treating every window as one uninterrupted load.
        startups = (len(windows) + 1) * measured.get("capture_startups_per_window", 1)
        extraction_seconds = tokens / rate + startups * measured["capture_startup_seconds"]
        budget = measured.get("hours", 38.) * 3600
        multiplier = measured.get("training_overhead_multiplier", 1.1)
        epochs = (budget - extraction_seconds) / (sequences / per_step * seconds_per_step * multiplier)
        if epochs < max(1.5, measured.get("min_epochs", 1.5)):
            raise ValueError("frozen corpus cannot fit1.5 epochs in the measured time budget")
        sizing_receipt = dict(training_sequences=sequences, prefill_tokens=tokens,
                              capture_seconds=extraction_seconds, epochs=epochs,
                              target_hours=budget / 3600, measured=measured,
                              effective_seconds_per_step=seconds_per_step)
    step = 0
    for window in windows:
        window["start_step"] = step
        window["sequence_passes"] = epochs
        window["capture_passes"] = 1
        step += math.ceil(len(window["ids"]) * epochs / per_step)
        window["stop_step"] = step
    run = {
        **base,
        "features": str(train_root / "shards"),
        "eval_features": str(val_root / "shards"),
        "total_steps": base.get("total_steps", 1_000_000) if base.get("append_open") else step,
        "epochs": epochs,
    }
    if sizing_receipt:
        run["budget_sizing_receipt"] = sizing_receipt
    plan = {
        "version": 1,
        "append_open": base.get("append_open", False),
        "source_config_sha256": source_hash,
        "excluded_ids": sorted(excluded),
        "feature_root": str(feature_root),
        "train_root": str(train_root),
        "validation_root": str(val_root),
        "window_budget": allocation,
        "validation_budget": validation_bytes,
        "feature_budget": max_feature_bytes,
        "baseline_feature_bytes": baseline,
        "feature_reuse_policy": "extract_once_then_consecutive_window_passes",
        "window_mixture_policy": "stratified_domain_proportions_with_seeded_shuffle_within_domain",
        "validation": val,
        "windows": windows,
        "training_records": sum(len(w["ids"]) for w in windows),
        "run": run,
    }
    if proposal_receipt:
        plan["proposal_vocabulary_receipt"] = proposal_receipt
    atomic_json(root / "run.json", run)
    atomic_json(path, plan)
    # The frozen manifests retain every selected generation; the build index is disposable.
    (root / "corpus.build.sqlite3").unlink()
    return plan


def feature_path(root, record_id):
    return (
        Path(root) / "shards" / (hashlib.sha256(record_id.encode()).hexdigest() + ".pt")
    )


def window_ready(root, window, run):
    for record_id in window["ids"]:
        path = feature_path(root, record_id)
        if not path.exists():
            return False
        row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (
            row["id"] != record_id
            or row["version"] != FEATURE_VERSION
            or row["model_rev"] != run["target_revision"]
            or tuple(row["layer_ids"]) != tuple(run["model"]["target_layer_ids"])
        ):
            raise ValueError(
                "feature shard contract differs from the frozen rolling plan"
            )
    return True


def latest_state(output):
    output = Path(output)
    pointer = output / "latest"
    if not pointer.exists():
        return None
    checkpoint = output / pointer.read_text().strip()
    state = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=False, mmap=True
    )
    return {
        "step": state["step"],
        "data_window": state.get("data_window"),
        "path": str(checkpoint),
    }


def purge_window(root, window):
    """Delete only manifest-owned cached features after the durable step receipt."""
    root = Path(root)
    expected = {
        hashlib.sha256(record_id.encode()).hexdigest() for record_id in window["ids"]
    }
    shards = root / "shards"
    owned = [p for p in shards.glob("*") if p.suffix in (".pt", ".partial")]
    if any(path.stem not in expected for path in owned):
        raise ValueError("refusing to prune a shard outside this rolling window")
    for path in owned:
        path.unlink()
    (shards / "manifest.jsonl").unlink(missing_ok=True)
    for path in (root / "inbox").glob("*.json"):
        if json.loads(path.read_text())["id"] in window["ids"]:
            path.unlink()


def request_stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def child(command, env=None):
    global CHILD
    CHILD = subprocess.Popen(
        command,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
        start_new_session=True,
    )
    try:
        return CHILD.wait()
    finally:
        CHILD = None


def capture_window(plan, window, destination, root, seconds):
    """Own and reconcile a unique container CID; never touch production services."""
    if plan["run"].get("device", "cuda") == "cuda":
        # Target extraction needs nearly the entire GPU. Owner-authorized
        # training co-runs must finish before the next target load.
        processes = subprocess.run(
            ["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
            check=True,capture_output=True,text=True).stdout.strip()
        if processes:
            print(json.dumps({"capture_waiting_for_gpu_processes":processes.splitlines(),
                              "resumable":True}),flush=True)
            return
    logs = Path(root) / "capture-runs"
    logs.mkdir(exist_ok=True)
    state_dir = Path(tempfile.mkdtemp(prefix="capture-", dir=logs))
    allocation = (
        plan["validation_budget"]
        if window["name"] == "validation"
        else plan["window_budget"]
    )
    outside = feature_bytes(plan["feature_root"], exclude=(destination,))
    available = min(
        allocation,
        plan["feature_budget"]
        - outside
        - min(4_000_000_000, plan["feature_budget"] // 100),
    )
    if available <= 0:
        raise BufferError("other feature caches fill the global feature budget")
    run = plan["run"]
    env = {
        **os.environ,
        "DSPINE_FEATURE_ROOT": str(destination),
        "DSPINE_CAPTURE_GENERATIONS": window["input"],
        "DSPINE_FEATURE_BUDGET": str(available),
        "DSPINE_FEATURE_BUDGET_ROOT": plan["feature_root"],
        "DSPINE_GLOBAL_FEATURE_BUDGET": str(
            plan["feature_budget"] - min(1_000_000_000, plan["feature_budget"] // 100)
        ),
        "DSPINE_CAPTURE_JOB_SECONDS": str(max(120, int(seconds))),
        "DSPINE_CAPTURE_STATE_DIR": str(state_dir),
        "DSPINE_TARGET_PATH": run["target"],
        "DSPINE_TARGET_REVISION": run["target_revision"],
        **(
            {"DSPINE_EXCLUDE_IDS": run["exclude_ids"]} if run.get("exclude_ids") else {}
        ),
        "DSPINE_CAPTURE_LAYER_IDS": ",".join(
            map(str, run["model"]["target_layer_ids"])
        ),
        "DSPINE_CONTEXT_WINDOW": str(run["model"]["context_window"]),
    }
    script = Path(__file__).resolve().parent / "scripts/capture_job.sh"
    try:
        result = child(["bash", str(script), "0", "1"], env)
        if result not in (0, 124, 143):
            raise RuntimeError(f"capture failed with exit {result}; see {state_dir}")
    finally:
        cid_file = state_dir / "container.cid"
        if cid_file.exists():
            cid = cid_file.read_text().strip()
            if not re.fullmatch("[0-9a-f]{64}", cid):
                raise ValueError("invalid owned capture container id")
            inspected = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "codex.task"}}',
                    cid,
                ],
                capture_output=True,
                text=True,
            )
            if inspected.returncode == 0:
                if inspected.stdout.strip() != "feature-capture":
                    raise ValueError("capture CID ownership label mismatch")
                subprocess.run(
                    ["docker", "stop", "-t", "30", cid],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )


def execute_plan(plan, root, max_minutes=165, capture=capture_window, max_actions=None):
    """One GPU-hold chunk; every phase/rotation is resumable and reviewable."""
    root = Path(root)
    path = root / "state.json"
    run = plan["run"]
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"phase": "validation", "window": 0, "complete": False}
    )
    deadline = time.monotonic() + max_minutes * 60
    actions = 0
    while (
        not STOP
        and not state["complete"]
        and (max_actions is None or actions < max_actions)
    ):
        remaining = deadline - time.monotonic()
        if (Path(run["output"]) / "pilot-no-go.json").exists():
            print(json.dumps({"pilot_no_go":True,"resumable":True}),flush=True)
            break
        if remaining < 120:
            break
        published = root / "plan.json"
        if published.exists():
            updated = json.loads(published.read_text())
            if (updated["run"] != run or updated["windows"][:len(plan["windows"])] != plan["windows"]
                    or updated["source_config_sha256"] != plan["source_config_sha256"]):
                raise ValueError("only immutable-window append is allowed during execution")
            plan = updated
        phase = state["phase"]
        if phase == "awaiting_data":
            if state["window"] < len(plan["windows"]):
                state["phase"] = "capture"
                continue
            if not plan.get("append_open"):
                state.update(complete=True,phase="complete")
            break
        window = plan["windows"][state["window"]]
        print(
            json.dumps(
                {
                    "rolling_phase": phase,
                    "window": state["window"],
                    "remaining_seconds": remaining,
                }
            ),
            flush=True,
        )
        if phase == "validation":
            if not window_ready(plan["validation_root"], plan["validation"], run):
                capture(
                    plan,
                    plan["validation"],
                    plan["validation_root"],
                    root,
                    min(run.get("capture_max_seconds", 9900), remaining - 60),
                )
                actions += 1
            if not window_ready(plan["validation_root"], plan["validation"], run):
                break
            state["phase"] = "capture"
        elif phase == "capture":
            if not window_ready(plan["train_root"], window, run):
                capture(
                    plan, window, plan["train_root"], root,
                    min(run.get("capture_max_seconds", 9900), remaining - 60),
                )
                actions += 1
            if not window_ready(plan["train_root"], window, run):
                break
            state["phase"] = "train"
        elif phase == "train":
            checkpoint = latest_state(run["output"])
            if checkpoint is None or checkpoint["step"] < window["stop_step"]:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "dspine.train",
                    "--config",
                    str(root / "run.json"),
                    "--resume",
                    "latest",
                    "--data-window",
                    window["name"],
                    "--stop-at-step",
                    str(window["stop_step"]),
                    "--max-minutes",
                    str(max(0.01, (remaining - 60) / 60)),
                ]
                if checkpoint and checkpoint["data_window"] != window["name"]:
                    command.append("--reset-data")
                result = child(command)
                actions += 1
                if result:
                    raise RuntimeError(
                        f"training exited with {result}; last checkpoint/features retained"
                    )
            checkpoint = latest_state(run["output"])
            if checkpoint is None or checkpoint["step"] < window["stop_step"]:
                break
            state["phase"] = "rotate"
        elif phase == "rotate":
            checkpoint = latest_state(run["output"])
            if (
                not checkpoint
                or checkpoint["data_window"] != window["name"]
                or checkpoint["step"] < window["stop_step"]
            ):
                raise ValueError(
                    "refusing feature pruning without a matching durable window checkpoint"
                )
            atomic_json(
                root / "receipts" / f"{window['name']}.json",
                {"window": window, "checkpoint": checkpoint},
            )
            purge_window(plan["train_root"], window)
            actions += 1
            state["window"] += 1
            if state["window"] == len(plan["windows"]):
                state["complete"] = not plan.get("append_open", False)
                state["phase"] = "complete" if state["complete"] else "awaiting_data"
            else:
                state["phase"] = "capture"
        else:
            raise ValueError("unknown rolling phase")
        atomic_json(path, state)
    atomic_json(path, state)
    print(
        json.dumps({"rolling_state": state, "resumable": not state["complete"]}),
        flush=True,
    )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=165)
    args = parser.parse_args()
    base = json.loads(Path(args.config).read_text())
    if base.get("sizing_status") != "measured_pilot":
        raise ValueError("full rolling run requires a measured pilot config")
    plan = prepare_plan(base, args.generations, args.root)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "windows": len(plan["windows"]),
                    "training_records": plan["training_records"],
                    "total_steps": plan["run"]["total_steps"],
                    "window_budget": plan["window_budget"],
                }
            ),
            flush=True,
        )
        return
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    execute_plan(plan, args.root, args.max_minutes)


if __name__ == "__main__":
    main()
