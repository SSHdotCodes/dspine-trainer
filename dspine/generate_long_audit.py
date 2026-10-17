"""Bounded actual-decode diagnostic trajectories on an owned capture server."""

import argparse
import json
import time
from pathlib import Path

from .data import records
from .generate_eval import generate_one


def diagnostic_requests(groups, long_tokens=6144, control_tokens=512):
    """One deep short-prompt trajectory and one long-prompt control per split."""
    primary, controls = [], []
    for kind, rows in groups.items():
        short = [row for row in rows if len(row["prompt_ids"]) <= 4096] or rows
        row = max(short, key=lambda r: (len(r["output_ids"]), r["id"]))
        primary.append((kind, row, long_tokens))
        long = [r for r in rows if len(r["prompt_ids"]) > 4096 and r["id"] != row["id"]]
        if long:
            controls.append(
                (kind, max(long, key=lambda r: len(r["prompt_ids"])), control_tokens)
            )
    return primary + controls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="./work")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30220")
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--long-tokens", type=int, default=6144)
    p.add_argument("--control-tokens", type=int, default=512)
    p.add_argument("--minimum-tokens-per-second", type=float, default=25)
    args = p.parse_args()
    project = Path(args.project)
    root = project / "runs/audit-long"
    target = args.target
    eos = json.loads((target / "generation_config.json").read_text())["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    groups = {
        kind: list(records([root / "requests" / kind / "records.jsonl"]))
        for kind in ("heldout", "training_prompt")
    }
    started = time.monotonic()
    for index, (kind, row, cap) in enumerate(
        diagnostic_requests(groups, args.long_tokens, args.control_tokens)
    ):
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining < 45:
            print(json.dumps({"stage": "generation_budget_exhausted"}), flush=True)
            return
        limit = min(cap, max(1, int((remaining - 20) * args.minimum_tokens_per_second)))
        result = generate_one(
            args.url,
            row,
            row["prompt_ids"],
            1.0,
            root / "generations" / kind,
            row["model_rev"],
            limit,
            eos,
            record_decode_probs=True,
        )
        result.pop("id", None)
        print(
            json.dumps(
                {
                    "group": kind,
                    "index": index,
                    "prompt_tokens": len(row["prompt_ids"]),
                    **result,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
