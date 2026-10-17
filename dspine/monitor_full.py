"""Read-only, low-noise event stream for a bounded rolling-training chunk."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from .quality_gate import pooled, read_metrics


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail_json(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except FileNotFoundError:
        pass
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--unit", default="ai-dspine-rolling.service")
    p.add_argument("--max-hours", type=float, default=4)
    args = p.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    initial = read_json(root / "state.json")
    phase = (initial.get("phase"), initial.get("window"))
    rows = read_metrics(output / "metrics.jsonl")
    last_eval = max((r["step"] for r in rows if "offline_tau" in r), default=-1)
    started = last_progress = time.monotonic()
    while time.monotonic() - started < args.max_hours * 3600:
        state = read_json(root / "state.json")
        current = (state.get("phase"), state.get("window"))
        if current != phase:
            print(
                json.dumps(
                    {
                        "event": "phase_changed",
                        "at": time.time(),
                        "phase": current[0],
                        "window": current[1],
                    }
                ),
                flush=True,
            )
            phase = current
        for row in read_metrics(output / "metrics.jsonl"):
            if "offline_tau" not in row or row["step"] <= last_eval:
                continue
            result = {
                "event": "paired_evaluation",
                "at": time.time(),
                "step": row["step"],
            }
            result["complete_tau"] = {
                str(b): pooled(row["offline_tau"], b) for b in (6, 8)
            }
            if "offline_first_anchor" in row:
                result["opening_tau"] = {
                    str(b): pooled(row["offline_first_anchor"], b) for b in (6, 8)
                }
            print(json.dumps(result), flush=True)
            last_eval = row["step"]
        guard = read_json(output / "pilot-no-go.json")
        if guard:
            print(json.dumps({"event": "pilot_no_go", **guard}), flush=True)
            return
        status = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", args.unit]
        )
        if status.returncode:
            print(
                json.dumps(
                    {
                        "event": "chunk_finished",
                        "at": time.time(),
                        "state": state,
                        "quality": read_json(output / "pilot-quality.json"),
                    }
                ),
                flush=True,
            )
            return
        if time.monotonic() - last_progress >= 600:
            paths = list(
                (root / "capture-runs").glob("capture-*/extraction-metrics.jsonl")
            )
            latest = max(paths, key=lambda x: x.stat().st_mtime) if paths else None
            progress = tail_json(latest) if latest else {}
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "at": time.time(),
                        "phase": phase,
                        "capture": {
                            k: progress[k]
                            for k in (
                                "sequences",
                                "prefill_tokens",
                                "tokens_per_second",
                                "bytes",
                            )
                            if k in progress
                        },
                    }
                ),
                flush=True,
            )
            last_progress = time.monotonic()
        time.sleep(10)


if __name__ == "__main__":
    main()
