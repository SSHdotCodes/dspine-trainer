"""Read-only acceptance trend analysis and optional bounded pilot stop watcher.

Complete-round and opening metrics have different anchor populations. A rising
opening metric cannot compensate for stalled body acceptance. The watcher only
stops its exact trainer process after a clear stall/regression; a GO decision
still requires reviewing the bounded chunk's diagnostics.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def pooled(domains, block):
    values = [v[str(block)] for v in domains.values() if str(block) in v]
    count = sum(v["anchors"] for v in values)
    return sum(v["tau"] * v["anchors"] for v in values) / count if count else None


def analyze(records, blocks=(6, 8), minimum_step=3000, final=False):
    evaluations = [
        r for r in records if "offline_tau" in r and "offline_first_anchor" in r
    ]
    if not evaluations:
        return dict(decision="pending", reason="no paired evaluation yet")
    first = evaluations[0]
    latest = evaluations[-1]
    result = dict(step=latest["step"], decision="pending", blocks={})
    for b in blocks:
        tail = evaluations[-3:]
        body = sum(pooled(r["offline_tau"], b) for r in tail) / len(tail)
        opening = sum(pooled(r["offline_first_anchor"], b) for r in tail) / len(tail)
        baseline = pooled(first["offline_tau"], b)
        base_open = pooled(first["offline_first_anchor"], b)
        result["blocks"][str(b)] = dict(
            complete_tau=body,
            opening_tau=opening,
            complete_gain=body - baseline,
            opening_gain=opening - base_open,
            gap=opening - body,
            baseline_complete_tau=baseline,
            baseline_opening_tau=base_open,
        )
    if latest["step"] < minimum_step or len(evaluations) < 4:
        result["reason"] = (
            "collecting paired evaluations through the pilot grace period"
        )
        return result
    if any(v["complete_gain"] < 0.03 for v in result["blocks"].values()):
        result.update(
            decision="no_go",
            reason="complete-round acceptance failed to improve after the grace period",
        )
    elif any(v["opening_gain"] < -0.1 for v in result["blocks"].values()):
        result.update(
            decision="no_go", reason="opening acceptance regressed materially"
        )
    elif len(evaluations) >= 7 and any(
        result["blocks"][str(b)]["complete_tau"]
        < sum(pooled(r["offline_tau"], b) for r in evaluations[-6:-3]) / 3 - 0.1
        for b in blocks
    ):
        result.update(
            decision="no_go",
            reason="complete-round acceptance regressed materially from the previous evaluations",
        )
    else:
        result.update(
            decision="review" if final else "progressing",
            reason="body acceptance improved; review long-offset diagnostics and loss curves before continuation",
        )
    return result


def read_metrics(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # writer may be appending the final line
    return rows


def stop_own_trainer(output, config):
    marker = Path(output) / "trainer-process.json"
    if not marker.exists():
        return False
    pid = json.loads(marker.read_text())["pid"]
    proc = Path("/proc") / str(pid)
    if not (proc / "cmdline").exists():
        return False
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    if (
        b"dspine.train" not in argv
        or os.fsencode(str(Path(config).resolve())) not in argv
    ):
        raise ValueError("pilot guard refuses an unrelated or reused process ID")
    os.kill(pid, signal.SIGTERM)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--controller-unit")
    parser.add_argument("--controller-root")
    parser.add_argument("--minimum-step", type=int, default=3000)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_step = -1
    while True:
        result = analyze(
            read_metrics(output / "metrics.jsonl"),
            minimum_step=args.minimum_step,
            final=not args.watch,
        )
        if result.get("step") != last_step or not args.watch:
            temp = output / "pilot-quality.partial"
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(output / "pilot-quality.json")
            print(json.dumps(result), flush=True)
            last_step = result.get("step")
        if result["decision"] == "no_go":
            (output / "pilot-no-go.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            result["stop_requested"] = stop_own_trainer(output, args.config)
            if args.controller_unit and args.controller_root:
                status = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.controller_unit,
                        "-p",
                        "MainPID",
                        "--value",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                pid = int(status.stdout.strip() or "0")
                proc = Path("/proc") / str(pid) / "cmdline"
                if pid and proc.exists():
                    argv = proc.read_bytes().split(b"\0")
                    if (
                        b"dspine.rolling" not in argv
                        or os.fsencode(str(Path(args.controller_root).resolve()))
                        not in argv
                    ):
                        raise ValueError("pilot guard refuses an unrelated controller")
                    os.kill(pid, signal.SIGTERM)
                    result["controller_stop_requested"] = True
            print(json.dumps(result), flush=True)
            return
        if not args.watch or time.monotonic() - started >= args.max_hours * 3600:
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
