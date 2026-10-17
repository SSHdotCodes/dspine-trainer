"""Append validated generations without rewriting windows, policy or schedules.

A published merge SUMMARY must match the selected source platforms exactly.
Incoming records are independently contract-checked and deduplicated against
all existing windows. CPU-only; safe while an earlier window is training.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from .data import generation_paths, load_exclusion_ids, records, split_for_id
from .rolling import atomic_json, estimated_feature_bytes


def append_windows(root, generations, summary, platform_prefix=None, seal=False):
    root = Path(root)
    with (root / "append.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = json.loads((root / "plan.json").read_text())
        run = plan["run"]
        if summary.get("errors") != [] or not summary.get("validated_template"):
            raise ValueError(
                "append requires a successful template-validated merge summary"
            )
        if summary.get("duplicate_records", 0):
            raise ValueError("validated append source reports duplicate records")
        if sorted(load_exclusion_ids(run.get("exclude_ids"))) != plan["excluded_ids"]:
            raise ValueError("append exclusions differ from frozen plan")
        paths = generation_paths(generations)
        if platform_prefix:
            paths = [
                p for p in paths if any(x.startswith(platform_prefix) for x in p.parts)
            ]
        if not paths:
            raise ValueError("no validated append source parts")
        seen = {x for w in plan["windows"] for x in w["ids"]} | set(
            plan["validation"]["ids"]
        )
        transaction = Path(tempfile.mkdtemp(prefix="append-", dir=root))
        connection = sqlite3.connect(transaction / "index.sqlite3")
        connection.execute(
            "CREATE TABLE incoming(id TEXT PRIMARY KEY, ordering TEXT, body TEXT, bytes INTEGER)"
        )
        source_counts = {}
        try:
            for path in paths:
                platform = next(
                    (
                        x
                        for x in path.parts
                        if platform_prefix and x.startswith(platform_prefix)
                    ),
                    "total",
                )
                for row in records([path], set(plan["excluded_ids"])):
                    if (
                        row["sampling"]
                        != {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
                        or row["model_rev"] != run["target_revision"]
                    ):
                        raise ValueError(
                            "append target revision/sampler differs from frozen run"
                        )
                    # datagen.merge excludes aborts before its validated counts.
                    # Keep the source receipt comparable without admitting an
                    # aborted continuation into a training manifest.
                    if row["finish_reason"] == "abort":
                        continue
                    count = source_counts.setdefault(platform, Counter())
                    count["records"] += 1
                    count["tokens"] += len(row["output_ids"])
                    if (
                        row["id"] in seen
                        or split_for_id(row["id"]) != "train"
                        or len(row["output_ids"]) < 2
                    ):
                        continue
                    ordering = hashlib.sha256(
                        f"{run.get('seed', 42)}:{row['id']}".encode()
                    ).hexdigest()
                    connection.execute(
                        "INSERT OR IGNORE INTO incoming VALUES(?,?,?,?)",
                        (
                            row["id"],
                            ordering,
                            json.dumps(
                                {**row, "split": "train"}, separators=(",", ":")
                            ),
                            estimated_feature_bytes(row, run["model"]),
                        ),
                    )
            connection.commit()
            validated = summary.get("counts", {})
            expected_counts = (
                {
                    key.removeprefix("platform:"): value
                    for key, value in validated.items()
                    if key.startswith("platform:" + platform_prefix)
                }
                if platform_prefix
                else {"total": validated.get("total", {})}
            )
            # Check both directions: an absent source platform must not silently
            # disappear from a SUMMARY that validated records for that platform.
            for platform in source_counts.keys() | expected_counts.keys():
                actual = source_counts.get(platform, {})
                expected = expected_counts.get(platform, {})
                if any(
                    actual.get(k, 0) != expected.get(k, 0)
                    for k in ("records", "tokens")
                ):
                    raise ValueError(
                        "append files changed or are not covered by the validated summary"
                    )
            added = connection.execute("SELECT COUNT(*) FROM incoming").fetchone()[0]
            if added and not plan.get("append_open"):
                raise ValueError("append source is sealed")
            windows = []
            per_step = (
                run.get("gradient_accumulation", 4)
                if run.get("packed", True)
                else run["batch_size"]
                * run["gradient_accumulation"]
                / run["anchors_per_sequence"]
            )
            step = plan["windows"][-1]["stop_step"]
            current, size = [], 0

            def flush():
                nonlocal current, size, step
                if not current:
                    return
                name = f"window-{len(plan['windows']) + len(windows):05d}"
                directory = transaction / name
                directory.mkdir()
                domains = Counter()
                prefill = response = 0
                with (directory / "records.jsonl").open("w") as stream:
                    for _, body, _ in current:
                        stream.write(body + "\n")
                        row = json.loads(body)
                        domains[row["domain"]] += 1
                        response += len(row["output_ids"])
                        prefill += len(row["prompt_ids"]) + len(row["output_ids"])
                    stream.flush()
                    os.fsync(stream.fileno())
                stop = step + math.ceil(len(current) * run["epochs"] / per_step)
                if stop > run["total_steps"]:
                    raise ValueError(
                        "append exceeds the immutable optimizer-step safety limit"
                    )
                windows.append(
                    dict(
                        name=name,
                        input=str(directory),
                        ids=[x[0] for x in current],
                        estimated_bytes=size,
                        prefill_tokens=prefill,
                        response_tokens=response,
                        domain_records=dict(domains),
                        sequence_passes=run["epochs"],
                        capture_passes=1,
                        start_step=step,
                        stop_step=stop,
                    )
                )
                step = stop
                current = []
                size = 0

            for item in connection.execute("""SELECT id,body,bytes FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY json_extract(body,'$.domain') ORDER BY ordering)
                  *1.0/COUNT(*) OVER(PARTITION BY json_extract(body,'$.domain')) AS progress FROM incoming
                ) ORDER BY progress,ordering"""):
                if item[2] > plan["window_budget"]:
                    raise BufferError("one append record exceeds feature window")
                if current and size + item[2] > plan["window_budget"]:
                    flush()
                current.append(item)
                size += item[2]
            flush()
            for directory in [transaction, *[Path(w["input"]) for w in windows]]:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            plan["windows"].extend(windows)
            plan["training_records"] += added
            plan["append_open"] = bool(plan.get("append_open")) and not seal
            receipt = dict(
                added_records=added,
                added_windows=len(windows),
                source_counts=source_counts,
                summary_sha256=hashlib.sha256(
                    json.dumps(summary, sort_keys=True).encode()
                ).hexdigest(),
                proposal_policy_unchanged=True,
                global_schedule_unchanged=True,
                sealed=seal,
            )
            plan.setdefault("append_receipts", []).append(receipt)
            atomic_json(root / "plan.json", plan)
            atomic_json(transaction / "receipt.json", receipt)
            return plan
        finally:
            connection.close()
            (transaction / "index.sqlite3").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--validated-summary", required=True)
    parser.add_argument("--platform-prefix", default="modal-c2-")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    plan = append_windows(
        args.root,
        args.generations,
        json.loads(Path(args.validated_summary).read_text()),
        args.platform_prefix,
        args.seal,
    )
    print(json.dumps(plan["append_receipts"][-1]), flush=True)


if __name__ == "__main__":
    main()
