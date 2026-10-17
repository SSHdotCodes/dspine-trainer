"""Drive the isolated capture server in microbatches and record throughput.

The caller owns the GPU hold and server lifecycle. A successful /generate with
max_new_tokens=1 is one full teacher prefill plus an unused sampled token, never
an autoregressive continuation. Feature capture occurs before the one-token
sampler, and p is computed for all response tokens from the prefill logits.
"""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import torch

from .data import generation_paths, load_exclusion_ids, records, split_for_id
from .decode_probs import attach_decode_probs


def validate_completed_shard(path, row):
    payload = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    expected = torch.tensor(row["prompt_ids"] + row["output_ids"])[payload["start"] :]
    if (
        payload["id"] != row["id"]
        or payload["model_rev"] != row.get("model_rev", "unknown")
        or payload["sampling"] != row["sampling"]
        or not torch.equal(payload["input_ids"], expected)
    ):
        raise ValueError("existing capture shard differs from the requested generation")
    return len(payload["input_ids"])


def request(url, row, root, destination=None):
    ids = row["prompt_ids"] + row["output_ids"]
    key = hashlib.sha256(
        torch.tensor(ids, dtype=torch.long).numpy().tobytes()
    ).hexdigest()
    expected = (
        root / "shards" / (hashlib.sha256(row["id"].encode()).hexdigest() + ".pt")
    )
    final = (
        Path(destination) / "shards" / expected.name
        if destination is not None
        else expected
    )
    metadata = root / "inbox" / f"{key}.json"
    if final.exists():
        validate_completed_shard(final, row)
        attach_decode_probs(final, row)
        metadata.unlink(missing_ok=True)
        return {"tokens": 0, "bytes": 0, "skipped": True}
    if expected.exists():
        validate_completed_shard(expected, row)
        final.parent.mkdir(parents=True, exist_ok=True)
        expected.replace(final)
        attach_decode_probs(final, row)
        metadata.unlink(missing_ok=True)
        return {"tokens": 0, "bytes": 0, "skipped": True}
    temp = metadata.with_suffix(".tmp")
    temp.write_text(json.dumps(row))
    temp.replace(metadata)
    body = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
    }
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        json.load(response)
    if not expected.exists():
        raise RuntimeError("capture server returned without writing features")
    if final != expected:
        final.parent.mkdir(parents=True, exist_ok=True)
        expected.replace(final)
    attach_decode_probs(final, row)
    metadata.unlink()
    retained = validate_completed_shard(final, row)
    return {
        "tokens": len(ids),
        "retained_tokens": retained,
        "response_tokens": len(row["output_ids"]),
        "prompt_tokens": len(row["prompt_ids"]),
        "bytes": final.stat().st_size,
        "skipped": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True)
    p.add_argument("--root", required=True)
    p.add_argument(
        "--destination",
        help="move completed shards into a separate feature root on the same filesystem (e.g. held-out T0)",
    )
    p.add_argument("--url", default="http://127.0.0.1:30220")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--allow-greedy",
        action="store_true",
        help="separate validation-only greedy capture root",
    )
    p.add_argument(
        "--target-revision", default="c8f2fb1b9869f686b214782036123b10ff96d14a"
    )
    p.add_argument(
        "--train-response-tokens",
        type=int,
        default=0,
        help="stop after this many selected training response tokens, including resumed shards",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--exclude-ids", default=None)
    p.add_argument("--max-minutes", type=float, default=160)
    args = p.parse_args()
    root = Path(args.root)
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    paths = generation_paths(args.generations)
    begin = time.monotonic()
    long_contexts = max_prompt_tokens = 0
    tokens = retained_tokens = response_tokens = size = count = (
        selected_train_tokens
    ) = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = set()

        def collect(done):
            nonlocal tokens, retained_tokens, response_tokens, size, count
            nonlocal long_contexts, max_prompt_tokens
            for future in done:
                result = future.result()
                tokens += result["tokens"]
                retained_tokens += result.get("retained_tokens", 0)
                response_tokens += result.get("response_tokens", 0)
                prompt_tokens = result.get("prompt_tokens", 0)
                long_contexts += int(prompt_tokens >= 16384)
                max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
                size += result["bytes"]
                count += 1
                elapsed = time.monotonic() - begin
                print(
                    json.dumps(
                        {
                            "sequences": count,
                            "prefill_tokens": tokens,
                            "retained_tokens": retained_tokens,
                            "response_tokens": response_tokens,
                            "long_context_sequences_ge_16384": long_contexts,
                            "max_prompt_tokens": max_prompt_tokens,
                            "bytes": size,
                            "seconds": elapsed,
                            "tokens_per_second": tokens / elapsed,
                            "bytes_per_prefill_token": size / max(1, tokens),
                            "bytes_per_retained_token": size / max(1, retained_tokens),
                        }
                    ),
                    flush=True,
                )

        seen = set()
        for row in records(paths, load_exclusion_ids(args.exclude_ids)):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            if row["finish_reason"] == "abort" or len(row["output_ids"]) < 2:
                continue
            if row.get("model_rev") != args.target_revision:
                raise ValueError("generation target revision mismatch")
            if args.allow_greedy and row["sampling"]["temperature"] == 0:
                row["split"] = "validation"
            elif row["sampling"] != {"temperature": 1.0, "top_k": 20, "top_p": 0.95}:
                raise ValueError("generation sampling mismatch")
            if row.get("split", split_for_id(row["id"])) == "train":
                selected_train_tokens += len(row["output_ids"])
            pending.add(pool.submit(request, args.url, row, root, args.destination))
            if len(pending) >= args.concurrency:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                collect(done)
            if (
                (
                    args.train_response_tokens
                    and selected_train_tokens >= args.train_response_tokens
                )
                or (args.limit and count + len(pending) >= args.limit)
                or (time.monotonic() - begin) / 60 >= args.max_minutes
            ):
                break
        collect(pending)


if __name__ == "__main__":
    main()
