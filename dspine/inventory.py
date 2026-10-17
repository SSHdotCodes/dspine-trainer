"""CPU-only generation inventory and full-corpus feature/capture projections."""

import argparse
import datetime
import json
import math
from collections import Counter
from pathlib import Path

from .data import generation_paths, load_exclusion_ids, records


def inventory(
    root, response_budgets=(200_000_000, 250_000_000), window=4096, exclude_ids=None
):
    counts, domains = Counter(), Counter()
    seen, lengths = set(), []
    for row in records(generation_paths(root), load_exclusion_ids(exclude_ids)):
        if row["id"] in seen:
            counts["duplicates"] += 1
            continue
        seen.add(row["id"])
        if row["finish_reason"] == "abort" or len(row["output_ids"]) < 2:
            continue
        prompt, response = len(row["prompt_ids"]), len(row["output_ids"])
        counts["records"] += 1
        counts["response_tokens"] += response
        counts["prefill_tokens"] += prompt + response
        counts["retained_tokens"] += min(prompt, window) + response
        counts["long_context_ge_16384"] += int(prompt >= 16384)
        domains[row["domain"]] += 1
        lengths.append(prompt)
    if not lengths:
        raise ValueError("no eligible generation records")
    lengths.sort()
    projections = {}
    for budget in response_budgets:
        ratio = budget / counts["response_tokens"]
        retained = counts["retained_tokens"] * ratio
        projections[str(budget)] = {
            "sequences": math.ceil(counts["records"] * ratio),
            "prefill_tokens": math.ceil(counts["prefill_tokens"] * ratio),
            "retained_tokens": math.ceil(retained),
            "feature_bytes_core": math.ceil(retained * 12978),
            "windows_at_400GB": math.ceil(retained * 12978 / 400_000_000_000),
        }
    return {
        "as_of_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts": dict(counts),
        "domains": dict(domains),
        "prompt_token_percentiles": {
            str(p): lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))]
            for p in (0.5, 0.9, 0.99, 1.0)
        },
        "projections": projections,
        "exclusions_applied": True,
        "production_paths_only": True,
        "note": "Current-mixture projection. Core size assumes Flash Next FP8/primary top20; add measured tie/archive overhead. Refresh when the corpus freezes.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-ids", default=None)
    parser.add_argument("--response-budgets", default="200000000,250000000")
    args = parser.parse_args()
    result = inventory(args.generations, response_budgets=tuple(map(int,args.response_budgets.split(','))),
                       exclude_ids=args.exclude_ids)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
