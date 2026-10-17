"""Actual decode distributions for exact stored-trajectory acceptance eval.

Finite-precision prefill and decode can differ, even with identical weights.
Capture features with prefill, but use the raw top-logprobs returned by the same
decode call that sampled each held-out token. Never silently substitute a
recomputed prefill distribution for this on-policy probability.
"""

import hashlib
import math
import os
from pathlib import Path


def processed_decode_probs(top_logprobs, output_ids, top_k=20, top_p=0.95):
    if len(top_logprobs) != len(output_ids):
        raise ValueError("decode logprob/token count mismatch")
    all_ids, all_probs = [], []
    for entries, token in zip(top_logprobs, output_ids):
        entries = sorted(
            ((float(x[0]), int(x[1])) for x in entries), key=lambda x: (-x[0], x[1])
        )
        if len(entries) <= top_k or not all(math.isfinite(x[0]) for x in entries):
            raise ValueError("incomplete raw decode top-logprobs")
        cutoff = entries[top_k - 1][0]
        greater_mass = sum(math.exp(lp) for lp, _ in entries if lp > cutoff)
        if entries[-1][0] >= cutoff and greater_mass < top_p:
            raise ValueError("decode cutoff ties exceed returned logprob support")
        retained, mass, index = [], 0.0, 0
        while index < len(entries):
            end = index + 1
            while end < len(entries) and entries[end][0] == entries[index][0]:
                end += 1
            probability = math.exp(entries[index][0])
            if index < top_k and mass < top_p:
                retained.extend((entry[1], probability) for entry in entries[index:end])
            mass += (end - index) * probability
            index = end
        ids = [x[0] for x in retained]
        if token not in ids:
            raise ValueError(
                "actual decode token is outside reconstructed joint support"
            )
        normalizer = sum(x[1] for x in retained)
        all_ids.append(ids)
        all_probs.append([x[1] / normalizer for x in retained])
    return all_ids, all_probs


def attach_decode_probs(path, record):
    """Atomically attach actual p to an owned, completed feature shard.

    Called on the extraction client after capture; mmap keeps rewrite scratch
    bounded to one sequence. Features and tokens remain bit-identical.
    """
    if "target_decode_ids" not in record:
        return
    import torch

    path = Path(path)
    row = torch.load(path, weights_only=True, mmap=True)
    digest = hashlib.sha256(
        repr((record["target_decode_ids"], record["target_decode_probs"])).encode()
    ).hexdigest()
    if row.get("decode_distribution_sha256") == digest:
        return
    n, start = len(row["input_ids"]), row["response_start"]
    if n - start != len(record["target_decode_ids"]):
        raise ValueError("actual decode p length differs from captured response")
    width = max(20, max(map(len, record["target_decode_ids"])))
    ids = torch.zeros(n, width, dtype=torch.int32)
    probs = torch.zeros(n, width)
    for offset, (index, p) in enumerate(
        zip(record["target_decode_ids"], record["target_decode_probs"])
    ):
        ids[start + offset, : len(index)] = torch.tensor(index)
        probs[start + offset, : len(p)] = torch.tensor(p)
    if not torch.allclose(probs[start:].sum(-1), torch.ones(n - start), atol=1e-6):
        raise ValueError("actual decode probabilities must sum to one")
    row["target_ids"], row["target_probs"] = ids[:, :20].clone(), probs[:, :20].clone()
    row.pop("target_tie_ids", None)
    row.pop("target_tie_probs", None)
    if width > 20:
        row["target_tie_ids"], row["target_tie_probs"] = (
            ids[:, 20:].clone(),
            probs[:, 20:].clone(),
        )
    row["target_distribution_source"] = "actual_decode_raw_top_logprobs_joint"
    row["decode_distribution_sha256"] = digest
    temp = path.with_suffix(".decode-p.partial")
    torch.save(row, temp)
    os.replace(temp, path)
