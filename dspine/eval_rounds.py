"""Walk complete speculative rounds on held-out target trajectories.

For Y~p, reverse acceptance A|Y~Bernoulli(min(1,q(Y)/p(Y))) has joint
P(A=1,Y)=min(p,q) and P(A=0,Y)=(p-q)+. These are exactly the accepted-draft
and rejected/residual-token laws of ordinary speculative sampling. Consequently
we may advance the next anchor by the sampled round length along the stored
target trajectory. Decisions depend only on the emitted prefix, so composition
across rounds remains exact. No p from a different proposed prefix is used.

This Monte Carlo evaluator complements the lower-cost first-anchor estimates
used during training. T=0 walks the real greedy continuation directly.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from .data import AnchorStream, dequantize_features, pad_support, target_distribution
from .eval_offline import greedy_acceptance, path_acceptance
from .modeling import DSpineDraft, TransitionCache
from .transfer_space import load_target_weights


def domain_balanced_paths(paths, seed=42):
    """Round-robin domains before any bounded prefix of the eval prompt set."""
    groups = defaultdict(list)
    for path in paths:
        row = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        key = hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()
        groups[row["domain"]].append((key, path))
    for values in groups.values():
        values.sort()
    ordered = []
    for index in range(max(map(len, groups.values()), default=0)):
        ordered.extend(
            groups[domain][index][1]
            for domain in sorted(groups)
            if index < len(groups[domain])
        )
    return ordered


def emitted_lengths(survival, labels, eos_ids):
    """Accepted prefix plus one residual/bonus, except after an accepted EOS."""
    accepted = survival.long()
    eos = torch.zeros_like(labels, dtype=torch.bool)
    for token in eos_ids:
        eos |= labels == token
    accepted_eos = (accepted.bool() & eos).any(-1)
    return accepted.sum(-1) + (~accepted_eos).long()


def offset_bucket(offset):
    if offset < 256:
        return "0-256"
    if offset < 1024:
        return "256-1K"
    if offset < 4096:
        return "1K-4K"
    return ">=4K"


def attach_diagnostics(result, breakdown, first_round):
    for group in breakdown.values():
        for item in group.values():
            item["tau"] = item["tokens"] / item["rounds"]
    for item in first_round.values():
        item["rao_blackwell_tau"] = item["expected_tokens"] / item["anchors"]
        item["sampled_tau"] = item["sampled_tokens"] / item["anchors"]
    result.update(by_anchor=breakdown, first_round=first_round)
    return result


class PackedReplay:
    """Lazy, bounded chunks of independent anchor caches, reused across blocks.

    Future anchors are independent masked blocks, not teacher-forced inputs to
    earlier anchors. Only target features strictly before each anchor are visible.
    q/p walking below is identical to the serial path. At FP32 the two paths are
    numerically equivalent; ordinary BF16 GEMM/attention rounding can differ.
    """

    def __init__(self, model, row, chunk_size=64, max_block=None):
        if chunk_size < 1:
            raise ValueError("packed replay chunk must be positive")
        self.model, self.row, self.chunk_size = model, row, chunk_size
        self.max_block = model.config.block_size if max_block is None else max_block
        if not 2 <= self.max_block <= model.config.block_size:
            raise ValueError("packed replay block is outside the draft architecture")
        device = next(model.parameters()).device
        self.features = dequantize_features(row,device=device,dtype=model.embedding.weight.dtype)
        self.ids = row['input_ids'].to(device)
        self.caches = {}

    @torch.inference_mode()
    def get(self, anchor, block):
        index = (anchor-self.row['response_start']) // self.chunk_size
        first = self.row['response_start'] + index*self.chunk_size
        if index not in self.caches:
            offsets = torch.arange(first,min(first+self.chunk_size,len(self.ids)),device=self.ids.device)
            out = self.model.forward_packed(self.features,self.ids,offsets,self.row['start'],
                                            return_layer_features=False)
            # Causality makes later block rows irrelevant. Avoid projecting them
            # through the large vocabulary head when evaluating only b6/b8.
            out = replace(out, hidden=out.hidden[:, :self.max_block],
                          pre_injection=out.pre_injection[:, :self.max_block])
            self.caches[index] = self.model.build_transition_cache(out)
        cache = self.caches[index]
        local = anchor-first
        return TransitionCache(cache.candidate_ids[local:local+1,:block-1],
            cache.initial_scores[local:local+1,:block-1], cache.transitions[local:local+1,:block-2])


@torch.inference_mode()
def evaluate_rounds(
    model,
    feature_root,
    blocks=(6, 8),
    temperature=1.0,
    sequences=64,
    seed=42,
    trials=1,
    max_seconds=None,
    exclude_ids=None,
    expected_revision=None,
    require_decode_probs=False,
    split="validation",
    collect_diagnostics=False,
    progress_callback=None,
    offsets=(0,),
    window_tokens=None,
    packed_chunk_size=0,
):
    if any(b < 2 or b > model.config.block_size for b in blocks):
        raise ValueError("evaluation block is outside the draft architecture")
    if not offsets or min(offsets) < 0 or (window_tokens is not None and window_tokens < 1):
        raise ValueError("invalid fixed-prefix walk windows")
    if packed_chunk_size < 0:
        raise ValueError("packed replay chunk must be nonnegative")
    device = next(model.parameters()).device
    stream = AnchorStream(
        feature_root,
        split=split,
        anchor_policy="first",
        expected_layer_ids=model.config.target_layer_ids,
        expected_revision=expected_revision,
        exclude_ids=exclude_ids,
    )
    started = time.monotonic()
    training = model.training
    model.eval()
    domains = {}
    seen = set()
    breakdown = {}
    first_round = {}
    last_progress = started
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for path in domain_balanced_paths(stream.paths, seed):
            if len(seen) >= sequences:
                break
            row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if (
                row["id"] in seen
                or row["id"] in stream.excluded
                or row["split"] != split
            ):
                continue
            stream.validate_contract(row)
            if (
                temperature > 0
                and require_decode_probs
                and row.get("target_distribution_source")
                != "actual_decode_raw_top_logprobs_joint"
            ):
                raise ValueError(
                    "exact production eval requires actual decode probabilities"
                )
            if temperature == 0:
                if row["sampling"]["temperature"] != 0:
                    raise ValueError(
                        "T=0 round evaluation requires greedy target trajectories"
                    )
            elif temperature != 1 or row["sampling"] != {
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
            }:
                raise ValueError(
                    "T=1 round evaluation requires production on-policy trajectories"
                )
            seen.add(row["id"])
            ids = row["input_ids"]
            target_ids, target_probs = target_distribution(row)
            replay = PackedReplay(model,row,packed_chunk_size,max(blocks)) if packed_chunk_size else None
            totals = domains.setdefault(row["domain"], {})
            for block in blocks:
                stats = totals.setdefault(
                    str(block),
                    {"tokens": 0, "rounds": 0, "sequences": 0, "censored_tails": 0},
                )
                stats["sequences"] += 1
                for _ in range(trials):
                    for begin_offset in offsets:
                        anchor = row["response_start"] + begin_offset
                        window_end = anchor + window_tokens if window_tokens is not None else len(ids)
                        while (
                            anchor < len(ids)
                            and anchor < window_end
                            and ids[anchor].item() not in model.config.eos_token_ids
                        ):
                            if (
                                max_seconds is not None
                                and time.monotonic() - started >= max_seconds
                            ):
                                result = finalize(
                                    domains,
                                    len(seen),
                                    trials,
                                    temperature,
                                    time.monotonic() - started,
                                    False,
                                )
                                return attach_diagnostics(result, breakdown, first_round) if collect_diagnostics else result
                            remainder = ids[anchor + 1 :]
                            terminates = any(
                                bool((remainder == token).any())
                                for token in model.config.eos_token_ids
                            )
                            if anchor + block >= len(ids) and not terminates:
                                stats["censored_tails"] += 1
                                break
                            if replay is not None:
                                cache = replay.get(anchor,block)
                            else:
                                start = max(0, anchor - model.config.context_window)
                                payload = {
                                    "features": row["features"][start:anchor],
                                    "encoding": row["encoding"],
                                    "scales": row["scales"][start:anchor]
                                    if row["scales"] is not None
                                    else None,
                                }
                                features = dequantize_features(
                                    payload, device=device, dtype=model.embedding.weight.dtype
                                )
                                out = model(
                                    features[None],
                                    ids[anchor : anchor + 1].to(device),
                                    (torch.arange(start, anchor, device=device) + row["start"])[
                                        None
                                    ],
                                    torch.tensor([anchor + row["start"]], device=device),
                                    block_size=block,
                                    return_layer_features=False,
                                )
                                cache = model.build_transition_cache(out)
                                cache = replace(
                                    cache,
                                    candidate_ids=cache.candidate_ids[:, : block - 1],
                                    initial_scores=cache.initial_scores[:, : block - 1],
                                    transitions=cache.transitions[:, : block - 2],
                                )
                            available = min(block - 1, len(ids) - anchor - 1)
                            labels = pad_support(
                                ids[anchor + 1 : anchor + block], block - 1
                            )[None].to(device)
                            valid = (torch.arange(block - 1, device=device) < available)[
                                None
                            ]
                            if temperature == 0:
                                survival = greedy_acceptance(model, cache, labels, valid)
                            else:
                                support = target_ids.shape[-1]
                                pi = torch.zeros(
                                    1, block - 1, support, dtype=torch.long, device=device
                                )
                                pp = torch.zeros(1, block - 1, support, device=device)
                                pi[0, :available] = target_ids[
                                    anchor + 1 : anchor + 1 + available
                                ].to(device)
                                pp[0, :available] = target_probs[
                                    anchor + 1 : anchor + 1 + available
                                ].to(device)
                                survival = path_acceptance(
                                    cache, labels, pi, pp, valid, 1.0, True, generator
                                )
                            emitted = emitted_lengths(
                                survival, labels, model.config.eos_token_ids
                            ).item()
                            if collect_diagnostics:
                                offset = anchor - row["response_start"]
                                window = "exceeds_W" if anchor + row["start"] > model.config.context_window else "within_W"
                                for group in ("offset:" + offset_bucket(offset), "window:" + window,
                                              "offset:" + offset_bucket(offset) + "/" + window):
                                    item = breakdown.setdefault(group, {}).setdefault(str(block), {"tokens": 0, "rounds": 0})
                                    item["tokens"] += emitted
                                    item["rounds"] += 1
                                if offset == 0:
                                    expected = survival if temperature == 0 else path_acceptance(cache, labels, pi, pp, valid)
                                    eos = torch.zeros_like(labels, dtype=torch.bool)
                                    for token in model.config.eos_token_ids:
                                        eos |= labels == token
                                    value = 1 + (expected * (~eos)).sum().item()
                                    item = first_round.setdefault(str(block), {"expected_tokens": 0., "sampled_tokens": 0, "anchors": 0})
                                    item["expected_tokens"] += value
                                    item["sampled_tokens"] += emitted
                                    item["anchors"] += 1
                            if anchor + emitted >= len(ids):
                                raise ValueError(
                                    "round consumed an unavailable bonus; capture must retain EOS"
                                )
                            stats["tokens"] += emitted
                            stats["rounds"] += 1
                            anchor += emitted
                            if progress_callback is not None and time.monotonic() - last_progress >= 30:
                                progress = finalize(domains, len(seen), trials, temperature,
                                                    time.monotonic()-started, False)
                                if collect_diagnostics:
                                    attach_diagnostics(progress, breakdown, first_round)
                                progress_callback(progress)
                                last_progress = time.monotonic()
        if not seen:
            raise ValueError("no eligible held-out feature shards")
        result = finalize(
            domains, len(seen), trials, temperature, time.monotonic() - started, True
        )
        if collect_diagnostics:
            attach_diagnostics(result, breakdown, first_round)
        result.update(walk_offsets=list(offsets), window_tokens=window_tokens,
                      walk_mode="whole_trajectory" if window_tokens is None else "fixed_prefix_windows",
                      draft_path="packed_replay" if packed_chunk_size else "serial", packed_chunk_size=packed_chunk_size)
        return result
    finally:
        model.train(training)


def finalize(domains, sequences, trials, temperature, seconds, complete):
    for values in domains.values():
        for stats in values.values():
            stats["tau"] = (
                stats["tokens"] / stats["rounds"] if stats["rounds"] else None
            )
    return {
        "temperature": temperature,
        "sequences": sequences,
        "trials": trials,
        "complete": complete,
        "seconds": seconds,
        "by_domain": domains,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--head-quantization", choices=("qwenfast_w8",), default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", default="6,8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exclude-ids", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--require-decode-probs", action="store_true")
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--collect-diagnostics", action="store_true")
    parser.add_argument("--offsets", default="0")
    parser.add_argument("--window-tokens", type=int)
    parser.add_argument("--proposal-vocab", help="Override optional hot-vocabulary policy with these global IDs")
    parser.add_argument("--proposal-quantization", choices=("row_int8","none"), default="row_int8")
    parser.add_argument("--packed-chunk-size", type=int, default=0)
    args = parser.parse_args()
    emb, head = load_target_weights(args.target, args.head_quantization)
    model = DSpineDraft.from_pretrained(args.checkpoint, emb, head).to(args.device)
    if args.proposal_vocab:
        from .hot_vocab import load_hot_ids

        model.configure_proposal_vocabulary(load_hot_ids(args.proposal_vocab,model.config.vocab_size),
                                            args.proposal_quantization)
    result = evaluate_rounds(
        model,
        args.features,
        tuple(map(int, args.blocks.split(","))),
        args.temperature,
        args.sequences,
        trials=args.trials,
        max_seconds=args.max_seconds,
        exclude_ids=args.exclude_ids,
        expected_revision=args.target_revision,
        require_decode_probs=args.require_decode_probs
        or args.head_quantization == "qwenfast_w8",
        split=args.split,
        collect_diagnostics=args.collect_diagnostics,
        offsets=tuple(map(int, args.offsets.split(','))),
        window_tokens=args.window_tokens,
        packed_chunk_size=args.packed_chunk_size,
    )
    path = Path(args.output)
    result['proposal_vocab_size'] = model.config.vocab_size if model.proposal_weight is None else len(model.proposal_weight)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
