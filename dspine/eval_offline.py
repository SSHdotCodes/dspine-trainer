"""Acceptance on held-out on-policy target trajectories, without off-policy p.

A single stored trajectory does NOT provide p for a different proposed prefix.
Instead use the exact identity, for Y~p:
  E_q[prod min(1,p(X)/q(X))] = E_p[prod min(1,q(Y)/p(Y))].
The products are prefix products. Rao-Blackwell averaging is the default;
--monte-carlo uses Bernoulli q/p trials on the held-out target paths. Both are
unbiased estimates of expected tau including the bonus. No teacher-forced,
off-policy target distributions are treated as a speculative verifier.
"""

import argparse
import json
from pathlib import Path

import torch

from .data import AnchorStream, collate_anchors
from .modeling import DSpineDraft
from .transfer_space import load_target_weights


@torch.no_grad()
def path_acceptance(
    cache,
    labels,
    target_ids,
    target_probs,
    valid=None,
    temperature=1.0,
    monte_carlo=False,
    generator=None,
):
    """Per-anchor survival of each proposed token. labels excludes anchor."""
    if temperature <= 0:
        raise ValueError("use greedy_acceptance with greedy target continuations")
    n, length, k = cache.candidate_ids.shape
    batch = torch.arange(n, device=labels.device)
    previous = torch.zeros(n, dtype=torch.long, device=labels.device)
    survival = torch.ones(n, device=labels.device)
    result = []
    for t in range(length):
        scores = (
            cache.initial_scores[:, 0]
            if t == 0
            else cache.transitions[:, t - 1][batch, previous]
        )
        q = (scores.float() / temperature).softmax(-1)
        match = cache.candidate_ids[:, t] == labels[:, t, None]
        q_y = (q * match).sum(-1)
        p_y = ((target_ids[:, t] == labels[:, t, None]) * target_probs[:, t]).sum(-1)
        if torch.any((p_y <= 0) & (valid[:, t] if valid is not None else True)):
            raise ValueError("on-policy token absent from processed target support")
        ratio = (q_y / p_y.clamp_min(1e-30)).clamp(max=1)
        if valid is not None:
            ratio *= valid[:, t]
        if monte_carlo:
            ratio = (
                torch.rand(ratio.shape, device=ratio.device, generator=generator)
                < ratio
            ).float()
        survival *= ratio
        result.append(survival.clone())
        previous = match.long().argmax(-1)
    return torch.stack(result, 1)


@torch.no_grad()
def greedy_acceptance(model, cache, greedy_labels, valid=None):
    proposal = model.propose(cache, temperature=0)
    matches = proposal.tokens == greedy_labels
    if valid is not None:
        matches &= valid
    return matches.long().cumprod(-1).float()


@torch.no_grad()
def evaluate(
    model,
    stream,
    anchors=128,
    batch_size=4,
    temperature=1.0,
    monte_carlo=False,
    seed=42,
    require_decode_probs=False,
    diagnostic_callback=None,
):
    if temperature > 0 and stream.anchor_policy != "first":
        raise ValueError(
            "exact stored-path sampling evaluation requires first-response anchors independent of future response length"
        )
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    domains = {}
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for start in range(0, anchors, batch_size):
            rows = [next(stream) for _ in range(min(batch_size, anchors - start))]
            if temperature > 0 and require_decode_probs and any(
                row.get("target_distribution_source") != "actual_decode_raw_top_logprobs_joint" for row in rows
            ):
                raise ValueError("exact production eval requires actual decode probabilities")
            batch = {
                k: v.to(device)
                for k, v in collate_anchors(rows, model.config.block_size).items()
            }
            batch["context_features"] = batch["context_features"].to(
                model.embedding.weight.dtype
            )
            out = model(
                batch["context_features"],
                batch["labels"][:, 0],
                batch["context_positions"],
                batch["anchor_positions"],
                batch["context_valid"],
                return_layer_features=False,
            )
            cache = model.build_transition_cache(out)
            if temperature == 0:
                if any(r["sampling"]["temperature"] != 0 for r in rows):
                    raise ValueError(
                        "T=0 requires greedy target continuations; stochastic gold is not a greedy oracle"
                    )
                survival = greedy_acceptance(
                    model, cache, batch["labels"][:, 1:], batch["valid"][:, 1:]
                )
            else:
                if any(
                    r["sampling"] != {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
                    for r in rows
                ):
                    raise ValueError(
                        "T=1 evaluation requires production on-policy trajectories"
                    )
                survival = path_acceptance(
                    cache,
                    batch["labels"][:, 1:],
                    batch["target_ids"][:, 1:],
                    batch["target_probs"][:, 1:],
                    batch["valid"][:, 1:],
                    temperature,
                    monte_carlo,
                    generator,
                )
            if diagnostic_callback is not None:
                diagnostic_callback(survival.detach(), batch["valid"][:, 1:])
            for i, row in enumerate(rows):
                values = domains.setdefault(
                    row["domain"],
                    {b: [] for b in (4, 6, 8, 12, 16) if b <= model.config.block_size},
                )
                for b in values:
                    # Exclude tail anchors without b real tokens; do not inflate bonus at sequence end.
                    eos = torch.zeros_like(batch["labels"][i, 1:], dtype=torch.bool)
                    for token in model.config.eos_token_ids:
                        eos |= batch["labels"][i, 1:] == token
                    terminated = bool((eos & batch["valid"][i, 1:]).any())
                    if (
                        row.get("available_tokens", len(row["labels"])) > b
                        or terminated
                    ):
                        accepted = survival[i, : b - 1]
                        # An accepted EOS ends the response, so there is no bonus after it.
                        values[b].append(
                            1
                            + accepted.sum().item()
                            - (accepted * eos[: b - 1]).sum().item()
                        )
        return {
            domain: {
                str(b): {"tau": sum(v) / len(v), "anchors": len(v)}
                for b, v in values.items()
                if v
            }
            for domain, values in domains.items()
        }
    finally:
        model.train(training)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--head-quantization", choices=("qwenfast_w8",), default=None)
    p.add_argument("--features", required=True)
    p.add_argument("--anchors", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--monte-carlo", action="store_true")
    p.add_argument("--require-decode-probs", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    emb, head = load_target_weights(args.target, args.head_quantization)
    model = DSpineDraft.from_pretrained(args.checkpoint, emb, head).to(args.device)
    stream = AnchorStream(
        args.features,
        block_size=model.config.block_size,
        context_window=model.config.context_window,
        split="validation",
        anchor_policy="first",
        expected_layer_ids=model.config.target_layer_ids,
    )
    result = evaluate(
        model,
        stream,
        args.anchors,
        temperature=args.temperature,
        monte_carlo=args.monte_carlo,
        require_decode_probs=args.require_decode_probs or args.head_quantization == "qwenfast_w8",
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
