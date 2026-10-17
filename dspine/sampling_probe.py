"""Reservation-gated numerical/empirical parity with the installed joint sampler.

Run inside the capture image before loading the target. Includes flat/cutoff
ties and BF16-rounded logits; checks both renormalization and the fused sampler.
"""

import argparse
import json
from pathlib import Path

import torch

from .capture import processed_top20


@torch.inference_mode()
def main():
    from flashinfer.sampling import top_k_top_p_sampling_from_probs
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = [torch.randn(257, device="cuda"), torch.zeros(29, device="cuda")]
    rows += [torch.randn(257, device="cuda").mul_(5).bfloat16().float()]
    tie = torch.full((29,), -1000.0, device="cuda")
    tie[:3] = torch.tensor([0.5, 0.25, 0.25], device="cuda").log()
    rows.append(tie)
    results = []
    for index, row in enumerate(rows):
        ids, p = processed_top20(
            torch.ones(1, 1, device="cuda"),
            row[:, None],
            top_p=0.6 if index == 3 else 0.95,
        )
        expected = torch.zeros_like(row).scatter_add_(0, ids[0], p[0])
        original = row.softmax(-1)[None]
        top_p = 0.6 if index == 3 else 0.95
        k = torch.tensor([20], dtype=torch.int32, device="cuda")
        cutoff = torch.tensor([top_p], dtype=torch.float32, device="cuda")
        actual = top_k_renorm_prob(original, k)
        nucleus = top_p_renorm_prob(original, cutoff)
        actual.masked_fill_(nucleus <= 0, 0)
        actual /= actual.sum(-1, keepdim=True)
        error = (actual[0] - expected).abs().max().item()
        if error > 2e-6:
            raise AssertionError(
                f"joint renormalization mismatch in case {index}: {error}"
            )
        inputs = original.expand(args.samples, -1).contiguous()
        samples = top_k_top_p_sampling_from_probs(
            inputs, 20, top_p, filter_apply_order="joint"
        )
        observed = (
            torch.bincount(samples.long(), minlength=row.numel()).float() / args.samples
        )
        empirical_error = (observed - expected).abs().max().item()
        if empirical_error > 0.006:
            raise AssertionError(
                f"fused sampling mismatch in case {index}: {empirical_error}"
            )
        results.append(
            {
                "case": index,
                "support": int(expected.count_nonzero()),
                "renorm_max_error": error,
                "empirical_max_error": empirical_error,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"samples": args.samples, "cases": results, "passed": True}, indent=2
        )
        + "\n"
    )
    print(output.read_text(), flush=True)


if __name__ == "__main__":
    main()
