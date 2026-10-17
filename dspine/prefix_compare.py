"""Prepare identical real prefixes and quantify base/production capture drift."""

import argparse
import json
from pathlib import Path

import torch

from .data import (
    dequantize_features,
    generation_paths,
    quantize_features,
    records,
    target_distribution,
)
from .transfer_space import qwenfast_w8_weights


def prepare(generations, output, lengths=(2048, 8192, 32768)):
    selected = {}
    for row in records(generation_paths(generations)):
        tokens = row["prompt_ids"] + row["output_ids"]
        for length in lengths:
            if length not in selected and len(tokens) >= length + 16:
                selected[length] = dict(
                    row,
                    id=f"capture-probe-{length}",
                    prompt_ids=tokens[:length],
                    output_ids=tokens[length : length + 16],
                    capture_probe=True,
                    split="validation",
                    finish_reason="length",
                )
        if len(selected) == len(lengths):
            break
    if not selected:
        raise RuntimeError("no real prefix long enough for the comparison")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        "".join(json.dumps(selected[k]) + "\n" for k in sorted(selected))
    )
    return {
        "prefix_lengths": sorted(selected),
        "missing_lengths": sorted(set(lengths) - selected.keys()),
    }


def difference(base, production):
    base, production = base.float(), production.float()
    delta = production - base
    return {
        "cosine_mean": torch.nn.functional.cosine_similarity(base, production, dim=-1)
        .mean()
        .item(),
        "relative_l2": (delta.norm() / base.norm().clamp_min(1e-20)).item(),
        "rms_error": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
    }


def compare(base_root, production_root):
    result = []
    base_root, production_root = map(Path, (base_root, production_root))
    for path in sorted((base_root / "shards").glob("*.pt")):
        base = torch.load(path, weights_only=True)
        production = torch.load(
            production_root / "shards" / path.name, weights_only=True
        )
        if not torch.equal(base["input_ids"], production["input_ids"]):
            raise ValueError("comparison prefixes differ")
        if base["encoding"] != "bf16" or production["encoding"] != "bf16":
            raise ValueError("compare runtime numerics before storage quantization")
        layers = len(base["layer_ids"])
        a = dequantize_features(base).reshape(len(base["input_ids"]), layers, -1)
        b = dequantize_features(production).reshape(
            len(production["input_ids"]), layers, -1
        )
        features = {
            str(layer): difference(a[:, i], b[:, i])
            for i, layer in enumerate(base["layer_ids"])
        }
        encoded, scales = quantize_features(b)
        roundtrip = dequantize_features(
            {"features": encoded, "scales": scales, "encoding": "fp8_e4m3fn"}
        ).reshape_as(b)
        storage_error = {
            str(layer): difference(b[:, i], roundtrip[:, i])
            for i, layer in enumerate(base["layer_ids"])
        }
        logits_a = torch.load(base_root / "probe-logits" / path.name, weights_only=True)
        logits_b = torch.load(
            production_root / "probe-logits" / path.name, weights_only=True
        )
        p, q = logits_a.softmax(-1), logits_b.softmax(-1)
        processed = []
        for payload in (base, production):
            ids, probs = target_distribution(payload)
            start = payload["response_start"]
            dense = torch.zeros(len(probs) - start, logits_a.shape[-1])
            dense.scatter_add_(1, ids[start:].long(), probs[start:])
            processed.append(dense)
        processed_a, processed_b = processed
        result.append(
            dict(
                id=base["id"],
                prefix_tokens=base["response_start"] + base["start"],
                retained_tokens=len(a),
                features=features,
                fp8_storage_error=storage_error,
                logits=difference(logits_a, logits_b),
                full_softmax_l1=(p - q).abs().sum(-1).mean().item(),
                argmax_agreement=(p.argmax(-1) == q.argmax(-1)).float().mean().item(),
                processed_l1=(processed_a - processed_b).abs().sum(-1).mean().item(),
                processed_support_disagreement=((processed_a > 0) != (processed_b > 0))
                .sum(-1)
                .float()
                .mean()
                .item(),
            )
        )
    if not result:
        raise RuntimeError("comparison has no completed probes")
    head_a = torch.load(base_root / "probe-head.pt", weights_only=True)
    head_b = torch.load(production_root / "probe-head.pt", weights_only=True)
    if not torch.equal(head_a["ids"], head_b["ids"]):
        raise ValueError("head probe rows differ")
    expected = qwenfast_w8_weights(head_a["weight"])
    error = (expected - head_b["weight"]).abs().max().item()
    if error != 0:
        raise ValueError(
            f"CPU effective-head reconstruction differs from native unpack: {error}"
        )
    return {
        "storage_encoding": "bf16",
        "probes": result,
        "training_capture_runtime": "production",
        "head_drift": difference(head_a["weight"], head_b["weight"]),
        "effective_head_reconstruction_max_error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--generations", required=True)
    prep.add_argument("--output", required=True)
    check = sub.add_parser("compare")
    check.add_argument("--base", required=True)
    check.add_argument("--production", required=True)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.generations, args.output)
    else:
        result = compare(args.base, args.production)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        result = {
            "output": args.output,
            "prefix_lengths": [x["prefix_tokens"] for x in result["probes"]],
            "effective_head_reconstruction_max_error": result[
                "effective_head_reconstruction_max_error"
            ],
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
