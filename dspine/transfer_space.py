"""Appendix A PCA whitening in bounded memory; save basis for adaptation/init."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F


@torch.no_grad()
def build_transfer_space(head, rank=1024, chunk_size=4096, eps=1e-10):
    if not 0 < rank <= head.shape[1]:
        raise ValueError("invalid PCA rank")
    device = head.device
    h = head.shape[1]
    n = head.shape[0]
    mean = torch.zeros(h, device=device, dtype=torch.float64)
    gram = torch.zeros(h, h, device=device, dtype=torch.float64)
    for x in head.split(chunk_size):
        x = F.normalize(x.double(), dim=-1)
        mean += x.sum(0)
        gram += x.T @ x
    mean /= n
    gram -= n * mean[:, None] * mean[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    values = eigenvalues[-rank:].flip(0).clamp_min(eps)
    basis = eigenvectors[:, -rank:].flip(1)
    # Fix arbitrary eigenvector signs for reproducible checkpoints.
    signs = basis.gather(0, basis.abs().argmax(0)[None]).squeeze(0).sign()
    basis *= signs
    codebook = torch.empty(n, rank, device=device, dtype=torch.bfloat16)
    for start in range(0, n, chunk_size):
        x = F.normalize(head[start : start + chunk_size].double(), dim=-1) - mean
        codebook[start : start + chunk_size] = F.normalize(
            (x @ basis) * values.rsqrt(), dim=-1
        ).bfloat16()
    return {
        "codebook": codebook.cpu(),
        "basis": basis.float().cpu(),
        "mean": mean.float().cpu(),
        "eigenvalues": values.float().cpu(),
        "method": "row_l2_center_gram_pca_whiten_row_l2_v1",
    }


@torch.no_grad()
def qwenfast_w8_weights(weight, rows_per_chunk=4096):
    """Production qwenfast w8_quant/unpack, including BF16 scale rounding.

    Return the effective BF16 weight used by its large-row prefill GEMM and by
    serving's pk_gather_rows. This optional path leaves dense targets unchanged.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("qwenfast W8 requires a BF16 matrix")
    width = weight.shape[1]
    group = 128 if width % 128 == 0 else 64
    if width % group:
        raise ValueError("qwenfast W8 width must be divisible by64")
    output = torch.empty_like(weight)
    for start in range(0, len(weight), rows_per_chunk):
        x = weight[start:start + rows_per_chunk].float().reshape(-1, width // group, group)
        scale = (x.abs().amax(-1) / 127).clamp_min(1e-30).bfloat16().float()
        q = (x / scale[..., None]).round().clamp(-127, 127)
        output[start:start + len(x)] = (q * scale[..., None]).reshape(-1, width).bfloat16()
    return output


def load_target_weights(path, head_quantization=None):
    """Safetensors only; load the two dense target tables without the MoE model."""
    import json

    from safetensors import safe_open

    path = Path(path)
    index = path / "model.safetensors.index.json"
    weights = json.loads(index.read_text())["weight_map"] if index.exists() else None
    aliases = {
        "embedding": (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ),
        "head": ("lm_head.weight", "language_model.lm_head.weight"),
    }
    found = {}
    for role, names in aliases.items():
        for name in names:
            if weights and name not in weights:
                continue
            files = (
                [path / weights[name]]
                if weights
                else sorted(path.glob("*.safetensors"))
            )
            for filename in files:
                with safe_open(str(filename), framework="pt", device="cpu") as f:
                    if name in f.keys():
                        found[role] = f.get_tensor(name)
                        break
            if role in found:
                break
        if role not in found:
            raise KeyError(f"missing {role} ({names}) in {path}")
    if head_quantization == "qwenfast_w8":
        found["head"] = qwenfast_w8_weights(found["head"])
    elif head_quantization is not None:
        raise ValueError("unsupported target head quantization")
    return found["embedding"], found["head"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--head-quantization", choices=("qwenfast_w8",), default=None)
    args = p.parse_args()
    _, head = load_target_weights(args.target, args.head_quantization)
    result = build_transfer_space(head.to(args.device), args.rank)
    result["head_quantization"] = args.head_quantization
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
