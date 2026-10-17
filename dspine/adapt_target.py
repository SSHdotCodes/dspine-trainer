"""Weights-only adaptation to a new target, including transfer-space alignment."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import DSpineConfig
from .modeling import DSpineDraft
from .transfer_space import build_transfer_space, load_target_weights


@torch.no_grad()
def procrustes_align(new_codebook, old_codebook, new_ids=None, old_ids=None):
    """Return Q minimizing ||new_C[ids] Q - old_C[ids]||_F.

    Token ids must identify the same token strings. For a changed tokenizer,
    pass an explicit new/old id mapping, never assume integer ids still match.
    """
    if new_codebook.shape[1] != old_codebook.shape[1]:
        raise ValueError("transfer rank must match for orthogonal alignment")
    if new_ids is None:
        if new_codebook.shape[0] != old_codebook.shape[0]:
            raise ValueError("changed vocab requires token mapping")
        new_ids = torch.arange(new_codebook.shape[0])
        old_ids = new_ids
    x = F.normalize(new_codebook[new_ids].double(), dim=-1)
    y = F.normalize(old_codebook[old_ids].double(), dim=-1)
    u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    return (u @ vh).float()


def main():
    from safetensors.torch import load_file

    p = argparse.ArgumentParser()
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-target", required=True)
    p.add_argument("--new-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--token-map",
        help="JSON list of [new_id,old_id] pairs; required for changed tokenizer",
    )
    p.add_argument(
        "--same-tokenizer",
        action="store_true",
        help="explicitly certify unchanged token ids",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--head-quantization", choices=("qwenfast_w8",), default=None)
    args = p.parse_args()
    if not args.token_map and not args.same_tokenizer:
        p.error("provide --token-map or certify --same-tokenizer")
    c = DSpineConfig.load(args.new_config)
    emb, head = load_target_weights(args.new_target, args.head_quantization)
    space = build_transfer_space(head.to(args.device), c.code_size)
    space["head_quantization"] = args.head_quantization
    old = load_file(str(Path(args.old_checkpoint) / "model.safetensors"))
    old_code = old["selected_prefix_bridge.codebook"]
    mapping = json.loads(Path(args.token_map).read_text()) if args.token_map else None
    new_ids = torch.tensor([x[0] for x in mapping]) if mapping else None
    old_ids = torch.tensor([x[1] for x in mapping]) if mapping else None
    rotation = procrustes_align(space["codebook"], old_code, new_ids, old_ids)
    space["raw_basis"] = space["basis"]
    space["alignment"] = rotation
    space["basis"] = space["basis"] @ rotation
    space["codebook"] = (space["codebook"].float() @ rotation).bfloat16()
    model = DSpineDraft(c, emb, head, space["codebook"], space["basis"])
    old_c = DSpineConfig.load(Path(args.old_checkpoint) / "config.json")
    state = model.state_dict()
    copied = []
    initialized = []
    for key, value in state.items():
        if (
            key.startswith(("embedding.", "lm_head."))
            or key == "selected_prefix_bridge.codebook"
        ):
            continue
        # Changed feature slots/dimensions require a fresh context fusion/readout.
        incompatible_context = key == "fc.weight" and (
            tuple(c.target_layer_ids) != tuple(old_c.target_layer_ids)
            or c.target_hidden_size != old_c.target_hidden_size
        )
        if key in old and old[key].shape == value.shape and not incompatible_context:
            state[key] = old[key]
            copied.append(key)
        else:
            initialized.append(key)
    model.load_state_dict(state)
    model.save_pretrained(args.output)
    torch.save(space, Path(args.output) / "transfer-space.pt")
    report = {
        "copied": copied,
        "reinitialized": initialized,
        "old_layers": old_c.target_layer_ids,
        "new_layers": c.target_layer_ids,
        "optimizer_resume": False,
        "note": "recapture new-target features, then fine-tune all draft parameters",
    }
    (Path(args.output) / "adaptation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
