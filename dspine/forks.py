"""Explicit recipe forks preserve optimizer/RNG without weakening exact resume."""

import copy
import hashlib
import json
import math
from pathlib import Path

FORK_FIELDS = {
    "output",
    "lambda_ce_reduction",
    "eval_step_offset",
    "batch_trace",
    "anchor_sampling",
    "refinement_start_position",
    "mask_unsupported_ce",
    "cuda_memory_limit_gb",
    "lr",
    "gradient_accumulation",
    "curriculum_p",
    "eval_every",
    "sample_trace",
    "embedding_loss_scale",
    "loss_alpha",
    "loss_gamma",
    "context_fusion",
    "pos1_weight",
}


def restore_optimizer(optimizer, state, model, *, new_context_fusion=False):
    """Keep old moments/IDs and lazily initialize only a newly added shortcut.

    The historical trainer has one AdamW group and no names in its optimizer
    state. DSpineDraft appends context_fusion last, so the old ordering is an
    unchanged prefix. Never guess a migration for another architecture change.
    """
    if new_context_fusion:
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        current = optimizer.state_dict()
        if (
            len(state["param_groups"]) != 1
            or len(current["param_groups"]) != 1
            or names[-1] != "context_fusion.weight"
            or len(state["param_groups"][0]["params"]) != len(names) - 1
        ):
            raise ValueError(
                "optimizer migration requires only one appended fusion weight"
            )
        # Do not duplicate multi-GB moment tensors on the shared host.
        state = {**state, "param_groups": copy.deepcopy(state["param_groups"])}
        old_ids = state["param_groups"][0]["params"]
        # IDs may be arbitrary in a serialized optimizer; use a fresh unused ID.
        old_ids.append(max(old_ids, default=-1) + 1)
    optimizer.load_state_dict(state)


def validate_recipe_values(config):
    """Validate opt-in recipe values before writing outputs or initializing CUDA."""
    for key, default, minimum, maximum in (
        ("embedding_loss_scale", 1.0, 0.0, None),
        ("loss_alpha", 0.1, 0.0, 1.0),
        ("loss_gamma", 7.0, 0.0, None),
        ("pos1_weight", 1.0, 0.0, None),
    ):
        value = config.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
            or (key == "loss_gamma" and value == 0)
            or (maximum is not None and value > maximum)
        ):
            raise ValueError(f"invalid {key}")
    mode = config.get("context_fusion", "off")
    if not isinstance(mode, str) or mode not in {"off", "anchor", "all"}:
        raise ValueError("context_fusion must be off, anchor or all")
    lr = config.get("lr", 6e-4)
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("lr must be finite and positive")
    accum = config.get("gradient_accumulation", 8)
    if type(accum) is not int or accum < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    probability = config.get("curriculum_p")
    if probability is not None and (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("curriculum_p must be null or a finite probability in [0,1]")


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_fork(source, destination, checkpoint):
    """Only explicit recipe overrides, diagnostics and output may differ."""
    changes = {
        key
        for key in source.keys() | destination.keys()
        if source.get(key) != destination.get(key)
    }
    if changes - FORK_FIELDS:
        raise ValueError(
            f"unsupported fork config changes: {sorted(changes - FORK_FIELDS)}"
        )
    validate_recipe_values(destination)
    old = Path(source["output"]).resolve()
    new = Path(destination["output"]).resolve()
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise ValueError("fork output must be separate from the parent output")
    if destination.get("lambda_ce_reduction", "position_sum") not in {
        "position_sum",
        "weighted_token_mean",
    }:
        raise ValueError("unsupported fork lambda CE reduction")
    if destination.get("anchor_sampling", "with_replacement") not in {
        "with_replacement",
        "without_replacement",
    }:
        raise ValueError("unsupported fork anchor sampling")
    if destination.get("refinement_start_position", 1) not in (1, 2):
        raise ValueError("unsupported fork refinement start")
    if not isinstance(destination.get("mask_unsupported_ce", False), bool):
        raise ValueError("CE support mask must be boolean")
    cap = destination.get("cuda_memory_limit_gb")
    if cap is not None and (not isinstance(cap, (int, float)) or cap <= 0):
        raise ValueError("allocator cap must be positive")
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "parent_run_config_sha256": config_digest(source),
        "fork_run_config_sha256": config_digest(destination),
        "changed_fields": sorted(changes),
    }
