"""Named recipe forks sharing ab_lambda's lifecycle, isolation and promotion.

All changes are explicit. A parent in the current feature window keeps its exact
cursor. Larger accumulation changes optimizer updates, not the per-sequence
anchor count or the cumulative response-token cosine/curriculum schedule.
"""

import copy
import itertools
import json
import math
import re
import shutil
import time
from pathlib import Path

import torch

from . import ab_lambda as manager
from .forks import FORK_FIELDS, config_digest, validate_fork, validate_recipe_values
from .quality_gate import pooled, read_metrics
from .rolling import atomic_json, capture_window, latest_state, source_fingerprint

BASE_RECIPES = {
    "A": dict(
        lambda_ce_reduction="position_sum",
        anchor_sampling="with_replacement",
        refinement_start_position=1,
        mask_unsupported_ce=False,
    ),
    "B": dict(
        lambda_ce_reduction="weighted_token_mean",
        anchor_sampling="with_replacement",
        refinement_start_position=1,
        mask_unsupported_ce=False,
    ),
    "C": dict(
        lambda_ce_reduction="weighted_token_mean",
        anchor_sampling="without_replacement",
        refinement_start_position=2,
        mask_unsupported_ce=True,
    ),
}
# Output, data, seeds, head policy and token schedule cannot be overridden.
ARM_OVERRIDES = {
    "lr",
    "gradient_accumulation",
    "curriculum_p",
    "embedding_loss_scale",
    "loss_alpha",
    "loss_gamma",
    "context_fusion",
    "pos1_weight",
}


def read_spec(path, base_recipe=None, reference_arm=None):
    spec = json.loads(Path(path).read_text())
    if set(spec) - {"base_recipe", "reference_arm", "arms"}:
        raise ValueError("unknown recipe specification fields")
    base = base_recipe or spec.get("base_recipe", "C")
    arms = spec.get("arms")
    if base not in BASE_RECIPES or not isinstance(arms, dict) or not arms:
        raise ValueError("recipe needs base A/B/C and a nonempty arms dictionary")
    for name, overrides in arms.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("invalid recipe arm name")
        if not isinstance(overrides, dict) or set(overrides) - ARM_OVERRIDES:
            raise ValueError(f"{name}: only {sorted(ARM_OVERRIDES)} may be overridden")
        validate_recipe_values(overrides)
    reference = reference_arm or spec.get("reference_arm", next(iter(arms)))
    if reference not in arms:
        raise ValueError("reference arm is missing from the recipe")
    return dict(base_recipe=base, reference_arm=reference, arms=arms)


def recipe_configs(parent_run, spec, root, start, *, eval_every=250, cap_gb=None):
    if type(eval_every) is not int or eval_every < 1:
        raise ValueError("recipe eval cadence must be a positive integer")
    configs = {}
    for name, overrides in spec["arms"].items():
        config = {
            **copy.deepcopy(parent_run),
            **BASE_RECIPES[spec["base_recipe"]],
            **overrides,
            "output": str(root / name),
            "eval_every": eval_every,
            "eval_step_offset": start,
            "batch_trace": True,
            "sample_trace": True,
        }
        if cap_gb is not None:
            config["cuda_memory_limit_gb"] = cap_gb
        validate_fork(parent_run, config, root / "parent")
        configs[name] = config
    return configs


def arm_goals(configs, reference, start, steps, equal_samples=False):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    reference_accum = configs[reference].get("gradient_accumulation", 8)
    goals = {}
    for arm, config in configs.items():
        accum = config.get("gradient_accumulation", 8)
        updates = steps
        if equal_samples:
            updates, remainder = divmod(steps * reference_accum, accum)
            if remainder or updates < 1:
                raise ValueError(
                    "equal-samples budget must divide exactly by every arm's accumulation"
                )
        goals[arm] = start + updates
    return goals


def validate_parent(plan, window, saved):
    """Keep immutable data/model/head/schedule fields identical to the source plan."""
    source, parent = plan["run"], saved["run_config"]
    changes = {
        k for k in source.keys() | parent.keys() if source.get(k) != parent.get(k)
    }
    if changes - FORK_FIELDS:
        raise ValueError(
            f"parent differs from source data/model/schedule: {sorted(changes - FORK_FIELDS)}"
        )
    if not source.get("schedule_response_tokens"):
        raise ValueError("recipe forks require the cumulative response-token schedule")
    if not window["start_step"] <= saved["step"] < window["stop_step"]:
        raise ValueError("parent step is outside the selected unfinished window")
    same_window = saved.get("data_window") == window["name"]
    if not same_window and saved["step"] != window["start_step"]:
        raise ValueError(
            "a mid-window parent must have that window's exact data cursor"
        )
    if parent.get("gradient_accumulation", 8) != source.get("gradient_accumulation", 8):
        raise ValueError(
            "parent accumulation differs from this plan; use its promoted plan"
        )
    return not same_window


def summarize_recipe(root, experiment, arms=None):
    """Pair individual sequence draws, including arms with unequal update counts."""
    start = experiment["start_step"]
    definitions = experiment["arm_definitions"]
    arms = list(arms or definitions)
    blocks = experiment["selection_blocks"]
    result = dict(
        kind="recipe_ablation", requested_arms=arms, branches={}, paired_samples={}
    )
    traces = {}
    for arm, definition in definitions.items():
        goal = definition["goal_step"]
        rows = read_metrics(root / arm / "metrics.jsonl")
        steps = {
            r["step"]: r
            for r in rows
            if "sample_traces" in r and start < r["step"] <= goal
        }
        # Compare only contiguous prefixes; never pair samples beyond a missing
        # record. A live writer's partial last JSON line is retried next poll.
        contiguous = list(
            itertools.takewhile(lambda s: s in steps, range(start + 1, goal + 1))
        )
        traces[arm] = [t for s in contiguous for t in steps[s]["sample_traces"]]
        accum = definition["gradient_accumulation"]
        if any(len(steps[s]["sample_traces"]) != accum for s in steps):
            raise ValueError(
                f"{arm}: sample trace length differs from configured accumulation"
            )
        token_count = experiment["parent_response_tokens_seen"]
        for s in contiguous:
            token_count += sum(t["response_tokens"] for t in steps[s]["sample_traces"])
            if steps[s]["response_tokens_seen"] != token_count:
                raise ValueError(
                    f"{arm}: cumulative response-token count differs from samples"
                )
        evaluations = {
            r["step"]: r
            for r in rows
            if "offline_tau" in r and start <= r["step"] <= goal
        }
        result["branches"][arm] = {
            "goal_step": goal,
            "last_step": max(steps, default=start),
            "sequence_samples": len(traces[arm]),
            "anchor_blocks": sum(t["anchors"] for t in traces[arm]),
            "evaluations": [
                dict(
                    step=step,
                    branch_updates=step - start,
                    sequence_samples=(step - start) * accum,
                    reference_updates=(step - start)
                    * accum
                    / experiment["reference_accumulation"],
                    response_tokens_seen=steps[step]["response_tokens_seen"]
                    if step in steps
                    else experiment["parent_response_tokens_seen"],
                    complete_tau={
                        str(b): pooled(row["offline_tau"], b) for b in blocks
                    },
                    opening_tau={
                        str(b): pooled(row["offline_first_anchor"], b) for b in blocks
                    },
                    t0_tau={str(b): pooled(row["offline_tau_t0"], b) for b in blocks},
                )
                for step, row in sorted(evaluations.items())
            ],
            "last250_means": {
                key: sum(steps[s][key] for s in sorted(steps)[-250:])
                / min(250, len(steps))
                for key in (
                    "ce",
                    "ce_token_mean",
                    "l1",
                    "auxiliary",
                    "distillation",
                    "seconds_per_step",
                    "anchors",
                )
            }
            if steps
            else {},
            "full_budget_means": {
                key: sum(steps[s][key] * accum for s in contiguous) / len(traces[arm])
                for key in ("ce", "ce_token_mean", "l1", "auxiliary", "distillation")
            }
            if traces[arm]
            else {},
            "complete": len(contiguous) == goal - start and goal in evaluations,
        }
    for a, b in itertools.combinations(definitions, 2):
        shared = min(len(traces[a]), len(traces[b]))
        for index, (left, right) in enumerate(zip(traces[a], traces[b])):
            for key in ("sequence", "data", "anchors", "response_tokens"):
                if left[key] != right[key]:
                    raise ValueError(f"{a}/{b}: sample{index} differs: {key}")
            # Curriculum overrides intentionally change masks. Different
            # accumulation can also change when the scheduled p is evaluated.
            if (
                left["curriculum_p"] == right["curriculum_p"]
                and left["teacher"] != right["teacher"]
            ):
                raise ValueError(f"{a}/{b}: teacher RNG differs at sample{index}")
        result["paired_samples"][a + "/" + b] = shared
    result["complete"] = all(result["branches"][a]["complete"] for a in arms)
    result["decision"] = (
        "manual_review" if result["complete"] else "incomplete_do_not_promote"
    )
    manager.attach_arm_statuses(result, root)
    atomic_json(root / "comparison.json", result)
    return result


def run_recipe_experiment(args, plan, root):
    deadline = time.monotonic() + args.max_minutes * 60
    spec = read_spec(args.recipe_file, args.base_recipe, args.reference_arm)
    selected = args.arms or list(spec["arms"])
    if not set(selected) <= spec["arms"].keys():
        raise ValueError("requested arm is absent from recipe specification")
    state = json.loads((args.plan.parent / "state.json").read_text())
    window = plan["windows"][args.window]
    co_run = getattr(args, "allow_active_main", False)
    if co_run and state["phase"] != "train":
        raise ValueError("co-run requires the source controller's pure training phase")
    pinned_view = root / "feature-view"
    if not (co_run and pinned_view.exists()) and (
        state["window"] != args.window or state["phase"] not in {"capture", "train"}
    ):
        raise ValueError("source controller must be on the selected feature window")
    if co_run and pinned_view.exists():
        data_read_root = manager.pin_feature_window(plan, window, root)
        ready = True
    else:
        data_read_root = None
        ready = manager.complete_window(plan, window)
    if not ready and args.finish_capture and not manager.STOP:
        seconds = min(
            args.capture_max_minutes * 60, deadline - time.monotonic() - 80 * 60
        )
        if seconds < 180:
            raise RuntimeError("insufficient time for capture and recipe arms")
        capture_window(plan, window, plan["train_root"], root, seconds)
        ready = manager.complete_window(plan, window)
    atomic_json(
        root / "feature-readiness.json", {"window": window["name"], "complete": ready}
    )
    if not ready:
        print(
            json.dumps(
                dict(
                    not_ready=window["name"],
                    gpu_training_started=False,
                    next="owner may finish missing features with --finish-capture",
                )
            ),
            flush=True,
        )
        raise SystemExit(75)
    if manager.STOP:
        raise SystemExit(75)
    required = (40 * len(selected) + 10) * 1_000_000_000
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(
            f"selected arms need {required / 1e9:g}GB free for checkpoints"
        )
    selector = getattr(args, "checkpoint_selector", args.checkpoint)
    moving_selector = selector in {
        Path("latest"),
        Path(plan["run"]["output"]),
        Path(plan["run"]["output"]) / "latest",
    }
    parent, args.checkpoint = manager.snapshot_live_parent(
        args.checkpoint.resolve(),
        root,
        latest_output=Path(plan["run"]["output"])
        if co_run and moving_selector
        else None,
    )
    saved = torch.load(
        parent / "trainer.pt", map_location="cpu", weights_only=False, mmap=True
    )
    reset_data = validate_parent(plan, window, saved)
    if co_run and data_read_root is None:
        data_read_root = manager.pin_feature_window(plan, window, root)
    run = saved["run_config"]
    if not (
        run.get("packed")
        and run.get("eval_mode") == "rounds"
        and run.get("eval_before_training")
        and run.get("eval_at_end")
        and run.get("eval_report_first_anchor")
        and run.get("eval_require_decode_probs")
        and run.get("eval_greedy_features")
    ):
        raise ValueError(
            "recipe forks require the complete strict-decode T1/opening/T0 evaluation suite"
        )
    start = saved["step"]
    configs = recipe_configs(
        run,
        spec,
        root,
        start,
        eval_every=250 if args.eval_every is None else args.eval_every,
        cap_gb=args.allocator_cap_gb,
    )
    goals = arm_goals(
        configs, spec["reference_arm"], start, args.steps, args.equal_samples
    )
    available_samples = (window["stop_step"] - start) * run.get(
        "gradient_accumulation", 8
    )
    for arm, goal in goals.items():
        if (goal - start) * configs[arm].get(
            "gradient_accumulation", 8
        ) > available_samples or goal > run["total_steps"]:
            raise ValueError(
                f"{arm}: requested sample/update budget exceeds the current feature window"
            )
    experiment = {
        "kind": "recipe_ablation",
        "co_run": co_run,
        "start_step": start,
        "window_index": args.window,
        "window_ids_sha256": config_digest(window["ids"]),
        "source_config_sha256": config_digest(plan["run"]),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "parent_config_sha256": config_digest(run),
        "parent_response_tokens_seen": saved.get("response_tokens_seen", 0),
        "parent_accumulation": run.get("gradient_accumulation", 8),
        "reset_data": reset_data,
        "spec": spec,
        "reference_accumulation": configs[spec["reference_arm"]].get(
            "gradient_accumulation", 8
        ),
        "equal_samples": args.equal_samples,
        "reference_steps": args.steps,
        "selection_blocks": run.get("selection_blocks", [6, 8]),
        "arm_definitions": {
            a: {
                "goal_step": goals[a],
                "gradient_accumulation": c.get("gradient_accumulation", 8),
                "run_config_sha256": config_digest(c),
            }
            for a, c in configs.items()
        },
        "code_sha256": {
            name: manager.file_digest(Path(__file__).parent / name)
            for name in (
                "recipe_ablation.py",
                "ab_lambda.py",
                "train.py",
                "forks.py",
                "data.py",
                "modeling.py",
                "losses.py",
            )
        },
    }
    path = root / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != experiment:
        raise ValueError("experiment definition changed; choose a new output")
    for arm, config in configs.items():
        config_path = root / arm / "run.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("arm config changed; choose a new output")
    atomic_json(path, experiment)
    for arm, config in configs.items():
        atomic_json(root / arm / "run.json", config)
    cap = max(configs[a].get("cuda_memory_limit_gb", 36) for a in selected)
    parallel_arms = min(
        getattr(args, "parallel_arms", None) or len(selected), len(selected)
    )
    mode = manager.choose_mode(manager.free_gpu_bytes(), cap, args.mode, parallel_arms)
    print(
        json.dumps(
            dict(
                mode=mode,
                base_recipe=spec["base_recipe"],
                goals=goals,
                equal_samples=args.equal_samples,
                reset_data=reset_data,
            )
        ),
        flush=True,
    )
    summarize = lambda: summarize_recipe(root, experiment, arms=selected)
    groups = (
        [
            tuple(selected[i : i + parallel_arms])
            for i in range(0, len(selected), parallel_arms)
        ]
        if mode == "parallel"
        else [(a,) for a in selected]
    )
    failures = manager.run_groups(
        groups,
        configs,
        root,
        parent,
        window,
        goals,
        deadline,
        reset_data=reset_data,
        summarizer=summarize,
        data_read_root=data_read_root,
        co_run_guard=(lambda: manager.source_training_phase(args.plan))
        if co_run
        else None,
    )
    result = summarize()
    if co_run and all(values["complete"] for values in result["branches"].values()):
        # All branches are done. Release retained feature bytes; checkpoints
        # contain logical source paths, never paths into this temporary view.
        shutil.rmtree(data_read_root)
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)
    if not result["complete"]:
        raise SystemExit(75)


def rescale_windows(plan, experiment, arm):
    """Preserve remaining sequence exposure after changing accumulation.

    The immutable old plan's step counts used its original accumulation. Reusing
    those counts with 4x accumulation would silently train 4x as many sequences.
    Preserve the consumed prefix; rescale only remaining current/future budgets.
    """
    windows = copy.deepcopy(plan["windows"])
    index, start = experiment["window_index"], experiment["start_step"]
    definition = experiment["arm_definitions"][arm]
    accum = definition["gradient_accumulation"]
    source_accum = plan["run"].get("gradient_accumulation", 8)
    step = definition["goal_step"]
    for i in range(index, len(windows)):
        original = plan["windows"][i]
        if i == index:
            remaining = (original["stop_step"] - start) * source_accum - (
                step - start
            ) * accum
        else:
            windows[i]["start_step"] = step
            remaining = (original["stop_step"] - original["start_step"]) * source_accum
        if remaining < 0:
            raise ValueError(
                "recipe has exceeded the planned current-window sample budget"
            )
        step += math.ceil(remaining / accum)
        windows[i]["stop_step"] = step
    return windows


def promote_recipe_plan(plan, root, destination, arm):
    experiment = json.loads((root / "experiment.json").read_text())
    if arm not in experiment["arm_definitions"]:
        raise ValueError("selected promotion arm is absent from the experiment")
    if not summarize_recipe(root, experiment)["complete"]:
        raise ValueError("all named comparison arms must finish before promotion")
    window = plan["windows"][experiment["window_index"]]
    if (
        config_digest(plan["run"]) != experiment["source_config_sha256"]
        or config_digest(window["ids"]) != experiment["window_ids_sha256"]
    ):
        raise ValueError("promotion source config/window differs from the experiment")
    run = json.loads((root / arm / "run.json").read_text())
    definition = experiment["arm_definitions"][arm]
    if config_digest(run) != definition["run_config_sha256"]:
        raise ValueError("promotion arm config differs from the experiment")
    checkpoint = latest_state(run["output"])
    if (
        not checkpoint
        or checkpoint["step"] != definition["goal_step"]
        or checkpoint["data_window"] != window["name"]
    ):
        raise ValueError("selected final optimizer/data checkpoint is unavailable")
    saved = torch.load(
        Path(checkpoint["path"]) / "trainer.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if saved["run_config"] != run:
        raise ValueError("selected checkpoint config differs from its frozen arm")
    if not manager.complete_window(plan, window):
        raise ValueError("current features must remain intact for exact cursor resume")
    if destination.exists():
        raise ValueError("new controller root must not exist")
    windows = rescale_windows(plan, experiment, arm)
    if windows[-1]["stop_step"] > run["total_steps"]:
        raise ValueError("rescaled plan exceeds the immutable total_steps cap")
    promoted = copy.deepcopy(plan)
    promoted.update(
        run=run,
        windows=windows,
        source_config_sha256=source_fingerprint(run),
        objective_fork={
            "experiment": str(root),
            "arm": arm,
            "checkpoint": checkpoint,
            "remaining_windows_rescaled_by_sequence_budget": True,
        },
    )
    for name, payload in (
        ("source.json", run),
        ("run.json", run),
        ("plan.json", promoted),
    ):
        atomic_json(destination / name, payload)
    atomic_json(
        destination / "state.json",
        dict(
            window=experiment["window_index"],
            phase="rotate"
            if checkpoint["step"] >= windows[experiment["window_index"]]["stop_step"]
            else "train",
            complete=False,
        ),
    )
    print(
        json.dumps(
            dict(
                promoted_root=str(destination),
                checkpoint=checkpoint,
                source_config=str(destination / "source.json"),
                note="owner must repoint driver and quality watcher; original controller stays stopped",
            )
        ),
        flush=True,
    )
