"""Isolated, optimizer-preserving lambda ablation; never controls the main job.

The caller owns the GPU hold. Main plan/config/checkpoints are read-only. Optional
capture completion writes only the existing feature cache and this experiment's
capture logs. Training writes only A/B/C outputs. Promotion creates a NEW controller
root; the frozen original root and full checkpoint output remain unchanged.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .forks import config_digest, validate_fork
from .quality_gate import pooled, read_metrics
from .rolling import (
    atomic_json,
    capture_window,
    latest_state,
    source_fingerprint,
    window_ready,
)

PROJECT = Path(os.environ.get("DSPINE_PROJECT", "./work"))
STOP = False
CHILDREN = []


@contextmanager
def experiment_lock(root):
    with (root / "experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another A/B manager or parent pin owns this output"
            ) from error
        yield


def request_stop(*_):
    global STOP
    STOP = True
    for process in CHILDREN:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    # Optional capture uses this module's separately owned child/CID cleanup.
    from . import rolling

    rolling.request_stop()


def require_main_inactive(unit):
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
        capture_output=True,
        text=True,
    )
    state = result.stdout.strip()
    if state in {"active", "activating", "deactivating", "reloading"}:
        raise RuntimeError(
            "main controller is running; wait for its chunk to end, without stopping it"
        )
    if state not in {"inactive", "failed"}:
        raise RuntimeError("cannot verify that the main controller is inactive")


def choose_mode(free_bytes, cap_gb, requested="auto", branches=2):
    single = cap_gb * 1_000_000_000
    if free_bytes < single + 6_000_000_000:
        raise RuntimeError(
            "insufficient free VRAM even for one branch; no other job will be stopped"
        )
    parallel = free_bytes >= branches * single + 6_000_000_000
    if requested == "parallel" and not parallel:
        raise RuntimeError(
            f"parallel mode requires {branches} allocator caps plus6GB free; use auto/sequential"
        )
    return "parallel" if requested != "sequential" and parallel else "sequential"


def free_gpu_bytes():
    # Driver query only: the CPU manager must not acquire a CUDA context, which
    # would block optional target capture and consume branch memory.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = result.stdout.splitlines()
    if len(values) != 1:
        raise ValueError("this bounded A/B launcher requires the configured single-GPU host")
    return int(values[0]) * 1024**2


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(selector, pinned_root=None):
    """Read an explicit checkpoint, output root, or its atomic latest pointer.

    Resolve once after the source controller is inactive. The returned immutable
    directory, not a moving pointer, is recorded in experiment provenance.
    """
    path = Path(selector).expanduser().resolve()
    # Existing immutable parent copies remain usable after source retention
    # removes their original explicit directory. snapshot_parent rehashes the
    # copy before use; moving latest selectors still resolve normally.
    if not path.exists() and pinned_root is not None:
        origin = Path(pinned_root) / "parent/origin.json"
        if origin.is_file() and json.loads(origin.read_text())["path"] == str(path):
            return path
    if path.is_dir() and (path / "trainer.pt").is_file():
        return path
    pointer = path / "latest" if path.is_dir() else path
    if pointer.name != "latest" or not pointer.is_file():
        raise FileNotFoundError(
            "checkpoint selector needs a complete checkpoint or latest pointer"
        )
    name = pointer.read_text().strip()
    if not re.fullmatch(r"step-[0-9]{8}", name):
        raise ValueError("latest pointer must name a single step-NNNNNNNN directory")
    checkpoint = (pointer.parent / name).resolve()
    if checkpoint.parent != pointer.parent.resolve():
        raise ValueError("latest checkpoint escapes its output directory")
    if not checkpoint.is_dir() or not (checkpoint / "trainer.pt").is_file():
        raise FileNotFoundError("latest checkpoint is missing its trainer state")
    return checkpoint


def snapshot_parent(checkpoint, root):
    destination = root / "parent"
    if not destination.exists():
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                "parent checkpoint was pruned; supply its pinned copy with --checkpoint"
            )
        temporary = root / ".parent.partial"
        temporary.mkdir(exist_ok=True)
        for name in (
            "config.json",
            "model.safetensors",
            "trainer.pt",
            "proposal_policy.json",
        ):
            source = checkpoint / name
            if source.exists():
                shutil.copyfile(source, temporary / name)
            elif name != "proposal_policy.json":
                raise ValueError("source parent checkpoint is incomplete")
            else:
                (temporary / name).unlink(missing_ok=True)
        for name in ("config.json", "model.safetensors", "trainer.pt"):
            if not (temporary / name).is_file():
                raise ValueError("parent checkpoint is incomplete")
        atomic_json(
            temporary / "origin.json",
            {
                "path": str(checkpoint),
                "sha256": {
                    name: file_digest(temporary / name)
                    for name in (
                        "config.json",
                        "model.safetensors",
                        "trainer.pt",
                        "proposal_policy.json",
                    )
                    if (temporary / name).exists()
                },
            },
        )
        temporary.rename(destination)
    origin = json.loads((destination / "origin.json").read_text())
    if origin["path"] != str(checkpoint):
        raise ValueError("experiment already has a different parent checkpoint")
    for name, digest in origin["sha256"].items():
        if file_digest(destination / name) != digest:
            raise ValueError("cached parent checkpoint changed")
    return destination


def pin_feature_window(plan, window, root):
    """Owned read-only view of immutable shards, surviving source rotation.

    Hard links allocate no second copy of feature bytes. They retain the old
    window's physical bytes after source purge until this view is removed.
    No source file contents or source directory entries are changed.
    """
    destination = root / "feature-view"
    expected = sorted(
        hashlib.sha256(x.encode()).hexdigest() + ".pt" for x in window["ids"]
    )
    manifest = {
        "window": window["name"],
        "shards": expected,
        "source": str((Path(plan["train_root"]) / "shards").resolve()),
    }
    if destination.exists():
        if json.loads((destination / "manifest.json").read_text()) != manifest:
            raise ValueError("feature view belongs to a different window")
        if sorted(p.name for p in destination.glob("*.pt")) != expected:
            raise ValueError("pinned feature view is incomplete")
        return destination
    temporary = root / ".feature-view.partial"
    temporary.mkdir(exist_ok=True)
    try:
        for name in expected:
            target = temporary / name
            if not target.exists():
                os.link(Path(manifest["source"]) / name, target)
        if sorted(p.name for p in temporary.glob("*.pt")) != expected:
            raise ValueError("partial feature view contains another window")
        atomic_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except OSError as error:
        raise RuntimeError(
            "cannot pin current features (rotation race or cross-filesystem output); "
            "use a fresh experiment on the current window, on the same filesystem"
        ) from error
    return destination


def snapshot_live_parent(checkpoint, root, latest_output=None):
    """Retry a live latest pin if retention removes it during the CPU copy."""
    for attempt in range(3):
        try:
            parent = snapshot_parent(checkpoint, root)
            return parent, checkpoint
        except (FileNotFoundError, ValueError) as error:
            if (
                latest_output is None
                or (root / "parent").exists()
                or attempt == 2
                or (isinstance(error, ValueError) and "incomplete" not in str(error))
            ):
                raise
            checkpoint = resolve_checkpoint(latest_output)
    raise AssertionError("unreachable")


def source_training_phase(plan_path):
    """Yield owned children if a foreign controller begins high-memory capture."""
    return (
        json.loads((Path(plan_path).parent / "state.json").read_text())["phase"]
        == "train"
    )


def branch_configs(run, root, step, allocator_cap_gb=None):
    configs = {}
    for branch, reduction in (
        ("A", "position_sum"),
        ("B", "weighted_token_mean"),
        ("C", "weighted_token_mean"),
    ):
        config = {
            **copy.deepcopy(run),
            "output": str(root / branch),
            "lambda_ce_reduction": reduction,
            "eval_step_offset": step,
            "batch_trace": True,
        }
        if allocator_cap_gb is not None:
            config["cuda_memory_limit_gb"] = allocator_cap_gb
        if branch == "C":
            config.update(
                anchor_sampling="without_replacement",
                refinement_start_position=2,
                mask_unsupported_ce=True,
            )
        validate_fork(run, config, root / "parent")
        path = root / branch / "run.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("branch config changed; use a new experiment output")
        atomic_json(path, config)
        configs[branch] = config
    return configs


def complete_window(plan, window):
    expected = {hashlib.sha256(x.encode()).hexdigest() + ".pt" for x in window["ids"]}
    found = {p.name for p in (Path(plan["train_root"]) / "shards").glob("*.pt")}
    if found - expected:
        raise ValueError("training cache contains shards outside the requested window")
    return window_ready(plan["train_root"], window, plan["run"])


def arm_status(root, branch):
    path = root / branch / "status.json"
    return json.loads(path.read_text()) if path.exists() else {"state": "not_started"}


def write_arm_status(root, branch, state, **details):
    status = {"state": state, "updated_at": time.time(), **details}
    atomic_json(root / branch / "status.json", status)
    return status


def attach_arm_statuses(result, root):
    """Persist failures separately while retaining every healthy arm's results."""
    result["failed_arms"] = {}
    for branch, values in result["branches"].items():
        status = arm_status(root, branch)
        values["process"] = status
        if status["state"] == "failed":
            result["failed_arms"][branch] = status
    return result


def summarize(root, blocks, start_step, goal, arms=("A", "B")):
    records = {b: read_metrics(root / b / "metrics.jsonl") for b in ("A", "B", "C")}
    training = {
        b: {
            r["step"]: r
            for r in rows
            if "data_fingerprint" in r and start_step < r["step"] <= goal
        }
        for b, rows in records.items()
    }
    paired = sorted(training["A"].keys() & training["B"].keys())
    with (root / "paired-metrics.jsonl").open("w") as log:
        for step in paired:
            a, b = training["A"][step], training["B"][step]
            for key in (
                "data_fingerprint",
                "teacher_fingerprint",
                "response_tokens_seen",
                "lr",
                "curriculum_p",
            ):
                if a[key] != b[key]:
                    raise ValueError(
                        f"A/B inputs or schedule differ at step{step}: {key}"
                    )
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "data_fingerprint": a["data_fingerprint"],
                        **{
                            branch + "_" + key: training[branch][step][key]
                            for branch in ("A", "B")
                            for key in (
                                "ce",
                                "ce_token_mean",
                                "l1",
                                "auxiliary",
                                "distillation",
                            )
                        },
                    }
                )
                + "\n"
            )
    # C intentionally changes anchors and therefore teacher-mask RNG consumption.
    # Its sequence order/token schedule still match both controls, across epochs.
    sequence_pairs = {}
    for control in ("A", "B"):
        shared = sorted(training[control].keys() & training["C"].keys())
        for step in shared:
            for key in (
                "sequence_fingerprint",
                "response_tokens_seen",
                "lr",
                "curriculum_p",
            ):
                if training[control][step][key] != training["C"][step][key]:
                    raise ValueError(
                        f"{control}/C sequence or schedule differs at step{step}: {key}"
                    )
        sequence_pairs[control + "_C"] = len(shared)
    result = {
        "paired_steps_verified": len(paired),
        "sequence_paired_steps_verified": sequence_pairs,
        "requested_arms": list(arms),
        "complete": False,
        "branches": {},
    }
    for branch, rows in records.items():
        evaluations = {r["step"]: r for r in rows if "offline_tau" in r}
        result["branches"][branch] = {
            "evaluations": [
                {
                    "step": step,
                    "branch_updates": step - start_step,
                    "complete_tau": {
                        str(b): pooled(r["offline_tau"], b) for b in blocks
                    },
                    "opening_tau": {
                        str(b): pooled(r["offline_first_anchor"], b) for b in blocks
                    },
                    "t0_tau": {str(b): pooled(r["offline_tau_t0"], b) for b in blocks},
                }
                for step, r in sorted(evaluations.items())
            ],
            "last_step": max(training[branch], default=start_step),
            "last250_means": {
                key: sum(
                    training[branch][s][key] for s in sorted(training[branch])[-250:]
                )
                / min(250, len(training[branch]))
                for key in (
                    "ce",
                    "ce_token_mean",
                    "l1",
                    "auxiliary",
                    "distillation",
                    "seconds_per_step",
                )
            }
            if training[branch]
            else {},
        }
    result["complete"] = all(
        len(training[b]) == goal - start_step
        and goal in {r["step"] for r in records[b] if "offline_tau" in r}
        for b in arms
    )
    result["decision"] = (
        "manual_review" if result["complete"] else "incomplete_do_not_promote"
    )
    attach_arm_statuses(result, root)
    atomic_json(root / "comparison.json", result)
    return result


def train_group(
    branches,
    configs,
    root,
    parent,
    window,
    goal,
    deadline,
    *,
    reset_data=True,
    summarizer=None,
    data_read_root=None,
    co_run_guard=None,
):
    handles, processes, failed_starts = [], [], []
    try:
        for branch in branches:
            if co_run_guard is not None and not co_run_guard():
                print(
                    json.dumps(
                        {"co_run_yield": "source controller left training phase"}
                    ),
                    flush=True,
                )
                request_stop()
                break
            config = configs[branch]
            branch_goal = goal[branch] if isinstance(goal, dict) else goal
            previous = latest_state(config["output"])
            evaluated = any(
                row.get("step") == branch_goal and "offline_tau" in row
                for row in read_metrics(root / branch / "metrics.jsonl")
            )
            if previous and previous["step"] >= branch_goal and evaluated:
                write_arm_status(
                    root,
                    branch,
                    "succeeded",
                    goal_step=branch_goal,
                    checkpoint=previous["path"],
                    exit_code=0,
                )
                continue
            command = [
                sys.executable,
                "-u",
                "-m",
                "dspine.train",
                "--config",
                str(root / branch / "run.json"),
                "--data-window",
                window["name"],
                "--stop-at-step",
                str(branch_goal),
                "--max-minutes",
                str(max(0, (deadline - time.monotonic() - 120) / 60)),
            ]
            command += (
                ["--resume", "latest"]
                if previous
                else ["--fork-from", str(parent)]
                + (["--reset-data"] if reset_data else [])
            )
            if data_read_root is not None:
                command += ["--data-read-root", str(data_read_root)]
            try:
                handle = (root / branch / "train.log").open("a", buffering=1)
                handles.append(handle)
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).resolve().parent.parent,
                    start_new_session=True,
                )
            except OSError as error:
                failed_starts.append(branch)
                status = write_arm_status(
                    root,
                    branch,
                    "failed",
                    error=str(error),
                    goal_step=branch_goal,
                    exit_code=None,
                )
                print(json.dumps({"branch_failed": branch, **status}), flush=True)
                continue
            processes.append((branch, process))
            CHILDREN.append(process)
            write_arm_status(
                root,
                branch,
                "running",
                pid=process.pid,
                goal_step=branch_goal,
                exit_code=None,
            )
            print(
                json.dumps({"branch_started": branch, "pid": process.pid}), flush=True
            )
        notified = set()
        last_check, last_eval = 0.0, None
        while any(p.poll() is None for _, p in processes):
            if not STOP and co_run_guard is not None and not co_run_guard():
                print(
                    json.dumps(
                        {"co_run_yield": "source controller left training phase"}
                    ),
                    flush=True,
                )
                request_stop()
            if time.monotonic() - last_check >= 15:
                progress = (
                    summarizer()
                    if summarizer is not None
                    else summarize(
                        root,
                        next(iter(configs.values())).get("selection_blocks", [6, 8]),
                        window["start_step"],
                        goal,
                        arms=branches,
                    )
                )
                newest = {
                    b: progress["branches"][b]["evaluations"][-1]
                    for b in branches
                    if progress["branches"][b]["evaluations"]
                }
                signature = tuple((b, row["step"]) for b, row in newest.items())
                if signature and signature != last_eval:
                    last_eval = signature
                    print(
                        json.dumps({"latest_evaluations": newest}),
                        flush=True,
                    )
                last_check = time.monotonic()
            if STOP or time.monotonic() >= deadline - 60:
                for _, process in processes:
                    if process.poll() is None:
                        process.send_signal(signal.SIGTERM)
            for branch, process in processes:
                if process.poll() is not None and branch not in notified:
                    write_arm_status(
                        root,
                        branch,
                        "failed" if process.returncode else "succeeded",
                        pid=process.pid,
                        exit_code=process.returncode,
                        goal_step=goal[branch] if isinstance(goal, dict) else goal,
                        train_log=str(root / branch / "train.log"),
                    )
                    print(
                        json.dumps(
                            {"branch_finished": branch, "exit_code": process.returncode}
                        ),
                        flush=True,
                    )
                    notified.add(branch)
            time.sleep(2)
        # A fast failure (including a single-arm group) can exit before the
        # polling loop runs. Always publish its terminal receipt here too.
        for branch, process in processes:
            if branch not in notified:
                write_arm_status(
                    root,
                    branch,
                    "failed" if process.returncode else "succeeded",
                    pid=process.pid,
                    exit_code=process.returncode,
                    goal_step=goal[branch] if isinstance(goal, dict) else goal,
                    train_log=str(root / branch / "train.log"),
                )
                print(
                    json.dumps(
                        {"branch_finished": branch, "exit_code": process.returncode}
                    ),
                    flush=True,
                )
        failed = failed_starts + [
            branch for branch, process in processes if process.returncode
        ]
        return failed
    finally:
        # A launch/log/manager failure must not orphan the already-started peer.
        # Popen handles identify only this experiment's own children.
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for _, process in processes:
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def run_groups(groups, configs, root, parent, window, goals, deadline, **kwargs):
    """Finish independent peers before reporting/retrying an individual failure."""
    failures = set()
    for group in groups:
        if STOP or time.monotonic() >= deadline - 120:
            break
        failed = train_group(
            group, configs, root, parent, window, goals, deadline, **kwargs
        )
        for branch in failed:
            log = root / branch / "train.log"
            oom = log.exists() and "CUDA out of memory" in log.read_text()
            if (
                len(group) > 1
                and oom
                and not STOP
                and time.monotonic() < deadline - 120
            ):
                print(
                    json.dumps(
                        {"oom_retry": branch, "mode": "sequential_after_peers_finish"}
                    ),
                    flush=True,
                )
                if not train_group(
                    (branch,), configs, root, parent, window, goals, deadline, **kwargs
                ):
                    continue
            failures.add(branch)
    return sorted(failures)


def promote_plan(plan, root, destination, arm="B"):
    """Prepare a new immutable controller fork, never rewrite the original run."""
    experiment = json.loads((root / "experiment.json").read_text())
    if experiment.get("kind") == "recipe_ablation":
        from .recipe_ablation import promote_recipe_plan

        return promote_recipe_plan(plan, root, destination, arm)
    if arm not in ("B", "C"):
        raise ValueError("legacy experiment promotion requires arm B or C")
    result = summarize(
        root,
        plan["run"].get("selection_blocks", [6, 8]),
        experiment["start_step"],
        experiment["goal_step"],
        arms=("A", "B", "C") if arm == "C" else ("A", "B"),
    )
    if not result["complete"]:
        raise ValueError(
            "all comparison branches must finish with verified paired inputs before promotion"
        )
    run = json.loads((root / arm / "run.json").read_text())
    checkpoint = latest_state(run["output"])
    window = plan["windows"][experiment["window_index"]]
    if (
        config_digest(plan["run"]) != experiment["source_config_sha256"]
        or config_digest(window["ids"]) != experiment["window_ids_sha256"]
    ):
        raise ValueError("promotion source config/window differs from the experiment")
    if (
        not checkpoint
        or checkpoint["step"] != experiment["goal_step"]
        or checkpoint["data_window"] != window["name"]
    ):
        raise ValueError(f"{arm} final optimizer/data checkpoint is unavailable")
    if not complete_window(plan, window):
        raise ValueError(
            "the current window must remain intact for exact branch cursor resume"
        )
    if destination.exists():
        raise ValueError("new controller root must not exist")
    promoted = copy.deepcopy(plan)
    promoted.update(
        run=run,
        source_config_sha256=source_fingerprint(run),
        objective_fork={"experiment": str(root), "checkpoint": checkpoint},
    )
    atomic_json(destination / "source.json", run)
    atomic_json(destination / "run.json", run)
    atomic_json(destination / "plan.json", promoted)
    atomic_json(
        destination / "state.json",
        {
            "window": experiment["window_index"],
            "phase": "rotate" if checkpoint["step"] >= window["stop_step"] else "train",
            "complete": False,
        },
    )
    print(
        json.dumps(
            {
                "promoted_root": str(destination),
                "source_config": str(destination / "source.json"),
                "checkpoint": checkpoint,
                "note": "owner must repoint the detached driver; original controller stays stopped",
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PROJECT / "runs/rolling/plan.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "ckpt/full/step-00004516",
        help="checkpoint directory, output/latest pointer, output root, or 'latest' from --plan",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "ckpt/ab-lambda")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=1500, help="optimizer updates per selected arm"
    )
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument(
        "--recipe-file", type=Path, help="named arm override dictionaries"
    )
    parser.add_argument("--base-recipe", choices=("A", "B", "C"))
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--equal-samples", action="store_true")
    parser.add_argument("--reference-arm", default=None)
    parser.add_argument(
        "--allow-active-main",
        action="store_true",
        help="recipe-only co-run; pin immutable current features, never control main",
    )
    parser.add_argument(
        "--parallel-arms",
        type=int,
        default=None,
        help="maximum experiment children at once (foreign jobs are left alone)",
    )
    parser.add_argument(
        "--allocator-cap-gb",
        type=float,
        default=None,
        help="same optional CUDA allocator cap for every experiment arm",
    )
    parser.add_argument(
        "--mode", choices=("auto", "parallel", "sequential"), default="auto"
    )
    parser.add_argument("--max-minutes", type=float, default=155)
    parser.add_argument("--finish-capture", action="store_true")
    parser.add_argument("--capture-max-minutes", type=float, default=40)
    parser.add_argument("--promote-root", type=Path)
    parser.add_argument("--promote-arm", default="B")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="CPU-only parent pin; allowed while main capture/training runs",
    )
    args = parser.parse_args()
    args.checkpoint_selector = args.checkpoint
    if args.parallel_arms is not None and args.parallel_arms < 1:
        parser.error("--parallel-arms must be positive")
    if args.allow_active_main and (
        not args.recipe_file
        or args.finish_capture
        or args.promote_root
        or args.snapshot_only
    ):
        parser.error(
            "--allow-active-main requires a recipe training fork, without capture/promotion"
        )
    if args.parallel_arms is not None and not args.recipe_file:
        parser.error("--parallel-arms requires --recipe-file")
    if args.arms and len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    if not args.recipe_file:
        args.arms = args.arms or ["A", "B"]
        if not set(args.arms) <= {"A", "B", "C"}:
            parser.error("custom arm names require --recipe-file")
        if (
            args.base_recipe
            or args.eval_every is not None
            or args.equal_samples
            or args.reference_arm
        ):
            parser.error("recipe options require --recipe-file")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    if args.snapshot_only:
        args.checkpoint = resolve_checkpoint(
            Path(json.loads(args.plan.read_text())["run"]["output"])
            if args.checkpoint == Path("latest")
            else args.checkpoint,
            pinned_root=args.output.resolve(),
        )
        root = args.output.resolve()
        for protected in (args.checkpoint.resolve().parent, args.plan.resolve().parent):
            if (
                root == protected
                or root.is_relative_to(protected)
                or protected.is_relative_to(root)
            ):
                raise ValueError(
                    "snapshot output overlaps the source checkpoint/controller output"
                )
        root.mkdir(parents=True, exist_ok=True)
        with experiment_lock(root):
            parent = snapshot_parent(args.checkpoint.resolve(), root)
        print(
            json.dumps({"parent_snapshot": str(parent), "cpu_only": True}), flush=True
        )
        return
    if not args.allow_active_main:
        require_main_inactive("ai-dspine-rolling.service")
    plan = json.loads(args.plan.read_text())
    root = args.output.resolve()
    for protected in (
        args.plan.parent,
        Path(plan["run"]["output"]),
        Path(plan["feature_root"]),
    ):
        protected = protected.resolve()
        if (
            root == protected
            or root.is_relative_to(protected)
            or protected.is_relative_to(root)
        ):
            raise ValueError("A/B output overlaps a protected source directory")
    if args.promote_root:
        destination = args.promote_root.resolve()
        for protected in (
            args.plan.parent,
            Path(plan["run"]["output"]),
            Path(plan["feature_root"]),
            root,
        ):
            protected = protected.resolve()
            if (
                destination == protected
                or destination.is_relative_to(protected)
                or protected.is_relative_to(destination)
            ):
                raise ValueError(
                    "new controller root overlaps a protected source/experiment directory"
                )
        with experiment_lock(root):
            promote_plan(plan, root, destination, args.promote_arm)
        return
    if (
        args.allow_active_main
        and (root / "parent/origin.json").exists()
        and args.checkpoint == Path("latest")
    ):
        args.checkpoint = Path(
            json.loads((root / "parent/origin.json").read_text())["path"]
        )
    args.checkpoint = resolve_checkpoint(
        Path(plan["run"]["output"])
        if args.checkpoint == Path("latest")
        else args.checkpoint,
        pinned_root=root,
    )
    root.mkdir(parents=True, exist_ok=True)
    with experiment_lock(root):
        run_experiment(args, plan, root)


def run_experiment(args, plan, root):
    if args.recipe_file:
        from .recipe_ablation import run_recipe_experiment

        return run_recipe_experiment(args, plan, root)
    deadline = time.monotonic() + args.max_minutes * 60
    window = plan["windows"][args.window]
    state = json.loads((args.plan.parent / "state.json").read_text())
    if state["window"] != args.window or state["phase"] not in {"capture", "train"}:
        raise ValueError("main is not paused on the requested feature window")
    ready = complete_window(plan, window)
    atomic_json(
        root / "feature-readiness.json", {"window": window["name"], "complete": ready}
    )
    if not ready and args.finish_capture and not STOP:
        seconds = min(
            args.capture_max_minutes * 60, deadline - time.monotonic() - 80 * 60
        )
        if seconds < 180:
            raise RuntimeError(
                "insufficient bounded time for capture plus both branches"
            )
        capture_window(plan, window, plan["train_root"], root, seconds)
        ready = complete_window(plan, window)
        atomic_json(
            root / "feature-readiness.json",
            {"window": window["name"], "complete": ready},
        )
    if not ready:
        print(
            json.dumps(
                {
                    "not_ready": window["name"],
                    "gpu_training_started": False,
                    "next": "relaunch this same job with --finish-capture in an owner-held slot; it finishes ONLY missing features, then A/B. Allow up to40min extra. No partial-window ablation is run.",
                }
            ),
            flush=True,
        )
        raise SystemExit(75)
    if STOP:
        raise SystemExit(75)
    required_disk = (len(args.arms) * 40 + 10) * 1_000_000_000
    if shutil.disk_usage(root).free < required_disk:
        raise RuntimeError(
            f"selected arms need {required_disk / 1e9:g}GB free for checkpoints and protected winners"
        )
    parent = snapshot_parent(args.checkpoint.resolve(), root)
    saved = torch.load(
        parent / "trainer.pt", map_location="cpu", weights_only=False, mmap=True
    )
    run = plan["run"]
    if not (
        run.get("packed")
        and run.get("eval_mode") == "rounds"
        and run.get("eval_before_training")
        and run.get("eval_report_first_anchor")
        and run.get("eval_require_decode_probs")
        and run.get("eval_greedy_features")
        and run.get("eval_every", 0) > 0
    ):
        raise ValueError(
            "A/B requires packed training and the complete strict-decode T1/opening/T0 evaluation suite"
        )
    if saved["run_config"] != run or saved["step"] != window["start_step"]:
        raise ValueError(
            "source must be the unchanged checkpoint at this window's start"
        )
    start, goal = saved["step"], saved["step"] + args.steps
    if args.steps < 1 or goal > window["stop_step"]:
        raise ValueError("requested A/B updates exceed the current window")
    experiment = {
        "start_step": start,
        "goal_step": goal,
        "window_index": args.window,
        "window_ids_sha256": config_digest(window["ids"]),
        "source_config_sha256": config_digest(run),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "code_sha256": {
            name: file_digest(Path(__file__).parent / name)
            for name in (
                "train.py",
                "losses.py",
                "modeling.py",
                "data.py",
                "forks.py",
                "ab_lambda.py",
            )
        },
    }
    if (root / "experiment.json").exists() and json.loads(
        (root / "experiment.json").read_text()
    ) != experiment:
        raise ValueError("experiment definition changed; choose a new output")
    atomic_json(root / "experiment.json", experiment)
    configs = branch_configs(run, root, start, args.allocator_cap_gb)
    mode = choose_mode(
        free_gpu_bytes(),
        configs["A"].get("cuda_memory_limit_gb", 36),
        args.mode,
        branches=len(args.arms),
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "arms": args.arms,
                "updates_per_branch": args.steps,
                "goal": goal,
            }
        ),
        flush=True,
    )
    groups = [tuple(args.arms)] if mode == "parallel" else [(b,) for b in args.arms]
    failures = run_groups(groups, configs, root, parent, window, goal, deadline)
    result = summarize(
        root, run.get("selection_blocks", [6, 8]), start, goal, arms=args.arms
    )
    print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(1)
    if not result["complete"]:
        raise SystemExit(75)


if __name__ == "__main__":
    # run_module.py uses runpy's __main__ namespace. Enter through the canonical
    # module so recipe_ablation and the signal handler share STOP/CHILDREN.
    from dspine.ab_lambda import main as canonical_main

    canonical_main()
