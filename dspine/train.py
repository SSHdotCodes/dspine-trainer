"""Single-GPU BF16 DSpine trainer with FP32 trainable parameters/Adam states.

Run as a module from the repository parent: python -m dspine.train --config ...
Atomic checkpoints retain optimizer, Python/NumPy/Torch/CUDA RNG and data cursor.
SIGTERM requests a checkpoint and clean exit at the next optimizer step.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .config import DSpineConfig
from .data import (
    AnchorStream,
    PackedSequenceStream,
    collate_anchors,
    dequantize_features,
)
from .eval_offline import evaluate
from .eval_rounds import domain_balanced_paths, evaluate_rounds
from .forks import validate_fork, validate_recipe_values
from .losses import curriculum_probability, lr_multiplier, training_loss
from .modeling import DSpineDraft
from .transfer_space import load_target_weights

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def atomic_torch_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(".partial")
    torch.save(payload, temp)
    os.replace(temp, path)


def fsync_path(path):
    """Make a completed checkpoint file or directory durable before pruning."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        # CPU jobs must not create a CUDA context merely to checkpoint RNG.
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(root, model, optimizer, step, stream, config, elapsed):
    root = Path(root)
    path = root / f"step-{step:08d}"
    temp = root / f".step-{step:08d}.partial"
    temp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp)
    atomic_torch_save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "rng": rng_state(),
            "data": stream.state_dict(),
            "data_window": getattr(stream, "data_window", None),
            "run_config": config,
            "elapsed_seconds": elapsed,
            "response_tokens_seen": getattr(stream, "response_tokens_seen", 0),
            "fork_provenance": getattr(stream, "fork_provenance", None),
        },
        temp / "trainer.pt",
    )
    if path.exists():
        # A signal immediately after periodic save may request the same step.
        import shutil

        shutil.rmtree(temp)
    else:
        for item in temp.iterdir():
            fsync_path(item)
        fsync_path(temp)
        os.replace(temp, path)
        fsync_path(root)
    pointer = root / "latest.tmp"
    pointer.write_text(path.name + "\n")
    fsync_path(pointer)
    os.replace(pointer, root / "latest")
    fsync_path(root)
    # Keep bounded disk use during multi-day runs. Only prune complete checkpoints
    # created by this trainer, after the new checkpoint and pointer are durable.
    import re
    import shutil

    complete = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and re.fullmatch(r"step-[0-9]{8}", p.name)
        and (p / "trainer.pt").is_file()
    )
    keep = max(1, config.get("checkpoint_keep", 3))
    for old in complete[:-keep]:
        shutil.rmtree(old)
    return path


def acceptance_summary(results, blocks=(6, 8)):
    """Pool each serving block's tau by its number of evaluated domain anchors."""
    summary = {}
    for block in blocks:
        values = [
            domain[str(block)] for domain in results.values() if str(block) in domain
        ]
        anchors = sum(value["anchors"] for value in values)
        if anchors:
            summary[str(block)] = {
                "tau": sum(value["tau"] * value["anchors"] for value in values)
                / anchors,
                "anchors": anchors,
            }
    return summary


def save_best(root, model, step, results, blocks=(6, 8)):
    """Retain independent b=6/b=8 winners, sharing weights when their step matches.

    ``best.json`` indexes both per-block pointers. A former single-block pointer
    is migrated when first encountered. Only superseded, unreferenced weights
    created by this trainer are pruned; optimizer checkpoint retention is separate.
    """
    import re
    import shutil

    root = Path(root)
    summary = acceptance_summary(results, blocks)
    index_path = root / "best.json"
    old_index = json.loads(index_path.read_text()) if index_path.exists() else {}
    previous = {}
    for block in summary:
        pointer = root / f"best-b{block}.json"
        if pointer.exists():
            previous[block] = json.loads(pointer.read_text())
        elif str(old_index.get("block")) == block:
            previous[block] = old_index
            temporary = pointer.with_suffix(".tmp")
            temporary.write_text(json.dumps(old_index) + "\n")
            os.replace(temporary, pointer)
    improved = {
        block: value
        for block, value in summary.items()
        if block not in previous or value["tau"] > previous[block]["tau"]
    }
    if not improved:
        return summary

    name = f"best-step-{step:08d}"
    destination = root / name
    if not destination.exists():
        temporary = root / f".{name}.partial"
        model.save_pretrained(temporary)
        os.replace(temporary, destination)
    for block, value in improved.items():
        metadata = {"step": step, **value, "block": int(block), "path": name}
        pointer = root / f"best-b{block}.json"
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata) + "\n")
        os.replace(temporary, pointer)

    # Include every pointer when checking references, so no other block's winner
    # is removed if a later run selects a different set of serving block sizes.
    pointers = {
        str(value["block"]): value
        for pointer in root.glob("best-b*.json")
        for value in [json.loads(pointer.read_text())]
    }
    temporary = index_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"by_block": pointers}, indent=2) + "\n")
    os.replace(temporary, index_path)
    referenced = {value["path"] for value in pointers.values()}
    for old in previous.values():
        if old["path"] not in referenced and re.fullmatch(
            r"best-step-[0-9]{8}", old["path"]
        ):
            directory = root / old["path"]
            if directory.is_dir():
                shutil.rmtree(directory)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--fork-from",
        help="explicit objective fork, restoring optimizer/RNG from this checkpoint",
    )
    p.add_argument("--max-minutes", type=float, default=165)
    p.add_argument("--data-window", default=None)
    p.add_argument(
        "--reset-data",
        action="store_true",
        help="start a new rolling data window, preserving optimizer and global RNG",
    )
    p.add_argument("--stop-at-step", type=int, default=None)
    p.add_argument(
        "--data-read-root",
        type=Path,
        help="owned immutable shard view; cursor paths still refer to config features",
    )
    args = p.parse_args()
    if args.fork_from and args.resume:
        p.error("--fork-from and --resume are mutually exclusive")
    run = json.loads(Path(args.config).read_text())
    validate_recipe_values(run)
    if run.get("sample_trace") and not run.get("packed"):
        raise ValueError("per-sequence sample traces require packed training")
    arch = DSpineConfig.from_dict(run["model"])
    if "context_fusion" in run:
        arch.context_fusion = run["context_fusion"]
    fork_state = None
    fork_provenance = None
    if args.fork_from:
        fork_state = torch.load(
            Path(args.fork_from) / "trainer.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        # Validate isolation before writing even the process marker or loading CUDA.
        fork_provenance = validate_fork(fork_state["run_config"], run, args.fork_from)
    if run.get("batch_trace") and not run.get("packed"):
        raise ValueError("batch fingerprints require packed sequence training")
    root = Path(run["output"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "pilot-no-go.json").exists():
        raise RuntimeError(
            "pilot quality gate paused this run; diagnose before resuming"
        )
    (root / "trainer-process.json").write_text(
        json.dumps({"pid": os.getpid(), "config": str(Path(args.config).resolve())})
        + "\n"
    )
    device = torch.device(run.get("device", "cuda"))
    if device.type == "cuda" and run.get("cuda_memory_limit_gb"):
        limit = run["cuda_memory_limit_gb"] * 1_000_000_000
        # set_per_process_memory_fraction rejects an index-less torch.device("cuda").
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, limit / torch.cuda.get_device_properties(index).total_memory),
            index,
        )
    seed = run.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(run.get("cpu_threads", 8))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    emb, head = load_target_weights(run["target"], run.get("target_head_quantization"))
    resume = args.resume or args.fork_from
    if args.resume == "latest":
        pointer = root / "latest"
        resume = str(root / pointer.read_text().strip()) if pointer.exists() else None
    if resume:
        source_arch = DSpineConfig.load(Path(resume) / "config.json")
        model = DSpineDraft.from_pretrained(
            resume,
            emb,
            head,
            trainable_dtype=torch.float32,
            context_fusion=arch.context_fusion if args.fork_from else None,
        )
        if model.config != arch:
            raise ValueError("resume architecture differs from run config")
    elif run.get("init_checkpoint"):
        model = DSpineDraft.from_pretrained(
            run["init_checkpoint"],
            emb,
            head,
            trainable_dtype=torch.float32,
            context_fusion=arch.context_fusion,
        )
        model.config.gradient_checkpointing = arch.gradient_checkpointing
        if model.config != arch:
            raise ValueError("init architecture differs from run config")
    else:
        space = torch.load(run["transfer_space"], map_location="cpu", weights_only=True)
        if space.get("head_quantization") != run.get("target_head_quantization"):
            raise ValueError("transfer-space and frozen-head quantization differ")
        model = DSpineDraft(arch, emb, head, space["codebook"], space["basis"])
    if run.get("proposal_vocabulary"):
        from .hot_vocab import load_hot_ids

        policy = run["proposal_vocabulary"]
        hot_ids = load_hot_ids(policy["path"], arch.vocab_size)
        if policy.get("sorted_ids_sha256"):
            if (
                hashlib.sha256(hot_ids.numpy().tobytes()).hexdigest()
                != policy["sorted_ids_sha256"]
            ):
                raise ValueError("proposal vocabulary fingerprint changed")
        previous_policy = model.proposal_policy
        model.configure_proposal_vocabulary(
            hot_ids,
            policy.get("quantization", "row_int8"),
            policy.get("loss", "restricted"),
        )
        if resume and previous_policy != model.proposal_policy:
            raise ValueError("exact resume requires an unchanged proposal policy")
    model.to(device=device)
    if device.type == "cpu":
        model.float()
    # FP32 master params and optimizer moments; frozen head/embedding stay BF16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.train()
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=run.get("lr", 6e-4),
        weight_decay=0.0,
        betas=(0.9, 0.999),
        fused=(device.type == "cuda"),
    )
    stream_class = PackedSequenceStream if run.get("packed", False) else AnchorStream
    stream_seed = (
        seed
        if args.data_window is None
        else seed
        ^ int.from_bytes(
            hashlib.sha256(args.data_window.encode()).digest()[:4], "little"
        )
    )
    stream = stream_class(
        run["features"],
        run.get("anchors_per_sequence", 32),
        arch.block_size,
        arch.context_window,
        stream_seed,
        expected_layer_ids=arch.target_layer_ids,
        expected_revision=run.get("target_revision"),
        exclude_ids=run.get("exclude_ids"),
        anchor_sampling=run.get("anchor_sampling", "with_replacement"),
        read_root=args.data_read_root,
    )
    stream.data_window = args.data_window
    stream.cold_data = run.get("benchmark_cold_data", False)
    step = 0
    previous_elapsed = 0.0
    stream.response_tokens_seen = 0
    if run.get("schedule_response_tokens") and not run.get("packed"):
        raise ValueError("token-progress schedules require packed sequence training")
    if resume:
        state = (
            fork_state
            if fork_state is not None
            else torch.load(
                Path(resume) / "trainer.pt", map_location="cpu", weights_only=False
            )
        )
        if args.fork_from:
            stream.fork_provenance = {
                **fork_provenance,
                "parent_step": state["step"],
                "previous_fork": state.get("fork_provenance"),
            }
            (root / "fork.json").write_text(
                json.dumps(stream.fork_provenance, indent=2) + "\n"
            )
        elif state["run_config"] != run:
            raise ValueError("exact resume requires identical run config")
        else:
            stream.fork_provenance = state.get("fork_provenance")
        from .forks import restore_optimizer

        restore_optimizer(
            optimizer,
            state["optimizer"],
            model,
            new_context_fusion=bool(args.fork_from)
            and source_arch.context_fusion == "off"
            and arch.context_fusion != "off",
        )
        changed = state.get("data_window") != args.data_window
        if changed != args.reset_data:
            raise ValueError(
                "data-window changes require --reset-data, which must not reset an unchanged window"
            )
        if not changed:
            stream.load_state_dict(state["data"])
        restore_rng(state["rng"])
        step = state["step"]
        previous_elapsed = state["elapsed_seconds"]
        stream.response_tokens_seen = state.get("response_tokens_seen", 0)
    # Compile the tensor-only layer functions, leaving cursor/checkpoints/eval eager.
    if run.get("compile", False):
        for layer in model.layers:
            layer.forward = torch.compile(layer.forward, dynamic=False)
    total = run["total_steps"]
    batch_size = run.get("batch_size", 4)
    accum = run.get("gradient_accumulation", 8)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    started = time.monotonic()
    last_save = started
    last_evaluated = -1
    model.train()
    amp = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (root / "metrics.jsonl").open("a", buffering=1) as log:

        def evaluate_now():
            nonlocal last_evaluated
            begin_eval = time.monotonic()

            def measured_eval(stream, temperature=1.0):
                if run.get("eval_mode", "first_anchor") != "rounds":
                    return evaluate(
                        model,
                        stream,
                        run.get("eval_anchors", 128),
                        batch_size=min(batch_size, 4),
                        temperature=temperature,
                        monte_carlo=run.get("eval_monte_carlo", False),
                        require_decode_probs=run.get(
                            "eval_require_decode_probs", False
                        ),
                    )
                walked = evaluate_rounds(
                    model,
                    stream.root,
                    blocks=run.get("selection_blocks", [6, 8]),
                    temperature=temperature,
                    sequences=run.get("eval_sequences", 18),
                    seed=seed,
                    exclude_ids=run.get("exclude_ids"),
                    expected_revision=run.get("target_revision"),
                    require_decode_probs=run.get("eval_require_decode_probs", False),
                    packed_chunk_size=run.get("eval_packed_chunk_size", 0),
                )
                return {
                    domain: {
                        b: {"tau": stats["tau"], "anchors": stats["rounds"]}
                        for b, stats in values.items()
                        if stats["rounds"]
                    }
                    for domain, values in walked["by_domain"].items()
                }

            val = AnchorStream(
                run.get("eval_features", run["features"]),
                run.get("anchors_per_sequence", 32),
                arch.block_size,
                arch.context_window,
                seed,
                split="validation",
                anchor_policy="first",
                expected_layer_ids=arch.target_layer_ids,
                expected_revision=run.get("target_revision"),
                exclude_ids=run.get("exclude_ids"),
            )
            with amp():
                results = measured_eval(val)
            selection_blocks = run.get(
                "selection_blocks",
                [run["selection_block"]] if "selection_block" in run else [6, 8],
            )
            selection = save_best(root, model, step, results, selection_blocks)
            record = {
                "step": step,
                "offline_tau": results,
                "selection_tau": selection,
                "evaluation_mode": run.get("eval_mode", "first_anchor"),
            }
            if run.get("eval_report_first_anchor"):
                # Use exactly the same domain-balanced prompt cohort as the
                # complete-round metric, independent of continuation length.
                cohort = domain_balanced_paths(val.paths, seed)[
                    : run.get("eval_sequences", 18)
                ]
                val.paths = [str(path) for path in cohort]
                val.index = 0
                with amp():
                    opening = evaluate(
                        model,
                        val,
                        anchors=len(cohort),
                        batch_size=1,
                        require_decode_probs=run.get(
                            "eval_require_decode_probs", False
                        ),
                    )
                record["offline_first_anchor"] = {
                    domain: {
                        b: stats
                        for b, stats in values.items()
                        if int(b) in selection_blocks
                    }
                    for domain, values in opening.items()
                }
                record["first_anchor_estimator"] = "rao_blackwell_same_prompt_cohort"
            if run.get("eval_greedy_features"):
                greedy = AnchorStream(
                    run["eval_greedy_features"],
                    block_size=arch.block_size,
                    context_window=arch.context_window,
                    seed=seed,
                    split="validation",
                    anchor_policy="first",
                    expected_layer_ids=arch.target_layer_ids,
                    expected_revision=run.get("target_revision"),
                    exclude_ids=run.get("exclude_ids"),
                )
                with amp():
                    record["offline_tau_t0"] = measured_eval(greedy, temperature=0)
            long_every = run.get("eval_long_every", 2000)
            if (
                run.get("eval_long_features")
                and long_every > 0
                and (step == total or step % long_every == 0)
            ):
                with amp():
                    record["offline_long_rounds"] = evaluate_rounds(
                        model,
                        run["eval_long_features"],
                        blocks=selection_blocks,
                        sequences=run.get("eval_long_sequences", 4),
                        seed=seed,
                        exclude_ids=run.get("exclude_ids"),
                        expected_revision=run.get("target_revision"),
                        require_decode_probs=run.get(
                            "eval_require_decode_probs", False
                        ),
                        packed_chunk_size=run.get("eval_long_packed_chunk_size", 64),
                        max_seconds=run.get("eval_long_max_seconds", 120),
                        collect_diagnostics=True,
                    )
            record["evaluation_seconds"] = time.monotonic() - begin_eval
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            last_evaluated = step

        if run.get("eval_before_training", False) and (step == 0 or args.fork_from):
            evaluate_now()
        stop_at = total if args.stop_at_step is None else min(total, args.stop_at_step)
        while step < stop_at and not STOP:
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            data_trace = hashlib.sha256() if run.get("batch_trace") else None
            teacher_trace = hashlib.sha256() if run.get("batch_trace") else None
            sequence_trace = hashlib.sha256() if run.get("batch_trace") else None
            actual_anchors = 0
            anchor_slots = 0
            sample_traces = [] if run.get("sample_trace") else None
            schedule_step = (
                total
                * min(
                    1.0, stream.response_tokens_seen / run["schedule_response_tokens"]
                )
                if run.get("schedule_response_tokens")
                else step
            )
            probability = run.get("curriculum_p")
            if probability is None:
                probability = curriculum_probability(schedule_step / total)
            for _ in range(accum):
                packed = run.get("packed", False)
                if packed:
                    row = next(stream)
                    if data_trace is not None or sample_traces is not None:
                        sequence_key = json.dumps(
                            [row["id"], row["sequence_start"]]
                        ).encode()
                        anchor_bytes = (
                            row["anchor_offsets"][row["anchor_valid"]].numpy().tobytes()
                        )
                    if data_trace is not None:
                        data_trace.update(sequence_key)
                        sequence_trace.update(sequence_key)
                        data_trace.update(anchor_bytes)
                    stream.response_tokens_seen += row["sequence_response_tokens"]
                    batch = {
                        k: v.to(device, non_blocking=True)
                        for k, v in row.items()
                        if isinstance(v, torch.Tensor)
                    }
                    batch["context_features"] = dequantize_features(
                        row["context_payload"],
                        device=device,
                        dtype=torch.bfloat16
                        if device.type == "cuda"
                        else torch.float32,
                    )
                    count = len(batch["anchor_offsets"])
                    real_count = int(row["anchor_valid"].sum())
                else:
                    rows = [next(stream) for _ in range(batch_size)]
                    batch = {
                        k: v.to(device, non_blocking=True)
                        for k, v in collate_anchors(rows, arch.block_size).items()
                    }
                    count = batch_size
                    real_count = count
                actual_anchors += real_count
                anchor_slots += count
                dtype = (
                    next(model.fc.parameters()).dtype
                    if device.type == "cpu"
                    else torch.bfloat16
                )
                batch["context_features"] = batch["context_features"].to(dtype)
                real_teacher = torch.rand(real_count, device=device) < probability
                teacher = (
                    real_teacher
                    if real_count == count
                    else torch.cat(
                        (
                            real_teacher,
                            torch.zeros(
                                count - real_count, dtype=torch.bool, device=device
                            ),
                        )
                    )
                )
                if teacher_trace is not None or sample_traces is not None:
                    teacher_bytes = real_teacher.cpu().numpy().tobytes()
                    if teacher_trace is not None:
                        teacher_trace.update(teacher_bytes)
                    if sample_traces is not None:
                        sample_traces.append(
                            {
                                "sequence": hashlib.sha256(sequence_key).hexdigest(),
                                "data": hashlib.sha256(
                                    sequence_key + anchor_bytes
                                ).hexdigest(),
                                "teacher": hashlib.sha256(teacher_bytes).hexdigest(),
                                "curriculum_p": probability,
                                "anchors": real_count,
                                "response_tokens": row["sequence_response_tokens"],
                            }
                        )
                with amp():
                    if packed:
                        out = model.forward_packed(
                            batch["context_features"],
                            batch["sequence_ids"],
                            batch["anchor_offsets"],
                            row["sequence_start"],
                            correct_ids=batch["labels"],
                            teacher_blocks=teacher,
                        )
                    else:
                        out = model(
                            batch["context_features"],
                            batch["labels"][:, 0],
                            batch["context_positions"],
                            batch["anchor_positions"],
                            batch["context_valid"],
                            correct_ids=batch["labels"],
                            teacher_blocks=teacher,
                        )
                    loss, metrics = training_loss(
                        model,
                        out,
                        batch["labels"],
                        batch["target_ids"],
                        batch["target_probs"],
                        schedule_step,
                        total,
                        batch["valid"],
                        alpha=run.get("loss_alpha", 0.1),
                        gamma=run.get("loss_gamma", 7.0),
                        embedding_loss_scale=run.get("embedding_loss_scale", 1.0),
                        pos1_weight=run.get("pos1_weight", 1.0),
                        chunk_size=run.get("head_chunk_size", 4096),
                        lambda_ce_reduction=run.get(
                            "lambda_ce_reduction", "position_sum"
                        ),
                        refinement_start_position=run.get(
                            "refinement_start_position", 1
                        ),
                        mask_unsupported_ce=run.get("mask_unsupported_ce", False),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                (loss / accum).backward()
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v.item() / accum
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            lr = run.get("lr", 6e-4) * lr_multiplier(schedule_step, total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.monotonic()
            elapsed = previous_elapsed + now - started
            record = {
                "step": step,
                **totals,
                "lr": lr,
                "gradient_norm": norm.item(),
                "curriculum_p": probability,
                "response_tokens_seen": stream.response_tokens_seen,
                "schedule_progress": schedule_step / total,
                "seconds_per_step": now - begin,
                "elapsed_seconds": elapsed,
                "data_epoch": stream.epoch,
                "anchors": actual_anchors,
                "anchor_slots": anchor_slots,
                "peak_memory_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
                "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved()
                if device.type == "cuda"
                else 0,
            }
            if data_trace is not None:
                record.update(
                    data_fingerprint=data_trace.hexdigest(),
                    teacher_fingerprint=teacher_trace.hexdigest(),
                    sequence_fingerprint=sequence_trace.hexdigest(),
                )
            if sample_traces is not None:
                record["sample_traces"] = sample_traces
            log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if (
                run.get("eval_every", 250) > 0
                and (step - run.get("eval_step_offset", 0)) % run.get("eval_every", 250)
                == 0
            ):
                evaluate_now()
            if step % run.get(
                "checkpoint_every", 100
            ) == 0 or now - last_save >= run.get("checkpoint_seconds", 600):
                path = save_checkpoint(
                    root, model, optimizer, step, stream, run, elapsed
                )
                last_save = time.monotonic()
                print(json.dumps({"checkpoint": str(path), "step": step}), flush=True)
            if (time.monotonic() - started) / 60 >= args.max_minutes:
                break
        if run.get("eval_at_end", False) and step != last_evaluated:
            evaluate_now()
        path = save_checkpoint(
            root,
            model,
            optimizer,
            step,
            stream,
            run,
            previous_elapsed + time.monotonic() - started,
        )
        print(
            json.dumps(
                {
                    "finished": step >= total,
                    "step": step,
                    "checkpoint": str(path),
                    "resumable": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
