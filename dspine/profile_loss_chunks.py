"""Paired end-to-end GPU calibration of the exact full-vocabulary loss.

Uses the same starting draft weights, sequence order, anchors and optimizer
initialization for every vocabulary chunk size. Discards all benchmark updates;
only a JSON receipt is written. Reserve the GPU before invoking this module.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .data import PackedSequenceStream, dequantize_features
from .hot_vocab import load_hot_ids
from .losses import training_loss
from .modeling import DSpineDraft
from .transfer_space import load_target_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchors", default="256,512")
    parser.add_argument("--chunks", default="4096,16384,32768,65536")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-order", choices=("shuffled", "longest"), default="shuffled")
    args = parser.parse_args()
    run = json.loads(Path(args.config).read_text())
    torch.set_num_threads(run.get("cpu_threads", 8))
    if run.get("cuda_memory_limit_gb"):
        torch.cuda.set_per_process_memory_fraction(
            min(1., run["cuda_memory_limit_gb"] * 1_000_000_000 /
                torch.cuda.get_device_properties(0).total_memory),0)
    emb, head = load_target_weights(run["target"], run.get("target_head_quantization"))
    model = (
        DSpineDraft.from_pretrained(
            args.checkpoint, emb, head, trainable_dtype=torch.float32
        )
        .cuda()
        .train()
    )
    if run.get("proposal_vocabulary"):
        policy = run["proposal_vocabulary"]
        model.configure_proposal_vocabulary(load_hot_ids(policy["path"], model.config.vocab_size),
            policy.get("quantization", "row_int8"), policy.get("loss", "restricted"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    # Host snapshots keep the measured GPU peak representative of training.
    initial = [p.detach().cpu().clone() for p in parameters]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    accumulation = run.get("gradient_accumulation", 4)
    for anchors in map(int, args.anchors.split(",")):
        for chunk in map(int, args.chunks.split(",")):
            with torch.no_grad():
                for parameter, value in zip(parameters, initial):
                    parameter.copy_(value)
            torch.manual_seed(run.get("seed", 42))
            stream = PackedSequenceStream(
                run["features"],
                anchors,
                model.config.block_size,
                model.config.context_window,
                run.get("seed", 42),
                expected_layer_ids=model.config.target_layer_ids,
                expected_revision=run.get("target_revision"),
                exclude_ids=run.get("exclude_ids"),
            )
            stream.cold_data = run.get("benchmark_cold_data", False)
            if args.sequence_order == "longest":
                def retained_length(path):
                    record = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                    return len(record["input_ids"])
                stream.paths.sort(key=retained_length, reverse=True)
            optimizer = torch.optim.AdamW(
                parameters, lr=run.get("lr", 6e-4), weight_decay=0.0, fused=True
            )
            timings, initial_loss = [], None
            for step in range(args.warmup + args.steps):
                if step == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for _ in range(accumulation):
                    row = next(stream)
                    batch = {
                        k: v.cuda()
                        for k, v in row.items()
                        if isinstance(v, torch.Tensor)
                    }
                    features = dequantize_features(
                        row["context_payload"], device="cuda", dtype=torch.bfloat16
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.forward_packed(
                            features,
                            batch["sequence_ids"],
                            batch["anchor_offsets"],
                            row["sequence_start"],
                            correct_ids=batch["labels"],
                            teacher_blocks=torch.zeros(
                                anchors, device="cuda", dtype=torch.bool
                            ),
                        )
                        loss, _ = training_loss(
                            model,
                            out,
                            batch["labels"],
                            batch["target_ids"],
                            batch["target_probs"],
                            100,
                            200,
                            batch["valid"],
                            chunk_size=chunk,
                        )
                    (loss / accumulation).backward()
                    total += float(loss.detach()) / accumulation
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                if initial_loss is None:
                    initial_loss = total
                if step >= args.warmup:
                    timings.append(time.perf_counter() - started)
            result = dict(
                anchors=anchors,
                head_chunk_size=chunk,
                median_seconds_per_step=statistics.median(timings),
                mean_seconds_per_step=statistics.mean(timings),
                timings=timings,
                initial_loss=initial_loss,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_memory_bytes=torch.cuda.max_memory_reserved(),
                proposal_vocab_size=len(model.lm_head.weight) if model.proposal_weight is None else len(model.proposal_weight),
                cold_shard_reads=stream.cold_data,
                sequence_order=args.sequence_order,
            )
            results.append(result)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            optimizer.zero_grad(set_to_none=True)
            del optimizer, out, loss, features, batch, row
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
