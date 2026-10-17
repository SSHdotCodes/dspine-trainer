"""Versioned rolling feature shards and deterministic response-only anchors.

Token index j's stored p[j] predicts ids[j], from target logits after token j-1.
Context features end strictly before the known anchor. Never shift either array
again in the trainer. FP8 bytes are stored as uint8 for portable torch.load.
"""

import hashlib
import json
import os
import random
from pathlib import Path

import torch

FEATURE_VERSION = "dspine-hc-mean-v2-joint-ties"


def pad_support(tensor, width):
    return torch.nn.functional.pad(tensor, (0, width - tensor.shape[-1]))


def target_distribution(row):
    """Primary top-20 plus joint-sampler cutoff ties, with zero-probability padding."""
    ids, probs = row["target_ids"], row["target_probs"]
    if "target_tie_ids" in row:
        ids = torch.cat((ids, row["target_tie_ids"]), -1)
        probs = torch.cat((probs, row["target_tie_probs"]), -1)
    return ids, probs


def load_exclusion_ids(path=None):
    """Apply the project exclusion list in generation and cached-feature loaders.

    Explicit paths must exist. Source checkouts find prompts/EXCLUDE_IDS.json in
    their repository or workspace root; reusable installations may omit it.
    """
    path = path or os.environ.get("DSPINE_EXCLUDE_IDS")
    if path is None:
        source = Path(__file__).resolve()
        path = next(
            (
                parent / "prompts/EXCLUDE_IDS.json"
                for parent in (source.parents[1], source.parents[2])
                if (parent / "prompts/EXCLUDE_IDS.json").exists()
            ),
            None,
        )
    if path is None:
        return frozenset()
    payload = json.loads(Path(path).read_text())
    ids = payload["ids"] if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("exclusion IDs must be a JSON list of strings")
    return frozenset(ids)


def records(paths, exclude_ids=None):
    excluded = load_exclusion_ids() if exclude_ids is None else exclude_ids
    for path in sorted(map(Path, paths)):
        if path.suffix == ".zst":
            import io

            import zstandard as zstd

            with (
                path.open("rb") as raw,
                zstd.ZstdDecompressor().stream_reader(raw) as src,
            ):
                for line in io.TextIOWrapper(src):
                    if line.strip():
                        row = json.loads(line)
                        if row["id"] not in excluded:
                            yield row
        else:
            with path.open() as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row["id"] not in excluded:
                            yield row


def generation_paths(root):
    """Match the campaign merge's production parts, excluding smoke/tune shards.

    Standalone JSONL directories remain supported for frozen manifests and
    held-out evaluation. Campaign roots contain vega/vast-*/modal-* platforms.
    """
    root = Path(root)
    platforms = [
        p
        for p in root.iterdir()
        if p.is_dir() and (p.name == "vega" or p.name.startswith(("vast-", "modal-")))
    ]
    if platforms:
        return sorted(
            part
            for platform in platforms
            if "smoke" not in platform.name and "test" not in platform.name
            for part in platform.glob("shard-*/part-*.jsonl.zst")
            if part.parent.name != "shard-99999"
        )
    return sorted(root.rglob("*.jsonl.zst")) + sorted(root.rglob("*.jsonl"))


def split_for_id(record_id, validation_fraction=0.01):
    value = int(hashlib.sha256(record_id.encode()).hexdigest()[:16], 16) / 2**64
    return "validation" if value < validation_fraction else "train"


def quantize_features(features, chunk_size=1024):
    """Per-token/layer FP8 scaling with bounded FP32 scratch on long prefills."""
    if features.ndim != 3:
        raise ValueError("expected [tokens,layers,hidden]")
    if chunk_size < 1:
        raise ValueError("quantization chunk size must be positive")
    limit = torch.finfo(torch.float8_e4m3fn).max
    encoded, scales = [], []
    for part in features.split(chunk_size):
        if not torch.isfinite(part).all():
            raise ValueError("nonfinite target features")
        # BF16 scales cover tiny ranges without FP16 underflow.
        scale = (
            (part.float().abs().amax(-1, keepdim=True) / limit)
            .clamp_min(1e-12)
            .bfloat16()
        )
        values = (
            (part.float() / scale.float()).clamp(-limit, limit).to(torch.float8_e4m3fn)
        )
        encoded.append(values.view(torch.uint8))
        scales.append(scale.squeeze(-1))
    return torch.cat(encoded).cpu(), torch.cat(scales).cpu()


def dequantize_features(payload, device="cpu", dtype=torch.bfloat16):
    if payload["encoding"] == "bf16":
        return payload["features"].to(device=device, dtype=dtype).flatten(-2)
    if payload["encoding"] != "fp8_e4m3fn":
        raise ValueError("unknown feature encoding")
    encoded = payload["features"].to(device).view(torch.float8_e4m3fn)
    return (
        (encoded.float() * payload["scales"].to(device).float()[..., None])
        .to(dtype)
        .flatten(-2)
    )


class FeatureStore:
    """Atomic sequence shards; refuses to exceed budget. No automatic deletion.

    The rolling controller can delete only *consumed* shards after an epoch or
    recapture them for subsequent epochs. manifest.jsonl is an append-only index;
    shards are the authority, and readers ignore missing consumed entries.
    """

    def __init__(
        self,
        root,
        max_bytes=450_000_000_000,
        min_free_bytes=50_000_000_000,
        budget_root=None,
        max_global_bytes=449_000_000_000,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self._bytes = sum(p.stat().st_size for p in self.root.glob("*.pt"))
        self._partials = {p: p.stat().st_size for p in self.root.glob("*.partial")}
        self.max_global_bytes = (
            max_global_bytes if budget_root is not None else max_bytes
        )
        self._outside_bytes = 0
        if budget_root is not None:
            budget_root = Path(budget_root).resolve()
            if not self.root.resolve().is_relative_to(budget_root):
                raise ValueError("feature shard root is outside the global budget root")
            # One writer owns capture during a GPU hold. Snapshot other caches
            # once, avoiding an O(shards) filesystem scan for every sequence.
            self._outside_bytes = sum(
                p.stat().st_size
                for p in budget_root.rglob("*")
                if p.is_file()
                and p.suffix in (".pt", ".partial")
                and not p.resolve().is_relative_to(self.root.resolve())
            )

    def write(
        self,
        record,
        features,
        target_ids,
        target_probs,
        start=0,
        encoding="fp8_e4m3fn",
        layer_ids=(1, 12, 23, 34, 45),
    ):
        if features.ndim != 3 or features.shape[1] != len(layer_ids):
            raise ValueError("feature layers do not match layer_ids")
        if encoding == "fp8_e4m3fn":
            encoded, scales = quantize_features(features)
        elif encoding == "bf16":
            encoded = features.cpu().bfloat16().clone()
            scales = None
        else:
            raise ValueError("invalid feature encoding")
        return self.write_encoded(
            record,
            encoded,
            scales,
            target_ids,
            target_probs,
            start,
            encoding,
            layer_ids,
        )

    def write_encoded(
        self,
        record,
        encoded,
        scales,
        target_ids,
        target_probs,
        start=0,
        encoding="fp8_e4m3fn",
        layer_ids=(1, 12, 23, 34, 45),
    ):
        """Commit already packed per-token features, including chunked captures."""
        n = len(record["prompt_ids"]) + len(record["output_ids"]) - start
        if (
            encoded.ndim != 3
            or encoded.shape[:2] != (n, len(layer_ids))
            or target_ids.ndim != 2
            or target_ids.shape[0] != n
            or target_ids.shape[1] < 20
            or target_probs.shape != target_ids.shape
        ):
            raise ValueError("feature/distribution length mismatch")
        if encoding == "fp8_e4m3fn":
            if (
                encoded.dtype != torch.uint8
                or scales is None
                or scales.shape != (n, len(layer_ids))
            ):
                raise ValueError("invalid packed FP8 feature/scales")
        elif (
            encoding != "bf16" or encoded.dtype != torch.bfloat16 or scales is not None
        ):
            raise ValueError("invalid feature encoding")
        ids = torch.tensor(
            record["prompt_ids"] + record["output_ids"], dtype=torch.long
        )[start:].clone()
        response_start = len(record["prompt_ids"]) - start
        validp = target_probs[response_start:]
        if not torch.allclose(
            validp.float().sum(-1),
            torch.ones(validp.shape[0], device=validp.device),
            atol=1e-4,
        ):
            raise ValueError("processed target probabilities must sum to one")
        payload = {
            "version": FEATURE_VERSION,
            "id": record["id"],
            "domain": record["domain"],
            "split": record.get("split", split_for_id(record["id"])),
            "model_rev": record.get("model_rev", "unknown"),
            "finish_reason": record.get("finish_reason", "unknown"),
            "sampling": record["sampling"],
            "start": start,
            "response_start": response_start,
            "input_ids": ids,
            "layer_ids": list(layer_ids),
            "features": encoded.cpu(),
            "scales": scales.cpu() if scales is not None else None,
            "encoding": encoding,
            "target_ids": target_ids[:, :20].cpu().int().clone(),
            "target_probs": target_probs[:, :20].cpu().float().clone(),
        }
        if target_ids.shape[-1] > 20:
            payload["target_tie_ids"] = target_ids[:, 20:].cpu().int().clone()
            payload["target_tie_probs"] = target_probs[:, 20:].cpu().float().clone()
        key = hashlib.sha256(record["id"].encode()).hexdigest()
        dest = self.root / f"{key}.pt"
        if dest.exists():
            return dest
        size = (
            sum(
                v.numel() * v.element_size()
                for v in payload.values()
                if isinstance(v, torch.Tensor)
            )
            + 16384
        )
        temp = dest.with_suffix(".partial")
        replaced_partial = self._partials.get(temp, 0)
        other_partials = sum(self._partials.values()) - replaced_partial
        free = (
            os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
            + replaced_partial
        )
        if (
            self._bytes + other_partials + size > self.max_bytes
            or self._outside_bytes + self._bytes + other_partials + size
            > self.max_global_bytes
            or free - size < self.min_free_bytes
        ):
            raise BufferError(
                "rolling feature budget reached; consume or rotate completed shards first"
            )
        try:
            torch.save(payload, temp)
            os.replace(temp, dest)
            self._bytes += dest.stat().st_size
        finally:
            if temp.exists():
                self._partials[temp] = temp.stat().st_size
            else:
                self._partials.pop(temp, None)
        with (self.root / "manifest.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "path": dest.name,
                        "id": record["id"],
                        "tokens": n,
                        "split": payload["split"],
                        "bytes": dest.stat().st_size,
                    }
                )
                + "\n"
            )
        return dest


class AnchorStream:
    """Resumable single-process cursor, bounded to one sequence on host memory.

    Sampling is with replacement inside assistant output only. A saved cursor
    includes the shuffled shard order and RNG, so interrupted runs are exact.
    """

    def __init__(
        self,
        root,
        anchors_per_sequence=32,
        block_size=16,
        context_window=4096,
        seed=42,
        split="train",
        expected_layer_ids=None,
        expected_revision=None,
        anchor_policy="random",
        exclude_ids=None,
        anchor_sampling="with_replacement",
        read_root=None,
    ):
        self.root = Path(root)
        # Optional immutable, owned hard-link view for concurrent recipe forks.
        # Cursor paths remain logical source paths, so checkpoint promotion does
        # not embed an experiment-specific data location.
        self.read_root = Path(read_root) if read_root is not None else self.root
        self._read_root_override = read_root is not None
        self.anchors = anchors_per_sequence
        self.block = block_size
        self.window = context_window
        self.rng = random.Random(seed)
        self.split = split
        self.expected_layer_ids = (
            tuple(expected_layer_ids) if expected_layer_ids is not None else None
        )
        self.expected_revision = expected_revision
        self.anchor_policy = anchor_policy
        if anchor_sampling not in ("with_replacement", "without_replacement"):
            raise ValueError("unknown anchor sampling mode")
        self.anchor_sampling = anchor_sampling
        self.excluded = load_exclusion_ids(exclude_ids)
        if anchor_policy not in ("random", "first"):
            raise ValueError("unknown anchor policy")
        self.epoch = 0
        self.index = 0
        self.anchor_index = 0
        self.paths = self.shard_paths()
        self.rng.shuffle(self.paths)
        self.current = None
        self.current_path = None
        self.sampled = None

    def shard_paths(self):
        return sorted(str(self.root / p.name) for p in self.read_root.glob("*.pt"))

    def read_path(self, logical_path):
        return (
            self.read_root / Path(logical_path).name
            if self._read_root_override
            else Path(logical_path)
        )

    def validate_contract(self, row):
        if row["version"] != FEATURE_VERSION:
            raise ValueError("wrong feature contract")
        if (
            self.expected_layer_ids is not None
            and tuple(row["layer_ids"]) != self.expected_layer_ids
        ):
            raise ValueError("captured target layers differ from model config")
        if (
            self.expected_revision is not None
            and row.get("model_rev") != self.expected_revision
        ):
            raise ValueError("captured target revision differs from model config")

    def state_dict(self):
        return {
            "paths": self.paths,
            "epoch": self.epoch,
            "index": self.index,
            "anchor_index": self.anchor_index,
            "sampled": self.sampled,
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        self.paths = state["paths"]
        self.epoch = state["epoch"]
        self.index = state["index"]
        self.anchor_index = state["anchor_index"]
        self.sampled = state["sampled"]
        self.rng.setstate(state["rng"])
        self.current = None
        self.current_path = None

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(max(1, len(self.paths)) * 2):
            if not self.paths:
                raise RuntimeError("no feature shards available")
            if self.index >= len(self.paths):
                self.epoch += 1
                self.index = 0
                self.anchor_index = 0
                self.sampled = None
                self.paths = self.shard_paths()
                self.rng.shuffle(self.paths)
            path = self.paths[self.index]
            if self.current_path != path:
                self.current = torch.load(
                    self.read_path(path), map_location="cpu", weights_only=False
                )
                self.current_path = path
            row = self.current
            length = len(row["input_ids"])
            if row["id"] not in self.excluded:
                self.validate_contract(row)
            if (
                row["id"] in self.excluded
                or row["split"] != self.split
                or length - row["response_start"] < 2
            ):
                self.index += 1
                self.sampled = None
                self.anchor_index = 0
                continue
            if self.sampled is None:
                self.sampled = (
                    [row["response_start"]]
                    if self.anchor_policy == "first"
                    else [
                        self.rng.randrange(row["response_start"], length - 1)
                        for _ in range(self.anchors)
                    ]
                )
            a = self.sampled[self.anchor_index]
            self.anchor_index += 1
            if self.anchor_index >= len(self.sampled):
                self.index += 1
                self.sampled = None
                self.anchor_index = 0
            end = min(length, a + self.block)
            start = max(0, a - self.window)
            # Dequantize the requested window only, rather than a long sequence.
            view = {**row, "features": row["features"][start:a]}
            if row["scales"] is not None:
                view["scales"] = row["scales"][start:a]
            features = dequantize_features(view)
            target_ids, target_probs = target_distribution(row)
            return {
                "context_features": features,
                "context_positions": torch.arange(start, a) + row["start"],
                "anchor_position": a + row["start"],
                "available_tokens": length - a,
                "finish_reason": row.get("finish_reason", "unknown"),
                "labels": row["input_ids"][a:end],
                "target_ids": target_ids[a:end].long(),
                "target_probs": target_probs[a:end],
                "domain": row["domain"],
                "id": row["id"],
                "sampling": row["sampling"],
                "target_distribution_source": row.get(
                    "target_distribution_source", "prefill_processed"
                ),
            }
        raise RuntimeError(f"no eligible {self.split} shards")


def collate_anchors(rows, block_size=16):
    n = len(rows)
    width = max(r["context_features"].shape[0] for r in rows)
    dim = rows[0]["context_features"].shape[-1]
    support = max(r["target_ids"].shape[-1] for r in rows)
    result = {
        "context_features": torch.zeros(
            n, width, dim, dtype=rows[0]["context_features"].dtype
        ),
        "context_positions": torch.zeros(n, width, dtype=torch.long),
        "context_valid": torch.zeros(n, width, dtype=torch.bool),
        "anchor_positions": torch.tensor([r["anchor_position"] for r in rows]),
        "labels": torch.zeros(n, block_size, dtype=torch.long),
        "target_ids": torch.zeros(n, block_size, support, dtype=torch.long),
        "target_probs": torch.zeros(n, block_size, support),
        "valid": torch.zeros(n, block_size, dtype=torch.bool),
    }
    for i, row in enumerate(rows):
        w = len(row["context_features"])
        b = len(row["labels"])
        if w:
            result["context_features"][i, -w:] = row["context_features"]
            result["context_positions"][i, -w:] = row["context_positions"]
            result["context_valid"][i, -w:] = True
        result["labels"][i, :b] = row["labels"]
        for key in ("target_ids", "target_probs"):
            result[key][i, :b, : row[key].shape[-1]] = row[key]
        result["valid"][i, :b] = True
    return result


class PackedSequenceStream(AnchorStream):
    """One sequence and fixed-capacity independent anchor blocks per microbatch.

    ``anchor_valid`` identifies the real prefix. Unique sampling pads shortages
    with safe repeated offsets, and ``valid`` masks every position in those
    dummy blocks. Consumers must normalize by real anchors, not slot capacity.
    """

    def __next__(self):
        for _ in range(max(1, len(self.paths)) * 2):
            if not self.paths:
                raise RuntimeError("no feature shards available")
            if self.index >= len(self.paths):
                self.epoch += 1
                self.index = 0
                self.paths = self.shard_paths()
                self.rng.shuffle(self.paths)
            if getattr(self, "cold_data", False) and hasattr(os, "posix_fadvise"):
                # Calibration only: evict this owned shard's clean OS cache
                # pages to approximate windows larger than host RAM.
                descriptor = os.open(
                    self.read_path(self.paths[self.index]), os.O_RDONLY
                )
                try:
                    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(descriptor)
            row = torch.load(
                self.read_path(self.paths[self.index]),
                map_location="cpu",
                weights_only=False,
            )
            self.index += 1
            if row["id"] not in self.excluded:
                self.validate_contract(row)
            length = len(row["input_ids"])
            if (
                row["id"] in self.excluded
                or row["split"] != self.split
                or length - row["response_start"] < 2
            ):
                continue
            # Consume the historical draw budget in both modes so the shuffled
            # sequence order and its resumable RNG remain paired across forks,
            # even after an epoch boundary. The cloned state supplies uniform
            # sampling without replacement without changing A/B's RNG stream.
            unique_rng = None
            if self.anchor_sampling == "without_replacement":
                unique_rng = random.Random()
                unique_rng.setstate(self.rng.getstate())
            draws = [
                self.rng.randrange(row["response_start"], length - 1)
                for _ in range(self.anchors)
            ]
            if unique_rng is not None:
                positions = range(row["response_start"], length - 1)
                draws = unique_rng.sample(positions, min(self.anchors, len(positions)))
            anchors = torch.tensor(draws, dtype=torch.long).sort().values
            real_count = len(anchors)
            # Keep N*block_size constant for the compiled FlexAttention backward.
            # Repeat a safe, nearby anchor rather than creating fully masked
            # attention rows. Blocks are independent; only their LOSSES are
            # masked. Invalid attention rows could introduce NaNs even at zero
            # loss weight. Padding follows the sorted real prefix.
            if real_count < self.anchors:
                anchors = torch.cat(
                    (anchors, anchors[-1:].expand(self.anchors - real_count))
                )
            anchor_valid = torch.arange(self.anchors) < real_count
            # Nearby anchors share FlexAttention query tiles, keeping each tile's
            # union of context windows small. Losses average over actual anchors.
            offsets = anchors[:, None] + torch.arange(self.block)
            valid = (offsets < length) & anchor_valid[:, None]
            offsets = offsets.clamp(max=length - 1)
            target_ids, target_probs = target_distribution(row)
            return {
                "context_payload": {
                    k: row[k] for k in ("features", "scales", "encoding")
                },
                "sequence_ids": row["input_ids"],
                "anchor_offsets": anchors,
                "anchor_valid": anchor_valid,
                "sequence_start": row["start"],
                "sequence_response_tokens": length - row["response_start"],
                "labels": row["input_ids"][offsets],
                "target_ids": target_ids[offsets].long(),
                "target_probs": target_probs[offsets],
                "valid": valid,
                "domain": row["domain"],
                "id": row["id"],
            }
        raise RuntimeError(f"no eligible {self.split} shards")
