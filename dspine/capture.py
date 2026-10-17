"""Prefill-only target feature capture helpers, called by the SGLang overlay.

All 48 target layers still run on the *entire* sequence, retaining its exact
history including PLE ngrams and recurrent states. Only stored features are
cropped to [max(0,response_start-W),end). Never prefill a cropped long prompt.
"""

import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F

from .capture_sequence import CaptureSequence
from .data import FeatureStore, load_exclusion_ids, pad_support


@lru_cache(maxsize=1)
def capture_layer_ids():
    return tuple(
        map(int, os.environ.get("DSPINE_CAPTURE_LAYER_IDS", "1,12,23,34,45").split(","))
    )


@torch.no_grad()
def processed_top20(hidden, head, temperature=1.0, top_p=0.95, chunk_size=256):
    """SGLang joint top-k/top-p support, then renormalize, retaining cutoff ties.

    Nucleus mass is measured in the full softmax, before truncating top-k. Using
    a top-k-renormalized mass here silently implements a different distribution.
    FlashInfer accepts a token when fewer than k tokens have strictly greater
    probability and their total mass is <p. Thus tied cutoff tokens can make
    support larger than 20. Output width includes those ties and zero padding.

    p at hidden[j] predicts token j+1. Return unshifted arrays here; capture_batch
    performs the one required shift when writing token-aligned feature shards.
    """
    ids = []
    probabilities = []
    for part in hidden.split(chunk_size):
        logits = (
            head(part) if callable(head) else F.linear(part, head)
        ).float() / temperature
        dense = logits.softmax(-1)
        cutoff = dense.topk(20, -1).values[:, -1:]
        width = max(20, int(((dense >= cutoff) & (dense > 0)).sum(-1).max()))
        if width > 4096:
            raise RuntimeError(
                "joint sampler tie support exceeds the 4096-token sparse capture limit"
            )
        p, index = dense.topk(width, -1, sorted=True)
        first = torch.cat(
            (torch.ones_like(p[:, :1], dtype=torch.bool), p[:, 1:] != p[:, :-1]), -1
        )
        positions = torch.arange(width, device=p.device).expand_as(p)
        group_start = torch.where(first, positions, 0).cummax(-1).values
        strictly_greater_mass = (p.cumsum(-1) - p).gather(-1, group_start)
        keep = (p >= cutoff) & (strictly_greater_mass < top_p)
        p = torch.where(keep, p, 0.0)
        p /= p.sum(-1, keepdim=True)
        ids.append(index)
        probabilities.append(p)
    width = max(part.shape[-1] for part in ids)
    return torch.cat([pad_support(part, width) for part in ids]), torch.cat(
        [pad_support(part, width) for part in probabilities]
    )


def capture_layer(model, layer_id, hidden, forward_batch):
    if (
        not os.environ.get("DSPINE_CAPTURE_ROOT")
        or not forward_batch.forward_mode.is_extend()
    ):
        return
    if layer_id == model.start_layer:
        model._dspine_means = []
    if layer_id in capture_layer_ids():
        if hidden.shape[-1] != model.hc_count * model.hidden_size:
            raise RuntimeError(
                "post-layer capture requires a canonical combined HC residual; disable deferred combine at selected layers"
            )
        mean = (
            hidden.reshape(-1, model.hc_count, model.hidden_size)
            .float()
            .mean(1)
            .to(hidden.dtype)
        )
        model._dspine_means.append(mean)


@lru_cache(maxsize=8)
def _load_metadata(path, modified):
    return json.loads(Path(path).read_text())


def _matching_record(root, tokens, store):
    matches = []
    excluded = load_exclusion_ids()
    for metadata in (Path(root) / "inbox").glob("*.json"):
        record = _load_metadata(str(metadata), metadata.stat().st_mtime_ns)
        if record["id"] in excluded:
            continue
        key = hashlib.sha256(record["id"].encode()).hexdigest()
        if (store.root / f"{key}.pt").exists():
            continue
        expected = record["prompt_ids"] + record["output_ids"]
        if expected[: len(tokens)] == tokens:
            matches.append(record)
    if len(matches) > 1:
        raise RuntimeError(
            "ambiguous capture prefixes; rerun extraction with --concurrency 1"
        )
    return matches[0] if matches else None


@torch.no_grad()
def capture_batch(model, input_ids, final_hidden, forward_batch, head, head_module=None):
    """TP=PP=1, no radix reuse; preserve state and p alignment across chunks."""
    root = os.environ.get("DSPINE_CAPTURE_ROOT")
    if not root or not forward_batch.forward_mode.is_extend():
        return
    # Startup profiling has no request/cache metadata. Never treat it as a
    # registered capture, including after an interrupted client's inbox survives.
    required = ("extend_seq_lens", "extend_prefix_lens", "req_pool_indices")
    if any(getattr(forward_batch, key, None) is None for key in required):
        model._dspine_means = []
        model._dspine_final_hidden = None
        return
    if len(model._dspine_means) != len(capture_layer_ids()):
        raise RuntimeError("selected post-layer HC means were not captured")
    lengths = forward_batch.extend_seq_lens.cpu().tolist()
    prefixes = forward_batch.extend_prefix_lens.cpu().tolist()
    slots = forward_batch.req_pool_indices.cpu().tolist()
    if sum(lengths) != input_ids.numel() or len(slots) != len(lengths):
        raise RuntimeError("capture batch segmentation mismatch")
    global _feature_store, _capture_sequences
    if (
        globals().get("_feature_store") is None
        or _feature_store.root != Path(root) / "shards"
    ):
        _feature_store = FeatureStore(
            Path(root) / "shards",
            max_bytes=int(os.environ.get("DSPINE_FEATURE_BUDGET", "450000000000")),
            budget_root=os.environ.get("DSPINE_FEATURE_BUDGET_ROOT"),
            max_global_bytes=int(
                os.environ.get("DSPINE_GLOBAL_FEATURE_BUDGET", "449000000000")
            ),
        )
        _capture_sequences = {}
    store = _feature_store
    offset = 0
    active = []
    tokens_cpu = input_ids.detach().cpu().long()
    for slot, prefix, length in zip(slots, prefixes, lengths):
        end = offset + length
        if prefix == 0:
            _capture_sequences.pop(slot, None)
            record = _matching_record(root, tokens_cpu[offset:end].tolist(), store)
            if record is not None:
                _capture_sequences[slot] = CaptureSequence(
                    record,
                    store,
                    capture_layer_ids(),
                    int(os.environ.get("DSPINE_CONTEXT_WINDOW", "4096")),
                    encoding=os.environ.get("DSPINE_CAPTURE_ENCODING", "fp8_e4m3fn"),
                )
        state = _capture_sequences.get(slot)
        if state is not None:
            active.append((slot, prefix, offset, end, state))
        offset = end
    if not active:
        model._dspine_means = []
        model._dspine_final_hidden = None
        return
    if head_module is not None and any(state.record.get("capture_probe") for *_, state in active):
        head_path = Path(root) / "probe-head.pt"
        if not head_path.exists():
            packed = getattr(head_module, "_qf_pk", None)
            vocab = packed["shape"][0] if packed is not None else head_module.weight.shape[0]
            row_ids = torch.linspace(0, vocab - 1, min(64, vocab), device=final_hidden.device).long()
            if packed is not None:
                from sglang.srt.qwenfast import pk_gather_rows

                weight = pk_gather_rows(packed, row_ids)
            else:
                weight = head_module.weight[row_ids]
            torch.save({"ids": row_ids.cpu(), "weight": weight.cpu()}, head_path)
    features = torch.stack(model._dspine_means, 1)
    model._dspine_means = []
    # Only response-token distributions are required. Skip the expensive head
    # projection of long prompt prefixes, including those whose features are cropped.
    written = []
    completed_tokens = 0
    for slot, prefix, start, end, state in active:
        first = start + max(0, len(state.record["prompt_ids"]) - 1 - prefix)
        last = end - int(prefix + (end - start) == len(state.tokens))
        selected_ids = selected_probs = None
        if first < last:
            def selected_head(hidden):
                logits = head(hidden) if callable(head) else F.linear(hidden, head)
                if state.record.get("capture_probe"):
                    if not hasattr(state, "probe_logits"):
                        state.probe_logits = []
                    state.probe_logits.append(logits.detach().cpu().float().clone())
                return logits

            selected_ids, selected_probs = processed_top20(
                final_hidden[first:last],
                selected_head,
                chunk_size=int(os.environ.get("DSPINE_CAPTURE_HEAD_CHUNK", "256")),
            )
        width = selected_ids.shape[-1] if selected_ids is not None else 20
        ids = torch.zeros(
            end - start, width, dtype=torch.long, device=final_hidden.device
        )
        p = torch.zeros(
            end - start, width, dtype=torch.float32, device=final_hidden.device
        )
        if selected_ids is not None:
            ids[first - start : last - start] = selected_ids
            p[first - start : last - start] = selected_probs
        path = state.append(
            prefix,
            tokens_cpu[start:end],
            features[start:end],
            ids,
            p,
        )
        if path is not None:
            if state.record.get("capture_probe"):
                probe_dir = Path(root) / "probe-logits"
                probe_dir.mkdir(exist_ok=True)
                torch.save(torch.cat(state.probe_logits), probe_dir / path.name)
            written.append(str(path))
            completed_tokens += len(state.tokens)
            del _capture_sequences[slot]
    if written:
        with (Path(root) / "capture-events.jsonl").open("a", buffering=1) as log:
            log.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "tokens": completed_tokens,
                        "chunk_tokens": input_ids.numel(),
                        "sequences": len(written),
                        "paths": written,
                    }
                )
                + "\n"
            )
    model._dspine_final_hidden = None
