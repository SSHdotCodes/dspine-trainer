"""Check prefill chunk boundaries without loading a target model or using GPU."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from dspine.capture import processed_top20
from dspine.capture_sequence import CaptureSequence
from dspine.data import FeatureStore, target_distribution


def row(key="example", tokens=None, prompt_length=7):
    tokens = list(range(13)) if tokens is None else tokens
    return {
        "id": key,
        "domain": "math",
        "prompt_ids": tokens[:prompt_length],
        "output_ids": tokens[prompt_length:],
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    }


def test_cutoff_tie_overflow_survives_varying_chunk_support(tmp_path):
    record = row(prompt_length=5)
    state = CaptureSequence(
        record, FeatureStore(tmp_path, min_free_bytes=0), (1, 3), window=3
    )
    prefix = 0
    expected = []
    for length, width in ((4, 25), (5, 20), (4, 23)):
        ids = torch.arange(width).expand(length, -1)
        probs = torch.ones(length, width) / width
        expected.extend([width] * length)
        path = state.append(
            prefix,
            torch.arange(prefix, prefix + length),
            torch.randn(length, 2, 8),
            ids,
            probs,
        )
        prefix += length
    saved = torch.load(path, weights_only=True)
    ids, probs = target_distribution(saved)
    assert ids.shape == probs.shape == (11, 25)
    for absolute in range(5, 13):
        count = expected[absolute - 1]
        p = probs[absolute - saved["start"]]
        torch.testing.assert_close(p[:count], torch.full((count,), 1 / count))
        assert not p[count:].count_nonzero()


@pytest.mark.parametrize("chunks", [(13,), (3, 4, 2, 4), (1,) * 13, (4, 9)])
@pytest.mark.parametrize("encoding", ["fp8_e4m3fn", "bf16"])
def test_chunked_capture_equals_whole_sequence_with_prompt_crop(tmp_path, chunks, encoding):
    torch.manual_seed(17)
    record = row()
    features = torch.randn(13, 2, 8)
    hidden = torch.randn(13, 12)
    head = torch.randn(29, 12)
    next_ids, next_probs = processed_top20(hidden, head)
    shifted_ids = torch.zeros_like(next_ids)
    shifted_p = torch.zeros_like(next_probs)
    shifted_ids[1:] = next_ids[:-1]
    shifted_p[1:] = next_probs[:-1]
    expected_store = FeatureStore(tmp_path / "whole", min_free_bytes=0)
    expected = expected_store.write(
        record, features[4:], shifted_ids[4:], shifted_p[4:], start=4, layer_ids=(1, 3), encoding=encoding
    )
    store = FeatureStore(tmp_path / "chunked", min_free_bytes=0)
    capture = CaptureSequence(record, store, (1, 3), window=3, encoding=encoding)
    prefix = 0
    actual = None
    for length in chunks:
        actual = capture.append(
            prefix,
            torch.arange(prefix, prefix + length),
            features[prefix : prefix + length],
            next_ids[prefix : prefix + length],
            next_probs[prefix : prefix + length],
        )
        prefix += length
        assert (actual is not None) == (prefix == 13)
    actual = torch.load(actual, weights_only=True)
    expected = torch.load(expected, weights_only=True)
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        else:
            assert actual[key] == expected[key]


def test_capture_rejects_out_of_order_or_wrong_tokens(tmp_path):
    record = row()
    store = FeatureStore(tmp_path, min_free_bytes=0)
    capture = CaptureSequence(record, store, (1, 3))
    features = torch.randn(2, 2, 8)
    ids = torch.arange(20).expand(2, -1)
    p = torch.ones(2, 20) / 20
    with pytest.raises(ValueError, match="ordered"):
        capture.append(2, torch.tensor([2, 3]), features, ids, p)
    with pytest.raises(ValueError, match="registered request"):
        capture.append(0, torch.tensor([9, 9]), features, ids, p)
    assert not list(tmp_path.glob("*.pt"))


def test_sglang_chunk_routing_keeps_requests_and_predecessor_probs_separate(
    tmp_path, monkeypatch
):
    import dspine.capture as capture

    monkeypatch.setenv("DSPINE_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setattr(capture, "capture_layer_ids", lambda: (1, 3))
    monkeypatch.setattr(
        capture,
        "FeatureStore",
        lambda root, max_bytes, **kwargs: FeatureStore(
            root, max_bytes, min_free_bytes=0, **kwargs
        ),
    )
    monkeypatch.setattr(capture, "_feature_store", None, raising=False)
    monkeypatch.setattr(capture, "_capture_sequences", {}, raising=False)
    (tmp_path / "inbox").mkdir()
    rows = [row("a", [1, 2, 3, 4, 5, 6], 2), row("b", [1, 2, 5, 6, 7, 8, 9], 2)]
    for item in rows:
        (tmp_path / "inbox" / f"{item['id']}.json").write_text(json.dumps(item))
    torch.manual_seed(22)
    all_features = [
        torch.randn(len(r["prompt_ids"]) + len(r["output_ids"]), 2, 8) for r in rows
    ]
    all_hidden = [torch.randn(len(f), 12) for f in all_features]
    head = torch.randn(29, 12)
    model = SimpleNamespace(_dspine_final_hidden=None)
    projected = []

    def projector(h):
        projected.append(len(h))
        return torch.nn.functional.linear(h, head)

    for prefix in (0, 3):
        end = [3, 3] if prefix == 0 else [6, 7]
        features = torch.cat([f[prefix:e] for f, e in zip(all_features, end)])
        hidden = torch.cat([f[prefix:e] for f, e in zip(all_hidden, end)])
        tokens = torch.tensor(
            sum(
                [
                    (r["prompt_ids"] + r["output_ids"])[prefix:e]
                    for r, e in zip(rows, end)
                ],
                [],
            )
        )
        model._dspine_means = [features[:, i] for i in range(2)]
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            extend_seq_lens=torch.tensor([e - prefix for e in end]),
            extend_prefix_lens=torch.tensor([prefix, prefix]),
            req_pool_indices=torch.tensor([5, 6]),
        )
        capture.capture_batch(model, tokens, hidden, batch, projector)
    assert capture._capture_sequences == {}
    assert sum(projected) == sum(len(r["output_ids"]) for r in rows)
    for r, h in zip(rows, all_hidden):
        path = (
            tmp_path / "shards" / (hashlib.sha256(r["id"].encode()).hexdigest() + ".pt")
        )
        saved = torch.load(path, weights_only=True)
        ids, p = processed_top20(h, head)
        torch.testing.assert_close(
            saved["target_probs"][2:], p[1:-1], atol=1e-6, rtol=1e-5
        )
        assert torch.equal(saved["target_ids"][2:].long(), ids[1:-1])
        assert torch.count_nonzero(saved["target_probs"][:2]) == 0


def test_partial_shards_count_toward_budget_but_can_be_replaced(tmp_path):
    record = row()
    features = torch.randn(13, 2, 8)
    ids = torch.arange(20).expand(13, -1)
    p = torch.ones(13, 20) / 20
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "interrupted.partial").write_bytes(b"x" * 10000)
    with pytest.raises(BufferError):
        FeatureStore(blocked, max_bytes=20000, min_free_bytes=0).write(
            record, features, ids, p, layer_ids=(1, 3)
        )
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    partial = resumed / (hashlib.sha256(record["id"].encode()).hexdigest() + ".partial")
    partial.write_bytes(b"x" * 10000)
    result = FeatureStore(resumed, max_bytes=20000, min_free_bytes=0).write(
        record, features, ids, p, layer_ids=(1, 3)
    )
    assert result.exists() and not partial.exists()


def test_capture_ignores_startup_profile_without_request_metadata(
    tmp_path, monkeypatch
):
    import dspine.capture as capture

    monkeypatch.setenv("DSPINE_CAPTURE_ROOT", str(tmp_path))
    model = SimpleNamespace(
        _dspine_means=[torch.randn(2, 8)], _dspine_final_hidden=torch.randn(2, 8)
    )
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))

    def unavailable_head(hidden):
        raise AssertionError("startup profiling must not project the capture head")

    capture.capture_batch(
        model, torch.tensor([1, 2]), torch.randn(2, 8), batch, unavailable_head
    )
    assert model._dspine_means == [] and model._dspine_final_hidden is None
