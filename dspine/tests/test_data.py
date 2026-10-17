import copy
import json

import pytest
import torch

from dspine.adapt_target import procrustes_align
from dspine.capture import processed_top20
from dspine.data import (
    AnchorStream,
    FeatureStore,
    PackedSequenceStream,
    collate_anchors,
    generation_paths,
    load_exclusion_ids,
    quantize_features,
    records,
    split_for_id,
    target_distribution,
)
from dspine.prepare_overlay import patch


def make_record(key="sample"):
    return {
        "id": key,
        "domain": "math",
        "prompt_ids": list(range(5)),
        "output_ids": list(range(5, 25)),
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    }


def test_store_cursor_alignment_and_packing(tmp_path):
    record = make_record()
    assert split_for_id(record["id"]) == "train"
    features = torch.randn(25, 2, 8)
    ids = torch.arange(20).expand(25, -1).clone()
    p = torch.rand(25, 20)
    p /= p.sum(-1, keepdim=True)
    store = FeatureStore(tmp_path, min_free_bytes=0)
    path = store.write(record, features, ids, p, layer_ids=(1, 3))
    raw = torch.load(path, weights_only=True)
    assert raw["response_start"] == 5 and raw["layer_ids"] == [1, 3]
    stream = AnchorStream(
        tmp_path, anchors_per_sequence=7, block_size=5, context_window=4
    )
    next(stream)
    state = copy.deepcopy(stream.state_dict())
    expected = [next(stream) for _ in range(5)]
    restored = AnchorStream(
        tmp_path, anchors_per_sequence=7, block_size=5, context_window=4
    )
    restored.load_state_dict(state)
    actual = [next(restored) for _ in range(5)]
    for a, b in zip(expected, actual):
        assert a["anchor_position"] == b["anchor_position"] >= 5
        assert a["context_positions"].max() < a["anchor_position"]
        torch.testing.assert_close(a["context_features"], b["context_features"])
        torch.testing.assert_close(a["labels"], b["labels"])
    batch = collate_anchors(actual, block_size=5)
    assert torch.equal(
        batch["labels"][:, 0], torch.tensor([x["anchor_position"] for x in actual])
    )
    packed = next(PackedSequenceStream(tmp_path, anchors_per_sequence=7, block_size=5))
    assert torch.all(packed["anchor_offsets"] >= 5)
    assert torch.all(packed["anchor_offsets"][1:] >= packed["anchor_offsets"][:-1])
    assert torch.equal(
        packed["labels"][:, 0], packed["sequence_ids"][packed["anchor_offsets"]]
    )
    with pytest.raises(ValueError, match="target layers"):
        next(AnchorStream(tmp_path, expected_layer_ids=(7, 9)))
    fixed = AnchorStream(tmp_path, anchor_policy="first")
    assert [next(fixed)["anchor_position"] for _ in range(4)] == [5] * 4
    limited = FeatureStore(tmp_path / "limited", max_bytes=100, min_free_bytes=0)
    with pytest.raises(BufferError):
        limited.write(record, features, ids, p, layer_ids=(1, 3))


def test_procrustes_recovers_known_rotation():
    torch.manual_seed(6)
    old = torch.randn(50, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    new = old @ rotation
    aligned = procrustes_align(new, old)
    torch.testing.assert_close(new @ aligned, old, atol=2e-6, rtol=2e-6)


def test_processed_sampling_matches_dense():
    torch.manual_seed(5)
    hidden = torch.randn(7, 12)
    head = torch.randn(29, 12)
    ids, p = processed_top20(hidden, head, chunk_size=3)
    dense = hidden @ head.T
    naive, index = dense.softmax(-1).sort(-1, descending=True)
    keep = (naive.cumsum(-1) - naive) <= 0.95
    keep[:, 20:] = False
    naive = torch.where(keep, naive, 0.0)
    naive /= naive.sum(-1, keepdim=True)
    assert torch.equal(ids, index[:, :20])
    torch.testing.assert_close(p, naive[:, :20])


def test_joint_sampling_does_not_apply_top_p_to_top_k_renormalized_mass():
    # Flat enough that the full nucleus is larger than k, so all k survive.
    # Sequential top-k then top-p would incorrectly remove the twentieth token.
    ids, p = processed_top20(torch.ones(1, 1), torch.linspace(0.1, 0, 29)[:, None])
    assert ids.tolist() == [list(range(20))]
    assert torch.all(p > 0)


def test_joint_sampling_preserves_cutoff_and_nucleus_ties():
    ids, p = processed_top20(torch.ones(1, 1), torch.zeros(29, 1))
    assert ids.shape == p.shape == (1, 29)
    assert len(set(ids[0].tolist())) == 29
    torch.testing.assert_close(p, torch.full_like(p, 1 / 29))
    head = torch.full((29, 1), -1000.0)
    head[:3, 0] = torch.tensor([0.5, 0.25, 0.25]).log()
    ids, p = processed_top20(torch.ones(1, 1), head, top_p=0.6)
    dense = torch.zeros(1, 29).scatter_add(1, ids, p)
    torch.testing.assert_close(dense[0, :3], torch.tensor([0.5, 0.25, 0.25]))


def test_tie_overflow_stays_separate_on_disk_and_rejoins_in_training(tmp_path):
    row = make_record()
    path = FeatureStore(tmp_path, min_free_bytes=0).write(
        row,
        torch.randn(25, 2, 8),
        torch.arange(25).expand(25, -1),
        torch.ones(25, 25) / 25,
        layer_ids=(1, 3),
    )
    saved = torch.load(path, weights_only=True)
    assert saved["target_ids"].shape == (25, 20)
    assert saved["target_tie_ids"].shape == (25, 5)
    ids, p = target_distribution(saved)
    torch.testing.assert_close(p.sum(-1), torch.ones(25))
    assert ids.shape == p.shape == (25, 25)
    for cls in (AnchorStream, PackedSequenceStream):
        item = next(cls(tmp_path, anchors_per_sequence=1, block_size=5))
        assert item["target_ids"].shape[-1] == 25
    batch = collate_anchors([next(AnchorStream(tmp_path, block_size=5))], 5)
    assert batch["target_probs"].shape[-1] == 25


def test_global_feature_budget_includes_other_caches_after_restart(tmp_path):
    outside = tmp_path / "heldout/T1/shards"
    outside.mkdir(parents=True)
    (outside / "old.pt").write_bytes(b"x" * 20000)
    root = tmp_path / "train/shards"
    local = FeatureStore(
        root, min_free_bytes=0, budget_root=tmp_path, max_global_bytes=30000
    )
    with pytest.raises(BufferError, match="budget reached"):
        local.write(
            make_record(),
            torch.randn(25, 2, 8),
            torch.arange(20).expand(25, -1),
            torch.ones(25, 20) / 20,
            layer_ids=(1, 3),
        )
    assert not list(root.glob("*.pt"))
    with pytest.raises(ValueError, match="outside the global budget root"):
        FeatureStore(root, min_free_bytes=0, budget_root=outside)


def test_chunked_feature_quantization_is_bit_exact():
    features = torch.randn(1033, 2, 17)
    chunked, scales = quantize_features(features, chunk_size=128)
    reference, reference_scales = quantize_features(features, chunk_size=2048)
    assert torch.equal(chunked, reference) and torch.equal(scales, reference_scales)


def test_overlay_refuses_unknown_source():
    with pytest.raises(ValueError):
        patch("class Unrelated: pass")


@pytest.mark.parametrize("compressed", [False, True])
def test_exclusions_apply_to_raw_generations(tmp_path, monkeypatch, compressed):
    import zstandard

    excluded = tmp_path / "exclude.json"
    excluded.write_text(json.dumps({"ids": ["excluded"]}))
    monkeypatch.setenv("DSPINE_EXCLUDE_IDS", str(excluded))
    body = "".join(
        json.dumps(make_record(key)) + "\n" for key in ("excluded", "sample")
    )
    path = tmp_path / ("records.jsonl.zst" if compressed else "records.jsonl")
    path.write_bytes(
        zstandard.ZstdCompressor().compress(body.encode())
        if compressed
        else body.encode()
    )
    assert [row["id"] for row in records([path])] == ["sample"]
    with pytest.raises(FileNotFoundError):
        load_exclusion_ids(tmp_path / "missing.json")
    excluded.write_text('{"ids": [123]}')
    with pytest.raises(ValueError, match="strings"):
        load_exclusion_ids(excluded)


@pytest.mark.parametrize("stream_type", [AnchorStream, PackedSequenceStream])
def test_exclusions_apply_after_restoring_existing_feature_cursor(
    tmp_path, stream_type
):
    features = tmp_path / "features"
    store = FeatureStore(features, min_free_bytes=0)
    for key in ("old-overlap", "sample"):
        store.write(
            make_record(key),
            torch.randn(25, 2, 8),
            torch.arange(20).expand(25, -1),
            torch.ones(25, 20) / 20,
            layer_ids=(1, 3),
        )
    stream = stream_type(features, anchors_per_sequence=1)
    state = stream.state_dict()
    excluded = tmp_path / "exclude.json"
    excluded.write_text(json.dumps(["old-overlap"]))
    restored = stream_type(features, anchors_per_sequence=1, exclude_ids=excluded)
    restored.load_state_dict(state)
    assert {next(restored)["id"] for _ in range(5)} == {"sample"}


def test_campaign_paths_match_production_merge_and_skip_tuning(tmp_path):
    for relative in (
        "vega/shard-00000/part-one.jsonl.zst",
        "vast-123/shard-00011/part-two.jsonl.zst",
        "vast-123/shard-99999/part-tune.jsonl.zst",
        "modal-smoke/shard-00022/part-smoke.jsonl.zst",
        "modal-bulk/shard-00033/part-three.jsonl.zst",
        "validation/old.jsonl",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert {p.name for p in generation_paths(tmp_path)} == {
        "part-one.jsonl.zst",
        "part-two.jsonl.zst",
        "part-three.jsonl.zst",
    }
