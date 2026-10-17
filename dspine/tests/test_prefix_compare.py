"""Compare per-layer runtime drift before flattening or storage quantization."""

import torch

from dspine.data import FeatureStore
from dspine.prefix_compare import compare
from dspine.transfer_space import qwenfast_w8_weights


def test_probe_comparison_preserves_layer_axis_and_checks_effective_head(tmp_path):
    torch.manual_seed(12)
    base, production = tmp_path / "base", tmp_path / "production"
    features = torch.randn(6, 2, 128).bfloat16()
    changed = features.clone()
    changed[:, 0] *= 1.01
    changed[:, 1] *= 1.1
    head = torch.randn(29, 128).bfloat16()
    record = dict(
        id="probe",
        domain="math",
        prompt_ids=[1, 2],
        output_ids=[3, 4, 5, 6],
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    )
    ids = torch.arange(20).expand(6, -1)
    p = torch.ones(6, 20) / 20
    for root, values, weight in (
        (base, features, head),
        (production, changed, qwenfast_w8_weights(head)),
    ):
        path = FeatureStore(root / "shards", min_free_bytes=0).write(
            record, values, ids, p, encoding="bf16", layer_ids=(1, 3)
        )
        (root / "probe-logits").mkdir()
        torch.save(torch.zeros(4, 29), root / "probe-logits" / path.name)
        torch.save({"ids": torch.arange(29), "weight": weight}, root / "probe-head.pt")
    result = compare(base, production)
    assert result["effective_head_reconstruction_max_error"] == 0
    probe = result["probes"][0]
    assert set(probe["features"]) == {"1", "3"}
    assert 0.005 < probe["features"]["1"]["relative_l2"] < 0.02
    assert 0.08 < probe["features"]["3"]["relative_l2"] < 0.12
    assert probe["processed_l1"] == 0
    assert probe["retained_tokens"] == 6
