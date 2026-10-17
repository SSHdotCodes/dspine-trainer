"""Actual decode p preserves the joint law and survives feature-cache resume."""

import pytest
import torch

from dspine.capture import processed_top20
from dspine.data import FeatureStore, target_distribution
from dspine.decode_probs import attach_decode_probs, processed_decode_probs


@pytest.mark.parametrize("ties", [False, True])
def test_raw_decode_logprobs_reconstruct_joint_distribution(ties):
    torch.manual_seed(2)
    logits = torch.randn(257)
    if ties:
        logits.fill_(-100.0)
        logits[:25] = 0
    raw = logits.log_softmax(-1)
    values, index = raw.topk(64)
    entries = [[float(p), int(i), None] for p, i in zip(values, index)]
    ids, probs = processed_decode_probs([entries], [int(logits.argmax())])
    actual = torch.zeros(257).scatter_add_(
        0, torch.tensor(ids[0]), torch.tensor(probs[0])
    )
    ri, rp = processed_top20(torch.ones(1, 1), logits[:, None])
    expected = torch.zeros(257).scatter_add_(0, ri[0], rp[0])
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_decode_probability_attachment_preserves_features_and_is_idempotent(tmp_path):
    row = dict(
        id="decode",
        domain="math",
        prompt_ids=[1, 2],
        output_ids=[3, 4, 5, 6],
        sampling={"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        target_decode_ids=[[i, 99] for i in (3, 4, 5, 6)],
        target_decode_probs=[[0.3, 0.7]] * 4,
    )
    path = FeatureStore(tmp_path, min_free_bytes=0).write(
        row,
        torch.randn(6, 2, 8),
        torch.arange(20).expand(6, -1),
        torch.ones(6, 20) / 20,
        layer_ids=(1, 3),
    )
    before = torch.load(path, weights_only=True)
    attach_decode_probs(path, row)
    after = torch.load(path, weights_only=True)
    for key in ("features", "scales", "input_ids"):
        assert torch.equal(before[key], after[key])
    ids, p = target_distribution(after)
    py = ((ids[2:] == after["input_ids"][2:, None]) * p[2:]).sum(-1)
    torch.testing.assert_close(py, torch.full((4,), 0.3))
    assert after["target_distribution_source"] == "actual_decode_raw_top_logprobs_joint"
    modified = path.stat().st_mtime_ns
    attach_decode_probs(path, row)
    assert path.stat().st_mtime_ns == modified
