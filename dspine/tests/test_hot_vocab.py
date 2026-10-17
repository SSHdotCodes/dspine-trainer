"""Restricted q retains exact full-p L1 and serving-aligned candidate rows."""

import torch
from torch.nn import functional as F

from dspine.hot_vocab import row_int8_weights
from dspine.losses import proposal_head_loss
from dspine.modeling import DSpineDraft
from dspine.tests.test_reference import tiny


def test_hot_loss_and_gradient_match_full_vocab_naive_reference():
    m = tiny()
    m.configure_proposal_vocabulary([20, 4, 0, 17, 8], quantization="none")
    h = torch.randn(3, 16, requires_grad=True)
    labels = torch.tensor([4, 28, 20])
    ids = torch.tensor([[0, 4, 8, 27], [4, 17, 26, 27], [0, 17, 20, 28]])
    p = torch.tensor([[0.1, 0.3, 0.2, 0.4], [0.2, 0.3, 0.4, 0.1], [0.1, 0.4, 0.4, 0.1]])
    ce, l1, _, candidates = proposal_head_loss(m, h, labels, ids, p, chunk_size=2)
    qhot = F.linear(h, m.proposal_weight).softmax(-1)
    q = torch.zeros(3, 29).scatter(1, m.proposal_token_ids.expand(3, -1), qhot)
    dense_p = torch.zeros_like(q).scatter_add(1, ids, p)
    reference_l1 = (q - dense_p).abs().sum(-1)
    local_y = m.proposal_lookup[labels]
    reference_ce = -qhot.gather(1, local_y.clamp_min(0)[:, None]).log().squeeze(-1) * (
        local_y >= 0
    )
    torch.testing.assert_close(ce, reference_ce)
    torch.testing.assert_close(l1, reference_l1)
    expected_grad = torch.autograd.grad((reference_ce + reference_l1).sum(), h)[0]
    got_grad = torch.autograd.grad((ce + l1).sum(), h)[0]
    torch.testing.assert_close(got_grad, expected_grad, atol=2e-6, rtol=2e-5)
    assert torch.all(m.proposal_lookup[candidates] >= 0)


def test_hot_policy_roundtrip_cache_and_serving_rows(tmp_path):
    m = tiny().bfloat16().eval()
    ids = torch.tensor([20, 4, 0, 17, 8])
    m.configure_proposal_vocabulary(ids)
    out = m(torch.randn(1, 7, 32).bfloat16(), torch.tensor([2]))
    cache = m.build_transition_cache(out, chunk_size=2)
    assert torch.all(m.proposal_lookup[cache.candidate_ids] >= 0)
    refined = m.last_write(
        out.pre_injection[:, 2:, None].expand(-1, -1, 3, -1),
        cache.candidate_ids[:, :-1],
    )
    expected = torch.einsum(
        "ntih,ntjh->ntij", refined, m.proposal_rows(cache.candidate_ids[:, 1:])
    ).float()
    torch.testing.assert_close(expected, cache.transitions, atol=0, rtol=0)
    m.save_pretrained(tmp_path)
    restored = DSpineDraft.from_pretrained(
        tmp_path, m.embedding.weight, m.lm_head.weight
    )
    assert restored.proposal_policy == m.proposal_policy
    assert not any(k.startswith("proposal_") for k in m.trainable_state_dict())
    torch.testing.assert_close(
        restored.proposal_weight, m.proposal_weight, atol=0, rtol=0
    )


def test_restricted_candidate_ties_use_global_id_order():
    m = tiny()
    m.lm_head.weight.data.zero_()
    m.configure_proposal_vocabulary([28, 4, 2, 1, 0], quantization="none")
    out = m(torch.randn(1, 7, 32), torch.tensor([2]))
    assert m.build_transition_cache(out, chunk_size=2).candidate_ids[0, 0].tolist() == [
        0,
        1,
        2,
    ]


def test_vocab_expansion_never_fits_validation_tokens(tmp_path):
    from dspine.data import FeatureStore
    from dspine.hot_vocab import expand_from_training

    store = FeatureStore(tmp_path, min_free_bytes=0)
    for name, domain, split, token in [
        ("a", "code", "train", 20),
        ("b", "math", "train", 21),
        ("v", "math", "validation", 28),
    ]:
        record = dict(
            id=name,
            domain=domain,
            split=split,
            prompt_ids=[0],
            output_ids=[token, token],
            sampling=dict(temperature=1.0, top_k=20, top_p=0.95),
        )
        ids = torch.zeros(3, 20, dtype=torch.long)
        ids[:, 1] = token
        probs = torch.zeros(3, 20)
        probs[:, :2] = 0.5
        store.write(record, torch.randn(3, 1, 4), ids, probs, layer_ids=(1,))
    ids, masses = expand_from_training([tmp_path], torch.tensor([0]), 29, 0.99)
    assert ids.tolist() == [0, 20, 21]
    assert set(masses) == {"code", "math"} and min(masses.values()) == 1


def test_raw_coverage_reports_independent_validation_domain():
    from collections import Counter

    from dspine.hot_vocab import raw_label_coverage

    counts = {
        "train": {"multilingual": Counter({1: 999, 2: 1})},
        "validation": {"multilingual": Counter({1: 8, 2: 2})},
    }
    result = raw_label_coverage(
        None, {"hot": torch.tensor([1])}, cached_counts=(counts, 2)
    )
    assert (
        result["groups"]["split:validation/domain:multilingual"]["coverage"]["hot"]
        == 0.8
    )
    assert (
        result["groups"]["split:train/domain:multilingual"]["coverage"]["hot"] == 0.999
    )


def test_row_int8_effective_rows_include_bf16_rounding():
    rows = torch.tensor(
        [[0.0, 1.0, -1.0], [2.0, -2.0, 0.4], [0.0, 0.0, 0.0]], dtype=torch.bfloat16
    )
    expected = torch.tensor(
        [[0.0, 1.0, -1.0], [2.0, -2.0, 50 / 127], [0.0, 0.0, 0.0]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(row_int8_weights(rows), expected, atol=0, rtol=0)


def test_optional_serving_head_rows_match_reference():
    import pytest

    module = pytest.importorskip(
        "serving.quantization",
        reason="optional serving parity package is not installed",
    )
    model = tiny().bfloat16().eval()
    ids = torch.tensor([20, 4, 0, 17, 8])
    model.configure_proposal_vocabulary(ids)
    serving = module.HotInt8Head(model.lm_head.weight, ids, qwenfast=False)
    torch.testing.assert_close(
        model.proposal_rows(ids), serving.rows(ids, torch.bfloat16), atol=0, rtol=0
    )
    torch.testing.assert_close(
        row_int8_weights(model.lm_head.weight[ids]),
        serving.rows(ids, torch.bfloat16),
        atol=0,
        rtol=0,
    )
