"""Packed training and isolated serving/eval agree beyond the context window."""

import pytest
import torch

from dspine.tests.test_reference import tiny


@pytest.mark.parametrize('anchor', [4, 12, 27, 50])
def test_deep_anchor_crop_positions_and_no_teacher_leak(anchor):
    m = tiny().eval()
    m.selected_prefix_bridge.feature_write_weights.data.normal_(std=.02)
    features = torch.randn(64, 32)
    ids = torch.randint(0, 29, (64,))
    absolute_start = 45001
    offsets = torch.tensor([anchor])
    packed = m.forward_packed(features, ids, offsets, absolute_start, reference=True)
    start = max(0, anchor - m.config.context_window)
    single = m(features[None, start:anchor], ids[anchor:anchor+1],
               (torch.arange(start, anchor) + absolute_start)[None],
               torch.tensor([anchor + absolute_start]), reference=True)
    torch.testing.assert_close(single.hidden, packed.hidden, atol=2e-6, rtol=2e-5)
    features[anchor:] += 1000  # Anchor and future target features are invisible.
    ids[anchor+1:] = (ids[anchor+1:] + 1) % 29
    labels = torch.randint(0, 29, (1, m.config.block_size))
    changed = m.forward_packed(features, ids, offsets, absolute_start,
                               correct_ids=labels, teacher_blocks=torch.zeros(1, dtype=torch.bool),
                               reference=True)
    torch.testing.assert_close(changed.hidden, packed.hidden, atol=0, rtol=0)


def test_lazy_packed_replay_matches_serial_caches_and_isolates_future_anchors():
    from dspine.eval_rounds import PackedReplay

    m = tiny().eval()
    m.selected_prefix_bridge.feature_write_weights.data.normal_(std=.02)
    row = dict(features=torch.randn(64,2,16).bfloat16(), scales=None, encoding='bf16',
               input_ids=torch.randint(0,29,(64,)), response_start=5, start=45001)
    replay = PackedReplay(m,row,4)
    for anchor in (5,7,18,30,50):
        first = max(0,anchor-m.config.context_window)
        out = m(row['features'][None,first:anchor].float().flatten(-2),row['input_ids'][anchor:anchor+1],
                (torch.arange(first,anchor)+row['start'])[None],torch.tensor([anchor+row['start']]),block_size=4)
        expected = m.build_transition_cache(out)
        got = replay.get(anchor,4)
        assert torch.equal(got.candidate_ids,expected.candidate_ids)
        torch.testing.assert_close(got.initial_scores,expected.initial_scores,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(got.transitions,expected.transitions,atol=3e-6,rtol=3e-5)
    original = replay.get(18,4)
    row['input_ids'][19:] = (row['input_ids'][19:]+1)%29
    row['features'][18:] += 100
    changed = PackedReplay(m,row,4).get(18,4)
    assert torch.equal(changed.candidate_ids,original.candidate_ids)
    torch.testing.assert_close(changed.transitions,original.transitions,atol=0,rtol=0)
