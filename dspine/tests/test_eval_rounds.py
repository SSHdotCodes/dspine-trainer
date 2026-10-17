"""Reverse coupling matches accepted/residual laws and advances real rounds."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dspine.data import FeatureStore
from dspine.eval_offline import path_acceptance
from dspine.eval_rounds import emitted_lengths, evaluate_rounds
from dspine.modeling import TransitionCache


def cache(batch=1, block=4):
    q = torch.tensor([0.75, 0.25]).log()
    return TransitionCache(
        torch.arange(2).expand(batch, block - 1, -1),
        q.expand(batch, block - 1, -1),
        q.expand(batch, block - 2, 2, -1),
    )


def test_reverse_acceptance_has_the_exact_accepted_and_residual_joint_law():
    n = 20000
    generator = torch.Generator().manual_seed(4)
    labels = torch.randint(2, (n, 1), generator=generator)
    support = torch.arange(2).expand(n, 1, -1)
    p = torch.full((n, 1, 2), 0.5)
    survival = path_acceptance(
        cache(n, 2), labels, support, p, monte_carlo=True, generator=generator
    )
    accepted = survival[:, 0].bool()
    # min(p,q) is (.5,.25); the residual mass is entirely on token1.
    assert abs(((labels[:, 0] == 0) & accepted).float().mean().item() - 0.5) < 0.02
    assert abs(((labels[:, 0] == 1) & accepted).float().mean().item() - 0.25) < 0.02
    assert not ((labels[:, 0] == 0) & ~accepted).any()
    assert (
        abs(emitted_lengths(survival, labels, ()).float().mean().item() - 1.75) < 0.02
    )


def test_emission_omits_bonus_after_accepted_eos():
    survival = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    labels = torch.tensor([[3, 28, 0], [2, 3, 0], [2, 3, 4]])
    assert emitted_lengths(survival, labels, (28,)).tolist() == [2, 2, 4]


class ConstantDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(29, 4)
        self.config = SimpleNamespace(
            block_size=4, context_window=8, target_layer_ids=(1,), eos_token_ids=(28,)
        )
        self.cache = cache()

    def forward(self, *args, **kwargs):
        return None

    def build_transition_cache(self, out):
        return self.cache

    def propose(self, cache, temperature):
        return SimpleNamespace(tokens=torch.zeros_like(cache.candidate_ids[..., 0]))


def store_row(root, ids, temperature=1.0):
    row = {
        "id": "heldout",
        "domain": "math",
        "prompt_ids": [3],
        "output_ids": ids,
        "split": "validation",
        "model_rev": "rev",
        "finish_reason": "length",
        "sampling": {"temperature": temperature, "top_k": 20, "top_p": 0.95},
    }
    n = len(ids) + 1
    probs = torch.zeros(n, 20)
    probs[:, :2] = 0.5
    FeatureStore(root, min_free_bytes=0).write(
        row, torch.randn(n, 1, 4), torch.arange(20).expand(n, -1), probs, layer_ids=(1,)
    )


def test_full_walk_matches_known_tau_and_keeps_model_training_state(tmp_path):
    ids = torch.randint(2, (3000,), generator=torch.Generator().manual_seed(9)).tolist()
    store_row(tmp_path, ids)
    model = ConstantDraft()
    report = evaluate_rounds(
        model, tmp_path, blocks=(2,), sequences=1, seed=5, expected_revision="rev"
    )
    stats = report["by_domain"]["math"]["2"]
    assert abs(stats["tau"] - 1.75) < 0.04
    assert stats["rounds"] > 1000 and stats["censored_tails"] == 1
    assert report["complete"] and model.training


def test_greedy_round_walk_counts_bonus_and_censors_only_partial_tail(tmp_path):
    store_row(tmp_path, [0] * 50, temperature=0.0)
    report = evaluate_rounds(
        ConstantDraft(), tmp_path, blocks=(4,), temperature=0.0, sequences=1
    )
    stats = report["by_domain"]["math"]["4"]
    assert stats["tau"] == 4 and stats["tokens"] == 48 and stats["rounds"] == 12
    with pytest.raises(ValueError, match="production on-policy"):
        evaluate_rounds(
            ConstantDraft(), tmp_path, blocks=(4,), temperature=1.0, sequences=1
        )


def test_fixed_prefix_windows_group_actual_complete_rounds(tmp_path):
    store_row(tmp_path, [0] * 5000, temperature=0.)
    report = evaluate_rounds(ConstantDraft(), tmp_path, blocks=(4,), temperature=0.,
        sequences=1, offsets=(0,256,1024,4096), window_tokens=128, collect_diagnostics=True)
    assert report['walk_mode'] == 'fixed_prefix_windows'
    assert report['by_domain']['math']['4']['tokens'] == 4 * 128
    assert report['by_domain']['math']['4']['tau'] == 4
    for name in ('0-256','256-1K','1K-4K','>=4K'):
        assert report['by_anchor']['offset:'+name]['4']['rounds'] == 32
    assert report['first_round']['4']['rao_blackwell_tau'] == 4


def test_packed_replay_walk_has_identical_sampling_and_boundaries(tmp_path):
    class ReplayDraft(ConstantDraft):
        def forward(self,features,*args,**kwargs):return len(features)

        def forward_packed(self,features,ids,offsets,*args,**kwargs):
            from dspine.modeling import DraftOutput

            h = torch.zeros(len(offsets),4,1)
            return DraftOutput(h,h,())

        def build_transition_cache(self,n):return cache(n.hidden.shape[0] if hasattr(n,'hidden') else n,4)

    ids = torch.randint(2,(1000,),generator=torch.Generator().manual_seed(17)).tolist()
    store_row(tmp_path,ids)
    serial = evaluate_rounds(ReplayDraft(),tmp_path,blocks=(4,),sequences=1,collect_diagnostics=True)
    packed = evaluate_rounds(ReplayDraft(),tmp_path,blocks=(4,),sequences=1,collect_diagnostics=True,packed_chunk_size=32)
    assert packed['by_domain'] == serial['by_domain']
    assert packed['by_anchor'] == serial['by_anchor']
    assert packed['first_round'] == serial['first_round']


def test_packed_replay_skips_unused_head_rows():
    from dspine.eval_rounds import PackedReplay
    from dspine.tests.test_reference import tiny

    model = tiny().eval()
    row = dict(features=torch.randn(24,2,16),scales=None,encoding="bf16",
               input_ids=torch.randint(0,29,(24,)),response_start=7,start=5000)
    with torch.inference_mode():
        complete = PackedReplay(model,row,chunk_size=4)
        bounded = PackedReplay(model,row,chunk_size=4,max_block=3)
        a,b = complete.get(8,3),bounded.get(8,3)
    assert torch.equal(a.candidate_ids,b.candidate_ids)
    torch.testing.assert_close(a.initial_scores,b.initial_scores)
    torch.testing.assert_close(a.transitions,b.transitions)
    assert bounded.caches[0].candidate_ids.shape[1] == 2
