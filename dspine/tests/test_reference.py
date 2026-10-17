import itertools
import tempfile

import pytest
import torch
from torch.nn import functional as F

from dspine import DSpineConfig, DSpineDraft
from dspine.data import dequantize_features, quantize_features
from dspine.eval_offline import path_acceptance
from dspine.losses import (
    chunked_head_loss,
    chunked_topk,
    curriculum_probability,
    refinement_loss,
    training_loss,
)
from dspine.modeling import TransitionCache, block_causal_mask
from dspine.transfer_space import build_transfer_space


def tiny():
    torch.manual_seed(7)
    c = DSpineConfig(
        hidden_size=16,
        target_hidden_size=16,
        vocab_size=29,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        target_layer_ids=(1, 3),
        block_size=5,
        context_window=12,
        code_size=8,
        message_size=4,
        candidate_count=3,
    )
    model = DSpineDraft(
        c,
        torch.randn(29, 16),
        torch.randn(29, 16),
        F.normalize(torch.randn(29, 8), dim=-1),
    )
    return model


def test_identity_and_anchor():
    model = tiny()
    bridge = model.selected_prefix_bridge
    u = torch.randn(2, 5, 16)
    anchor = torch.tensor([2, 3])
    assert torch.equal(u, bridge(u, 0, anchor))
    bridge.feature_write_weights.data.normal_()
    assert torch.equal(u[:, :1], bridge(u, 0, anchor)[:, :1])
    z = bridge.read(u, 0)
    serial = torch.stack(
        [
            bridge.write(
                u[:, t], z[:, t], bridge.token(anchor) if t == 1 else z[:, t - 1], 0
            )
            for t in range(1, 5)
        ],
        1,
    )
    torch.testing.assert_close(bridge(u, 0, anchor)[:, 1:], serial)


def test_mask():
    valid = torch.tensor([[False, True, True]])
    mask = block_causal_mask(valid, 3)[0, 0]
    assert mask.tolist() == [
        [False, True, True, True, False, False],
        [False, True, True, True, True, False],
        [False, True, True, True, True, True],
    ]


def test_reference_sdpa_and_prefix():
    m = tiny()
    m.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.1)
    features = torch.randn(2, 7, 32)
    anchor = torch.tensor([2, 3])
    fast = m(features, anchor)
    ref = m(features, anchor, reference=True)
    torch.testing.assert_close(fast.hidden, ref.hidden, atol=2e-6, rtol=2e-5)
    prefix = m(features, anchor, block_size=3)
    torch.testing.assert_close(prefix.hidden, fast.hidden[:, :3], atol=2e-6, rtol=2e-5)
    valid = torch.tensor([[False, False, True, True, True, True, True]]).expand(2, -1)
    first = m(features, anchor, context_valid=valid)
    features[:, :2].normal_(std=100)
    second = m(features, anchor, context_valid=valid)
    torch.testing.assert_close(first.hidden, second.hidden)


def test_transition_cache_serial_and_proposal():
    m = tiny()
    m.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.1)
    out = m(torch.randn(2, 7, 32), torch.tensor([2, 3]))
    cache = m.build_transition_cache(out, chunk_size=7)
    for t in range(1, 4):
        for i in range(3):
            refined = m.last_write(
                out.pre_injection[:, t + 1], cache.candidate_ids[:, t - 1, i]
            )
            logits = torch.einsum(
                "nh,nkh->nk",
                refined,
                F.embedding(cache.candidate_ids[:, t], m.lm_head.weight),
            )
            torch.testing.assert_close(
                cache.transitions[:, t - 1, i], logits, atol=5e-6, rtol=1e-5
            )
    # Exhaustively sum all 3^4 paths for one anchor, verifying q is normalized.
    total = 0.0
    for path in itertools.product(range(3), repeat=4):
        prob = cache.initial_scores[0, 0].softmax(-1)[path[0]]
        for t in range(1, 4):
            prob = prob * cache.transitions[0, t - 1, path[t - 1]].softmax(-1)[path[t]]
        total += float(prob.detach())
    assert total == pytest.approx(1.0, abs=1e-6)
    prop = m.propose(
        cache, temperature=0.7, generator=torch.Generator().manual_seed(42)
    )
    for n in range(2):
        last = None
        for t in range(4):
            index = (cache.candidate_ids[n, t] == prop.tokens[n, t]).nonzero().item()
            scores = (
                cache.initial_scores[n, 0]
                if t == 0
                else cache.transitions[n, t - 1, last]
            )
            q = (scores / 0.7).softmax(-1)
            torch.testing.assert_close(prop.probabilities[n, t], q[index])
            torch.testing.assert_close(prop.conditional_probs[n, t], q)
            last = index
    greedy = m.propose(cache, temperature=0)
    assert torch.equal(greedy.probabilities, torch.ones(2, 4))


@pytest.mark.parametrize("chunk", [3, 11, 29])
def test_chunked_loss_value_and_gradient(chunk):
    torch.manual_seed(9)
    h = torch.randn(6, 9, requires_grad=True)
    head = torch.randn(29, 9)
    y = torch.randint(29, (6,))
    ids = torch.stack([torch.randperm(29)[:5] for _ in range(6)])
    p = torch.rand(6, 5)
    p /= p.sum(-1, keepdim=True)
    ce, l1 = chunked_head_loss(h, head, y, ids, p, chunk)
    logits = F.linear(h, head)
    dense = torch.zeros_like(logits).scatter(1, ids, p)
    naive_ce = F.cross_entropy(logits, y, reduction="none")
    naive_l1 = (logits.softmax(-1) - dense).abs().sum(-1)
    torch.testing.assert_close(ce, naive_ce)
    torch.testing.assert_close(l1, naive_l1)
    weights = torch.rand(6)
    loss = ((0.1 * ce + 0.9 * l1) * weights).sum()
    actual = torch.autograd.grad(loss, h, retain_graph=True)[0]
    expected = torch.autograd.grad(
        ((0.1 * naive_ce + 0.9 * naive_l1) * weights).sum(), h
    )[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    values, indices = chunked_topk(h, head, 4, chunk)
    torch.testing.assert_close(values, logits.topk(4).values)
    assert torch.equal(indices, logits.topk(4).indices)


def test_refinement_loss_and_curriculum():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], requires_grad=True)
    cand = torch.tensor([[1, 3, 5], [2, 4, 6]])
    y = torch.tensor([3, 7])
    ids = torch.tensor([[1, 2, 3], [1, 3, 5]])
    p = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    ce, l1 = refinement_loss(logits, cand, y, ids, p)
    torch.testing.assert_close(ce[0], -logits[0].log_softmax(-1)[1])
    assert ce[1] == 0 and l1[1] == 0
    torch.testing.assert_close(
        l1[0], (logits[0].softmax(-1) - torch.tensor([2 / 7, 5 / 7, 0.0])).abs().sum()
    )
    assert curriculum_probability(0) == 0.5 and curriculum_probability(1 / 6) == 0.5
    assert (
        curriculum_probability(1 / 4) == pytest.approx(0.25)
        and curriculum_probability(1 / 3) == 0
    )


def test_fp8_and_pca():
    x = torch.randn(50, 3, 16)
    raw, scales = quantize_features(x)
    decoded = dequantize_features(
        {"encoding": "fp8_e4m3fn", "features": raw, "scales": scales},
        dtype=torch.float32,
    ).view_as(x)
    assert F.cosine_similarity(x, decoded, dim=-1).mean() > 0.999
    head = torch.randn(100, 16)
    result = build_transfer_space(head, 8, chunk_size=13)
    torch.testing.assert_close(
        result["basis"].T @ result["basis"], torch.eye(8), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        result["codebook"].float().norm(dim=-1), torch.ones(100), atol=0.004, rtol=0.004
    )
    mean = F.normalize(head.double(), dim=-1).mean(0)
    code = F.normalize(
        ((F.normalize(head.double(), dim=-1) - mean) @ result["basis"].double())
        * result["eigenvalues"].double().rsqrt(),
        dim=-1,
    )
    torch.testing.assert_close(
        result["codebook"].float(), code.float(), atol=0.004, rtol=0.008
    )


def test_exact_on_policy_acceptance_identity():
    # Enumerate target trajectories, comparing reverse q/p to forward p/q.
    p0 = torch.tensor([0.8, 0.2])
    p1 = torch.tensor([[0.4, 0.6], [0.9, 0.1]])
    q0 = torch.tensor([0.3, 0.7])
    q1 = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    expected = torch.zeros(2)
    reverse = torch.zeros(2)
    cache = TransitionCache(
        torch.tensor([[[0, 1], [0, 1]]]),
        torch.stack((q0.log(), q0.log()))[None],
        q1.log()[None, None],
    )
    for a, b in itertools.product(range(2), repeat=2):
        probp = p0[a] * p1[a, b]
        probq = q0[a] * q1[a, b]
        expected[0] += probq * min(1.0, float(p0[a] / q0[a]))
        expected[1] += (
            probq
            * min(1.0, float(p0[a] / q0[a]))
            * min(1.0, float(p1[a, b] / q1[a, b]))
        )
        survival = path_acceptance(
            cache,
            torch.tensor([[a, b]]),
            torch.tensor([[[0, 1], [0, 1]]]),
            torch.stack((p0, p1[a]))[None],
        )[0]
        reverse += probp * survival
    torch.testing.assert_close(reverse, expected)


def test_training_backward_checkpoint_and_roundtrip():
    m = tiny()
    m.config.gradient_checkpointing = True
    labels = torch.randint(29, (2, 5))
    ids = torch.stack([torch.randperm(29)[:20] for _ in range(10)]).view(2, 5, 20)
    probs = torch.rand(2, 5, 20)
    probs /= probs.sum(-1, keepdim=True)
    features = torch.randn(2, 7, 32)
    out = m(
        features,
        labels[:, 0],
        correct_ids=labels,
        teacher_blocks=torch.tensor([True, False]),
    )
    loss, metrics = training_loss(m, out, labels, ids, probs, 10, 100, chunk_size=7)
    loss.backward()
    assert torch.isfinite(loss) and m.fc.weight.grad.isfinite().all()
    assert m.embedding.weight.grad is None and m.lm_head.weight.grad is None
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = DSpineDraft.from_pretrained(d, m.embedding.weight, m.lm_head.weight)
        m.eval()
        loaded.eval()
        torch.testing.assert_close(
            m(features, labels[:, 0]).hidden, loaded(features, labels[:, 0]).hidden
        )


def test_packed_attention_matches_independent_anchors_and_gradients():
    model = tiny()
    model.config.context_window = 4
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.1)
    features = torch.randn(12, 32)
    ids = torch.randint(29, (12,))
    anchors = torch.tensor([2, 6, 9])
    packed = model.forward_packed(
        features, ids, anchors, sequence_start=17, reference=True
    )
    independent = []
    for a in anchors:
        start = max(0, int(a) - 4)
        independent.append(
            model(
                features[start:a][None],
                ids[a][None],
                torch.arange(start, a)[None] + 17,
                torch.tensor([int(a) + 17]),
                reference=True,
            ).hidden
        )
    independent = torch.cat(independent)
    torch.testing.assert_close(packed.hidden, independent, atol=3e-6, rtol=1e-5)
    projection = torch.randn_like(packed.hidden)
    gp = torch.autograd.grad(
        (packed.hidden * projection).sum(), model.fc.weight, retain_graph=True
    )[0]
    gi = torch.autograd.grad((independent * projection).sum(), model.fc.weight)[0]
    torch.testing.assert_close(gp, gi, atol=5e-6, rtol=1e-4)


def test_topk_ties_are_token_id_order_and_independent_of_chunks():
    from dspine.losses import stable_topk

    h = torch.ones(2, 3)
    head = torch.zeros(29, 3)
    for chunk in (3, 11, 29):
        values, ids = chunked_topk(h, head, 5, chunk_size=chunk)
        assert torch.equal(ids, torch.arange(5).expand(2, -1))
        assert torch.equal(values, torch.zeros(2, 5))
    scores = torch.tensor([[-3.0, -1.0, -1.0, 0.0, 4.0, -0.0]])
    ids = torch.tensor([[6, 4, 2, 9, 8, 1]])
    value, index = stable_topk(scores, ids, 6)
    assert index.tolist() == [[8, 1, 9, 2, 4, 6]]


def test_entire_objective_matches_naive_reference():
    m = tiny()
    m.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.05)
    labels = torch.randint(29, (2, 5))
    ids = torch.stack([torch.randperm(29)[:20] for _ in range(10)]).view(2, 5, 20)
    p = torch.rand(2, 5, 20)
    p /= p.sum(-1, keepdim=True)
    out = m(torch.randn(2, 7, 32), labels[:, 0])
    loss, _ = training_loss(m, out, labels, ids, p, 2, 100, chunk_size=7)
    logits = F.linear(out.hidden[:, 1:], m.lm_head.weight)
    logq = logits.log_softmax(-1)
    q = logq.exp()
    dense = torch.zeros_like(q).scatter(-1, ids[:, 1:], p[:, 1:])
    ce = -logq.gather(-1, labels[:, 1:, None]).squeeze(-1)
    l1 = (q - dense).abs().sum(-1)
    _, candidates = chunked_topk(out.hidden[:, 1:], m.lm_head.weight, 3, 7)
    refined = m.last_write(out.pre_injection[:, 1:], labels[:, :-1])
    rlogits = torch.einsum(
        "nth,ntkh->ntk", refined, F.embedding(candidates, m.lm_head.weight)
    )
    rlogq = rlogits.log_softmax(-1)
    rq = rlogq.exp()
    rp = dense.gather(-1, candidates)
    mass = rp.sum(-1)
    rp /= mass.clamp_min(1e-30)[..., None]
    rce = -(rlogq * (candidates == labels[:, 1:, None])).sum(-1)
    rl1 = torch.where(mass > 1e-4, (rq - rp).abs().sum(-1), 0.0)
    weights = torch.exp(-torch.arange(4) / 7.0)
    reduce = lambda x: (x * weights).sum() / 2
    emb = sum(
        reduce(
            1
            - F.cosine_similarity(
                read[:, 1:], m.selected_prefix_bridge.codebook[labels[:, 1:]], dim=-1
            )
        )
        for read in out.layer_features
    )
    lam = 0.5 * (2 / 4) * reduce(ce).detach()
    expected = reduce(0.1 * ce + 0.9 * l1) + reduce(0.1 * rce + 0.9 * rl1) + lam * emb
    torch.testing.assert_close(loss, expected, atol=2e-5, rtol=2e-6)
