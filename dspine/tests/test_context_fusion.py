"""Zero-init checkpoint compatibility and explicit/packed/serving fusion parity."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from dspine import DSpineDraft
from dspine.forks import restore_optimizer
from dspine.losses import training_loss
from dspine.modeling import block_causal_mask
from dspine.tests.test_reference import tiny


@pytest.mark.parametrize("mode", ["anchor", "all"])
def test_zero_init_is_exact_and_preserves_rng_and_old_parameter_order(mode):
    model = tiny()
    features = torch.randn(32, 32)
    ids = torch.randint(0, 29, (32,))
    offsets = torch.tensor([0, 4, 18])
    old_names = [n for n, p in model.named_parameters() if p.requires_grad]
    before = model.forward_packed(features, ids, offsets)
    reference = model(features[None, :4], ids[4:5], reference=True)
    rng = torch.get_rng_state().clone()
    model.configure_context_fusion(mode)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.context_fusion.bias is None
    assert not model.context_fusion.weight.count_nonzero()
    assert [n for n, p in model.named_parameters() if p.requires_grad] == old_names + [
        "context_fusion.weight"
    ]
    after = model.forward_packed(features, ids, offsets)
    for left, right in zip(
        (before.hidden, before.pre_injection, *before.layer_features),
        (after.hidden, after.pre_injection, *after.layer_features),
    ):
        assert torch.equal(left, right)
    assert torch.equal(
        reference.hidden, model(features[None, :4], ids[4:5], reference=True).hidden
    )
    after.hidden.square().sum().backward()
    assert model.context_fusion.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["anchor", "all"])
@pytest.mark.parametrize("anchor", [0, 4, 27])
def test_nonzero_fusion_matches_manual_reference_packed_propose_and_no_future_leak(
    mode, anchor
):
    model = tiny().eval()
    model.configure_context_fusion(mode)
    model.context_fusion.weight.data.normal_(std=0.1)
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.02)
    features = torch.randn(40, 32)
    ids = torch.randint(0, 29, (40,))
    start = max(0, anchor - model.config.context_window)
    cp = (torch.arange(start, anchor) + 45001)[None]
    ap = torch.tensor([anchor + 45001])
    prefix = features[None, start:anchor]
    anchor_ids = ids[anchor : anchor + 1]
    ref = model(prefix, anchor_ids, cp, ap, reference=True)
    fast = model(prefix, anchor_ids, cp, ap)
    packed = model.forward_packed(features, ids, torch.tensor([anchor]), 45001)
    packed_ref = model.forward_packed(
        features, ids, torch.tensor([anchor]), 45001, reference=True
    )

    # Manual decoder execution establishes the precise input addition contract.
    context = model.hidden_norm(model.fc(prefix))
    previous = context[:, -1] if anchor else context.new_zeros(1, 16)
    delta = model.context_fusion(previous)[:, None]
    x = torch.cat(
        (
            model.embedding(anchor_ids)[:, None],
            model.mask_embedding[None, None].expand(1, 4, -1),
        ),
        1,
    )
    x = x + delta if mode == "all" else torch.cat((x[:, :1] + delta, x[:, 1:]), 1)
    mask = block_causal_mask(torch.ones(1, anchor - start, dtype=torch.bool), 5)
    bp = ap[:, None] + torch.arange(5)
    for l, layer in enumerate(model.layers):
        u = layer(x, context, cp, bp, mask, reference=True)
        x = model.selected_prefix_bridge(u, l, anchor_ids)
    torch.testing.assert_close(ref.hidden, model.norm(x), atol=0, rtol=0)
    reference_cache = model.build_transition_cache(ref)
    for output in (fast, packed, packed_ref):
        torch.testing.assert_close(output.hidden, ref.hidden, atol=3e-6, rtol=3e-5)
        cache = model.build_transition_cache(output)
        for temperature in (0, 1):
            got = model.propose(
                cache,
                temperature=temperature,
                generator=torch.Generator().manual_seed(20),
            )
            expected = model.propose(
                reference_cache,
                temperature=temperature,
                generator=torch.Generator().manual_seed(20),
            )
            assert torch.equal(got.tokens, expected.tokens)
            torch.testing.assert_close(
                got.probabilities, expected.probabilities, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(
                got.conditional_probs, expected.conditional_probs, atol=2e-6, rtol=2e-5
            )
    features[anchor:] += 1000
    changed = model.forward_packed(
        features, ids, torch.tensor([anchor]), 45001, reference=True
    )
    torch.testing.assert_close(changed.hidden, packed_ref.hidden, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["anchor", "all"])
def test_padding_selects_exact_absolute_predecessor(mode):
    model = tiny()
    model.configure_context_fusion(mode)
    model.context_fusion.weight.data.normal_(std=0.1)
    features = torch.randn(1, 6, 32)
    # Last physical row is padding, not the predecessor at absolute position 102.
    positions = torch.tensor([[0, 100, 101, 102, 0, 0]])
    valid = torch.tensor([[False, True, True, True, False, False]])
    args = features, torch.tensor([2]), positions, torch.tensor([103]), valid
    padded = model(*args, reference=True)
    cropped = model(
        features[:, 1:4], args[1], positions[:, 1:4], args[3], reference=True
    )
    torch.testing.assert_close(padded.hidden, cropped.hidden, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["anchor", "all"])
def test_old_checkpoint_missing_new_weight_loads_zero_and_new_weight_roundtrips(
    tmp_path, mode
):
    old = tiny()
    old.save_pretrained(tmp_path)
    loaded = DSpineDraft.from_pretrained(
        tmp_path, old.embedding.weight, old.lm_head.weight, context_fusion=mode
    )
    assert not loaded.context_fusion.weight.count_nonzero()
    assert "context_fusion.weight" not in load_file(str(tmp_path / "model.safetensors"))
    features, ids = torch.randn(1, 6, 32), torch.tensor([4])
    assert torch.equal(old(features, ids).hidden, loaded(features, ids).hidden)
    loaded.context_fusion.weight.data.normal_(std=0.1)
    loaded.save_pretrained(tmp_path)
    restored = DSpineDraft.from_pretrained(
        tmp_path, old.embedding.weight, old.lm_head.weight
    )
    assert restored.config.context_fusion == mode
    assert torch.equal(restored(features, ids).hidden, loaded(features, ids).hidden)
    state = load_file(str(tmp_path / "model.safetensors"))
    del state["context_fusion.weight"]
    save_file(state, str(tmp_path / "model.safetensors"))
    assert not DSpineDraft.from_pretrained(
        tmp_path, old.embedding.weight, old.lm_head.weight
    ).context_fusion.weight.count_nonzero()


def test_new_fusion_optimizer_keeps_every_old_moment_and_only_adds_fresh_state():
    old = tiny()
    optimizer = torch.optim.AdamW(
        [p for p in old.parameters() if p.requires_grad], lr=0.001
    )
    old(torch.randn(1, 5, 32), torch.tensor([2])).hidden.square().sum().backward()
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    new = copy.deepcopy(old)
    new.configure_context_fusion("anchor")
    fork_optimizer = torch.optim.AdamW(
        [p for p in new.parameters() if p.requires_grad], lr=0.001
    )
    restore_optimizer(fork_optimizer, saved, new, new_context_fusion=True)
    new_state = fork_optimizer.state_dict()
    assert (
        len(new_state["param_groups"][0]["params"])
        == len(saved["param_groups"][0]["params"]) + 1
    )
    assert new.context_fusion.weight not in fork_optimizer.state
    for key, values in saved["state"].items():
        for name, tensor in values.items():
            assert torch.equal(tensor, new_state["state"][key][name])
    assert len(saved["param_groups"][0]["params"]) + 1 == len(
        new_state["param_groups"][0]["params"]
    )
    new(torch.randn(1, 5, 32), torch.tensor([2])).hidden.square().sum().backward()
    fork_optimizer.step()
    assert fork_optimizer.state[new.context_fusion.weight]["step"].item() == 1


def test_pos1_weight_scales_all_position_one_terms_and_nominal_lambda_mass():
    model = tiny()
    features = torch.randn(2, 6, 32)
    labels = torch.randint(0, 29, (2, 5))
    output = model(features, labels[:, 0])
    ids = torch.arange(29).expand(2, 5, 29)
    probs = torch.ones(2, 5, 29) / 29
    args = model, output, labels, ids, probs, 10, 100
    plain, metrics = training_loss(
        *args, lambda_ce_reduction="weighted_token_mean", refinement_start_position=2
    )
    explicit, default_metrics = training_loss(
        *args,
        pos1_weight=1,
        lambda_ce_reduction="weighted_token_mean",
        refinement_start_position=2,
    )
    assert torch.equal(plain, explicit)
    assert all(torch.equal(v, default_metrics[k]) for k, v in metrics.items())
    # With only t1 valid and no auxiliary, every active term scales by exactly3;
    # C never refines t1. With all positions, nominal lambda mass increases by2.
    valid = torch.zeros(2, 5, dtype=torch.bool)
    valid[:, :2] = True
    base, _ = training_loss(
        *args, valid=valid, embedding_loss_scale=0, refinement_start_position=2
    )
    weighted, _ = training_loss(
        *args,
        valid=valid,
        pos1_weight=3,
        embedding_loss_scale=0,
        refinement_start_position=2,
    )
    torch.testing.assert_close(weighted, 3 * base)
    _, weighted_metrics = training_loss(
        *args,
        pos1_weight=3,
        lambda_ce_reduction="weighted_token_mean",
        refinement_start_position=2,
    )
    torch.testing.assert_close(
        weighted_metrics["lambda_ce_normalizer"], metrics["lambda_ce_normalizer"] + 2
    )
