"""Corrected recipe isolation, inference alignment, and resumable unique anchors."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from dspine import ab_lambda
from dspine.data import FeatureStore, PackedSequenceStream, split_for_id
from dspine.losses import proposal_head_loss, refinement_loss, training_loss
from dspine.rolling import execute_plan, latest_state
from dspine.tests.test_ab_lambda import cpu_experiment
from dspine.tests.test_data import make_record
from dspine.tests.test_reference import tiny


def test_unique_anchors_cap_and_resume_keep_control_sequence_order(tmp_path):
    store = FeatureStore(tmp_path, min_free_bytes=0)
    for index in range(8):
        key = f"unique-{index}"
        if split_for_id(key) != "train":
            continue
        row = make_record(key)
        length = 25 if index % 2 else 8
        row["output_ids"] = list(range(5, length))
        store.write(
            row,
            torch.randn(length, 2, 8),
            torch.arange(20).expand(length, -1),
            torch.ones(length, 20) / 20,
            layer_ids=(1, 3),
        )
    kwargs = dict(anchors_per_sequence=12, block_size=5, seed=9)
    control = PackedSequenceStream(tmp_path, **kwargs)
    corrected = PackedSequenceStream(
        tmp_path, **kwargs, anchor_sampling="without_replacement"
    )
    for _ in range(30):  # Several epoch boundaries, not just the initial shuffle.
        a, c = next(control), next(corrected)
        assert a["id"] == c["id"]
        n_valid = len(c["sequence_ids"]) - 1 - 5
        assert len(c["anchor_offsets"]) == 12
        real = c["anchor_offsets"][c["anchor_valid"]]
        assert len(real) == min(12, n_valid)
        assert len(real.unique()) == len(real)
        assert not c["valid"][~c["anchor_valid"]].any()
        assert (c["anchor_offsets"] >= 5).all()
        assert (c["anchor_offsets"] < len(c["sequence_ids"]) - 1).all()
        assert control.rng.getstate() == corrected.rng.getstate()
    state = copy.deepcopy(corrected.state_dict())
    restored = PackedSequenceStream(
        tmp_path, **kwargs, anchor_sampling="without_replacement"
    )
    restored.load_state_dict(state)
    for _ in range(10):
        a, b = next(corrected), next(restored)
        assert a["id"] == b["id"]
        torch.testing.assert_close(
            a["anchor_offsets"], b["anchor_offsets"], rtol=0, atol=0
        )


def loss_fixture():
    model = tiny()
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.2)
    anchors = torch.tensor([2, 3])
    output = model(torch.randn(2, 7, 32), anchors)
    labels = torch.randint(0, 20, (2, 5))
    labels[:, 0] = anchors
    ids = torch.arange(20).expand(2, 5, -1).clone()
    probs = torch.ones(2, 5, 20) / 20
    return model, output, labels, ids, probs


def test_position_one_is_already_written_and_inference_never_refines_it():
    model, output, labels, _, _ = loss_fixture()
    serial = model.last_write(output.pre_injection[:, 1], labels[:, 0])
    torch.testing.assert_close(serial, output.hidden[:, 1], atol=1e-6, rtol=1e-6)
    cache = model.build_transition_cache(output)
    assert cache.transitions.shape[1] == labels.shape[1] - 2
    original = model.propose(cache, temperature=0)
    altered = copy.copy(cache)
    altered.initial_scores = cache.initial_scores.clone()
    altered.initial_scores[:, 1:] = (
        torch.randn_like(altered.initial_scores[:, 1:]) * 1e5
    )
    actual = model.propose(altered, temperature=0)
    torch.testing.assert_close(original.tokens, actual.tokens, atol=0, rtol=0)


def test_corrected_refinement_drops_only_t1_without_reweighting():
    model, output, labels, ids, probs = loss_fixture()
    args = (model, output, labels, ids, probs, 10, 100)
    _, b = training_loss(*args, lambda_ce_reduction="weighted_token_mean")
    _, c = training_loss(
        *args, lambda_ce_reduction="weighted_token_mean", refinement_start_position=2
    )
    _, _, _, candidates = proposal_head_loss(
        model, output.hidden[:, 1:], labels[:, 1:], ids[:, 1:], probs[:, 1:], 4096
    )
    refined = model.last_write(output.pre_injection[:, 1:], labels[:, :-1])
    logits = torch.einsum("nth,ntkh->ntk", refined, model.proposal_rows(candidates))
    ce, l1 = refinement_loss(
        logits, candidates, labels[:, 1:], ids[:, 1:], probs[:, 1:]
    )
    removed = (0.1 * ce[:, 0] + 0.9 * l1[:, 0]).mean()
    torch.testing.assert_close(c["refinement"], b["refinement"] - removed)
    for key in ("backbone", "ce", "l1", "lambda", "embedding"):
        torch.testing.assert_close(c[key], b[key], atol=0, rtol=0)


def test_zero_support_masks_only_ce_with_exact_naive_value_and_gradient():
    model, output, labels, ids, probs = loss_fixture()
    # Every label is unsupported; distribution remains normalized.
    probs.scatter_(-1, labels[..., None], 0)
    probs /= probs.sum(-1, keepdim=True)
    args = (model, output, labels, ids, probs, 10, 100)
    _, unmasked = training_loss(
        *args, lambda_ce_reduction="weighted_token_mean", refinement_start_position=2
    )
    loss, masked = training_loss(
        *args,
        lambda_ce_reduction="weighted_token_mean",
        refinement_start_position=2,
        mask_unsupported_ce=True,
    )
    assert masked["ce"].item() == masked["lambda"].item() == 0
    assert masked["unsupported_ce_fraction"].item() == 1
    torch.testing.assert_close(masked["l1"], unmasked["l1"], atol=0, rtol=0)
    torch.testing.assert_close(
        masked["embedding"], unmasked["embedding"], atol=0, rtol=0
    )
    # With no hard targets and lambda=0, the objective is exactly the two L1s.
    _, l1, _, candidates = proposal_head_loss(
        model, output.hidden[:, 1:], labels[:, 1:], ids[:, 1:], probs[:, 1:], 4096
    )
    refined = model.last_write(output.pre_injection[:, 1:], labels[:, :-1])
    logits = torch.einsum("nth,ntkh->ntk", refined, model.proposal_rows(candidates))
    _, rl1 = refinement_loss(
        logits, candidates, labels[:, 1:], ids[:, 1:], probs[:, 1:]
    )
    w = torch.exp(-torch.arange(4) / 7)
    naive = 0.9 * ((l1 * w).sum() + (rl1[:, 1:] * w[1:]).sum()) / 2
    torch.testing.assert_close(loss, naive)
    parameter = model.fc.weight
    a = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    b = torch.autograd.grad(naive, parameter)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_three_arm_parallel_cap_guard():
    assert ab_lambda.choose_mode(100e9, 30, "parallel", branches=3) == "parallel"
    with pytest.raises(RuntimeError, match="3 allocator"):
        ab_lambda.choose_mode(90e9, 30, "parallel", branches=3)
    assert ab_lambda.choose_mode(100e9, 36, branches=3) == "sequential"


def test_cpu_AB_then_C_same_steps_and_immutable_C_promotion(tmp_path, monkeypatch):
    root, plan, parent, _ = cpu_experiment(tmp_path, anchors_per_sequence=16)
    protected = {
        name: (root / name).read_bytes()
        for name in ("plan.json", "run.json", "state.json")
    }
    output = tmp_path / "ab"
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", lambda: 100_000_000_000)
    monkeypatch.setattr(
        ab_lambda.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    argv = [
        "ab",
        "--plan",
        str(root / "plan.json"),
        "--checkpoint",
        str(parent),
        "--output",
        str(output),
        "--steps",
        "2",
        "--max-minutes",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv + ["--arms", "A", "B", "--mode", "parallel"])
    ab_lambda.main()
    monkeypatch.setattr(sys, "argv", argv + ["--arms", "C"])
    ab_lambda.main()
    # C-only completion is useful for staged launches; the final comparison
    # verifies A/B identical inputs and C identical sequence/schedule inputs.
    result = ab_lambda.summarize(output, [4], 3, 5, arms=("A", "B", "C"))
    assert result["complete"] and result["paired_steps_verified"] == 2
    assert result["sequence_paired_steps_verified"] == {"A_C": 2, "B_C": 2}
    for branch in ("A", "B", "C"):
        assert result["branches"][branch]["last_step"] == 5
        assert [r["step"] for r in result["branches"][branch]["evaluations"]] == [
            3,
            4,
            5,
        ]
    ab_lambda.main()  # Idempotent C re-entry must not add more optimizer updates.
    destination = tmp_path / "rolling-C"
    ab_lambda.promote_plan(plan, output, destination, arm="C")
    config = json.loads((destination / "run.json").read_text())
    saved = torch.load(output / "C/step-00000005/trainer.pt", weights_only=False)
    assert saved["run_config"] == config
    assert config["anchor_sampling"] == "without_replacement"
    assert config["refinement_start_position"] == 2 and config["mask_unsupported_ce"]
    metrics = [
        json.loads(line)
        for line in (output / "C/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["anchors"] for r in metrics if "anchors" in r] == [7, 7]
    continued = json.loads((destination / "plan.json").read_text())
    assert execute_plan(continued, destination, max_actions=1)["phase"] == "rotate"
    final = latest_state(config["output"])
    assert final["step"] == 6
    resumed = torch.load(output / "C/step-00000006/trainer.pt", weights_only=False)
    assert resumed["run_config"] == config
    assert all((root / name).read_bytes() == value for name, value in protected.items())
