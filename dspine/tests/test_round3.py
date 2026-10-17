"""Opt-in recovery losses and read-only forks from a paused main's latest."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from dspine import ab_lambda
from dspine.forks import validate_fork, validate_recipe_values
from dspine.losses import training_loss
from dspine.recipe_ablation import read_spec, recipe_configs
from dspine.rolling import execute_plan, latest_state, prepare_plan
from dspine.tests.test_ab_lambda import cpu_experiment
from dspine.tests.test_reference import tiny

ROUND3 = Path(__file__).resolve().parents[1] / "configs/round3-recipe.json"


def fixture_loss():
    model = tiny()
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.1)
    anchors = torch.tensor([2, 3])
    output = model(torch.randn(2, 7, 32), anchors)
    labels = torch.randint(0, 20, (2, 5))
    labels[:, 0] = anchors
    ids = torch.arange(20).expand(2, 5, -1)
    probs = torch.rand(2, 5, 20)
    probs /= probs.sum(-1, keepdim=True)
    probs[0, 2, labels[0, 2]] = 0  # Keep C's zero-support masking active.
    probs[0, 2] /= probs[0, 2].sum()
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, 4] = False
    return model, output, labels, ids, probs, valid


def dense_reference(model, output, labels, ids, probs, valid, alpha, gamma, scale):
    w = torch.exp(-torch.arange(4) / gamma) * valid[:, 1:]
    count = valid[:, 0].sum()
    logits = F.linear(output.hidden[:, 1:], model.lm_head.weight)
    p = torch.zeros_like(logits).scatter_add(-1, ids[:, 1:], probs[:, 1:])
    ce = -logits.log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    supported = p.gather(-1, labels[:, 1:, None]).squeeze(-1) > 0
    ce = ce * supported
    l1 = (logits.softmax(-1) - p).abs().sum(-1)
    candidates = logits.detach().argsort(dim=-1, descending=True, stable=True)[..., :3]
    refined = model.last_write(output.pre_injection[:, 1:], labels[:, :-1])
    rlogits = torch.einsum("nth,ntkh->ntk", refined, model.proposal_rows(candidates))
    q = rlogits.softmax(-1)
    selected_p = p.gather(-1, candidates)
    mass = selected_p.sum(-1)
    normalized_p = selected_p / mass[..., None].clamp_min(1e-30)
    match = candidates == labels[:, 1:, None]
    rce = -(rlogits.log_softmax(-1) * match).sum(-1) * supported
    rl1 = (q - normalized_p).abs().sum(-1) * (mass > 1e-4)
    backbone = ((alpha * ce + (1 - alpha) * l1) * w).sum() / count
    refinement = ((alpha * rce + (1 - alpha) * rl1)[:, 1:] * w[:, 1:]).sum() / count
    code = model.selected_prefix_bridge.codebook[labels[:, 1:]]
    embedding = sum(
        ((1 - F.cosine_similarity(z[:, 1:].float(), code.float(), dim=-1)) * w).sum()
        / count
        for z in output.layer_features
    )
    coefficient = (
        0.5
        * (ce * w).sum().detach()
        / count
        / torch.exp(-torch.arange(4) / gamma).sum()
        * scale
    )
    return backbone + refinement + coefficient * embedding, backbone + refinement


@pytest.mark.parametrize(
    "alpha,gamma,scale",
    [(0.1, 7, 1), (0.1, 7, 0), (1, 7, 0), (0.1, 2, 0), (0.1, 7, 0.25)],
)
def test_recovery_losses_and_gradients_match_dense_reference(alpha, gamma, scale):
    model, output, labels, ids, p, valid = fixture_loss()
    args = model, output, labels, ids, p, 10, 100, valid
    actual, metrics = training_loss(
        *args,
        alpha=alpha,
        gamma=gamma,
        embedding_loss_scale=scale,
        lambda_ce_reduction="weighted_token_mean",
        refinement_start_position=2,
        mask_unsupported_ce=True,
        chunk_size=7,
    )
    expected, distillation = dense_reference(
        model, output, labels, ids, p, valid, alpha, gamma, scale
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    params = tuple(x for x in model.parameters() if x.requires_grad)
    ga = torch.autograd.grad(actual, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(expected, params, retain_graph=True, allow_unused=True)
    for left, right in zip(ga, gb):
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=3e-5)
    if scale == 0:
        assert metrics["lambda"].item() == metrics["auxiliary"].item() == 0
        torch.testing.assert_close(actual, distillation)
        # Disabling supervision leaves the bridge active and trainable.
        assert any(
            g is not None and g.abs().max() > 0
            for name, g in zip(
                (
                    name
                    for name, param in model.named_parameters()
                    if param.requires_grad
                ),
                ga,
            )
            if "feature_write_weights" in name
        )
    if (alpha, gamma, scale) == (0.1, 7, 1):
        defaults, values = training_loss(
            *args,
            lambda_ce_reduction="weighted_token_mean",
            refinement_start_position=2,
            mask_unsupported_ce=True,
            chunk_size=7,
        )
        torch.testing.assert_close(defaults, actual, rtol=0, atol=0)
        for key in values:
            torch.testing.assert_close(values[key], metrics[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("embedding_loss_scale", -1),
        ("embedding_loss_scale", float("nan")),
        ("embedding_loss_scale", True),
        ("embedding_loss_scale", None),
        ("loss_alpha", -0.01),
        ("loss_alpha", 1.01),
        ("loss_alpha", float("inf")),
        ("loss_alpha", True),
        ("loss_gamma", 0),
        ("loss_gamma", -2),
        ("loss_gamma", float("nan")),
        ("loss_gamma", True),
    ],
)
def test_bad_recovery_knobs_rejected_before_gpu(key, value, tmp_path):
    with pytest.raises(ValueError, match=key):
        validate_recipe_values({key: value})
    spec = tmp_path / "bad.json"
    spec.write_text(json.dumps({"arms": {"bad": {key: value}}}))
    with pytest.raises(ValueError, match=key):
        read_spec(spec)


def test_round3_config_inherits_lr_and_all_other_hyperparameters(tmp_path):
    parent = dict(
        output=str(tmp_path / "main"),
        lr=3e-4,
        gradient_accumulation=4,
        anchors_per_sequence=256,
        loss_alpha=0.1,
        loss_gamma=7,
        embedding_loss_scale=1,
        schedule_response_tokens=540000000,
    )
    frozen = copy.deepcopy(parent)
    configs = recipe_configs(
        parent, read_spec(ROUND3), tmp_path / "r3", 6400, eval_every=150, cap_gb=21
    )
    assert list(configs) == ["control", "noaux", "noaux_ce", "noaux_g2"]
    for config in configs.values():
        assert config["lr"] == 3e-4 and config["gradient_accumulation"] == 4
        assert config["eval_every"] == 150 and config["eval_step_offset"] == 6400
        assert config["cuda_memory_limit_gb"] == 21
        validate_fork(parent, config, tmp_path / "parent")
    assert configs["control"]["embedding_loss_scale"] == 1
    assert configs["noaux"]["loss_alpha"] == 0.1
    assert configs["noaux_ce"]["loss_alpha"] == 1
    assert configs["noaux_g2"]["loss_gamma"] == 2
    assert parent == frozen
    assert ab_lambda.choose_mode(96e9, 21, "parallel", branches=4) == "parallel"


def test_latest_selector_is_read_only_and_rejects_escaping_or_incomplete_pointers(
    tmp_path,
):
    source = tmp_path / "main"
    checkpoint = source / "step-00006400"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer.pt").write_bytes(b"fixture")
    pointer = source / "latest"
    pointer.write_text(checkpoint.name + "\n")
    before = pointer.read_bytes()
    for path in (checkpoint, pointer, source):
        assert ab_lambda.resolve_checkpoint(path) == checkpoint
    assert pointer.read_bytes() == before
    pointer.write_text("../escape\n")
    with pytest.raises(ValueError, match="single step"):
        ab_lambda.resolve_checkpoint(source)
    pointer.write_text("step-00006401\n")
    with pytest.raises(FileNotFoundError, match="missing"):
        ab_lambda.resolve_checkpoint(pointer)


def test_latest_is_not_read_until_main_is_stopped(tmp_path, monkeypatch):
    def active(*_):
        raise RuntimeError("main controller is running")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("read source checkpoint while main was active")

    monkeypatch.setattr(ab_lambda, "require_main_inactive", active)
    monkeypatch.setattr(ab_lambda, "resolve_checkpoint", unexpected)
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    output = tmp_path / "r3"
    monkeypatch.setattr(
        sys, "argv", ["ab", "--checkpoint", "latest", "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="running"):
        ab_lambda.main()
    assert not output.exists()


def test_round3_from_promoted_main_latest_never_writes_source_and_promotes_winner(
    tmp_path, monkeypatch
):
    root, plan, parent, generations = cpu_experiment(tmp_path, anchors_per_sequence=12)
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", lambda: 100e9)
    monkeypatch.setattr(
        ab_lambda.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    # Create the same shape of promoted source as the real C/lr3 main run.
    seed_spec = tmp_path / "seed.json"
    seed_spec.write_text(
        json.dumps(dict(base_recipe="C", arms={"lr3": {"lr": 0.0003}}))
    )
    seed = tmp_path / "recipe-r2c"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ab",
            "--plan",
            str(root / "plan.json"),
            "--checkpoint",
            str(parent),
            "--output",
            str(seed),
            "--recipe-file",
            str(seed_spec),
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--max-minutes",
            "5",
        ],
    )
    ab_lambda.main()
    main_root = tmp_path / "rolling-r2c-lr3"
    ab_lambda.promote_plan(plan, seed, main_root, arm="lr3")
    main_plan = json.loads((main_root / "plan.json").read_text())
    main_checkpoint = Path(latest_state(main_plan["run"]["output"])["path"])
    source_files = [
        main_root / name
        for name in ("source.json", "run.json", "plan.json", "state.json")
    ]
    source_files += [
        seed / "lr3/latest",
        main_checkpoint / "trainer.pt",
        main_checkpoint / "model.safetensors",
    ]
    hashes = {path: ab_lambda.file_digest(path) for path in source_files}
    output = tmp_path / "recipe-r3"
    argv = [
        "ab",
        "--plan",
        str(main_root / "plan.json"),
        "--checkpoint",
        "latest",
        "--output",
        str(output),
        "--recipe-file",
        str(ROUND3),
        "--steps",
        "1",
        "--eval-every",
        "1",
        "--allocator-cap-gb",
        "21",
        "--mode",
        "parallel",
        "--max-minutes",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ab_lambda.main()
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["source_checkpoint"] == str(main_checkpoint)
    assert experiment["reset_data"] is False
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["complete"] and not comparison["failed_arms"]
    assert set(comparison["paired_samples"].values()) == {1}
    assert all(ab_lambda.file_digest(path) == digest for path, digest in hashes.items())
    for arm in ("control", "noaux", "noaux_ce", "noaux_g2"):
        rows = [
            json.loads(line)
            for line in (output / arm / "metrics.jsonl").read_text().splitlines()
        ]
        training = next(row for row in rows if "sample_traces" in row)
        if arm != "control":
            assert training["lambda"] == training["auxiliary"] == 0
    # Verify the CLI promotion route into another main root, with no default
    # source checkpoint lookup and no mutation of the old promoted controller.
    destination = tmp_path / "rolling-r3-noaux"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ab",
            "--plan",
            str(main_root / "plan.json"),
            "--output",
            str(output),
            "--promote-arm",
            "noaux",
            "--promote-root",
            str(destination),
        ],
    )
    ab_lambda.main()
    config = json.loads((destination / "source.json").read_text())
    continued = prepare_plan(config, generations, destination)
    assert config["embedding_loss_scale"] == 0 and config["lr"] == 0.0003
    assert execute_plan(continued, destination, max_actions=1)["phase"] == "rotate"
    assert latest_state(config["output"])["step"] == 6
    assert all(ab_lambda.file_digest(path) == digest for path, digest in hashes.items())
