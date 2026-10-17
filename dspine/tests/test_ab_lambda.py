"""A/B changes only auxiliary scaling, and preserves optimizer/RNG/data inputs."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dspine import ab_lambda
from dspine.data import FeatureStore, records
from dspine.forks import validate_fork
from dspine.losses import training_loss
from dspine.rolling import (
    atomic_json,
    execute_plan,
    latest_state,
    prepare_plan,
    purge_window,
)
from dspine.tests.test_reference import tiny
from dspine.tests.test_rolling import fake_capture, fixture, plan_for


def test_lambda_mean_changes_only_auxiliary_factor_and_gradient():
    torch.manual_seed(12)
    model = tiny()
    output = model(torch.randn(2, 7, 32), torch.tensor([2, 3]))
    labels = torch.randint(0, 20, (2, 5))
    ids = torch.arange(20).expand(2, 5, -1)
    probs = torch.ones(2, 5, 20) / 20
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, 3:] = False  # B retains the same tail mask and nominal normalizer.
    args = (model, output, labels, ids, probs, 10, 100, valid)
    original, old = training_loss(*args, chunk_size=7)
    a, am = training_loss(*args, chunk_size=7, lambda_ce_reduction="position_sum")
    b, bm = training_loss(
        *args, chunk_size=7, lambda_ce_reduction="weighted_token_mean"
    )
    torch.testing.assert_close(original, a, atol=0, rtol=0)
    mass = torch.exp(-torch.arange(4) / 7).sum()
    for key in ("ce", "l1", "backbone", "refinement", "embedding", "distillation"):
        torch.testing.assert_close(am[key], bm[key], atol=0, rtol=0)
    torch.testing.assert_close(bm["lambda"], am["lambda"] / mass)
    torch.testing.assert_close(b, am["distillation"] + am["auxiliary"] / mass)
    parameter = model.selected_prefix_bridge.read_weight
    ga = torch.autograd.grad(a, parameter, retain_graph=True)[0]
    gb = torch.autograd.grad(b, parameter)[0]
    torch.testing.assert_close(gb, ga / mass, atol=1e-7, rtol=1e-5)
    assert old["lambda_ce_normalizer"].item() == 1


@pytest.mark.parametrize(
    "changed", ["head_chunk_size", "features", "seed", "schedule_response_tokens"]
)
def test_fork_rejects_unrelated_hyperparameter_changes(changed, tmp_path):
    source = {"output": str(tmp_path / "full"), changed: 1}
    destination = {**source, "output": str(tmp_path / "B"), changed: 2}
    with pytest.raises(ValueError, match="unsupported fork"):
        validate_fork(source, destination, tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="separate"):
        validate_fork(source, source, tmp_path / "checkpoint")


def test_mode_selection_uses_two_allocator_caps_not_optimistic_peaks():
    assert ab_lambda.choose_mode(80_000_000_000, 36) == "parallel"
    assert ab_lambda.choose_mode(60_000_000_000, 36) == "sequential"
    with pytest.raises(RuntimeError, match="insufficient"):
        ab_lambda.choose_mode(40_000_000_000, 36)


def test_experiment_lock_prevents_two_managers_writing_the_same_outputs(tmp_path):
    with ab_lambda.experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another A/B"):
            with ab_lambda.experiment_lock(tmp_path):
                pytest.fail("concurrent experiment lock admitted")


def test_snapshot_only_never_queries_gpu_or_main_unit_and_survives_source_pruning(
    tmp_path, monkeypatch
):
    source = tmp_path / "full/step-00004516"
    source.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "trainer.pt"):
        (source / name).write_bytes(name.encode())
    output = tmp_path / "ab"

    def unexpected(*_):
        raise AssertionError("snapshot-only queried GPU or main unit")

    monkeypatch.setattr(ab_lambda, "require_main_inactive", unexpected)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", unexpected)
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ab", "--checkpoint", str(source), "--output", str(output), "--snapshot-only"],
    )
    ab_lambda.main()
    import shutil

    shutil.rmtree(source)
    assert ab_lambda.snapshot_parent(source, output) == output / "parent"
    ab_lambda.main()  # CLI selectors also honor an existing pin after pruning.


def cpu_experiment(tmp_path, anchors_per_sequence=2):
    base, generations = fixture(tmp_path)
    root = tmp_path / "rolling"
    hot = tmp_path / "hot.json"
    hot.write_text(json.dumps(list(range(25))))
    base.update(
        append_open=True,
        anchors_per_sequence=anchors_per_sequence,
        total_steps=1_000_000,
        schedule_response_tokens=200,
        eval_every=1,
        eval_before_training=True,
        eval_at_end=True,
        eval_sequences=1,
        eval_mode="rounds",
        eval_packed_chunk_size=2,
        eval_require_decode_probs=True,
        eval_report_first_anchor=True,
        selection_blocks=[4],
        eval_greedy_features=str(tmp_path / "greedy"),
        proposal_vocabulary={
            "path": str(hot),
            "quantization": "row_int8",
            "loss": "restricted",
        },
    )
    _, plan = plan_for(base, generations, root)
    fake_capture(plan, plan["validation"], plan["validation_root"], root, 120)
    for path in (Path(plan["validation_root"]) / "shards").glob("*.pt"):
        row = torch.load(path, weights_only=True)
        row["target_distribution_source"] = "actual_decode_raw_top_logprobs_joint"
        torch.save(row, path)
    row = next(records(Path(plan["validation"]["input"]).glob("*.jsonl")))
    row["sampling"] = {**row["sampling"], "temperature": 0.0}
    FeatureStore(tmp_path / "greedy", min_free_bytes=0).write(
        row,
        torch.randn(13, 2, 16),
        torch.arange(20).expand(13, -1),
        torch.ones(13, 20) / 20,
        layer_ids=(1, 3),
    )
    fake_capture(plan, plan["windows"][0], plan["train_root"], root, 120)
    command = [
        sys.executable,
        "-m",
        "dspine.train",
        "--config",
        str(root / "run.json"),
        "--data-window",
        plan["windows"][0]["name"],
        "--stop-at-step",
        "3",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    parent = Path(plan["run"]["output"]) / "step-00000003"
    purge_window(plan["train_root"], plan["windows"][0])
    fake_capture(plan, plan["windows"][1], plan["train_root"], root, 120)
    atomic_json(root / "state.json", {"window": 1, "phase": "train", "complete": False})
    return root, plan, parent, generations


def test_cpu_ab_and_exact_B_promotion_leave_original_controller_unchanged(
    tmp_path, monkeypatch
):
    root, plan, parent, generations = cpu_experiment(tmp_path)
    protected = {
        name: (root / name).read_bytes()
        for name in ("plan.json", "run.json", "state.json")
    }
    parent_hash = ab_lambda.file_digest(parent / "trainer.pt")
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
    monkeypatch.setattr(sys, "argv", argv)
    ab_lambda.main()
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["complete"] and comparison["paired_steps_verified"] == 2
    assert [r["step"] for r in comparison["branches"]["B"]["evaluations"]] == [3, 4, 5]
    states = {
        b: torch.load(output / b / "step-00000005/trainer.pt", weights_only=False)
        for b in ("A", "B")
    }
    assert (
        states["A"]["response_tokens_seen"] == states["B"]["response_tokens_seen"] == 40
    )
    assert states["A"]["fork_provenance"]["parent_step"] == 3
    for name in ("A", "B"):
        assert states[name]["data_window"] == "window-00001"
    # A completed job rerun is idempotent: it does not train extra updates.
    ab_lambda.main()
    changed = {**plan, "run": {**plan["run"], "lr": 1.0}}
    with pytest.raises(ValueError, match="source config/window"):
        ab_lambda.promote_plan(changed, output, tmp_path / "invalid-promotion")
    assert not (tmp_path / "invalid-promotion").exists()
    promoted = tmp_path / "rolling-B"
    monkeypatch.setattr(sys, "argv", argv + ["--promote-root", str(promoted)])
    ab_lambda.main()
    new_source = json.loads((promoted / "source.json").read_text())
    fork = prepare_plan(new_source, generations, promoted)
    assert fork["windows"] == plan["windows"]
    state = execute_plan(fork, promoted, max_actions=1)
    assert state["phase"] == "rotate"
    checkpoint = latest_state(new_source["output"])
    assert checkpoint["step"] == 6
    saved = torch.load(Path(checkpoint["path"]) / "trainer.pt", weights_only=False)
    assert saved["response_tokens_seen"] == 48
    assert saved["run_config"] == new_source
    assert parent_hash == ab_lambda.file_digest(parent / "trainer.pt")
    assert all(
        (root / name).read_bytes() == content for name, content in protected.items()
    )


def test_incomplete_capture_exits_before_querying_gpu(tmp_path, monkeypatch):
    root, plan, parent, _ = cpu_experiment(tmp_path)
    path = next((Path(plan["train_root"]) / "shards").glob("*.pt"))
    path.unlink()
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)

    def unexpected():
        raise AssertionError("incomplete capture queried GPU")

    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", unexpected)
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
            str(tmp_path / "ab"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        ab_lambda.main()
    assert error.value.code == 75
    assert not (tmp_path / "ab/A").exists()
