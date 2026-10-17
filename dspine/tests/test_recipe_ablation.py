"""Round-two overrides preserve token schedules, cursors and sequence budgets."""

import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dspine import ab_lambda
from dspine.data import PackedSequenceStream
from dspine.forks import validate_fork, validate_recipe_values
from dspine.losses import lr_multiplier
from dspine.recipe_ablation import (
    BASE_RECIPES,
    arm_goals,
    read_spec,
    recipe_configs,
    rescale_windows,
    summarize_recipe,
    validate_parent,
)
from dspine.rolling import atomic_json, execute_plan, latest_state, prepare_plan
from dspine.tests.test_ab_lambda import cpu_experiment


@pytest.mark.parametrize("base", ["A", "B", "C"])
def test_explicit_base_recipe_replaces_parent_recipe_and_preserves_schedule(
    tmp_path, base
):
    parent = dict(
        output=str(tmp_path / "parent-run"),
        lr=6e-4,
        gradient_accumulation=4,
        schedule_response_tokens=10000,
        total_steps=1000000,
        **BASE_RECIPES["C"],
    )
    spec = dict(
        base_recipe=base,
        arms={
            "control": {},
            "low": {"lr": 3e-4},
            "big": {"gradient_accumulation": 16},
            "off": {"curriculum_p": 0},
        },
    )
    frozen = copy.deepcopy(parent)
    configs = recipe_configs(parent, spec, tmp_path / "experiment", 5516)
    for config in configs.values():
        for key, value in BASE_RECIPES[base].items():
            assert config[key] == value
        assert config["schedule_response_tokens"] == parent["schedule_response_tokens"]
        assert config["total_steps"] == parent["total_steps"]
        assert config["eval_every"] == 250 and config["eval_step_offset"] == 5516
    assert parent == frozen
    goals = arm_goals(configs, "control", 5516, 1000, equal_samples=True)
    assert goals == dict(control=6516, low=6516, big=5766, off=6516)
    assert all(
        goal == 6516 for goal in arm_goals(configs, "control", 5516, 1000).values()
    )
    with pytest.raises(ValueError, match="divide exactly"):
        arm_goals(configs, "control", 5516, 1001, equal_samples=True)
    for progress in (0.1, 0.4, 0.8):
        old = parent["lr"] * lr_multiplier(
            progress * parent["total_steps"], parent["total_steps"]
        )
        low = configs["low"]["lr"] * lr_multiplier(
            progress * parent["total_steps"], parent["total_steps"]
        )
        assert low == pytest.approx(old * 0.5)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lr", 0),
        ("lr", float("nan")),
        ("lr", True),
        ("gradient_accumulation", 0),
        ("gradient_accumulation", 1.5),
        ("curriculum_p", -0.1),
        ("curriculum_p", 1.1),
        ("curriculum_p", True),
    ],
)
def test_invalid_overrides_fail_before_cuda(key, value):
    with pytest.raises(ValueError):
        validate_recipe_values({key: value})


def test_recipe_rejects_path_names_unlisted_overrides_and_unknown_reference(tmp_path):
    path = tmp_path / "recipe.json"
    for spec in (
        {"arms": {"../escape": {}}},
        {"arms": {"control": {"seed": 13}}},
        {"arms": {"control": {}}, "reference_arm": "missing"},
        {"arms": {"control": {}}, "unknown": True},
    ):
        path.write_text(json.dumps(spec))
        with pytest.raises(ValueError):
            read_spec(path)
    path.write_text(json.dumps({"arms": {"control": {}, "low": {"lr": 0.0003}}}))
    assert read_spec(path, base_recipe="B")["base_recipe"] == "B"


def test_mid_window_parent_requires_exact_cursor_and_immutable_data():
    plan = dict(
        run=dict(gradient_accumulation=4, schedule_response_tokens=10000, seed=42)
    )
    window = dict(start_step=4516, stop_step=9032, name="window-00001")
    saved = dict(
        run_config={**plan["run"], "lr": 0.0003}, step=5516, data_window=window["name"]
    )
    assert validate_parent(plan, window, saved) is False
    with pytest.raises(ValueError, match="cursor"):
        validate_parent(plan, window, {**saved, "data_window": "window-00000"})
    with pytest.raises(ValueError, match="data/model/schedule"):
        validate_parent(
            plan, window, {**saved, "run_config": {**saved["run_config"], "seed": 3}}
        )
    with pytest.raises(ValueError, match="promoted plan"):
        validate_parent(
            plan,
            window,
            {
                **saved,
                "run_config": {**saved["run_config"], "gradient_accumulation": 16},
            },
        )


def test_promotion_rescales_remaining_samples_not_token_schedule():
    plan = dict(
        run={"gradient_accumulation": 4},
        windows=[
            dict(start_step=0, stop_step=4516),
            dict(start_step=4516, stop_step=9032),
            dict(start_step=9032, stop_step=13032),
        ],
    )
    exp = dict(
        start_step=5516,
        window_index=1,
        arm_definitions={"big": dict(goal_step=5766, gradient_accumulation=16)},
    )
    windows = rescale_windows(plan, exp, "big")
    assert windows[0] == plan["windows"][0]
    assert windows[1]["stop_step"] == 6395
    assert windows[2] == dict(start_step=6395, stop_step=7395)
    assert plan["windows"][1]["stop_step"] == 9032


@pytest.mark.filterwarnings(
    "ignore:'dspine.ab_lambda' found in sys.modules:RuntimeWarning"
)
def test_runpy_entry_uses_canonical_manager_for_signals_and_children(monkeypatch):
    called = []
    monkeypatch.setattr(ab_lambda, "main", lambda: called.append(True))
    runpy.run_module("dspine.ab_lambda", run_name="__main__")
    assert called == [True]


def test_cpu_round2_from_midwindow_equal_samples_and_accum_promotion(
    tmp_path, monkeypatch
):
    root, plan, parent, generations = cpu_experiment(tmp_path, anchors_per_sequence=12)
    # Give the tiny local fixture enough remaining sample budget for round two.
    plan["windows"][1]["stop_step"] = 20
    atomic_json(root / "plan.json", plan)
    protected = {
        name: (root / name).read_bytes()
        for name in ("plan.json", "run.json", "state.json")
    }
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", lambda: 200_000_000_000)
    monkeypatch.setattr(
        ab_lambda.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    first = tmp_path / "round1"
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
            str(first),
            "--steps",
            "1",
            "--arms",
            "B",
            "--max-minutes",
            "5",
        ],
    )
    ab_lambda.main()
    winner = first / "B/step-00000004"
    before = torch.load(winner / "trainer.pt", weights_only=False)
    parent_hash = ab_lambda.file_digest(winner / "trainer.pt")
    expected = PackedSequenceStream(
        plan["run"]["features"],
        anchors_per_sequence=12,
        block_size=5,
        anchor_sampling="without_replacement",
    )
    expected.load_state_dict(before["data"])
    next_row = next(expected)
    expected_key = json.dumps([next_row["id"], next_row["sequence_start"]]).encode()
    expected_data = hashlib.sha256(
        expected_key
        + next_row["anchor_offsets"][next_row["anchor_valid"]].numpy().tobytes()
    ).hexdigest()
    spec_path = tmp_path / "recipe.json"
    spec_path.write_text(
        json.dumps(
            dict(
                base_recipe="C",
                reference_arm="control",
                arms={
                    "control": {},
                    "low": {"lr": 0.0003},
                    "high": {"lr": 0.0012},
                    "big": {"gradient_accumulation": 4},
                    "off": {"curriculum_p": 0.0},
                },
            )
        )
    )
    output = tmp_path / "round2"
    argv = [
        "ab",
        "--plan",
        str(root / "plan.json"),
        "--checkpoint",
        str(winner),
        "--output",
        str(output),
        "--recipe-file",
        str(spec_path),
        "--steps",
        "4",
        "--equal-samples",
        "--eval-every",
        "2",
        "--mode",
        "parallel",
        "--max-minutes",
        "5",
    ]
    # Exercise staged launches using one shared immutable experiment definition.
    monkeypatch.setattr(sys, "argv", argv + ["--arms", "control", "low", "high"])
    ab_lambda.main()
    monkeypatch.setattr(sys, "argv", argv + ["--arms", "big", "off"])
    ab_lambda.main()
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["reset_data"] is False
    comparison = summarize_recipe(output, experiment)
    assert comparison["complete"]
    assert all(
        row["full_budget_means"]["ce"] > 0 for row in comparison["branches"].values()
    )
    assert all(count == 4 for count in comparison["paired_samples"].values())
    for arm in experiment["arm_definitions"]:
        metrics = [
            json.loads(line)
            for line in (output / arm / "metrics.jsonl").read_text().splitlines()
        ]
        training = [r for r in metrics if "sample_traces" in r]
        assert training[0]["sample_traces"][0]["data"] == expected_data
        assert sum(len(r["sample_traces"]) for r in training) == 4
        assert (
            training[-1]["response_tokens_seen"] == before["response_tokens_seen"] + 32
        )
        config = json.loads((output / arm / "run.json").read_text())
        for row in training:
            assert row["lr"] == pytest.approx(
                config.get("lr", 0.0006)
                * lr_multiplier(
                    row["schedule_progress"] * config["total_steps"],
                    config["total_steps"],
                )
            )
            if arm == "off":
                assert row["curriculum_p"] == 0
                assert all(t["curriculum_p"] == 0 for t in row["sample_traces"])
    assert comparison["branches"]["big"]["last_step"] == 5
    assert comparison["branches"]["control"]["last_step"] == 8
    assert [r["step"] for r in comparison["branches"]["control"]["evaluations"]] == [
        4,
        6,
        8,
    ]
    assert comparison["branches"]["big"]["evaluations"][-1]["reference_updates"] == 4
    ab_lambda.main()  # Completed rerun is idempotent, including the shorter arm.
    promoted = tmp_path / "rolling-big"
    ab_lambda.promote_plan(plan, output, promoted, arm="big")
    config = json.loads((promoted / "source.json").read_text())
    new_plan = prepare_plan(config, generations, promoted)
    assert new_plan["windows"][1]["stop_step"] == 8
    assert (
        new_plan["run"]["schedule_response_tokens"]
        == plan["run"]["schedule_response_tokens"]
    )
    # A real continuation restores the selected arm's moments/RNG/exact cursor.
    execute_plan(new_plan, promoted, max_actions=1)
    final = latest_state(config["output"])
    assert final["step"] == 8
    resumed = torch.load(Path(final["path"]) / "trainer.pt", weights_only=False)
    assert resumed["run_config"] == config
    assert resumed["response_tokens_seen"] == before["response_tokens_seen"] + 128
    assert parent_hash == ab_lambda.file_digest(winner / "trainer.pt")
    assert all((root / name).read_bytes() == data for name, data in protected.items())
    # Ordinary resume is still strict, even for a flag allowed in explicit forks.
    changed = {**config, "lr": 0.123}
    validate_fork(config, {**changed, "output": str(tmp_path / "new-fork")}, winner)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(changed))
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dspine.train",
            "--config",
            str(wrong),
            "--resume",
            "latest",
            "--data-window",
            "window-00001",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "identical run config" in result.stderr
