"""Recipe co-runs never control main; immutable feature views survive rotation."""

import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dspine import ab_lambda
from dspine.data import AnchorStream, FeatureStore, PackedSequenceStream
from dspine.forks import validate_recipe_values
from dspine.recipe_ablation import read_spec, recipe_configs
from dspine.rolling import latest_state
from dspine.tests.test_ab_lambda import cpu_experiment
from dspine.tests.test_data import make_record

ROUND4 = Path(__file__).resolve().parents[1] / "configs/round4-recipe.json"


@pytest.mark.parametrize(
    "key,value",
    [
        ("context_fusion", "bad"),
        ("context_fusion", None),
        ("context_fusion", True),
        ("pos1_weight", -1),
        ("pos1_weight", True),
        ("pos1_weight", float("nan")),
    ],
)
def test_bad_fusion_recipe_values_are_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        validate_recipe_values({key: value})


def test_round4_defaults_and_inherited_schedule(tmp_path):
    parent = dict(
        output=str(tmp_path / "main"),
        lr=3e-4,
        gradient_accumulation=4,
        anchors_per_sequence=256,
        schedule_response_tokens=540000000,
    )
    frozen = copy.deepcopy(parent)
    configs = recipe_configs(
        parent, read_spec(ROUND4), tmp_path / "r4", 7122, cap_gb=21
    )
    assert list(configs) == ["control", "fusion", "fusion_p1"]
    assert "context_fusion" not in configs["control"]
    assert configs["fusion"]["context_fusion"] == "anchor"
    assert configs["fusion_p1"]["pos1_weight"] == 3
    assert all(c["lr"] == 3e-4 and c["eval_every"] == 250 for c in configs.values())
    assert parent == frozen


@pytest.mark.parametrize("stream_type", [AnchorStream, PackedSequenceStream])
def test_hard_link_view_keeps_logical_cursor_and_rng_after_source_purge(
    tmp_path, stream_type
):
    source = tmp_path / "source"
    record = make_record()
    store = FeatureStore(source, min_free_bytes=0)
    path = store.write(
        record,
        torch.randn(25, 2, 8),
        torch.arange(20).expand(25, -1),
        torch.ones(25, 20) / 20,
        layer_ids=(1, 3),
    )
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    # FeatureStore is already a shard root here; plan train_root uses /shards.
    plan = {"train_root": str(tmp_path), "run": {}}
    (tmp_path / "shards").symlink_to(source, target_is_directory=True)
    window = {"name": "window-1", "ids": [record["id"]]}
    data_read_root = ab_lambda.pin_feature_window(plan, window, experiment)
    before = path.read_bytes()
    assert (data_read_root / path.name).stat().st_ino == path.stat().st_ino
    kwargs = dict(
        anchors_per_sequence=3,
        block_size=5,
        context_window=4,
        anchor_sampling="without_replacement",
    )
    control = stream_type(source, **kwargs)
    next(control)
    saved = copy.deepcopy(control.state_dict())
    expected = [next(control) for _ in range(8)]  # Includes epoch boundaries.
    path.unlink()  # The foreign controller can rotate; pinned files remain.
    pinned = stream_type(source, read_root=data_read_root, **kwargs)
    pinned.load_state_dict(saved)
    actual = [next(pinned) for _ in range(8)]
    for a, b in zip(actual, expected):
        assert a["id"] == b["id"]
        assert torch.equal(a["labels"], b["labels"])
        if stream_type is PackedSequenceStream:
            assert torch.equal(a["anchor_offsets"], b["anchor_offsets"])
        else:
            assert a["anchor_position"] == b["anchor_position"]
    assert control.state_dict() == pinned.state_dict()
    assert all(Path(p).parent == source for p in pinned.state_dict()["paths"])
    assert ab_lambda.pin_feature_window(plan, window, experiment) == data_read_root
    assert (data_read_root / path.name).read_bytes() == before


def test_active_main_opt_in_is_recipe_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ab", "--allow-active-main"])
    with pytest.raises(SystemExit) as e:
        ab_lambda.main()
    assert e.value.code == 2


def test_live_latest_pin_retries_retention_and_records_actual_parent(
    tmp_path, monkeypatch
):
    output = tmp_path / "main"
    for step in (1, 2):
        checkpoint = output / f"step-{step:08d}"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "trainer.pt"):
            (checkpoint / name).write_bytes(f"{step}{name}".encode())
    first, second = output / "step-00000001", output / "step-00000002"
    (output / "latest").write_text(first.name + "\n")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    original = ab_lambda.snapshot_parent

    def prune_during_copy(checkpoint, root):
        if checkpoint == first:
            shutil.rmtree(first)
            (output / "latest").write_text(second.name + "\n")
        return original(checkpoint, root)

    monkeypatch.setattr(ab_lambda, "snapshot_parent", prune_during_copy)
    parent, actual = ab_lambda.snapshot_live_parent(
        first, experiment, latest_output=output
    )
    assert actual == second
    assert json.loads((parent / "origin.json").read_text())["path"] == str(second)
    with pytest.raises(FileNotFoundError):
        ab_lambda.snapshot_live_parent(first, tmp_path / "explicit")


def test_corun_phase_guard_does_not_control_foreign_unit(tmp_path):
    plan = tmp_path / "plan.json"
    state = tmp_path / "state.json"
    for phase in ("train", "rotate", "capture"):
        state.write_text(json.dumps({"phase": phase}))
        assert ab_lambda.source_training_phase(plan) == (phase == "train")


def test_round4_coruns_bounded_groups_exact_resume_and_promotion(tmp_path, monkeypatch):
    main_root, plan, parent, _ = cpu_experiment(tmp_path, anchors_per_sequence=12)
    main_files = [main_root / name for name in ("plan.json", "run.json", "state.json")]
    main_files += [
        parent / name for name in ("model.safetensors", "trainer.pt", "config.json")
    ]
    source_shards = list((Path(plan["train_root"]) / "shards").glob("*.pt"))
    hashes = {path: ab_lambda.file_digest(path) for path in main_files + source_shards}

    def foreign_active(*_):
        raise AssertionError("co-run queried/controlled the foreign main unit")

    monkeypatch.setattr(ab_lambda, "require_main_inactive", foreign_active)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", lambda: 50e9)  # Fits2caps, not3.
    monkeypatch.setattr(
        ab_lambda.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    output = tmp_path / "recipe-r4"
    groups = []
    original = ab_lambda.train_group

    def record_group(branches, *args, **kwargs):
        groups.append(tuple(branches))
        return original(branches, *args, **kwargs)

    monkeypatch.setattr(ab_lambda, "train_group", record_group)
    argv = [
        "ab",
        "--plan",
        str(main_root / "plan.json"),
        "--checkpoint",
        str(parent),
        "--output",
        str(output),
        "--recipe-file",
        str(ROUND4),
        "--steps",
        "1",
        "--eval-every",
        "1",
        "--allocator-cap-gb",
        "21",
        "--mode",
        "parallel",
        "--parallel-arms",
        "2",
        "--allow-active-main",
        "--max-minutes",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ab_lambda.main()
    assert groups == [("control", "fusion"), ("fusion_p1",)]
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["complete"] and not comparison["failed_arms"]
    assert set(comparison["paired_samples"].values()) == {1}
    assert all(ab_lambda.file_digest(path) == digest for path, digest in hashes.items())
    for name in ("fusion", "fusion_p1"):
        checkpoint = Path(latest_state(output / name)["path"])
        from safetensors.torch import load_file

        weights = load_file(str(checkpoint / "model.safetensors"))
        assert weights["context_fusion.weight"].abs().sum() > 0
        state = torch.load(checkpoint / "trainer.pt", weights_only=False)
        assert all(
            Path(p).parent == Path(plan["run"]["features"])
            for p in state["data"]["paths"]
        )
        assert state["fork_provenance"]["parent_step"] == 3
    # Idempotent owned retry resumes complete arms, without changing source.
    ab_lambda.main()
    assert all(ab_lambda.file_digest(path) == digest for path, digest in hashes.items())
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    destination = tmp_path / "promoted"
    ab_lambda.promote_plan(plan, output, destination, arm="fusion")
    assert (
        json.loads((destination / "run.json").read_text())["context_fusion"] == "anchor"
    )
