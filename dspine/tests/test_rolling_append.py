"""Append-only manifests preserve exact training continuation and schedules."""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from dspine.data import records, split_for_id
from dspine.rolling import execute_plan, latest_state
from dspine.rolling_append import append_windows
from dspine.tests.test_rolling import fake_capture, fixture, plan_for


def incoming(tmp_path, original):
    destination = tmp_path / "incoming"
    destination.mkdir()
    template = next(records(original.glob("*.jsonl")))
    names = [
        f"append-{i}" for i in range(20) if split_for_id(f"append-{i}") == "train"
    ][:2]
    rows = [{**template, "id": name} for name in names]
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    summary = dict(
        errors=[],
        validated_template=True,
        duplicate_records=0,
        counts={"total": {"records": 2, "tokens": 16}},
    )
    return destination, summary


def test_append_before_or_after_prefix_training_is_identical(tmp_path):
    models = []
    tokens = []
    for variant in ("early", "late"):
        directory = tmp_path / variant
        directory.mkdir()
        base, source = fixture(directory)
        base.update(append_open=True, total_steps=100, schedule_response_tokens=1000)
        root = directory / "plan"
        _, plan = plan_for(base, source, root)
        original = copy.deepcopy(plan["windows"])
        run_bytes = (root / "run.json").read_bytes()
        if variant == "late":
            state = execute_plan(plan, root, capture=fake_capture)
            assert state["phase"] == "awaiting_data" and not state["complete"]
        added, summary = incoming(directory, source)
        appended = append_windows(root, added, summary, seal=True)
        assert appended["windows"][: len(original)] == original
        assert (
            appended["run"] == plan["run"]
            and (root / "run.json").read_bytes() == run_bytes
        )
        assert appended["windows"][2]["start_step"] == 6
        assert appended["windows"][2]["stop_step"] == 9
        # The running controller refreshes an old in-memory plan safely.
        state = execute_plan(plan, root, capture=fake_capture)
        assert state["complete"] and state["window"] == 3
        checkpoint = latest_state(plan["run"]["output"])
        saved = torch.load(Path(checkpoint["path"]) / "trainer.pt", weights_only=False)
        tokens.append(saved["response_tokens_seen"])
        models.append(load_file(str(Path(checkpoint["path"]) / "model.safetensors")))
    assert tokens == [72, 72]
    for key in models[0]:
        torch.testing.assert_close(models[0][key], models[1][key], atol=0, rtol=0)


def test_append_rejects_stale_validation_without_publishing(tmp_path):
    base, source = fixture(tmp_path)
    base.update(append_open=True, total_steps=100)
    root = tmp_path / "plan"
    _, plan = plan_for(base, source, root)
    added, summary = incoming(tmp_path, source)
    before = (root / "plan.json").read_bytes()
    summary["counts"]["total"]["records"] = 3
    with pytest.raises(ValueError, match="not covered"):
        append_windows(root, added, summary, seal=True)
    assert (root / "plan.json").read_bytes() == before
    assert json.loads(before)["windows"] == plan["windows"]


def test_append_cannot_change_a_sealed_run(tmp_path):
    base, source = fixture(tmp_path)
    root = tmp_path / "plan"
    plan_for(base, source, root)
    added, summary = incoming(tmp_path, source)
    with pytest.raises(ValueError, match="sealed"):
        append_windows(root, added, summary, seal=True)


def test_validated_append_counts_exclude_aborted_generations(tmp_path):
    base, source = fixture(tmp_path)
    base.update(append_open=True, total_steps=100)
    root = tmp_path / "plan"
    _, plan = plan_for(base, source, root)
    added, summary = incoming(tmp_path, source)
    path = added / "records.jsonl"
    aborted = {**next(records([path])), "id": "aborted", "finish_reason": "abort"}
    with path.open("a") as stream:
        stream.write(json.dumps(aborted) + "\n")
    # The real merge validator omits aborts from records/tokens, retaining them
    # only in counts.excluded.abort_records. It still validates this source.
    summary["counts"]["excluded"] = {"abort_records": 1}
    appended = append_windows(root, added, summary, seal=True)
    assert appended["training_records"] == plan["training_records"] + 2
    assert "aborted" not in {x for w in appended["windows"] for x in w["ids"]}
    assert appended["append_receipts"][-1]["source_counts"]["total"] == {
        "records": 2,
        "tokens": 16,
    }


def test_initial_freeze_defers_campaign2_and_append_validates_its_platform(tmp_path):
    import zstandard

    base, source = fixture(tmp_path)
    original = (source / "records.jsonl").read_bytes()
    incoming_root, summary = incoming(tmp_path, source)
    new = (incoming_root / "records.jsonl").read_bytes()
    for platform, payload in [("vega", original), ("modal-c2-example", new)]:
        part = source / platform / "shard-00000" / "part-00000.jsonl.zst"
        part.parent.mkdir(parents=True)
        part.write_bytes(zstandard.ZstdCompressor().compress(payload))
    base.update(
        append_open=True, total_steps=100, deferred_platform_prefixes=["modal-c2-"]
    )
    root = tmp_path / "plan"
    _, plan = plan_for(base, source, root)
    assert plan["training_records"] == 4
    summary["counts"] = {"platform:modal-c2-example": summary["counts"]["total"]}
    appended = append_windows(root, source, summary, "modal-c2-", seal=True)
    assert appended["training_records"] == 6
    assert appended["append_receipts"][-1]["source_counts"]["modal-c2-example"] == {
        "records": 2,
        "tokens": 16,
    }


def test_append_refuses_a_missing_validated_platform(tmp_path):
    import zstandard

    base, source = fixture(tmp_path)
    added, summary = incoming(tmp_path, source)
    part = added / "modal-c2-present/shard-00000/part-00000.jsonl.zst"
    part.parent.mkdir(parents=True)
    part.write_bytes(
        zstandard.ZstdCompressor().compress((added / "records.jsonl").read_bytes())
    )
    summary["counts"] = {
        "platform:modal-c2-present": {"records": 2, "tokens": 16},
        "platform:modal-c2-not-exported": {"records": 1, "tokens": 8},
    }
    base.update(append_open=True, total_steps=100)
    root = tmp_path / "plan"
    plan_for(base, source, root)
    before = (root / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="not covered"):
        append_windows(root, added, summary, "modal-c2-", seal=True)
    assert (root / "plan.json").read_bytes() == before


def test_capture_yields_to_authorized_training_corunners(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dspine.rolling import capture_window

    monkeypatch.setattr(
        "dspine.rolling.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout="12345\n"),
    )
    assert (
        capture_window({"run": {"device": "cuda"}}, None, None, tmp_path, 120) is None
    )
    assert not (tmp_path / "capture-runs").exists()
