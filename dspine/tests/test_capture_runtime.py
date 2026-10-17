"""Pinned numerical runtime must survive changes to its production source."""

import json
import shlex

import pytest

from dspine import capture_runtime


def test_runtime_snapshot_is_immutable_and_checks_its_hashes(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "ref").mkdir(parents=True)
    model = tmp_path / "qwen4_exp.py"
    model.write_text("original model")
    binary = tmp_path / "qwenfast.so"
    binary.write_bytes(b"original kernel")
    unit = tmp_path / "unit"
    args = [
        "docker",
        "run",
        "-v",
        f"{model}:{capture_runtime.MODEL_DEST}:ro",
        "-v",
        f"{binary}:/opt/qwenfast/qwenfast.so:ro",
        "-e",
        "QWENFAST_W8=head",
        "-e",
        "UNRELATED_SECRET=must-not-copy",
        "image",
    ]
    unit.write_text("ExecStart=" + shlex.join(args))
    monkeypatch.setattr(capture_runtime, "patch", lambda text: text + " patched")
    first = capture_runtime.prepare(project, "production", unit)
    assert "QWENFAST_W8=head" in first
    assert all("SECRET" not in x for x in first)
    model.write_text("changed model")
    binary.write_bytes(b"changed kernel")
    unit.unlink()
    second = capture_runtime.prepare(project, "production", unit)
    assert first == second
    assert (
        project / "ref/qwen4_exp-capture-production.py"
    ).read_text() == "original model patched"
    manifest = json.loads(
        (project / "ref/production-capture-runtime/manifest.json").read_text()
    )
    pinned_binary = next(name for name in manifest["hashes"] if name.endswith(".so"))
    from pathlib import Path

    Path(pinned_binary).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="pinned capture runtime"):
        capture_runtime.prepare(project, "production", unit)
