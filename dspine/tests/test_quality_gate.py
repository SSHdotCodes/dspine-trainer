import pytest

from dspine.quality_gate import analyze


def row(step, body, opening):
    return {
        "step": step,
        "offline_tau": {"a": {str(b): {"tau": body, "anchors": 1000} for b in (6, 8)}},
        "offline_first_anchor": {
            "a": {str(b): {"tau": opening, "anchors": 18} for b in (6, 8)}
        },
    }


def test_opening_gain_cannot_hide_stalled_complete_rounds():
    rows = [row(0, 1.12, 1.91)] + [
        row(s, 1.13, 3.0) for s in (500, 1000, 1500, 2000, 2500, 3000)
    ]
    assert analyze(rows)["decision"] == "no_go"
    assert analyze(rows)["blocks"]["8"]["complete_gain"] == pytest.approx(0.01)


def test_grace_period_and_paired_progress_require_final_review():
    assert analyze([row(0, 1.12, 1.91), row(500, 1.12, 3.0)])["decision"] == "pending"
    rows = [row(0, 1.12, 1.91)] + [
        row(s, 1.5 + s / 10000, 2.5) for s in (500, 1000, 1500, 2000, 2500, 3000)
    ]
    assert analyze(rows)["decision"] == "progressing"
    assert analyze(rows, final=True)["decision"] == "review"


def test_material_opening_regression_stops_even_when_body_improves():
    rows = [row(0, 1.12, 1.91)] + [row(s, 1.5, 1.5) for s in (1000, 2000, 3000)]
    assert analyze(rows)["decision"] == "no_go"


def test_body_regression_is_detected_even_above_initial_baseline():
    rows = [row(0, 1.12, 1.91)] + [
        row(s, b, 2.5)
        for s, b in [
            (500, 2.0),
            (1000, 2.1),
            (1500, 2.2),
            (2000, 1.5),
            (2500, 1.5),
            (3000, 1.5),
        ]
    ]
    assert analyze(rows)["decision"] == "no_go"


@pytest.mark.parametrize("related", [True, False])
def test_stop_guard_checks_exact_trainer_config_before_signalling(
    tmp_path, monkeypatch, related
):
    import os
    import signal
    from pathlib import Path

    import dspine.quality_gate as gate

    output = tmp_path / "output"
    output.mkdir()
    config = tmp_path / "run.json"
    (output / "trainer-process.json").write_text('{"pid":12345}')
    proc = tmp_path / "proc/12345"
    proc.mkdir(parents=True)
    argv = [
        b"python",
        b"-m",
        b"dspine.train",
        b"--config",
        os.fsencode(str(config if related else tmp_path / "other.json")),
    ]
    (proc / "cmdline").write_bytes(b"\0".join(argv))
    actual = Path
    monkeypatch.setattr(
        gate,
        "Path",
        lambda value: actual(tmp_path / "proc")
        if str(value) == "/proc"
        else actual(value),
    )
    calls = []
    monkeypatch.setattr(gate.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    if related:
        assert gate.stop_own_trainer(output, config)
        assert calls == [(12345, signal.SIGTERM)]
    else:
        with pytest.raises(ValueError, match="unrelated"):
            gate.stop_own_trainer(output, config)
        assert not calls
