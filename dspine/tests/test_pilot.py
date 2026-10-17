"""Sizing accounts for extraction and measured eval/checkpoint overhead."""

import pytest

from dspine.pilot import size_run


def measurement(layers, seconds, extra=0):
    return {
        "config": {
            "model": {"num_hidden_layers": layers},
            "anchors_per_sequence": 32,
            "packed": True,
            "gradient_accumulation": 4,
        },
        "metrics": [
            {
                "step": i + 1,
                "seconds_per_step": seconds,
                "elapsed_seconds": (i + 1) * (seconds + extra),
                "peak_memory_bytes": 80_000_000_000,
            }
            for i in range(10)
        ],
    }


def test_sizing_subtracts_capture_and_uses_measured_total_step_cost():
    report = size_run(
        {"output": "full"},
        [measurement(6, 10, extra=10)],
        corpus_sequences=100,
        hours=1,
        capture_seconds=1000,
    )
    assert report["chosen"]["seconds_per_step"] == 20
    assert report["epochs"] == pytest.approx(2600 / (500 * 1.1))
    assert report["config"]["epochs"] == report["epochs"]
    assert report["config"]["eval_every"] == 250
    assert report["config"]["checkpoint_every"] == 100
    assert report["capture_seconds"] == 1000


def test_sizing_uses_five_layers_if_six_cannot_cover_minimum_epochs():
    report = size_run(
        {"output": "full"},
        [measurement(6, 300), measurement(5, 10)],
        corpus_sequences=100,
        hours=1,
        capture_seconds=1000,
    )
    assert report["chosen"]["layers"] == 5
    failed = size_run(
        {"output": "full"},
        [measurement(6, 300)],
        corpus_sequences=100,
        hours=1,
        capture_seconds=1000,
    )
    assert failed["status"] == "budget_not_met" and failed["config"] is None


def test_sizing_compares_measured_tau_among_feasible_architectures():
    six, five = measurement(6, 10), measurement(5, 10)
    for item, tau in ((six, 1.8), (five, 2.1)):
        item["eval"] = [{"selection_tau": {str(b): {"tau": tau} for b in (6, 8)}}]
    report = size_run({"output": "full"}, [six, five], corpus_sequences=100, hours=1)
    assert report["chosen"]["layers"] == 5
    assert report["chosen"]["validation_tau"] == 2.1
