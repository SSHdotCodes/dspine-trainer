"""Fixed query shapes, zero-weight padding, and independent arm failures."""

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dspine import ab_lambda
from dspine.data import FeatureStore, PackedSequenceStream, dequantize_features
from dspine.losses import training_loss
from dspine.tests.test_ab_lambda import cpu_experiment
from dspine.tests.test_reference import tiny


def write_lengths(root, length=25):
    store = FeatureStore(root, min_free_bytes=0)
    for real_positions, prompt in ((1, length - 2), (4, length - 5), (length - 6, 5)):
        row = dict(
            id=f"positions-{real_positions}",
            domain="math",
            split="train",
            prompt_ids=[i % 29 for i in range(prompt)],
            output_ids=[i % 29 for i in range(prompt, length)],
            sampling=dict(temperature=1.0, top_k=20, top_p=0.95),
        )
        ids = (torch.arange(20)[None] + torch.arange(length)[:, None]) % 29
        store.write(
            row,
            torch.randn(length, 2, 16),
            ids,
            torch.ones(length, 20) / 20,
            layer_ids=(1, 3),
            encoding="bf16",
        )


def padded_rows(root, capacity=8):
    stream = PackedSequenceStream(
        root,
        anchors_per_sequence=capacity,
        block_size=5,
        anchor_sampling="without_replacement",
    )
    return [next(stream) for _ in range(3)]


def corrected_loss(model, output, row):
    return training_loss(
        model,
        output,
        row["labels"],
        row["target_ids"],
        row["target_probs"],
        10,
        100,
        row["valid"],
        chunk_size=7,
        lambda_ce_reduction="weighted_token_mean",
        refinement_start_position=2,
        mask_unsupported_ce=True,
    )


def test_256_anchor_slots_for_one_four_and_many_real_positions(tmp_path):
    write_lengths(tmp_path, length=320)
    counts = set()
    for row in padded_rows(tmp_path, capacity=256):
        mask = row["anchor_valid"]
        counts.add(int(mask.sum()))
        assert row["anchor_offsets"].shape == (256,)
        assert row["labels"].shape == row["valid"].shape == (256, 5)
        assert mask.equal(row["valid"][:, 0])
        assert not row["valid"][~mask].any()
        real = row["anchor_offsets"][mask]
        assert real.unique().numel() == real.numel()
        if not mask.all():
            assert (row["anchor_offsets"][~mask] == real[-1]).all()
    assert counts == {1, 4, 256}


@pytest.mark.parametrize("checkpointing", [False, True])
def test_padded_forward_losses_and_all_parameter_gradients_equal_unpadded(
    tmp_path, checkpointing
):
    write_lengths(tmp_path)
    model = tiny()
    model.config.gradient_checkpointing = checkpointing
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.05)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    query_shapes = []
    original = model.layers[0].self_attn.forward_packed

    def track(x, *args, **kwargs):
        query_shapes.append((x.shape[0], x.shape[1]))
        return original(x, *args, **kwargs)

    model.layers[0].self_attn.forward_packed = track
    for row in padded_rows(tmp_path):
        context = dequantize_features(row["context_payload"], dtype=torch.float32)
        mask = row["anchor_valid"]
        teacher = (torch.arange(8) % 2 == 0) & mask
        padded = model.forward_packed(
            context,
            row["sequence_ids"],
            row["anchor_offsets"],
            row["sequence_start"],
            correct_ids=row["labels"],
            teacher_blocks=teacher,
        )
        real = {
            key: value[mask]
            if isinstance(value, torch.Tensor) and value.shape[:1] == (8,)
            else value
            for key, value in row.items()
        }
        unpadded = model.forward_packed(
            context,
            row["sequence_ids"],
            real["anchor_offsets"],
            row["sequence_start"],
            correct_ids=real["labels"],
            teacher_blocks=teacher[mask],
        )
        torch.testing.assert_close(
            padded.hidden[mask], unpadded.hidden, atol=2e-6, rtol=2e-5
        )
        a, metrics_a = corrected_loss(model, padded, row)
        b, metrics_b = corrected_loss(model, unpadded, real)
        for key in metrics_a:
            torch.testing.assert_close(
                metrics_a[key], metrics_b[key], atol=2e-6, rtol=2e-5
            )
        ga = torch.autograd.grad(a, parameters, allow_unused=True)
        gb = torch.autograd.grad(b, parameters, allow_unused=True)
        for left, right in zip(ga, gb):
            if left is None or right is None:
                assert left is right
            else:
                torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-4)
    assert (8, 5) in query_shapes and (1, 5) in query_shapes and (4, 5) in query_shapes


def test_cpu_aot_backward_has_constant_query_shape_as_real_anchor_count_changes(
    tmp_path,
):
    """Exercise the old 1/4/8-anchor path through CPU AOTAutograd, padded to8.

    This cannot reproduce Triton's CUDA LoweringException. It verifies the data
    shape contract and compiled backward on the equivalent CPU attention path.
    """
    write_lengths(tmp_path)
    model = tiny()
    from torch._dynamo.backends.registry import lookup_backend

    graphs = []
    aot_eager = lookup_backend("aot_eager")

    def backend(graph, inputs):
        graphs.append(graph)
        return aot_eager(graph, inputs)

    compiled = torch.compile(
        model.forward_packed, backend=backend, dynamic=False, fullgraph=True
    )
    counts, shapes = set(), set()
    for row in padded_rows(tmp_path):
        model.zero_grad(set_to_none=True)
        counts.add(int(row["anchor_valid"].sum()))
        output = compiled(
            dequantize_features(row["context_payload"], dtype=torch.float32),
            row["sequence_ids"],
            row["anchor_offsets"],
            row["sequence_start"],
        )
        shapes.add(tuple(output.hidden.shape))
        loss, _ = corrected_loss(model, output, row)
        loss.backward()
        assert torch.isfinite(loss) and model.fc.weight.grad.isfinite().all()
    assert counts == {1, 4, 8}
    assert shapes == {(8, 5, 16)}
    assert len(graphs) == 1


def test_process_failure_does_not_signal_healthy_peer(tmp_path, monkeypatch):
    for arm in ("bad", "healthy"):
        (tmp_path / arm).mkdir()
    configs = {arm: {"output": str(tmp_path / arm)} for arm in ("bad", "healthy")}
    original = subprocess.Popen
    worker = """
import json, pathlib, signal, sys, time
root, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
if arm == 'bad':
    print('simulated FlexAttention LoweringException', flush=True)
    sys.exit(7)
def stopped(*_):
    (root / arm / 'unexpected-signal').write_text('stopped')
    sys.exit(99)
signal.signal(signal.SIGTERM, stopped)
for _ in range(300):
    path = root / 'bad/status.json'
    if path.exists() and json.loads(path.read_text())['state'] == 'failed':
        (root / arm / 'goal-reached').write_text('yes')
        sys.exit(0)
    time.sleep(.02)
sys.exit(98)
"""

    def launch(command, **kwargs):
        arm = Path(command[command.index("--config") + 1]).parent.name
        return original([sys.executable, "-c", worker, str(tmp_path), arm], **kwargs)

    monkeypatch.setattr(ab_lambda.subprocess, "Popen", launch)
    monkeypatch.setattr(ab_lambda, "latest_state", lambda *_: None)
    summary = lambda: {"branches": {arm: {"evaluations": []} for arm in configs}}
    failed = ab_lambda.train_group(
        tuple(configs),
        configs,
        tmp_path,
        tmp_path / "parent",
        {"name": "window", "start_step": 0},
        7,
        time.monotonic() + 180,
        summarizer=summary,
    )
    assert failed == ["bad"]
    assert (tmp_path / "healthy/goal-reached").exists()
    assert not (tmp_path / "healthy/unexpected-signal").exists()
    result = ab_lambda.attach_arm_statuses(summary(), tmp_path)
    assert result["failed_arms"]["bad"]["exit_code"] == 7
    assert result["branches"]["healthy"]["process"]["state"] == "succeeded"


@pytest.mark.parametrize("failure", ["exit", "launch"])
def test_legacy_manager_reports_failure_after_healthy_arms_reach_goal(
    tmp_path, monkeypatch, failure
):
    root, plan, parent, _ = cpu_experiment(tmp_path)
    output = tmp_path / "abc"
    monkeypatch.setattr(ab_lambda, "require_main_inactive", lambda *_: None)
    monkeypatch.setattr(ab_lambda, "free_gpu_bytes", lambda: 200_000_000_000)
    monkeypatch.setattr(
        ab_lambda.shutil, "disk_usage", lambda *_: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(ab_lambda.signal, "signal", lambda *_: None)
    original = subprocess.Popen

    def launch(command, **kwargs):
        if (
            "--config" in command
            and Path(command[command.index("--config") + 1]).parent.name == "C"
        ):
            if failure == "launch":
                raise OSError("simulated per-arm startup error")
            return original([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)
        return original(command, **kwargs)

    monkeypatch.setattr(ab_lambda.subprocess, "Popen", launch)
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
            str(output),
            "--arms",
            "A",
            "B",
            "C",
            "--steps",
            "2",
            "--mode",
            "parallel",
            "--max-minutes",
            "5",
        ],
    )
    with pytest.raises(SystemExit) as error:
        ab_lambda.main()
    assert error.value.code == 1
    report = json.loads((output / "comparison.json").read_text())
    assert set(report["failed_arms"]) == {"C"}
    for arm in ("A", "B"):
        assert report["branches"][arm]["last_step"] == 5
        assert report["branches"][arm]["evaluations"][-1]["step"] == 5
        assert report["branches"][arm]["process"]["exit_code"] == 0


def test_failed_sequential_group_does_not_skip_later_healthy_groups(
    tmp_path, monkeypatch
):
    calls = []

    def group(branches, *_args, **_kwargs):
        calls.append(branches)
        return ["bad"] if branches == ("bad",) else []

    monkeypatch.setattr(ab_lambda, "train_group", group)
    assert ab_lambda.run_groups(
        [("bad",), ("healthy",)], {}, tmp_path, tmp_path, {}, 9, time.monotonic() + 180
    ) == ["bad"]
    assert calls == [("bad",), ("healthy",)]
