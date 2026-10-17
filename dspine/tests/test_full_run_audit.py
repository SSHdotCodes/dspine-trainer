"""CPU coverage of full-run branches absent from the short GPU pilot."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dspine.data import FeatureStore, records
from dspine.rolling import execute_plan, latest_state
from dspine.tests.test_reference import tiny
from dspine.tests.test_rolling import fake_capture, fixture, plan_for


@pytest.mark.parametrize("device,index", [("cuda", 1), ("cuda:2", 2), ("cpu", None)])
def test_memory_cap_resolves_cuda_index_without_gpu(
    tmp_path, monkeypatch, device, index
):
    import dspine.train as trainer

    run = dict(
        model=asdict(tiny().config),
        output=str(tmp_path / "output"),
        device=device,
        cuda_memory_limit_gb=36,
        target="unused",
    )
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config)])
    monkeypatch.setattr(torch, "manual_seed", lambda *_: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: None)
    calls = []

    def current_device():
        assert device == "cuda", (
            "CPU/explicit-index startup queried the current CUDA device"
        )
        return 1

    def properties(selected):
        assert selected == index and isinstance(selected, int)
        return SimpleNamespace(total_memory=80_000_000_000)

    def stop_before_loading(*_):
        raise RuntimeError("CPU audit stops before loading target/model")

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a)
    )
    monkeypatch.setattr(trainer, "load_target_weights", stop_before_loading)
    with pytest.raises(RuntimeError, match="CPU audit stops"):
        trainer.main()
    assert calls == ([] if index is None else [(0.45, index)])


def test_strict_decode_evals_hot_policy_rotation_and_checkpoint_retention(tmp_path):
    base, generations = fixture(tmp_path)
    base["model"]["gradient_checkpointing"] = True
    root = tmp_path / "plan"
    hot_ids = tmp_path / "hot.json"
    hot_ids.write_text(json.dumps(list(range(25))))
    base.update(
        append_open=True,
        total_steps=1_000_000,
        schedule_response_tokens=200,
        proposal_vocabulary=dict(
            path=str(hot_ids), quantization="row_int8", loss="restricted"
        ),
        checkpoint_every=1,
        checkpoint_keep=1,
        eval_every=1,
        eval_before_training=True,
        eval_at_end=True,
        eval_mode="rounds",
        eval_require_decode_probs=True,
        eval_sequences=1,
        eval_packed_chunk_size=2,
        eval_report_first_anchor=True,
        selection_blocks=[4],
        eval_greedy_features=str(tmp_path / "greedy"),
        eval_long_features=str(root / "features/rolling-validation/shards"),
        eval_long_every=1,
        eval_long_sequences=1,
        eval_long_packed_chunk_size=2,
        eval_long_max_seconds=30,
    )
    _, plan = plan_for(base, generations, root)
    row = next(records(Path(plan["validation"]["input"]).glob("*.jsonl")))
    row["sampling"] = {**row["sampling"], "temperature": 0.0}
    FeatureStore(tmp_path / "greedy", min_free_bytes=0).write(
        row,
        torch.randn(13, 2, 16),
        torch.arange(20).expand(13, -1),
        torch.ones(13, 20) / 20,
        layer_ids=(1, 3),
    )

    def capture(*args):
        fake_capture(*args)
        if args[1]["name"] == "validation":
            for path in (Path(args[2]) / "shards").glob("*.pt"):
                payload = torch.load(path, weights_only=True)
                # This synthetic p is uniform on IDs0..19 at every prefix,
                # so the stored trajectory uses that same exact conditional p.
                payload["target_distribution_source"] = (
                    "actual_decode_raw_top_logprobs_joint"
                )
                torch.save(payload, path)

    paused = execute_plan(plan, root, capture=capture, max_actions=3)
    assert paused["phase"] == "rotate" and paused["window"] == 0
    # Resume the controller at the rotation boundary. Its next trainer loads
    # optimizer/RNG/policy and resets ONLY the new window's data cursor.
    final = execute_plan(plan, root, capture=capture)
    assert final["phase"] == "awaiting_data" and final["window"] == 2
    checkpoint = latest_state(plan["run"]["output"])
    state = torch.load(Path(checkpoint["path"]) / "trainer.pt", weights_only=False)
    assert state["response_tokens_seen"] == 48
    assert checkpoint["data_window"] == "window-00001"
    output = Path(plan["run"]["output"])
    assert [p.name for p in output.glob("step-*")] == ["step-00000006"]
    pointer = json.loads((output / "best-b4.json").read_text())
    assert (output / pointer["path"] / "model.safetensors").is_file()
    policy = json.loads((output / pointer["path"] / "proposal_policy.json").read_text())
    assert policy["token_ids"] == list(range(25))
    evaluations = [
        r
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if "offline_tau" in (r := json.loads(line))
    ]
    assert [r["step"] for r in evaluations] == list(range(7))
    assert all(r["offline_long_rounds"]["complete"] for r in evaluations)
    assert all(r["offline_tau_t0"] and r["offline_first_anchor"] for r in evaluations)


def test_optimizer_retention_preserves_both_block_winners(tmp_path):
    from dspine.train import save_best, save_checkpoint

    model = tiny()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    stream = SimpleNamespace(state_dict=lambda: {})
    for step, tau6, tau8 in [(1, 2.0, 3.0), (2, 3.0, 2.0)]:
        save_best(
            tmp_path,
            model,
            step,
            {
                "domain": {
                    "6": {"tau": tau6, "anchors": 10},
                    "8": {"tau": tau8, "anchors": 10},
                }
            },
        )
    for step in (3, 4):
        save_checkpoint(
            tmp_path, model, optimizer, step, stream, {"checkpoint_keep": 1}, 0
        )
    assert [p.name for p in tmp_path.glob("step-*")] == ["step-00000004"]
    for block, step in [(6, 2), (8, 1)]:
        pointer = json.loads((tmp_path / f"best-b{block}.json").read_text())
        assert pointer["step"] == step
        assert (tmp_path / pointer["path"] / "model.safetensors").exists()
