"""Held-out sampling remains separate by temperature, with exact output ids."""

import io
import json
import subprocess
import sys
from collections import Counter

import pytest
import torch

from dspine.data import FeatureStore
from dspine.extract_features import request
from dspine.generate_eval import generate_one, render_prompt, select_prompts
from dspine.tests.test_data import make_record
from dspine.tests.test_rolling import fake_capture, fixture, plan_for


def test_heldout_selection_and_template_preserve_prompt_contract():
    prompts = [{"id": str(i), "domain": "a" if i < 9 else "b"} for i in range(18)]
    selected = select_prompts(prompts, 8)
    assert Counter(row["domain"] for row in selected) == {"a": 4, "b": 4}
    assert select_prompts(reversed(prompts), 8) == selected

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "hello"}]
            assert kwargs["reasoning_effort"] == "xhigh"
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
            return [12, 23]

    row = {
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning_effort": "xhigh",
        "enable_thinking": False,
        "prompt_tokens": 2,
    }
    assert render_prompt(Tokenizer(), row) == [12, 23]
    with pytest.raises(ValueError, match="count differs"):
        render_prompt(Tokenizer(), {**row, "prompt_tokens": 3})


@pytest.mark.parametrize("eval_mode", ["first_anchor", "rounds"])
def test_pilot_baseline_and_final_eval_report_both_temperatures(tmp_path, eval_mode):
    from dspine.data import records

    base, generations = fixture(tmp_path)
    root = tmp_path / "plan"
    _, plan = plan_for(base, generations, root)
    fake_capture(plan, plan["windows"][0], plan["train_root"], root, 120)
    fake_capture(plan, plan["validation"], plan["validation_root"], root, 120)
    row = next(records((root / "manifests/validation").glob("*.jsonl")))
    row["sampling"] = {"temperature": 0.0, "top_k": 20, "top_p": 0.95}
    greedy = root / "greedy"
    FeatureStore(greedy, min_free_bytes=0).write(
        row,
        torch.randn(13, 2, 16),
        torch.arange(20).expand(13, -1),
        torch.ones(13, 20) / 20,
        layer_ids=(1, 3),
    )
    run = {
        **plan["run"],
        "eval_greedy_features": str(greedy),
        "eval_anchors": 1,
        "eval_before_training": True,
        "eval_at_end": True,
        "eval_monte_carlo": True,
        "eval_mode": eval_mode,
        "eval_sequences": 1,
        "selection_blocks": [4],
    }
    if eval_mode == 'rounds':
        run.update(eval_long_features=run['eval_features'],eval_long_sequences=1,
                   eval_long_every=1,eval_long_packed_chunk_size=2,eval_report_first_anchor=True)
    path = root / "eval-run.json"
    path.write_text(json.dumps(run))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dspine.train",
            "--config",
            str(path),
            "--stop-at-step",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metrics = [
        json.loads(line)
        for line in (root / "checkpoints/metrics.jsonl").read_text().splitlines()
    ]
    evaluations = [row for row in metrics if "offline_tau" in row]
    assert [row["step"] for row in evaluations] == [0, 1]
    counts = [row["offline_tau_t0"]["math"]["4"]["anchors"] for row in evaluations]
    assert all(count == 1 for count in counts) if eval_mode == "first_anchor" else all(count > 0 for count in counts)
    assert all(row["evaluation_mode"] == eval_mode for row in evaluations)
    if eval_mode == 'rounds':
        assert all(row['offline_long_rounds']['complete'] for row in evaluations)
        assert all(row['offline_first_anchor']['math']['4']['anchors']==1 for row in evaluations)
        assert all('offset:0-256' in row['offline_long_rounds']['by_anchor'] for row in evaluations)


def test_eval_generation_uses_server_tokens_and_separate_temperature_roots(
    tmp_path, monkeypatch
):
    row = {
        "id": "heldout",
        "domain": "math",
        "reasoning_effort": "medium",
        "max_new_tokens": 16,
    }

    def response(request, timeout):
        body = json.loads(request.data)
        assert body["sampling_params"]["no_stop_trim"] is True
        return io.StringIO(
            json.dumps(
                {
                    "output_ids": [7, 8, 9],
                    "meta_info": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "finish_reason": {"type": "stop"},
                    },
                }
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    for temperature in (0.0, 1.0):
        generate_one("http://test", row, [2, 3], temperature, tmp_path, "rev", 16, [9])
        path = next((tmp_path / f"T{temperature:g}").glob("*.jsonl"))
        saved = json.loads(path.read_text())
        assert saved["output_ids"] == [7, 8, 9] and saved["split"] == "validation"
        assert saved["sampling"]["temperature"] == temperature
        assert generate_one(
            "http://test", row, [2, 3], temperature, tmp_path, "rev", 16, [9]
        )["resumed"]


def test_capture_move_resume_and_sampler_validation(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    (root / "inbox").mkdir(parents=True)
    destination = tmp_path / "eval/T1"
    row = {**make_record(), "model_rev": "rev", "split": "validation"}
    store = FeatureStore(root / "shards", min_free_bytes=0)

    def response(request, timeout):
        assert (
            json.loads(request.data)["input_ids"]
            == row["prompt_ids"] + row["output_ids"]
        )
        assert len(list((root / "inbox").glob("*.json"))) == 1
        store.write(
            row,
            torch.randn(25, 2, 8),
            torch.arange(20).expand(25, -1),
            torch.ones(25, 20) / 20,
            layer_ids=(1, 3),
        )
        return io.StringIO("{}")

    monkeypatch.setattr("urllib.request.urlopen", response)
    result = request("http://test", row, root, destination)
    assert result["retained_tokens"] == 25 and result["response_tokens"] == 20
    assert result["bytes"] > 0 and not list((root / "shards").glob("*.pt"))
    assert len(list((destination / "shards").glob("*.pt"))) == 1
    assert request("http://test", row, root, destination)["skipped"]
    with pytest.raises(ValueError, match="differs from the requested generation"):
        request(
            "http://test", {**row, "sampling": {"temperature": 0}}, root, destination
        )


def test_long_diagnostic_request_budget_covers_depth_and_long_prompts():
    from dspine.generate_long_audit import diagnostic_requests

    groups={kind:[dict(id='short',prompt_ids=[0]*100,output_ids=[0]*10000),
                  dict(id='long',prompt_ids=[0]*22000,output_ids=[0]*16000)]
            for kind in ('heldout','training_prompt')}
    selected=diagnostic_requests(groups)
    assert [cap for _,_,cap in selected]==[6144,6144,512,512]
    assert all(len(r['prompt_ids'])<=4096 for _,r,_ in selected[:2])
    assert all(len(r['prompt_ids'])>4096 for _,r,_ in selected[2:])
    assert sum(cap for _,_,cap in selected)/25 < 600
