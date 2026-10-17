"""An interrupted CPU training run must equal uninterrupted training exactly."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file, save_file

from dspine.config import DSpineConfig
from dspine.data import FeatureStore
from dspine.transfer_space import build_transfer_space


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("hot", [False, True])
def test_trainer_exact_resume(tmp_path, packed, hot):
    torch.manual_seed(4)
    target = tmp_path / "target"
    target.mkdir()
    features = tmp_path / "features"
    emb = torch.randn(29, 16)
    head = torch.randn(29, 16)
    save_file(
        {"model.embed_tokens.weight": emb, "lm_head.weight": head},
        str(target / "model.safetensors"),
    )
    space = tmp_path / "space.pt"
    torch.save(build_transfer_space(head, 8), space)
    record = {
        "id": "sample",
        "domain": "math",
        "prompt_ids": list(range(5)),
        "output_ids": list(range(5, 25)),
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
    }
    ids = torch.arange(20).expand(25, -1)
    probs = torch.ones(25, 20) / 20
    FeatureStore(features, min_free_bytes=0).write(
        record, torch.randn(25, 2, 16), ids, probs, layer_ids=(1, 3)
    )
    c = DSpineConfig(
        hidden_size=16,
        target_hidden_size=16,
        vocab_size=29,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        target_layer_ids=(1, 3),
        block_size=5,
        context_window=8,
        code_size=8,
        message_size=4,
        candidate_count=3,
    )
    run = {
        "model": asdict(c),
        "packed": packed,
        "target": str(target),
        "transfer_space": str(space),
        "features": str(features),
        "device": "cpu",
        "total_steps": 3,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "anchors_per_sequence": 4,
        "eval_every": 0,
        "checkpoint_every": 100,
        "cpu_threads": 1,
        "head_chunk_size": 7,
    }
    if packed:
        run['schedule_response_tokens'] = 200
    if hot:
        hot_ids = tmp_path / 'hot.json'
        hot_ids.write_text(json.dumps(list(range(23))))
        run['proposal_vocabulary'] = dict(path=str(hot_ids), quantization='row_int8', loss='restricted')
    for name, interrupted in [("full", False), ("resumed", True)]:
        run["output"] = str(tmp_path / name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(run))
        cmd = [sys.executable, "-m", "dspine.train", "--config", str(path)]
        if interrupted:
            r = subprocess.run(
                cmd + ["--max-minutes", "0"], capture_output=True, text=True
            )
            assert r.returncode == 0, r.stderr
            latest = (tmp_path / name / "latest").read_text().strip()
            assert latest == "step-00000001"
            cmd += ["--resume", "latest"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    if packed:
        for name in ('full','resumed'):
            saved = torch.load(tmp_path/name/'step-00000003/trainer.pt',weights_only=False)
            assert saved['response_tokens_seen'] == 60
    full = load_file(str(tmp_path / "full/step-00000003/model.safetensors"))
    resumed = load_file(str(tmp_path / "resumed/step-00000003/model.safetensors"))
    assert full.keys() == resumed.keys()
    for key in full:
        torch.testing.assert_close(full[key], resumed[key], rtol=0, atol=0)


def test_fp32_master_weights_survive_bf16_target_checkpoint_roundtrip(tmp_path):
    from dspine.modeling import DSpineDraft
    from dspine.tests.test_reference import tiny

    model = tiny().bfloat16()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.float().data + torch.randn_like(p.float()) * 1e-5
    model.save_pretrained(tmp_path)
    restored = DSpineDraft.from_pretrained(
        tmp_path,
        model.embedding.weight,
        model.lm_head.weight,
        trainable_dtype=torch.float32,
    )
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(
                p, dict(restored.named_parameters())[name], atol=0, rtol=0
            )


def test_chunked_backward_bf16_head_with_fp32_residual():
    from dspine.losses import chunked_head_loss

    h = torch.randn(3, 8, requires_grad=True)
    head = torch.randn(29, 8).bfloat16()
    ids = torch.arange(20).expand(3, -1)
    probs = torch.ones(3, 20) / 20
    ce, l1 = chunked_head_loss(
        h, head, torch.tensor([1, 2, 3]), ids, probs, chunk_size=7
    )
    (ce + l1).sum().backward()
    assert h.grad.dtype == torch.float32 and h.grad.isfinite().all()


def test_cpu_checkpoint_rng_does_not_initialize_cuda(monkeypatch):
    from dspine.train import restore_rng, rng_state

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def unexpected():
        raise AssertionError("CPU checkpoint attempted CUDA context initialization")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected)
    state = rng_state()
    assert state["cuda"] == []
    restore_rng(state)


def test_checkpoint_selection_keeps_independent_serving_block_winners(tmp_path):
    from dspine.tests.test_reference import tiny
    from dspine.train import acceptance_summary, save_best

    model = tiny()
    # Domain means must be weighted by anchors; both blocks initially share one save.
    first = {
        "math": {"6": {"tau": 4.0, "anchors": 3}, "8": {"tau": 5.0, "anchors": 3}},
        "code": {"6": {"tau": 2.0, "anchors": 1}, "8": {"tau": 3.0, "anchors": 1}},
    }
    assert acceptance_summary(first) == {
        "6": {"tau": 3.5, "anchors": 4},
        "8": {"tau": 4.5, "anchors": 4},
    }
    save_best(tmp_path, model, 10, first)
    assert len(list(tmp_path.glob("best-step-*"))) == 1
    save_best(
        tmp_path,
        model,
        20,
        {"all": {"6": {"tau": 3.6, "anchors": 4}, "8": {"tau": 4.4, "anchors": 4}}},
    )
    pointers = json.loads((tmp_path / "best.json").read_text())["by_block"]
    assert pointers["6"]["step"] == 20 and pointers["8"]["step"] == 10
    assert (tmp_path / "best-step-00000010").exists()
    save_best(
        tmp_path,
        model,
        30,
        {"all": {"6": {"tau": 3.7, "anchors": 4}, "8": {"tau": 4.6, "anchors": 4}}},
    )
    pointers = json.loads((tmp_path / "best.json").read_text())["by_block"]
    assert pointers["6"]["step"] == pointers["8"]["step"] == 30
    assert sorted(p.name for p in tmp_path.glob("best-step-*")) == [
        "best-step-00000030"
    ]
    save_best(
        tmp_path,
        model,
        40,
        {"all": {"6": {"tau": 3.7, "anchors": 4}, "8": {"tau": 4.5, "anchors": 4}}},
    )
    assert not (tmp_path / "best-step-00000040").exists()


def test_single_block_best_pointer_migrates_without_losing_its_winner(tmp_path):
    from dspine.tests.test_reference import tiny
    from dspine.train import save_best

    model = tiny()
    model.save_pretrained(tmp_path / "best-step-00000005")
    (tmp_path / "best.json").write_text(
        json.dumps({"step": 5, "tau": 4.0, "block": 6, "path": "best-step-00000005"})
    )
    save_best(
        tmp_path,
        model,
        10,
        {"all": {"6": {"tau": 3.5, "anchors": 4}, "8": {"tau": 4.5, "anchors": 4}}},
    )
    pointers = json.loads((tmp_path / "best.json").read_text())["by_block"]
    assert pointers["6"]["step"] == 5 and pointers["8"]["step"] == 10
    assert (tmp_path / pointers["6"]["path"]).is_dir()
    assert (tmp_path / pointers["8"]["path"]).is_dir()
