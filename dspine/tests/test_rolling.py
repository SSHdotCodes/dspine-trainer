"""Storage rotation is gated by checkpoints and preserves training across stops."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from dspine.config import DSpineConfig
from dspine.data import FeatureStore, records, split_for_id
from dspine.rolling import (
    execute_plan,
    feature_path,
    latest_state,
    prepare_plan,
    purge_window,
    window_ready,
)
from dspine.transfer_space import build_transfer_space


def fixture(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    generator = torch.Generator().manual_seed(21)
    head = torch.randn(29, 16, generator=generator)
    save_file(
        {
            "model.embed_tokens.weight": torch.randn(29, 16, generator=generator),
            "lm_head.weight": head,
        },
        str(target / "model.safetensors"),
    )
    space = tmp_path / "space.pt"
    torch.save(build_transfer_space(head, 8), space)
    config = DSpineConfig(
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
    ids = {"train": [], "validation": []}
    for index in range(10000):
        name = f"rolling-{index}"
        split = split_for_id(name)
        if len(ids[split]) < (4 if split == "train" else 1):
            ids[split].append(name)
        if len(ids["train"]) == 4 and ids["validation"]:
            break
    generations = tmp_path / "gen"
    generations.mkdir()
    rows = [
        {
            "id": name,
            "domain": "math",
            "model_rev": "test-revision",
            "finish_reason": "stop",
            "prompt_ids": list(range(5)),
            "output_ids": list(range(5, 13)),
            "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        }
        for names in ids.values()
        for name in names
    ]
    # Replayed identical generation parts must not change epoch coverage.
    (generations / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [*rows, rows[0]])
    )
    run = {
        "model": asdict(config),
        "packed": True,
        "target": str(target),
        "target_revision": "test-revision",
        "transfer_space": str(space),
        "device": "cpu",
        "gradient_accumulation": 1,
        "anchors_per_sequence": 2,
        "batch_size": 1,
        "epochs": 1.5,
        "eval_every": 0,
        "checkpoint_every": 100,
        "cpu_threads": 1,
        "head_chunk_size": 7,
    }
    return run, generations


def plan_for(base, generations, root):
    base = {
        **base,
        "output": str(root / "checkpoints"),
        "features": str(root / "features/shards"),
        "feature_budget_root": str(root / "features"),
    }
    plan = prepare_plan(
        base,
        generations,
        root,
        window_bytes=43000,
        validation_records=1,
        validation_bytes=25000,
        max_feature_bytes=100000,
    )
    return base, plan


def fake_capture(plan, window, destination, root, seconds):
    del root, seconds
    store = FeatureStore(Path(destination) / "shards", min_free_bytes=0)
    for row in records(Path(window["input"]).glob("*.jsonl")):
        generator = torch.Generator().manual_seed(
            int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "little")
        )
        n = len(row["prompt_ids"]) + len(row["output_ids"])
        store.write(
            row,
            torch.randn(n, 2, 16, generator=generator),
            torch.arange(20).expand(n, -1),
            torch.ones(n, 20) / 20,
            layer_ids=plan["run"]["model"]["target_layer_ids"],
        )


def test_plan_is_frozen_disjoint_and_budgeted(tmp_path):
    base, generations = fixture(tmp_path)
    root = tmp_path / "plan"
    base, plan = plan_for(base, generations, root)
    windows = plan["windows"]
    assert len(windows) == 2
    assert [len(w["ids"]) for w in windows] == [2, 2]
    assert [w["stop_step"] for w in windows] == [3, 6]
    assert plan["training_records"] == 4 and plan["run"]["total_steps"] == 6
    train_ids = {name for w in windows for name in w["ids"]}
    assert not train_ids.intersection(plan["validation"]["ids"])
    assert all(w["estimated_bytes"] <= plan["window_budget"] for w in windows)
    assert prepare_plan(base, generations, root) == plan
    with pytest.raises(ValueError, match="unchanged source run config"):
        prepare_plan({**base, "anchors_per_sequence": 9}, generations, root)


def test_rotation_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    base, generations = fixture(tmp_path)
    outputs = []
    states = []
    # ai-job has no guaranteed cwd. The module's children must still import dspine.
    monkeypatch.chdir(tmp_path)
    for name in ("full", "resumed"):
        root = tmp_path / name
        _, plan = plan_for(base, generations, root)
        if name == "resumed":
            state = execute_plan(plan, root, capture=fake_capture, max_actions=3)
            assert state["phase"] == "rotate" and state["window"] == 0
            checkpoint = latest_state(plan["run"]["output"])
            assert checkpoint["step"] == 3
            assert checkpoint["data_window"] == "window-00000"
            assert not (root / "receipts/window-00000.json").exists()
            assert window_ready(plan["train_root"], plan["windows"][0], plan["run"])
        state = execute_plan(plan, root, capture=fake_capture)
        assert state["complete"] and state["window"] == 2
        assert not list((Path(plan["train_root"]) / "shards").glob("*.pt"))
        assert window_ready(plan["validation_root"], plan["validation"], plan["run"])
        assert len(list((root / "receipts").glob("*.json"))) == 2
        checkpoint = latest_state(plan["run"]["output"])
        assert checkpoint["step"] == 6
        outputs.append(load_file(str(Path(checkpoint["path"]) / "model.safetensors")))
        states.append(
            torch.load(Path(checkpoint["path"]) / "trainer.pt", weights_only=False)
        )
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        torch.testing.assert_close(outputs[0][name], outputs[1][name], atol=0, rtol=0)
    assert (
        states[0]["optimizer"]["param_groups"] == states[1]["optimizer"]["param_groups"]
    )
    for key, state in states[0]["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, states[1]["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    assert states[0]["rng"]["python"] == states[1]["rng"]["python"]
    torch.testing.assert_close(
        states[0]["rng"]["torch"], states[1]["rng"]["torch"], atol=0, rtol=0
    )


def test_rotation_refuses_unowned_shards_before_deleting_any(tmp_path):
    root = tmp_path / "features"
    known = feature_path(root, "known")
    known.parent.mkdir(parents=True)
    known.write_bytes(b"owned")
    unknown = feature_path(root, "unknown")
    unknown.write_bytes(b"unowned")
    with pytest.raises(ValueError, match="outside this rolling window"):
        purge_window(root, {"ids": ["known"]})
    assert known.read_bytes() == b"owned"
    assert unknown.read_bytes() == b"unowned"


def test_rotation_refuses_without_durable_matching_checkpoint(tmp_path):
    base, generations = fixture(tmp_path)
    root = tmp_path / "plan"
    _, plan = plan_for(base, generations, root)
    fake_capture(plan, plan["windows"][0], plan["train_root"], root, 120)
    (root / "state.json").write_text(
        json.dumps({"phase": "rotate", "window": 0, "complete": False})
    )
    with pytest.raises(ValueError, match="matching durable window checkpoint"):
        execute_plan(plan, root, capture=fake_capture)
    assert window_ready(plan["train_root"], plan["windows"][0], plan["run"])


def test_plan_freezes_exclusions_and_refuses_changed_list(tmp_path):
    base, generations = fixture(tmp_path)
    excluded_id = next(records(generations.glob("*.jsonl")))["id"]
    path = tmp_path / "exclude.json"
    path.write_text(json.dumps({"ids": [excluded_id]}))
    base["exclude_ids"] = str(path)
    root = tmp_path / "plan"
    base, plan = plan_for(base, generations, root)
    assert plan["training_records"] == 3
    assert excluded_id not in {name for w in plan["windows"] for name in w["ids"]}
    path.write_text(json.dumps({"ids": []}))
    with pytest.raises(ValueError, match="exclusions changed"):
        prepare_plan(base, generations, root)


def test_external_heldout_cache_is_preserved_and_never_trained(tmp_path):
    base, generations = fixture(tmp_path)
    heldout = next(
        row
        for row in records(generations.glob("*.jsonl"))
        if split_for_id(row["id"]) == "train"
    )
    source = tmp_path / "heldout-generations"
    source.mkdir()
    (source / "record.jsonl").write_text(json.dumps(heldout) + "\n")
    root = tmp_path / "plan"
    features = root / "features"
    base.update(
        {
            "features": str(features / "shards"),
            "feature_budget_root": str(features),
            "output": str(root / "checkpoints"),
            "eval_generations": str(source),
            "eval_features": str(features / "heldout/T1/shards"),
        }
    )
    plan = prepare_plan(
        base,
        generations,
        root,
        window_bytes=43000,
        validation_bytes=25000,
        max_feature_bytes=100000,
    )
    assert plan["run"]["eval_features"] == base["eval_features"]
    assert plan["validation"]["ids"] == [heldout["id"]]
    assert heldout["id"] not in {
        name for window in plan["windows"] for name in window["ids"]
    }
    fake_capture(plan, plan["validation"], plan["validation_root"], root, 120)
    assert window_ready(plan["validation_root"], plan["validation"], plan["run"])
    assert prepare_plan(base, generations, root) == plan


def test_final_corpus_rebudgets_epochs_from_measured_rates(tmp_path):
    base, generations = fixture(tmp_path)
    base['budget_sizing'] = dict(hours=1., seconds_per_step=2.,
        capture_tokens_per_second=100., capture_startup_seconds=3.,
        capture_startups_per_window=2, training_overhead_multiplier=1.1)
    _, plan = plan_for(base, generations, tmp_path/'budget')
    receipt = plan['run']['budget_sizing_receipt']
    assert receipt['training_sequences'] == 4
    assert receipt['prefill_tokens'] == 65
    assert receipt['capture_seconds'] == pytest.approx(
        65 / 100 + (len(plan['windows']) + 1) * 2 * 3)
    assert receipt['epochs'] > 1.5
    training_seconds = 4 * receipt['epochs'] * 2 * 1.1
    assert training_seconds + receipt['capture_seconds'] == pytest.approx(3600.)


def test_window_passes_reuse_extraction_and_stratify_domains(tmp_path):
    base, generations = fixture(tmp_path)
    rows = [json.loads(line) for line in (generations / 'records.jsonl').read_text().splitlines()]
    train_ids = sorted({r['id'] for r in rows if split_for_id(r['id']) == 'train'})
    for row in rows:
        row['domain'] = 'code' if row['id'] in train_ids[:2] else 'math'
    (generations / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    _, first = plan_for(base, generations, tmp_path/'first')
    base['epochs'] = 3.
    _, second = plan_for(base, generations, tmp_path/'second')
    assert len(first['windows']) == len(second['windows']) == 2
    assert sum(w['prefill_tokens'] for w in first['windows']) == sum(w['prefill_tokens'] for w in second['windows'])
    assert second['run']['total_steps'] == 2 * first['run']['total_steps']
    assert all(w['domain_records'] == {'code': 1, 'math': 1} for w in second['windows'])
    assert second['feature_reuse_policy'] == 'extract_once_then_consecutive_window_passes'


def test_frozen_proposal_map_fits_train_only_and_resumes_source_config(tmp_path):
    base, generations = fixture(tmp_path)
    rows = [json.loads(line) for line in (generations/'records.jsonl').read_text().splitlines()]
    for row in rows:
        row['output_ids'] = [28,28] if split_for_id(row['id'])=='validation' else [20,21]
    (generations/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    initial = tmp_path/'base-hot.json'
    initial.write_text('[0]')
    base['proposal_vocabulary'] = dict(base_path=str(initial),fit_from_frozen_generations=True,
                                      minimum_domain_coverage=.99,quantization='row_int8',loss='restricted')
    source, plan = plan_for(base,generations,tmp_path/'plan-hot')
    policy = plan['run']['proposal_vocabulary']
    assert json.loads(Path(policy['path']).read_text()) == [0,20,21]
    assert not policy['fit_from_frozen_generations']
    assert plan['proposal_vocabulary_receipt']['source'] == 'deduplicated_frozen_training_labels_only'
    assert prepare_plan(source,generations,tmp_path/'plan-hot') == plan
