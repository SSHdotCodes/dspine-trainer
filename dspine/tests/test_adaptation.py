"""Changing feature-layer identities must not silently copy old fusion slots."""

import json
import sys
from dataclasses import replace

import torch
from safetensors.torch import save_file

from dspine.adapt_target import main
from dspine.tests.test_reference import tiny


def test_changed_layer_ids_reinitialize_fusion_and_keep_compatible_decoder(
    tmp_path, monkeypatch
):
    model = tiny()
    old = tmp_path / "old"
    model.save_pretrained(old)
    target = tmp_path / "target"
    target.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": model.embedding.weight,
            "lm_head.weight": model.lm_head.weight,
        },
        str(target / "model.safetensors"),
    )
    config = tmp_path / "new.json"
    replace(model.config, target_layer_ids=(1, 4)).save(config)
    output = tmp_path / "adapted"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapt",
            "--old-checkpoint",
            str(old),
            "--new-target",
            str(target),
            "--new-config",
            str(config),
            "--output",
            str(output),
            "--same-tokenizer",
        ],
    )
    main()
    report = json.loads((output / "adaptation.json").read_text())
    assert "fc.weight" in report["reinitialized"]
    assert "layers.0.input_layernorm.weight" in report["copied"]
    space = torch.load(output / "transfer-space.pt", weights_only=True)
    assert space["head_quantization"] is None
