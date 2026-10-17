"""GPU tests are opt-in so CPU validation never touches another agent's hold."""

import os
from dataclasses import replace

import pytest
import torch

from dspine.losses import training_loss
from dspine.modeling import DSpineDraft
from dspine.tests.test_reference import tiny

pytestmark = pytest.mark.skipif(
    os.environ.get("DSPINE_RUN_CUDA_TESTS") != "1",
    reason="requires explicit GPU reservation and DSPINE_RUN_CUDA_TESTS=1",
)


def cuda_model():
    template = tiny()
    # Match the real model's head width; tiny CPU-only tests use width four.
    config = replace(template.config, head_dim=128)
    return DSpineDraft(
        config,
        template.embedding.weight,
        template.lm_head.weight,
        template.selected_prefix_bridge.codebook,
    ).cuda()


def test_flex_matches_explicit_and_backward():
    model = cuda_model()
    model.selected_prefix_bridge.feature_write_weights.data.normal_(std=0.05)
    features = torch.randn(12, 32, device="cuda")
    ids = torch.randint(29, (12,), device="cuda")
    anchors = torch.tensor([2, 6, 9], device="cuda")
    actual = model.forward_packed(features, ids, anchors)
    reference = model.forward_packed(features, ids, anchors, reference=True)
    torch.testing.assert_close(actual.hidden, reference.hidden, atol=2e-4, rtol=2e-4)
    projection = torch.randn_like(actual.hidden)
    actual_grad = torch.autograd.grad(
        (actual.hidden * projection).sum(), model.fc.weight
    )[0]
    reference_grad = torch.autograd.grad(
        (reference.hidden * projection).sum(), model.fc.weight
    )[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=5e-4, rtol=1e-3)


def test_bf16_packed_training_checkpoint_backward():
    model = cuda_model().bfloat16()
    model.config.gradient_checkpointing = True
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    features = torch.randn(12, 32, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(29, (12,), device="cuda")
    anchors = torch.tensor([2, 6], device="cuda")
    offsets = anchors[:, None] + torch.arange(5, device="cuda")
    labels = ids[offsets]
    target_ids = torch.arange(20, device="cuda").expand(2, 5, -1)
    target_probs = torch.ones(2, 5, 20, device="cuda") / 20
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_packed(
            features,
            ids,
            anchors,
            correct_ids=labels,
            teacher_blocks=torch.tensor([True, False], device="cuda"),
        )
        assert output.hidden.dtype == torch.bfloat16
        loss, _ = training_loss(
            model, output, labels, target_ids, target_probs, 10, 100, chunk_size=7
        )
    loss.backward()
    assert torch.isfinite(loss) and model.fc.weight.grad.isfinite().all()
