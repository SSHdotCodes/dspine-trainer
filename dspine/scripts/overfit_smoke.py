"""Small CPU optimization check, not evidence of real-target convergence."""
import argparse
import json
from pathlib import Path

import torch

from dspine import DSpineConfig, DSpineDraft
from dspine.losses import training_loss
from dspine.transfer_space import build_transfer_space


def run(steps=250):
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    config = DSpineConfig(hidden_size=32, target_hidden_size=32, vocab_size=23,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, intermediate_size=64,
                         target_layer_ids=(1,), block_size=4, context_window=4,
                         code_size=8, message_size=8, candidate_count=8,
                         eos_token_ids=())
    embedding, head = torch.randn(23, 32), torch.randn(23, 32)
    space = build_transfer_space(head, 8)
    model = DSpineDraft(config, embedding, head, space["codebook"].float(), space["basis"])
    context = torch.randn(4, 3, 32)
    # Identical anchors require the context reader to distinguish four examples.
    labels = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9], [0, 10, 11, 12]])
    target_ids = labels[..., None]
    target_probs = torch.ones_like(target_ids, dtype=torch.float32)
    frozen = {k: v.clone() for k, v in model.state_dict().items()
              if k.startswith(("embedding.", "lm_head.")) or k.endswith("codebook")}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(context, labels[:, 0], reference=True, return_layer_features=True)
        loss, _ = training_loss(model, out, labels, target_ids, target_probs, 100, 1000,
                               lambda_ce_reduction="weighted_token_mean",
                               refinement_start_position=2, embedding_loss_scale=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        out = model(context, labels[:, 0], reference=True, return_layer_features=False)
        proposed = model.propose(model.build_transition_cache(out), temperature=0).tokens
        accuracy = float((proposed == labels[:, 1:]).float().mean())
        shuffled = model(context.flip(0), labels[:, 0], reference=True, return_layer_features=False)
        shuffled_tokens = model.propose(model.build_transition_cache(shuffled), temperature=0).tokens
        shuffled_accuracy = float((shuffled_tokens == labels[:, 1:]).float().mean())
    for key, value in frozen.items():
        assert torch.equal(model.state_dict()[key], value), key
    return {"kind": "synthetic_cpu_overfit", "steps": steps, "initial_loss": losses[0],
            "final_loss": losses[-1], "greedy_proposal_accuracy": accuracy,
            "shuffled_context_accuracy": shuffled_accuracy,
            "auxiliary_scale": 0.0, "frozen_tables_unchanged": True,
            "limitation": "Four synthetic examples; does not establish real-target convergence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    assert result["final_loss"] < result["initial_loss"] * 0.15, result
    assert result["greedy_proposal_accuracy"] >= 0.99, result
    assert result["shuffled_context_accuracy"] < 0.5, result
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
