"""Reproduce Tencent b16 acceptance with BF16 HF Qwen3-8B greedy generations.

Cache continuations and features locally, then ablate scale on the SAME
anchors. This is a method/loader validation, never training data or a final
quality benchmark. Thinking is explicitly disabled in the chat template.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .modeling import DSpineDraft


def greedy_rounds(model, payload, block_size=16):
    """Follow actual greedy speculative round boundaries, including the bonus."""
    features = payload["features"]
    ids = payload["ids"]
    start = payload["response_start"]
    end = len(ids)
    anchor = start
    accepted = rounds = 0
    accepted_counts = []
    device = next(model.parameters()).device
    while anchor + block_size < end:
        context = max(0, anchor - model.config.context_window)
        out = model(
            features[context:anchor][None].to(device),
            ids[anchor : anchor + 1].to(device),
            torch.arange(context, anchor, device=device)[None],
            torch.tensor([anchor], device=device),
            return_layer_features=False,
        )
        cache = model.build_transition_cache(out)
        proposal = model.propose(cache, temperature=0)
        correct = proposal.tokens[0].cpu() == ids[anchor + 1 : anchor + block_size]
        length = int(correct.long().cumprod(0).sum()) + 1
        accepted += length
        rounds += 1
        accepted_counts.append(length)
        anchor += length
    return accepted, rounds, accepted_counts


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--scales", default="32,10,1")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(args.device)
        .eval()
    )
    emb = target.get_input_embeddings().weight.detach()
    head = target.get_output_embeddings().weight.detach()
    draft = DSpineDraft.from_pretrained(args.draft, emb, head).to(args.device).eval()
    questions = load_dataset("openai/gsm8k", "main", split="test")
    begin = time.monotonic()
    paths = []
    for index in range(min(args.prompts, len(questions))):
        path = root / f"example-{index:04d}.pt"
        paths.append(path)
        if path.exists():
            continue
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": questions[index]["question"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(args.device)
        # Some tokenizer releases return BatchEncoding for return_tensors.
        if not isinstance(prompt, torch.Tensor):
            prompt = prompt["input_ids"]
        with torch.inference_mode():
            ids = target.generate(
                prompt,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            result = target(ids, output_hidden_states=True, use_cache=False)
            features = torch.cat(
                [result.hidden_states[i + 1] for i in draft.config.target_layer_ids], -1
            )[0].cpu()
        torch.save(
            {
                "ids": ids[0].cpu(),
                "features": features,
                "response_start": prompt.shape[1],
                "question_index": index,
                "thinking": False,
                "sampling": {"temperature": 0},
            },
            path,
        )
        del result, features
        print(
            json.dumps({"generated": index + 1, "seconds": time.monotonic() - begin}),
            flush=True,
        )
    results = {}
    with torch.inference_mode():
        for scale in map(float, args.scales.split(",")):
            draft.config.feature_scale = scale
            tokens = rounds = 0
            per_prompt = []
            for index, path in enumerate(paths):
                item = torch.load(path, map_location="cpu", weights_only=True)
                accepted, count, lengths = greedy_rounds(draft, item)
                tokens += accepted
                rounds += count
                per_prompt.append(
                    {
                        "question_index": index,
                        "tokens": accepted,
                        "rounds": count,
                        "tau": accepted / max(1, count),
                    }
                )
                print(
                    json.dumps(
                        {
                            "scale": scale,
                            "evaluated": index + 1,
                            "tau": tokens / max(1, rounds),
                            "rounds": rounds,
                        }
                    ),
                    flush=True,
                )
            results[str(scale)] = {
                "tau": tokens / max(1, rounds),
                "tokens": tokens,
                "rounds": rounds,
                "per_prompt": per_prompt,
            }
            (root / "results.json").write_text(
                json.dumps(
                    {
                        "temperature": 0,
                        "block": 16,
                        "thinking": False,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "complete": True,
                "results": {
                    k: {n: v for n, v in x.items() if n != "per_prompt"}
                    for k, x in results.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
