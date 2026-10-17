"""Small, separately stored T=0/T=1 held-out continuations on an owned server.

This is a pilot evaluation helper, not the training generation campaign. Input
prompts must be held out before generation. Token ids come directly from the
target tokenizer and server; text is never retokenized to recover outputs.
"""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from .data import load_exclusion_ids, records
from .decode_probs import processed_decode_probs


def select_prompts(prompts, count, seed=42):
    """Domain-balanced selection independent of continuation content/length."""
    groups = defaultdict(list)
    for row in prompts:
        groups[row["domain"]].append(row)
    for domain in groups:
        groups[domain].sort(
            key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()
        )
    selected = []
    index = 0
    while len(selected) < count:
        added = False
        for domain in sorted(groups):
            if index < len(groups[domain]):
                selected.append(groups[domain][index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        index += 1
    return selected


def render_prompt(tokenizer, row):
    kwargs = dict(row.get("chat_template_kwargs") or {})
    if set(kwargs) - {"reasoning_effort", "enable_thinking", "preserve_thinking"}:
        raise ValueError("unsupported template option")
    if (
        kwargs.get("reasoning_effort", row["reasoning_effort"])
        != row["reasoning_effort"]
    ):
        raise ValueError("conflicting reasoning effort")
    kwargs["reasoning_effort"] = row["reasoning_effort"]
    if "enable_thinking" in row:
        if (
            kwargs.get("enable_thinking", row["enable_thinking"])
            != row["enable_thinking"]
        ):
            raise ValueError("conflicting thinking option")
        kwargs["enable_thinking"] = row["enable_thinking"]
    ids = tokenizer.apply_chat_template(
        row["messages"],
        tools=row.get("tools"),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("invalid prompt ids")
    if "prompt_tokens" in row and row["prompt_tokens"] != len(ids):
        raise ValueError("held-out prompt count differs from the pinned chat template")
    return ids


def generate_one(url, row, ids, temperature, output, revision, max_tokens, eos_ids, record_decode_probs=False):
    directory = Path(output) / f"T{temperature:g}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(row["id"].encode()).hexdigest() + ".jsonl")
    sampling = {"temperature": temperature, "top_k": 20, "top_p": 0.95}
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved["id"] != row["id"]
            or saved["prompt_ids"] != ids
            or saved["model_rev"] != revision
            or saved["sampling"] != sampling
            or (record_decode_probs and temperature == 1 and "target_decode_ids" not in saved)
        ):
            raise ValueError("existing evaluation generation has a different contract")
        return {"id": row["id"], "temperature": temperature, "resumed": True}
    body = {
        "input_ids": ids,
        "sampling_params": {
            **sampling,
            "max_new_tokens": min(max_tokens, row["max_new_tokens"]),
            "stop_token_ids": eos_ids,
            "skip_special_tokens": False,
            "no_stop_trim": True,
        },
        "stream": False,
    }
    if record_decode_probs and temperature == 1:
        body.update(return_logprob=True, top_logprobs_num=64, logprob_start_len=-1)
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    begin = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        reply = json.load(response)
    meta = reply["meta_info"]
    outputs = reply.get("output_ids")
    reason = meta.get("finish_reason", {})
    reason = reason.get("type") if isinstance(reason, dict) else reason
    if (
        meta.get("prompt_tokens") != len(ids)
        or reply.get("prompt_token_ids", ids) != ids
        or not isinstance(outputs, list)
        or any(type(token) is not int or token < 0 for token in outputs)
        or meta.get("completion_tokens", len(outputs)) != len(outputs)
        or reason not in ("stop", "length")
    ):
        raise ValueError("server did not return a complete token-exact continuation")
    record = {
        "id": row["id"],
        "domain": row["domain"],
        "reasoning_effort": row["reasoning_effort"],
        "prompt_ids": ids,
        "output_ids": outputs,
        "model_rev": revision,
        "finish_reason": reason,
        "sampling": sampling,
        "split": "validation",
        "gen_seconds": time.monotonic() - begin,
    }
    if record_decode_probs and temperature == 1:
        index, probabilities = processed_decode_probs(meta["output_top_logprobs"], outputs)
        record.update(target_decode_ids=index, target_decode_probs=probabilities)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record) + "\n")
    temp.replace(path)
    return {
        "id": row["id"],
        "temperature": temperature,
        "output_tokens": len(outputs),
        "seconds": record["gen_seconds"],
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30220")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--exclude-ids", default=None)
    parser.add_argument("--temperatures", default="1,0")
    parser.add_argument("--record-decode-probs", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    template = Path(args.target) / "chat_template.jinja"
    tokenizer.chat_template = template.read_text()
    generation = json.loads((Path(args.target) / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    prompts = select_prompts(
        records([args.prompts], load_exclusion_ids(args.exclude_ids)), args.count
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = []
        for row in prompts:
            ids = render_prompt(tokenizer, row)
            for temperature in map(float, args.temperatures.split(",")):
                pending.append(
                    pool.submit(
                        generate_one,
                        args.url,
                        row,
                        ids,
                        temperature,
                        args.output,
                        args.target_revision,
                        args.max_new_tokens,
                        eos,
                        args.record_decode_probs,
                    )
                )
        for future in concurrent.futures.as_completed(pending):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
