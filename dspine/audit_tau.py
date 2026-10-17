"""Same-checkpoint acceptance, deep-prefix mechanics and generalization audit.

Exact walks require actual-decode p. Existing training shards contain teacher-
prefill p, so their gold-path replay is explicitly a diagnostic surrogate, never
an exact speculative-sampling estimate. First-proposal overlap does not need a
sample from p and is exact for each supplied prefix distribution.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import (
    AnchorStream,
    dequantize_features,
    load_exclusion_ids,
    target_distribution,
)
from .eval_offline import evaluate
from .eval_rounds import domain_balanced_paths, evaluate_rounds, offset_bucket
from .modeling import DSpineDraft
from .transfer_space import load_target_weights


def save(path, result):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def payload(row, start=0, end=None):
    return {"features": row["features"][start:end], "encoding": row["encoding"],
            "scales": None if row["scales"] is None else row["scales"][start:end]}


@torch.inference_mode()
def prefix_probe(model, row, anchor, mechanics=False):
    device = next(model.parameters()).device
    dtype = model.embedding.weight.dtype
    start = max(0, anchor - model.config.context_window)
    features = dequantize_features(payload(row, start, anchor), device, dtype)
    ids = row['input_ids'].to(device)
    absolute = row['start'] + anchor
    cp = (torch.arange(start, anchor, device=device) + row['start'])[None]
    out = model(features[None], ids[anchor:anchor+1], cp,
                torch.tensor([absolute], device=device), return_layer_features=False)
    cache = model.build_transition_cache(out)
    pi, pp = target_distribution(row)
    count = model.config.block_size - 1
    pi, pp = pi[anchor+1:anchor+1+count].to(device), pp[anchor+1:anchor+1+count].to(device)
    labels = ids[anchor+1:anchor+1+count]
    q = cache.initial_scores[0, 0].softmax(-1)
    match = cache.candidate_ids[0, 0, :, None] == pi[0, None]
    target_on_candidates = (match * pp[0, None]).sum(-1)
    overlap = torch.minimum(q, target_on_candidates).sum().item()
    previous = 0
    survival = 1.
    values = []
    zero_p = 0
    for t in range(count):
        logits = cache.initial_scores[0, 0] if t == 0 else cache.transitions[0, t-1, previous]
        local = cache.candidate_ids[0, t] == labels[t]
        qy = (logits.softmax(-1) * local).sum().item()
        py = (pp[t] * (pi[t] == labels[t])).sum().item()
        zero_p += py <= 0
        survival *= min(1., qy / py) if py > 0 else 0.
        values.append(survival)
        previous = int(local.long().argmax())
    report = dict(offset=anchor-row['response_start'], absolute_anchor=absolute,
                  offset_bucket=offset_bucket(anchor-row['response_start']),
                  exceeds_window=absolute > model.config.context_window,
                  context_first_absolute=row['start']+start, context_last_absolute=absolute-1,
                  block_positions=[absolute, absolute+model.config.block_size-1],
                  first_proposal_overlap=overlap, unsupported_gold_labels=zero_p,
                  gold_path_surrogate_tau={str(b):1+sum(values[:b-1]) for b in (6,8)},
                  p_source=row.get('target_distribution_source','teacher_prefill'))
    if mechanics:
        whole = dequantize_features(payload(row), device, dtype)
        offsets = torch.tensor([anchor], device=device)
        packed = model.forward_packed(whole, ids, offsets, row['start'], return_layer_features=False)
        hidden_diff = (packed.hidden.float()-out.hidden.float()).abs()
        pre_diff = (packed.pre_injection.float()-out.pre_injection.float()).abs()
        # False curriculum mask must ignore arbitrary correct future labels.
        changed = whole.clone()
        changed[anchor:] += 100
        future_ids = ids.clone()
        future_ids[anchor+1:] = (future_ids[anchor+1:] + 1) % model.config.vocab_size
        falsified = model.forward_packed(changed, future_ids, offsets, row['start'],
            correct_ids=future_ids[anchor:anchor+model.config.block_size][None],
            teacher_blocks=torch.zeros(1, dtype=torch.bool, device=device),
            return_layer_features=False)
        report['mechanics'] = dict(packed_vs_cropped_hidden_max_abs=float(hidden_diff.max()),
            packed_vs_cropped_hidden_relative_l2=float(torch.linalg.vector_norm(hidden_diff)/torch.linalg.vector_norm(out.hidden.float())),
            pre_injection_max_abs=float(pre_diff.max()),
            future_feature_or_label_leak_max_abs=float((falsified.hidden-packed.hidden).abs().max()),
            teacher_forcing=False, target_features_exclude_anchor=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--exact-features', required=True)
    p.add_argument('--cache-features', required=True)
    p.add_argument('--training-cursor', help='Own trainer.pt: identify the actually presented training slice')
    p.add_argument('--training-sequences', type=int, help='Limit cursor prefix to the checkpoint step times accumulation (e.g.150*4)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--sequences', type=int, default=64)
    p.add_argument('--prefixes-per-group', type=int, default=8)
    p.add_argument('--skip-round-walk', action='store_true')
    p.add_argument('--skip-prefix-probes', action='store_true')
    p.add_argument('--skip-first-anchor', action='store_true')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--proposal-vocab', help='Optional serving-aligned global token IDs')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(args.cpu_threads)
    emb, head = load_target_weights(args.target, 'qwenfast_w8')
    model = DSpineDraft.from_pretrained(args.checkpoint, emb, head).to(args.device).eval()
    del emb, head
    if args.proposal_vocab:
        from .hot_vocab import load_hot_ids

        model.configure_proposal_vocabulary(load_hot_ids(args.proposal_vocab, model.config.vocab_size))
    if args.fp32:
        model = model.float()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), device=args.device,
                  precision=str(model.embedding.weight.dtype), protocol='Same checkpoint and frozen effective W8 head for every metric.',
                  checkpoint_model_sha256=file_sha256(Path(args.checkpoint)/'model.safetensors'))
    result['proposal_vocab_size'] = model.config.vocab_size if model.proposal_weight is None else len(model.proposal_weight)
    result['proposal_vocab_path'] = args.proposal_vocab
    start = time.monotonic()
    stream = AnchorStream(args.exact_features, block_size=model.config.block_size,
                          context_window=model.config.context_window, split='validation', anchor_policy='first')
    for mc in (() if args.skip_first_anchor else (False, True)):
        survival_totals = torch.zeros(model.config.block_size-1)
        survival_counts = torch.zeros_like(survival_totals)

        def collect_survival(survival, valid):
            survival_totals.add_(survival.float().cpu().sum(0))
            survival_counts.add_(valid.cpu().sum(0))

        result['first_anchor_MC' if mc else 'first_anchor_RB'] = evaluate(model, stream, anchors=args.sequences,
            batch_size=1, monte_carlo=mc, require_decode_probs=True,
            diagnostic_callback=None if mc else collect_survival)
        if not mc:
            result['first_anchor_RB_prefix_survival'] = (survival_totals/survival_counts.clamp_min(1)).tolist()
        stream = AnchorStream(args.exact_features, block_size=model.config.block_size,
                              context_window=model.config.context_window, split='validation', anchor_policy='first')
        save(output, result);print(json.dumps({'stage':'first_anchor', 'MC':mc, 'seconds':time.monotonic()-start}), flush=True)
    if not args.skip_prefix_probes:
        seen_train = None
        excluded = load_exclusion_ids()
        if args.training_cursor:
            cursor = torch.load(args.training_cursor, map_location='cpu', weights_only=False, mmap=True)['data']
            seen_train = set()
            for name in cursor['paths'][:cursor['index']]:
                sample = torch.load(name, map_location='cpu', weights_only=True, mmap=True)
                if sample['split']=='train' and sample['id'] not in excluded:
                    seen_train.add(name)
                    if args.training_sequences and len(seen_train) >= args.training_sequences:
                        break
        groups = {}
        probes = []
        mechanics_done = set()
        for path in domain_balanced_paths(sorted(Path(args.cache_features).glob('*.pt'))):
            row = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if row['id'] in excluded:
                continue
            if row['split']=='train' and seen_train is not None and str(path) not in seen_train:
                continue
            for offset in (0, 128, 256, 768, 1024, 3072, 4096, 8192):
                anchor = row['response_start'] + offset
                if anchor + model.config.block_size >= len(row['input_ids']):
                    continue
                key = (row['split'], offset_bucket(offset), anchor+row['start'] > model.config.context_window)
                if groups.get(key,0) >= args.prefixes_per_group:
                    continue
                probe = prefix_probe(model, row, anchor, mechanics=key not in mechanics_done)
                probe.update(split=row['split'], domain=row['domain'], record_sha256=hashlib.sha256(row['id'].encode()).hexdigest())
                probes.append(probe);groups[key] = groups.get(key,0)+1;mechanics_done.add(key)
                result['prefix_probes'] = probes;save(output,result)
                print(json.dumps({'stage':'prefix_probe','split':row['split'],'offset':offset,'seconds':time.monotonic()-start}),flush=True)
        result['training_prefix_note']='Actually presented training cursor slice; gold-path surrogate uses teacher-prefill p and is not exact sampling tau. First-proposal overlap and packed/cropped parity do not require on-policy labels.'
    if not args.skip_round_walk:
        def progress(report):
            result['complete_rounds'] = report
            save(output, result)
            print(json.dumps({'stage':'round_walk','sequences':report['sequences'],
                              'seconds':report['seconds']}), flush=True)

        result['complete_rounds'] = evaluate_rounds(model, args.exact_features,
            blocks=(6,8), sequences=args.sequences, require_decode_probs=True, collect_diagnostics=True,
            progress_callback=progress)
    result['seconds'] = time.monotonic()-start
    save(output,result);print(json.dumps({'stage':'complete','seconds':result['seconds']}),flush=True)


if __name__ == '__main__':
    main()
