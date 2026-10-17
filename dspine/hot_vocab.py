"""Frozen serving-aligned proposal vocabulary and CPU coverage measurements.

The target p and transfer space stay full-vocabulary. A restricted draft q has
zero mass outside its hot set; its L1 loss still includes that target mass.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .data import (
    generation_paths,
    load_exclusion_ids,
    records,
    split_for_id,
    target_distribution,
)


def load_hot_ids(path, vocab_size):
    ids = torch.tensor(json.loads(Path(path).read_text()), dtype=torch.long)
    if ids.ndim != 1 or not len(ids) or len(ids.unique()) != len(ids):
        raise ValueError("hot vocabulary must be a nonempty list of unique token IDs")
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError("hot vocabulary token ID outside target vocabulary")
    return ids.sort().values


def row_int8_weights(weight):
    """Effective rows of serving.quantization.HotInt8Head, including BF16 cast.

    This is its additional per-row INT8 quantization, applied AFTER the optional
    target group128 W8 reconstruction. Activations remain BF16.
    """
    rows = weight.float()
    scales = rows.abs().amax(-1).clamp_min(1e-30) / 127
    quantized = (rows / scales[:, None]).round().clamp(-127, 127)
    return (quantized * scales[:, None]).to(weight.dtype)


def coverage(roots, hot_ids, vocab_size, exclude_ids=None):
    membership = torch.zeros(vocab_size, dtype=torch.bool)
    membership[hot_ids] = True
    excluded = load_exclusion_ids(exclude_ids)
    groups = defaultdict(lambda: dict(tokens=0, sampled_inside=0, mass_sum=0.,
                                    minimum_mass=1., mass_below_099=0))
    for root in roots:
        for path in sorted(Path(root).glob("*.pt")):
            row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if row["id"] in excluded:
                continue
            ids, probs = target_distribution(row)
            start = row["response_start"]
            ids, probs = ids[start:], probs[start:]
            mass = (membership[ids.long()] * probs).sum(-1)
            inside = membership[row["input_ids"][start:]]
            for group in ("all", "split:" + row["split"], "domain:" + row["domain"],
                          "split:" + row["split"] + "/domain:" + row["domain"]):
                g = groups[group]
                g["tokens"] += len(mass)
                g["sampled_inside"] += int(inside.sum())
                g["mass_sum"] += float(mass.double().sum())
                g["minimum_mass"] = min(g["minimum_mass"], float(mass.min()))
                g["mass_below_099"] += int((mass < .99).sum())
    for g in groups.values():
        g["mean_target_mass"] = g.pop("mass_sum") / g["tokens"]
        g["sampled_token_coverage"] = g["sampled_inside"] / g["tokens"]
    return dict(hot_vocab_size=len(hot_ids), vocab_size=vocab_size,
                sorted_ids_sha256=hashlib.sha256(hot_ids.numpy().tobytes()).hexdigest(),
                groups=dict(groups), note="Target mass includes all stored joint cutoff ties. Training cache p is teacher-prefill p; exact held-out p is actual decode p.")


def expand_from_training(roots, hot_ids, vocab_size, minimum_domain_mass=.9995, exclude_ids=None):
    """Union the base set with enough frequent training mass per domain.

    Validation records never influence selection. Keep the original hot set and
    add tokens greedily by per-domain target probability mass, without oversampling.
    """
    if not 0 < minimum_domain_mass <= 1:
        raise ValueError("minimum domain mass must be in (0,1]")
    excluded = load_exclusion_ids(exclude_ids)
    mass = defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.float64))
    for root in roots:
        for path in sorted(Path(root).glob("*.pt")):
            row = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if row["split"] != "train" or row["id"] in excluded:
                continue
            ids, probs = target_distribution(row)
            start = row["response_start"]
            mass[row["domain"]].scatter_add_(0, ids[start:].long().flatten(), probs[start:].double().flatten())
    return expand_from_mass(mass,hot_ids,vocab_size,minimum_domain_mass)


def expand_from_mass(mass, hot_ids, vocab_size, minimum_domain_mass=.9995):
    if not 0 < minimum_domain_mass <= 1:
        raise ValueError("minimum domain mass must be in (0,1]")
    inside = torch.zeros(vocab_size, dtype=torch.bool)
    inside[hot_ids] = True
    for domain in sorted(mass):
        values = mass[domain]
        needed = minimum_domain_mass * values.sum() - values[inside].sum()
        if needed <= 0:
            continue
        scores = values.clone()
        scores[inside] = 0
        ordering = scores.argsort(descending=True, stable=True)
        count = int(torch.searchsorted(scores[ordering].cumsum(0), needed)) + 1
        inside[ordering[:count]] = True
    result = inside.nonzero().flatten()
    return result, {domain: float(values[inside].sum() / values.sum()) for domain, values in mass.items()}


def expand_from_raw_training(root, hot_ids, vocab_size, minimum_domain_mass=.9995, exclude_ids=None):
    counts, _ = collect_raw_counts(root,exclude_ids)
    return expand_from_counts(counts['train'],hot_ids,vocab_size,minimum_domain_mass)


def collect_raw_counts(root,exclude_ids=None):
    counts = {'train':defaultdict(Counter),'validation':defaultdict(Counter)}
    seen = set()
    for row in records(generation_paths(root),load_exclusion_ids(exclude_ids)):
        if row['id'] in seen or row['finish_reason']=='abort' or len(row['output_ids'])<2:continue
        seen.add(row['id'])
        counts[split_for_id(row['id'])][row['domain']].update(row['output_ids'])
    return counts,len(seen)


def expand_from_counts(counts,hot_ids,vocab_size,minimum_domain_mass=.9995):
    mass = {}
    for domain,values in counts.items():
        vector = torch.zeros(vocab_size,dtype=torch.float64)
        vector[torch.tensor(list(values),dtype=torch.long)] = torch.tensor(list(values.values()),dtype=torch.float64)
        mass[domain] = vector
    return expand_from_mass(mass,hot_ids,vocab_size,minimum_domain_mass)


def raw_label_coverage(root, vocabularies, exclude_ids=None, cached_counts=None):
    """Full-corpus sampled-token coverage; separate from teacher-p mass coverage."""
    sets = {name:set(ids.tolist()) for name,ids in vocabularies.items()}
    groups = defaultdict(lambda: dict(tokens=0, inside={k:0 for k in sets}))
    counts,count = cached_counts or collect_raw_counts(root,exclude_ids)
    for split,domains in counts.items():
        for domain,frequency in domains.items():
            inside = {name:sum(n for token,n in frequency.items() if token in values) for name,values in sets.items()}
            for key in ('all','domain:'+domain,'split:'+split,'split:'+split+'/domain:'+domain):
                groups[key]['tokens'] += sum(frequency.values())
                for name,value in inside.items():groups[key]['inside'][name] += value
    for g in groups.values():
        g['coverage'] = {k:v/g['tokens'] for k,v in g['inside'].items()}
    return dict(records=count, groups=dict(groups),
                note='Sampled-token frequencies on deduplicated raw generations, not exact teacher target-p mass. No validation token influences map fitting.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot-ids", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--exclude-ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expanded-ids-output")
    parser.add_argument("--minimum-domain-mass", type=float, default=.9995)
    parser.add_argument("--generations")
    parser.add_argument("--fit-raw", action="store_true", help="Fit additions from all raw TRAIN labels instead of the smaller feature cache")
    parser.add_argument("--sweep", action="store_true", help="Also measure coverage/size tradeoffs from the same frozen training counts")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ids = load_hot_ids(args.hot_ids, args.vocab_size)
    result = coverage(args.features, ids, args.vocab_size, args.exclude_ids)
    raw_cache = None
    if args.expanded_ids_output:
        if args.fit_raw:
            if not args.generations:parser.error('--fit-raw requires --generations')
            raw_cache = collect_raw_counts(args.generations,args.exclude_ids)
            ids, fitted = expand_from_counts(raw_cache[0]['train'],ids,args.vocab_size,args.minimum_domain_mass)
            if args.sweep:
                sweep = []
                for keep_base in (True,False):
                    base = load_hot_ids(args.hot_ids,args.vocab_size) if keep_base else torch.empty(0,dtype=torch.long)
                    for goal in (.98,.99,.995,.999,.9995):
                        candidate,masses = expand_from_counts(raw_cache[0]['train'],base,args.vocab_size,goal)
                        candidate_path = Path(args.expanded_ids_output).with_name(f'proposal-base{int(keep_base)}-mass{goal:g}.json')
                        candidate_path.write_text(json.dumps(candidate.tolist())+'\n')
                        swept = raw_label_coverage(args.generations,{'selected':candidate},cached_counts=raw_cache)
                        sweep.append(dict(preserve_base=keep_base,goal=goal,vocab_size=len(candidate),
                                          path=str(candidate_path),training_mass=masses,coverage=swept))
                result['size_coverage_sweep'] = sweep
        else:
            ids, fitted = expand_from_training(args.features, ids, args.vocab_size,
                                              args.minimum_domain_mass, args.exclude_ids)
        Path(args.expanded_ids_output).write_text(json.dumps(ids.tolist()) + "\n")
        result["expanded"] = coverage(args.features, ids, args.vocab_size, args.exclude_ids)
        result["expanded"]["training_selection_mass"] = fitted
        result["expanded"]["fitting_source"] = "raw_training_labels" if args.fit_raw else "training_teacher_p"
    if args.generations:
        result['raw_generation_labels'] = raw_label_coverage(args.generations,
            {'original':load_hot_ids(args.hot_ids,args.vocab_size),'expanded':ids},args.exclude_ids,raw_cache)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
