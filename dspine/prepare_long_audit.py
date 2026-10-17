"""Select targeted long-response/prompt diagnostics before new target sampling.

All generated trajectories are NEW. Selection uses a previous response only to
find diagnostic prompts likely to reach >4K; these are not population estimates.
The training group contains prompts actually seen by the audited checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch

from .data import generation_paths, load_exclusion_ids, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='./work')
    p.add_argument('--training-sequences', type=int, default=600)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.project)
    output = root/'runs/audit-long'
    output.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusion_ids(root/'prompts/EXCLUDE_IDS.json')
    cursor = torch.load(root/'runs/pilot-mid/L6-A256/step-00000200/trainer.pt',
                        map_location='cpu', weights_only=False, mmap=True)['data']
    seen = []
    for path in cursor['paths'][:cursor['index']]:
        row = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if row['split']=='train' and row['id'] not in excluded:
            seen.append(row['id'])
            if len(seen) >= args.training_sequences:
                break
    seen = set(seen)
    groups = {'heldout':[], 'training_prompt':[]}
    for path in (root/'features/shards').glob('*.pt'):
        row = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if row['id'] in excluded:
            continue
        item = dict(id=row['id'], domain=row['domain'],
                    previous_response_tokens=len(row['input_ids'])-row['response_start'],
                    prompt_tokens=row['response_start']+row['start'])
        if row['split']=='validation':groups['heldout'].append(item)
        elif row['id'] in seen:groups['training_prompt'].append(item)
    selected = {}
    for kind, values in groups.items():
        longest = sorted(values, key=lambda r:(-r['previous_response_tokens'],r['id']))
        selected[kind] = longest[:1]
        long_prompt = sorted((r for r in values if r['prompt_tokens']>4096),
                             key=lambda r:(-r['previous_response_tokens'],r['id']))
        second = next((r for r in long_prompt if r['id']!=longest[0]['id']),None)
        second = second or next(r for r in longest if r['id']!=longest[0]['id'])
        selected[kind].append(second)
    wanted = {r['id']:kind for kind, values in selected.items() for r in values}
    found = {kind:[] for kind in selected}
    for row in records(generation_paths(root/'gen'),excluded):
        kind = wanted.pop(row['id'],None)
        if kind is not None:
            found[kind].append(row)
        if not wanted:break
    if wanted:raise ValueError('selected diagnostic records missing from generation archive')
    # Add two genuine held-out prompts with long prefixes, independent of their
    # new response lengths. The prior 64-token cache was never used for training.
    external = list(records(generation_paths(root/'runs/pilot/eval-generations/T1'),excluded))
    existing = {r['id'] for r in found['heldout']}
    for row in sorted(external,key=lambda r:(-len(r['prompt_ids']),r['id'])):
        if row['id'] not in existing:
            found['heldout'].append(row);existing.add(row['id'])
        if len(found['heldout'])>=4:break
    receipt = {}
    for kind, values in found.items():
        directory = output/'requests'/kind
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'records.jsonl').open('w')as stream:
            for row in values:
                row = {**row,'max_new_tokens':8192}
                stream.write(json.dumps(row)+'\n')
        receipt[kind] = [{'id':r['id'],'domain':r['domain'],'prompt_tokens':len(r['prompt_ids']),
                          'previous_response_tokens':len(r['output_ids'])}for r in values]
    (output/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'selected':{k:len(v)for k,v in found.items()},
                      'max_prompt_tokens':max(len(r['prompt_ids'])for v in found.values()for r in v)}),flush=True)


if __name__=='__main__':main()
