"""Actual-checkpoint FP32 packed-cache versus serial-cache numerical gate."""

import argparse
import json
from pathlib import Path

import torch

from .data import dequantize_features, load_exclusion_ids
from .eval_rounds import PackedReplay
from .modeling import DSpineDraft
from .transfer_space import load_target_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--target',required=True)
    p.add_argument('--features',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--cases',type=int,default=4)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    emb,head = load_target_weights(args.target,'qwenfast_w8')
    model = DSpineDraft.from_pretrained(args.checkpoint,emb,head,trainable_dtype=torch.float32)
    del emb,head
    model = model.float().to(args.device).eval()
    excluded = load_exclusion_ids()
    candidates = []
    for path in Path(args.features).glob('*.pt'):
        row = torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        if row['id'] in excluded:continue
        candidates.append((len(row['input_ids']),row['response_start']+row['start'],str(path)))
    selected = list(dict.fromkeys([x[2] for x in sorted(candidates,reverse=True)[:args.cases//2]
                                  + sorted(candidates,key=lambda x:(x[1],x[0]),reverse=True)[:(args.cases+1)//2]]))
    reports = []
    with torch.inference_mode():
        for path in selected:
            row = torch.load(path,weights_only=True,map_location='cpu',mmap=True)
            replay = PackedReplay(model,row,8,max_block=8)
            for offset in (0,257,1025,4097):
                anchor = row['response_start']+offset
                if anchor+model.config.block_size>=len(row['input_ids']):continue
                first = max(0,anchor-model.config.context_window)
                features = dequantize_features({**row,'features':row['features'][first:anchor],
                    'scales':row['scales'][first:anchor] if row['scales'] is not None else None},
                    device=args.device,dtype=torch.float32)
                for block in (6,8):
                    out = model(features[None],row['input_ids'][anchor:anchor+1].to(args.device),
                        (torch.arange(first,anchor,device=args.device)+row['start'])[None],
                        torch.tensor([anchor+row['start']],device=args.device),block_size=block,return_layer_features=False)
                    serial = model.build_transition_cache(out)
                    packed = replay.get(anchor,block)
                    equal = torch.equal(serial.candidate_ids,packed.candidate_ids)
                    score_error = float((serial.initial_scores-packed.initial_scores).abs().max())
                    transition_error = float((serial.transitions-packed.transitions).abs().max())
                    q_error = max(float((serial.initial_scores.softmax(-1)-packed.initial_scores.softmax(-1)).abs().max()),
                                  float((serial.transitions.softmax(-1)-packed.transitions.softmax(-1)).abs().max()))
                    reports.append(dict(offset=offset,absolute_anchor=anchor+row['start'],block=block,
                        candidates_equal=equal,score_max_abs=score_error,transition_max_abs=transition_error,q_max_abs=q_error))
            print(json.dumps({'checked_prefixes':len(reports),'device':args.device}),flush=True)
    result = dict(device=args.device,precision='float32',checkpoint=args.checkpoint,cases=reports,
                  passed=bool(reports) and all(r['candidates_equal'] and r['q_max_abs']<3e-5 for r in reports))
    path = Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise AssertionError('actual-checkpoint packed replay disagrees with serial numerical reference')


if __name__=='__main__':main()
