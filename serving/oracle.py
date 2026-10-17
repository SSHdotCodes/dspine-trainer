"""Explicit test-only frozen target conditional table, never a serving default."""
import hashlib
import json
from pathlib import Path

import torch


class FrozenOracle:
    def __init__(self, path, expected_sha256, device):
        path=Path(path);payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected_sha256:
            raise ValueError('frozen oracle hash mismatch')
        self.manifest=json.loads(payload)
        if self.manifest['protocol']['joint_tokens']!=3:
            raise ValueError('oracle requires complete depth3 enumeration')
        self.prefixes={tuple(row['input_ids']):row for row in self.manifest['prefixes']}
        self.nodes={row['name']:{tuple(node['path']):node for node in row['nodes']}
                    for row in self.manifest['prefixes']}
        self.device=device;self.histories={};self.names={}
        for name,nodes in self.nodes.items():
            for path,node in nodes.items():
                p=torch.tensor(node['probabilities'],dtype=torch.float32,device=device)
                if p.numel()!=len(node['ids']) or bool((p<=0).any()) or abs(float(p.sum())-1)>1e-5:
                    raise ValueError('invalid frozen probability row')
                node['_p']=p
                node['_ids']=torch.tensor(node['ids'],dtype=torch.long,device=device)
                if len(path)<2:
                    for token in node['ids']:
                        if (*path,token) not in nodes:
                            raise ValueError('missing positive-probability oracle branch')

    def prefill(self, requests):
        samples=[];logprobs=[]
        for req in requests:
            if req.sampling_params.max_new_tokens!=3 or not req.sampling_params.ignore_eos:
                raise ValueError('frozen oracle requires exactly3 outputs and ignore_eos')
            prefix=tuple(req.origin_input_ids)
            if prefix not in self.prefixes:
                raise ValueError('unknown frozen-oracle prefix')
            name=self.prefixes[prefix]['name'];node=self.nodes[name][()]
            index=torch.multinomial(node['_p'],1)
            token=node['_ids'][index].reshape(())
            samples.append(token);logprobs.append(node['_p'][index].log().reshape(()))
            self.names[req.rid]=name;self.histories[req.rid]=[int(token)]
        return torch.stack(samples),torch.stack(logprobs)

    def override(self, p, candidates, requests):
        candidates_cpu=candidates.cpu().tolist()
        for i,(chain,req) in enumerate(zip(candidates_cpu,requests)):
            history=self.histories[req.rid];nodes=self.nodes[self.names[req.rid]]
            if chain[0]!=history[-1]:raise ValueError('oracle anchor/history mismatch')
            for j in range(p.shape[1]):
                path=tuple(history+chain[1:j+1])
                if len(path)>=3:break
                # Impossible proposal histories have zero acceptance upstream;
                # their unused later rows only need a normalized placeholder.
                node=nodes.get(path,nodes[()])
                p[i,j].zero_().scatter_(0,node['_ids'],node['_p'])
        return p

    def advance(self, requests, outputs, commits):
        for req,tokens,commit in zip(requests,outputs.cpu().tolist(),commits.cpu().tolist()):
            history=self.histories[req.rid];nodes=self.nodes[self.names[req.rid]]
            for token in tokens[:min(commit,3-len(history))]:
                if token not in nodes[tuple(history)]['ids']:
                    raise ValueError('oracle emitted an impossible token')
                history.append(token)
            if len(history)==3:
                del self.histories[req.rid];del self.names[req.rid]
