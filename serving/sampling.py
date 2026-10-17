"""Exact sparse-q speculative verification against processed target probabilities.

The proposal distribution may have arbitrary support. Target processors must be
applied first, including penalties, grammar, temperature, top-k and top-p.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


def process_logits(logits, temperature=1.0, top_k=20, top_p=.95):
    """Native FlashInfer joint top-k/top-p reference, including cutoff ties.

    Both filters see the full softmax. A token survives when fewer than k tokens
    have strictly greater probability and their total probability is below p.
    """
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('invalid sampling parameters')
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    scores = logits.float() / temperature
    probs, ids = scores.softmax(-1).sort(dim=-1, descending=True)
    first = torch.cat((torch.ones_like(probs[..., :1], dtype=torch.bool),
                       probs[..., 1:] != probs[..., :-1]), -1)
    ranks = torch.arange(probs.shape[-1], device=probs.device).expand_as(ids)
    group_start = torch.where(first, ranks, 0).cummax(-1).values
    greater_mass = (probs.cumsum(-1) - probs).gather(-1, group_start)
    keep = greater_mass < top_p
    if top_k and top_k < scores.shape[-1]:
        keep &= probs >= probs[..., top_k-1:top_k]
    probs = torch.where(keep, probs, 0.)
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.zeros_like(scores).scatter(-1, ids, probs)


def categorical(probs, uniforms):
    """Inverse-CDF draw; one uniform in [0,1) per probability row."""
    # Last positive support member handles fp32 cumulative-roundoff safely.
    cdf = probs.cumsum(-1)
    index = (uniforms[..., None] >= cdf).sum(-1)
    last = torch.where(probs > 0, torch.arange(probs.shape[-1], device=probs.device), 0).amax(-1)
    return torch.minimum(index, last).long()


@dataclass
class VerifyResult:
    output_ids: torch.Tensor  # [bs,b], only first commit_lens positions are valid
    commit_lens: torch.Tensor # accepted proposals + one corrected/bonus token
    accepted: torch.Tensor


def verify_sparse(proposals, candidate_ids, conditional_probs, target_probs,
                  accept_uniforms=None, correction_uniforms=None):
    """Verify a chain, including a bonus row, with exact sparse proposals.

    ``target_probs[:, t]`` is p(next | accepted prefix, proposals[:t]);
    ``conditional_probs[:, t]`` is exactly the q from which proposal t was drawn.
    Supports greedy target/proposal point masses without a separate approximation.
    Inputs have shapes [N,M], [N,M,K], [N,M,K], and [N,M+1,V].
    This is graph-capture safe for fixed shapes and contains no device/host sync.
    """
    bs, length = proposals.shape
    if target_probs.shape[:2] != (bs, length + 1):
        raise ValueError('verification needs one p row per proposal plus the bonus')
    if candidate_ids.shape != conditional_probs.shape or candidate_ids.shape[:2] != proposals.shape:
        raise ValueError('sparse q layout mismatch')
    device=proposals.device
    if accept_uniforms is None:
        accept_uniforms=torch.rand((bs,length),device=device)
    if correction_uniforms is None:
        correction_uniforms=torch.rand((bs,),device=device)
    q_chosen=torch.where(candidate_ids==proposals[...,None],conditional_probs,0.).sum(-1)
    p_chosen=target_probs[:,:length].gather(-1,proposals[...,None]).squeeze(-1)
    ratio=torch.minimum(torch.ones_like(p_chosen),p_chosen/q_chosen.clamp_min(torch.finfo(torch.float32).tiny))
    # q_chosen=0 means the caller supplied an impossible proposal. Never accept it.
    pass_mask=(accept_uniforms<ratio) & (q_chosen>0)
    prefix=pass_mask.to(torch.int32).cumprod(-1)
    accepted=prefix.sum(-1).long()
    row=torch.arange(bs,device=device)
    p=target_probs[row,accepted].clone()
    if length:
        rejection=accepted.clamp_max(length-1)
        ids=candidate_ids[row,rejection]
        q=conditional_probs[row,rejection].float()
        q=torch.where((accepted<length)[:,None],q,0.)
        # scatter_add handles support collisions defensively. Proposers should use
        # unique top-K ids; duplicate masses would otherwise be under-subtracted.
        delta=torch.zeros_like(p).scatter_add(-1,ids,q)
        residual=(p-delta).clamp_min(0.)
        total=residual.sum(-1,keepdim=True)
        # A rejected event with exactly p==q has probability zero. This fallback
        # only defines that unreachable case under finite precision/user uniforms.
        p=torch.where(total>0,residual/total.clamp_min(1e-30),p)
    corrected=categorical(p,correction_uniforms)
    output=torch.cat((proposals,torch.zeros((bs,1),dtype=torch.long,device=device)),1)
    output.scatter_(1,accepted[:,None],corrected[:,None])
    return VerifyResult(output,accepted.to(torch.int32)+1,accepted.to(torch.int32))


def walk_cache(candidate_ids, initial_scores, transitions, temperature=1.0, uniforms=None):
    """Select a path through precomputed K×K transitions, retaining every q row."""
    bs,length,k=candidate_ids.shape
    device=candidate_ids.device
    if uniforms is None: uniforms=torch.rand((bs,length),device=device)
    if not torch.is_tensor(temperature):
        temperature=torch.full((bs,),float(temperature),device=device)
    greedy=temperature==0
    temperatures=temperature.clamp_min(1e-5)[:,None]
    row=torch.arange(bs,device=device)
    tokens=[]; probabilities=[]; indices=[]
    for t in range(length):
        scores=initial_scores[:,0] if t==0 else transitions[:,t-1][row,indices[-1]]
        probs=(scores.float()/temperatures).softmax(-1)
        choice=categorical(probs,uniforms[:,t])
        choice=torch.where(greedy,scores.argmax(-1),choice)
        probs=torch.where(greedy[:,None],F.one_hot(choice,k).float(),probs)
        indices.append(choice); tokens.append(candidate_ids[:,t][row,choice]); probabilities.append(probs)
    return torch.stack(tokens,1),torch.stack(probabilities,1)
