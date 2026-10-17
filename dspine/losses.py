"""Exact sparse-target L1 and token CE with a recomputed, chunked frozen head.

For sparse p, ||q-p||_1 = 1 + sum_{v in supp(p)} (|q_v-p_v|-q_v).
Only vocabulary chunks, sparse probabilities, and hidden states are retained.
No [N,V] float32 activation is ever allocated. The backward is analytic and
recomputes one chunk at a time. Head/embedding are intentionally frozen.
"""

import math

import torch
from torch.nn import functional as F


def stable_topk(scores, ids, k):
    """Descending finite FP32 score, ascending token id at ties.

    Integer lexicographic keys avoid changing logits/probabilities with epsilon
    perturbations. This makes candidate selection independent of chunk size.
    """
    scores = scores.float().contiguous()
    scores = torch.where(scores == 0, 0.0, scores)  # canonicalize negative zero
    bits = scores.view(torch.int32).long()
    ordered = torch.where(bits < 0, -2147483649 - bits, bits)
    keys = (ordered << 32) | (0xFFFFFFFF - ids.long())
    index = keys.topk(k, -1).indices
    return scores.gather(-1, index), ids.expand_as(scores).gather(-1, index)


class _FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head, labels, target_ids, target_probs, chunk_size, topk):
        if head.requires_grad:
            raise ValueError("chunked loss requires a frozen target head")
        ctx.input_dtype = hidden.dtype
        # Match autocast's BF16 head input and keep backward independent of its context.
        hidden = hidden.to(head.dtype)
        n, vocab = hidden.shape[0], head.shape[0]
        lse = torch.full((n,), -torch.inf, device=hidden.device)
        selected = hidden.new_empty(target_ids.shape, dtype=torch.float32)
        correct = hidden.new_empty((n,), dtype=torch.float32)
        best = hidden.new_empty((n, 0), dtype=torch.float32)
        best_ids = torch.empty((n, 0), dtype=torch.long, device=hidden.device)
        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            logits = F.linear(hidden, head[start:end]).float()
            lse = torch.logaddexp(lse, logits.logsumexp(-1))
            in_chunk = (target_ids >= start) & (target_ids < end)
            selected = torch.where(
                in_chunk,
                logits.gather(1, (target_ids - start).clamp(0, end - start - 1)),
                selected,
            )
            correct = torch.where(
                (labels >= start) & (labels < end),
                logits.gather(
                    1, (labels - start).clamp(0, end - start - 1)[:, None]
                ).squeeze(1),
                correct,
            )
            if topk:
                token_ids = torch.arange(start, end, device=hidden.device).expand(n, -1)
                values, indices = stable_topk(logits, token_ids, min(topk, end - start))
                values = torch.cat((best, values), -1)
                indices = torch.cat((best_ids, indices), -1)
                best, best_ids = stable_topk(
                    values, indices, min(topk, values.shape[-1])
                )
        sparse_q = (selected - lse[:, None]).exp()
        a = (sparse_q - target_probs).sign() - 1
        a_q = (a * sparse_q).sum(-1)
        ce = lse - correct
        l1 = 1 + ((sparse_q - target_probs).abs() - sparse_q).sum(-1)
        ctx.save_for_backward(hidden, head, labels, target_ids, lse, a, a_q)
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(best, best_ids)
        return ce, l1, best, best_ids

    @staticmethod
    def backward(ctx, gce, gl1, _gbest, _gids):
        hidden, head, labels, ids, lse, a, a_q = ctx.saved_tensors
        gce = torch.zeros_like(lse) if gce is None else gce
        gl1 = torch.zeros_like(lse) if gl1 is None else gl1
        dh = torch.zeros_like(hidden, dtype=torch.float32)
        vocab = head.shape[0]
        for start in range(0, vocab, ctx.chunk_size):
            end = min(start + ctx.chunk_size, vocab)
            logits = F.linear(hidden, head[start:end]).float()
            q = (logits - lse[:, None]).exp()
            grad = q * (gce - gl1 * a_q)[:, None]
            correction = torch.zeros_like(q)
            mask = (ids >= start) & (ids < end)
            local = (ids - start).clamp(0, end - start - 1)
            correction.scatter_add_(1, local, torch.where(mask, a, 0))
            grad += gl1[:, None] * q * correction
            ymask = (labels >= start) & (labels < end)
            grad.scatter_add_(
                1,
                (labels - start).clamp(0, end - start - 1)[:, None],
                (-gce * ymask)[:, None],
            )
            dh += (grad.to(head.dtype) @ head[start:end]).float()
        return dh.to(ctx.input_dtype), None, None, None, None, None, None


def chunked_head_loss(
    hidden, head, labels, target_ids, target_probs, chunk_size=4096, topk=0
):
    shape = hidden.shape[:-1]
    ce, l1, scores, ids = _FrozenHeadLoss.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        head,
        labels.reshape(-1),
        target_ids.reshape(-1, target_ids.shape[-1]),
        target_probs.float().reshape(-1, target_probs.shape[-1]),
        chunk_size,
        topk,
    )
    if topk:
        return (
            ce.reshape(shape),
            l1.reshape(shape),
            scores.reshape(*shape, topk),
            ids.reshape(*shape, topk),
        )
    return ce.reshape(shape), l1.reshape(shape)


@torch.no_grad()
def chunked_topk(hidden, head, k=16, chunk_size=4096):
    values = hidden.new_empty((*hidden.shape[:-1], 0), dtype=torch.float32)
    indices = torch.empty(
        (*hidden.shape[:-1], 0), dtype=torch.long, device=hidden.device
    )
    for start in range(0, head.shape[0], chunk_size):
        scores = F.linear(hidden, head[start : start + chunk_size]).float()
        ids = torch.arange(
            start, start + scores.shape[-1], device=hidden.device
        ).expand_as(scores)
        v, i = stable_topk(scores, ids, min(k, scores.shape[-1]))
        v = torch.cat((values, v), -1)
        i = torch.cat((indices, i), -1)
        values, indices = stable_topk(v, i, min(k, v.shape[-1]))
    return values, indices


def proposal_head_loss(
    model, hidden, labels, target_ids, target_probs, chunk_size=4096
):
    """Full-vocabulary default or the configured hot-vocabulary draft objective.

    For restricted q, CE is defined only for y in H. L1 stays EXACT against the
    original full-vocabulary p: the outside mass is included, not renormalized.
    Candidate IDs always use the original tokenizer's global vocabulary.
    """
    if model.proposal_weight is None:
        return chunked_head_loss(
            hidden,
            model.lm_head.weight,
            labels,
            target_ids,
            target_probs,
            chunk_size,
            min(model.config.candidate_count, model.config.vocab_size),
        )
    head = model.proposal_weight
    k = min(model.config.candidate_count, len(head))
    if model.proposal_policy["loss"] == "full_backbone":
        ce, l1 = chunked_head_loss(
            hidden, model.lm_head.weight, labels, target_ids, target_probs, chunk_size
        )
        scores, local_candidates = chunked_topk(hidden, head, k, chunk_size)
    else:
        local_y = model.proposal_lookup[labels]
        local_p = model.proposal_lookup[target_ids]
        supported_probs = target_probs * (local_p >= 0)
        ce, l1, scores, local_candidates = chunked_head_loss(
            hidden,
            head,
            local_y.clamp_min(0),
            local_p.clamp_min(0),
            supported_probs,
            chunk_size,
            k,
        )
        ce = ce * (local_y >= 0)
        l1 = l1 + (target_probs * (local_p < 0)).sum(-1)
    return ce, l1, scores, model.proposal_token_ids[local_candidates]


def candidate_targets(candidate_ids, target_ids, target_probs):
    return (
        (candidate_ids[..., :, None] == target_ids[..., None, :])
        * target_probs[..., None, :]
    ).sum(-1)


def refinement_loss(logits, candidate_ids, labels, target_ids, target_probs):
    p = candidate_targets(candidate_ids, target_ids, target_probs)
    mass = p.sum(-1)
    p = p / mass.clamp_min(1e-30)[..., None]
    log_q = logits.float().log_softmax(-1)
    q = log_q.exp()
    matches = candidate_ids == labels[..., None]
    ce = -(log_q * matches).sum(-1)  # zero unless y belongs to candidates
    l1 = torch.where(mass > 1e-4, (q - p).abs().sum(-1), 0.0)
    return ce, l1


def curriculum_probability(progress):
    if progress <= 1 / 6:
        return 0.5
    return max(0.0, 0.5 * (1 / 3 - progress) / (1 / 6))


def lr_multiplier(step, total_steps, warmup_fraction=0.04):
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (
        1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup)))
    )


def training_loss(
    model,
    output,
    labels,
    target_ids,
    target_probs,
    step,
    total_steps,
    valid=None,
    alpha=0.1,
    gamma=7.0,
    chunk_size=4096,
    lambda_ce_reduction="position_sum",
    refinement_start_position=1,
    mask_unsupported_ce=False,
    embedding_loss_scale=1.0,
    pos1_weight=1.0,
):
    """labels [N,B] includes anchor. Sparse p at index t predicts labels[:,t].

    Sums over the block and over layers, averages across real anchors (not valid
    token positions or padded slot capacity). ``valid[:,0]`` identifies real
    anchors; false rows receive zero weight everywhere. Tail blocks are masked.
    Both distillation terms have equal weight, as in Appendix A.
    """
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    active = valid[:, 1:] & valid[:, :1]
    # valid[:,0] identifies real anchors, not the number of valid tail tokens.
    # A fixed-size padded batch must equal the unpadded real-anchor objective.
    anchor_count = valid[:, 0].sum().clamp_min(1)
    if lambda_ce_reduction not in ("position_sum", "weighted_token_mean"):
        raise ValueError("unsupported lambda CE reduction")
    if refinement_start_position not in (1, 2):
        raise ValueError("refinement must start at block position 1 or 2")
    position_weights = torch.exp(
        -torch.arange(labels.shape[1] - 1, device=labels.device) / gamma
    )
    position_weights[0] *= pos1_weight
    w = position_weights * active
    reduce = lambda x: (x * w).sum() / anchor_count
    ce, l1, _, candidates = proposal_head_loss(
        model,
        output.hidden[:, 1:],
        labels[:, 1:],
        target_ids[:, 1:],
        target_probs[:, 1:],
        chunk_size,
    )
    supported = ((target_ids[:, 1:] == labels[:, 1:, None]) * target_probs[:, 1:]).sum(
        -1
    ) > 0
    if mask_unsupported_ce:
        # A sampled label outside the stored processed distribution is an
        # inconsistent hard target. Preserve the distributional L1 supervision.
        ce = ce * supported
    refined = model.last_write(output.pre_injection[:, 1:], labels[:, :-1])
    logits = torch.einsum("nth,ntkh->ntk", refined, model.proposal_rows(candidates))
    rce, rl1 = refinement_loss(
        logits, candidates, labels[:, 1:], target_ids[:, 1:], target_probs[:, 1:]
    )
    if mask_unsupported_ce:
        rce = rce * supported
    backbone = reduce(alpha * ce + (1 - alpha) * l1)
    refinement_terms = alpha * rce + (1 - alpha) * rl1
    # Inference takes t=1 directly from backbone scores: the known anchor was
    # already injected. The transition cache refines only positions t>=2.
    # Leave remaining position weights unchanged; no extra renormalization.
    refinement = (
        refinement_terms[:, refinement_start_position - 1 :]
        * w[:, refinement_start_position - 1 :]
    ).sum() / anchor_count
    target_code = model.selected_prefix_bridge.codebook[labels[:, 1:]].float()
    emb = sum(
        reduce(1 - F.cosine_similarity(read[:, 1:].float(), target_code, dim=-1))
        for read in output.layer_features
    )
    # Preserve the historical objective by default. The explicit A/B alternative
    # removes the nominal block's weight mass ONLY from lambda. Tail masks and
    # every distillation/embedding reduction remain unchanged in both branches.
    ce_backbone = reduce(ce)
    weight_mass = position_weights.sum()
    warmup = max(1, round(total_steps * 0.04))
    beta = 0.5 * min(1.0, step / warmup)
    coefficient = beta * ce_backbone.detach()
    if lambda_ce_reduction == "weighted_token_mean":
        coefficient = coefficient / weight_mass
    # Opt-in recovery ablation: leave the trained bridge active/trainable and
    # scale only its auxiliary supervision, after the nominal CE normalizer.
    coefficient = coefficient * embedding_loss_scale
    total = backbone + refinement + coefficient * emb
    metrics = {
        "loss": total.detach(),
        "ce": ce_backbone.detach(),
        "l1": reduce(l1).detach(),
        "backbone": backbone.detach(),
        "refinement": refinement.detach(),
        "embedding": emb.detach(),
        "lambda": coefficient.detach(),
        "ce_token_mean": (ce_backbone / weight_mass).detach(),
        "lambda_ce_normalizer": weight_mass.detach()
        if lambda_ce_reduction == "weighted_token_mean"
        else weight_mass.new_tensor(1.0),
        "auxiliary": (coefficient * emb).detach(),
        "distillation": (backbone + refinement).detach(),
        "unsupported_ce_fraction": ((~supported) & active).sum().detach()
        / active.sum().clamp_min(1),
    }
    return total, metrics
