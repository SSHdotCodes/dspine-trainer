"""Numerical DSpine reference. All positions include the known anchor at index 0.

Context features exclude the anchor, use absolute RoPE positions, and are fused
once, unchanged across layers. Only block states are transformed by the decoder.
``reference=True`` uses explicit attention; the training path uses masked SDPA.
No target-model dependency is required: pass its frozen embedding/head weights.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import DSpineConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        z = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return self.weight.to(x.dtype) * z.to(x.dtype)


def block_causal_mask(context_valid, block_size):
    """SDPA boolean mask [batch,1,B,W+B], True means visible, left padding allowed."""
    batch, width = context_valid.shape
    causal = torch.ones(
        block_size, block_size, dtype=torch.bool, device=context_valid.device
    ).tril()
    return torch.cat(
        (
            context_valid[:, None, :].expand(batch, block_size, width),
            causal.expand(batch, -1, -1),
        ),
        -1,
    )[:, None]


def rotary(x, positions, theta):
    d = x.shape[-1]
    freq = 1.0 / theta ** (
        torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d
    )
    angle = positions.float()[..., None] * freq
    angle = torch.cat((angle, angle), -1)[:, None]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    half = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), -1)
    return x * cos + half * sin


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(
            c.hidden_size, c.num_attention_heads * c.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            c.num_attention_heads * c.head_dim, c.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def forward(
        self, x, context, context_positions, block_positions, mask, reference=False
    ):
        c = self.c
        batch, length = x.shape[:2]
        kv = torch.cat((context, x), 1)
        q = self.q_norm(
            self.q_proj(x).view(batch, length, c.num_attention_heads, c.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(kv).view(batch, -1, c.num_key_value_heads, c.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(kv)
            .view(batch, -1, c.num_key_value_heads, c.head_dim)
            .transpose(1, 2)
        )
        q = rotary(q, block_positions, c.rope_theta)
        k = rotary(k, torch.cat((context_positions, block_positions), 1), c.rope_theta)
        groups = c.num_attention_heads // c.num_key_value_heads
        if reference or not x.is_cuda:
            k = k.repeat_interleave(groups, 1)
            v = v.repeat_interleave(groups, 1)
            if reference:
                scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(
                    c.head_dim
                )
                out = (
                    scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.float()
                ).to(x.dtype)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True
            )
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))

    def forward_packed(
        self, x, context, anchor_offsets, sequence_start=0, reference=False
    ):
        """Share context K/V across many independent anchors from one sequence.

        FlexAttention skips off-window/future/cross-block keys. The CPU numerical
        path constructs the equivalent boolean mask and uses explicit softmax.
        """
        c = self.c
        n, b = x.shape[:2]
        length = n * b
        width = context.shape[1]
        flattened = x.reshape(1, length, c.hidden_size)
        kv = torch.cat((context, flattened), 1)
        q = self.q_norm(
            self.q_proj(flattened).view(1, length, c.num_attention_heads, c.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(kv).view(1, -1, c.num_key_value_heads, c.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(kv)
            .view(1, -1, c.num_key_value_heads, c.head_dim)
            .transpose(1, 2)
        )
        bp = (
            anchor_offsets[:, None] + torch.arange(b, device=x.device) + sequence_start
        ).reshape(1, -1)
        cp = torch.arange(width, device=x.device)[None] + sequence_start
        q = rotary(q, bp, c.rope_theta)
        k = rotary(k, torch.cat((cp, bp), 1), c.rope_theta)

        def allowed(batch, head, qi, ki):
            # Clamp the padded tail used when building FlexAttention block masks.
            a = anchor_offsets[(qi // b).clamp(max=n - 1)]
            in_context = (
                (ki < width) & (ki < a) & (ki >= (a - c.context_window).clamp(min=0))
            )
            local = ki - width
            in_block = (ki >= width) & (local // b == qi // b) & (local % b <= qi % b)
            return (in_context | in_block) & (qi < length) & (ki < width + length)

        if x.is_cuda and not reference:
            from torch.nn.attention.flex_attention import (
                create_block_mask,
                flex_attention,
            )

            global _compiled_flex_attention
            if "_compiled_flex_attention" not in globals():
                _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
            mask = create_block_mask(
                allowed,
                B=1,
                H=None,
                Q_LEN=length,
                KV_LEN=width + length,
                device=x.device,
            )
            out = _compiled_flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
        else:
            qi = torch.arange(length, device=x.device)[:, None]
            ki = torch.arange(width + length, device=x.device)[None, :]
            mask = allowed(0, 0, qi, ki)[None, None]
            groups = c.num_attention_heads // c.num_key_value_heads
            k = k.repeat_interleave(groups, 1)
            v = v.repeat_interleave(groups, 1)
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(c.head_dim)
            out = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.float()).to(
                x.dtype
            )
        return self.o_proj(out.transpose(1, 2).reshape(n, b, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Attention(c)
        self.mlp = MLP(c)

    def forward(self, x, context, cp, bp, mask, reference=False):
        x = x + self.self_attn(
            self.input_layernorm(x), context, cp, bp, mask, reference
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class AdjacentBridge(nn.Module):
    """Names/shapes intentionally match Tencent's independent-write checkpoint."""

    def __init__(self, c, codebook, basis=None):
        super().__init__()
        self.c = c
        self.register_buffer("codebook", codebook.detach().clone())
        self.early_read_weights = nn.Parameter(
            torch.empty(c.num_hidden_layers - 1, c.code_size, c.hidden_size)
        )
        self.read_weight = nn.Parameter(torch.empty(c.code_size, c.hidden_size))
        self.feature_write_weights = nn.Parameter(
            torch.zeros(c.num_hidden_layers, c.hidden_size, c.message_size)
        )
        self.gate = nn.Linear(c.code_size, c.message_size)
        self.predecessor_gate = nn.Linear(c.code_size, c.message_size, bias=False)
        self.prev = nn.Linear(c.code_size, c.message_size, bias=False)
        nn.init.zeros_(self.predecessor_gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            if basis is None:
                basis = torch.linalg.qr(
                    torch.randn(c.hidden_size, c.code_size), mode="reduced"
                ).Q
            if basis.shape != (c.hidden_size, c.code_size):
                raise ValueError("PCA basis must have shape [hidden,code]")
            self.early_read_weights.copy_(basis.T)
            self.read_weight.copy_(basis.T)

    def read(self, u, layer):
        weight = (
            self.read_weight
            if layer == self.c.num_hidden_layers - 1
            else self.early_read_weights[layer]
        )
        return (
            F.normalize(
                F.linear(u, weight).float(), dim=-1, eps=self.c.rms_norm_eps
            ).to(u.dtype)
            * self.c.scale
        )

    def token(self, ids):
        return F.embedding(ids, self.codebook) * self.c.scale

    def write(self, u, receiver, predecessor, layer):
        gate = torch.sigmoid(self.gate(receiver) + self.predecessor_gate(predecessor))
        msg = gate * self.prev(predecessor)
        rms = torch.sqrt(
            u.float().square().mean(-1, keepdim=True) + self.c.rms_norm_eps
        ).to(u.dtype)
        return u + rms * F.linear(msg, self.feature_write_weights[layer])

    def forward(self, u, layer, anchor_ids, correct_ids=None, teacher_blocks=None):
        z = self.read(u, layer)
        pred = torch.cat((self.token(anchor_ids)[:, None], z[:, 1:-1]), 1)
        # For receivers 1..B-1, predecessor 0 is ALWAYS the known anchor.
        if (
            correct_ids is not None
            and teacher_blocks is not None
            and layer < math.ceil(self.c.num_hidden_layers / 2)
        ):
            pred = torch.where(
                teacher_blocks[:, None, None], self.token(correct_ids[:, :-1]), pred
            )
        updated = self.write(u[:, 1:], z[:, 1:], pred, layer)
        return torch.cat((u[:, :1], updated), 1)


@dataclass
class DraftOutput:
    hidden: torch.Tensor  # [N,B,H], final normalized post-injection
    pre_injection: torch.Tensor  # [N,B,H], cached final layer u
    layer_features: tuple[torch.Tensor, ...]  # post-injection readouts, [N,B,r]


@dataclass
class TransitionCache:
    candidate_ids: torch.Tensor  # [N,B-1,K], excludes anchor
    initial_scores: torch.Tensor  # [N,B-1,K], backbone logits
    transitions: torch.Tensor  # [N,B-2,K,K], sender then receiver


@dataclass
class Proposal:
    tokens: torch.Tensor  # [N,B-1], excludes anchor
    probabilities: torch.Tensor  # selected q, [N,B-1]
    candidate_ids: torch.Tensor  # support, [N,B-1,K]
    conditional_probs: torch.Tensor  # full selected-row q, [N,B-1,K]


class DSpineDraft(nn.Module):
    def __init__(self, config, embedding_weight, lm_head_weight, codebook, basis=None):
        super().__init__()
        self.config = config
        c = config
        if (
            embedding_weight.shape != (c.vocab_size, c.hidden_size)
            or lm_head_weight.shape != embedding_weight.shape
        ):
            raise ValueError("target embedding/head shape mismatch")
        if codebook.shape != (c.vocab_size, c.code_size):
            raise ValueError("codebook shape mismatch")
        self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=True)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.mask_embedding = nn.Parameter(
            embedding_weight.float().mean(0).to(embedding_weight.dtype)
        )
        self.fc = nn.Linear(
            len(c.target_layer_ids) * c.target_hidden_size, c.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DecoderLayer(c) for _ in range(c.num_hidden_layers)]
        )
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.selected_prefix_bridge = AdjacentBridge(c, codebook, basis)
        # Runtime policy buffers are reconstructed from the external frozen head.
        # Existing bridge/decoder tensors and DraftOutput remain unchanged.
        self.register_buffer("proposal_token_ids", None, persistent=False)
        self.register_buffer("proposal_weight", None, persistent=False)
        self.register_buffer("proposal_lookup", None, persistent=False)
        self.proposal_policy = None
        # Append after every existing parameter: old optimizer parameter IDs keep
        # their meaning when an explicit fork adds this one trainable tensor.
        self.context_fusion = None
        self.configure_context_fusion(c.context_fusion)

    def configure_context_fusion(self, mode):
        """Fuse the normalized a-1 target row into block inputs, once before L0.

        Anchor mode mirrors MTP's previous-state + known-token input. All mode
        additionally broadcasts that same state to every masked block row.
        Creation preserves RNG, so a zero-init fork keeps paired teacher draws.
        """
        if not isinstance(mode, str) or mode not in {"off", "anchor", "all"}:
            raise ValueError("context_fusion must be off, anchor or all")
        if mode == "off" and self.context_fusion is not None:
            raise ValueError("cannot silently discard an existing fusion weight")
        if mode != "off" and self.context_fusion is None:
            # fork_rng only needs CPU state: nn.Linear is constructed on CPU.
            with torch.random.fork_rng(devices=[]):
                layer = nn.Linear(
                    self.config.hidden_size, self.config.hidden_size, bias=False
                )
            nn.init.zeros_(layer.weight)
            self.context_fusion = layer.to(
                device=self.fc.weight.device, dtype=self.fc.weight.dtype
            )
        self.config.context_fusion = mode

    def _fuse_block_inputs(self, x, previous_context):
        if self.context_fusion is None:
            return x
        delta = self.context_fusion(previous_context).to(x.dtype)[:, None]
        if self.config.context_fusion == "all":
            return x + delta
        return torch.cat((x[:, :1] + delta, x[:, 1:]), dim=1)

    def configure_proposal_vocabulary(
        self, token_ids, quantization="row_int8", loss="restricted"
    ):
        """Optional serving-aligned hot head; default remains the full target head.

        Global-ID ascending order makes ties independent of the input hot-map
        order. No target p or embedding/codebook row is removed. Restricted CE
        is masked on unproposable labels; its L1 includes all outside target mass.
        """
        from .hot_vocab import row_int8_weights

        if loss not in ("restricted", "full_backbone") or quantization not in (
            "row_int8",
            "none",
        ):
            raise ValueError("unsupported proposal vocabulary policy")
        ids = torch.as_tensor(
            token_ids, device=self.lm_head.weight.device, dtype=torch.long
        )
        if ids.ndim != 1 or not len(ids) or len(ids.unique()) != len(ids):
            raise ValueError("proposal token IDs must be nonempty and unique")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("proposal token ID outside vocabulary")
        self.proposal_token_ids = ids.sort().values
        rows = self.lm_head.weight[self.proposal_token_ids].detach()
        self.proposal_weight = (
            row_int8_weights(rows) if quantization == "row_int8" else rows
        )
        self.proposal_lookup = torch.full(
            (self.config.vocab_size,), -1, device=ids.device, dtype=torch.long
        )
        self.proposal_lookup[self.proposal_token_ids] = torch.arange(
            len(ids), device=ids.device
        )
        self.proposal_policy = {
            "version": 1,
            "token_ids": self.proposal_token_ids.cpu().tolist(),
            "quantization": quantization,
            "loss": loss,
        }

    def proposal_rows(self, global_ids):
        if self.proposal_weight is None:
            return F.embedding(global_ids, self.lm_head.weight)
        local = self.proposal_lookup[global_ids]
        if torch.any(local < 0):
            raise ValueError("candidate outside configured proposal vocabulary")
        return F.embedding(local, self.proposal_weight)

    def forward(
        self,
        context_features,
        anchor_ids,
        context_positions=None,
        anchor_positions=None,
        context_valid=None,
        block_size=None,
        correct_ids=None,
        teacher_blocks=None,
        reference=False,
        return_layer_features=True,
    ):
        c = self.config
        n, w = context_features.shape[:2]
        b = block_size or c.block_size
        if w > c.context_window or not 2 <= b <= c.block_size:
            raise ValueError("context/block exceeds configured window")
        device = context_features.device
        if context_positions is None:
            context_positions = torch.arange(w, device=device).expand(n, -1)
        if anchor_positions is None:
            anchor_positions = (
                context_positions[:, -1] + 1
                if w
                else torch.zeros(n, device=device, dtype=torch.long)
            )
        bp = anchor_positions[:, None] + torch.arange(b, device=device)
        if context_valid is None:
            context_valid = torch.ones(n, w, device=device, dtype=torch.bool)
        mask = block_causal_mask(context_valid, b)
        context = self.hidden_norm(self.fc(context_features))
        x = torch.cat(
            (
                self.embedding(anchor_ids)[:, None],
                self.mask_embedding.to(self.embedding.weight.dtype)[None, None].expand(
                    n, b - 1, -1
                ),
            ),
            1,
        )
        if self.context_fusion is not None:
            # Select by absolute position, respecting left padding and validity.
            # No predecessor (including an empty prefix) means a zero shortcut.
            predecessor = context_valid & (
                context_positions == anchor_positions[:, None] - 1
            )
            if w:
                index = predecessor.long().argmax(1)
                previous = context[torch.arange(n, device=device), index]
                previous = torch.where(predecessor.any(1)[:, None], previous, 0)
            else:
                previous = context.new_zeros(n, c.hidden_size)
            x = self._fuse_block_inputs(x, previous)
        reads = []
        for l, layer in enumerate(self.layers):

            def step(x, context, layer=layer, l=l):
                u = layer(x, context, context_positions, bp, mask, reference)
                h = self.selected_prefix_bridge(
                    u, l, anchor_ids, correct_ids, teacher_blocks
                )
                read = (
                    self.selected_prefix_bridge.read(h, l)
                    if return_layer_features
                    else h.new_empty(0)
                )
                return h, u, read

            if c.gradient_checkpointing and self.training:
                x, u, read = checkpoint(step, x, context, use_reentrant=False)
            else:
                x, u, read = step(x, context)
            if return_layer_features:
                reads.append(read)
        return DraftOutput(self.norm(x), u, tuple(reads))

    def forward_packed(
        self,
        sequence_features,
        sequence_ids,
        anchor_offsets,
        sequence_start=0,
        correct_ids=None,
        teacher_blocks=None,
        reference=False,
        return_layer_features=True,
    ):
        """Training path: fuse/project each sequence context only once per layer.

        ``anchor_offsets`` index into the stored sequence; absolute RoPE positions
        add ``sequence_start``. Anchors must be response tokens (enforced by data).
        Public serving/reference ``forward`` and bridge APIs remain unchanged.
        """
        c = self.config
        n = anchor_offsets.numel()
        b = c.block_size
        context = self.hidden_norm(self.fc(sequence_features))[None]
        anchor_ids = sequence_ids[anchor_offsets]
        x = torch.cat(
            (
                self.embedding(anchor_ids)[:, None],
                self.mask_embedding.to(self.embedding.weight.dtype)[None, None].expand(
                    n, b - 1, -1
                ),
            ),
            1,
        )
        if self.context_fusion is not None:
            previous = context[0, (anchor_offsets - 1).clamp_min(0)]
            previous = previous * (anchor_offsets > 0)[:, None]
            x = self._fuse_block_inputs(x, previous)
        reads = []
        for l, layer in enumerate(self.layers):

            def step(x, context, layer=layer, l=l):
                u = x + layer.self_attn.forward_packed(
                    layer.input_layernorm(x),
                    context,
                    anchor_offsets,
                    sequence_start,
                    reference,
                )
                u = u + layer.mlp(layer.post_attention_layernorm(u))
                h = self.selected_prefix_bridge(
                    u, l, anchor_ids, correct_ids, teacher_blocks
                )
                read = (
                    self.selected_prefix_bridge.read(h, l)
                    if return_layer_features
                    else h.new_empty(0)
                )
                return h, u, read

            if c.gradient_checkpointing and self.training:
                x, u, read = checkpoint(step, x, context, use_reentrant=False)
            else:
                x, u, read = step(x, context)
            if return_layer_features:
                reads.append(read)
        return DraftOutput(self.norm(x), u, tuple(reads))

    def last_write(self, u, predecessor_ids):
        """Serial reference, with arbitrary leading dimensions on u and token ids."""
        bridge = self.selected_prefix_bridge
        l = self.config.num_hidden_layers - 1
        return self.norm(
            bridge.write(u, bridge.read(u, l), bridge.token(predecessor_ids), l)
        )

    def build_transition_cache(self, output, candidate_count=None, chunk_size=4096):
        from .losses import chunked_topk

        weight = (
            self.lm_head.weight
            if self.proposal_weight is None
            else self.proposal_weight
        )
        k = min(candidate_count or self.config.candidate_count, len(weight))
        scores, ids = chunked_topk(output.hidden[:, 1:], weight, k, chunk_size)
        if self.proposal_token_ids is not None:
            ids = self.proposal_token_ids[ids]
        # Receivers 2..B-1 by all K candidates of their predecessor.
        u = output.pre_injection[:, 2:, None, :].expand(-1, -1, k, -1)
        refined = self.last_write(u, ids[:, :-1])
        weights = self.proposal_rows(ids[:, 1:])
        transitions = torch.einsum("ntih,ntjh->ntij", refined, weights).float()
        return TransitionCache(ids, scores.float(), transitions)

    @torch.no_grad()
    def propose(self, cache, temperature=1.0, generator=None):
        """K-way q only, no target top-k/top-p processor. T=0 returns point masses."""
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        n, length, k = cache.candidate_ids.shape
        batch = torch.arange(n, device=cache.candidate_ids.device)
        previous = None
        tokens = []
        chosen = []
        distributions = []
        for t in range(length):
            logits = (
                cache.initial_scores[:, 0]
                if t == 0
                else cache.transitions[:, t - 1][batch, previous]
            )
            if temperature == 0:
                index = logits.argmax(-1)
                q = F.one_hot(index, k).float()
            else:
                q = (logits.float() / temperature).softmax(-1)
                index = torch.multinomial(q, 1, generator=generator).squeeze(-1)
            tokens.append(cache.candidate_ids[:, t][batch, index])
            chosen.append(q[batch, index])
            distributions.append(q)
            previous = index
        return Proposal(
            torch.stack(tokens, 1),
            torch.stack(chosen, 1),
            cache.candidate_ids,
            torch.stack(distributions, 1),
        )

    def trainable_state_dict(self):
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith(("embedding.", "lm_head."))
        }

    @classmethod
    def from_pretrained(
        cls,
        path,
        embedding_weight,
        lm_head_weight,
        feature_scale=None,
        trainable_dtype=None,
        context_fusion=None,
    ):
        """Load our format or Tencent independent-write b16. Never silently drop weights."""
        from safetensors.torch import load_file

        path = Path(path)
        c = DSpineConfig.load(path / "config.json")
        if feature_scale is not None:
            c.feature_scale = feature_scale
        if context_fusion is not None:
            c.context_fusion = context_fusion
        state = load_file(str(path / "model.safetensors"))
        legacy = "selected_prefix_bridge.out.weight" in state
        code = state["selected_prefix_bridge.codebook"]
        model = cls(c, embedding_weight, lm_head_weight, code).to(
            dtype=embedding_weight.dtype
        )
        if legacy:
            # Terminal branch disabled by config/version; public checkpoint's out is exactly zero.
            terminal = state.pop("selected_prefix_bridge.out.weight")
            if torch.count_nonzero(terminal):
                raise ValueError(
                    "nonzero legacy terminal branch is unsupported; do not guess its semantics"
                )
            expected_versions = {
                "draft_block_causal_version",
                "last_write_refine_version",
                "selected_prefix_bridge.independent_write_version",
                "selected_prefix_bridge.terminal_disabled_version",
            }
            for name in expected_versions:
                if int(state.pop(name)) != 1:
                    raise ValueError(f"unsupported checkpoint version: {name}")
            state["mask_embedding"] = embedding_weight[c.mask_token_id].clone()
            model.mask_embedding.requires_grad_(False)
        if trainable_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(trainable_dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = {"embedding.weight", "lm_head.weight"}
        # Old checkpoints have no shortcut. An enabled missing weight remains
        # exactly zero; every other missing/unexpected tensor is still an error.
        if model.context_fusion is not None and "context_fusion.weight" not in state:
            expected_missing.add("context_fusion.weight")
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        policy = path / "proposal_policy.json"
        if policy.exists():
            import json

            saved = json.loads(policy.read_text())
            if saved["version"] != 1:
                raise ValueError("unsupported proposal policy version")
            model.configure_proposal_vocabulary(
                saved["token_ids"], saved["quantization"], saved["loss"]
            )
        return model

    def save_pretrained(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.json")
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.trainable_state_dict().items()
            },
            str(path / "model.safetensors"),
        )
        if self.proposal_policy is not None:
            import json

            (path / "proposal_policy.json").write_text(
                json.dumps(self.proposal_policy) + "\n"
            )
        else:
            (path / "proposal_policy.json").unlink(missing_ok=True)
