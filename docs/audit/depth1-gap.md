> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Why is DSpine's position-1 quality far below the target's MTP? (AUDIT-MODEL-EVAL, 2026-10-17)

CPU-only study. No GPU, no edits to `dspine/`. Scratch code and raw JSON are in `docs/audit/depth1/`.
Target facts come from the SGLang 0.5.21 sources on the workstation (`ref/sglang/srt/models/qwen4_exp.py`,
`qwen4_exp_mtp.py`, `layers/hyperconnection.py`) and from safetensors headers and small tensor reads.
Probe data: 200K response positions from 971 window sequences of `features/rolling-train/shards`
(teacher p), extracted on the workstation with `nice` and copied to the Mac. The two extractions peaked at
3.4 GB and 2.2 GB RSS, above the 2 GB guideline for ~3 s each; host RAM had 106 GB available.
Held-out evaluation uses the same 3,072 positions as `score_checkpoint.py`'s `proposal_accept_t1`:
64 pilot-eval T1 prompts, anchors at response offsets 1..48, actual decode p.

## Answer in one paragraph

Position 1 is weak mainly because the drafter **under-uses the inputs it already has**, not because
those inputs are too poor. A 2-layer MLP probe that sees only the *last* context position (stored
HC-mean after layer 45) plus the anchor embedding, trained on CPU for 4 minutes on 200K examples, beats
the 6-layer drafter on held-out depth-1 overlap: 0.461 vs 0.430 at full context, and 0.406 at W=1.
The drafter lineage has trained on ~5.5M blocks. The probe's held-out quality is still rising steeply
with data (+0.055 per doubling), so these inputs support at least ~0.5. The inputs are also
degraded relative to MTP's, and that sets the long-run ceiling:
- MTP reads the target's full 4-stream hyper-connection state after the **last** layer (47).
- We store the plain mean of the raw streams after layer 45. That drops layers 46 (GDN) and 47 (QSA),
  three quarters of the state's dimensions, and the per-stream normalization and gating that both
  the LM head and MTP apply.
- Even for the target's own next token, a learned linear readout of our f45 recovers only ~0.79 of
  the head's distribution.
- Of the five captured layers, the drafter effectively uses only layer 45: it supplies 97.5% of the
  fused-context energy because the raw HC-mean RMS grows 58× from layer 1 to layer 45.

## 1. What MTP consumes (exact)

`Qwen4ExpModel.forward` (`qwen4_exp.py:1719-1789`) returns two things after the last decoder layer:
the head input `hyper_connection_mixer.mix(H)` and the raw state `hc_hidden_states = H`, which is
4×2560 = 10,240 dims. The MTP draft model (`qwen4_exp_mtp.py:104-114`, `_fuse_residual_linear_shared`)
then does:

```
e   = fc_embedding( GemmaRMSNorm_2560(embed(x_{t+1})) )          # next token, shared target embedding
Hn  = GemmaRMSNorm_10240(H_47)                                     # ONE joint RMS over all 4 streams, learned (1+w)
in_k = fc_hidden(Hn_k) + e       for each stream k = 0..3          # shared 2560x2560, streams kept separate
-> 1 decoder layer: full attention (24 gated heads x 256, 2 KV heads, q/k norm, RoPE 1e7; indexer weights present)
   + MoE 512 experts top-10 (640 intermediate) + shared expert, each with GatedResidual HC mix/combine
-> own hyper_connection_mixer (per-stream GemmaRMSNorm + low-rank sigmoid gates) -> shared target LM head
```

MTP's attention runs over its own KV of all earlier positions, which draft-extend computes during
prefill. `model_mtp.safetensors` is 5.21 GB BF16. Of that, experts are 5.03 GB (2.5B parameters) and
attention ~0.1B; the active compute per token is ~0.15B. MTP was trained by the model developers
jointly with the target at pretraining scale. Config: `mtp_num_hidden_layers 1`,
`mtp_use_hidden_state_from_layer None` (the final layer), `mtp_use_dedicated_embeddings False`.

How the LM head reads the same state (`hyperconnection.py:184-200`, `hc_per_branch_norm=True`):
`mean_k( sigmoid(up(silu(down(Hn)/4)))_k ⊙ RMSNorm_k(H_k)·(1+w_k) )`. Each stream is normalized
separately, then gated element-wise as a function of the input.

## 2. What our drafter receives, and what is missing or degraded

`dspine/capture.py:73-92` stores `H_l.reshape(4,2560).mean(streams)` of the raw streams after layers
[1,12,23,34,45], as FP8. The drafter sees `hidden_norm(fc(concat))` as context K/V, plus the anchor
embedding at block position 0.

| aspect | MTP | ours | evidence |
|---|---|---|---|
| depth of last state | after layer 47 (last) | after layer 45 | tuned-lens readout of the target's own next token at 100K examples: f34 0.287, f45 0.771; the last layers do most of the work |
| stream structure | all 4 streams, 10,240 dims | raw mean, 2,560 dims | consumers weight streams very differently (below); a fixed mean cannot serve them all |
| normalization/gating | joint RMS (MTP) / per-stream RMS + gates (head) | none (raw magnitudes) | per-stream gains: head mixer 2.4/3.6/5.0/4.0; MTP `pre_fc_norm_hidden` 1.02/**0.26**/0.75/0.66; layer-46 input 0.86/1.46/1.45/1.35 |
| next-token embedding | yes, `fc_embedding(norm(emb))` added to every stream | yes, as block position 0 | — |
| multi-layer context | no (final state only) | 5 layers, but **layers 1-34 are muted** | raw RMS per layer 0.008/0.014/0.025/0.054/0.461; share of `fc` output energy 0.1/0.2/0.6/1.6/**97.5%** (step 5543 and 4516 alike); `fc` blocks still ~uniform (0.033-0.040) |
| capacity | 1 layer, 2.5B-parameter MoE, ~0.15B active | 6 dense layers, ~0.57B, all active | — |
| training | pretraining scale, jointly with the target | ~6M blocks so far | — |

The orchestrator asked which HC combination best predicts x_{j+2}. Only the stream mean is
captured, so other combinations (per-stream normalized, mixer-gated, full streams) **cannot be
reconstructed** from stored data. Section 5 gives the capture needed to measure them.

## 3. Logit lens and probes (held-out, the 3,072 `proposal_accept_t1` positions)

Overlap = Σ_v min(p,q). acc16 = Σ min(p, q renormalized to its top-16), comparable to the
drafter's `proposal_accept_t1`. Probes predict over the 48-65K-token support of all stored target
distributions, through the frozen production W8 head (`probes.py`; scaling in `probes-scaling.json`).

Depth 0, the target's own next token x_{j+1} from f_j (an information check on the features):

| readout | overlap | top-1 |
|---|---|---|
| zero-shot lens, f45 (`lens_by_layer.py`) | 0.585 | 0.574 |
| zero-shot lens, f1/f12/f23/f34 | ≤0.009 | ≤0.03 |
| tuned lens (affine), f34, 100K | 0.287 | 0.315 |
| tuned lens (affine), f45, 100K / 200K | 0.771 / **0.785** | 0.730 / 0.743 |
| the head's real input (mixer of H_47) | 1.0 by definition | — |

Depth 1, x_{j+2} from f_j and the known x_{j+1} (DSpine position 1 and MTP's task):

| predictor | train examples | overlap | acc16 |
|---|---|---|---|
| MLP on emb(x_{j+1}) only (bigram) | 100K | 0.067 | 0.090 |
| MLP f34 + emb | 100K | 0.200 | 0.215 |
| linear all-5 + emb | 100K | 0.331 | 0.351 |
| MLP all-5 + emb | 100K / 200K | 0.339 / 0.391 | 0.359 / 0.412 |
| MLP f45 + emb | 25K / 50K / 100K / 200K | 0.181 / 0.326 / 0.406 / **0.461** | 0.207 / 0.344 / 0.424 / **0.479** |
| **drafter C-lr3 step 5543, W=1** (same inputs as the probe) | ~5.5M blocks | 0.406 | 0.418 |
| drafter C-lr3 step 5543, W=4096 | ~5.5M blocks | 0.430 | 0.442 |
| drafter parent step 4516, W=4096 | ~4.6M blocks | 0.392 | 0.407 |
| target MTP (production, 3 drafts, tau 2.25) | pretraining | — | ~0.62-0.68 |

Loss form is not the cause (`probes-loss.json`, MLP f45 + emb, 100K, 4 epochs):

| probe loss | overlap | acc16 |
|---|---|---|
| soft CE to p (KL) | 0.406 | 0.424 |
| DSpine/paper eq. 7: 0.1·CE(sampled y) + 0.9·‖q−p‖₁ | 0.401 | 0.410 |
| ‖q−p‖₁ only | 0.093 | 0.093 |

The paper's mix learns as well as KL, because its small CE term does the learning. L1 alone barely
moves from scratch. The drafter has the CE term, so its gap to the probe is not the loss form.

## 4. Capacity, data or input?

- **Not input-limited yet.** From last-position f45 + anchor alone, a small probe goes past the
  drafter at 200K examples and is still climbing (0.18 → 0.33 → 0.41 → 0.46 per doubling of data).
- **Not capacity-limited.** The drafter has far more compute than the probe, and context.
- **It is optimization/objective-limited.** The drafter has ~30× more examples than the probe yet
  sits at the probe's 100K level. At W=1 it is exactly the 100K probe (0.406). The loss form is
  ruled out (section 3). What remains:
  - *access*: position 1 sees f_{a-1} only through attention to context K/V (8 KV heads × 128 per
    layer), and the drafter barely attends to context at all (W8/W4096 ratio 0.97);
  - *budget*: position 1 is 1/6.63 of `L_dist`. The step is dominated by λ·L_emb and clipped from
    norm ~16 to 1. Half the blocks are still teacher-forced at layers 1-3.
- **Inputs set the long-run ceiling.** Even at depth 0, f45 gives only ~0.79 linearly. The final
  state, which MTP uses, is complete by construction. The depth trend (f34 → f45: depth-1 probe
  0.20 → 0.41, depth-0 0.29 → 0.79) says layers 46-47 are worth having.
- Shallow layers do not help a probe at this scale: all-5 is worse than f45 alone at 100K and
  200K. So un-muting layers 1-34 is not the first lever.

## 5. Changes, ranked by expected gain, that can be adopted mid-run

Gains are for position-1 acceptance (acc16, held-out) unless stated. "No capture" means it works
with the current windows, so it can switch at any checkpoint.

**1. Direct last-state fusion into the block inputs** (MTP-style; no capture; +6.6M parameters;
negligible compute).
- Add `x_t += W_f · RMSNorm(f45_{a-1})` for block positions t ≥ 1, optionally also t = 0. W_f is
  2560×2560 and **zero-initialized**, so the function is unchanged at the switch, and it gets fresh
  AdamW state. An optional gate or a second matrix for the anchor embedding mirrors MTP's
  `fc_embedding`/`fc_hidden`.
- Rationale: the probe that sees f45_{a-1} and the anchor directly reaches 0.46 at 200K examples
  and is still climbing. The drafter must route the same vector through attention and reaches only
  0.43 at ~5.5M.
- Expected: +0.03 to +0.10 acc16. The lower end is the 200K probe level, which the drafter should
  meet quickly; the upper end extrapolates the probe's data scaling.
- Changes the parameter set, so serving needs the same addition: one GEMV per round.
- Decisive test: fork the current checkpoint into control vs fusion, 500 steps each (~11 min per arm
  at 1.33 s/step), and score with `score_checkpoint.py`.

**2. MTP-first hybrid at serving** (no training; serving engineering; `hybrid_estimate.py`).
- The target's own MTP proposes x_1. DSpine proposes x_2..x_{b-1}, with position 2 last-written from
  the actual x_1, which is one extra K-row even when x_1 is outside DSpine's top-16.
- Exact stored-path pieces on today's C-lr3 step 5543 (b8, offsets 0..56, 3,578 anchors):
  - DSpine alone: fixed-offset RB tau **1.84** (complete-round 1.68);
  - continuation after an accepted position 1: E[C] = 1.65;
  - correlation factor ρ = 1.15;
  - with MTP at position 1 (a1 = 0.62/0.65/0.68): RB tau **2.02-2.18 / 2.07-2.24 / 2.12-2.29**
    (independent vs DSpine-like correlation), i.e. +0.2 to +0.45;
  - oracle position 1: 2.65.
- It does **not** beat production MTP (2.25) by itself until DSpine's continuation improves: with
  a1 = 0.65 that needs E[C] ≳ 1.9, i.e. 1 + 0.65·E[C] > 2.25. It multiplies every DSpine gain at
  positions 2+.
- Costs:
  - one MTP layer forward per round (production already runs it), which can run concurrently with the
    DSpine forward because their inputs are independent;
  - the target must relay H_47, as it does for MTP today;
  - offline eval needs MTP's q at position 1: capture MTP logits for the 64+64 eval prompts (minutes).

**3. Capture the final 4-stream state H_47 (MTP's exact input) as an extra feature** (raises the
ceiling; capture change at a window boundary).
- Capture-overlay change: also store `hc_hidden_states` after layer 47. Feed it either through new
  **zero-initialized** `fc` columns or, better, into the fusion path of (1).
- With H_47, the fusion path can be **initialized from MTP's own input fusion**: `pre_fc_norm_hidden`
  (joint norm) and the shared per-stream `fc_hidden`, averaged over streams. That is the one
  well-defined way to initialize part of the drafter from the MTP layer.
- Expected: depth-0 information goes from ~0.79 to 1.0 by construction (the head reads it). The
  depth trend (f34 → f45: depth-1 probe 0.20 → 0.41) suggests a further +0.05 to +0.15 at depth 1.
  This is unmeasured until the capture test below.
  **Measured 2026-10-17 (`depth1/finalfeat/RESULTS.md`): +0.03 to +0.045, below the +0.05 bar; not adopted.**
- Costs:
  - storage +10,248 B/token, +79% (12,978 → 23,226 B), so a 450 GB window holds 19.4M instead of
    34.7M tokens. That means ~1.8× as many windows: ≈ +1 GPU-hour of capture start-ups over the run.
    Prefill compute per token is unchanged.
  - fc/data-loader cost < 2% of the step;
  - re-capture of the pilot-eval T1/T0 and audit-long sets (~10 min).
- Cheaper variant: the mixer output only (+2,562 B, +20%). It gives exact depth-0 information but
  not MTP's stream structure.
- Decisive capture test first: one capture-server session storing H_45 (4 streams), H_47 (4 streams)
  and the mixer output for ~200 training sequences (~0.5M tokens, ~6 GB, ~10 min including start-up).
  Then rerun `probes.py` on CPU at equal data: mean45 (0.461 at 200K) vs mixer47 vs streams47 vs
  streams45. Adopt (3) if depth-1 gains ≥0.05.

**4. Re-balance the position-1 budget** (no capture, trivial compute, any checkpoint).
- Lower γ or add an extra weight on t = 1, which is 15% of `L_dist` today.
- Let the curriculum p decay as scheduled; it already ends at 1/3.
- Revisit λ: per the round-2 A/B it was neutral short-term, but λ·L_emb still dominates the
  clipped gradient.
- Expected: small to moderate. Test it in the same fork as (1).

**5. Un-mute layers 1-34** (exact reparametrization; no capture).
- Scale each captured layer's input by a fixed s_l = 0.461/median-RMS_l and divide its `fc`
  columns by s_l. The function is identical at the switch, and Adam then moves shallow contributions
  at a comparable rate. Equivalently, use per-block LR multipliers.
- Expected: low. The all-5 probe is *worse* than f45 alone at 100K and 200K. Tencent's trained `fc`
  does weight its early layers up (block rms 0.17…0.05).

**6. Not recommended mid-run:** making MTP's 2.5B MoE a drafter layer. That forward would read up
to 160 experts' weights (~1.6 GB) for 16 positions and does not fit our dense layers.

Decisive GPU budget for the top items: fork arms for (1) and (4) at ~11 min each, the capture test
for (3) at ~10 min, and MTP eval logits for (2) at a few minutes.

## Files
`docs/audit/depth1/`: `extract_probe_data.py`, `probes.py` → `probes.json`, `probes-scaling.json`,
`probes-loss.json`; `lens_by_layer.py` → `lens-by-layer.json`; `feature_scales.py`; `hybrid_estimate.py`
→ `hybrid-estimate-c5543.json`; `mtp-header.txt`.
