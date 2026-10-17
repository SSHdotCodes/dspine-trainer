> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Paper-to-code checklist: DSpine (arXiv 2609.36173v2) vs `dspine/`

AUDIT-PAPER, 2026-10-17. Sources read in full: `docs/reference/dspine-paper-2609.36173v2.txt`
(Secs. 2-5, App. A, App. B, App. C, Table 4), `docs/reference/tencent-Qwen3-8B-DSpine-b16-config.json`,
`tencent-Qwen3-8B-DSpine-b16-safetensors-header.json`, `tencent-Qwen3-8B-DSpine-b16-README.md`,
`docs/SPEC.md`, `dspine/STATUS.md`, plus all of `modeling.py`, `losses.py`, `train.py`, `data.py`,
`transfer_space.py`, `eval_rounds.py`, `eval_offline.py`, `config.py`, `configs/qwen38-flash-next-full.json`.
No GPU used. Statuses: **MATCH**, **DEVIATION** (with justification judgment), **RISK**, **MISSING**.
Expected acceptance impact is stated per item; tau = average acceptance length.

Live run facts used (brief + `docs/audit/loss/live-run.json`): L=6, hidden 2560, r=1024, a=512,
K=16, block 16, 4 sequences x 256 anchors = 1,024 blocks/step, 55,877 steps, 540M response-token
schedule, 2.25 passes over a 240M-token corpus, peak LR 6e-4, warmup 4% by response tokens,
hot proposal vocab 189,090 (`restricted` loss, row_int8).

---

## A. Architecture (paper Sec. 4.1-4.3, App. A "Architecture"; Table 4)

| # | Paper element | Where in code | Status | Notes / impact if wrong |
|---|---|---|---|---|
| A1 | DFlash-style backbone: L Transformer layers, block-causal attention in **all** layers; position t attends to context K/V + block positions 0..t | `modeling.py:32-46` (`block_causal_mask`), `modeling.py:150-157` (`allowed`, packed), `modeling.py:60-116` (Attention) | MATCH | Packed FlexAttention mask reproduces the reference mask exactly (tests cover parity). If wrong, training/inference diverge; validated by CPU/CUDA parity tests and tencent reproduction. |
| A2 | Draft layers L = 5 (Table 4) | `config.py:11` (`num_hidden_layers=6`), SPEC | DEVIATION (justified) | Our L=6 is a SPEC decision for a 48-layer target. Side effects: (a) `L_emb` sums 6 layers vs paper's 5, so lambda*emb is 1.2x larger (see D4); (b) curriculum ceiling `ceil(L/2)=3` still gives layers 1-3 (matches). Minor positive for capacity, mild negative via D4. |
| A3 | Hidden 4096 (Table 4) | `config.py:10` (2560) | DEVIATION (justified) | Must equal target hidden (shares frozen embedding/LM head); target is 2560. Not optional. |
| A4 | Target feature layers {1,9,17,25,33}; features fused from 5 target layers, supplied as K/V to every layer | `config.py:12` `(1,12,23,34,45)`, `modeling.py:327-330` (`fc` + `hidden_norm` fused once), `modeling.py:118-216` (context K/V per layer) | DEVIATION (justified) | Evenly spaced over our 48 layers, HC-stream mean captured (`capture.py:80-92`). Justified by target arch (dense -> GDN/QSA, hc_count 4). |
| A5 | Shared frozen target input embedding and LM head | `modeling.py:318-323` (embedding frozen, `from_pretrained`), `modeling.py:596+` (`from_pretrained` strict missing=={embedding,lm_head}) | MATCH | Loader fails closed on any other missing tensor. |
| A6 | Block = 1 anchor + 15 candidates (M=16) | `config.py:18` (`block_size=16`); `modeling.py:394-399` (anchor + 15 mask embeddings) | MATCH | Serving uses b<=16 prefixes (SPEC); block-causal makes prefixes consistent. |
| A7 | Injection after **every** layer ℓ, message from immediate predecessor only; anchor unchanged; all messages use pre-update features | `modeling.py:271-285` (`AdjacentBridge.forward`: `pred = [anchor_tok, z_1..z_{B-2}]`, receivers 1..B-1, `u[:, :1]` passthrough) | MATCH | This is the core mechanism; a shift-by-one here would be catastrophic and is covered by tests (injection identity, packed/reference parity). |
| A8 | Eq. 4: `g = sigmoid(G_q z_t + G_p z_{t-1} + b)`; A, G_q, G_p, b shared across layers | `modeling.py:232-236` (`gate` = G_q+b, `predecessor_gate` = G_p, `prev` = A; single shared Linears) | MATCH | |
| A9 | Eq. 5: `h = u + RMS(u) * W_ℓ (g ⊙ m)`, per-layer W_ℓ | `modeling.py:263-269` (`write`: `u + rms * F.linear(msg, feature_write_weights[layer])`, unweighted RMS) | MATCH | |
| A10 | Eq. 4/5 message dims: z is r-dim, message a=512 (Table 4) | `modeling.py:225-234` (code_size 1024, message_size 512) | MATCH | |
| A11 | z_{ℓ,t} = sqrt(r) * normalize(R_ℓ u_{ℓ,t}); known token enters as sqrt(r) C_x | `modeling.py:247-259` (`read`: normalize * `c.scale`), `modeling.py:260-261` (`token`: codebook * scale), `config.py:48-49` (scale = sqrt(1024) = 32) | MATCH | Scale 32 was decisively validated on tencent/Qwen3-8B-DSpine-b16: tau 9.86 (scale 32) vs 7.26 (kappa=10) vs 3.04 (scale 1) (`STATUS.md`, `results/tencent-reproduction-summary.json`). Using kappa=10 would have cost ~2.6 tau. |
| A12 | R_ℓ per layer; separate final-layer readout is fine | `modeling.py:225-228` (`early_read_weights[L-1]` + `read_weight`) | MATCH | Same convention as tencent header (`early_read_weights[4,1024,4096]` + `read_weight[1024,4096]`). |
| A13 | "Injection adds about 33M parameters" (App. A) | Our count: 25.17M for L=6/H=2560; the paper formula with tencent shapes gives 33.03M exactly | MATCH | Confirms the parameterization interpretation (per-layer R_ℓ/W_ℓ, shared A/G_q/G_p/b) is the one tencent shipped. |
| A14 | Mask embedding for the 15 unknown positions | `modeling.py:324-326` learned, init = mean of target embedding rows; tencent freezes target mask row 151669 (`modeling.py:596-601` legacy path) | DEVIATION (minor, not paper-specified) | Reference implementation freezes a target row; ours learns from mean-init. Validated path unaffected. Impact: negligible. |
| A15 | RoPE on draft attention | `modeling.py:49-58`, `config.py:23` theta 1e7 | MATCH | Equals our target's `rope_theta` (target config line 1197). Paper/tencent used 1e6 for Qwen3-8B; matching the *target* is the correct rule. |
| A16 | MLP SwiGLU, intermediate 12288 (tencent) | `config.py:13` (10240) | DEVIATION (justified) | SPEC: 10240 so the qwenfast INT8 decode GEMM needs no padding. Hardware/serving constraint, not a paper violation. |
| A17 | Top-K determined by the ordinary final-layer output (full vocab) | `modeling.py:508-524` (`build_transition_cache`) uses `proposal_weight` (hot head) when configured; `losses.py:147-189` | DEVIATION (see E1) | Paper's candidates come from the full-vocab shared head; ours from the 189,090-token hot head. Justified by production serving alignment; lossless for verification but caps drafter recall. |
| A18 | Last-write refinement: replace z_{L,t-1} with sqrt(r) C_s, redo final-layer write from cached u_{L,t}, final RMSNorm, shared head | `modeling.py:500-506` (`last_write`), `modeling.py:508-524`; training `losses.py:263-264` | MATCH | Receiver's own read is recomputed from u (identical to the forward's z), only the predecessor input changes, matching Eq. 4-5 semantics. |
| A19 | Transition cache T_t[i,j] = S_t(v_{t-1,i}, v_{t,j}), K^2 pairs; receivers 2..B-1 | `modeling.py:514-523` (`transitions` einsum; `u = pre_injection[:, 2:]`, senders `ids[:, :-1]`) | MATCH | Position 1 uses `initial_scores` (backbone top-K), matching "t=1's predecessor is the known anchor". |
| A20 | Block-causal ensures training-injected info cannot flow to earlier positions (Sec. 4.2) | `modeling.py:32-46`, `modeling.py:150-157` | MATCH | Packed mask keys are restricted to own block and pre-anchor context. |

## B. Unified transfer space (Sec. 4.1, App. A "Unified transfer space")

| # | Paper element | Where | Status | Notes / impact |
|---|---|---|---|---|
| B1 | l2-normalize LM-head rows; mu = mean; centered Gram; top-r eigenvectors/values; C_v = normalize(Lambda_r^{-1/2} P_r^T (Ebar_v - mu)) | `transfer_space.py:11-41` | MATCH | Exact: `gram -= n*mu mu^T` (line 24), `basis` = P_r, `values.rsqrt()` = Lambda^{-1/2} (line 34-36), re-normalized rows. Built from the same effective W8 head used in training/serving (`target_head_quantization: qwenfast_w8`), consistent by construction. |
| B2 | r = 1024 (Table 4) | `config.py:16` | MATCH | |
| B3 | C fixed, shared by all layers | `modeling.py:220-221` (registered buffer, not trainable) | MATCH | |
| B4 | Init: R_ℓ = P_r^T; W_ℓ and G_p zero; b = 0, "every injection starts as the identity map" | `modeling.py:225-245` (`zeros_` on `feature_write_weights`, `predecessor_gate.weight`, `gate.bias`; `basis.T` into readouts) | MATCH | Random-QR fallback (`modeling.py:241-243`) only if `basis=None`; `train.py` always passes the PCA basis (`space["basis"]`) when training from scratch (line 404-407) or loads a checkpoint. If the basis were dropped, shallow-layer reads would start noise-dominant; guarded by config checks. |
| B5 | Readout/projection sharing: the same f_ℓ is used for injection features and the L_emb readout (Sec. 4.4 "With the projection f_ℓ used for injection") | `modeling.py:420-425` / `477-486`: `read = bridge.read(h, l)` from **post**-injection states; `losses.py:270-274` uses the same `read` | MATCH | Sharing is exactly what the paper specifies. If someone split them, L_emb would stop supervising the injection pathway. |

## C. Losses (Sec. 4.4-4.5, App. A "Losses")

| # | Paper element | Where | Status | Notes / impact |
|---|---|---|---|---|
| C1 | L_dist = Σ_i w_i [ -alpha log q_i(y_i) + (1-alpha) ||q_i - p_i||_1 ], alpha = 0.1 | `losses.py:268-269`, `alpha=0.1` default (`losses.py:232`), CE = `-log q(y)` in `losses.py:66-74` (`ce = lse - correct`) | MATCH | |
| C2 | w_i = exp(-(t_i - 1)/gamma), gamma = 7 (Table 4) | `losses.py:250-253` (`exp(-arange(15)/7)`), `gamma=7.0` default | MATCH | Weight mass = 6.6306 (checked numerically); first weight 1.0, last exp(-14/7)=0.135. |
| C3 | Reduction: sum over positions within a block, averaged over blocks | `losses.py:254` `reduce = (x*w).sum()/N_blocks` | MATCH (paper's batch reduction unspecified; per-block mean is the natural reading) | Note this is a per-anchor position-weighted **sum** (not a mean over positions); this matters only through lambda (see C6). |
| C4 | Backbone loss over the full vocabulary | `losses.py:147-189` full-vocab branch when `proposal_weight is None`; ours runs `restricted` (hot head) | DEVIATION (E1) | |
| C5 | Refinement output included in L_dist with **equal weight**; distributions restricted to top-K; target renormalized to candidates | `losses.py:269` (equal weight), `losses.py:191-196` (`candidate_targets` + renormalize), `losses.py:198-208` | MATCH | Equal weight = 1.0 confirmed by tencent config `last_write_refine.weight: 1.0`. |
| C6 | Refinement CE only where y in the K=16 candidates | `losses.py:205` (`ce = -(log_q*matches).sum(-1)`, zero unless matched) | MATCH | |
| C7 | Refinement L1 only where candidates hold > 1e-4 of target probability, against the renormalized target | `losses.py:206` (`torch.where(mass > 1e-4, ...)`) | MATCH | |
| C8 | Refinement uses the **correct** predecessor token (as at inference, but ground truth) | `losses.py:263` (`labels[:, :-1]` = y_{t-1}, anchor included) | MATCH | |
| C9 | Refinement covers receivers t>=2 at inference; **training applies it to positions 1..15** | `losses.py:263` uses `pre_injection[:, 1:]` (receivers 1..15) | DEVIATION (paper ambiguous; inference path is t>=2) | At t=1 the predecessor is the anchor, whose embedding is already what the forward pass used, so last-write refinement at t=1 reproduces the backbone output exactly. Effect: position-1 distillation (the largest weight, w=1.0) is double-counted, once full-vocab, once restricted-to-top-K. Mildly over-weights the easiest position. Expected impact on tau: small (<= a few 0.01); fix is to slice receivers 2.. and renormalize... (see proposed fixes). |
| C10 | L_emb = Σ_ℓ Σ_i w_i [1 - cos(e_ℓ,i, C_y_i)], read from **post**-injection states, all L layers | `losses.py:270-274` (sum over `layer_features`, post-injection reads `modeling.py:420-425`), anchor excluded (`[:, 1:]`) | MATCH | Anchor exclusion is correct (anchor is known; paper's i indexes candidate positions). |
| C11 | L = L_dist + lambda L_emb; lambda = beta_t sg(CE_bb); CE_bb = "the token cross-entropy of the backbone output"; beta_t rises linearly 0 -> 0.5 over the first 146 steps (4% of schedule) | `losses.py:278-285` (`coefficient = beta * ce_backbone.detach()`; `beta = 0.5*min(1, step/warmup)`, warmup = 4% of schedule), `losses.py:281` | DEVIATION — the headline one | Our `ce_backbone = reduce(ce)` is the **position-weighted sum over 15 positions** (weight mass 6.63), not a token CE. Live metrics: ce 50.4 -> 37.8 per anchor; lambda -> ~17-19; lambda*emb ~360 vs distillation ~13 (brief). The auxiliary embedding objective dominates the gradient ~28x. If the paper means per-token CE (weighted token mean = ce/6.63 ~ 5.7 -> lambda ~ 2.9 at beta=0.5), lambda*emb would drop ~6.3x; if it means the unweighted token mean (ce/15 ~ 2.5 -> lambda ~ 1.25), ~14x. The `weighted_token_mean` branch already exists (`losses.py:283-285`, `lambda_ce_reduction` run key) and the TRAINER agent owns the A/B (not duplicated here). Expected impact: largest single lever on post-warmup tau; an embedding loss 28x the distillation loss plausibly explains the slow +0.09/2000-step gains (the model is spending most of its gradient budget on cosine alignment). |
| C12 | alpha, gamma, K, r, a coefficients (Table 4) | `config.py:16-17,20`, `losses.py:232-233` | MATCH | 0.1 / 7 / 16 / 1024 / 512 all exact. |
| C13 | L1 term exactness: ||q - p||_1 with the **processed** target p (top-20 + joint-tie support, sums to 1) | `losses.py:46-47` (`l1 = 1 + Σ_S(|q_v - p_v| - q_v)`), exact analytic identity given full-support p; support = stored top-20(+ties) which is the complete processed distribution (validated at capture, `data.py:236-244`) | MATCH (exact given the stored p is the whole processed support) | Paper does not state temperature/processing for training p; ours is the production T1/top-20/top-p-0.95 p (SPEC decision). Truncation error would show up as a biased L1; capture validation makes it exact. |
| C14 | Distillation target y_i = actual on-policy token under the correct prefix | `data.py` labels = sampled output ids; p stored at each position | MATCH | |
| C15 | Teacher-prefill p vs production decode p | training uses prefill-computed p; STATUS measured 10/4078 sampled held-out tokens absent from recomputed prefill p; exact stochastic **eval** refuses prefill p (`eval_rounds.py:180-186`) | DEVIATION (documented, SPEC-sanctioned) | Affects training only, ~0.24% of positions; expected tau impact small but nonzero. |

## D. Curriculum and schedules (App. A "Correct-predecessor curriculum", "Training setup")

| # | Paper element | Where | Status | Notes / impact |
|---|---|---|---|---|
| D1 | Each draft block selected with prob p by **one draw** shared by layers 1-3; selected blocks use sqrt(r) C_{y_{t-1}} as predecessor input; receiver features stay model-derived | `train.py:601` (`teacher = torch.rand(count) < probability`, one draw per block), `modeling.py:276-281` (substitution `pred` only, receiver z untouched, layers `l < ceil(L/2)` = 0,1,2 = paper layers 1-3) | MATCH | Draw is per micro-batch anchor row; shared across the 3 curriculum layers within a block. Matches tencent `teacher_layers: [1,2,3]`, `block_bernoulli`. |
| D2 | p = 0.5 for the first sixth, decays linearly to 0 at one third | `losses.py:210-214` (`curriculum_probability`), `train.py:564` | MATCH | Verified numerically: 0.5 at <= 1/6, 0.25 at 1/4, 0 at 1/3. Matches tencent `hold_fraction: 1/6, zero_fraction: 1/3, peak: 0.5`. |
| D3 | Progress measured in steps (paper) | `train.py:556-562`: progress = response_tokens_seen / 540M | DEVIATION (minor, equivalent under uniform throughput) | Warmup 21.6M tokens, curriculum hold to 90M tokens, zero at 180M. With rolling windows the token rate is near-constant, so fractions match the paper's. Impact: negligible. |
| D4 | beta_t ramp and lambda schedule (see C11) | `losses.py:280-282` | DEVIATION (C11) | Note the interaction: our L=6 makes L_emb 1.2x the paper's per-block sum, amplifying the C11 overshoot by another 20%. |
| D5 | Last-write refinement always uses correct predecessors (no curriculum there) | `losses.py:263` always `labels`; `modeling.py:271-285` substitution only in-layer | MATCH | |
| D6 | AdamW without weight decay | `train.py:328-334` (`weight_decay=0.0`) | MATCH | |
| D7 | Peak LR 6e-4 | `configs/qwen38-flash-next-full.json` `"lr": 0.0006`, `train.py:646` | MATCH numerically, RISK in context | The paper tuned 6e-4 at a 57,344-block global batch; ours runs 1,024-block batches (56x smaller). AdamW is fairly scale-insensitive but not batch-insensitive; with clip 1.0 this is the second-largest training-dynamics risk (see F-tests). |
| D8 | 4% linear warmup (146 steps), cosine decay to zero | `losses.py:216-223` (`lr_multiplier`: linear (step+1)/warmup then cosine ending at 0), `train.py:556-562,646` | MATCH | Ours: 4% of the response-token schedule = 21.6M tokens ~ 2,235 steps of 55,877. Same fraction as the paper's 146/3655. |
| D9 | Gradient-norm clip 1.0 (Table 4) | `train.py:643` (`clip_grad_norm_(parameters, 1.0)`) | MATCH | Nonfinite norm aborts the run (fail-closed). |
| D10 | Global batch 112 sequences | `configs/...full.json`: `batch_size: 4` (1 packed sequence per micro-step), `gradient_accumulation: 4` -> 4 sequences/step | DEVIATION (see E2) | |
| D11 | Anchors per sequence 512 (Table 4) | `"anchors_per_sequence": 256` | DEVIATION (see E2) | |
| D12 | Six epochs (3,655 steps) | `epochs: 2.25` passes over 240M response tokens; 55,877 steps | DEVIATION (see E2) | |
| D13 | Max training sequence length 3,072 | `context_window: 4096` sliding window before the anchor; sequences up to 16K+ served from stored windows | DEVIATION (justified) | Ours must handle much longer production contexts; W=4096 bounds attention cost. Feature capture crops to W before the response (`STATUS.md`). Impact: contexts up to 4K match the paper's effective conditioning range. |
| D14 | Draft blocks sampled from training sequences; anchors within assistant responses | `data.py:545-575` (PackedSequenceStream: response-only anchors, with replacement, sorted for tile sharing), `data.py:444-451` (AnchorStream same) | MATCH | |

## E. Training-setup deviations that set the acceptance ceiling

**E1. Restricted 189,090-token hot proposal head (`DEVIATION`, justified, bounded).**
`modeling.py:234-263` (`configure_proposal_vocabulary`, row_int8 rows of the hot head),
`losses.py:147-189` (`proposal_head_loss`: CE masked where y outside the hot set; L1 exact against
the full processed p including outside mass — verified algebraically), `modeling.py:508-524`
(candidates/inference from the hot head). The paper always scores the full shared vocab. Impact:
(a) candidates outside the hot set can never be proposed — acceptance ceiling loss bounded by the
measured coverage (>= 0.9995 per domain, so < 0.05% of positions); (b) backbone CE is silently
dropped at out-of-hot labels (0 CE contribution) — biases q at exactly the positions where the
drafter cannot help anyway. Justified by production serving alignment (SPEC Sec. 2), and lossless
overall since verification corrects. Expected tau impact: small (< 0.05), but the interaction with
checkpoint selection at b6/b8 is nonzero.

**E2. Data/schedule scale (`DEVIATION`, budget-justified, largest ceiling risk after C11).**
Paper: 3,655 steps x 112 sequences x 512 anchors = 209.6M block presentations (6 epochs).
Ours: 55,877 steps x 1,024 blocks = 57.2M block presentations (2.25 passes over 240M response
tokens) — 27% of the paper's block count, 38% of its epoch count. The paper's Table 3 ablations
were themselves trained 3 epochs; our 2.25 is below even that. Combined with D7 (peak LR tuned for a
56x larger batch), this is the most likely structural reason the run sits at tau 1.57 with MTP
parity at ~2.7 still far away. Expected impact: this alone likely caps tau well below the paper's
4.82 even with a perfect loss.

**E3. Warm-start.** The full run warm-starts from a ~200-step pilot (`init_checkpoint`), so
curriculum/warmup progress is token-consistent (response_tokens_seen restored, `train.py:562,600`).
OK.

## F. Inference / decoding rules (Sec. 4.3, App. C)

| # | Paper rule | Where | Status | Notes / impact |
|---|---|---|---|---|
| F1 | tau = E[N_acc] + 1; bonus/residual token included | `eval_rounds.py:69-76` (`emitted_lengths`: accepted prefix + 1 unless accepted EOS), `eval_rounds.py:72` (`tau = tokens/rounds`) | MATCH | EOS handling: no bonus after an accepted EOS (correct — nothing follows). |
| F2 | Acceptance min(1, p/q) per position, prefix-ordered, first rejection truncates | `eval_offline.py:24-61` (`path_acceptance`), T0 `eval_offline.py:69-76` (`greedy_acceptance`, cumulative product) | MATCH | Monte Carlo uses exact q/p trials on stored on-policy trajectories (Rao-Blackwell alternative available). |
| F3 | Correction distribution [p - q]+ on first rejection (Eq. 11) | `eval_rounds.py` Monte Carlo path: emitted length +1 covers the residual sample; exact residual resampling is a serving-side duty (SGLang) | MATCH (offline estimator) | Offline tau is an unbiased estimate of the same quantity; no code computes the corrected sample itself, which is fine for the metric. |
| F4 | T=1: proposals sampled from softmax of transition-cache scores over the top-16 candidates; rejection with these proposal probabilities | `modeling.py:526-554` (`propose`: softmax over K, position 0 from `initial_scores`), `eval_offline.py:33-35` (q = softmax over candidates) | MATCH | Position-0 proposal from backbone top-K softmax matches "over the top-16 candidates of each position". |
| F5 | T=0: greedy argmax through the cache | `modeling.py:536-538` (argmax), `eval_offline.py:69-76` | MATCH | |
| F6 | Greedy acceptance = candidate matches target's greedy token under the same prefix | `eval_offline.py:69-76` vs stored greedy trajectories | MATCH | |
| F7 | All methods propose 15 draft tokens per round; evaluation complete test sets | Ours evaluates b6/b8 prefixes (serving budget constraint, SPEC) | DEVIATION (justified) | Production kernels cover 1..8 verify rows; b6/b8 is the payable range. Block-causal makes b<=16 exact prefixes of the b16 distribution, so this is measurement, not mechanism. |
| F8 | Selection proceeds through the cache, no sequential matmuls | `modeling.py:526-554` serial over positions but K-way only; `build_transition_cache` precomputes K^2 | MATCH | |

## G. Tencent config / header -> our modules

| Tencent artifact | Value | Our module | Status |
|---|---|---|---|
| `selected_prefix_code_bridge.code_size` | 1024 | `config.code_size` | MATCH |
| `...rank` (message dim) | 512 | `config.message_size`, `prev` [512,1024] | MATCH |
| `...kappa` | 10.0 | not used; `config.feature_scale=None` -> sqrt(r)=32 | OK (settled) | Ablation the real checkpoint: scale32 tau 9.86 vs kappa10 7.26 (`results/tencent-reproduction-summary.json`). Kappa is retained metadata; its training-time meaning is undocumented. Using 10 in training would have been a ~2.6-tau-scale mistake. |
| `selected_prefix_bridge.codebook` | [151936, 1024] | `codebook` buffer [vocab, 1024] | MATCH (shape parity; ours [248320,1024]) |
| `...early_read_weights` | [4, 1024, 4096] | ours [5, 1024, 2560] (L=6) | MATCH (same convention: L-1 early readouts) |
| `...read_weight` | [1024, 4096] | ours [1024, 2560] | MATCH (final-layer readout) |
| `...feature_write_weights` | [5, 4096, 512] | ours [6, 2560, 512] | MATCH (`independent` write mode = per-layer W_ℓ) |
| `...gate.weight/bias` | [512,1024] + [512] | `gate` (G_q + b) | MATCH |
| `...predecessor_gate.weight` | [512, 1024] | `predecessor_gate` (G_p), zero-init | MATCH |
| `...prev.weight` | [512, 1024] | `prev` (A) | MATCH |
| `...out.weight` | [1024, 512], exactly zero | omitted with a fail-closed nonzero guard (`modeling.py:588-599`) | OK | `selected_prefix_terminal: false` in tencent config; the paper has no terminal branch. Loading a nonzero `out` is refused rather than guessed. |
| `selected_prefix_feature_teacher_layers` | [1,2,3] | `layer < ceil(L/2)` = layers 0-2 | MATCH |
| `...teacher_schedule` | peak 0.5, hold 1/6, zero 1/3, `block_bernoulli` | `losses.py:210-214` + `train.py:601` | MATCH |
| `selected_prefix_feature_gt_training` | true | correct-predecessor substitution | MATCH |
| `selected_prefix_last_write_refine` | head_topk 16, exact_sender, weight 1.0 | K=16, correct predecessor, equal weight | MATCH |
| `selected_prefix_feature_layers` | [1,2,3,4,5] | injection after every layer | MATCH |
| `draft_block_causal` | true | block-causal masks | MATCH |
| `selected_prefix_decode_mode` | serial | transition-cache (parallel precompute) | OK (paper Sec. 4.3 describes the cache; tencent ships the serial mode) |
| `target_layer_ids` | [1,9,17,25,33] | [1,12,23,34,45] | DEVIATION (justified, target arch) |
| version scalars (`draft_block_causal_version` etc. = 1) | I64 [] | checked ==1 on load | MATCH |

"Whether our training reproduces the parameters those tensors imply": yes — the parameter census
(B4, A13) reproduces exactly the family tencent shipped (per-layer R_ℓ and W_ℓ, shared A/G_q/G_p/b,
zero-init W/G_p/b, PCA-basis init for readouts), rescaled to L=6/H=2560. The only tencent tensors
without a paper counterpart are the zero `out` (refused-if-nonzero, disabled) and the frozen mask row
(A14).

## H. Data pipeline (App. A "Training setup", SPEC Sec. 4)

| # | Element | Where | Status |
|---|---|---|---|
| H1 | ShareGPT data, responses regenerated by the target | on-policy campaign generations; features = target HC-mean residuals (`capture.py:80-92`) | MATCH in kind (our corpus is domain-balanced production prompts, not ShareGPT; justified — SPEC data contract, and the paper's corpus is not our serving distribution) |
| H2 | Response-only anchors, correct-prefix labels, stored processed p per position | `data.py:545-575`, capture validation `data.py:236-244` (p sums to 1) | MATCH |
| H3 | Tail blocks masked | `data.py:563` (`valid = offsets < length`), `losses.py:247` (`active = valid[:, 1:]`) | MATCH |
| H4 | No training on benchmark/eval prompts | exclusions + 1% validation split (`data.py:81-87`, SPEC 4.4) | MATCH |

## I. Proposed fixes (not applied)

1. **C11 lambda reduction** (owned by TRAINER A/B; do not double-edit). If the A/B confirms
   `weighted_token_mean`, the switch is the existing run key:
   ```json
   "lambda_ce_reduction": "weighted_token_mean"
   ```
   If the A/B instead points to the unweighted token mean (ce/15, not ce/6.63), the divisor should
   be the number of active positions, not the weight mass — a third mode worth including in the A/B:
   `coefficient = coefficient * valid_positions_per_anchor / 15` scale.
2. **C9 position-1 refinement double-count**:
   ```diff
   --- a/dspine/losses.py
   +++ b/dspine/losses.py
   -    refined = model.last_write(output.pre_injection[:, 1:], labels[:, :-1])
   +    refined = model.last_write(output.pre_injection[:, 2:], labels[:, 1:-1])
   ```
   (keeps receivers 2..15, predecessors y_1..y_14, matching inference indexing).
3. **D7 LR risk**: nothing to change blind; decisive <= 20-min GPU test below.

## J. Decisive GPU tests (<= 20 min each, RTX PRO 6000)

1. **Lambda A/B** (already owned by TRAINER): fork at step ~4,500 with
   `lambda_ce_reduction=weighted_token_mean`, 300 steps, compare complete-round tau b6/b8 and the
   distillation/embedding gradient-norm split. Cheap, directly targets C11.
2. **LR-at-small-batch probe**: 250-step fork at peak LR 3e-4 vs 6e-4 (identical data cursor),
   watch distillation term slope and gradient-norm distribution. Targets D7/E2 without touching the
   production run.
3. **Position-1 double-count ablation**: 150-step fork with the I.2 diff; compare position-1 vs
   position-2..6 per-position acceptance (available from `attach_diagnostics` first-round stats).
   Confirms or kills C9 cheaply.

## K. Things checked and found correct that are easy to get subtly wrong (explicitly verified)

- Reductions: per-anchor position-weighted **sum**, mean over blocks only (C3); no accidental mean
  over positions in the distillation terms; `emb` not divided by L (matches the paper's Σ_ℓ).
- One curriculum draw per block shared across layers 1-3 (D1), substitution the predecessor
  input only.
- sqrt(r)=32 on both normalized readouts and unit codebook rows; kappa=10 ruled out by ablation.
- Zero-init of W_ℓ, G_p, b; R_ℓ init from the PCA basis (not the random QR fallback) on the
  training path.
- Readout sharing between injection and L_emb (same `bridge.read`, post-injection states).
- Refinement masking (CE only if y in candidates; L1 only if candidate mass > 1e-4) and target
  renormalization to candidates.
- Transition-cache indexing (receivers 2..B-1; position 0/1 from initial scores) identical in
  training, `propose`, and both eval paths.
- Exact L1 identity against the sparse processed p; exactness of the restricted-head L1 including
  outside mass (checked algebraically).
- Position decay exp(-(t-1)/7) over the 15 candidate positions; anchor excluded from every loss.
- Warmup/cosine by response-token fraction = the paper's 4% fraction; beta ramp to 0.5 over the
  same warmup; clip 1.0; AdamW wd 0.
- Block-causal masks on both reference and packed paths; anchor never updated; context strictly
  pre-anchor, W=4096 sliding window.
