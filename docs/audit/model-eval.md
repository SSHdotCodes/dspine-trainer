> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# AUDIT-MODEL-EVAL: drafter model and evaluation (2026-10-17)

Scope: `dspine/modeling.py`, `dspine/transfer_space.py`, train/inference parity, and the eval stack
(`eval_rounds.py`, `eval_offline.py`, `audit_tau.py`, the train-loop eval in `train.py:416-540`).
All numbers below are CPU FP32 runs on copies of the step-4516 checkpoint, the 189,090-id row-INT8
proposal policy, the production qwenfast-W8 target head (byte ranges cut from the target safetensors) and
the held-out `pilot-eval/T1` shards, unless labelled otherwise. Scripts and raw JSON are in
`docs/audit/model-eval/`. Nothing in `dspine/` was edited; the the workstation GPU was not used.

## Verdict on the puzzle (opening tau 2.85 vs complete-round tau 1.57)

**The gap is not a bug and not a train/eval mismatch.** The opening round is systematically easy: every
eval response starts `<think>\n` + a stock phrase ("The user wants/is asking…", "We need answer…",
"Let me work through…"). Opening tau is one anchor per prompt (18 anchors, 2 per domain). The
complete-round number is the honest one, and it shows **genuine drafter weakness** at step 4516: the
backbone matches the target's next token (position 1) only ~40% of the time. The model has not yet
learned to use its target-context attention (finding 1: shrinking its window from 4096 to 1 token
costs it 6%; the same ablation costs the public tencent DSpine 64% of its tau).

Exact RB tau by fixed response offset, step 4516, same 18-prompt cohort (`offset-tau-step4516.json`).
Every cohort response is a 64-token length-capped sample, so offsets ≤ 56 are never censored and each
estimate is exact for its anchor:

| response offset k | 0 | 1 | 2 | 3 | 4-7 | 8-15 | 16-31 | 32-56 | all 1-56 |
|---|---|---|---|---|---|---|---|---|---|
| RB tau b6 | **2.839** | 2.051 | 1.667 | 2.058 | 1.939 | 1.594 | 1.624 | 1.581 | 1.639 |
| RB tau b8 | **2.869** | 2.051 | 1.684 | 2.061 | 1.948 | 1.597 | 1.624 | 1.581 | 1.641 |
| position-1 overlap Σmin(p,q) | 0.777 | 0.708 | 0.374 | 0.583 | 0.458 | 0.360 | 0.358 | 0.405 | |

- The complete-round MC walk on CPU reproduces the GPU log exactly: b6/b8 **1.562/1.601**, opening RB
  **2.839/2.869** (GPU step 4516: 1.570/1.590 and 2.838/2.870). BF16-vs-FP32 and packed-vs-serial
  numerics therefore play no part.
- Three other cohorts (`seed` 43/44/45 also reorders `domain_balanced_paths`) give complete-round
  1.574/1.620/1.571 against opening 2.978/2.918/2.695. The gap is robust; it is not one lucky cohort.
- Target p is ≈1.0 on draft positions 1-2 for 14 of the 18 openings (`inspect_openings.py`). The
  run's own long held-out eval (audit-long, 4 sequences, step 4000) is flat across depth: offset
  0-256 1.49, 256-1K 1.50, 1K-4K 1.53, ≥4K 1.57. Its first-round RB tau is 2.38, the same pattern.
- **How the T=1 walk works** (`eval_rounds.py:213-321`, `eval_offline.py:24-65`). For a stored
  on-policy trajectory Y~p it accepts position t with Bernoulli min(1, q_t(Y_t)/p_t(Y_t)). Here q_t is
  the transition row conditioned on the *trajectory's* token Y_{t-1} (`previous = match.argmax`). The
  emitted length is accepted+1 (no bonus after an accepted EOS), and the next anchor is
  Y[anchor+emitted]. That token is the residual or bonus token of the round. In law it equals real
  speculative sampling: P(accept, Y=y)=min(p,q) and P(reject, Y=v)=(p−q)+. The trajectory therefore
  never diverges from the cached one, and the next round's context features are the stored target
  features of exactly the emitted prefix (what the verifier would have produced). It is **not** a
  "must match the cached token" rule. Independent check `test_round_walk_vs_real_spec.py`: Markov
  target, predecessor-conditioned draft through a real `TransitionCache`. Real speculative sampling
  gives tau 1.617, the repo's walk 1.604 (bigram law within 0.01), and the wrong match-the-cached-token
  rule gives 1.323. **PASS.**

## Ranked findings

### 1. RISK, highest impact, confirmed by measurement: the drafter has not learned to use its target context; the cause is training, not eval or a mismatch
Position- and domain-matched comparison, same step-4516 weights (`generalization_gap.py` →
`generalization-gap.json`). 7 common domains, response offsets 1-48, equal domain weight:

| set | overlap t1 | t2 | t3 | weighted ce | weighted l1 |
|---|---|---|---|---|---|
| held-out (decode p), 64 prompts | 0.393 | 0.276 | 0.184 | 43.3 | 11.17 |
| window-0 TRAIN shards (prefill p), 40 seqs | 0.407 | 0.268 | 0.185 | 41.9 | 11.11 |
| same TRAIN shards, random offsets | 0.440 | 0.343 | 0.239 | 39.2 | 10.61 |
| pilot-init warm start (L6 step150), held-out | 0.078 | 0.022 | 0.015 | 54.8 | 12.93 |
| pilot-init warm start, TRAIN offsets 1-48 | 0.075 | 0.020 | 0.015 | 54.0 | 12.95 |

- **No generalization gap.** Train ≈ held-out at matched positions, both before window 0 (warm start)
  and after it, although window 0 is on its 2nd+ pass (`data_epoch: 2` in `metrics.jsonl`). Training
  has raised t1 overlap ~5× since the warm start, so learning works; it is just far from done. The held-out set covers only the first 64 tokens of each
  response, which are slightly harder than random positions. That bias is small (t1 0.39→0.44).
- The trainer's logged `ce` 39.8 / `l1` 9.96 at step 4516 match these per-anchor sums: the
  position-weighted sums with w_t=exp(-(t-1)/7), Σw=6.63. So the logged loss is consistent with
  what eval sees. Mean weighted ||q−p||₁ = 1.5, i.e. the average single-position overlap is ~0.25.
- **Feature alignment is correct** (`feature_alignment.py`). A logit lens on the captured layer-45 HC
  mean, `features[j]`, hits `ids[j+1]` 57.5% of the time on held-out shards and 53.2% on training
  shards, versus `ids[j]` 0.4% and `ids[j-1]` 0.5%. No off-by-one exists in capture, and train and
  eval shards agree.
- Qualitatively, the misses are context content: after "用户问的是" it proposes "在" instead of "瑞典"
  from the prompt; "The edit placed…" draws "issue/bug/fix/PR"; "the modeling" draws "model/system"
  (`longctx_opening.py`). That is what drives the longctx opening RB of ~1.00 at every step.
- **The drafter barely uses its context window** (`context_ablation.py` → `context-ablation-step4516.json`;
  the packed mask's window is shrunk, the same 18 prompts, offsets 1..48 step 2):

  | context window W | 4096 | 512 | 64 | 8 | 1 |
  |---|---|---|---|---|---|
  | overlap t1 / t2 / t3 | 0.396 / 0.254 / 0.159 | 0.395 / 0.255 / 0.160 | 0.393 / 0.255 / 0.157 | 0.385 / 0.241 / 0.140 | 0.373 / 0.226 / 0.106 |
  | weighted ce | 43.05 | 43.09 | 43.29 | 44.56 | 48.04 |

  With one context feature (the target state at the token before the anchor), the drafter keeps 94%
  of its t1 overlap. Everything beyond the last ~64 tokens is worth ≤0.003.
- **The reference DSpine depends on context heavily; ours does not** (`tencent_context_ablation.py` →
  `tencent-context-ablation.json`). The public tencent/Qwen3-8B-DSpine-b16 was run on the cached
  Qwen3-8B GSM8K greedy continuations from `runs/tencent-validation` (16 prompts, 1,127 fixed
  anchors, packed path, greedy b8):

  | context window W | 4096 | 512 | 64 | 8 | 1 |
  |---|---|---|---|---|---|
  | tencent greedy tau b8 | 6.91 | 6.91 | 5.77 | 3.60 | 2.47 |
  | tencent pos-1 accuracy | 0.967 | 0.966 | 0.912 | 0.803 | 0.652 |
  | **ours** t1 overlap (T1, from the table above) | 0.396 | 0.395 | 0.393 | 0.385 | 0.373 |

  A trained DSpine loses 64% of its tau and a third of its position-1 accuracy when it can only see
  the last context feature. Ours loses 6% of t1 overlap. At step 4516 our drafter behaves like a
  weak MTP head on the last few target features; it has not learned the DFlash context attention
  (copying prompt content, using long-range state). **This is where the missing tau is.** Parity tests
  show the attention mask, positions and K/V path are correct, so this is learned behaviour, not a
  masking bug. The pathway is not dead, only young (`context_path_weights.py` →
  `context-path-weights.txt`). `fc` rms has gone from 0.0063 at the warm start to 0.034 now
  (init ≈0.005, tencent 0.123). `layers.0.k_proj` has gone from 0.012 to 0.028 (tencent 0.082). The
  per-target-layer `fc` blocks are still uniform (tencent weights them 0.17…0.05).
- Interpretation: everything the model sees and every eval mechanic checks out, so the low
  accuracy is learning-limited. The relevant facts are outside my scope and belong to the trainer and
  loss auditors. (a) lambda·emb ≈ 360 vs backbone ≈ 13, with the gradient norm ~16 clipped to 1.0
  every step, so the update direction is mostly the cosine loss. (b) Per optimizer step we see
  4 sequences × 256 anchors = 1,024 blocks; the paper sees 112 × 512 = 57,344 (56× more) at the same
  6e-4 peak LR. At step 4516 we have trained ~4.6M blocks, about the paper's step 81 of 3,655 by
  block count. (c) The curriculum is still at p=0.5 (finding 3).

### 2. OK (eval exact) + DEVIATION in reporting: opening tau is a biased, noisy progress signal
- `train.py:479-505` measures opening RB tau on 18 anchors, one per prompt and two per domain, all at
  the formulaic `<think>\n` start. Per-domain opening values are dominated by single phrases:
  `longctx` ≈1.00 and `multilingual`/`writing` ≈2.0 at every step.
- Proposed reporting fix (not applied):
```diff
--- a/dspine/train.py
+++ b/dspine/train.py
@@ def evaluate_now():
-            if run.get("eval_report_first_anchor"):
+            # Opening tau is a formulaic-start metric (one anchor/prompt); never use it for selection or progress.
+            if run.get("eval_report_first_anchor"):
```
  Also set `"eval_sequences": 64` in the run config. The eval costs 3-17 s now, and 64 prompts cut
  the cohort spread (±0.03 between cohorts) by ~2×. Add `trials: 4` to the complete-round walk.
  Treat the audit-long rounds (offset-bucketed, deep context) as the second selection signal.

### 3. RISK (corrects the brief): the correct-predecessor curriculum is ACTIVE at p=0.5
`metrics.jsonl` at step 4516 shows `curriculum_p: 0.5` and `schedule_progress: 0.079`
(`train.py:559-567`, `losses.curriculum_probability`). Half the training blocks feed `sqrt(r)·C[y_{t-1}]`
as the predecessor at layers 1-3 (`modeling.py:275-282`, `layer < ceil(L/2)`). That matches paper
Appendix A: p=0.5 for the first sixth of training, linear to 0 at one third, i.e. schedule steps
~9.3K-18.6K here. It is not a bug. But eval and serving never see those blocks, and the logged
training ce/l1/embedding average both kinds. Expect held-out tau to move when p decays. Checkpoint
selection before step ~18.6K mixes two regimes.

### 4. RISK (small): teacher-prefill p (training) vs actual decode p (eval)
On identical prefixes (`prefill_vs_decode_p.py`, 486 positions from `pilot-eval-prefill`), mean
Σmin(p_prefill, p_decode) = **0.985** (p10 0.952, min 0.433). The earlier receipt
`results/decode-p-validation.json` found 10/4078 sampled tokens absent from prefill p. This caps a
perfect prefill-trained drafter at ~98.5% per-position overlap. It is real but second-order.

### 5. OK: proposal sharpening does not help
Exact RB tau over offsets 0-56 vs the K-way proposal temperature T_q (`proposal_temperature.py`).
The rule stays lossless for any q because verification uses that q. b6: T_q 1.0 → 1.660,
0.8 → 1.670, 0.6 → 1.670, 0.3 → 1.662, 0.05 → 1.651. The L1-heavy loss already makes q about as sharp
as is useful. Keep T_q=1 (paper Appendix C).

### 6. OK: train/inference parity of the fast packed path (our config)
`test_packed_parity.py` (10 passed, 4 skipped; output in `packed-parity-cpu.txt`). Injection is fully
active (random nonzero W_l, G_p, b, A). `forward_packed(reference=True)` is compared with per-anchor
`forward(reference=True)` built exactly like `eval_rounds.py:242-263`. Anchors cover a<W, a=W, a>W,
overlapping and duplicate blocks, a tail block, and sequence_start>0.
- Tiny model, full and restricted row-INT8 head: hidden 3.9e-6, pre_injection 4.6e-5, layer
  features ≤3.1e-6, transitions 2.1e-5. Candidates and `propose()` T=1 tokens are identical; q differs
  by ≤1e-6. The b6 prefix of a b16 forward equals a b6 forward.
- The FlexAttention branch, run eagerly on CPU from the real `Attention.forward_packed` source (only
  `is_cuda` and `torch.compile` removed), is **bitwise equal** to the explicit mask. Q_LEN 16/112/144/208
  covers padded tail tiles and the `qi//b` clamp. A wrong window changes the output, so the test is
  sensitive. Compiled CPU flex is unsupported on macOS (the 4 skips). The compiled CUDA kernel is
  covered by `results/prerun/gpu/packed-parity-gpu.json` (FP32 CUDA, real shapes, transitions ≤6e-6).
- Real config: step-4516 weights, real W8 head, 189,090-id INT8 head, FP32, W=4096. On a synthetic
  4,200-token sequence (anchors 100/4095/4096/4100/4150) hidden is ≤1.8e-5, with one FP32 near-tie
  candidate swap at rank 7/8 (scores 3e-6 apart). On two real long-context held-out shards, offsets
  +0..+61, hidden ≤6.4e-6 and candidates, q and sampled tokens are equal.
- Anchor: input `embedding(anchor)`, never injected, and it always sends `sqrt(r)·C[anchor]`
  (`modeling.py:273`). Training feeds the same anchor id and position: `sequence_ids[offsets]` and
  `a+start` (`modeling.py:460`, `:143-146`; `data.py:549-575`).
- Labels and p are aligned identically in training (`hidden[:,1:]` vs `labels[:,1:]`, p[a+t]) and
  eval (`ids[a+1:]`, p[a+1:]). Refinement in training uses the same `last_write` and the same
  `proposal_rows` INT8 rows as `build_transition_cache`. Packing never mixes sequences: one sequence
  per micro-batch (`train.py:568-572`), with many anchors masked to their own block and window
  (`modeling.py:150-158`).

### 7. OK: model vs paper (Sections 4.1-4.5, Appendix A)
- Transfer space (`transfer_space.py:10-43`) matches Appendix A: row-L2, centered Gram, top-r eigh,
  C = normalize(Λ^{-1/2} Pᵀ(Ē−μ)), built from the effective W8 head that serving uses.
- Readout `z = sqrt(r)·normalize(R_l u)` on the pre-injection residual (`modeling.py:247-258`). R_l
  is initialized to P_rᵀ (`:244-245`). There is a separate final-layer read weight.
- Gate σ(G_q z_t + G_p z_{t-1} + b), message A z_{t-1}, write `u + RMS(u)·W_l(g⊙m)` (`:263-269`).
  W_l and G_p are zero-initialized with b=0, so injection starts as the identity (`:229-236`).
  A, G_q, G_p and b are shared and W_l is per-layer. Senders and receivers are pre-update, and the
  anchor is unchanged (`:271-284`).
- Block-causal attention over [context ‖ block] (`:34-46`, `:150-158`). Context K/V go through each
  layer's own k_proj/v_proj, with k_norm, from `hidden_norm(fc(features))`, without the input norm
  (`:85-98`, `:406`), as in DFlash. RoPE uses absolute positions: context at its own positions and the
  block at anchor_pos+t (`:97-98`, `:402`).
- Last-write refinement redoes the final write from cached u with `sqrt(r)·C_s` (`:500-506`). The
  K×K transition cache uses predecessor rows and receiver columns (`:508-523`). `propose()` samples
  the K-way softmax and returns the exact q (`:525-557`).
- DEVIATION (by design, harmless): a learned mask vector (mean-embedding init) in place of tencent's
  frozen mask-token row. Tencent loading still uses its row (`:604`).
- DEVIATION (sampling, trainer-owned, low): 256 anchors per sequence uniformly **with replacement**,
  whatever the length (`data.py:549-558`). A 50-token response gets ~5 copies of every anchor per
  pass. That over-weights short responses per token and wastes compute. Proposed fix: draw
  `min(256, length-1-response_start)` anchors without replacement
  (`torch.randperm(n)[:k] + response_start`).

### 8. RISK (coverage): the selection cohort only sees the first 64 response tokens
Every pilot-eval response is a 64-token `length`-capped sample. Deeper training positions score better
(finding 1, t1 0.44 vs 0.39). The only deep-context eval is audit-long (4 sequences, mostly code,
120 s cap). Whole-response tau on real prompts is unmeasured offline. The decisive number remains
the serving bench.

## Decisive GPU tests (≤20 min each, after the lambda A/B frees the GPU)
1. **Curriculum effect on held-out tau (finding 3):** fork from step 4516 into two arms,
   `curriculum_p` forced to 0.5 vs 0.0, 400 steps each (~9 min each at 1.33 s/step), same data order
   and lambda. Measure complete-round T1 tau on all 64 held-out prompts with 4 MC trials, plus
   audit-long. If the p=0 arm is ≥0.05 higher, consider bringing the decay forward. The paper's
   schedule assumes 57K blocks per step.
2. **BF16 autocast training forward vs FP32 reference on real shapes:** run step-4516 weights on two
   real training sequences × 256 anchors through `forward_packed` (CUDA, autocast BF16, compiled flex)
   and through the FP32 reference. Compare Σmin(p,q) at t1..3 and the transition-cache candidates
   (<2 min). This closes the last parity gap (BF16 rounding at real shapes). CPU and FP32-CUDA parity
   are already proven.

3. **Does the lambda fix teach context use?** Settled on CPU above: tencent depends on context and
   ours does not. Score every lambda A/B arm, and any later checkpoint, with the window ablation:
   `context_ablation.py` on CPU, or on GPU in <2 min by setting `model.config.context_window` to
   4096/64/8/1 in `PackedReplay`. Add "Δ t1 overlap W4096→W8" to the eval log as a health metric.
   A healthy run should show a clear W8 drop (tencent: −0.16 accuracy at pos-1, −48% tau). Hypothesis
   for the trainer: with lambda·emb ≈ 360 vs backbone ≈ 13 and every step clipped from norm ~16 to 1,
   the update is mostly the cosine loss. That loss can largely be met from local features, so the
   context attention gets a small share of a tiny batch (1,024 blocks/step vs the paper's 57,344).

## Files
- `docs/audit/model-eval/offset_tau.py` → `offset-tau-step4516.json` (CPU repro of the GPU eval +
  per-offset RB tau)
- `docs/audit/model-eval/test_round_walk_vs_real_spec.py` → `round-walk-vs-real-spec.txt`
- `docs/audit/model-eval/test_packed_parity.py` → `packed-parity-cpu.txt`
- `docs/audit/model-eval/generalization_gap.py` → `generalization-gap.json`; `train_vs_heldout.py` →
  `train-vs-heldout-step4516.json` (unmatched sample, superseded by the matched one)
- `docs/audit/model-eval/tencent_context_ablation.py` → `tencent-context-ablation.json`;
  `context_path_weights.py` → `context-path-weights.txt`
- `docs/audit/model-eval/feature_alignment.py`, `prefill_vs_decode_p.py`, `proposal_temperature.py` →
  `proposal-temperature-step4516.json`, `inspect_openings.py`, `longctx_opening.py`,
  `context_ablation.py` → `context-ablation-step4516.json`
- Data used (local copies, not committed): `/tmp/audit-model/{ckpt,ckpt-init,target,eval,eval-prefill,train,tencent}`.
