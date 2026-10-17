> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Training-loop mechanics audit (AUDIT-LOOP), 2026-10-17 03:2x EDT

Scope: `dspine/train.py`, `dspine/data.py`, `dspine/rolling.py`, `dspine/config.py` and their use of
`dspine/modeling.py`, `dspine/losses.py`. Live evidence: `./work/ckpt/full/metrics.jsonl`
(4,516 step records + 11 eval records, window-00000 complete), `runs/rolling/plan.json`, `ckpt/full/step-00004516/`,
`ckpt/pilot-init/`. Read-only; no GPU use. The running the workstation code differs from local only by the TRAINER's
in-flight fork/batch-trace/lambda-reduction additions (diffed; mechanics identical), so findings apply to the live run.

Verdict up front: **no CONFIRMED mechanics bug found**. The optimizer/param set, accumulation, clip placement,
step accounting, RNG/cursor resume, window rotation, eval isolation and LR schedule are all implemented correctly
and verified against live checkpoints. Four RISks, one DEVIATION note, one dead-code note. The known
lambda-position-sum suspect (TRAINER's) is excluded here except where it interacts mechanically with clipping.

---

## Ranked findings

### RISK 1 (highest impact): gradient clipping is active on 100% of optimizer steps; median clip ratio ~16x

Evidence (metrics.jsonl, all 4,516 steps):
- `gradient_norm` (pre-clip): min 1.142, p50 15.86, p90 21.36, p99 29.14, max 56.83.
- Clip threshold is 1.0 (`train.py` `clip_grad_norm_(parameters, 1.0)`): **4,516 of 4,516 steps clipped** (0 unclipped).
- Trend: p50 rises 3.4 (steps 0-500) -> ~15-20 through step 4516; never approaches 1.0 after step ~200.

Why: the objective is a position-weighted SUM over 15 positions (weight mass ~6.63,
`losses.py:236-241` `reduce = (x*w).sum()/labels.shape[0]`), and the auxiliary embedding term has
coefficient lambda ~17-21 (metrics), so `lambda*emb ~ 360` dwarfs distillation ~13 (brief's numbers). The
clip threshold 1.0 was sized (per `docs/notes/trainer.md` "clip1") for a loss scale that no longer matches
this objective.

Mechanical consequences:
1. The configured schedule "peak LR 6e-4" is not the delivered semantics: after clipping, every update has
   norm exactly lr, i.e. the LR schedule only sets a *normalized* step size. Adam's m/sqrt(v) is invariant to a
   *constant* rescale, so the damage is muted - but the clip factor *varies* 1.1x-56.8x step to step, which
   distorts the m/v ratios and makes per-step effective magnitudes noise dominated by the clip ratio.
2. Gradient direction is dominated by the lambda*embedding term (it is ~96% of the loss magnitude). This is
   the mechanical face of the known lambda suspect: the distillation gradients ride on top of an embedding-
   alignment update. The TRAINER's `weighted_token_mean` A/B (`coefficient / weight_mass`, local
   `losses.py:283-284`) reduces lambda ~6.63x and would also cut the gradient norm by the same order,
   putting the clip back in its intended rare-safety-net regime. My independent read supports that direction.

Proposed fix (do not apply without coordinating with the TRAINER, who owns losses.py): either
(a) adopt per-token-mean reduction for the whole loss (divide `reduce` by `weight_mass`) and re-tune LR, or
(b) keep the sum scale but raise the clip to ~30 (p99+ margin) so clipping engages only on genuine spikes.
Numeric check already available in metrics: `gradient_norm` is logged pre-clip, so no GPU time is needed to
monitor the regime after any change.

Decisive GPU test (<= 20 min, if wanted): run ~30 training steps in a fork with
`torch.autograd.grad` decomposing `total` into distillation vs auxiliary components, logging
`||g_dist||`, `||g_aux||` separately. This pins the direction-dominance number (~expected >90% aux)
that the lambda A/B will change.

### RISK 2: response-length-dependent per-token gradient weighting (up to ~40x skew)

`data.py:PackedSequenceStream.__next__` draws `anchors_per_sequence=256` anchors **with replacement** from
`[response_start, length-1)`. The loss is normalized per anchor (`/labels.shape[0]`), so each *sequence*
contributes equal total gradient weight regardless of response length:
- median-length sequence (~2,370 response tokens, measured avg 2,370.5 over window 0): 256 draws -> ~242.7
  unique positions, ~5.2% duplicated anchors, each position weighted ~0.108x.
- short responses: metrics show `response_tokens_seen` deltas as low as 240/step (4 sequences of ~60
  response tokens). For a 60-token response, 256 draws from 59 positions -> every position drawn ~4.3x.

Net effect: tokens in short responses receive up to ~40x more gradient than tokens in long responses, purely
as a function of sequence length. The paper (Sec 4.x) does not specify anchor counts per sequence; this is a
sampling-design risk, not a contract violation, and it plausibly biases toward short-response domains and
away from deep-body positions (long responses), consistent with - though not proof of - the
opening-vs-complete tau gap. (Anchor *positions* are uniform over the response, so early/late positions are
not directly skewed; only cross-sequence weighting is.)

Proposed fix (diff, not applied - changes the exact-resume data contract, so it needs a fork/new-run decision):

```diff
--- a/dspine/data.py
+++ b/dspine/data.py
@@             anchors = (
                 torch.tensor(
-                    [
-                        self.rng.randrange(row["response_start"], length - 1)
-                        for _ in range(self.anchors)
-                    ]
+                    self.rng.sample(
+                        range(row["response_start"], length - 1),
+                        min(self.anchors, length - 1 - row["response_start"]),
+                    )
                 )
                 .sort()
                 .values
             )
```

(with a matching cap in the `AnchorStream.random` policy if independent-anchor mode is ever revived).
This removes duplicates for any response longer than 256 tokens and caps total draws otherwise.

### RISK 3: LR/curriculum hit exactly 0 at the 540M-token endpoint; overshoot trains at lr=0

`train.py` computes `schedule_step = total * min(1.0, response_tokens_seen / 540e6)` and
`losses.lr_multiplier` returns `0.5*(1+cos(pi*min(1.0, progress)))` -> exactly 0 at progress 1.0. The
schedule denominator is the *planned* 240M x 2.25; actual presentations are the sum of per-window
`ceil(ids*2.25/4)*4` sequence visits (14 windows, 99,326 records; 55,877 steps). Measured window-0 tokens
(42,821,556 over 4,516 steps) and corpus counts put total presentations at ~540M +/- <1%, so the endpoint
is consistent today. But nothing guards the overshoot direction: if appended campaign-2 windows carry more
tokens than planned, the final steps run at lr=0 (silent wasted compute), and if fewer, the tail of the
cosine never completes (mild). Watch `schedule_progress` in the last window; expected final value 1.00.

Proposed fix (diagnostic, minimal):

```diff
--- a/dspine/train.py
+++ b/dspine/train.py
@@             if device.type == "cuda":
                 torch.cuda.synchronize()
+            if record["schedule_progress"] >= 0.999 and step + 1 < stop_at:
+                print(json.dumps({"warning": "schedule exhausted before step budget",
+                                  "step": step, "stop_at": stop_at}), flush=True)
```

### RISK 4 (low): `compile` flag is a dead knob for packed training

`train.py` compiles `layer.forward` when `run["compile"]` is true, but the packed training path
(`DSpineDraft.forward_packed`) calls `layer.self_attn.forward_packed(...)` and `layer.mlp(...)`
directly - `DecoderLayer.forward` is never invoked in packed mode. Today `compile: false`, so no live
impact; but anyone setting `compile: true` for the speedup will silently get nothing. FlexAttention itself
is separately compiled (`torch.compile(flex_attention, dynamic=True)`), which is where the real win is.

Proposed fix (if compiled layers are ever wanted):

```diff
--- a/dspine/train.py
+++ b/dspine/train.py
@@     if run.get("compile", False):
         for layer in model.layers:
-            layer.forward = torch.compile(layer.forward, dynamic=False)
+            layer.self_attn.forward_packed = torch.compile(
+                layer.self_attn.forward_packed, dynamic=False
+            )
+            layer.mlp = torch.compile(layer.mlp, dynamic=False)
```

(Verify no interaction with `checkpoint(step, ...)` recompute first; the current config keeps this off.)

### DEVIATION FROM PAPER (accepted, flagging for completeness)

`losses.training_loss` ramps `beta` (lambda coefficient) over the full 4% response-token warmup
(~2,280 steps / ~21.6M tokens). The paper (Appendix A, line 978) ramps beta 0 -> 0.5 over 146 steps. The
4%-by-response-tokens warmup is the documented design for this run (BRIEF, config), so this is a deliberate
scale-up, not a transcription error. Curriculum p (0.5 flat until 1/6, linear to 0 at 1/3, shared draw for
layers 1-3) matches paper line 982 exactly (`losses.curriculum_probability`, `modeling.AdjacentBridge.forward`
`layer < ceil(6/2)`).

---

## Verified OK (with evidence)

**Parameters / optimizer (task question 1)**
- Trainable set = 77 parameters: `mask_embedding`, `fc.weight`, `hidden_norm.weight`, `norm.weight`,
  6 layers x 11 (`input/post_attention_layernorm`, `q/k/v/o_proj`, `q/k_norm`, `mlp gate/up/down`),
  bridge `early_read_weights`, `read_weight`, `feature_write_weights`, `gate.{weight,bias}`,
  `predecessor_gate.weight`, `prev.weight`. Verified: `step-00004516/model.safetensors` has exactly 78 keys
  (77 params + frozen `selected_prefix_bridge.codebook` buffer), and `trainer.pt` optimizer state has
  exactly 77 entries. Nothing trainable is outside the optimizer; nothing frozen is inside it.
- Frozen as intended: target embedding (`nn.Embedding.from_pretrained(..., freeze=True)`), LM head
  (`requires_grad=False`; `_FrozenHeadLoss` re-checks `head.requires_grad`), transfer table C
  (`codebook` is a registered buffer, not a Parameter). `proposal_weight`/`proposal_lookup` are
  non-persistent buffers rebuilt deterministically from the frozen head on resume
  (`configure_proposal_vocabulary`, `row_int8_weights` is a pure function); `sorted_ids_sha256` is
  validated on resume, so no silent head drift.
- No accidental freeze/detach/re-init on resume: `from_pretrained` requires `missing == {embedding.weight,
  lm_head.weight}` and zero unexpected keys; `mask_embedding` is present, FP32, and in the optimizer both at
  pilot-init and at step 4516 (dtypes checked: mask F32, fc F32, codebook BF16 both checkpoints).
  `pilot-init` is our own format (no legacy `out.weight`; legacy Tencent path, which freezes the mask row,
  is not in play). `mask_embedding.to(...)` inside forward is differentiable (no detach); gradient flows
  through `expand`.

**Gradient accumulation (task question 2)**
- `optimizer.zero_grad(set_to_none=True)` at the top of each step; `(loss/accum).backward()` per micro-batch;
  `clip_grad_norm_` after all 8... actually 4 micro-batches (accum=4); one `optimizer.step()` per logged step.
  Adam's internal step counter in `step-00004516/trainer.pt` is exactly 4516 == logged step count == metrics
  record count (unique, gapless 1..4516). LR set per-group from the schedule before `step()`; checkpointed
  group lr (5.97525e-4 at step 4516) equals the logged lr.
- Loss scaling across micro-batches is exact: each packed micro-batch is one sequence with exactly 256
  anchors, `reduce` divides by `labels.shape[0]`, metrics are averaged with equal 1/accum weights.
- Effective tokens per optimizer step: 4 sequences x 256 anchors x 16 positions = 16,384 draft positions
  (metrics `anchors: 1024` = 256x4); ~9,484 response tokens counted toward the schedule (mean; median 6,554,
  min 240, max 55,979 - heavy-tailed response lengths, see RISK 2).

**torch.compile / FlexAttention (task question 3)**
- `compile: false` in the installed config, so no compiled trainer function is live except FlexAttention's
  internal `torch.compile(flex_attention, dynamic=True)` (compiled once, module-global).
- No stale-constant capture: the FlexAttention block mask is built per micro-batch from `anchor_offsets`
  (an input tensor) inside `create_block_mask`; curriculum probability and teacher tensors are created in
  the eager step loop and passed as inputs; lambda/beta are Python floats computed per step in eager code.
  Nothing curriculum- or schedule-related can be baked into a compiled graph.
- No recompile pathology in the data: `seconds_per_step` p50 1.219, p90 1.388, p99 1.577, max 1.79 over
  4,516 steps (stable; the measured pilot value was 1.183 plus eval/checkpoint overhead).
- `create_block_mask` runs inside the checkpointed region and is recomputed on backward recompute - overhead
  only, not correctness.

**Activation checkpointing + RNG (task question 4)**
- `config.gradient_checkpointing: true` and training runs under `model.train()`, so the checkpointed path
  is active (peak 19.0 GB alloc / 24.7 GB reserved under the 36 GB cap is consistent with recompute).
- There is no dropout or in-region randomness anywhere in `DSpineDraft`; the only per-step RNG use
  (`torch.rand(count) < probability` for teacher blocks) happens in the eager training loop, *outside* the
  checkpointed `step` closures. `use_reentrant=False` is used. Recompute is therefore bit-identical.

**Resume / warm start (task question 5)**
- `pilot-init` contains only `config.json`, `model.safetensors`, `initialization.json` - **weights only, no
  optimizer state** ("optimizer": "fresh on full run"). The full run therefore started with fresh Adam
  moments and the LR/curriculum/lambda schedules restarted at progress 0: warmup 4% by response tokens
  (~2,280 steps), beta ramp tied to the same warmup. This is a sensible warm-start (fresh second moments +
  warmup protect the borrowed weights), and it is exactly what the receipt documents.
- Exact resume requires `state["run_config"] == run` (bitwise dict compare), config-arch equality check, and
  unchanged proposal policy. Checkpoint payload retains optimizer, python/numpy/torch/cuda RNG, data cursor
  (`paths`, `epoch`, `index`, stream RNG), `data_window`, `response_tokens_seen`, elapsed seconds.
  Verified live: cursor epoch 2, index 2008 of 8,028 - exactly 2.25 passes at step 4,516; RNG block has all
  four engines; Adam step counter matches. Metrics stream across restarts is gapless and duplicate-free;
  `response_tokens_seen` is strictly monotonic; `elapsed_seconds` accumulates across restarts.
- Window rotation (`--reset-data`) resets only the data cursor and reseeds the *stream* RNG
  (`seed ^ sha(window_name)`); global torch/CUDA RNG, optimizer state and the token schedule continue
  (rolling.py train phase passes `--reset-data` only when the window actually changed, and the trainer
  refuses a mismatch). SIGTERM checkpoints at the next step boundary (tested earlier, 131->132 resume).

**Data sampling (task question 6)**
- 2.25 passes per window are implemented as a per-window step budget: `stop_step = ceil(ids*2.25/4)` with 4
  sequences/step. Window 0: 8,028 ids -> ceil(18,063/4)=4,516 steps; consumed 4,516*4=18,064 sequence visits
  = 2.2502 passes; measured 42,821,556 response tokens = 8,028 x 2,370.5 x 2.25 exactly.
- No skipped/duplicated sequences across epochs: `paths` reglobbed and reshuffled at each epoch boundary
  (fixed shard set within a window), `index` restored exactly on resume; ineligible rows (excluded ids,
  wrong split, <2 response tokens) are advanced over without consuming anchor RNG. Duplicate *anchors*
  within a sequence exist by design (RISK 2).
- Train/val separation: window manifests are built from `split="train"` only; validation is a separate
  captured root (pilot-eval T1/T0) - the trainer's rounds/first-anchor evals read `eval_features`, not the
  training shards (checked `measured_eval` passes the `val` stream's root). `data_epoch` in metrics cycles
  0..2 as expected.
- The 14 windows sum to 55,877 steps; ~99,326 training records; schedule endpoint math checks out to
  ~540M presentations (see RISK 3 for the guard).

**Eval (task question 7)**
- Evaluated weights: **live** weights at the eval step (there is no EMA anywhere in the trainer). `save_best`
  snapshots the live model and maintains independent `best-b6.json`/`best-b8.json` by complete-round tau,
  pooled across domains weighted by round counts; pruning protects referenced winners. Current best:
  step 4500 (b6 1.635).
- Eval runs in `model.eval()` with `model.train(training)` restored in `finally` (both `eval_offline.evaluate`
  and `eval_rounds.evaluate_rounds`); the eval forward passes call `forward`/`forward_packed` **without**
  `teacher_blocks`, so no teacher forcing enters eval.
- Eval cannot perturb training: all eval sampling uses a dedicated `torch.Generator(device).manual_seed(seed)`
  (Monte Carlo acceptance) or argmax (T=0); the training stream is untouched (fresh `AnchorStream`
  instances); everything runs under `no_grad`/`inference_mode`. The training CUDA RNG sequence is therefore a
  pure function of checkpoints + teacher draws - resume determinism holds.
- `eval_at_end` fired at window end (eval at step 4516 after the 4500 periodic eval), and
  `eval_before_training` ran the step-0 baseline (22s incl. T0 + long).

**Live metrics anomalies (task question 8)**
- LR: matches `lr_multiplier(schedule_progress*1e6, 1e6)` for **all 4,516 steps** (0 mismatches at 1e-9
  rel); warmup completed ~step 2,280 (progress 4%), peak 6e-4 held after.
- `curriculum_p`: flat 0.5 through step 4516 - correct (decay starts at progress 1/6 = 16.7%; run is at 7.9%).
- No loss spikes: per-window loss medians smooth (80 -> 440 as lambda ramps in, then 440 -> 405 declining);
  the trainer raises on any nonfinite loss/grad, and none fired.
- `seconds_per_step`: p50 1.219 vs measured 1.183 budget - overhead is eval/checkpoint amortization, as
  planned (eval 4-22s every 500 steps, checkpoints every 200 steps).
- Memory: 19.0 GB peak alloc / 24.7 GB reserved, stable, under the 36 GB cap.
- The one genuine anomaly is the gradient-norm regime (RISK 1): never unclipped, p50 ~16x over threshold.

---

## What I could not settle on CPU

- Per-term gradient decomposition (distillation vs auxiliary) - needs the short GPU test in RISK 1, or falls
  out of the TRAINER's lambda A/B directly.
- FlexAttention GPU numerics vs the explicit-mask CPU reference on this exact target (previously validated on
  the pilot path per `docs/notes/trainer.md`; the packed parity gates passed in the pre-run audit, 14 cases).

## Bottom line for the acceptance symptom

The loop's mechanics deliver exactly the updates the loss asks for - nothing is frozen, dropped, double-
counted or stale. The slow post-warmup tau gains and the opening/complete gap are therefore most plausibly
objective-side (the lambda scale the TRAINER is A/B-ing) and data-weighting-side (RISK 2's length skew),
not optimizer/data plumbing. RISK 1 is the mechanical amplifier of the objective-side problem: with ~96% of
the gradient norm coming from the lambda*embedding term and the clip permanently engaged, the distillation
signal the tau metric needs is a minor component of each update.
