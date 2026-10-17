> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# AUDIT-LOSS: objective and optimization

Reviewed 2026-10-17. Scope: `dspine/losses.py`, its model/trainer call sites, paper §§4.4–4.5, Appendix A and Table 4. No `dspine/` files were edited. All numerical work ran on the Mac CPU with Python 3.13 / torch 2.6.0. the workstation access was read-only SSH for metrics, run configuration and source inspection; no GPU was used.

## Ranked verdict

1. **RISK, highest expected impact: lambda is probably 6.63061 times too large.** My reading of “the token cross-entropy of the backbone output” is a token mean, not the position-weighted block sum currently used. The paper does not explicitly specify this reduction, so this is not a proven implementation bug. The implemented coefficient is unambiguously the block sum. Independent tiny-model measurements show its auxiliary gradient is **43.9–54.8 times** the distillation gradient on shared trunk parameters, accounting for **96.9–97.4%** of the sum of individual term norms. Normalizing lambda alone reduces that ratio to **6.61–8.27 times**. This supports the TRAINER agent's existing A/B; I did not duplicate or modify it.
2. **DEVIATION / RISK: same peak LR with a much smaller and less diverse batch.** Actual training uses four sequences and 1,024 anchors/update, versus 112 sequences and 57,344 anchors in Table 4: 28 times fewer independent sequences and 56 times fewer anchors. Keeping `6e-4` is not paper-validated for this regime. A different LR or batch is not *clearly* better from CPU evidence alone; compare after resolving objective balance.
3. **RISK: accumulation changes the adaptive auxiliary objective.** Lambda is computed inside each microbatch, so training averages `beta * CE_j.detach() * Emb_j`, rather than multiplying a single global-batch CE by global-batch embedding loss. This is a confirmed mathematical difference, although the paper leaves microbatch handling unspecified. The CPU reproduction changes the full parameter gradient by **7.74% in relative L2**. Last-100 live auxiliary loss is **4.42%** above the product of logged within-step means because of this covariance.
4. **DEVIATION, probably small on the measured corpus: the proposal objective is restricted, not full-vocabulary.** The 189,090-row head drops CE for unproposable labels. This differs from §4.5, but the implemented L1 correctly retains outside target mass and has the correct gradient. A matching archived coverage snapshot reports mean retained target mass **0.99968835**, so the measured average missing-mass penalty is small. This is not a loss-kernel bug.
5. **OK: custom backward, position weights, refinement masking/renormalization, post-injection supervision, live curriculum, optimizer enrollment, clipping and token schedule.** No confirmed production-impacting arithmetic or indexing bug was found in those paths. BF16 chunked backward has small rounding differences, not a sign, scale or omitted-mass error.

## Evidence and reproducibility

Run from the repository root:

```sh
PYTHONPATH=. python3 docs/audit/loss/check_loss.py
```

The script includes analytic-vs-autograd comparisons, an independent central finite difference, sparse padding and equality cases, exact restricted-vocabulary size, tie ordering, refinement edge cases, component gradients for three seeds, post-injection hooks, curriculum capture, accumulation comparison and an actual mixed-precision optimizer update. Results are in `loss/results.json` (`loss/results.json`, retained internal evidence); successful console output is in `loss/check.stdout.txt` (`loss/check.stdout.txt`, retained internal evidence).

The read-only metrics snapshot contains 4,516 training rows through step 4,516, plus evaluation rows. Original uncompressed SHA-256: `ff717d24e5563b2380cd972c1a29e46d013c9572f7f881bbd7333980745267ed`. The archived input is `loss/live-metrics.jsonl.gz` (`loss/live-metrics.jsonl.gz`, retained internal evidence), with `loss/live-run.json` (`loss/live-run.json`, retained internal evidence). Refreshing the snapshot is optional; the committed inputs reproduce the report.

The local worktree already contained another agent's lambda A/B additions. I tested the historical `position_sum` branch and its explicit mean alternative. The remote production loss still has the original position-summed coefficient and no A/B flag. Local and remote `modeling.py` have identical SHA-256 `83801bed6c57b2fe580c67809bd19fb91bce1e49464d34fa10479be6141db343`. Remote source hashes read during the audit:

| Source | SHA-256 |
|---|---|
| production `losses.py` | `fb6a45ae5f173d5a3827eff8034435363d1e600d58921a13668bcf48a16afff7` |
| production `train.py` | `e47bb68b5e0ad40f10eed2300d2c9729179977be9ff435c064f49f0bf0716198` |

Local hashes used by the numerical script are recorded in its JSON output. Line references below refer to the reviewed local files; concurrent TRAINER changes can move lines.

## 1. Chunked CE and sparse L1: derivation and numerical result

Let `z = W h`, `q = softmax(z)`, and `S` be the unique positive-probability sparse support. Since `sum(q)=1` and `p=0` outside `S`,

\[
L_1 = 1 + \sum_{v\in S} (|q_v-p_v|-q_v).
\]

Set `a_v = sign(q_v-p_v)-1` on `S`, zero elsewhere, and `A = sum_v a_v q_v`. Then

\[
\frac{\partial L_1}{\partial z_j}=q_j(a_j-A),\qquad
\frac{\partial CE}{\partial z_j}=q_j-\mathbf 1[j=y].
\]

For arbitrary upstream per-row gradients `gCE`, `gL1`,

\[
\frac{\partial L}{\partial z_j}
=q_j(gCE-gL1 A)+gL1 q_j a_j-gCE\mathbf 1[j=y],
\qquad
\frac{\partial L}{\partial h}=\sum_j\frac{\partial L}{\partial z_j}W_j.
\]

These are exactly the operations in `dspine/losses.py:69–74` and `:80–103`. The global softmax normalization is retained across all chunks by FP32 `logaddexp(logsumexp(chunk))` at `:39–47`. The backward subtracts the global `A`, not a per-chunk value. Head parameters and target probabilities intentionally receive no gradients. At `q=p`, the implementation uses the same zero subgradient of absolute value as PyTorch.

Numerical coverage:

| Case | Maximum relative gradient L2 error | Maximum absolute error |
|---|---:|---:|
| FP32, 5 seeds × chunks 1/7/16/31/40, 31-row head | 2.84e-7 | 4.17e-7 |
| BF16 frozen head, same cases, FP32 hidden input | 3.14e-3 | 3.60e-3 |
| Effective row-INT8 reconstructed BF16 head, same cases | 2.78e-3 | 2.81e-3 |
| Restricted 61/97 vocabulary, FP32 | below 1e-5 | below 2e-6 |
| Restricted 189,090/248,320, row-INT8 BF16, chunks 4,096/8,191 | about 2.2e-3 | below 1e-4 |
| Independent central finite difference, FP32 | 1.98e-4 | 9.16e-5 |
| Explicit `q=p` kink | 0 | 0 |

BF16 differences arise from rounding each `grad_logits @ W_chunk` to BF16 before accumulating chunks in FP32 (`:102`), versus rounding a full dense matmul once. They are bounded and small in these tests. The full-head `full_backbone` policy was also compared against plain autograd: relative L2 error 2.41e-7 in FP32 and 1.55e-3 in BF16.

For the large reference, normalization uses FP64 after the same BF16 logits. CPU FP32 dense softmax over 189K entries itself showed accumulated normalization error of several e-6; the FP64 reference avoids treating that reference error as a loss bug. Small-input comparisons use ordinary FP32 autograd throughout.

The script covers non-divisible final chunks, chunks larger than the vocabulary, labels at both ends of the head, repeated **zero-probability** sparse padding, and separate signed upstream weights for CE/L1. Positive-probability sparse IDs must be unique; arbitrary duplicated positive entries would need coalescing before this sparse identity. The feature format represents tokenizer support, with repeated IDs used only for zero padding. Raw `chunked_head_loss` also assumes in-range IDs; production's restricted wrapper maps them to in-range indices before calling it.

### Restricted vocabulary and outside mass

At `dspine/losses.py:174–187`, missing sparse IDs map to local row zero with probability zero, and the outside probability is then added explicitly to L1. This is correct:

\[
L_1=1+\sum_{v\in S\cap H}(|q_v-p_v|-q_v)+p(H^c),\quad q(H^c)=0.
\]

No outside probability is silently discarded or renormalized into `H`. The repeated zero-probability placeholders have `a=0` when `q>0` and do not contaminate the backward. If every target token is outside `H`, L1 is exactly **2** and its hidden gradient is zero, as it must be for any normalized `q` with disjoint support. Mixed inside/outside cases match a dense full-vocabulary zero-padded `q` in both value and gradient. More generally the minimum attainable L1 is `2*p(H^c)`.

CE is multiplied by `local_y >= 0`. Thus an outside label contributes **zero CE and zero CE gradient**, rather than an impossible `-log(0)`. That masking is a deliberate objective change, not full-vocabulary paper CE. A correct embedding target for such a label is still available through full-vocabulary `C`.

`dspine/hot_vocab.py:33–42` reconstructs per-row INT8 weights and casts them to the original BF16 weight dtype. It matches the non-fused serving head's effective rows (`serving/quantization.py:42–69`). CPU checks establish this matrix path, not exact parity with the CUDA `w8_gemm` implementation; include its logits in the GPU diagnostic below.

### Top-K and refinement

`stable_topk` (`dspine/losses.py:15–27`) uses lexicographic score bits then ascending token ID, canonicalizing signed zero. Tests include exact positive/negative/zero ties, ties split across chunk boundaries, and `K=16`. Local hot IDs are sorted globally (`dspine/modeling.py:359`), so local ascending ties are also global ascending ties. Selection is detached, as expected for a discrete candidate set. No epsilon modifies the logits or distributions.

Refinement (`dspine/losses.py:198–207`) passes the paper's requirements:

- CE is zero unless the correct label is in the candidate set; it is not divided by the eligible-position fraction.
- L1 is zero unless candidate target mass is **strictly greater** than `1e-4`. Tests cover zero, exactly `1e-4`, and `1.001e-4`.
- Eligible target probabilities are renormalized by their sum on the candidates. Both values and gradients match an independent dense candidate reference.
- Candidate head rows are frozen; gradients flow to the refined state and the bridge/backbone that generated it.

No arithmetic correction is proposed for these paths.

## 2. Equations 6–8 and objective balance

`dspine/losses.py:250–274` uses `alpha=.1`, `gamma=7`, and `w_t=exp(-(t-1)/7)` for the 15 predicted positions, excluding the known anchor. It sums over positions, divides by the number of anchors, and masks tail positions. Backbone and refinement are added with equal coefficient; each has the same CE/L1 mixture. Those reductions match Eq. 7. Summing over layers and positions for the embedding term matches Eq. 6.

The readouts are genuinely **post-injection**, including the last layer before terminal normalization: `dspine/modeling.py:420–437` and `:474–498` call the same `selected_prefix_bridge.read` used inside injection. The CPU hook test sets nonzero write weights, so pre/post states differ, then verifies every returned readout equals `bridge.read(post_state, layer)` exactly in both ordinary and packed paths. Pre/post read differences are 0.096–0.166, so this test would detect a read-before-write bug. Cosine similarity removes the `sqrt(r)` scale, as intended. The target is the fixed full-vocabulary codebook row for `y_i`.

### My CE_bb verdict

I would implement `CE_bb` as a **scalar token-mean cross-entropy**. Eq. 7 explicitly defines the position-summed *distillation objective*. Appendix A instead calls the quantity used for lambda the “token cross-entropy of the backbone output”; it does not say to reuse the block-summed distillation CE. A token mean also keeps the adaptive coefficient independent of block length and the total position weight. This is an interpretation of an underspecified paper, not access to the authors' unavailable training code.

Current code (`dspine/losses.py:278–285`) uses

```
lambda = beta * mean_blocks(sum_positions(w * CE)).detach()
```

For a full 16-token block, `sum(w)=6.630610466`. Relative to a weighted token mean, lambda and every auxiliary gradient are exactly 6.63061 times larger. Preferred normalization would average over valid supervised positions. The existing `weighted_token_mean` A/B divides by nominal `sum(w)`; this is an appropriate controlled first test, although masked tail positions can still make it slightly lower than a valid-token mean.

Beta itself is correct: it rises linearly from zero to .5 during the same 4% warmup as LR. `CE_bb.detach()` correctly prevents a gradient through the coefficient. This prevents a CE-gradient shortcut; it does not reduce the auxiliary gradient.

### Component gradient measurements

Random-init CPU model: six layers, hidden 32, code 16, block 16, `K=16`, eight anchors, vocabulary 97, fixed normalized random codebook, paper zero-initialized write matrices, three seeds. Shared trunk means mask embedding, feature projection/norms, all decoder parameters and terminal norm, excluding bridge-specific parameters and frozen tensors. Both distillation and embedding losses train this group. No optimizer update precedes measurement. Coefficient uses saturated `beta=.5`; this is an objective-scale probe at random initialization, not an estimate of current checkpoint gradients.

Seed 42, term-weighted gradient norms on the shared trunk:

| Term | L2 gradient norm | Share of sum of individual weighted norms |
|---|---:|---:|
| `.1 * backbone CE` | 12.353 | 0.885% |
| `.9 * backbone L1` | 9.973 | 0.714% |
| `.1 * refinement CE` | 3.139 | 0.225% |
| `.9 * refinement L1` | 18.060 | 1.293% |
| `lambda_sum * L_emb` | 1,352.875 | **96.883%** |
| Combined distillation vector | 30.845 | not an additive share |
| `lambda_mean * L_emb` | 204.035 | alternative coefficient |

Across seeds 42/43/44, auxiliary-to-distillation norm ratios are 43.86/51.68/54.80, or 6.61/7.79/8.27 after normalizing lambda. Auxiliary/distillation gradient cosine is .162/-.139/.188: they are not merely parallel gradients being rescaled. The first layer attention query projection shows the same imbalance (seed 42 ratio 44.71), so this is not just a large bridge-only auxiliary gradient. Norm shares are descriptive, not an orthogonal decomposition of the total vector.

Normalizing lambda can therefore materially change optimization direction under global clipping. Adam's normalization means that clipping every step is not by itself proof of a tiny effective LR, and these tiny-model ratios do not prove production tau improves. Measure per-term gradients on the actual checkpoint before declaring the A/B settled.

Proposed first fix, **only in the isolated A/B run configuration**, using the already prepared TRAINER implementation:

```diff
--- a/fork-run.json
+++ b/fork-run.json
@@
+  "lambda_ce_reduction": "weighted_token_mean",
```

Do not change `alpha`, divide all losses by 15, or average the embedding term over layers to compensate: those would change explicit Eq. 6/7 reductions too.

## 3. Accumulation-dependent lambda

Evidence: `training_loss` is invoked inside the accumulation loop (`dspine/train.py:565–640`); each invocation computes its own coefficient from that sequence's 256 anchors. Dividing each loss by accumulation is correct for fixed-coefficient objectives, but not equivalent to a larger batch when the coefficient depends on the same microbatch:

\[
\frac{\beta}{M}\sum_j CE_j^{sg}\,\nabla Emb_j
\ne \beta\left(\frac1M\sum_j CE_j^{sg}\right)
\left(\frac1M\sum_j\nabla Emb_j\right).
\]

The isolated CPU reproduction uses the same four fixed forward outputs for both expressions. Weighted block CEs are 39.619/42.731/41.626/35.180. Global coefficient loss is 804.334; accumulated per-micro coefficient loss is 801.985; parameter gradients differ by 7.74% relative L2. With a fixed shared coefficient the accumulation gradient agrees **exactly**. Thus there is no missing factor of four; the difference is the adaptive coefficient.

Live last-100 auxiliary contribution, calculated as `loss - backbone - refinement`, is 386.622. Mean of each step's logged `lambda * embedding` is 370.247. Their 4.42% difference is within-step coefficient/embedding covariance across the four sequences, not multiplying averages across unrelated steps. This upweights harder sequences with larger CE; it may help or hurt, so I rank it below the much larger normalization concern.

If lambda is intended as a global optimizer-batch scalar, the following is an exact proposed patch shape, to be evaluated after the existing A/B. It adds one FP32 embedding-gradient buffer and an extra backward sweep. It should be profiled before adoption. The shown normalizer follows the current A/B's nominal-weight convention.

```diff
--- a/dspine/losses.py
+++ b/dspine/losses.py
@@
     lambda_ce_reduction="position_sum",
+    return_components=False,
@@
-    coefficient = beta * ce_backbone.detach()
+    lambda_ce = ce_backbone.detach()
     if lambda_ce_reduction == "weighted_token_mean":
-        coefficient = coefficient / weight_mass
+        lambda_ce = lambda_ce / weight_mass
+    coefficient = beta * lambda_ce
@@
-    return total, metrics
+    if return_components:
+        return (backbone + refinement, emb, lambda_ce), metrics
+    return total, metrics
--- a/dspine/train.py
+++ b/dspine/train.py
@@
             probability = curriculum_probability(schedule_step / total)
+            embedding_grads = [torch.zeros_like(p) for p in parameters]
+            global_lambda_ce = 0.0
             for _ in range(accum):
@@
-                    loss, metrics = training_loss(
+                    components, metrics = training_loss(
@@
                         lambda_ce_reduction=run.get(
                             "lambda_ce_reduction", "position_sum"
                         ),
+                        return_components=True,
                     )
-                if not torch.isfinite(loss):
+                distillation, embedding, lambda_ce = components
+                if not (torch.isfinite(distillation) and torch.isfinite(embedding)):
                     raise FloatingPointError(f"nonfinite loss at step {step}")
-                (loss / accum).backward()
+                grads = torch.autograd.grad(
+                    embedding / accum, parameters,
+                    retain_graph=True, allow_unused=True,
+                )
+                for buffer, grad in zip(embedding_grads, grads):
+                    if grad is not None:
+                        buffer.add_(grad)
+                global_lambda_ce += lambda_ce.item() / accum
+                (distillation / accum).backward()
@@
-            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
+            beta = 0.5 * min(1.0, schedule_step / max(1, round(total * 0.04)))
+            coefficient = beta * global_lambda_ce
+            for parameter, grad in zip(parameters, embedding_grads):
+                if parameter.grad is None:
+                    parameter.grad = torch.zeros_like(parameter)
+                parameter.grad.add_(grad, alpha=coefficient)
+            totals["lambda"] = coefficient
+            totals["auxiliary"] = coefficient * totals["embedding"]
+            totals["loss"] = totals["backbone"] + totals["refinement"] + totals["auxiliary"]
+            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
```

This is proposed only, not applied or GPU-tested. It deliberately leaves the existing batch scaling, position sums, layer sums and clipping order intact.

## 4. Curriculum and training-only use

The live trainer samples `torch.rand(count) < probability` once per block (`dspine/train.py:601`), passes that same Boolean vector through all layers, and does not resample in checkpoint recomputation. `AdjacentBridge.forward` uses it in layers `layer < ceil(L/2)` (`dspine/modeling.py:271–284`). For the production six-layer model this is exactly layers 1–3; it is also layers 1–3 for the paper's five-layer model.

The receiver feature always remains the model readout. The replacement is exactly `sqrt(r) * C[correct_ids[:, :-1]]`; receiver positions 1–15 receive the correct predecessor positions 0–14. Receiver position 1 already uses the known anchor regardless of the draw. The anchor itself is not updated. Hook capture confirms selected/unselected blocks, every layer, receiver states and predecessor indexing.

`curriculum_probability` (`dspine/losses.py:210–213`) returns .5 through progress 1/6, .25 at 1/4 and zero at 1/3 and afterward. The live 7.93% progress legitimately still has `p=.5`. Last-write supervision always uses the correct predecessor (`losses.py:263`), independent of curriculum, as Appendix A requires.

Actual serving/evaluation calls omit `correct_ids`/`teacher_blocks`, so they cannot use the replacement. Evaluation restores the model's prior train/eval flag (`eval_offline.py:180`, `eval_rounds.py:341`). The bridge API itself has no `self.training` guard: explicitly supplying teacher labels and a teacher mask in `eval()` still activates replacement. This is a low-priority defensive API concern, not evidence of leakage in current evaluation. Optional exact guard:

```diff
--- a/dspine/modeling.py
+++ b/dspine/modeling.py
@@
         if (
-            correct_ids is not None
+            self.training
+            and correct_ids is not None
             and teacher_blocks is not None
```

That guard is not necessary to explain the observed run.

## 5. Optimizer, dtypes, clipping and schedule

Source evidence: `dspine/train.py:318–333`, `:407–411`, `:638–649`.

| Item | Actual behavior | Verdict |
|---|---|---|
| Optimizer | AdamW, `weight_decay=0`, betas `.9/.999`, implicit torch default epsilon `1e-8` | Correct; paper does not specify betas/epsilon |
| Trainable parameters | All `requires_grad=True` tensors are passed to optimizer | Correct |
| Frozen parameters | Target embedding and LM-head parameters have `requires_grad=False` | Correct |
| Frozen transfer table | `C` is a detached registered buffer, not an optimizer parameter | Correct |
| Other buffers | Proposal rows/IDs/lookups are frozen buffers | Correct |
| Other trainables | Mask embedding, feature FC/norm, decoder/terminal norms, all bridge read/write/gate/message parameters | Enrolled; zero initial gate/read-through-write gradients are expected from zero writes |
| Precision | Trainable masters converted to FP32, CUDA autocast BF16; frozen embedding/head remain BF16 | Correct in code |
| Accumulation | Loss divided by `accum`; zero-grad before loop, one optimizer update after loop | Correct scalar scaling, with adaptive-coefficient caveat above |
| Clipping | Global norm 1.0 after all microbatches, before AdamW | Table 4 matched |
| LR assignment | Every optimizer group updated before `optimizer.step()` | Actually applied |

The CPU mixed-precision test executes the same master conversion, BF16 autocast forward, backward, clipping and AdamW update with a restricted row-INT8/BF16 head. All trainable tensors receive FP32 gradients; Adam moments stay FP32; frozen embedding/head/C remain bitwise unchanged. Its pre/post clipping norms are recorded in the JSON. CUDA fused AdamW and CUDA matmul parity still require the proposed GPU check.

### Response-token schedule, checked against every logged update

The installed run's immutable `total_steps=1,000,000` is a safety cap, not the intended campaign length or LR duration. The live schedule uses `s = total_steps * R/540,000,000`, with `R` the response tokens presented **before** the current optimizer batch. Logged token count is after the batch; accounting for this avoids a false off-by-one diagnosis. The maximum difference between logged progress and previous row's token count divided by 540M is **1.39e-17**.

Let `T=1,000,000`, `W=round(.04*T)=40,000`, `eta_max=.0006`. Code applies

\[
\eta(s)=\eta_{max}\begin{cases}(s+1)/W&s<W\\
\tfrac12[1+\cos(\pi\min(1,(s-W)/(T-W)))]&s\ge W.
\end{cases}
\]

This is 4% response-token warmup followed by cosine decay. The `+1` virtual-step offset adds `1.5e-8` during warmup, negligible here. Warmup ends around update 2,278 / 21.6M response tokens, rather than the paper's 146 updates. That absolute-step difference follows the intentionally changed batch/campaign and is not a failure to apply warmup.

| Update | Logged pre-batch schedule progress | Logged/applied LR |
|---:|---:|---:|
| 1 | 0 | 0.000000015 |
| 1,000 | .01777357037 | .000266618556 |
| 2,000 | .03510309444 | .000526561417 |
| 2,278 | .04000197222 | .000600000000 |
| 3,000 | .05222616296 | .000599759912 |
| 4,516 | .07926744259 | .000597526476 |

Maximum residual of logged LR against the implementation's formula: **0.0** across 4,516 rows. Maximum residual of curriculum probability: **0.0**. Maximum residual of logged lambda against `beta * logged_ce`: **1.27e-6**, consistent with FP32 averaging. Thus neither a stuck LR, missing warmup nor stale curriculum explains the symptom.

### Batch/LR assessment

In packed mode `batch_size=4` does not create four sequences inside each microbatch. Each microbatch is **one** sequence with 256 anchors (`data.py:517–580`); accumulation 4 gives four sequences total. Increasing `batch_size` alone would not increase the packed training batch.

Matching the paper at 256 anchors/sequence would require accumulation 224 for the same anchor count, but only accumulation 112 for the same independent-sequence count. These are different matching criteria. Blindly increasing anchors within one sequence also does not restore sequence diversity; anchors are correlated and sampled with replacement.

A square-root anchor-count heuristic gives `6e-4/sqrt(56)=8.02e-5`; a sequence-count heuristic gives `1.13e-4`. Neither is a theorem for this model/Adam setup. I would test a peak LR of **1e-4** against 6e-4 at equal response-token exposure after fixing/choosing lambda. Proposed isolated-fork change:

```diff
--- a/fork-run.json
+++ b/fork-run.json
@@
-  "lr": 0.0006,
+  "lr": 0.0001,
```

Alternatively test more sequences per update by increasing `gradient_accumulation`, holding the response-token schedule fixed. Do not infer that the paper LR must fail: the current CE/L1 and tau are improving. Archived GPU timings (`dspine/results/prerun/gpu/batch-expanded_quality.json`) show 256 anchors take 1.183s/update versus 512 taking 2.684s at the same accumulation, so simply doubling anchors was not a throughput win in that measurement.

## 6. Live metrics: anomalies and limits

100-step means, historical production objective:

| Window | CE block sum | L1 block sum | Backbone | Refinement | Embedding | Lambda | Auxiliary, exact | Preclip norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–100 | 50.36 | 12.21 | 16.03 | 3.95 | 25.26 | .42 | 10.82 | 1.81 |
| 901–1,000 | 42.65 | 10.81 | 13.99 | 4.40 | 23.31 | 8.99 | 217.71 | 14.04 |
| 1,901–2,000 | 39.21 | 10.11 | 13.02 | 4.37 | 21.88 | 16.76 | 384.13 | 19.97 |
| 2,901–3,000 | 38.48 | 9.87 | 12.73 | 4.23 | 21.76 | 19.24 | 439.96 | 17.63 |
| 3,901–4,000 | 36.93 | 9.55 | 12.29 | 4.15 | 21.09 | 18.47 | 411.54 | 15.97 |
| 4,417–4,516 | 35.30 | 9.33 | 11.93 | 4.13 | 20.68 | 17.65 | 386.62 | 15.10 |

All **4,516/4,516** updates have norm above 1.0. Preclip norm min/median/p90/p99/max: **1.142 / 15.863 / 21.355 / 29.124 / 56.826**. The largest spike is update 2,223 near the end of warmup. `clip_grad_norm_` returns the norm **before** clipping; values above 1 are not evidence that clipping failed. The CPU optimizer test verifies postclip norm is approximately one.

The auxiliary scalar is about 24 times combined backbone/refinement in the last 100 steps. Loss rising during warmup is mainly the ramping coefficient, not CE divergence. CE, L1 and embedding alignment all improve overall; refinement worsens initially then changes slowly, and its aggregate obscures CE eligibility, candidate target mass and L1. There is no logged per-term **gradient** information and no separate refinement CE/L1 or candidate-coverage trends. Therefore production gradient dominance and candidate starvation remain hypotheses to measure, not conclusions from scalar losses alone.

Useful diagnostic logging, proposed for an isolated probe or normal reporting after review:

```diff
--- a/dspine/losses.py
+++ b/dspine/losses.py
@@
         "refinement": refinement.detach(),
+        "refinement_ce": reduce(rce).detach(),
+        "refinement_l1": reduce(rl1).detach(),
+        "candidate_label_coverage": (
+            (candidates == labels[:, 1:, None]).any(-1) * w
+        ).sum().detach() / w.sum().clamp_min(1),
+        "candidate_target_mass": (
+            candidate_targets(candidates, target_ids[:, 1:], target_probs[:, 1:])
+            .sum(-1) * w
+        ).sum().detach() / w.sum().clamp_min(1),
```

## 7. Restricted-head deviation: measured size and proposed comparison

The matching archived policy hash in `dspine/results/prerun/hot-coverage-sweep.json` is `743ab092...119db`, identical to `loss/live-run.json`. On its 5,040,346-token proxy cache, mean retained target probability is .99968835 and sampled-label coverage .99975121. On 34,161 validation tokens the retained mass is .99992073. Some rows still have zero retained mass. This is proxy teacher-prefill coverage, not proof of current actual-decode coverage on all domains or later rolling windows.

The average theoretical unavoidable L1 floor from missing mass is `2*(1-.99968835)=.00062330` per token in that cache. This is much smaller than current measured losses and is unlikely to explain the large tau shortfall by itself. Rare domain-specific exclusions can still be harmful. Preserve outside mass in L1; do not “fix” it by renormalizing p onto the hot vocabulary.

If coverage is worse on actual held-out trajectories, use the existing full-backbone diagnostic first:

```diff
--- a/fork-run.json
+++ b/fork-run.json
@@
   "proposal_vocabulary": {
@@
-    "loss": "restricted",
+    "loss": "full_backbone",
```

This restores full-vocabulary backbone CE/L1 while still choosing refinement candidates from the hot head. It is a diagnostic, not strict paper equivalence: strict equivalence would also use full-vocabulary candidate generation and the same frozen head for both branches. There is no proposed correction to the currently correct restricted L1 arithmetic.

## Decisive GPU test, at most 20 minutes

**Attach diagnostics to the TRAINER agent's existing lambda A/B, when the training owner schedules it. Do not run a second overlapping test.** Starting from the same current checkpoint, optimizer moments, RNG, sequence/anchor trace and token progress:

1. **Up to 90 seconds:** on 4–8 real microbatches, measure separate parameter gradients for `.1 CE_bb`, `.9 L1_bb`, `.1 CE_ref`, `.9 L1_ref`, and `lambda*Emb`, including their norms/cosines on shared trunk and first-layer attention. Report pre/post clipping and candidate coverage/mass by position. Compare historical lambda and lambda divided by 6.63061, with the same fixed activations.
2. **Up to 60 seconds:** on a few real hidden rows, compare the chunked CUDA BF16/row-INT8 loss backward with ordinary autograd using the same frozen rows, and compare logits with serving's fused `HotInt8Head` CUDA path. Use chunks 4,096 and a non-divisor, preserve sparse outside mass, and include deliberately tied rows. Expect FP32 errors near numerical precision and BF16 relative gradient error comfortably below 1%; large discrepancies would promote the kernel concern to a confirmed bug.
3. **About 13–15 minutes:** run **300 updates per arm**, historical vs normalized lambda, at the same LR/token schedule and exact paired batches. At the archived/current ~1.2–1.4 seconds/update this is 12–14 minutes; enforce an 18-minute total training/diagnostic deadline and reduce paired update count equally if slower. Do not reset Adam moments, curriculum or warmup.
4. **Remaining 1–2 minutes:** evaluate the same fixed complete-round held-out trajectories at steps 0 and 300, emphasizing T=1 b6/b8 tau, per-position acceptance, backbone candidate label/target mass, and refinement CE/L1 on eligible positions. Opening tau alone is not the decision metric.

The decisive evidence is whether the normalized arm restores a substantial distillation-gradient component and improves candidate coverage/complete-round tau at matched data. Scalar total loss is incomparable across the two objectives. This short test can settle gross gradient starvation and establish an A/B direction; a tiny tau difference would still need a larger held-out/longer run. Test LR/batch or the global-microbatch coefficient only afterward so the interpretation stays clear.
