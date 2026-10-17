> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# BUILDER-CLEANROOM differential audit

Completed on 2026-10-17, before 03:15 EDT. No confirmed structural implementation
bug was found in the CPU paths tested. The largest paper-interpretation issue
is the already-known lambda reduction, owned by TRAINER. The running restricted
proposal head is an intentional departure from the paper's full-vocabulary
training objective. Neither finding is a new causal explanation of low tau.

## Independence and artifacts

Phase 1 was committed as **`2e69a17`**, with commit message
`audit: clean-room DSpine reference written without reading dspine/`.
`ref.py` and `AMBIGUITIES.md` were written from the local paper's Sections
4.1–4.5, Appendix A and Table 4, plus SPEC's adapted target shape/transformer
contract. No `dspine/*.py` was opened or imported until that commit succeeded.
The committed phase-1 `ref.py` has not been modified in phase 2.

The reference independently implements dense block-causal attention, context
fusion, Qwen3 pre-norm/GQA/RoPE/SwiGLU, anchor/mask embeddings, every-layer
parallel injection, PCA whitening, post-injection embedding supervision,
top-16 candidate generation, correct-predecessor refinement and candidate-pair
transition tables. It scatters sparse p into a dense vocabulary tensor and
uses ordinary autograd, independently of the implementation's sparse L1
identity and custom backward. Interpretation switches are documented in
`AMBIGUITIES.md`.

Phase 2 adds a name/shape-only weight adapter in `compare.py`. Distinct
Parameters share immutable weight storage to bound CPU RAM, but retain
independent gradients. No production helper is used inside `ref.py`.
`policy.py` is an explicitly phase-2 extension for the run's restricted head,
independently computing its full-vocabulary L1 by scattering restricted q.

All changes are under `docs/audit/cleanroom/`. No `dspine/` file was edited.
Local Python 3.13 / PyTorch 2.6.0 ran the comparisons on the Mac CPU. No GPU,
optimizer state or training unit was used; no service or production project
artifact was changed. The only new remote-to-Mac data copy was an 11,583,734-byte
finished feature shard, copied with normal-priority `scp`.

The step-4516 model was reused from `/tmp/audit-model/ckpt`, linked at
`/tmp/cleanroom/step-00004516`. Both remote and local SHA256 are
`1c7155610d9cc0e8979c203dfba4d8e53913a179866ec7e49b985a1c3d77a073`.
It contains 3,257,047,192 bytes of model weights, without trainer state.
The reused external embedding/head copies remain in `/tmp/audit-model/target`.
`RECEIPT.json` records paths, sizes, hashes and feature metadata. This audit
did not independently hash those external raw table copies against their
remote safetensors ranges, so their source identity is inherited from
AUDIT-MODEL-EVAL. Group-128 W8 reconstruction was independently implemented and
checked exactly against the production formula on a small BF16 matrix.

## Ranked findings and exact counterfactual fixes

### 1. RISK: lambda's “token CE” interpretation changes auxiliary pressure 6.63x

**Evidence:** `dspine/losses.py:250–254,278–285`; Appendix A, Losses.
The implementation defaults to position-weighted CE summed over positions,
averaged over blocks. The paper explicitly sums positions in the two losses,
but calls the detached multiplier “token cross-entropy” without specifying
its reduction. This is an interpretation difference, not a confirmed bug.
This corroborates TRAINER's existing suspect; no duplicate A/B was run.

For 15 predicted positions, `sum(exp(-t/7)) = 6.630610787009076`.
On the real checkpoint, W8 target head, N=2/W=8 batch, curriculum off:

| Lambda interpretation | lambda | Total loss |
|---|---:|---:|
| Current position sum | 12.24711435 | 234.21282528 |
| Weighted mean over valid predicted tokens | 1.84705674 | 45.90975167 |
| Unweighted mean over valid predicted tokens | 2.26887792 | 53.54723100 |

`L_emb` is 18.10596448 and distillation is 12.46700793. Only the multiplier
changes in this comparison; it is detached in all branches.

The current TRAINER A/B's `weighted_token_mean` switch divides by the nominal
block weight mass. That equals this reference's weighted mean for full blocks.
On tail-masked batches, an actual valid-token weighted mean instead divides
by the active mass. This is a second denominator choice, documented here
without claiming the tail convention is a bug.

An exact optional alternative, if the team chooses valid-token weighted CE,
would be the following change inside the existing opt-in branch, not an
unconditional replacement of the default:

```diff
--- a/dspine/losses.py
+++ b/dspine/losses.py
@@
     if lambda_ce_reduction == "weighted_token_mean":
-        coefficient = coefficient / weight_mass
+        token_ce = (ce * w).sum() / w.sum().clamp_min(1e-30)
+        coefficient = beta * token_ce.detach()
```

For unweighted token CE, replace that branch's `token_ce` expression with
`(ce * active).sum() / active.sum().clamp_min(1)`. Reference switches support
both. Use TRAINER's already-planned <=20-minute A/B to select a convention;
this audit does not establish which one improves tau.

### 2. DEVIATION FROM PAPER: the saved run trains a restricted, row-INT8 head

**Evidence:** `dspine/configs/qwen38-flash-next-full.json:96–103`,
`dspine/losses.py:156–188`, `dspine/modeling.py:343–365`.
The paper's backbone trains full-vocabulary q. The saved checkpoint policy
uses 189,090 proposal tokens, row-INT8 effective rows, and masks CE on labels
outside the hot set. This differs from Section 4.5 and SPEC's v1 training
contract. It is explicit, not an accidental masking implementation error.

On the identical real states and target batch, curriculum off, current policy
versus full-vocabulary W8 head changes CE from 24.49424553 to 24.46652412 and
total loss from 234.21302795 to 233.95915222. Fifty of 480 ranked candidate
slots differ, but only **4 of 480 candidate set members** differ; most slot
differences are reordering caused by row quantization. With fixed curriculum,
52 ranked slots differ and candidate set membership is unchanged; total loss
changes from 195.25187683 to 195.32568359. These two blocks cannot estimate a
tau effect.

The current restricted objective itself matches independent dense math:
maximum total discrepancy 1.8311e-4, matching candidate IDs in both curriculum
conditions, maximum parameter-gradient relative error 6.843e-4.

For an explicitly new paper-exact training fork, an exact way to clear the
policy loaded from a checkpoint and suppress reconfiguration is:

```diff
--- a/dspine/train.py
+++ b/dspine/train.py
@@
-    if run.get("proposal_vocabulary"):
+    if run.get("paper_full_vocabulary", False):
+        model.proposal_token_ids = None
+        model.proposal_weight = None
+        model.proposal_lookup = None
+        model.proposal_policy = None
+    if run.get("proposal_vocabulary") and not run.get("paper_full_vocabulary", False):
```

Set `paper_full_vocabulary: true` only in that new fork's configuration. This
changes candidate generation, CE, L1 and refinement consistently. Merely
setting the existing policy's loss to `full_backbone` still leaves restricted
candidates, so it is not a fully paper-exact counterfactual. Do not alter the
frozen production run or call such a fork an exact resume. A bounded later
test could compare fixed-checkpoint tau for both heads and then <=100 updates
per fork within a 20-minute GPU slot, with matching data and lambda convention.

### 3. OK / numerical RISK: wide softmax and chunked log-normalizer precision

**Evidence:** `dspine/losses.py:39–47,69–74,89–103`.
The original pure-FP32 dense reference initially differed by total loss
0.011322 and a worst parameter-gradient relative error of 0.003336 on the real
checkpoint. This was largely a reference-side numerical issue: native CPU
FP32 `log_softmax` across 248,320 classes loses small denominator terms on
high-confidence positions. Using identical logits, its CE differs from a
float64 softmax by up to 8.484e-4, and its hidden gradient by relative 0.001754.

The production chunked loss is closer to float64 math. With chunk size 4096,
isolated CE differs by at most 5.066e-6 and L1 by 9.470e-6. Remaining chunk
normalizer/gradient arithmetic has hidden-gradient relative error 3.891e-4.
With one full-vocabulary chunk, that error falls to 2.685e-6. The analytic
backward is therefore corroborated rather than disproved by this comparison.

End-to-end, using float64 reference softmax, the ordinary 4096-chunk real W8
comparison has worst parameter error 0.001121 and total discrepancy <=1.984e-4.
The raw-head single-chunk countercheck lowers worst parameter error to
3.086e-5 across all packed/non-packed and curriculum cases. Candidate IDs
remain identical. This magnitude is not evidence for the observed tau deficit.

No production arithmetic change is recommended from this CPU-only result.
An exact diagnostic counterfactual for higher-precision normalization is:

```diff
--- a/dspine/losses.py
+++ b/dspine/losses.py
@@
-        lse = torch.full((n,), -torch.inf, device=hidden.device)
+        lse = torch.full((n,), -torch.inf, device=hidden.device, dtype=torch.float64)
@@
-            lse = torch.logaddexp(lse, logits.logsumexp(-1))
+            lse = torch.logaddexp(lse, logits.double().logsumexp(-1))
```

This is a diagnostic proposal, not a performance-reviewed fix. On a later
<=20-minute GPU slot, compare fixed BF16 training microbatches against the CPU
reference, including hidden-gradient agreement, and benchmark 20 iterations
of the normal and float64-normalizer paths before considering adoption.
BF16 head-gradient projection rounding remains separate from logZ accuracy.

### 4. OK: literal RMS and transfer-table storage precision

**Evidence:** `dspine/modeling.py:263–269`, `dspine/transfer_space.py:19–36`.
Equation 5 writes mathematical RMS without epsilon; the implementation adds
1e-6. Toggling only this interpretation changes the tiny total by 9.155e-5 and
worst parameter gradient by relative 1.3174e-5. The literal alternative is:

```diff
--- a/dspine/modeling.py
+++ b/dspine/modeling.py
@@
-            u.float().square().mean(-1, keepdim=True) + self.c.rms_norm_eps
+            u.float().square().mean(-1, keepdim=True)
```

Retaining the epsilon is reasonable numerical regularization; there is no
acceptance-based reason here to change it.

PCA C differs from the float64 mathematical reference by maximum 0.00192249
and relative 0.00163086 solely because production stores BF16. After BF16
rounding and eigenvector sign alignment, C is **bitwise equal**. The basis,
mean and eigenvalues agree at their stored FP32 precision. No fix is needed.
Ordinary FP32 RoPE, attention and reduction ordering also explain residual
forward differences around 1e-6 relative, with no changed candidate IDs.

## Coverage and numerical results

All gradient errors use `||g_impl - g_ref||2 / max(||g_ref||2, 1e-30)` per
production parameter tensor. All **77 trainable tensors** receive gradients
and are compared in every case; embedding and target head are frozen.
JSON results retain every tensor's error, norms and maximum absolute error.

| Test | Cases | Largest total-loss difference | Worst parameter-gradient relative error |
|---|---:|---:|---:|
| Tiny L6, explicit attention, includes literal RMS toggle | 5 | 9.155e-5 | 1.318e-5 |
| Tiny L6, CPU SDPA plus packed CPU path | 5 | 9.155e-5 | 1.308e-5 |
| Tiny L6, non-reentrant gradient checkpointing | 5 | 9.155e-5 | 1.318e-5 |
| Random real L6/H2560/V248320 shape | 2 | 2.441e-4 | 9.201e-6 |
| Step 4516 raw head, original FP32 reference softmax | 4 | 1.133e-2 | 3.337e-3 |
| Step 4516 raw head, float64 reference softmax | 4 | 1.374e-4 | 1.130e-3 |
| Step 4516 raw head, one chunk, float64 reference softmax | 4 | 1.984e-4 | 3.086e-5 |
| Step 4516 effective W8 head, float64 reference softmax | 4 | 1.984e-4 | 1.121e-3 |
| Step 4516 effective W8 + actual restricted policy | 2 | 1.832e-4 | 6.843e-4 |

Tiny uses N=3, B=6, W=7, H=32, r=12, K=16 with nonzero randomized writes and
predecessor gates. This exercises live gradients through every injection
parameter rather than testing only identity initialization. It includes
left-padded contexts, nonzero absolute positions and tail-masked losses.

The full random model uses the exact real L6/H2560, 32x128 query / 8x128 KV,
intermediate 10240, feature width 12800, r=1024, message=512, vocabulary=248320,
B=16, K=16, with N=2/W=8.

Real data is a finished 890-token code sequence with response start 205,
anchors 205 and 224, K=16 and W=8. It contains FP8 features plus joint-cutoff
tie supports. The effective frozen W8 table is reconstructed independently;
FP32 master drafter weights are retained exactly. Frozen BF16 values are
represented exactly in FP32 for the mathematical CPU comparison. This is
not a claim of bitwise BF16-autocast parity.

For the real W8 cases, maximum final hidden error is 2.337e-5, hidden relative
error <=1.278e-6, and maximum full-vocabulary logit error is 4.912e-5. Backbone
loss differs by <=1.622e-5, refinement by <=2.981e-6, and embedding loss by
<=3.815e-6. Large raw residual states have maximum absolute error 0.03223 but
relative error about 1.4e-6; final normalization removes their scale.

Curriculum off and on use identical inputs. On uses fixed uniforms
`linspace(0.17,0.83,N)` and threshold 0.5, shared across layers 1–3 of each
block. The reference and implementation agree on selected blocks and all
downstream outputs/gradients. Position 1 always receives the known anchor.

`checks.py` separately verifies:

- PCA centering, row normalization, whitening and signs, including exact BF16 C.
- Refinement CE membership independently from the L1 target-mass threshold,
  including target masses exactly at, just below and just above 1e-4, and zero.
- Zero-probability duplicate-ID padding in the sparse target head loss.
- Curriculum schedule boundaries and the shared-draw interpretation.
- Direct independently indexed real labels, p arrays, features and context
  positions against both AnchorStream/collation and PackedSequenceStream.
  All indices and valid supervised values match exactly; no extra token shift.
- Independent group-128 W8 reconstruction against production's definition.

The reference self-check also verifies identity zero-write initialization,
anchor invariance, causal prefix invariance, first-position last-write
equivalence, and finite gradients.

## Reproduction and remaining limits

```sh
python3 docs/audit/cleanroom/ref.py
python3 docs/audit/cleanroom/compare.py --mode tiny --output docs/audit/cleanroom/tiny.json
python3 docs/audit/cleanroom/compare.py --mode tiny --sdpa --output docs/audit/cleanroom/tiny-sdpa.json
python3 docs/audit/cleanroom/compare.py --mode tiny --gradient-checkpointing --output docs/audit/cleanroom/tiny-checkpointing.json
python3 docs/audit/cleanroom/compare.py --mode full --output docs/audit/cleanroom/full.json
python3 docs/audit/cleanroom/compare.py --mode real --output docs/audit/cleanroom/real.json
python3 docs/audit/cleanroom/compare.py --mode real --softmax64 --output docs/audit/cleanroom/real-fp64-softmax.json
python3 docs/audit/cleanroom/compare.py --mode real --softmax64 --loss-chunk 248320 --output docs/audit/cleanroom/real-single-chunk.json
python3 docs/audit/cleanroom/compare.py --mode real --effective-w8 --softmax64 --output docs/audit/cleanroom/real-w8.json
python3 docs/audit/cleanroom/policy.py
python3 docs/audit/cleanroom/precision.py --output docs/audit/cleanroom/precision.json
python3 docs/audit/cleanroom/checks.py
python3 docs/audit/cleanroom/validate.py
```

Validation passes **35 differential cases**, each comparing all 77 trainable
tensors, plus the isolated adversarial checks. The JSON files include source
SHA256s to identify the concurrently evolving implementation under test.

The CPU packed path's mask and semantics are covered, but CUDA FlexAttention
compilation, GPU fused kernels, BF16-autocast rounding, long W=4096 contexts,
real-batch stochastic selection statistics and any causal effect on tau remain
outside this result. A decisive later <=20-minute GPU parity check should run
the same checkpoint and fixed data with reference/off and FlexAttention/on,
curriculum off and the same explicit teacher blocks, W in {8,128,4096}, N=2,
and compare all readouts, per-term losses and gradients. Export compact results
and selected tensor slices for CPU inspection. No GPU test was executed here.
