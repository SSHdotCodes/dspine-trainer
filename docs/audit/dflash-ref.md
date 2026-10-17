> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# AUDIT-DFLASH-REF: What the working reference actually trained

2026-10-17. Research and CPU only. No GPU use, service changes, or edits to `dspine/`.

## Result

**The strongest actionable difference is the training objective and its allocation of early-position signal, not a missing context-feature injection.** The private DFlash2 draft reads context through the same `hidden_norm(fc(features)) → per-layer K/V` mechanism as ours. It does not add aligned target features to masked-token inputs. Its successful `fc` is actually slightly smaller than ours at step 4516.

The reference used output cross-entropy, short blocks initially, eight distinct sequences per update, and many passes over its small corpus. Our current recipe still combines 90% L1 with a six-layer directional auxiliary loss. Recipe C fixed the earlier lambda normalization; that does **not** establish that the remaining auxiliary weight is appropriate for this target or stage of learning.

**First recommendation:** fork the current checkpoint and run 300 paired steps with the auxiliary coefficient set to zero, keeping every other recipe-C choice and optimizer state. Measure actual depth-1 acceptance and context ablation, not opening tau. The complete experiment fits within 20 GPU minutes; protocol below. This is a **RISK/hypothesis**, not a confirmed implementation bug or a claim that DSpine's published objective is wrong.

## What DFlash2 measured at depth 1

The backup's **0.969× MTP is a throughput ratio**, not acceptance. It is the calibrated `kd8-best-007000` model, served with block 6, selector K32, and unary-logit multiplier 1.6. The model card reports 169.218 geometric-mean end-to-end tokens/s and **2.893 request-mean useful tokens per verification**. The corpus contains **8,005,796 unique generated output tokens**, not nine million optimizer-training tokens.

I extracted the saved benchmark archive and reconstructed the missing depth breakdown from `spec_correct_drafts_histogram`. Histogram bin `j` counts rounds accepting exactly `j` draft tokens, excluding the verifier token. Therefore:

`depth1 = 1 − histogram[0] / sum(histogram)`

| Same historical benchmark, sampled requests only | Calibrated DFlash2, b6 | Native MTP, b4 |
|---|---:|---:|
| Requests | 24 | 24 |
| Verification rounds | 6,353 | 6,048 |
| **P(at least 1 draft accepted), pooled rounds** | **0.63986** | **0.76306** |
| P(at least 2 accepted) | 0.41051 | 0.58052 |
| P(at least 3 accepted) | 0.26680 | 0.44015 |
| P(at least 4 / 5 accepted) | 0.17692 / 0.11333 | unavailable |
| P(position 2 accepted given position 1 accepted) | 0.64157 | 0.76078 |
| Pooled `1 + accepted drafts / rounds` | 2.60743 | 2.78373 |
| Request-mean reported tau | **2.89347** | **2.84725** |
| Request-mean depth-1 acceptance | 0.69573 | 0.78704 |

Different round counts weight requests differently, explaining the pooled/request-mean gap. Use the pooled depth-1 figure when discussing all verification rounds. Both histogram identities, sum of bins = verify count and weighted sum of bins = accepted-draft count, pass for every request. No independence assumption or `tau → acceptance` conversion was used.

| Domain | DFlash2 pooled depth 1 | MTP pooled depth 1 |
|---|---:|---:|
| Code | 0.7595 | 0.8298 |
| General | 0.5306 | 0.6909 |
| Tools | 0.6692 | 0.7767 |

Evidence: [`reference-summary.json`](dflash-ref/reference-summary.json), reproducible with `summarize_reference.py` (`dflash-ref/summarize_reference.py`, retained internal evidence), plus copied DFlash measurements (`dflash-ref/private-reference/evaluation/dev-calib-b6-k32-m1.6-dflash/measurements.jsonl`, retained internal evidence) and MTP measurements (`dflash-ref/private-reference/evaluation/dev-kd8-mtp/measurements.jsonl`, retained internal evidence). Warmups and synthetic greedy checks are excluded.

**Context use was not measured in the retained DFlash2 evidence.** I found no W=1/W=8 ablation, shuffled-context control, or matched no-context evaluation. The surviving offline scores are greedy sampled-token prefix length and teacher-forced overlap-chain metrics, not the current complete-round T1 evaluator. Per-position diagnostic code existed, but its detailed validation JSON/logs were not retained in the backup. The original project, raw features, optimizer states, and other checkpoints were deleted after export; the historical Mac review export adds aggregate scores, not a context ablation.

Consequently, the older draft is evidence that this architecture and target family can learn substantially better first-token acceptance. It is **not** evidence that it reached 0.97 acceptance, or proof that it learned long-context retrieval. Its target was RadixArk revision `7b719225...`, whereas ours is RedHatAI revision `c8f2fb1...`; cohorts, reasoning mix, output depths, proposal policy, and runtime also differ. The backup itself records unresolved greedy/runtime equivalence issues. Our user-provided step-5543 overlap 0.442 and native-MTP ≈0.65 must not be treated as a matched comparison to the historical 0.640/0.763.

## Item-by-item comparison

Reference source root below is `/tmp/dflash-ref`, extracted on this Mac from the workstation's `./work/ref/dflash2-backup/training-source.tar.gz`. `SF` denotes `vendor/SpecForge/specforge/`; all line numbers below refer to that extracted source. Private project files are copied under `dflash-ref/private-reference/` (`dflash-ref/private-reference/`, retained internal evidence). Archive hash and current source hashes are in `source-receipt.json` (`dflash-ref/source-receipt.json`, retained internal evidence).

| Item | Working private DFlash2 | Ours, promoted recipe C | Assessment |
|---|---|---|---|
| Context projection | One bias-free 12800→2560 `fc`, then trainable RMSNorm | Same shape and order | **OK**, same mechanism. `SF/modeling/draft/dflash.py:710–724,798–809`; `dspine/modeling.py:327–330,406,459` |
| Initialization | HF `post_init`, normal std 0.02 for linear projections; norm scale 1 | Default PyTorch linear initialization: fc std ≈0.00510, q/k/v std ≈0.01141; norm scale 1 | **RISK**, different from reference at initialization; not evidence for rescaling trained weights |
| Feature scale | No explicit context multiplier after normalization | No explicit context multiplier. `feature_scale` controls bridge token codes, not context amplitude | **OK**. `modeling.py:254–261,406`; do not confuse these two scales |
| Where features enter | Context K/V at every draft layer, separately projected by that layer; no context Q or context residual/MLP pass | Same | **OK**. `SF/modeling/draft/dflash.py:367–392,584–602,798–817`; `modeling.py:85–98,211–215,459–490` |
| Add target features to block input? | **No**. Block input is anchor embedding plus mask-token embeddings | **No**. Anchor plus learned mask vector | **OK**. Adding feature at the anchor would use a verifier state not yet available for that bonus/residual token |
| Same target layers in every draft layer? | Yes. Concatenate post-layers `[3,13,23,33,43]` once and reuse fused tensor | Yes. Post-layers `[1,12,23,34,45]`, reused in all six layers | **OK** structurally. Not a target-layer-to-draft-layer correspondence. Different layer IDs are a contract change, not a mid-run config tweak |
| Hyper-connections | Four post-layer streams averaged in FP32, cast to BF16 | Same reduction; stored features then FP8 compressed/dequantized | **OK** reduction; **RISK** additional compression. `patches-history/qwen4_exp-contract1.py:1672–1680`; `dspine/capture.py:73–92`; `data.py:112–146` |
| Anchor | A sampled real response token at absolute position `a`; first predicted token is `a+1` | Same | **OK**. It represents the known last token at the next draft round; its own target feature is excluded. It need not be the last *draft* token accepted |
| Corruption/noise | `[embedding(ids[a]), embedding(mask_id), …]`; all future positions masked | Same pattern, using a learned mean-initialized mask vector | **RISK**, minor reference difference. No random Gaussian noise or random partial unmasking in either recipe |
| Anchor sampling | Unique random anchors, valid anchor and next response label | Unique random anchors in C, padded dummy slots loss-masked | **OK**, the old duplication issue is already fixed. `src/dflash_distill_base.py:429–501`; `data.py:561–606` |
| Positions | Original context positions, block positions `a+t`; anchor excluded from target features | Same, plus stored sequence-start offset | **OK**. `src/dflash_distill_base.py:471–477,702–706`; `modeling.py:143–158,402,453` |
| Attention masks | Actual private config is all sliding layers, W=2048. Within-block causal; context `<a`, lower bound `a+t−(W−1)` | Causal within block; context `a−4096,a)` shared across positions | **OK** direction/availability. Small window-edge difference; ours has a larger context allowance, not less |
| Full-attention DFlash | Public full-attention mode can see all slots in its own masked block | DSpine intentionally stays block-causal | **OK**, a designed DSpine difference. Do not turn bidirectional attention while teacher-token injection exists |
| Architecture | 5 layers, 20 Q heads/4 KV heads, head dim 128, FFN 6912, grouped conv + selector rank 256 | 6 layers, 32/8 heads, FFN 10240, DSpine bridge r=1024/message 512 | **RISK** optimization/capacity differences, not a missing context path. No mid-run tensor-compatible conversion to DFlash2 |
| Block schedule | **b4 pretraining**, then **b8 KD**, later b6 serving | **b16 throughout**, serve/evaluate b6/b8 prefixes | **RISK**, much less early-position training per head/loss budget |
| Backbone loss | Initial weighted **hard CE**; KD **0.9 full teacher CE + 0.1 sampled-label CE** | **0.1 sampled-label CE + 0.9 L1** against processed top-k/top-p target p | **RISK**, strongest objective difference. Our choice follows DSpine Appendix A; it is not a loss-kernel bug |
| Auxiliary losses | Selector categorical CE, ramped over first 100 steps. No layer-wise cosine/feature loss | Six-layer cosine supervision, lambda = 0.5×mean token CE after warmup, plus equal-weight last-write refinement | **RISK**, most direct ablation to run first |
| Position weighting | Same `exp(−(t−1)/7)` in both private stages, **normalized by total supervised weight** | gamma=7; sums over positions, averages anchors. Lambda alone divides by nominal weight mass | **RISK** relative signal allocation; blanket normalization of every term does not fix the relative auxiliary/CE issue |
| Teacher targets | KD uses a final hidden state at label position minus one, full frozen BF16 head, unprocessed full softmax | Cached processed sparse p, W8 target head; row-INT8 restricted draft head of 189,090 IDs | **RISK** objective difference; head/vocabulary changes are not the first fix. `src/dflash_distill_base.py:535–552,885–895`; `losses.py:147–188` |
| Refinement | Selector CE restricted to strict top-K candidates; KD soft selector CE weighted by teacher candidate coverage | Repeated final bridge write, CE if gold candidate exists, candidate-renormalized L1; C skips refinement at position 1 | **OK**, intentional architectures differ. C already fixed double refinement at t1 |
| Optimizer | AdamW (.9,.95), decay .01, clip 1; fresh optimizer for KD | AdamW (.9,.999), decay 0, clip 1; optimizer retained across recipe forks | **RISK** adaptation speed after changing objectives; zero decay agrees with DSpine paper |
| LR/warmup | Initial 3e-4, KD 1e-4; 100-step warmup, cosine with .15 floor | Peak 3e-4 in C, inherited 4%-of-540M-token warmup/cosine-to-zero progress | **OK** same order of magnitude. Current LR ≈2.97e-4; not still in warmup |
| Optimizer batch | 8 sequences × up to 128 anchors initially; 8×96 in KD | 4 sequences × up to 256 anchors | **RISK** sequence diversity. Block counts are comparable, not 56× smaller than this working same-target reference |
| Reuse/epochs | Initial ends at step 20,000/epoch 22; chosen initial checkpoint 17,214. KD ends 7,843/epoch 8; chosen at 7,000 | Rolling corpus, planned 2.25 passes; ≈55M response-token exposures at the later audit snapshot | **RISK** misleading “only 8M tokens” comparison: reference repeatedly reused that corpus |
| Data/thinking | Target-generated T1/top-p .95/top-k20, mixture of low/medium/xhigh reasoning; code/general/tools; prompt and response caps 1536 each | Target-generated reasoning continuations, wider domains/lengths, same nominal T1 sampling policy | **OK** thinking is not uniquely new to our failing run. Tencent paper used thinking disabled, but the working same-target reference did not |
| Training context length | All source sequences ≤4096, padded to 2048/4096; generation caps imply nominal total ≤3072. Actual attention is sliding 2048 | Full-prefix target capture, stored left context up to4096 and full response; draft attention W4096 | **OK** ours is not deprived of context by a shorter configured window. Actual usable lengths vary by anchor |

Important source locations: [initial trainer (`dflash-ref/private-reference/src/train.py`, retained internal evidence) lines 35–43,73–99,113–132; KD trainer (`dflash-ref/private-reference/src/train_kd8.py`, retained internal evidence) lines 76–108,134–166; draft config (`dflash-ref/private-reference/configs/draft.json`, retained internal evidence); generation (`dflash-ref/private-reference/src/generate.py`, retained internal evidence) lines 25–40; prompt effort selection (`dflash-ref/private-reference/src/prepare_prompts.py`, retained internal evidence) lines 44–47. Current config: `promoted-run.json` (`dflash-ref/promoted-run.json`, retained internal evidence), not the stale 6e-4 checked-in full-run JSON.

## Ranked recommendations from the current checkpoint

### 1. RISK, highest priority: test a backbone-recovery phase with no cosine auxiliary loss

The known-working private recipe trains its context reader directly through output CE. It has no six-layer cosine objective. Ours sums that auxiliary term over six layers and 15 positions (`losses.py:294–309`). Even after the lambda fix, the later live snapshot, last 100 updates through step 5935, has mean auxiliary **52.63** versus distillation **14.70**, gradient norm **2.97**, and a clip threshold of 1. Scalars do not establish gradient dominance. The earlier loss auditor's tiny-model experiment found auxiliary/distillation gradient ratios 6.6–8.3 after normalization, including the first attention query projection; those are a warning, not a current-checkpoint measurement. See `promoted-metrics-summary.json` (`dflash-ref/promoted-metrics-summary.json`, retained internal evidence) and [`loss.md`](loss.md).

The hypothesis is that matching every layer to token-code directions is still consuming optimization effort before this target's context reader is good. It need not be intrinsically harmful at convergence. Keep the trained bridge active and trainable; disable only its auxiliary supervision for a bounded recovery phase. Do not reset the model or replace feature layers.

Exact proposed code changes are in `recovery-proposals.patch` (`dflash-ref/recovery-proposals.patch`, retained internal evidence), **not applied**. After adding the knob, the treatment config differs from the paired control by:

```diff
--- a/isolated-control-run.json
+++ b/isolated-noaux-run.json
@@
-  "output": "./work/ckpt/audit-ref/control",
+  "output": "./work/ckpt/audit-ref/noaux",
+  "embedding_loss_scale": 0.0,
```

All existing C choices remain: mean-token lambda branch, unique anchors, LR3e-4, CE support mask, refinement starting at2. Defaults preserve existing behavior. The knob multiplies the effective coefficient after its normalizer, so logged `auxiliary` and effective `lambda` both become zero.

### 2. RISK: give output CE a stronger bootstrap role

Private pretraining uses hard CE, and its stronger KD continuation uses 90% full-distribution teacher CE. Our alpha=.1 is explicitly prescribed by DSpine; it is still a meaningful transfer-risk on this target. L1 supplies weak recovery gradients when a correct token has almost no draft probability. For a one-hot teacher, `L1=2(1−q_y)` and `∇L1=2q_y∇CE`. The present mixture has only `(0.1+1.8q_y)∇CE`: at q_y=.001 it is 0.1018 of pure CE, before the auxiliary loss competes with it. Hard context-content misses are exactly where this distinction matters.

Once the no-aux result is known, compare the same recovery checkpoint with alpha .1 versus1.0. The existing loss already supports alpha; the proposal patch exposes it in run config:

```diff
--- a/recovery-run.json
+++ b/recovery-ce-run.json
@@
   "embedding_loss_scale": 0.0,
+  "loss_alpha": 1.0,
```

This matches the **hard-label pretraining objective family**, not the KD stage. Preserve processed-p L1 as the eventual acceptance-oriented objective if pure CE recovery helps. A closer KD follow-up is sparse teacher CE using existing cached p, with head-logit derivative `m*q − p_hot`, where `m=sum(p_hot)`; retain L1 for outside-hot mass. That is not equivalent to the reference's full raw teacher CE and needs a tested chunked-head extension, not a config-only claim. No new teacher capture is needed for the sparse version.

### 3. RISK: concentrate a short recovery phase on early draft positions

The reference did not learn from 8M tokens once. At the selected checkpoints, the nominal upper bound is **17,627,136 b4 blocks + 5,376,000 b8 blocks =23,003,136 first-position training examples**, before short-sequence/tail effects. Our 5543 steps×1024 slots is at most5,676,032 blocks, with actual count lower after unique-anchor padding. This is roughly **4× less first-position exposure**, despite similar nominal masked-slot work. Reference initial training reached epoch22, followed by roughly eight more KD passes; 8M unique output tokens considerably understate repeated training exposure. Exact token exposure cannot be reconstructed from the retained status files.

At gamma7:

| Block | Masked input fraction | Position-1 loss-weight share | Positions1–3 share |
|---|---:|---:|---:|
| Reference initial b4 | 75% | **38.19%** | 100% |
| Reference KD b8 | 87.5% | 21.06% | 55.14% |
| Ours b16 | 93.75% | **15.08%** | **39.49%** |

In our b16 objective, positions6–15 receive **42.17%** of the position-weight mass, although they depend on unknown future predecessors and currently contribute little useful accepted length. These are loss-budget shares, not measured gradient fractions or a declaration that all suffix supervision is noise.

Avoid changing the checkpoint's structural block size mid-run: `train.py:288–289` rejects architecture changes, and the packed mask also uses that config. A tensor-compatible diagnostic is gamma2 for 250–500 recovery steps, then restore7. Position1 gets39.37% and positions1–3 get77.73%, while positions6–15 drop to8.16%. It preserves the b16 model/inference contract and all weights:

```diff
--- a/recovery-run.json
+++ b/recovery-early-run.json
@@
+  "loss_gamma": 2.0,
```

The proposal patch passes this already-existing loss argument from config and allows it in an explicit fork. Measure full b6/b8 tau before retaining this change: stronger t1 can trade off later-position quality. The DFlash paper itself supports position decay, but it does not establish that gamma2 is optimal here.

### 4. RISK: trade correlated anchors for more sequences at fixed nominal block count

The reference initially used8×128, versus our4×256. Their nominal blocks/update match. The source does **not** support an argument that we simply need56× more blocks per update to match this particular success. At the later snapshot we actually supervise about909 of1024 slots/update because short responses are padded.

Try eight sequences×128 anchors once the objective is settled. This doubles sequence diversity and reduces wasted duplicate-capacity slots on short responses without increasing nominal blocks. It changes which data are consumed per optimizer update, so pair by response-token exposure as well as wall time. It is lower priority than fixing the objective allocation.

```diff
--- a/recovery-run.json
+++ b/recovery-diverse-run.json
@@
-  "gradient_accumulation": 4,
-  "anchors_per_sequence": 256,
+  "gradient_accumulation": 8,
+  "anchors_per_sequence": 128,
```

The proposal patch allows the anchor-count fork. Packed mode actually uses accumulation as its sequence count; changing the separate `batch_size` field alone does not achieve this. Preserved cursor/RNG state resumes from the current data position; newly drawn anchors and subsequent sequence grouping intentionally differ.

### Additional RISK when interpreting recovery tests: inherited Adam history is slow to forget

DFlash2 used beta2=.95 and a **new optimizer** for KD. Ours uses beta2=.999 and carries moments from the older high-auxiliary regime. The saved fork chain changed lambda at4516 and moved into C at5043. At5543, the exponential multiplicative carry from the pre-fix second moment is `.999**1027 =0.35790`. This is not its measured fraction of the current second moment, but it shows why a500-step change can retain substantial history. Beta2=.95 has a13.5-step half-life; .999 has a692.8-step half-life.

For the first A/B, preserve optimizer state in both arms to isolate the loss. If both appear unusually slow, separately test a **named weights-only recovery fork with a fresh optimizer and short LR warmup**, against a retained-moments control. Do not edit beta2 in a loaded optimizer and pretend its old moments/bias correction correspond to the new beta. I do not recommend that unpaired change mid-production. There is no config-only exact-resume fix for stale moments; a new experiment must record the intentional state reset.

## Negative findings: changes that are not justified now

**OK, projection amplitude:** I read the actual preserved DFlash2 weights on CPU, not just its config. RMS values:

| Tensor RMS | DFlash2 selected | Ours step4516 | Tencent DSpine |
|---|---:|---:|---:|
| fc.weight | **0.03101** | **0.03356** | 0.12308 |
| hidden_norm.weight | 0.99480 | 1.00869 | 1.05732 |
| layer0 k_proj | 0.03196 | 0.02785 | 0.08166 |
| layer0 v_proj | 0.03115 | 0.02743 | 0.05736 |
| final norm.weight | 2.26013 | 1.09864 | 1.76880 |

The same-target draft also has nearly uniform per-source-layer fc-block RMS, approximately.029–.034. The older model audit's suggestion that uniform fc blocks and small weights relative to Tencent indicate an immature context path is not diagnostic by itself: the working DFlash2 has both. Different widths, targets and objectives matter.

On64 real training-context positions, our pre-normalization projection RMS is1.2237 and normalized RMS is.9870. Uniformly scaling fc by4 changes its normalized output by only **6.68e-7 relative L2** in FP32. RMSNorm cancels that amplitude, apart from epsilon and floating-point effects. Rescaling parameters can change future optimization geometry, but it does not directly supply a missing context signal. **Do not multiply fc to match Tencent, remove hidden_norm, or initialize over the current checkpoint.** `tensor-summary.json` (`dflash-ref/tensor-summary.json`, retained internal evidence), `tensor_inspect.py` (`dflash-ref/tensor_inspect.py`, retained internal evidence).

**OK, teacher curriculum cannot directly explain t1:** causal attention blocks later-position information from flowing backward; position1's predecessor is always the known anchor (`modeling.py:271–284`). The p=.5 curriculum only replaces predecessors that differ at positions≥2. Lowering p may change shared-weight learning over updates and suffix performance, but toggling it cannot change the fixed-checkpoint t1 forward. A p=.0 test is useful for suffix dependency, lower priority for this depth-1 deficit.

**RISK, initializations are different, but changing them now discards learning:** the default fc initialization was3.9× smaller than HF .02; q/k/v were1.75× smaller. The learned mask grew from RMS.00382 at pilot-init to.03284 at4516. The reference used frozen mask-token248070. These may affect early learning dynamics, but the measured trained context-projection amplitudes already overlap. There is no evidence for a mid-run reinitialization.

**OK, thinking and context length:** the same-target reference also trained reasoning continuations, with a shorter 2048 draft-attention window. Neither turning thinking off nor expanding the current window follows from this comparison. Changing captured target-layer IDs requires matching new features and adaptation, not relabeling cached columns.

**RISK, old calibration is not a training recipe:** output multiplier1.6 was selected after training and improved its live throughput. Its offline calibration also used top-k-renormalized top-p (`calibrate_draft.py:54–57`), whereas our current joint top-k/top-p implementation explicitly avoids that policy mismatch. The previous model audit already found only marginal benefit from proposal sharpening. Copying1.6 is not a context-learning fix.

## Decisive GPU A/B, hard cap20 minutes, not executed

The GPU remains occupied by the authorized trainer. This is a prepared experiment for its owner, not a new job request.

1. **Prepare off-GPU:** apply/review the proposed opt-in knobs; make two separate output roots outside the live run. Fork both from the **same current checkpoint**, including optimizer/RNG/data cursor. Do not use an older4516 parent. Freeze feature-shard availability for the test, record checkpoint and shard fingerprints, preserve all C settings, and restore identical RNG before each arm. Verify default-scale1 gives the old scalar loss on a fixed batch, and scale0 removes only the auxiliary gradient. This is a meaningful new check required by the new knob.
2. **Baseline/probes, ≤2min total:** cache one 18–32-sequence held-out cohort with non-opening anchors at fixed response offsets1,9,17,25,33,41, plus a handful of deeper audit-long anchors. On a common checkpoint measure position1 `Σmin(p,q_actual)` at W4096,64,8,1, using the actual top-K transition-cache proposal and processed target p; report both differences and absolute values. Also log full restricted-head backbone overlap separately to match the older model audit. Evaluate b6/b8 complete-round T1 tau with fixed draws or exact/Rao-Blackwell anchor scores. Also record strict top16 recall and gold-token CE at t1. Count unsupported labels rather than silently including them as CE targets.
3. **Two arms, 300 updates each:** A=`embedding_loss_scale:1`, B=`0`; unchanged alpha=.1, gamma7, LR3e-4, p=.5, accumulation4, unique256. Use `--fork-from` with separate output directories and the same data-window identifier. For both isolated arms set `eval_every:0`, `eval_before_training:false`, `eval_at_end:false`, `checkpoint_every:1000000000`, and `checkpoint_seconds:1000000000`; the proposal patch permits these cadence-only fork changes. Run the explicit probes at the end and retain each final checkpoint. The trainer still saves its final checkpoint when the bounded run ends. Restore the same source state for B. Use the minimum extra evaluation instrumentation needed for the probes.
Use `python3 -m dspine.train --config ARM.json --fork-from CHECKPOINT --data-window WINDOW --stop-at-step END_STEP --max-minutes 6.5`, replacing the placeholders with the frozen run values and setting `END_STEP` to parent step+300. Keep the parent data-window identifier; do not pass `--reset-data` for unchanged data. The wall limit is a backup to the exact stop step, not permission to run longer.

4. **Repeat probes, ≤2min:** all comparisons use the same prompts, offsets, target distributions and seeds. Log actual auxiliary/distillation gradients on one small fixed microbatch for `fc`, layer0 q/k/v, and shared trunk, with gradient cosine; do not infer gradient shares from loss scalar sizes. The two training arms should have equal sequence/anchor fingerprints and equal unsupported-label counts.
5. **Budget and decision:** the live last100 training-step mean is1.222s, so600 updates cost about733s. Allow≤120s startup/exports plus≤240s all probes, totaling about18.2min. Enforce a1200s wall-clock stop, trim updates equally if startup or current contention costs more, and reserve time to export results. This estimate assumes draft-only replay of cached features; no target loading/capture or serving benchmark fits this budget. Run only when the owner schedules it.

Predeclare a useful positive result as ≥.02 absolute t1 overlap gain **over the paired control**, with improved or stable b6 tau and consistent prompt-level direction. Inspect paired prompt differences rather than treating thousands of correlated anchors as independent. A larger W4096→W8 drop is supportive **only when full-window acceptance also increases**; degrading the short-window score alone is not progress. Small/no separation after300 steps is inconclusive about long-run benefit, especially with retained Adam moments. A clear regression rejects this recovery setting.

If no-aux wins, continue that arm for a bounded500–1000 additional updates before deciding whether to reintroduce a small auxiliary coefficient. If it loses, preserve C and move to the CE/early-weighting tests separately. Do not promote a three-knob recipe based on one confounded result.

## Public primary-source cross-check

The [DFlash paper, §4.2 and Appendix A.1](https://arxiv.org/html/2602.06036v1) describes frozen shared embedding/head, output cross-entropy with exponential position decay, random anchors, six epochs, LR6e-4, clip1,4% warmup,512 anchors/sequence and max length3072 (4096 for Coder). Its standard b16 gamma is7; b8 uses4. The private b4/b8 run instead used gamma7 in both stages. The paper's feature fusion is a shared context conditioning path, not a requirement to add target states to block inputs. These published settings do not establish that our auxiliary loss balance is right for the new target.

The official [SpecForge DFlash source at pinned revision `fe694806...`](https://github.com/sgl-project/SpecForge/blob/fe694806d83704d9300db40111173e5b28836f0d/specforge/modeling/draft/dflash.py) confirms the K/V-only path, `fc→hidden_norm`, and shared fused features. Its [training wrapper](https://github.com/sgl-project/SpecForge/blob/fe694806d83704d9300db40111173e5b28836f0d/specforge/algorithms/common/dflash_family_model.py) confirms unique-anchor construction, absolute positions, gold/mask blocks, and position-weight normalization. These fetched files exactly match the extracted vendored counterparts byte for byte. The private KD changes reside in `src/dflash_distill_base.py` and are explicitly identified above; the public wrapper alone is not evidence for their use.

The official [DFlash2 explanation](https://inco.ai/blog/dflash2/) describes local convolution and a candidate selector as separate improvements, while attention reads target context. Its attention study reports within-block attention mass falling from30% at layer1 to8% at layer5 on its studied model. That supports preserving the context-attention pathway, not adding an unreported residual injection. Its first-position recall numbers are a different target/task and are not acceptance baselines for this run.

Tencent's actual tensor layout/config matches this conclusion: five shared target-feature blocks into fc, per-layer K/V, and bridge tensors; no separate target-to-block residual adapter. Its own code in this repo shows strong context sensitivity, as documented in [`model-eval.md`](model-eval.md). Config/tensors establish architecture, not unrecorded optimizer settings or acceptance for Qwen3.8.

## Reproduction and scope

Copied from the workstation by SCP: source archive, config/provenance/evaluation, selected model weights, promoted run config and small metrics log, about1.06GB total, below the brief's2GB limit. All tensor/metric analysis ran on the Mac with four CPU threads. Existing local `/tmp/audit-model` copies supplied ours/Tencent tensors and real context features; no remote Python model execution was needed.

Commands:

```sh
python3 docs/audit/dflash-ref/summarize_reference.py /tmp/dflash-ref
python3 docs/audit/dflash-ref/tensor_inspect.py
python3 docs/audit/dflash-ref/verify_proposals.py
```

The report distinguishes: current step5543 symptom supplied by the requester; directly reconstructed historical live depth-1 acceptance; older4516 tensor measurements; and a later step5935 recipe-C scalar snapshot. The proposed patch was generated against current source and passed `git apply --check` without being applied. CPU checks confirm default-scale1 loss and parameter gradients are bitwise identical to the existing objective; scale0 loss and gradients are bitwise identical to an independently constructed distillation-only objective. Invalid knob values are rejected. With nonzero bridge writes, teacher-on/off produces bitwise-identical t1 states while suffix states change (maximum difference.9373). See `proposal-checks.json` (`dflash-ref/proposal-checks.json`, retained internal evidence) and `verify_proposals.py` (`dflash-ref/verify_proposals.py`, retained internal evidence). No new trained acceptance or context-use result is claimed. Only this report and `docs/audit/dflash-ref/` are committed.
