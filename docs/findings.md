# Findings: an independently validated trainer, an unsuccessful target speedup

Training for RedHatAI/Qwen3.8-Flash-Next-NVFP4 stopped by owner decision at step
9,145 on October 17, 2026. The target's native MTP head is unusually strong.
This DSpine head remains an experimental research preview and warm-start base.
The serving-cost model estimates **0.6–0.7x native MTP throughput**. This is a
model using measured components and offline acceptance, not a matched live
benchmark of the final head. No quality or speed benefit is claimed.

## Final checkpoint metrics

| Temperature | b6 pooled tau | b8 pooled tau | Prompts |
|---|---:|---:|---:|
| T1 | 1.799928 | 1.823011 | 64 |
| T0 | 1.849828 | 1.867662 | 64 |

CPU FP32 evaluation completed in 662.2 seconds. T1 Monte Carlo standard errors
are 0.000565 (b6) and 0.004993 (b8); prompt-level standard errors are
0.042596 and 0.044432. These measure different sources of uncertainty.

The final score uses the audited scorer on CPU FP32, 64 short held-out
trajectories per temperature, four T1 trials, complete uncensored rounds,
actual-decode p and the saved proposal policy. The original scorer was extended
only to report T0 b6 alongside b8. See the [full score](release/final-score.json).
These final values supersede approximate in-run estimates.

## Validation and its scope

The Tencent checkpoint reproduction the first 100 GSM8K test prompts, Qwen3-8B
BF16, greedy, thinking off, 512-token cap, b16, gives prompt-mean tau **10.67034**
and pooled tau **9.85681** (25,608 emitted tokens / 2,598 verification rounds).
The pooled prompt-bootstrap interval is [9.32635, 10.40984]. Use sqrt(r)=32;
Tencent's `kappa=10` is not the inference scale. See the
[protocol and ablation receipt](../dspine/results/tencent-reproduction-summary.json).
This validates the inference path, not convergence on a different target.

A separate four-example synthetic CPU overfit smoke (auxiliary supervision off)
reached 100% proposal accuracy; shuffled-context accuracy was 0%, with frozen
tables unchanged. This is not a real-target overfit/convergence result. Its
[receipt](release/overfit-smoke.json) records the scope. The tests include independent full-head loss and
gradient comparisons, packed/reference parity, future-feature leak checks,
cache/proposal probabilities, and exact CPU optimizer/RNG/cursor continuation.
CUDA parity was measured during the experiment, separately from this release's
CPU verification. See [GPU parity evidence](../dspine/results/prerun/gpu/packed-parity-gpu.json).

Seven independent audit workstreams checked the paper, data, loss, loop,
model/evaluation, a clean-room differential reference and a working same-target
DFlash reference. The original reports retain their dates and hypotheses;
later fixes and measurements below supersede stale recommendations.

## What changed in the recipe

The historical lambda used a position-weighted CE sum, increasing the auxiliary
coefficient by 6.63061 relative to a nominal weighted token mean at b16. The
corrected recipe sets `lambda_ce_reduction="weighted_token_mean"`. The paper's
reduction is underspecified; this is an explicit interpretation backed by loss
and gradient audits, not a newly discovered arithmetic bug. Short A/Bs did not
show that normalization alone solved convergence.

The promoted recipe also samples unique anchors with fixed-capacity zero-weight
padding, starts last-write refinement at position 2, and masks hard CE outside
stored teacher support. At four sequences/up to 1,024 anchors per update,
3e-4 LR beat 6e-4 in follow-up tests. Do not transplant the paper's LR unchanged
across very different optimizer batches. Sequence diversity and per-sequence
weighting still matter. Historical exact-resume defaults remain in code; the
editable release configs opt into the corrected recipe.

Target probabilities must match the runtime's processed distribution. Joint
top-k/top-p with ties differs from sequential top-k then top-p. Prefill teacher p
can differ from actual decode p: sampled labels outside prefill support were
observed. Exact stored-trajectory T1 evaluation uses actual-decode probabilities,
separate from those training distillation targets.

## Why the initial tau reports were misleading

At step 4,516, opening RB tau b8 was about 2.87, while complete-round tau was
about 1.60. Repeated CPU checks found the evaluator correct. Formulaic
`<think>` openings are easier than later response tokens. The checkpoint
under-used target context, and position-1 quality was weak. Opening scores are
diagnostics; complete-round scores are the selection metric. Even those scores
use only 64-token held-out continuations and do not establish full-answer speed.

The native MTP receives all four target residual streams after the final layer;
the draft receives layer means ending at layer 45. Direct probes showed unused
predictive signal in the existing inputs. Follow-up capture found that extra
final-layer features improved probe overlap only 0.03–0.045, below the +0.05
adoption rule. That result revised the earlier optimistic feature recommendation.
It does not prove that data, architecture or optimization is fully solved.

The historical DFlash2 ratio 0.969x was throughput relative to MTP, not 0.969
acceptance. Reconstructed sampled-round depth-1 acceptance was 0.640 for that
draft and 0.763 for its MTP baseline, on a different target revision/cohort.
Do not compare those directly to this preview's fixed-anchor overlap.

An MTP-first/DSpine-tail hybrid is mathematically exact when each proposal row's
actual conditional q accompanies verification. Its proposed engineering path
and CPU cost model do not demonstrate a speed win. Stronger tail continuation,
native MTP-q capture and matched live timing remain necessary.

## Detailed reports

- [Paper-to-code checklist](audit/paper-checklist.md)
- [Data pipeline and actual-decode p](audit/data.md)
- [Loss arithmetic, gradients and lambda](audit/loss.md)
- [Training-loop mechanics](audit/loop.md)
- [Model/evaluation and opening bias](audit/model-eval.md)
- [Clean-room differential audit](audit/cleanroom/REPORT.md)
- [Depth-1 input/optimization gap](audit/depth1-gap.md)
- [Final-layer feature measurement and decision](audit/depth1/finalfeat/RESULTS.md)
- [Historical DFlash comparison](audit/dflash-ref.md)
- [MTP-first hybrid design](notes/hybrid-mtp-dspine.md)

Reports cite retained internal snapshots that are deliberately not distributed:
private production code, target weights, paper/checkpoint files, feature caches
and raw run dumps. Only project-authored reports and small aggregate evidence
are included. Reproduction needs separately obtained third-party artifacts.
