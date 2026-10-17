# Lambda and corrected-recipe A/B/C

Prepared 2026-10-17, CPU only. No remote commands, GPU job, reservation, rsync,
main-job lifecycle action, or writes to the live controller/checkpoint directories
were performed. The orchestrator owns execution and promotion. The parent at
step4516 has already been pinned under `ckpt/ab-lambda/parent`.

## Arms

All arms restore the same full model, Adam moments, RNG states and cumulative
response-token progress from4516, reset ONLY the data cursor to complete window1,
and preserve the main LR, warmup, cosine schedule, curriculum and head policy.
The default sampler and main run objective remain unchanged.

| Arm | Lambda CE | Anchors | Refinement positions | Hard CE outside stored p support |
| --- | --- | --- | --- | --- |
| A | historical weighted position sum | with replacement,256 | t=1..15 | included |
| B | A divided by nominal weight mass S | same as A | same as A | same as A |
| C | same as B | 256 slots,min(256,n_valid) real unique anchors | t=2..15 | masked |

For block16/gamma7, S=sum(exp(-i/7),i=0..14)=6.630610787. The paper calls
CE_bb the token cross-entropy but does not specify its reduction; a weighted mean
is better motivated than a second block-position sum. This remains an
interpretation to test, not a proven reconstruction of unavailable Tencent code.
Neither B nor C renormalizes lambda for tail masks, nor divides the layer sum.

C's refinement skip agrees with inference: `build_transition_cache` writes only
receivers2..15 and `propose` uses `initial_scores[:,0]` directly for t=1. A serial
last-write at t=1 reproduces the forward hidden state because the known anchor
was already injected. The two losses' distributions are not identical (full hot
head versus candidate-renormalized head), but the extra t=1 refinement objective
is absent at inference. C removes both CE and L1 refinement at t=1 without
renormalizing the remaining weights. No modeling/serving API or weights change.

C masks backbone CE AND refinement CE when the sampled label has zero stored
processed probability; the masked backbone CE also supplies lambda. L1 retains
all stored probability mass and embedding targets remain unchanged. This avoids
contradictory hard labels without pretending to reconstruct missing actual-decode
probabilities. It does not repair numerical mismatches between target runtimes.

**The full-optimizer-batch lambda change was deliberately skipped**, as allowed.
Lambda remains per microbatch in all arms. Exact implementation would require
retaining multiple sequence graphs or splitting/accumulating embedding gradients
with another backward pass and extra parameter-sized buffers. That changes the
memory/timing comparison and deserves a separate test. The loss audit measured a
much smaller covariance effect than the6.63x normalization issue.

**Sampling caveat:** each microbatch loss still averages over its actual anchors,
then microbatches are averaged. Without replacement removes duplicate supervision; fixed query padding retains
full compute on short responses. It does NOT remove equal-total-weight-per-sequence
sampling or all response-length skew. No unrequested sequence-weighting change
is bundled. Actual anchor counts, rather than the old nominal1024, are logged.

C preserves A/B's shuffled sequence order even across epochs: sample unique
anchors from a clone of the current stream RNG, while advancing the saved order
RNG by the same legacy256-draw budget as A/B. This is uniform sampling without
replacement, with exact resumability and no effect on A/B. C's anchor and
teacher-mask fingerprints intentionally differ; its sequence fingerprints,
response-token progress, LR and curriculum must still match A/B at every step.

## Owner sync, after the main chunk ends

Required runtime files (relative to the repo):

```text
dspine/ab_lambda.py
dspine/data.py
dspine/forks.py
dspine/losses.py
dspine/train.py
dspine/scripts/ab_lambda_job.sh
```

The shell wrapper was already prepared in the preceding A/B commit, but is
included so the commands are self-contained. `modeling.py` is unchanged. Example
owner-only sync, from the repository root, AFTER the main chunk has ended:

```bash
rsync -av --relative dspine/ab_lambda.py dspine/data.py dspine/forks.py dspine/losses.py dspine/train.py dspine/scripts/ab_lambda_job.sh compute-host:./work/code/
```

Keep the detached main driver waiting for its GO file throughout the experiment.
The harness refuses an active main controller and never stops/reserves anything
for it. Register these units as work under the owner's hold.

## Recommended launch: A+B together, then C

`--steps 1000` means1000 optimizer updates for EVERY selected arm, from4516 to5516.
All three evaluate at4516,5016 and5516 with complete-round T1, same-cohort opening
T1 and T0, using b6/b8. The global schedule never restarts. These two jobs must
run consecutively, not concurrently; the second waits for the first to exit:

```bash
ssh compute-host ./work/bin/ai-job run dspine-ab-lambda bash ./work/code/dspine/scripts/ab_lambda_job.sh --arms A B --steps 1000 --mode parallel --max-minutes 100
ssh compute-host ./work/bin/ai-job wait dspine-ab-lambda --timeout 6600 --lines 12
ssh compute-host ./work/bin/ai-job run dspine-c-recipe bash ./work/code/dspine/scripts/ab_lambda_job.sh --arms C --steps 1000 --mode sequential --max-minutes 65
ssh compute-host ./work/bin/ai-job wait dspine-c-recipe --timeout 4200 --lines 12
```

Expected total65–80 minutes for3000 updates plus startup/checkpoint/eval, based on
previous1.18–1.3s/step measurements. These are estimates, not new GPU measurements.
A+B likely reserve44–52GB combined, each with the unchanged36GB allocator cap;
C alone is expected to use the same22–26GB range; its query capacity is now fixed
even for short responses. No new GPU memory measurement was made. A+B parallel
requires at least78GB free (two caps plus6GB headroom). If unavailable, use
`--mode auto` to select sequential instead; allow more startup time as needed.
C-only requires42GB free under the unchanged cap. Each invocation checkpoints
near its bound; rerun exactly the same invocation to resume without extra steps.

## Optional all three concurrently

If no other work is resident and at least96GB is free, launch all arms with a
common30GB allocator cap (all recipe differences remain identical):

```bash
ssh compute-host ./work/bin/ai-job run dspine-abc-recipe bash ./work/code/dspine/scripts/ab_lambda_job.sh --arms A B C --steps 1000 --mode parallel --allocator-cap-gb 30 --max-minutes 155
```

Measured22–26GB per arm suggests66–78GB actual combined, but this has NOT been GPU
validated for three processes or C's variable query shapes. The guard requires
three30GB caps plus6GB. Do not assume3x speedup on the shared GPU. If allocator OOM occurs, healthy peers finish their goals; only the failed arm
is retried sequentially from its exact checkpoint. The cap must be identical in subsequent invocations for this
output; do not switch between36GB A+B and30GB C in one experiment. A dedicated
experiment lock prevents two managers from writing any shared output concurrently.

## Incomplete window1 capture

The default exits75 BEFORE any training/GPU query if even one manifest feature
shard is missing. No partial-window experiment is run. In an exclusive owner-held
slot, add `--finish-capture` to the FIRST invocation, for example:

```bash
ssh compute-host ./work/bin/ai-job run dspine-ab-lambda bash ./work/code/dspine/scripts/ab_lambda_job.sh --arms A B --steps 1000 --mode parallel --max-minutes 155 --finish-capture
```

At most40 extra minutes finish ONLY missing features using the existing private
production-pinned capture implementation. Its logs/CID go under this experiment,
not `runs/rolling`. Capture needs~80GB GPU/~95GB pinned host memory, so co-runners
must leave first. It stops the target before starting arms. If still incomplete,
exit75 preserves progress; finish another bounded capture chunk and retry.

## Comparison and continuation

Outputs live under `ckpt/ab-lambda/{A,B,C}`. Each step logs CE, nominal per-token
CE, L1, auxiliary, distillation, unsupported-CE fraction, actual anchors, LR,
progress, gradient norm and timing. `comparison.json` contains all available
branches, complete/opening/T0 trajectories and last250-step means;
`paired-metrics.jsonl` contains A/B losses on identical batches. A/B must match
full data/anchor/teacher fingerprints; C must match sequence/schedule fingerprints.
C intentionally changes the anchor sample, so CE/L1 are not paired token-for-token
between C and A/B. Compare acceptance as well as loss; total losses use different
objectives. There is no automatic winner or GO decision.

If B or C wins, create a NEW immutable controller root, continuing its final5516
model/optimizer/RNG/window1 cursor. For C, require completed A/B/C and then run:

```bash
ssh compute-host python3 ./work/code/dspine/scripts/run_module.py dspine.ab_lambda --promote-arm C --promote-root ./work/runs/rolling-recipeC
```

For B use `--promote-arm B --promote-root ./work/runs/rolling-lambdaB`.
The original `runs/rolling` and `ckpt/full` stay untouched. Promotion verifies the
source/window fingerprint and intact features, creates new source/run/plan/state
files, retains all window ranges and the global schedule, and keeps the selected
branch's existing output/config so strict exact-resume checks continue to pass.
The owner must repoint its driver to the new root/source, update its output-specific
NO-GO path, and re-arm any quality watcher against the selected branch. The old
controller must stay stopped because the feature cache is shared. No utility
edits TRAIN_GO, the detached driver, or an existing quality watcher.

The source parent and copied checkpoints require substantial disk: the launcher
requires90GB free for A+B,50GB for C-only, or130GB for all3. No automatic deletion.
Once a winner is safely continued, the owner can archive unneeded branches.

CPU validation: **127 passed,2 CUDA skips in46.12s**; receipt
`results/cpu-tests-corrected-recipe-full.txt`. Tests cover loss/gradient reference,
inference position1, unique-anchor cap and exact resume across epochs, staged
A+B then C, matching per-arm step limits/eval offsets, unchanged original
controller files, and an actual next optimizer update after immutable C promotion.
Ruff F/I, compileall, shell syntax and whitespace checks pass. No GPU arm was run.

## 11:45 correction: fixed query capacity and independent failures

Round1 failed at04:23 (owner report): C's variable query shapes triggered an
Inductor/FlexAttention backward LoweringException, and the old manager terminated
healthy A/B peers. That variable-length sampler was the trainer's implementation
error. The corrected sampler always emits the configured anchor capacity256.
It samples256 unique positions if available; otherwise it takes every valid
position once and appends repeats of the final safe anchor as dummy blocks.
`anchor_valid` and every position in `valid` are false for those dummy blocks.
Attention itself remains well-defined, with each dummy seeing its own block,
and real blocks cannot attend to dummies. All objective terms and lambda use
only real-anchor weights and divide by `valid[:,0].sum()`, not by256.
Refinement still starts at2 and zero-support CE masking remains enabled for C.

Real sampler/teacher RNG draws are preserved; teacher bits are padded with false
only after drawing for real anchors. Training logs distinguish real `anchors`
from physical `anchor_slots`. Fixed capacity removes the earlier proposed
short-sequence compute saving.

A failed child or child-launch error no longer terminates healthy peers or skips
later groups. Per-arm `status.json` records failure immediately; the final
`comparison.json` includes `failed_arms` alongside healthy final metrics. The
manager exits1 after peers finish if a failure remains. Only an OOM arm is retried,
after its peers finish. Explicit stop/deadline and manager-failure cleanup still
apply to owned children. Never sync during an active job; corrected experiments
need a NEW output root because the old manifests pin their code hashes.
