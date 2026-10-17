# Round 4: previous-state input fusion (CPU preparation, 2026-10-17)

The owner reports round 3 tied or lost to control. The running main remains
recipe C, LR3e-4, under `runs/rolling-r2c-lr3`, with output
`ckpt/recipe-r2c/lr3`. This change is opt-in. No GPU, remote command, reservation,
rsync, job launch or main-root write was performed by the trainer.

## Model and serving contract

Read `docs/audit/depth1-gap.md`. For anchor absolute position `a`, take the exact
normalized fused context row already used for context K/V:

```python
context = hidden_norm(fc(features))
previous = context_at_absolute_position(a - 1)  # strictly before anchor
delta = F.linear(previous, context_fusion.weight)  # [hidden, hidden], no bias
x[:, 0] += delta                  # context_fusion="anchor"
# Alternative: x[:, :] += delta[:, None]       # context_fusion="all"
```

Apply this once, after assembling anchor/mask input embeddings and before
decoder layer0/input_layernorm. Context K/V rows, RoPE positions, bridge
read/write, DraftOutput, last-write cache and proposal APIs are unchanged.
An empty prefix or missing/invalid `a-1` row contributes zero. Ordinary forward
selects by absolute position plus validity; packed forward gathers local
`anchor_offsets-1`, never target features at or after the anchor.

Anchor mode is the experiment default because MTP fuses the previous target
state with the known next-token embedding. Here that known token is the anchor;
later masked rows consume the enhanced anchor through causal attention. All mode
broadcasts the same preceding target state to masked positions as well, and is
available for a separate override. This follows the requested five-layer fused
feature contract; it adds no new capture and does not reproduce MTP's full HC
state or MoE architecture.

The projection is zero initialized and preserves RNG when created. Off mode
registers no new parameter. Existing checkpoints can load with fusion enabled
and no saved weight, which stays zero. New checkpoints save the mode and
`context_fusion.weight`. Explicit forks append its optimizer parameter without
changing any existing parameter ID or Adam moment; its own Adam state starts
fresh. Normal resume still requires the identical frozen run config. A trained
fusion weight cannot be silently discarded by loading with mode off.

Serving must load the new weight/mode and perform this input addition using its
cached normalized fused `a-1` row, once per request/block. Reuse the same feature
row and arithmetic dtype as the context projection path, including the sum's
cast back to input-embedding dtype. For FP32 oracle parity, load trainable weights
as FP32. The BF16 training path uses autocast for the linear projection. GPU
parity and timing of the new serving operation remain to be measured.

`pos1_weight` defaults to1 and multiplies the first element of the existing
position-weight vector. It applies to backbone CE/L1 and embedding loss. Recipe C
continues to skip t1 refinement. Its weighted-token-mean lambda normalizer uses
the modified vector too, so weight3 increases nominal weight mass by2. This
preserves the interpretation of lambda's CE as a weighted per-token mean.

## Recipe, source isolation and co-running

`configs/round4-recipe.json` defines base C and three arms: control{},
fusion{context_fusion:anchor}, fusion_p1{context_fusion:anchor,pos1_weight:3}.
LR, token cosine, curriculum, anchor capacity, batch, data RNG and all other
knobs are inherited from the chosen main checkpoint. Each arm runs500 updates,
with complete-round T1 b6/b8, opening T1 and T0 at start, +250 and +500.

`--allow-active-main` bypasses the inactive-unit requirement only for an explicit
recipe training fork. Capture completion and promotion cannot use this flag.
The tool never stops/changes the foreign controller. It first requires the
source state to be in pure training. While children run it polls that phase;
if the foreign controller begins rotation/capture, only owned experiment
children receive SIGTERM, with partial checkpoints and exit75 when incomplete.
This avoids continuing co-training into the target's high-memory capture phase.

`--checkpoint latest` resolves the selected source plan's output pointer to:

```text
./work/ckpt/recipe-r2c/lr3/step-NNNNNNNN
```

The immutable directory is copied/hashed into `ckpt/recipe-r4/parent`. A live
latest retention race retries up to3 pins; explicit checkpoint selectors never
silently change. Retries with `latest` reuse an existing pinned parent even
while the main advances. All three arms start from this one snapshot.

Co-runs hard-link the complete current window into the experiment's owned
`feature-view`. Source contents/directory entries stay unchanged. This requires
experiment and features on the same filesystem, allocates no second feature
copy, and keeps reads valid across source purge. Both initial cursor reload and
epoch reshuffle read that fixed view. Checkpoints keep logical original shard
paths, preserving the promotion/data contract. Source feature files are
immutable. Hard links retain physical bytes after source purge; the manager
removes the view after ALL recipe arms complete. An interrupted experiment
retains its view for resume; remove it manually only when abandoning that
experiment. Do not keep abandoned views across future capture windows.

No file in `runs/rolling-r2c-lr3` or `ckpt/recipe-r2c/lr3` is written. Experiment
configs, parent copy, optimizer checkpoints, metrics and status live solely in
`ckpt/recipe-r4`. A failed child does not stop healthy peers.

## Files to sync

From the repository root, the owner syncs these nine runtime/config files:

```bash
rsync -aR dspine/config.py dspine/modeling.py dspine/losses.py dspine/train.py dspine/forks.py dspine/data.py dspine/ab_lambda.py dspine/recipe_ablation.py dspine/configs/round4-recipe.json compute-host:./work/code/
```

The existing `scripts/ab_lambda_job.sh` and `scripts/run_module.py` are unchanged.
Tests/documentation are optional to sync for execution.

## Exact launch alongside main training

```bash
ssh compute-host ./work/bin/ai-job run dspine-recipe-r4 bash ./work/code/dspine/scripts/ab_lambda_job.sh --plan ./work/runs/rolling-r2c-lr3/plan.json --checkpoint latest --recipe-file ./work/code/dspine/configs/round4-recipe.json --output ./work/ckpt/recipe-r4 --window 1 --steps 500 --eval-every 250 --allocator-cap-gb 21 --mode parallel --parallel-arms 2 --allow-active-main --max-minutes 105
```

This runs control+fusion, then fusion_p1. At most TWO experiment children run
beside the main. The free-memory guard needs48GB free for two21GB allocator caps
plus6GB headroom. Expected total is about60-63GB including a20GB main; no new GPU
fit measurement was made. The new 2560² FP32 projection adds about105MB per arm
for weight/gradient/Adam moments, plus small activations/casts. Budget about
60-85min including shared-GPU contention, nine evaluation suites and checkpoint
I/O, based on earlier timings; the105min bound is a guard, not a benchmark.
Conservative existing checkpoint-space check requires130GB free. The owner
continues to manage GPU holds/renewals and any foreign workloads.

All args and code/spec must remain fixed to resume this experiment. Rerun the
same command to continue a partial run or finish its missing final eval. The
source controller must again be in training, and the pinned feature view must
remain for a resume after rotation. If it advances to another window before
initial pinning, select that current window in a NEW experiment output; an
unrelated mid-window checkpoint is refused.

## Promotion

After all three finish, select a winner. The main must be stopped before
promotion; omit `--allow-active-main`. Example for fusion:

```bash
ssh compute-host python3 ./work/code/dspine/scripts/run_module.py dspine.ab_lambda --plan ./work/runs/rolling-r2c-lr3/plan.json --output ./work/ckpt/recipe-r4 --promote-arm fusion --promote-root ./work/runs/rolling-r4-fusion
```

This stages a NEW controller root with the winner's checkpoint, optimizer,
fusion mode, RNG, token progress and cursor. It never writes the old controller
or old output. Promotion intentionally follows the fork lineage rather than
combining it with further main updates. The source window's original features
must still exist for exact resume; if main already rotated/purged them,
promotion refuses and requires a separately staged restoration/plan decision.
The owner repoints the driver and quality watcher to the new root/output.

## CPU validation

See `results/cpu-tests-round4-full.txt`. Tests cover exact zero-init equality,
RNG/parameter-order preservation, nonzero manual/reference/SDPA/packed/cache
proposal parity at T0/T1, empty prefixes and absolute positions beyond W,
padding validity, future-feature isolation, old/new checkpoint loading,
optimizer moment migration, position1 weighting, two-child grouping beside an
unqueried foreign active unit, feature-view cursor/RNG stability after source
purge and epoch boundaries, live-latest retention races, source-file hashes,
paired samples, idempotent retry and winner promotion.

Full CPU result: **199 passed, 2 CUDA tests skipped in81.71s**. Additional CPU BF16 autocast zero-init equality and checkpointed-backward gradient smoke passed for anchor/all; Ruff F/I, format, compileall and whitespace checks passed.
