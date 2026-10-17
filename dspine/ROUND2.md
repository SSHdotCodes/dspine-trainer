# Round-two recipe forks

Prepared locally, CPU only. The orchestrator owns all GPU execution. Do not sync
these runtime files until `dspine-abc-recipe` has finished. Nothing here launches,
stops, reserves, or modifies the frozen main run or the running round-one arms.

## Small, explicit arm dictionaries

The generalized entry point is still `dspine.ab_lambda`. Legacy A/B/C commands
remain available. `--recipe-file` opts into the named-arm mode; the checked-in
`configs/round2-recipe.json` contains:

```json
{
  "base_recipe": "C",
  "reference_arm": "control",
  "arms": {
    "control": {},
    "lr_low": {"lr": 0.0003},
    "lr_high": {"lr": 0.0012},
    "accum4": {"gradient_accumulation": 16},
    "no_teacher": {"curriculum_p": 0.0}
  }
}
```

`--base-recipe A|B|C` overrides the file's base. The common base explicitly sets
all four loss/sampling switches, so choosing B from a C parent does not silently
retain C's unique-anchor or masking changes. All other values inherit the parent.
Per-arm overrides are deliberately limited to LR, accumulation and curriculum;
model/data/head/seeds/global token schedule cannot be changed in this experiment.
The recipe JSON can contain any subset of these arms or additional valid names.
`--arms` stages a subset from one immutable experiment definition; promotion
requires every arm defined in that JSON to finish. Edit the JSON before the first
launch if only a smaller comparison is desired.

`lr` is the peak coefficient in `lr * lr_multiplier(cumulative_token_progress)`.
Thus3e-4/1.2e-3 multiply the original6e-4 curve by0.5/2, including the remaining
cosine. No schedule, warmup, token counter, RNG, optimizer moment or Adam step
counter restarts. Curriculum is unchanged when the key is absent or null;0
explicitly disables teacher replacement. Uniform teacher RNG draws still occur
when p=0 so controls keep their random-number streams paired.

Accumulation4->16 keeps256 nominal anchors per sequence (capped for C), forwards
one sequence at a time, and increases sequence/anchor exposure per update4x. It
does not retain16 activation graphs. Per-sequence loss and accumulation averaging
remain unchanged; this is still equal-total-weight per sequence.

## Parent and equal-samples semantics

Supply ANY complete optimizer checkpoint compatible with the selected immutable
plan/window. A mid-window parent must carry that window's cursor, which is
restored exactly. At a new-window boundary only, the stream resets to the new
window seed. A parent with changed accumulation must use its promoted plan so
remaining step budgets have the correct units. The normal round-one winners at
5516 all fit the original window1 plan.

`--steps 1000` is the reference arm's update budget. Without `--equal-samples`,
every arm runs1000 updates. With the flag, each arm gets
`1000 * reference_accumulation / arm_accumulation` updates; non-integral or zero
budgets are rejected rather than rounded. The example runs control/lr_low/lr_high/
no_teacher1000 updates each and accum4 only250, consuming exactly the same4000
sequence draws and sampled anchors in all arms. No arm may cross the current
feature window's remaining sample budget.

The default named-arm eval cadence is250 actual optimizer updates, relative to
the parent, plus a final eval. From5516, normal arms evaluate5516/5766/6016/6266/
6516; accum4 evaluates5516/5766. Every evaluation runs the same complete-round T1,
opening T1 and T0 b6/b8 suite. Compare final points by sequence samples/token
progress, not raw step number. `comparison.json` includes `reference_updates`
and `sequence_samples` plus cumulative response tokens for each evaluation.
`full_budget_means` averages CE/L1 across the full sequence budget; the diagnostic
`last250_means` covers250 updates and spans different sample counts across
accumulations. A larger-batch arm has fewer interim
evaluations under this actual-update cadence.

Every training step logs CE/L1, LR, curriculum, anchors and per-sequence sample
fingerprints. The manager pairs the flattened sequence/anchor stream across
unequal update counts, verifies cumulative response tokens, and checks teacher
RNG when probabilities match. Different accumulation can update a scheduled
curriculum probability at different token boundaries; different p intentionally
permits different teacher masks. Data/anchor mismatches fail the experiment.

## Files the owner must sync after round one ends

```bash
rsync -av --relative dspine/ab_lambda.py dspine/recipe_ablation.py dspine/forks.py dspine/train.py dspine/configs/round2-recipe.json compute-host:./work/code/
```

The existing `scripts/ab_lambda_job.sh` and `scripts/run_module.py` are unchanged
and already installed for round one. No modeling/serving API changes are needed.
The canonical module entry point keeps signal handling and child ownership in
one module, including when invoked through the existing runpy wrapper.

## Exact launch after choosing the round-one winner

Example assumes C wins; replace BOTH the checkpoint letter and `--base-recipe`
with A or B if selecting that recipe. Use a NEW experiment root. Keep the main
controller/driver paused and register this ai-job under the owner's GPU hold.

```bash
ssh compute-host ./work/bin/ai-job run dspine-recipe-r2 bash ./work/code/dspine/scripts/ab_lambda_job.sh --recipe-file ./work/code/dspine/configs/round2-recipe.json --base-recipe C --checkpoint ./work/ckpt/ab-lambda/C/step-00005516 --output ./work/ckpt/recipe-r2 --window 1 --steps 1000 --equal-samples --eval-every 250 --allocator-cap-gb 30 --mode auto --max-minutes 155
```

With all5 selected, auto mode runs sequentially because five30GB caps cannot fit.
This gives each arm the same dedicated-GPU conditions. Budget roughly110–140min
for5000 normal-update-equivalents plus22 evaluations and checkpoints, based on earlier
1.18–1.3s normal-step measurements, not a new GPU measurement. Accum4 is roughly4x
slower per update but has1/4 as many updates. The same one-sequence activations and
30GB cap apply to each arm. New sample traces require only a few KB per update.
Required free disk is210GB for all5; no automatic deletion is performed.

Alternatively stage three arms in parallel, then the other two under the SAME
parent/output/spec/options, changing only `--arms` and bounded manager duration:

```text
first:  --arms control lr_low lr_high --mode parallel --max-minutes 100
then:   --arms accum4 no_teacher       --mode parallel --max-minutes 65
```

Three30GB caps require>=96GB free; two require>=66GB. These are allocator-based
guards, not claims of measured new peak VRAM. Do not run two managers for the same
experiment simultaneously. Register sequential jobs in the owner's hold and wait
for success between them. A partially completed manager rerun resumes each arm's
own optimizer/RNG/cursor and stops at its original arm-specific goal. Healthy
peers continue after any child failure. Only failed OOM arms retry sequentially
after peers finish; other failed arms retain their logs/checkpoints and are
reported separately. Incomplete capture exits75 before GPU training/query; the
existing optional `--finish-capture` completes missing features exclusively.

## Same immutable promotion path, including larger accumulation

After manual review, promote any completed arm by NAME, for example:

```bash
ssh compute-host python3 ./work/code/dspine/scripts/run_module.py dspine.ab_lambda --output ./work/ckpt/recipe-r2 --promote-arm accum4 --promote-root ./work/runs/rolling-r2-accum4
```

The new controller continues the chosen arm's exact final checkpoint/config;
strict ordinary resume remains intact. If accumulation changes, only the new
plan's REMAINING current/future window update budgets are rescaled to preserve
remaining sequence exposure. Reusing the old step counts with4x accumulation
would silently overtrain4x. The original manifests, old controller state/config,
checkpoint output and global token schedule remain unchanged. Per-window ceiling
rounding can add less than one new optimizer batch at each boundary.

The owner must repoint its driver to the new root/source, update the output-specific
NO-GO check, and re-arm any quality watcher against the selected arm. Keep the old
controller stopped because the feature cache is shared. No code edits TRAIN_GO,
the driver or a quality watcher.

## CPU validation

Full suite: **143 passed,2 CUDA tests skipped in46.05s**. Receipt:
`results/cpu-tests-recipe-round2-full.txt`. Coverage includes A/B/C base replacement,
LR scaling, invalid overrides, equal-samples budgets, parent-cursor isolation,
sample pairing across unequal accumulation, staged launches, idempotent resume,
new-plan-only sequence-budget rescaling, and an actual next optimizer update after
promoting the larger-batch arm. Ordinary resume still rejects changed config.
Ruff F/I, formatting, compileall, shell syntax and whitespace checks pass.
No remote commands, GPU tests, reservation, rsync or run launch were performed.

## Fixed-capacity C and failure isolation (11:45 update)

C's old variable-length anchor tensors caused the reported GPU backward compile
failure. The corrected sampler uses256 query blocks at all response lengths,
with unique real anchors and zero-weight padded dummy blocks. All loss reductions
normalize by real anchors. Refinement starts at2; stored-zero-support labels
remain masked in CE. Teacher RNG consumes only real-anchor draws, as before.
`anchors` logs real supervision while `anchor_slots` logs physical capacity.

Both harness modes now allow healthy peers and later groups to complete after an
individual exit/launch failure. Each arm writes a live `status.json`; the final
comparison retains all metrics and lists `failed_arms`. The manager returns1
only after healthy peers have finished. OOM fallback retries only failing OOM
arms after peers finish. User stop, bounded deadlines and global manager errors
still clean up owned processes. No external notification or watcher was launched.

Owner sync paths for this correction, AFTER the active round2a job ends:

```text
dspine/data.py
dspine/losses.py
dspine/train.py
dspine/ab_lambda.py
dspine/recipe_ablation.py
```

No modeling/serving API, recipe flag, config JSON, or shell wrapper changes. Use a
NEW experiment output when applying the correction: old manifests intentionally
reject changed code hashes. Start from any compatible durable parent checkpoint;
the current owner-reported round2a parent is B/step-00005043, base B,500updates.
The earlier5516 commands are illustrative original-plan examples; round1 did not
reach that goal. No sync, GPU action or launch was performed by the trainer.

CPU regression tests cover full256-slot data with1/4/256 real anchors, equality
of padded and unpadded values/all parameter gradients with checkpointing, and
one full AOTAutograd forward/backward graph across1/4/8 real anchors at a fixed
8-slot numerical-test capacity. This validates shape/masking semantics, not the
unavailable CUDA/Triton lowering. Subprocess tests inject exit/launch failures
and prove healthy arms finish their goal and retain final evaluations.

Fixed-capacity validation: **151 passed,2 reservation-gated CUDA skips in58.88s**,
receipt `results/cpu-tests-fixed-anchors-full.txt`. Ruff F/I, compileall, formatting
and whitespace checks pass. No GPU validation was run.
