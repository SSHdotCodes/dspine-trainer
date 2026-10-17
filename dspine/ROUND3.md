# Round-three recovery objective test

Prepared locally and CPU-only from the13:45 request. No GPU access, reservation,
remote command, sync, run lifecycle action, or live run-file edit was performed.
The orchestrator owns stopping the main controller and executing this experiment.
Read `docs/audit/dflash-ref.md`, its ranked recommendations and proposed patch;
only the three requested objective knobs were added. No architecture, feature,
initialization, anchor count, optimizer state or sampling-law change is included.

## Knobs and exact arm definitions

New run config keys default to the current objective:

| Key | Default | Allowed values | Effect |
| --- | --- | --- | --- |
| embedding_loss_scale | 1.0 | finite, >=0 | Multiplies effective lambda AFTER its CE normalizer |
| loss_alpha | 0.1 | finite,0..1 | Same CE/L1 mixture in backbone and refinement |
| loss_gamma | 7.0 | finite,>0 | Position weight exp(-(t-1)/gamma); nominal lambda weight mass follows gamma |

Boolean/null/NaN/infinite/out-of-range values are rejected before CUDA loading.
Scale0 makes effective logged lambda and auxiliary exactly0. The trained bridge
stays active and trainable through distillation. Raw embedding loss is still
logged; the control's computation and default objective are otherwise unchanged.
Alpha1 is sampled-label CE recovery, not full-distribution teacher CE. C's support
mask, fixed256 slots/real-anchor normalization and refinement fromt=2 remain.

`configs/round3-recipe.json`:

```json
{
  "base_recipe": "C",
  "reference_arm": "control",
  "arms": {
    "control": {},
    "noaux": {"embedding_loss_scale": 0.0},
    "noaux_ce": {"embedding_loss_scale": 0.0, "loss_alpha": 1.0},
    "noaux_g2": {"embedding_loss_scale": 0.0, "loss_gamma": 2.0}
  }
}
```

LR3e-4 is inherited from the paused current parent; the file contains no LR
change. All arms restore that parent's complete optimizer/RNG/cumulative-token
progress and exact mid-window cursor. There is no warmup/cosine restart.300
updates per arm consume the same sequence/anchor stream and use the same teacher
RNG. Evaluation is before training, after150 and300 updates: complete-round T1,
opening T1 and T0 b6/b8. Existing optional long-eval policy is inherited. No new
depth/context probe is silently added to this short job; the independent depth1
audit can inspect the saved checkpoints separately.

## Owner stop, sync, checkpoint selection

The owner stops `dspine-rolling` gracefully and waits until its unit is inactive
and final checkpoint complete. The harness checks `ai-dspine-rolling.service`
before resolving a latest selector. Keep the detached main driver/controller
paused throughout the experiment so it cannot resume and rotate the shared
feature cache. Do not sync code while the main job runs.

Exact runtime sync list:

```text
dspine/ab_lambda.py
dspine/forks.py
dspine/losses.py
dspine/train.py
dspine/recipe_ablation.py
dspine/configs/round3-recipe.json
```

Owner-only command from the local repo root, AFTER stopping the main chunk:

```bash
rsync -av --relative dspine/ab_lambda.py dspine/forks.py dspine/losses.py dspine/train.py dspine/recipe_ablation.py dspine/configs/round3-recipe.json compute-host:./work/code/
```

Use the promoted main plan, not the historical `runs/rolling` plan:
`./work/runs/rolling-r2c-lr3/plan.json`.

The simplest selector is `--checkpoint latest`, which resolves the `run.output`
from that plan. The concrete pointer is:

```text
./work/ckpt/recipe-r2c/lr3/latest
```

It contains a single basename. The resolved checkpoint pattern is:

```text
./work/ckpt/recipe-r2c/lr3/step-NNNNNNNN
```

The CLI also accepts that explicit checkpoint, the pointer's absolute path, or
the output root. It resolves ONCE after the main is inactive, then pins/copies and
hashes that exact directory under the NEW experiment root. Provenance records
the resolved directory, not a moving pointer. An existing explicit parent pin
remains usable if source retention later prunes the original checkpoint.

The source `ckpt/recipe-r2c/lr3` and `runs/rolling-r2c-lr3` are read-only inputs.
No source pointer, checkpoint, plan, config or cursor is written. All training,
metrics, status, winners and parent snapshot go to `ckpt/recipe-r3`. Features are
read from the same completed window1 cache and are not purged by this harness.

## Exact launch, under the owner's hold

```bash
ssh compute-host ./work/bin/ai-job run dspine-recipe-r3 bash ./work/code/dspine/scripts/ab_lambda_job.sh --plan ./work/runs/rolling-r2c-lr3/plan.json --checkpoint latest --recipe-file ./work/code/dspine/configs/round3-recipe.json --output ./work/ckpt/recipe-r3 --window 1 --steps 300 --eval-every 150 --allocator-cap-gb 21 --mode parallel --max-minutes 55
```

Four21GB allocator caps plus6GB headroom require at least90GB free. This is an
allocator guard, not a new GPU fit measurement. No VRAM or timing validation was
performed for this four-process setup. The parent main used a22GB cap. If an arm
OOMs, healthy peers continue to their goals; only failed OOM arms retry later
sequentially. A true per-arm live allocation above21GB will still need a higher
cap and fewer simultaneous arms in a new experiment output. Do not assume that
removing the auxiliary reduces peak memory: its raw metric remains computed.

Budget roughly30–45min for1200 normal-update-equivalents, startup, checkpointing
and12 paired-suite evaluations using earlier step timings, not a new GPU
measurement. The55-minute manager bound leaves checkpointing time. Required
free disk:170GB under the existing conservative retention guard. A partial run
can resume with identical args while code/spec/source latest stay unchanged;
otherwise pass the recorded explicit parent checkpoint or its pinned origin.
No hold or driver registration is done automatically by this tool.

## Promote the selected winner as a new main root

After all four arms finish and the owner selects a winner, example for `noaux`:

```bash
ssh compute-host python3 ./work/code/dspine/scripts/run_module.py dspine.ab_lambda --plan ./work/runs/rolling-r2c-lr3/plan.json --output ./work/ckpt/recipe-r3 --promote-arm noaux --promote-root ./work/runs/rolling-r3-noaux
```

Replace the arm/root suffix when choosing another winner. The source main stays
untouched. The new root's source/run/plan/state point to the winner's final
optimizer checkpoint in `ckpt/recipe-r3/<arm>` and preserve its exact flags/RNG/
cursor/token schedule. Accumulation is unchanged, so remaining window budgets
are unchanged. Strict ordinary resume is preserved. The owner repoints its
detached driver to the new root/source, updates its output-specific NO-GO path,
and re-arms any quality watcher. Nothing edits TRAIN_GO or the existing driver.

Continuation payload for that example:

```text
dspine.rolling --config ./work/runs/rolling-r3-noaux/source.json --generations ./work/gen --root ./work/runs/rolling-r3-noaux --max-minutes 165
```

## CPU validation

**172 passed,2 CUDA skips in77.30s**, receipt
`results/cpu-tests-round3-full.txt`. Tests cover defaults/scale1 exact scalar
parity, dense-reference loss and parameter-gradient parity for every requested
objective, zero auxiliary with live bridge gradients, invalid scalar values,
latest resolution and active-main refusal, survival of existing source pins
after pruning, and a real four-process CPU experiment from a promoted main.
That integration verifies source configs/checkpoint/pointer hashes, identical
sample streams, all four completions, CLI promotion and a real next optimizer
update under the winner's exact saved config/cursor. No GPU memory or acceptance
result is claimed.
