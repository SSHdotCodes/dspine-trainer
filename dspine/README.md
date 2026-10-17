# Trainer and numerical reference

See the [project overview](../README.md), [findings](../docs/findings.md), and
[historical technical contract](../docs/SPEC.md). The current source retains the
historical objective defaults for exact old-run resume. The editable Qwen example
configs opt into corrected recipe C and LR 3e-4.

## Files and entry points

- `modeling.py`, `config.py`: block-causal draft, independent writes, last-write
  transition cache, exact K-way proposal q, checkpoint loading.
- `losses.py`, `train.py`: chunked CE/exact sparse L1, auxiliary objective,
  curriculum, FP32 optimizer masters, checkpoint/RNG/data-cursor resume.
- `data.py`, `capture.py`, `capture_sequence.py`, `extract_features.py`: portable
  feature format and capture callbacks. Labels/p at j refer to token j; feature j
  is after target token j. Draft context is strictly before the known anchor.
- `transfer_space.py`, `adapt_target.py`: PCA whitening and aligned adaptation.
- `eval_rounds.py`, `eval_offline.py`: complete-round and opening estimators.
- `rolling.py`, `rolling_append.py`: bounded feature windows and exact continuation.
- `ab_lambda.py`, `recipe_ablation.py`, `forks.py`: explicit, isolated recipe forks.

## Paths and capture

Set `DSPINE_PROJECT` to an absolute workspace and `DSPINE_CODE_ROOT` to this
checkout for shell capture jobs. Set `DSPINE_TARGET_PATH` to a local target
snapshot. `DSPINE_PYTHON` defaults to `python3`; `DSPINE_CAPTURE_IMAGE` defaults to
`lmsysorg/sglang:v0.5.21`. Shell capture requires Linux, Docker and a compatible GPU.
No models or images are downloaded automatically.

`capture_server.sh` exports the installed model source and applies our small
capture-hook transformation. It defaults to the public/base runtime. The hook
fails closed when the installed source layout differs. Adapt the hook against
your runtime and verify layer/stream/position semantics before a large capture.
Optional `DSPINE_CAPTURE_RUNTIME=production` and `DSPINE_RUNTIME_UNIT` let a local
operator pin their own read-only numerical runtime; no such source is bundled.

Capture must prefill the entire original target prefix to preserve recurrent
and positional state. Only stored draft-context features are cropped. FP8 bytes
are dequantized with per-token/per-layer scales. Persist all top-k cutoff ties.
The default exclusion file is `prompts/EXCLUDE_IDS.json`; set `exclude_ids` in
your run config to its absolute path so resumed data also honor it.

To extract from an already running capture server:

```bash
python -m dspine.extract_features --generations ./work/gen \
  --root ./work/features --url http://127.0.0.1:30220 \
  --exclude-ids ./prompts/EXCLUDE_IDS.json
```

## Training and evaluation

Edit config paths, generation revision, model shape and numerical head policy.
The small example is a starting point, not a measured full-run budget. Full
configs need a locally prepared proposal-vocabulary JSON. A full-vocabulary run
can omit `proposal_vocabulary` entirely. Build C from exactly the frozen head
used by the drafter and verifier. Training is intended for CUDA; CPU is useful
for tests and small references.

```bash
python -m dspine.train --config ./config/train.json
python -m dspine.train --config ./config/train.json --resume latest
python -m dspine.eval_rounds --checkpoint ./work/ckpt/preview \
  --target ./models/target --features ./work/features/pilot-eval/T1/shards \
  --blocks 6,8 --temperature 1 --sequences 64 --trials 4 \
  --require-decode-probs --output ./work/eval-T1.json
```

Use separate greedy continuations for T0 and pass `--temperature 0`. Use
`--head-quantization qwenfast_w8` only when that head policy was actually used.
Sampling and proposal probabilities must match their real processors. Offline
tau does not measure serving throughput or prove live cache correctness.

The ablation documents describe historical experiments; new users should begin
with the corrected example config and [adaptation guide](../docs/adapting-targets.md).
