# Adapting DSpine to a new target

`python -m dspine.adapt_target --help` documents the weights-only adaptation CLI.
This applies to a future Qwen 4 Flash only after its actual architecture,
tokenizer, license and supported runtime are available. No compatibility with
an unreleased model is assumed.

1. Obtain the target locally and inspect the real config. Set hidden size,
   vocabulary, query/KV heads, head dimension, intermediate size, RoPE, EOS IDs,
   hyper-connection count and selected post-layer feature IDs. The supplied
   `qwen38-flash-next.json` model object is an example, not a future-model spec.
2. Implement and validate the target feature hook. Dense residuals have one
   stream; hyper-connected targets need an explicit stream reduction. Preserve
   the full original prefix, absolute positions, and strictly pre-anchor context.
   A feature after token j predicts token j+1 through the target head, while the
   draft also receives the known anchor token j+1 and predicts j+2 at position 1.
3. Rebuild the transfer space from the target's *effective frozen head*. The
   current safetensors loader recognizes the embedding/head names listed in
   `transfer_space.py`; a new architecture may need new explicit aliases.
4. Align to the old code space with the CLI below. Keep transfer rank equal for
   orthogonal Procrustes. Certify unchanged tokenizer IDs only after comparing
   tokenizer files and special tokens. Otherwise provide explicit pairs of
   matching token strings, never pairs based solely on equal integer IDs.

```bash
python -m dspine.adapt_target \
  --old-checkpoint ./models/dspine-preview \
  --new-target ./models/new-target \
  --new-config ./config/new-model.json \
  --token-map ./config/new-old-token-pairs.json \
  --output ./work/adapted --device cpu
```

For a certified identical tokenizer, replace `--token-map ...` with
`--same-tokenizer`. `new-model.json` must be a plain `DSpineConfig` object, extracted from the run
config’s `model` field. Choose a compatible rank and dimensions deliberately.
Only use `--head-quantization qwenfast_w8` if the new target really uses that
effective arithmetic.

The tool recomputes PCA whitening, aligns common unit token codes and saves the
raw/aligned bases, eigenvalues, rotation and `adaptation.json`. Compatible draft
tensors are copied; changed shapes initialize fresh tensors. Changed feature
layer IDs or feature dimensions reinitialize `fc.weight`, even when its shape
is unchanged. Review every initialized/copied entry in the adaptation report.
The old head/embedding and proposal policy must not silently carry over. Build
and freeze a new proposal map from training data if using restricted proposals.

5. Generate on-policy continuations using the target's own tokenizer/template and
   declared sampling processor. Preserve prompt source attribution and licenses,
   benchmark exclusions, hash-based validation split and `EXCLUDE_IDS.json`.
   Capture a new feature corpus and independent T1/greedy T0 validation caches.
6. Set `init_checkpoint` to the adapted checkpoint in a fresh run. Train all
   draft parameters with fresh optimizer/RNG state. Do not use `--resume` to
   import old-target optimizer/cursor state. Recalibrate LR, lambda, sequence
   diversity, anchor count and feature budget on the new target.
7. Run CPU tests, overfit and packed/reference parity, check actual decode p,
   then score complete rounds across domains and response depths. Validate live
   exactness/cache commits and benchmark against the new target's own baseline.
   Architecture transfer and good offline tau do not prove a serving speedup.

For Qwen3.8-Flash-Next, the follow-up final-feature probes added only
0.03–0.045 depth-1 overlap, below the prespecified +0.05 adoption threshold.
See [the decision](audit/depth1/finalfeat/RESULTS.md). Reusing that target's
five-layer mean features for a new model is an experiment, not a guarantee.
