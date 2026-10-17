# DSpine Trainer

The first open DSpine trainer, to our knowledge: an independent, config-driven
implementation of [DSpine](https://arxiv.org/abs/2609.36173) with training,
target-feature capture, transfer-space construction, exact proposal probabilities,
and CPU reference verification. No training implementation was publicly available
when this project began on October 16, 2026. This is a research implementation
by SSHdotCodes, not an official Tencent release.

**The Qwen3.8-Flash-Next head is an experimental preview.** Training stopped by
owner decision at step 9,145 on October 17, 2026. It did not achieve the speed goal.
Our serving-cost model estimates **0.6–0.7x native MTP throughput**, meaning slower
than the base model's strong native MTP. This estimate is not a matched end-to-end
benchmark of the final head. Use the head for research and as a warm-start base.

Final 64-prompt complete-round tau (CPU FP32): **T1 b6/b8 1.799928/1.823011**,
**T0 b6/b8 1.849828/1.867662**. See [the score and limitations](docs/findings.md).

## What is validated

- Loading `tencent/Qwen3-8B-DSpine-b16` with Qwen3-8B reproduces **10.670 prompt-mean
  tau** and **9.857 pooled tau** on the first 100 GSM8K test prompts, greedy,
  thinking off, block 16, 512-token cap. Tau counts emitted tokens per verification,
  including the bonus. These two aggregations must not be interchanged.
- A four-example synthetic CPU overfit smoke (auxiliary off), independent
  dense-loss/gradient, packed/reference, proposal/cache,
  cursor/resume, and rolling-window tests exercise the training mechanics.
- A seven-agent audit covered the paper, loss, data, loop, model/evaluation,
  clean-room differential reference, and same-target DFlash comparison. It found
  recipe and evaluation limitations; it did not establish that this target will
  converge to a useful serving speed. [Findings and evidence](docs/findings.md).

## Recipe lessons

Use `lambda_ce_reduction="weighted_token_mean"`: the detached CE multiplier for
the auxiliary loss should be on a per-token scale. The historical position sum
increased it by about 6.63x at block 16. This interpretation is a controlled recipe
choice, not a claim of access to Tencent's unpublished training code. At our
roughly 1,024-anchor/four-sequence update size, **3e-4 LR beat 6e-4** in follow-up
forks. The example configs use the corrected recipe.

Sample unique response anchors with fixed-capacity, zero-weight padding for short
sequences; skip duplicated first-position refinement; mask CE labels outside
stored teacher support. Use actual decode probabilities for exact stochastic
evaluation. Teacher-prefill probabilities can differ numerically. Target sampling
is T=1, top-k=20, top-p=0.95 with the runtime's **joint, cutoff-tie-preserving**
processor, rather than sequential top-k then top-p. Preserve the full processed
target law even when draft proposals use a restricted vocabulary.

Opening-only tau is optimistic on formulaic reasoning prefixes. Select using
complete-round tau and report the prompt cohort, response cap, temperature,
proposal policy, and aggregation. The retained 64-prompt cohort has short,
64-token continuations; it is not a completed-answer benchmark.

## Quickstart

Python 3.10+, PyTorch 2.6+ for CPU reference tests. Full training uses a compatible
CUDA/FlexAttention stack and separately obtained target weights. From this root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './dspine[validation]'
bash dspine/scripts/run_tests.sh
python -m dspine.train --help
```

`serving/` contains our portable exact-sampling, frozen-oracle and randomized PIT
references and tests. It is not a deployment runtime. The
[exactness protocol](serving/EXACTNESS_PROTOCOL.md) separates controlled algorithm
tests from floating-point differences in live target kernels. The
[MTP-first hybrid note](docs/notes/hybrid-mtp-dspine.md) is a design and CPU study,
not an implemented or measured serving speedup.

For a real training run, obtain the frozen target snapshot locally and prepare
generation records and feature shards using the [data contract](docs/SPEC.md#4-data-contracts).
Set paths in a copy of `dspine/configs/qwen38-flash-next.json`. Paths are relative
to your launch directory; use absolute paths for detached jobs. Build C first:

```bash
python -m dspine.transfer_space --target ./models/target \
  --rank 1024 --head-quantization qwenfast_w8 \
  --output ./work/ref/transfer-space-qwenfast-w8.pt --device cpu
python -m dspine.train --config ./config/train.json
```

For this preview's exact target-head arithmetic, pass
`--head-quantization qwenfast_w8`, set `target_head_quantization` to the same value,
and use the matching transfer space. This is a portable PyTorch reconstruction;
private qwenfast kernels and runtime overlays are not included. A public runtime
capture needs its own numerical validation and may require adapting the capture
hook to the installed model source. See [training and capture](dspine/README.md).

## Train on a new target

[Adapting a target, including a future Qwen 4 Flash](docs/adapting-targets.md)
covers feature hooks, token maps, PCA/Procrustes alignment and fresh fine-tuning.
No future Qwen 4 Flash dimensions or compatibility are assumed. Change the model
config, recompute C, recapture features, and validate end-to-end before serving.

## Scope and license

Code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Paper text, Tencent
checkpoint files, target weights, private production/overlay code, private kernels,
cloud generation orchestration, credentials, and raw datasets are not bundled.
Obtain third-party artifacts from their authors and follow their licenses.

Proposed preview head: `ProCreations/Qwen3.8-Flash-Next-DSpine-preview`.
Base model: [RedHatAI/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-Flash-Next-NVFP4).
Final checkpoint metrics are recorded in [findings](docs/findings.md).
