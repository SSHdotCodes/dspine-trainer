> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Decision probe for final-layer features (depth1-gap recommendation 3)

Question: does a position-1 probe on the target's **final-layer state** beat our current HC-mean@45 by
≥ +0.05 held-out overlap at equal training examples? If yes, adopt recommendation 3 (capture H_47
as an extra feature at a window boundary). See `docs/audit/depth1-gap.md`.

## What is captured (200 held-out sequences, every response position)

| | tensor | source in the target forward |
|---|---|---|
| (a) | HC-mean after layer 45 | unchanged production capture (`dspine.capture.capture_layer`) |
| (b) | 4 raw HC streams after layer 47, 10,240 dims | `hc_hidden_states`, exactly what the target relays to its MTP head |
| (c) | exact LM-head input | `hyper_connection_mixer.mix(H_47)` (per-stream GemmaRMSNorm + low-rank gates); the head reads it directly |
| (d) | MTP input fusion | computed on CPU from (b): `fc_hidden(GemmaRMSNorm_10240(b))_k + fc_embedding(GemmaRMSNorm(emb(x_{j+1})))`, using MTP's own weights (`fetch_mtp_fusion.sh`). This avoids loading the speculative worker. It is a fixed linear function of (b) and the anchor embedding, so it tests MTP's projection as a prior, not new information. |

All four are FP8 (per token, per pseudo-layer BF16 scale), exactly as training stores features. p is the
production processed top-20 distribution (teacher prefill), as in training shards.

**Selection:** `select_heldout.py`, already run →
`diag/finalfeat/selection/heldout.jsonl` + `selection-receipt.json`.
- The corpus's deterministic 1% validation split (`split_for_id`), which training never uses.
- 200 records: 24 each of agentic/chat/code/math/multilingual, 23 each of stem/structured/writing, 11 longctx.
- 396,405 prefill tokens; 203,270 response positions.

## Production identity, read-only

- Same image as `dspine/scripts/capture_server.sh`: `lmsysorg/sglang:v0.5.21`, checked at launch as
  `sha256:b1259f3ea327…` (`--pull never`).
- Same server flags and the same pinned production overlays: `build_runtime.py` re-verifies all 15
  SHA-256 hashes in `ref/production-capture-runtime/manifest.json`.
- Same `dspine` capture code. The model overlay is `dspine.prepare_overlay.patch(pinned source)` plus 3
  edits. After undoing those edits it is byte-identical to `ref/qwen4_exp-capture-production.py`
  (`test_finalfeat.py`). The edits:
  1. import the hook;
  2. skip pseudo ids ≥ 1000 when setting `_is_layer_to_capture`;
  3. call `finalfeat_capture.extra_capture` after the existing `self._dspine_final_hidden = hidden_states`.
- The extra tensors ride as pseudo-layers `1047,2047,3047,4047` (streams) and `5047` (head input) through
  the unchanged `CaptureSequence`/`FeatureStore`.
- qwenfast defers the HC combine only for ≤4-row chunks (`QWENFAST_HCCOMB_MAXROWS`), and the last layer
  has no successor to defer into. So H_47 in prefill is the materialized residual.
- Differences from the training capture:
  - own container `dspine-finalfeat-capture` on port 30221;
  - `DSPINE_CONTEXT_WINDOW=0` (response rows only);
  - capture root and a copied JIT cache under `diag/finalfeat/`;
  - the sampler-parity probe is skipped (already verified for this image in `runs/sampling-probe.json`).
- The container mounts `./work` **read-only** and only `diag/finalfeat` read-write.
  Nothing touches `ref/`, the production service or other containers. Cleanup stops and removes only
  its own container, by CID.

## Launch (caller holds the GPU and has paused training)

```bash
ssh compute-host ./work/bin/ai-job run dspine-finalfeat env CUDA_VISIBLE_DEVICES=0 \
  bash ./work/diag/finalfeat/tools/finalfeat_job.sh
ssh compute-host ./work/bin/ai-job wait dspine-finalfeat --timeout 1500 --lines 30
```

The job refuses to start, before anything heavy, when:
- the image ID differs;
- the GPU has >6 GB in use;
- host `available` RAM is <108 GB;
- a `dspine-finalfeat-capture` container exists;
- the selection is missing.

Output: `diag/finalfeat/run-<time>/{job.log,client.jsonl,server.log,resources.csv,receipt.json}` and
`diag/finalfeat/features/shards/*.pt`.

| expectation | value | basis |
|---|---|---|
| wall time | **~5-7 min** (hard ceiling ~19) | startup 194 s (last capture run); 396K prefill tokens at 4.9-7.2K tok/s ≈ 1-1.5 min; teardown <1 min; startup timeout 10 min, client cap 8 min |
| peak VRAM | **~85-90 GB** (one GPU) | same server: weights 71.5 GB + KV 3 GB within `--mem-fraction-static 0.93`, 18.4 GB headroom; the hook adds ~0.5 GB transient per 8K chunk |
| peak host RAM | **~108-110 GB** | PLE table ~102 GB pinned, plus server and client; `receipt.json` records the measured peaks (2 s sampling) |
| disk | ~5.3 GB features | +3.1 GB compact probe rows |

## After the GPU run (CPU; AUDIT-MODEL-EVAL does this)

```bash
ssh compute-host 'nice -n 19 python3 -I ./work/diag/finalfeat/tools/compact.py \
  ./work/diag/finalfeat/features/shards ./work/diag/finalfeat/probe 1'
scp -r compute-host:./work/diag/finalfeat/probe /tmp/audit-depth1/finalfeat-probe
python3 docs/audit/depth1/finalfeat/finalfeat_probes.py --data /tmp/audit-depth1/finalfeat-probe \
  --mtp /tmp/audit-depth1/mtp-fusion --output docs/audit/depth1/finalfeat/finalfeat-probes.json   # ~35-40 CPU minutes
```

`finalfeat_probes.py` trains the same MLP probe on a, c, b, d, a+c and a+b, with soft CE to
p_{j+2}, 4 epochs and identical positions (~170K train / ~30K held-out by sequence-id hash). It then
prints `decision.adopt_final_layer_features`.

Built-in correctness check: the zero-shot head on (c) must reproduce p_{j+1}. Expect overlap ≳ 0.9, not
exactly 1: p is the top-k/top-p processed distribution, so its nucleus mass bounds the overlap, plus FP8 error. On (a) it should give ≈0.58, the earlier audit value. Afterwards delete
`diag/finalfeat/{features,probe,cache}`.

## Files

| file | role |
|---|---|
| `finalfeat_capture.py` | container hook |
| `build_runtime.py` | pinned overlay + docker arguments |
| `select_heldout.py` | held-out selection |
| `finalfeat_job.sh` | GPU job |
| `compact.py` | probe rows |
| `finalfeat_probes.py` | decision probes |
| `fetch_mtp_fusion.sh` | MTP fusion weights |
| `test_finalfeat.py` | CPU smoke tests: 4 passed |
