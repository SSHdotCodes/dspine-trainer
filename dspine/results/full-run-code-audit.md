# Full-run CPU code audit

2026-10-17 00:47 EDT. Local CPU only; no GPU job, reservation, lifecycle operation, rsync, source-config edit, or campaign append. Remote actions were reads of frozen run.json and existing metrics/state. Chunk ownership belongs exclusively to the orchestrator driver.

## Confirmed fixes, local only

Both changes are in `dspine/rolling_append.py`. Sync this file before the campaign2 append; no trainer/model/config change is required.

1. **Valid source with aborts rejected:** `datagen.merge` excludes abort records before its validated record/token totals. The append counted them before skipping their training data, so a valid SUMMARY mismatched and the append failed. Fix: validate revision/sampling, skip aborts, then increment source counts. Exclusions still apply first; short non-aborted records remain in source totals, matching merge semantics.
2. **Missing exported platform silently omitted:** validation compared only platforms discovered in current source files. A SUMMARY could include a campaign2 platform whose export was absent, yet append+seal would publish the other platform and silently omit its data. Fix: compare record/token counts across the union of discovered platforms and every matching platform in SUMMARY. Missing positive counts now fail without changing plan.json.

Regression tests cover both cases, stale validation, immutability, append timing, and end-to-end continuation.

## Frozen-config path review

| Path | Evidence / outcome |
|---|---|
| CUDA36GB cap | Orchestrator fixed index resolution in b39e9b0. New CPU mocks execute actual startup for cuda/current index, cuda:2/explicit index and cpu/no CUDA call. No CUDA context is created. |
| 189090-ID row-INT8 head | Observed restarted GPU startup and optimizer updates succeed. Frozen sorted-ID hash and resume policy equality are checked; saved policy buffers reconstruct from the same effective BF16 head. Existing CPU tests compare restricted CE/full-p L1 and gradients to a full-vocabulary naive reference, and quantized rows to serving. |
| eval_before_training + exact decode p | Existing GPU metrics show step0 complete-round T1, same-cohort opening, T0 and long diagnostics all succeeded. Evaluation took22.066s, including14.304s for all3 long shards, complete=true. This is observation of the orchestrator's run, not a new trainer GPU test. |
| eval_long_every=2000 | The same branch already runs at step0 because0%2000==0. Future periodic calls use the same implementation. Timeout reports complete=false explicitly and long diagnostics do not select best checkpoints. |
| Packed/checkpointed training with token schedules | New CPU integration combines activation checkpointing, restricted row-INT8 policy, exact synthetic decode-p validation, whole-round/opening/T0/long eval, fixed1M safety cap and cumulative response-token schedule. Optimizer updates continue after each eval. |
| Window change / --reset-data | Existing bitwise tests compare interrupted vs uninterrupted rotation and early vs late append. New combined integration resumes a controller at rotate, loads optimizer/RNG/policy, resets only the next window's data cursor, and preserves the48-token cumulative counter. |
| Rotation / purge | Matching durable checkpoint and data_window are required; receipt is fsynced before deleting only manifest-owned features. Interrupted purge is idempotent; unowned shards fail before any deletion. External eval/long features are outside the rolling-train root. |
| Checkpoint retention | New test keeps independent b6/b8 winners while optimizer checkpoint_keep=1 deletes earlier step directories. Winners use separate best-step directories and policy files; existing tests check pointer migration and superseded-winner pruning. |

No additional confirmed trainer runtime crash or numerical-contract mismatch was found. CPU tests cannot certify future CUDA kernel shapes or a GPU rotation that has not yet happened; prior GPU optimizer resume and the now-observed full-config startup complement these tests.

Full CPU suite:111passed,2CUDA-skipped,29.69s. Receipt: cpu-tests-full-run-audit.txt. No executable training/model code changed in this audit.
