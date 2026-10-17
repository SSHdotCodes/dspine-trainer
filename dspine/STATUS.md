# Release status, 2026-10-17

Owner stopped the Qwen3.8-Flash-Next training run at step 9145. The preview is a
research artifact and warm-start base; the serving-cost estimate is 0.6–0.7x
native MTP throughput. No final-head end-to-end speedup is claimed.

See [findings](../docs/findings.md) for validation, recipe lessons and limitations.
CPU test and release-scan receipts are under `../docs/release/`.
Private production runtimes, kernels, cloud orchestration and data are excluded.
