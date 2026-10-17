> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Final-layer feature decision: results (2026-10-17)

**Decision: do not adopt.** No input reaches +0.05 position-1 overlap over (a) alone. The best, a+d, gains
+0.045 (95% CI 0.029 to 0.064), and the bootstrap puts P(gain ≥ 0.05) at 25%. The gains are real
(every CI excludes 0) but modest. They stay flat or shrink over epochs, the pattern of a better
starting point rather than missing information.

**Run.**
- Capture: `run-20261017-142924`, 200 held-out sequences, 203,270 response positions. It took 277 s
  wall and peaked at 81.0 GB GPU and 107.0 GB host.
- Compaction the workstation: 3 s, peak RSS 1.1 GB.
- Probes: CPU, 162,702 train and 40,168 held-out positions from 38 held-out sequences, identical for
  every probe. Each probe trains the same MLP for 4 epochs with soft CE to p at depth 1.

**Sanity (depth 0, zero-shot target head).**

| input | overlap | acc16 | top-1 |
|---|---|---|---|
| (c) LM-head input | **0.979** | 0.985 | 0.841 |
| (a) HC-mean@45 | 0.545 | 0.546 | 0.519 |

On (c) the overlap is bounded by p's top-p nucleus, so the capture reproduces the head's input.

## Position-1 (depth-1) held-out overlap

| input | capture slots (×2560 B) | overlap | gain over a | paired 95% CI (by sequence) | acc16 |
|---|---|---|---|---|---|
| a = HC-mean@45 (today) | 1 | 0.464 | — | — | 0.476 |
| c = LM-head input | 1 | 0.452 | −0.012 | | 0.464 |
| b = 4 streams after L47 | 4 | 0.492 | +0.028 | | 0.506 |
| d = MTP input fusion of b | 4 | 0.507 / 0.505 | +0.043 / +0.040 | 0.022 to 0.062 | 0.518 |
| a+c | +1 | 0.497 / 0.498 | +0.033 / +0.034 | 0.017 to 0.055 | 0.510 |
| a+b | +4 | 0.494 | +0.030 | | 0.508 |
| a+d | +4 | 0.509 | **+0.045** | 0.029 to 0.064 | 0.520 |
| a+m (HC-mean@47) | +1 | 0.493 | +0.029 | 0.016 to 0.044 | 0.504 |
| a+dm (stream-mean of d) | +1 | 0.493 | +0.029 | 0.017 to 0.042 | 0.503 |

Where two values appear, they come from the main run (`finalfeat-probes.json`) and the rerun with
the bootstrap (`finalfeat-probes-ci.json`), which used a different initialization order. Seed noise
is ~±0.003. The CIs resample the 38 held-out sequences, 10,000 draws.

## Reading

- **(c), the head's input, is the wrong feature for position 1 on its own.** It is tuned to emit
  x_{j+1}, which is already known at depth 1. It only helps alongside (a).
- **The 4 raw streams carry a little more than any one-slot summary.** b is +0.028, and MTP's
  pretrained projection of them (d) is +0.04. d is a fixed linear map of b plus the anchor embedding,
  so the extra +0.012-0.015 over b is a better starting point, not new information.
- **The gain does not grow with training.** By epoch, for d: +0.048, +0.050, +0.044, +0.043. For b:
  +0.048, +0.036, +0.028, +0.028. The drafter trains on ~30× more examples. This suggests the gain
  there is at most this size, though that is an extrapolation.
- **This revises `depth1-gap.md` rec. 3,** which expected +0.05 to +0.15 at depth 1 from the depth
  trend. The measured value is +0.03 to +0.045. Layers 46-47 add less than layers 34→45 did.

## Cost, had the rule passed (for reference)

- A one-slot option (a+c or a+m) costs **+20%** feature bytes: +2,562 B/token, so 12,978 → 15,540.
  Adding b costs +79%: +10,248 B/token, so a 450 GB window holds 19.4M instead of 34.7M tokens.
- Prefill time does not change: this capture ran at 7.2K tok/s, the same as production capture.
- The change would mean a new rolling plan, because `rolling.window_ready` rejects shards with
  different `layer_ids`. It would also need a capture.py id for the head input/streams and new
  zero-initialized `fc` columns (`adapt_target` reinitializes `fc` when layer ids change).

Not recommended now. Round 4's last-state fusion targets the larger access/optimization gap and needs
no capture change.

## Reproduce

```bash
python3 docs/audit/depth1/finalfeat/finalfeat_probes.py --data /tmp/audit-depth1/finalfeat-probe \
  --mtp /tmp/audit-depth1/mtp-fusion --output docs/audit/depth1/finalfeat/finalfeat-probes.json            # ~21 min, 16 threads
python3 docs/audit/depth1/finalfeat/finalfeat_probes.py ... --probes a,d,a+d,a+c,a+m,a+dm \
  --output docs/audit/depth1/finalfeat/finalfeat-probes-ci.json                                              # ~20 min
```

The features (4.9 GB), probe rows (3.0 GB) and JIT cache copy (1.1 GB) are still in
`diag/finalfeat/{features,probe,cache}` on the workstation. Delete them when no longer needed.
