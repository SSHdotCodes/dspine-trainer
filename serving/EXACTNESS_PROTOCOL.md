# Pre-registered exactness and numerics protocol

Version1, prepared 2026-10-17 before any new speculative outcomes. Dense target:
Qwen/Qwen3-8B, Tencent b16 draft, T1/top-p.95/top-k20/min-p0. Pinned production
SGLang image. TP1, synchronous scheduling, radix cache disabled. All GPU work is
in project containers under bounded reservations. Results may span holds, with
unchanged source/checkpoint/oracle hashes and unique draw ids.

## A. Primary controlled algorithm test

`profile/build_oracle.py` selects one creative-writing, chat and code prefix from
its target-only pool. The original36 candidates were expanded to72 on observing
low first-position entropy, before any speculative outcomes. A second expansion to108 establishes space-delimited output with three example
words after none of the72 qualified at all four greedy positions. The expansions add
creative noun brainstorming, hobby suggestions and valid Python identifier
brainstorming, all permitting successive alternatives without forced punctuation.
After the108-candidate pilot still exposed forced activity/identifier syntax in
chat/code,24 more target-only candidates use a chat word-association game and
a Python triple-quoted word corpus, for132 total. No criterion was relaxed. Measure processed support and entropy at
the first four positions along a deterministic greedy continuation. Require
support>=8 at every position and maximize the minimum entropy within each genre.
If none qualifies, expand the target-only pool and record that expansion before
collecting any speculative outcomes. No selection may use speculative samples.

Enumerate every positive target path through depth2 with full-sequence BF16 HF
forwards, no KV cache, and freeze the resulting conditional rows plus input ids
and source/model hashes in an immutable JSON/SHA256 pair. Sampling filters use
the tie-preserving native joint rule. Because CPU BF16 and GPU verification are
not globally identical, an explicit `DSPINE_TEST_ORACLE` mode feeds these exact
frozen p rows into the installed live chain rejection kernel. The real drafter,
conditional q, proposal walk, target forwards and state updates remain active.
The first target-prefill output also comes from the frozen root distribution.
This mode is an algorithm test, never native serving or a throughput result.
It is incompatible with native PIT logging and fails on a manifest hash mismatch,
unknown prefix, missing positive branch or impossible emitted token.

Collect **50000 independent three-token draws per selected prefix**. Preserve all
raw ids, verification counters, mode, server metadata and unique sample indices.
The expected joint2/joint3 distributions are products of the frozen conditional
probabilities. No sampled or unobserved positive branch is omitted.

Primary endpoint: one goodness-of-fit G-test for joint2 and joint3 per prefix,
six tests with family alpha=.01, threshold=.01/6. Pool low-expected-count leaves
in fixed canonical path order into bins with expected count>=10; merge the last
small remainder. A single global rare-tail bin would erase joint3 power when all
20^3 outcomes have expected count<10, so it is not used. Calibrate G using20000
multinomial null simulations, seed912671. Pearson chi-square is descriptive.
Report plug-in TV,2000 bootstrap percentile intervals and null-sampling TV
intervals. Bootstrap TV is descriptive because its finite-sample bias is nonzero
even under a correct high-dimensional null. Do not interpret a positive TV
interval as an algorithm mismatch.

## B. Native numerical differences, descriptive

Run unmodified DSpine and ordinary target sampling sequentially on the same fixed
prefixes, with50000 draws per prefix and ordinary concurrency1 and2. Compare
joint2/joint3 empirical distributions with TV and bootstrap intervals. These
are descriptive, without rejection gates: BF16 decode, prefill and verify
kernel shapes can produce different probabilities. Separately rerun the old
prefix-2/anchor785 conditional group with>=20000 speculative and ordinary draws,
ten times the original count. Preserve its old exact ids and token17330/17788
counts. Production MTP is not the ordinary control because its private sampling
path uses sequential top-k/top-p rather than the ordinary joint rule.

## Native live PIT endpoint

This test uses **no oracle probability override**. Generate52 distinct prompts
(13 scenario variants of the four production profiles),4096 tokens each,
T1/p.95/k20, ignore EOS, no custom stops. This yields212992 API output tokens;
require>=200000 actually logged tokens for each of DSpine and ordinary control.
Use bs2 initially. No repeated fixed-prefix trials are necessary. Dense runs now;
Flash Next repeats once a trained head exists. Instrumentation is excluded from
throughput measurements.

The verifier retains exactly the processed p used by rejection for each committed
output, including accepted proposals, correction and all-accepted bonus. The
ordinary sampler hook captures the native joint distribution immediately before
sampling. Trim target-verify rows to each request's actual remaining output cap.
Prefill roots are included; these short prompts do not use chunked prefill.

For canonical descending p, ascending id ties, log the complete positive support,
selected token, row index, outcome type, F(y-), p(y), independent V and randomized
u=F(y-)+V*p(y). Float64 normalization of the logged float32 weights handles tiny
roundoff. Native ties can make support exceed20; retain all ties, fail if support
exceeds64 instead of truncating. V uses NumPy PCG64 independent of the torch/native
sampler RNG. Seeds:851984106 spec,851984107 ordinary.

Analyze uniformity overall, among non-point-mass rows (support>1, selected before
the token), and by **reached row index**, including prefill row0. Report support
counts and entropy quantiles so point-mass V randomization cannot hide weak power.
Only groups with>=100 observations receive uniform rejection gates; all counts
and available descriptive statistics are reported. Predeclare66 uniform-null endpoint/statistic pairs for b16 spec plus ordinary:
KS, known-uniform Anderson-Darling, and Ljung-Box lags1..16. Family alpha=.01;
threshold=.01/66. KS uses its uniform asymptotic null. AD uses100000 simulations
of its known-uniform eigenvalue law,1024 components with exact tail-mean correction,
seed170263. Preserve observed statistics and raw p-values, not only pass/fail.

Also report accepted-draft k=1..b-1, correction and bonus outcome groups, **without
uniform rejection gates**. Conditioning on acceptance selects mass proportional
to min(p,q); correction selects (p-q)+. Their PIT distributions need not be uniform
even under exact rejection. Grouping only by the reached row avoids that selection
on the current token. Sparse receipts and reconstruction checks make the source p
and independent randomization auditable.

PIT uniformity is a useful live diagnostic, not a proof of conditional equality:
biases that cancel across prefixes can escape a marginal test. Interpret it with
A, the full-forward/state audit, and downstream quality results.

## C. User-facing quality gate

Compare full GSM8K thinking-off and the fixed agent benchmark against isolated
production MTP with identical runtime/model/sampling settings. Report accuracy,
agent outcomes and uncertainty. No “no quality loss” claim before these checks.
The random Flash Next fixture only validates shapes, ring bounds and state commits.

## Commands and receipts

- `profile/cpu_ground_truth.md`: completed full CPU FP32/BF16 audit and receipts.
- `profile/build_oracle.py`: target-only pilot and complete frozen oracle.
- `profile/joint_exactness.py`: resumable collection and analytic joint fit.
- `profile/native_numerics.py`: descriptive joint TV/bootstrap and10x borderline report.
- `pit.py`, `sglang_overlay/pit_observer.py`: independent PIT RNG and live hooks.
- `profile/pit_live.py`: distinct long prompts, collection and KS/AD/serial analysis.
- Dense launcher: `--test-oracle /experiment/runs/joint-oracle.json --oracle-sha256 HASH`,
  or native `--pit-dir /experiment/runs/pit-spec --pit-seed 851984106`.
- Ordinary launcher: `--baseline --pit-dir /experiment/runs/pit-ordinary --pit-seed 851984107`.
- PIT analyzer: `--max-row 16` for b16 spec, `--max-row 0` for ordinary,
  `--family-size 66` for both. Raw chunks stay under ignored `runs/`; commit summaries.
