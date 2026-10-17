# Release validation

The staged portable tree passed 209 CPU tests. Three tests skip: two CUDA-only
parity tests and the optional private serving-head quantization parity test.
The live overlay PIT-observer integration test is excluded along with that
runtime; the portable PIT distribution/receipt tests are included and pass.
No private production runtime or overlay was added to make a test pass.

The separate synthetic CPU overfit smoke uses four context-dependent examples,
250 AdamW updates and the real chunked CE/L1/refinement path with auxiliary
supervision disabled. It reached 100% greedy proposal accuracy and 0% after
shuffling contexts, with frozen embedding/head/codebook unchanged. This does not
validate real-target convergence or the full auxiliary objective.

Package installation, Python/shell syntax, corrected-config validation, local
links, credential scanning and private-path scanning are also checked. The
secret scanner uses redacted regexes because gitleaks/trufflehog are not installed.
Passing a pattern scan is not a proof that every possible secret format is absent.

All release validation is CPU-only. Historical GPU receipts are identified as
historical evidence, not rerun for this package. No publishing has occurred.
