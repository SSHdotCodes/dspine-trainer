> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# AUDIT-DATA: What the trainer sees

Independent read-only audit, 2026-10-17. Scope: data, feature capture, batch construction, and offline evaluation data paths. No `dspine/` files were edited. No the workstation GPU, containers, services, training state, or model lifecycle controls were used.

## Ranked findings

1. **CONFIRMED BUG: the training data violates the claimed on-policy `(y, p)` contract.** On 10 finished window-1 shards, **18,894 / 18,990 response labels have positive stored probability (99.494471%)**. Through the actual `PackedSequenceStream` batch constructor, excluding the anchor and invalid tail slots exactly as training does, the rate is **37,370 / 37,508 (99.632078%)**, with **138 supervised positions having `p(y)=0`**. Training stores recomputed prefill probabilities; its generation records do not contain actual-decode probabilities. All **4,090 / 4,090** response tokens in the 64-sequence actual-decode eval cache pass, including **1,152 / 1,152** in the active 18-sequence cohort. This is a confirmed contract mismatch, not evidence of a wholesale indexing shift. Its effect on tau needs a matched-runtime A/B; these frequencies alone do not explain a factor-of-two acceptance deficit.
2. **RISK: the main “complete-round” metric covers only the first 64 response tokens.** Every sequence in the actual active cohort is **64 tokens, `finish_reason=length`**. **17 / 18** remain inside an unclosed thinking segment throughout; the other prompt closes thinking in the template. The metric walks successive rounds correctly, but does not measure response bodies, long response offsets, or completed answers. Two held-out prompts per domain is also a small independent sample. This can materially distort checkpoint selection and interpretation of the opening/body gap. Separate long diagnostics already exist in the run configuration, so this is a limitation of the main metric, not a claim that no long evaluation exists.
3. **RISK: exact numerical equality of target prefill features and live decode/verify features is not established by this CPU audit.** The selected layer IDs, HC mean computation, placement, feature order, window, and positions agree with serving source, and real-feature CPU tests pass. However, raw four-stream target tensors were not saved with the sampled shards. CPU checks cannot prove the live target's prefill/decode kernels produce identical HC means. A bounded GPU test is specified below; no GPU work was attempted.
4. **DEVIATION FROM PAPER, intentional in the project contract:** this corpus includes thinking, long sequences, and 256 anchors per sequence, versus Appendix A's thinking-disabled ShareGPT, maximum training sequence length 3,072, and 512 anchors. In the full frozen window, approximately **72.547% of generated tokens are in the thinking segment**, counting its closing marker and accounting for templates that already close thinking. This is not a prompt/response boundary error. Do not change the production training distribution merely to imitate the paper; compare under matched evaluation.
5. **OK: no observed feature/label/position shift or future leak.** Actual batch labels/p cover `anchor + 1 .. anchor + 15`; contexts end at `anchor - 1`. Independent scaled-FP8 reconstruction matches **222,208,000 / 222,208,000** sampled values exactly. Ten real-feature packed/cropped FP32 comparisons have maximum hidden-state error **8.34465e-7**; perturbing future features and future token IDs causes exactly zero change. The processed probability code matches an independent joint top-k/top-p oracle on saved real full-vocabulary logits.

## Evidence and reproducibility

Read first: `docs/audit/BRIEF.md`, `docs/SPEC.md` sections 3–4, the paper section 4 and Appendix A. Local `dspine/data.py` was byte-identical to the workstation's installed code mirror when copied. The actual frozen `runs/rolling/run.json` confirms packed training, 256 anchors, four accumulated sequences, W=4096, layers `[1,12,23,34,45]`, and an 18-sequence packed-replay eval cohort.

Copied with `scp` to `/tmp/audit-data/`:

| Source on the workstation, relative to `./work/` | Sample |
|---|---|
| `features/rolling-train/shards/` | 10 completed `.pt` files, 352,494,108 bytes, all nine domains; no `.partial` files |
| `features/pilot-eval/T1/shards/` | All 64 actual-decode feature shards |
| `features/pilot-eval-prefill/T1/shards/` | All 64 older prefill feature shards |
| `runs/rolling/manifests/window-00001/records.jsonl` | Complete frozen window, 8,229 generation records |
| `runs/pilot/eval-generations/T1/`, `eval-generations-prefill/T1/` | Matching generation records for the two distinct eval trajectories |
| `gen/<platform>/<shard>/part-*.jsonl.zst` | Original generation parts containing all 10 sampled training records; exact equality to frozen records verified |
| `features/prefix-production/` | One real BF16 feature probe plus its 16 full-vocabulary target-logit rows |
| `ref/qwen4_exp-capture-production.py`, installed `data.py`, pinned tokenizer files | Provenance, capture placement, token boundary inspection |

`pilot-eval-decode/` is empty; the actual-decode distributions are attached to `pilot-eval/T1/shards/`. Total local sample and metadata footprint was **1,953,316,178 bytes**, below the brief's 2 GB ceiling. Analysis ran on the Mac with torch 2.6.0. The only bespoke remote Python streamed original generation parts at `nice -n 19`, with a **1.5 GB address-space hard limit**, no torch import, and no remote writes. No raw features or generation records are committed.

Reproduction scripts, writing their numeric outputs only into `/tmp/audit-data/`:

```sh
python3 docs/audit/data/scp_matching.py
PYTHONPATH=. python3 docs/audit/data/check_data.py
PYTHONPATH=. python3 docs/audit/data/deep_checks.py
```

The first script locates original parts for the selected records. The two analysis scripts assume the above small samples already exist. Exact selected shard names, generation metadata, original-part locations, and full numeric outputs are retained locally in `meta/selected-train.json`, `meta/original-gen-parts.json`, `results.json`, and `deep-results.json`. The sample is deliberately domain-spanning plus one long response; its failure rate is **not** a population estimate.

## 1. Labels and target distributions

### Actual batch measurements

`data.py:550` samples response-only anchors. `data.py:562` builds offsets, then indexes **the same offsets** into token IDs, sparse probability IDs and probabilities at `data.py:574`. `train.py:568` consumes these batches without another shift. The loss input slicing drops index zero consistently for labels, p, and masks (`losses.py:247`, `:258`). The audit calls these batch constructors, rather than implementing a parallel loader.

| Cohort | All response tokens with `p(y)>0` | Packed valid candidate slots with `p(y)>0` | First-anchor valid candidate slots |
|---|---:|---:|---:|
| 10 training sequences | 18,894 / 18,990, **99.494471%** | 37,370 / 37,508, **99.632078%** | 149 / 150 |
| 64 actual-decode eval sequences | 4,090 / 4,090, **100%** | 219,244 / 219,244, **100%** | 960 / 960 |
| 64 older prefill eval sequences | 4,068 / 4,078, **99.754782%** | 218,510 / 218,975, **99.787647%** | 957 / 960 |

Packed measurements use 256 anchors/sequence, seed 42, one pass over the copied shards, with replacement. “Packed eval” is a batch-alignment stress check, not the production eval's anchor policy or an estimate of model quality. The production cohort is selected with the actual `domain_balanced_paths(..., seed=42)[:18]`; it has 1,152 / 1,152 positive-support response labels.

The literal first-20-column training support rate is **18,887 / 18,990 (99.457609%)**. Seven additional labels live in the stored cutoff-tie extension. `target_distribution()` correctly concatenates this extension. Discarding it would falsely add seven failures. There are **1,203 / 18,990** training positions with more than 20 positive entries, mostly from the long response, and 25 / 4,090 in eval. “Top-20” here legitimately means top-20 plus strict-greater cutoff ties.

| Training domain / record | Response tokens | Zero-probability labels |
|---|---:|---:|
| chat, `b7d99d12d3772891b980b2624644b69cbfc791a0` | 563 | 1 |
| multilingual, `c5c6bb70c539ffb902eca781cfe937c7b60d122e` | 807 | 1 |
| math, `827f277e59d09d477356662d6d95ac93b3768bbc` | 810 | 1 |
| agentic, `db519f5d253cef14ea77fec3d48a6c08b282468e` | 90 | 1 |
| code, `560324f27a837b17c082cc32dbfbe466aa46a72d` | 1,855 | 3 |
| structured, `54878c7a330b2c9a49512b171eae913ca7a8ebd2` | 499 | 1 |
| writing, `1249ea0805034b8e9c744ba22c0fd76cc007928b` | 361 | 0 |
| stem, `566f6ca142e563ace6f4bc692d14061ae64ff13b` | 1,093 | 2 |
| longctx, `e5a072b5a960b06b97e3ae02b8fddc3b292d5820` | 95 | 1 |
| writing, `ac392fb94c8ad1dc028d33c5ad41b3a8ebadc22a` | 12,817 | 85 |

These are scattered positions, not just EOS or chunk boundaries. Examples: structured response offset 13 has label ` that` with stored probability zero while ` deployed` has probability 1; math offset 558 has label ` every` outside the processed support; long writing has failures at offsets 96, 296, 1092, 1318, and many later locations.

As a negative control, pairing each label with the **previous** stored distribution gives only **2,119 / 18,990 (11.1585%)** support, and with the **next** gives **108 / 18,980 (0.5690%)**. The high unshifted rate and exact common-offset indexing rule out a global one-token shift.

### Why training differs from eval

`capture.py:211` recomputes p from prefill logits. `CaptureSequence.append` makes the one necessary next-token-to-token shift (`capture_sequence.py:56`), carrying the last row across chunk boundaries (`:58`, `:74`). `extract_features.py:81` calls `attach_decode_probs`; however, `decode_probs.py:56` silently returns when a generation lacks `target_decode_ids`. All 10 sampled train records lack these fields and all ten shards have the prefill source. Campaign generation (`datagen/client/generate.py:124`) requests neither output logprobs nor actual p. Eval generation explicitly requests raw top-64 logprobs (`generate_eval.py:105`) and attaches reconstructed joint probabilities.

Every actual-decode eval sparse distribution matches its generation record's decoded probability map to <1e-7. The older prefill eval cache is **not the same generated trajectory**: only 1 / 64 complete token sequences is identical. Therefore, comparing their full arrays position-by-position would be invalid. Restricting comparison to **550 positions with identical correct prefixes**, through the first differing emitted token, gives prefill-versus-actual-decode probability L1: **mean 0.0354833, median 0, p95 0.185769, max 1.152048**. That is numerical/runtime target drift, not a loader shift. Cross-platform campaign generation adds another possible source of drift; actual B200 p alone would restore that trajectory's on-policy property but would not prove equality with the workstation's serving distribution.

### Joint processing is correct

Across all 138 primary shards, there are no nonfinite probabilities and no response-row mass errors above 1e-4. Maximum observed mass error is **2.38419e-7**.

For 16 saved real target-logit rows over the full 248,320 vocabulary, the audit independently keeps a token iff its probability meets the 20th cutoff and the **full-softmax mass of strictly greater tokens is <0.95**, then renormalizes. Maximum L1 error against `processed_top20` is **2.38419e-7**; against stored p it is **1.19209e-7**. `processed_decode_probs` reconstructed from real-logit top-64 logprobs agrees to **1.19209e-7**. A sequential top-k-renormalize-then-top-p implementation differs on **9 / 16** rows, by up to **0.109254 L1**. The current code implements joint semantics, not that incorrect sequential alternative.

## 2. Features, HC mean, quantization, windows and positions

All 138 train/eval primary shards have layers **`[1,12,23,34,45]`**, feature shape `[tokens,5,2560]`, FP8 bytes in uint8, and BF16 scales of shape `[tokens,5]`. Features flatten layer-major to 12,800 values/token. Capture invokes the hook **after** `self.layers[i]`, so the IDs are zero-indexed post-layer outputs, not hidden-state tuple indices or pre-layer states. The copied live generated overlay verifies this placement. `capture.py:86` takes the float32 mean over four HC streams, then casts to the residual dtype. Serving's `prepare_target_overlay.py:19` performs the same reshape/mean/cast at the same point, in increasing layer order. Both set the selected-layer canonicalization flag to prevent deferred HC combine.

`data.py:145` reinterprets uint8 as `float8_e4m3fn`, multiplies by the per-token/per-layer scale in float32, then casts/concatenates. An independent reconstruction of 128 sampled tokens per shard, or all tokens when shorter, agrees in **222,208,000 values with zero unequal or nonfinite values**. Scales are not transposed, broadcast across the wrong layer, or applied twice.

| Layer | Train reconstructed RMS | Eval reconstructed RMS | Real BF16 probe RMS | Probe after FP8 round trip RMS | Relative FP8 L2 error |
|---|---:|---:|---:|---:|---:|
| 1 | 0.00989 | 0.01036 | 0.009868 | 0.009865 | 2.586% |
| 12 | 0.03688 | 0.03805 | 0.039012 | 0.038988 | 2.640% |
| 23 | 0.04555 | 0.04724 | 0.048574 | 0.048541 | 2.618% |
| 34 | 0.07257 | 0.07586 | 0.080576 | 0.080491 | 2.571% |
| 45 | 0.42767 | 0.38906 | 0.490749 | 0.490538 | 2.637% |

The BF16 probe consists of real captured HC means, not random tensors. Its FP8 round trip directly measures quantizer damage. Unscaled training FP8 RMS would be approximately 56–82, grossly larger than the correct values, so the magnitude check is informative. The actual loader applies the scales correctly. Serving receives BF16 means directly, leaving a normal ~2.6% FP8 training perturbation; whether that perturbation affects tau merits the paired test below, not a claim of corrupted features.

The target prefills the complete original sequence; only stored features are cropped at `max(0,prompt_length-4096)` (`capture_sequence.py:21`). All 138 primary shards exactly match the matching generation's token IDs and boundary, and the crop formula. A sampled long prompt has original response start **8,963**, stored sequence start **4,867**, and stored response offset **4,096**. At its first anchor the context positions are **4,867..8,962**, anchor position **8,963**, block positions **8,963..8,978**. They do not restart at zero.

For a second sampled sequence with a 12,817-token response, real CPU comparisons cover anchors around **4,095/4,096/4,097**, at 8,000 and near the end. W remains 4,096; e.g. anchor 8,000 sees positions 3,904..7,999, while block positions start at 8,000. Across ten cases, a small random two-layer FP32 drafter using the **real full 12,800-dimensional features** agrees between packed and individually cropped paths within **8.34465e-7**. This verifies data/position semantics, not the trained checkpoint or CUDA FlexAttention kernel. Future-feature and future-ID perturbations produce exact equality.

Packed training dequantizes the entire stored sequence (`train.py:580`), but its attention mask admits only feature positions `max(0,a-W) <= k < a` (`modeling.py:153`), and only causal positions of the same draft block. Live serving uses exactly `anchor-W <= ring_pos < anchor` (`serving/runtime.py:95`). The anchor's target feature is **not** available to either path: it is the residual/bonus token whose state enters target KV only during the next verify round (`serving/sglang_overlay/worker.py:197`). Training supplies the anchor's known **token embedding**, which is correct.

## 3. Response anchors, masks and template boundary

`PackedSequenceStream` samples uniformly with replacement from `[response_start, length-2]`; it does not anchor inside prompt tokens or on terminal EOS. Labels include the known anchor at column zero. All subsequent columns index token and p at `anchor + j`, for j=1..15. The audit checks every sampled offset, token, and validity mask. It observes **892 invalid tail candidate slots / 38,400 (2.322917%)** in the 10-shard training sample, and every such slot is masked. Clamped tail labels/p are placeholders, not additional valid supervision. End-of-sequence matches are not counted as real positions.

All 10 train records match the original `gen/` records exactly, and all 128 eval feature records match the appropriate eval generations. No output retokenization is involved. `datagen/common.py:74` applies the pinned chat template and sends exact input IDs; capture validates concatenated token IDs (`capture_sequence.py:36`). Across the 10 train plus 64 actual-decode eval records, 71 prompt suffixes end in `assistant\n<think>\n`; three close the empty thinking segment in the prompt. The stored response boundary starts after all of those prompt/template tokens. Thinking tokens generated after that boundary remain valid assistant response tokens and eligible anchors. This is intentional and consistent with inference.

## 4. Training versus opening and complete-round evaluation

Opening evaluation uses `AnchorStream(anchor_policy="first")`, `collate_anchors`, the known anchor in label column zero, and labels/p/masks sliced identically at column one (`eval_offline.py:112`, `:137`). Its context crop and absolute positions come from the same loader used by the unpacked training path.

Complete-round serial evaluation explicitly slices features before the current anchor and adds `row['start']` to positions (`eval_rounds.py:242`). It slices both labels and p at `anchor+1` (`:270`, `:285`) with the same near-end mask. Packed replay calls the same `forward_packed` path used by training (`:106`); later anchors are independent blocks masked from earlier ones. It reuses caches across different block lengths, which is valid under block causality. It censors length-truncated tails when a complete round lacks its bonus (`:236`), rather than pretending an unavailable token exists.

The intended differences are **anchor policy** (uniform random training versus first response/opening versus emitted-round walk), **p provenance** (prefill training versus actual decode eval), and teacher substitution (training curriculum versus none in eval). No extra label/feature shift is found. The current first-anchor metric uses the same selected prompt cohort as complete-round evaluation in `train.py`; its higher tau is not evidence that one loader includes the anchor feature. The active cohort contains **two sequences per each of nine domains**, but only **1,152 generated tokens total**, all truncated at 64. Its main quality number cannot establish body acceptance.

## 5. Shard and batching statistics

The frozen window-1 generation manifest contains **8,229 unique records**, **11,589,013 prompt tokens**, and **19,068,416 response tokens**. The sampled append-only capture manifest listed **4,039 finished shards** while capture was in progress. Reported full-window statistics come from the frozen generation manifest, not an assumption that all its features were already finished.

| Length | Min | p25 | Median | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Prompt | 14 | 110 | 229 | 612 | 10,627 | 32,019 |
| Response | 2 | 235 | 664 | 1,896 | 15,056.2 | 16,384 |

| Domain | Sequences |
|---|---:|
| agentic | 2,070 |
| code | 1,640 |
| chat | 1,069 |
| multilingual | 988 |
| math | 983 |
| writing | 655 |
| structured | 330 |
| stem | 248 |
| longctx | 246 |

There are **7,681 stop / 548 length** completions; reasoning effort counts are **1,636 low / 2,506 medium / 4,087 xhigh**. Platforms are Modal v3 4,082, the workstation 836, Vast 2,754, and earlier Modal 557. No record IDs or complete prompt+output token sequences are duplicated. There are **two repeated output-only sequences** across distinct prompts. Neither current exclusion-list ID occurs in the window, and its IDs do not overlap the 64 held-out eval IDs.

Each microbatch is one sequence, hence one domain, with **256 anchors**. Four such microbatches accumulate into an optimizer step, giving **1,024 nominal blocks**. A reproduction of the actual seeded initial shard shuffle on the 4,039-finished-shard snapshot yields 1,009 complete four-sequence groups: **349 with four domains, 495 with three, 155 with two, 10 with one**. Thus **999 / 1,009 (99.009%)** groups are domain-mixed, without a guarantee of one-per-domain. This is consistent with `rolling.py:260` stratified window construction followed by random shard order. A finished window's complete order can differ from this in-progress snapshot.

Anchors are intentionally sampled with replacement. The ten real sample batches contain **1,920 unique offsets / 2,560 draws**, i.e. **25% repeated offsets** within their sequences. Across the full window response-length distribution, the analytic expected duplicate-anchor fraction is **24.9533%**. **2,227 / 8,229** responses are <=256 tokens; **1,217** exceed 4,096. Short sequences therefore repeat more anchors and receive higher per-token sampling weight than long sequences. This is the implemented per-sequence policy, not duplicate input data or an accidental replay bug. It should be considered when comparing to the paper's batch/anchor count, but changing it changes the training objective and is not recommended without an A/B.

## Proposed changes, not applied

### A. Restore an explicit on-policy probability contract

The exact corrective route is to obtain `(y,p)` from the **same numerical target runtime** used for the intended inference distribution. Record processed actual-decode p when generating new data, then let the existing attachment code install it. Do not add epsilon to zero-probability sampled labels or shift arrays: neither restores the original probability law. Existing campaign records lack the original probabilities and need matched regeneration/replay if strict on-policy status is required.

The following concrete minimal producer/consumer diff uses the existing decode helper. Packaging must make `dspine.decode_probs` available in each generation image before rollout. The schema extension preserves strict validation while allowing only the two documented optional fields. The extraction guard makes the strict contract explicit; it would intentionally reject the current legacy train records, so introduce it only for a newly named strict corpus/run, not by hot-patching the active run.

```diff
--- a/datagen/client/generate.py
+++ b/datagen/client/generate.py
@@
+from dspine.decode_probs import processed_decode_probs
@@
                     payload = {"input_ids": ids, "sampling_params": dict(SAMPLING, max_new_tokens=max_new,
                                stop_token_ids=EOS, skip_special_tokens=False, no_stop_trim=True), "stream": False}
+                    payload.update(return_logprob=True, top_logprobs_num=64, logprob_start_len=-1)
@@
                                       "gen_seconds": time.monotonic() - t0, "server": args.server}
+                            di, dp = processed_decode_probs(meta["output_top_logprobs"], outputs)
+                            record.update(target_decode_ids=di, target_decode_probs=dp)
                             validate_record(record)
--- a/datagen/common.py
+++ b/datagen/common.py
@@
 def validate_record(record):
-    if set(record) != RECORD_KEYS:
+    optional = {"target_decode_ids", "target_decode_probs"}
+    if set(record) not in (RECORD_KEYS, RECORD_KEYS | optional):
         raise ValueError(f"record keys differ: {set(record) ^ RECORD_KEYS}")
+    if "target_decode_ids" in record:
+        import math
+        di, dp = record["target_decode_ids"], record["target_decode_probs"]
+        if len(di) != len(record["output_ids"]) or len(dp) != len(di):
+            raise ValueError("decode probability length mismatch")
+        for token, ids, probs in zip(record["output_ids"], di, dp):
+            if (len(ids) != len(probs) or not ids or len(set(ids)) != len(ids)
+                    or any(type(i) is not int or not 0 <= i < 248320 for i in ids)
+                    or any(not math.isfinite(p) or p < 0 for p in probs)
+                    or abs(sum(probs) - 1) > 1e-6
+                    or sum(p for i, p in zip(ids, probs) if i == token) <= 0):
+                raise ValueError("invalid on-policy decode distribution")
--- a/dspine/extract_features.py
+++ b/dspine/extract_features.py
@@
 def request(url, row, root, destination=None):
+    if row["sampling"]["temperature"] > 0 and "target_decode_ids" not in row:
+        raise ValueError("strict on-policy capture requires actual decode probabilities")
     ids = row["prompt_ids"] + row["output_ids"]
```

This is a proposed integration change, not a tested patch. It must retain the helper's fail-closed behavior if raw top-64 support cannot resolve cutoff ties. The rollout also needs a new immutable data-contract/run identifier; mixing strict and legacy provenance would hide the distinction again.

### B. Make checkpoint selection include body offsets

Generate a **new** immutable held-out actual-decode cohort with the existing `generate_eval` option `--max-new-tokens 4096`, same prompt IDs, actual decode probabilities, and the production numerical runtime. Capture it under a separate root. Keep the old 64-token cohort for cheap opening diagnostics. Once the new root exists and passes the support check, the concrete config change is:

```diff
--- a/dspine/configs/qwen38-flash-next-full.json
+++ b/dspine/configs/qwen38-flash-next-full.json
@@
-  "eval_sequences": 18,
+  "eval_sequences": 36,
@@
-  "eval_features": "./work/features/pilot-eval/T1/shards",
+  "eval_features": "./work/features/body-eval-v1/T1/shards",
@@
-  "eval_generations": "./work/runs/pilot/eval-generations/T1",
+  "eval_generations": "./work/runs/body-eval-v1/T1",
```

No data under those proposed paths has been created. This belongs in a new immutable run configuration or explicit evaluation-only override, not an edit to an active frozen run. Use fixed-prefix windows at offsets 0, 256, 1,024 and 4,096 where available, report censored tails, and pool uncertainty by prompt rather than treating repeated rounds as independent prompts. For the intentional paper deviation, no production fix is proposed: retain production thinking and length, and add a separately named thinking-disabled, <=3,072-token comparison only if paper parity is needed.

## Decisive GPU tests for unresolved impact, <=20 minutes each

These are proposed for a future approved free GPU slot. They were not run and require no production changes.

1. **Matched-target probability/short-run A/B, 20-minute cap.** Reuse one isolated production-pinned target load. Generate 18 held-out continuations of 256 tokens with raw top-64 decode logprobs, preserving the selected prompts and sampling. Capture their teacher-prefill p and features. Report per-layer feature cosine/relative L2, actual versus prefill p L1, support failures, and q/p loss differences. On a disposable checkpoint copy, alternate 100–200 fixed batches between actual-p and prefill-p branches with identical labels, anchors and RNG; evaluate both on the same actual-p cohort. Charge model startup to the cap and reduce updates if needed. This determines whether the confirmed mismatch has material optimization/acceptance impact; it does not retrofit the original campaign's probabilities.
2. **Live HC mean and FP8 parity, 15-minute cap.** With the isolated target, record all four canonical streams immediately after the five selected layers on one short prefix and one >W prefix, in prefill and normal decode/verify modes. Independently average streams, compare to the capture and serving overlays before FP8, and compare drafter q with BF16 means versus their FP8 round trip at identical anchors. Require stream/layer/position agreement; report kernel-induced drift separately from quantization. This settles the raw-tensor numerical limitation of finding 3. If a mismatch is found, the exact fix must be derived from the differing hook/kernel path; there is no justified source edit from current CPU evidence.
3. **Body validation, 15-minute cap after a target slot is available.** Four held-out 2,048-token actual-decode trajectories plus packed b6/b8 checkpoint evaluation at fixed offsets 0/256/1,024. Report thinking-versus-post-thinking and offset-specific tau against the existing 64-token metric. This tests the representativeness risk without regenerating an entire evaluation set first.

No changes are proposed for the passing index, mask, joint-sampler, layer-ID, HC-mean formula, dequantization, or absolute-position code.
