# DSpine for Qwen3.8 Flash Next: technical spec (v1)

This is the single source of truth for the model, the data contracts and the runtime contract. The training
reference implementation (`dspine/`) and the SGLang integration (`serving/`) must both follow it, and
must agree numerically: the same weights and inputs give the same candidates and proposal probabilities.
Change this file and say so in your report before you diverge from it.

Primary sources are the [DSpine paper](https://arxiv.org/abs/2609.36173), [Tencent checkpoint](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16), and [target model](https://huggingface.co/RedHatAI/Qwen3.8-Flash-Next-NVFP4). Their text, files and production runtime are not bundled. This document is the historical experiment contract; see findings (`findings.md`, retained internal evidence) for corrections and final results.

## 1. Target model facts

- `RedHatAI/Qwen3.8-Flash-Next-NVFP4` @ `c8f2fb1b9869f686b214782036123b10ff96d14a` (public, ungated, ~186 GB:
  NVFP4 W4A4 routed experts, BF16 dense/attention/MTP, 102.4 GB BF16 PLE n-gram table). Arch `qwen4_exp`:
  48 layers (3 Gated-DeltaNet linear-attention : 1 QSA sparse-attention, QSA at layers 3,7,...,47), hidden 2560,
  hyper-connections `hc_count=4` (4 residual streams of 2560), MoE 512 experts top-10 (+ shared expert),
  `moe_intermediate_size=640`, vocab 248320 (real tokens 0..248076, ids 248077..248319 unused padding),
  untied LM head, native 1-layer MTP. Chat template: `reasoning_effort` in {xhigh (default), medium, low};
  `<think>`=248068, `</think>`=248069, `<tool_call>`=248058, `<|im_end|>`=248046, eos {248046, 248044}.
- Production sampling (generation_config): temperature 1.0, top_p 0.95, top_k 20. **Every acceptance
  number we report is at this setting unless labelled otherwise.**
- On the workstation the production server is SGLang 0.5.21 (image `sha256:ed0946e2...`) with private overlays and
  `qwenfast` SM120 decode kernels: native NEXTN MTP, 3 steps / 4 verify tokens, exact rejection sampling,
  FP8 KV, BF16 SSM, PLE in pinned host RAM, 262144 context, 2 concurrent requests. Currently ~111 decode
  steps/s and ~250 tok/s at bs=1 (`./work/ai/qwen-speed-20261010/REPORT.md`). Its custom decode kernels cover
  **1..8 rows**; more verify rows fall back to slower prefill paths. The QSA pending ring needs **n + 3 slots**
  for n verify rows (it includes the prior partial compression group). Production allocates 16, so n >= 14 needs a
  wider ring (serving agent, 2026-10-16).
- Measured target verify cost (production kernels, ctx 2K, bs1, CUDA graphs, LM head included; serving agent
  2026-10-16): n1 5.43 ms, n2 6.68, n3 6.98, n4 7.70, n6 9.11, n8 10.10, **n9 13.84** (cliff: the custom kernels stop at
  8 rows), n12 15.28, n13 15.69; bs2 n8 16.26. **Serving target: b = 6..8** unless the kernels are extended past 8 rows.
  Training stays at block 16 (block-causal, so it is still a valid base for longer blocks later), but acceptance at
  positions 1..7 is what pays: select checkpoints by offline tau at b = 6 and b = 8. Public SGLang v0.5.21 and vLLM
  also support `qwen4_exp`, PLE offload included.

## 2. Draft architecture (DSpine-QF v1)

Per the paper (Section 4, Appendix A) plus the tencent checkpoint layout, adapted to this target.

| item | value |
|---|---|
| draft layers L | 6 (Qwen3-style decoder layers; configurable) |
| hidden | 2560 (must equal target hidden: shares frozen target embedding + LM head) |
| attention | 32 q heads x 128, 8 kv heads x 128, q/k RMSNorm, RoPE theta 1e7 full head dim, no bias |
| MLP | SwiGLU, intermediate **10240** (changed from 9728 on 2026-10-16 so the qwenfast w8m INT8 decode GEMM needs no padding; serving is adding K=4096/10240 instantiations) |
| target feature layers | `[1, 12, 23, 34, 45]` (0-indexed layer outputs). Each feature is the **mean of the 4 HC streams** of the residual after that layer, so 5 x 2560 = 12800 per token |
| context fusion | `fc: Linear(12800 -> 2560, no bias)` then `hidden_norm` (RMSNorm). The fused context state gives each draft layer's context K/V through that layer's own k_proj/v_proj (DFlash style) |
| block | B = 16 positions: anchor + 15 masked positions. **Train at 16**; serve with any prefix length b <= 16 (block-causal, so the first b positions do not depend on later ones) |
| mask input | learned `mask_embedding` vector [2560] (init: mean of target input-embedding rows) |
| attention mask | **block-causal**: block position t attends to the context window and block positions 0..t |
| context window | sliding window W = 4096 most recent context tokens before the anchor (train and serve identically; configurable) |
| transfer space | r = 1024. C = PCA-whitened, re-normalized L2-normalized LM-head rows (Appendix A formula), fixed buffer [248320, 1024] |
| message dim | a = 512 |
| candidates K | 16 |
| draft vocab | full 248320 for v1 training. Serving may restrict proposals to a hot set (for example the production 97,800-id hot map). This is still lossless, because rejection sampling corrects for it |

Injection after **every** layer l = 1..L (paper eq. 4-5), using pre-injection (layer output) states u_l:
- `z_{l,t} = s * normalize(R_l u_{l,t})`, with `R_l: 2560 -> 1024` per layer (init P_r^T in the 2560-dim LM-head
  space) and scale s = sqrt(r) = 32 (the paper's choice; tencent's config also has `kappa: 10.0`, whose
  meaning is unknown. Do not depend on it).
- A known token x (the anchor, or a decoded predecessor) sends `s * C_x` in place of z.
- `g = sigmoid(G_q z_{l,t} + G_p z_{l,t-1} + b)` (G_q, G_p: 1024 -> 512, shared across layers).
- `m = A z_{l,t-1}` (A: 1024 -> 512, shared).
- `h_{l,t} = u_{l,t} + RMS(u_{l,t}) * W_l (g * m)` (W_l: 512 -> 2560 per layer, **zero-init**; G_p zero-init, b = 0).
- All messages use pre-update features, so the update runs in parallel over positions. The anchor (t = 0) is not
  updated. Position 1's predecessor is the anchor, which always sends `s * C_{x_0}`.
- Tencent's tensor names, for reference: `selected_prefix_bridge.{codebook, early_read_weights[L-1], read_weight,
  feature_write_weights[L], gate.{weight,bias}=G_q,b, predecessor_gate.weight=G_p, prev.weight=A}`. Our
  loader must accept tencent's checkpoint (Qwen3-8B, L=5, hidden 4096) for validation.

Output: final RMSNorm, then the frozen target LM head gives logits; take the top-K candidates per position.

**Last-write refinement / transition cache** (Section 4.3): for position t and each candidate s in V_{t-1} (the
top-K of t-1), replace the predecessor feature in the **final-layer** injection with `s * C_s`, redo the final
write from cached u_{L,t}, apply the final norm, and score the K candidates of t: `T_t[i, j] = S_t(v_{t-1,i},
v_{t,j})`, a K x K table per transition. Position 1 has a known predecessor (the anchor), so its scores are the
backbone top-K logits.

**Proposal at T > 0 (production):** sample x_1 from softmax over V_1 of the backbone logits. For t >= 2, sample
x_t from `softmax_j T_t[idx(x_{t-1}), j]` over V_t. The proposal probability q_t is exactly that K-way softmax (0
outside V_t), and verification is exact speculative rejection sampling against the **processed** target
distribution p_t (after temperature, top-k and top-p), with the residual `(p - q)+` on the first rejection.
At T = 0, take the argmax at each step instead.

## 3. Training objective

Following paper eq. 6-8 and Appendix A:
- `L_dist = sum_i w_i [ -alpha log q_i(y_i) + (1-alpha) ||q_i - p_i||_1 ]`, with alpha = 0.1 and w_i = exp(-(t_i-1)/gamma), gamma = 7.
  - p_i = the **processed** target distribution at production sampling (T1, top_k 20, top_p 0.95), from
    stored target top-20 (ids, probs), so ||q - p||_1 = 2(1 - acceptance probability). y_i = the actual
    on-policy token.
  - Applied to (a) the backbone output over the full vocabulary and (b) the last-write-refinement output with
    the **correct** predecessor y_{t-1}, restricted to the K candidates of position t (CE only where y_t is in V_t; L1
    only where the candidates hold > 1e-4 of the target mass, with the target renormalized to the candidates). Equal weight.
- `L_emb = sum_l sum_i w_i [1 - cos(f_l(h_{l,i}), C_{y_i})]`, read from **post**-injection states.
- `L = L_dist + lambda L_emb`, with lambda = beta_t * stopgrad(CE_backbone) and beta rising linearly 0 -> 0.5 over the warmup.
- Correct-predecessor curriculum: per block, with probability p, layers 1..ceil(L/2) use `s * C_{y_{t-1}}` as the
  predecessor input (receiver features stay model-derived). p = 0.5 for the first 1/6 of training, decaying
  linearly to 0 at 1/3.
- AdamW without weight decay, peak LR 6e-4, 4% linear warmup, cosine to 0, grad clip 1.0. Anchors are sampled within
  assistant-generated tokens only.

## 4. Data contracts

### 4.1 Prompt shard (`prompts/out/shards/shard-XXXXX.jsonl.zst`)
One JSON object per line:
```json
{"id": "sha1-of-canonical-prompt", "domain": "code|agentic|math|stem|chat|writing|multilingual|structured|longctx|...",
 "subdomain": "free text", "lang": "en|zh|...", "source": "hf-dataset-id[:split]",
 "messages": [{"role": "system|user|assistant|tool", "content": "...", "tool_calls": [...]}],
 "tools": [ ...OpenAI-style tool schemas or null... ],
 "reasoning_effort": "xhigh|medium|low",
 "max_new_tokens": 16384}
```
The last message is the one the target answers (user or tool). Earlier assistant turns are context.

### 4.2 Generation record (`datagen/out/<platform>/<shard>/part-*.jsonl.zst`)
```json
{"id": "...", "shard": "...", "platform": "vega|vast-<id>|modal-<id>", "model_rev": "c8f2fb1b...",
 "prompt_ids": [...], "output_ids": [...], "finish_reason": "stop|length|abort", "sampling": {"temperature":1.0,"top_p":0.95,"top_k":20},
 "reasoning_effort": "...", "domain": "...", "gen_seconds": 12.3, "server": "sglang-0.5.21 ..."}
```
`prompt_ids` must be exactly the tokens the server consumed (chat template applied client-side with the model's own
`chat_template.jinja` and tokenizer, sent to the native `/generate` endpoint as `input_ids`). Sampling is the
production setting. No speculative decoding settings may change the output distribution.

### 4.3 Training features (the workstation only, produced by `dspine/` feature extraction)
Per sequence, over the window `[max(0, resp_start - W), end)`: the 12800-dim HC-mean features (FP8 e4m3 with a
per-token, per-layer scale, or BF16), plus processed target top-20 ids and probs at every response position. Rolling
windows on disk (budget 450 GB at a time) or online streaming. Never upload features off the workstation.

### 4.4 Exclusions
Training data loaders and the merge step must drop every generation record whose `id` appears in
`./work/prompts/EXCLUDE_IDS.json` (repo copy: `prompts/EXCLUDE_IDS.json`). Revision 2 of shards
00002 and 00031 removed benchmark overlaps found inside decoded tool arguments, so records generated from the old versions
must not be used.

## 5. Evaluation contract

- Offline: simulated acceptance tau (tokens per verify round, including the bonus) at b in {4,6,8,12,16}, T=1 processed
  sampling (Monte Carlo with the exact q / p acceptance rule) and T=0, on a held-out set of on-policy
  generations per domain. This is fast and used for checkpoint selection.
- Online (the deciding number): production-identical bench on the workstation (`bench.py` style: 4 profiles code / agent /
  prose / chinese x 4096 tokens, xhigh, T1/top_p .95/top_k 20, bs=1; then bs=2), DSpine vs. the current production MTP
  runtime, same image/kernels/context/KV/capacity. Report tok/s, steps/s and tau. Quality must be unchanged:
  speculative decoding is exact. Check it with GSM8K (thinking off, full) and the agent bench, compared within noise
  against MTP production.
- Never tune on the final test prompts. Hold out the eval prompts from training data (decontaminate: GSM8K test,
  MATH-500, HumanEval, MBPP test, LiveCodeBench, MT-Bench, Arena-Hard, and the the workstation bench prompts).
