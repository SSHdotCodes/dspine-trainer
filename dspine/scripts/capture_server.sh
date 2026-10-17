#!/usr/bin/env bash
set -euo pipefail
project=$(realpath "${DSPINE_PROJECT:-./work}")
code_root=${DSPINE_CODE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
model=${DSPINE_TARGET_PATH:-./models/target}
image=${DSPINE_CAPTURE_IMAGE:-lmsysorg/sglang:v0.5.21}
feature_root=${DSPINE_FEATURE_ROOT:-$project/features}
mkdir -p "$project/ref" "$feature_root/inbox" "$feature_root/shards" "$project/cache"
# CPU-only source export. Refuse unsupported upstream layouts before GPU startup.
docker run --rm --pull never --entrypoint cat "$image" /sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py > "$project/ref/qwen4_exp-original.py"
cd "$code_root"
mapfile -d '' -t runtime_extra < <("${DSPINE_PYTHON:-python3}" -m dspine.capture_runtime --project "$project" --runtime "${DSPINE_CAPTURE_RUNTIME:-base}" --unit "${DSPINE_RUNTIME_UNIT:-./config/target.service}")
test ${#runtime_extra[@]} -gt 0
# Exercise the exact installed joint sampler before writing any real features.
if [[ ${DSPINE_VERIFY_SAMPLER:-1} == 1 ]]; then
  docker run --rm --pull never --gpus all --ipc host \
    -e PYTHONPATH="$code_root:/sgl-workspace/sglang/python" \
    -v "$project:$project" -v "$code_root:$code_root:ro" -v "$project/cache:/root/.cache" \
    "$image" python -m dspine.sampling_probe --output "$project/runs/sampling-probe.json"
fi
capture_docker_extra=()
if [[ -n ${DSPINE_CAPTURE_CIDFILE:-} ]]; then
  capture_docker_extra+=(--cidfile "$DSPINE_CAPTURE_CIDFILE")
fi
exec docker run "${capture_docker_extra[@]}" --rm --pull never --gpus all --ipc host --shm-size 32g \
  --name dspine-feature-capture --label codex.project=dspine-qwen38 --label codex.task=feature-capture -p 127.0.0.1:30220:30220 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,pinned_max_round_threshold_mb:128 \
  -e DSPINE_CAPTURE_LAYER_IDS="${DSPINE_CAPTURE_LAYER_IDS:-1,12,23,34,45}" -e DSPINE_CONTEXT_WINDOW="${DSPINE_CONTEXT_WINDOW:-4096}" \
  -e DSPINE_CAPTURE_HEAD_CHUNK="${DSPINE_CAPTURE_HEAD_CHUNK:-256}" \
  -e DSPINE_CAPTURE_ENCODING="${DSPINE_CAPTURE_ENCODING:-fp8_e4m3fn}" \
  -e PYTHONPATH="$code_root:/sgl-workspace/sglang/python" -e DSPINE_CAPTURE_ROOT="$feature_root" -e DSPINE_FEATURE_BUDGET="${DSPINE_FEATURE_BUDGET:-450000000000}" \
  -e DSPINE_FEATURE_BUDGET_ROOT="${DSPINE_FEATURE_BUDGET_ROOT:-$project/features}" -e DSPINE_GLOBAL_FEATURE_BUDGET="${DSPINE_GLOBAL_FEATURE_BUDGET:-449000000000}" \
  -v "$project:$project" -v "$code_root:$code_root:ro" -v "$model:$model:ro" \
  "${runtime_extra[@]}" \
  -v "$project/cache:/root/.cache" \
  "$image" python -m sglang.launch_server --model-path "$model" --host 0.0.0.0 --port 30220 \
  --trust-remote-code --ple-offload-embedding --tp-size 1 \
  --model-loader-extra-config '{"enable_multithread_load":false}' \
  --quantization compressed-tensors --fp4-gemm-backend flashinfer_cutlass --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 --linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer \
  --page-size 64 --mamba-track-interval 64 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 12 --sampling-defaults model --skip-tokenizer-init \
  --mem-fraction-static 0.93 --context-length 262144 --max-total-tokens 262144 \
  --max-running-requests "${DSPINE_CAPTURE_REQUESTS:-1}" --chunked-prefill-size "${DSPINE_CAPTURE_CHUNK:-8192}" --disable-radix-cache --disable-cuda-graph
