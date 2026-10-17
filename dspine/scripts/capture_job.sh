#!/usr/bin/env bash
# Bounded local capture job; launches only its own container.
set -euo pipefail
if [[ ${DSPINE_CAPTURE_BOUNDED:-0} != 1 ]]; then
  exec timeout --signal=TERM --kill-after=60s "${DSPINE_CAPTURE_JOB_SECONDS:-3300}" env DSPINE_CAPTURE_BOUNDED=1 bash "$0" "$@"
fi
project=$(realpath "${DSPINE_PROJECT:-./work}")
code_root=${DSPINE_CODE_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
train_tokens=${1:-5000000}
concurrency=${2:-1}
feature_root=${DSPINE_FEATURE_ROOT:-$project/features}
generations=${DSPINE_CAPTURE_GENERATIONS:-$project/gen}
target=${DSPINE_TARGET_PATH:-./models/target}
revision=${DSPINE_TARGET_REVISION:-c8f2fb1b9869f686b214782036123b10ff96d14a}
mkdir -p "$project/runs"
capture_state=${DSPINE_CAPTURE_STATE_DIR:-$(mktemp -d "$project/runs/capture-XXXXXX")}
mkdir -p "$capture_state"
export DSPINE_CAPTURE_CIDFILE="$capture_state/container.cid"
if [[ -f "$DSPINE_CAPTURE_CIDFILE" ]]; then
  previous_cid=$(cat "$DSPINE_CAPTURE_CIDFILE")
  if docker inspect "$previous_cid" >/dev/null 2>&1; then
    previous_state=$(docker inspect --format '{{.State.Running}} {{index .Config.Labels "codex.task"}}' "$previous_cid")
    if [[ $previous_state != 'false feature-capture' ]]; then
      echo "capture state still belongs to an active or unrecognized container" >&2
      exit 1
    fi
  fi
  rm -- "$DSPINE_CAPTURE_CIDFILE"
fi
server_pid=
cleanup() {
  if [[ -s "$DSPINE_CAPTURE_CIDFILE" ]]; then
    docker stop -t 30 "$(cat "$DSPINE_CAPTURE_CIDFILE")" >/dev/null 2>&1 || true
  fi
  if [[ -n $server_pid ]]; then wait "$server_pid" || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
capture_deadline=$("${DSPINE_PYTHON:-python3}" -c 'import os,time; print(time.monotonic()+int(os.environ.get("DSPINE_CAPTURE_JOB_SECONDS","3300"))-60)')
cd "$code_root"
bash dspine/scripts/capture_server.sh > "$capture_state/server.log" 2>&1 &
server_pid=$!
"${DSPINE_PYTHON:-python3}" -u - "$server_pid" <<'PY'
import os,sys,time,urllib.request
pid=int(sys.argv[1]); start=time.monotonic()
while time.monotonic()-start<900:
    try:
        with urllib.request.urlopen('http://127.0.0.1:30220/health',timeout=3) as reply:
            if reply.status==200:
                if not os.path.isfile(os.environ["DSPINE_CAPTURE_CIDFILE"]):
                    raise RuntimeError("port 30220 belongs to a server outside this capture job")
                print('capture server ready',flush=True); break
    except (OSError,urllib.error.URLError): pass
    try: os.kill(pid,0)
    except ProcessLookupError: raise RuntimeError('capture server exited during startup')
    time.sleep(3)
else: raise TimeoutError('capture server startup exceeded 900 seconds')
PY
"${DSPINE_PYTHON:-python3}" - "$capture_deadline" "$capture_state" <<'PY'
import json,os,sys,time
from pathlib import Path
seconds=int(os.environ.get("DSPINE_CAPTURE_JOB_SECONDS","3300"))
started=float(sys.argv[1])+60-seconds
(Path(sys.argv[2])/"timing.json").write_text(json.dumps({"startup_seconds":time.monotonic()-started})+"\n")
PY
if [[ -n ${DSPINE_HELDOUT_PROMPTS:-} ]]; then
  eval_root=${DSPINE_EVAL_ROOT:-$project/features/pilot-eval}
  eval_generations=${DSPINE_EVAL_GENERATIONS:-$project/runs/pilot/eval-generations}
  "${DSPINE_PYTHON:-python3}" -u -m dspine.generate_eval \
    --prompts "$DSPINE_HELDOUT_PROMPTS" --target "$target" --target-revision "$revision" \
    --output "$eval_generations" --count "${DSPINE_EVAL_COUNT:-64}" --max-new-tokens 64 \
    --record-decode-probs \
    > "$capture_state/eval-generation-metrics.jsonl"
  for temperature in T1 T0; do
    eval_minutes=$("${DSPINE_PYTHON:-python3}" -c 'import sys,time; print(max(0.,(float(sys.argv[1])-time.monotonic())/60))' "$capture_deadline")
    "${DSPINE_PYTHON:-python3}" -u -m dspine.extract_features \
      --generations "$eval_generations/$temperature" --root "$feature_root" \
      --destination "$eval_root/$temperature" --allow-greedy --concurrency 1 \
      --target-revision "$revision" --max-minutes "$eval_minutes" \
      > "$capture_state/eval-$temperature-capture.jsonl"
  done
fi
capture_minutes=$("${DSPINE_PYTHON:-python3}" -c 'import sys,time; print(max(0.,(float(sys.argv[1])-time.monotonic())/60))' "$capture_deadline")
"${DSPINE_PYTHON:-python3}" -u -m dspine.extract_features \
  --generations "$generations" --root "$feature_root" --concurrency "$concurrency" \
  --train-response-tokens "$train_tokens" --max-minutes "$capture_minutes" \
  --target-revision "$revision" \
  | tee "$capture_state/extraction-metrics.jsonl"
