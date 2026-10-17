#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'usage: launch_train.sh CONFIG [trainer arguments...]' >&2
  exit 2
fi
exec "${DSPINE_PYTHON:-python3}" -m dspine.train --config "$@"
