#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec "${DSPINE_PYTHON:-python3}" -m pytest dspine/tests serving/tests serving/hybrid/test_exactness.py -q "$@"
