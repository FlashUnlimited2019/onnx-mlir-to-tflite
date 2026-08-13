#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/many_hardswish_2_3_4_5d.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/many_hardswish_2_3_4_5d.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "HardSwish rank-cases model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/build/bin/onnx-to-tflite" ]]; then
  echo "onnx-to-tflite is not built; run ./scripts/bootstrap_and_build.sh" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n onnx python)
else
  PYTHON_CMD=(python)
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}" \
  --verify-each
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 63 \
  --max-constant-tensors 45 \
  --forbid-op CUSTOM \
  --forbid-op Flex
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260806 \
  --num-threads 8 \
  --min-cosine 0.99999 \
  --max-relative-euclidean 0.001
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260806 \
  --stddev 3 \
  --num-threads 8 \
  --min-cosine 0.99999 \
  --max-relative-euclidean 0.001
