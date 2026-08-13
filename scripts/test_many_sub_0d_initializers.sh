#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/many_sub_with_0d_initializers.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/many_sub_with_0d_initializers.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Sub 0D-initializer model not found: ${MODEL}" >&2
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
  --max-operators 10 \
  --max-constant-tensors 8 \
  --forbid-op CUSTOM \
  --forbid-op Flex
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260806 \
  --stddev 0.5 \
  --num-threads 8 \
  --min-cosine 0.999999 \
  --max-relative-euclidean 0.000001
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260807 \
  --stddev 2.0 \
  --num-threads 8 \
  --min-cosine 0.999999 \
  --max-relative-euclidean 0.000001
