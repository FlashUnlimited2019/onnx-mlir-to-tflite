#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/vgg16-7.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/vgg16-7.tflite}"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -f "${MODEL}" ]]; then
  echo "VGG16 model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/build/bin/onnx-to-tflite" ]]; then
  echo "onnx-to-tflite is not built; run ./scripts/bootstrap_and_build.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}" \
  --verify-each
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_BIN}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 23 \
  --max-constant-tensors 34 \
  --forbid-op BATCH_MATMUL \
  --forbid-op RELU
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_BIN}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260805 \
  --num-threads 8
