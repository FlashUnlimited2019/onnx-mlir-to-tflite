#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/yolov8m_416.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/yolov8m_416.tflite}"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -f "${MODEL}" ]]; then
  echo "YOLOv8m model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/build/bin/onnx-to-tflite" ]]; then
  echo "onnx-to-tflite is not built; run ./scripts/bootstrap_and_build.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}"
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_BIN}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 321 \
  --max-constant-tensors 199
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_BIN}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260805 \
  --num-threads 8
