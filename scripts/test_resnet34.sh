#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/resnet34-v2-7-224.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/resnet34-v2-7-224.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "ResNet model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/build/bin/onnx-to-tflite" ]]; then
  echo "onnx-to-tflite is not built; run ./scripts/bootstrap_and_build.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}"
TF_CPP_MIN_LOG_LEVEL=3 python "${REPO_ROOT}/utils/inspect_tflite.py" \
  "${OUTPUT}" \
  --max-operators 96 \
  --max-constant-tensors 99 \
  --forbid-op RESHAPE \
  --forbid-op RELU \
  --forbid-op BATCH_MATMUL
TF_CPP_MIN_LOG_LEVEL=3 python "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260804
