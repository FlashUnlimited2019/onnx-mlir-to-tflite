#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/Qwen3-VL-Embedding-2B-image-frontend.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/Qwen3-VL-Embedding-2B-image-frontend.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Qwen3-VL image frontend model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${REPO_ROOT}/build/bin/onnx-to-tflite" ]]; then
  echo "onnx-to-tflite is not built; run ./scripts/bootstrap_and_build.sh" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n onnx python)
else
  PYTHON_CMD=(python)
fi

mkdir -p "$(dirname "${OUTPUT}")"
if [[ "${REUSE_OUTPUT:-0}" != 1 || ! -f "${OUTPUT}" ]]; then
  "${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}" \
    --use-buffer-offset \
    --verify-each
fi
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 1650 \
  --max-constant-tensors 250 \
  --max-tensor-rank 4 \
  --forbid-op CONV_3D \
  --forbid-op CUSTOM \
  --forbid-op Flex

for SEED_AND_STDDEV in 20260807:0.5 20260808:2.0; do
  SEED="${SEED_AND_STDDEV%%:*}"
  STDDEV="${SEED_AND_STDDEV##*:}"
  TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${MODEL}" \
    --tflite "${OUTPUT}" \
    --seed "${SEED}" \
    --stddev "${STDDEV}" \
    --num-threads 8 \
    --min-cosine 0.999999 \
    --max-relative-euclidean 0.0001
done
