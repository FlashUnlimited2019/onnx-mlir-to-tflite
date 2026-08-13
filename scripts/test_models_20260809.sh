#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${1:-${REPO_ROOT}/../models/test_models_20260809}"
OUTPUT_DIR="${2:-${REPO_ROOT}/build/test/models}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "constructed-model directory not found: ${MODEL_DIR}" >&2
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

MODELS=(
  common_ops_siso_sim
  diverse_rank_common_ops_sim
  multirank_float_operator_coverage_sim
)

mkdir -p "${OUTPUT_DIR}"
for NAME in "${MODELS[@]}"; do
  MODEL="${MODEL_DIR}/${NAME}.onnx"
  OUTPUT="${OUTPUT_DIR}/${NAME}.tflite"
  if [[ ! -f "${MODEL}" ]]; then
    echo "constructed model not found: ${MODEL}" >&2
    exit 1
  fi
  "${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}" \
    --verify-each
  TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
    --max-operators 250 \
    --max-constant-tensors 100 \
    --max-tensor-rank 5 \
    --forbid-op CUSTOM \
    --forbid-op Flex
  TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${MODEL}" \
    --tflite "${OUTPUT}" \
    --seed 20260809 \
    --num-threads 4 \
    --min-cosine 0.9999 \
    --max-relative-euclidean 0.001
done
