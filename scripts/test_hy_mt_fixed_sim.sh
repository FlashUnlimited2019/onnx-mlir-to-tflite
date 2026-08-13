#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/../models/hy_mt_fixed_sim"
MODEL="${1:-${MODEL_DIR}/hy_mt_fixed_sim.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/hy_mt_fixed_sim.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "hy_mt fixed model not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}.data" ]]; then
  echo "hy_mt external data file not found: ${MODEL}.data" >&2
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
  --max-operators 2577 \
  --max-constant-tensors 319 \
  --max-tensor-rank 5 \
  --forbid-op CUSTOM \
  --forbid-op Flex

for CASE_SPEC in "20260810 1 2" "20260811 0 2"; do
  read -r SEED MASK_LOW MASK_HIGH <<<"${CASE_SPEC}"
  TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${MODEL}" \
    --tflite "${OUTPUT}" \
    --seed "${SEED}" \
    --integer-input-range input_ids:0:120818 \
    --integer-input-range attention_mask:"${MASK_LOW}":"${MASK_HIGH}" \
    --num-threads 8 \
    --min-cosine 0.999999 \
    --max-relative-euclidean 0.0001
done
