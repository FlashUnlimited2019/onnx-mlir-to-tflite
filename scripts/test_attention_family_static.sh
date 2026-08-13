#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/test_models_20260810/attention_family_static_sim.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/attention_family_static_sim.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "model not found: ${MODEL}" >&2
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
"${REPO_ROOT}/build/bin/onnx-to-tflite" "${MODEL}" -o "${OUTPUT}" \
  --verify-each

TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 250 \
  --max-constant-tensors 100 \
  --max-tensor-rank 4 \
  --forbid-op CUSTOM \
  --forbid-op Flex \
  --forbid-op WHILE \
  --forbid-op IF

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
ORT_MODEL="${TMP_DIR}/attention_family_ort_compat.onnx"
"${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/prepare_ort_attention_baseline.py" \
  "${MODEL}" "${ORT_MODEL}"

TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${ORT_MODEL}" \
  --tflite "${OUTPUT}" \
  --seed 20260810 \
  --num-threads 4 \
  --min-cosine 0.999999 \
  --max-relative-euclidean 0.00001
