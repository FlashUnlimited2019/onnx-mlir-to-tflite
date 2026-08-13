#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_MODEL="${1:-${REPO_ROOT}/../models/conv_many_3_4_5d.onnx}"
OUTPUT_DIR="${2:-${REPO_ROOT}/build/test/conv_rank_decomposition}"

if [[ ! -f "${SOURCE_MODEL}" ]]; then
  echo "Conv rank test model not found: ${SOURCE_MODEL}" >&2
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

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_CMD[@]}" "${REPO_ROOT}/utils/generate_conv_rank_test_models.py" \
  --source-model "${SOURCE_MODEL}" \
  --output-dir "${OUTPUT_DIR}"

MODELS=(
  conv_many_supported
  conv3d_k1hw
  conv3d_kd11
  conv3d_kd1w
  conv3d_group2_k1hw
  conv3d_depthwise_kd11
)
for name in "${MODELS[@]}"; do
  "${REPO_ROOT}/build/bin/onnx-to-tflite" \
    "${OUTPUT_DIR}/${name}.onnx" \
    -o "${OUTPUT_DIR}/${name}.tflite" \
    --verify-each
  TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/utils/inspect_tflite.py" \
    "${OUTPUT_DIR}/${name}.tflite" \
    --forbid-op CONV_3D
  TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${OUTPUT_DIR}/${name}.onnx" \
    --tflite "${OUTPUT_DIR}/${name}.tflite" \
    --seed 20260805
done

"${REPO_ROOT}/build/bin/onnx-to-tflite" \
  "${SOURCE_MODEL}" \
  -o "${OUTPUT_DIR}/conv_many_full.tflite" \
  --verify-each
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" \
  "${OUTPUT_DIR}/conv_many_full.tflite"
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/test/e2e/run_similarity.py" \
  --onnx "${SOURCE_MODEL}" \
  --tflite "${OUTPUT_DIR}/conv_many_full.tflite" \
  --seed 20260805
