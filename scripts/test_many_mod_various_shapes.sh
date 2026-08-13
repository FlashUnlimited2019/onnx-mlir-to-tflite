#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/many_mod_various_shapes.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/many_mod_various_shapes.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "Mod various-shapes model not found: ${MODEL}" >&2
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
    --verify-each
fi
TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" "${OUTPUT}" \
  --max-operators 350 \
  --max-constant-tensors 105 \
  --max-tensor-rank 5 \
  --forbid-op CUSTOM \
  --forbid-op Flex

# The fixture constructs two runtime i64 divisors as round(5*tanh(x)) + 3.
# Wide zero-centered random inputs can therefore create an invalid zero
# divisor. These deterministic distributions exercise distinct ranges while
# keeping every integer Mod operation in its defined input domain.
for CASE in 20260807:0.0:0.1 20260808:1.0:0.2; do
  IFS=: read -r SEED MEAN STDDEV <<<"${CASE}"
  TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${MODEL}" \
    --tflite "${OUTPUT}" \
    --seed "${SEED}" \
    --mean "${MEAN}" \
    --stddev "${STDDEV}" \
    --num-threads 4 \
    --min-cosine 0.99999 \
    --max-relative-euclidean 0.0001
done
