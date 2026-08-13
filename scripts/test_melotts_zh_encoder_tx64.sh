#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL="${1:-${REPO_ROOT}/../models/melotts_zh_encoder_tx64.onnx}"
OUTPUT="${2:-${REPO_ROOT}/build/test/models/melotts_zh_encoder_tx64.tflite}"

if [[ ! -f "${MODEL}" ]]; then
  echo "MeloTTS Chinese encoder model not found: ${MODEL}" >&2
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
  --max-operators 655 \
  --max-constant-tensors 181 \
  --forbid-op CUSTOM \
  --forbid-op Flex

for SEED_AND_STDDEV in 20260806:0.5 20260807:2.0; do
  SEED="${SEED_AND_STDDEV%%:*}"
  STDDEV="${SEED_AND_STDDEV##*:}"
  TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_CMD[@]}" \
    "${REPO_ROOT}/test/e2e/run_similarity.py" \
    --onnx "${MODEL}" \
    --tflite "${OUTPUT}" \
    --seed "${SEED}" \
    --stddev "${STDDEV}" \
    --num-threads 8 \
    --integer-input-range x:0:112 \
    --integer-input-range tone:0:11 \
    --integer-input-range language:0:4 \
    --integer-input-range sid:0:256 \
    --min-cosine 0.9999 \
    --max-relative-euclidean 0.001
done
