#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

set -euo pipefail

readonly LLVM_COMMIT=1053047a4be7d1fece3adaf5e7597f838058c947
readonly TENSORFLOW_COMMIT=eef0088899ce97a533463fade6b54b3140f515e5
readonly BAZEL_VERSION=7.7.0
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DEPS_ROOT="${REPO_ROOT}/.deps"
readonly CMAKE_DEPS_ROOT="${DEPS_ROOT}/cmake"
readonly NINJA_BIN="${DEPS_ROOT}/bin/ninja"
readonly BAZEL_BIN="${DEPS_ROOT}/bin/bazel"
readonly LLVM_ROOT="${LLVM_ROOT:-${REPO_ROOT}/llvm-project}"
readonly TENSORFLOW_ROOT="${TENSORFLOW_ROOT:-${REPO_ROOT}/tensorflow}"
readonly BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT:-${REPO_ROOT}/.bazel-output}"
readonly BUILD_ROOT="${REPO_ROOT}/build"
readonly PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

# Bazel's Java downloader rejects socks5h:// proxy URLs. Translate that common
# environment form into JVM SOCKS properties while removing proxy variables
# from the server process. HTTP(S) proxy URLs are left untouched.
BAZEL_ENV=(env)
BAZEL_STARTUP_ARGS=()
BAZEL_PROXY_URL="${HTTPS_PROXY:-${ALL_PROXY:-}}"
if [[ "${BAZEL_PROXY_URL}" =~ ^socks5h?://([^:/]+):([0-9]+)$ ]]; then
  BAZEL_ENV+=(
    -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy
    -u HTTPS_PROXY -u https_proxy
  )
  BAZEL_STARTUP_ARGS+=(
    "--host_jvm_args=-DsocksProxyHost=${BASH_REMATCH[1]}"
    "--host_jvm_args=-DsocksProxyPort=${BASH_REMATCH[2]}"
  )
fi

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

for tool in git cmake curl c++; do
  if ! command -v "${tool}" >/dev/null; then
    echo "error: required tool not found: ${tool}" >&2
    exit 1
  fi
done

run mkdir -p "${DEPS_ROOT}" "${CMAKE_DEPS_ROOT}"
if [[ ! -x "${NINJA_BIN}" ]]; then
  run "${PYTHON_BIN}" -m pip install --prefix "${DEPS_ROOT}" ninja==1.13.0
fi
if [[ ! -x "${BAZEL_BIN}" ]]; then
  run curl -fL \
    "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64" \
    -o "${BAZEL_BIN}"
  run chmod +x "${BAZEL_BIN}"
fi

run git -C "${REPO_ROOT}" submodule update --init --recursive

if [[ ! -d "${LLVM_ROOT}/.git" ]]; then
  run git clone --filter=blob:none --no-checkout \
    https://github.com/llvm/llvm-project.git "${LLVM_ROOT}"
fi
run git -C "${LLVM_ROOT}" checkout "${LLVM_COMMIT}"
run cmake -G Ninja -S "${LLVM_ROOT}/llvm" -B "${LLVM_ROOT}/build" \
  -DLLVM_ENABLE_PROJECTS=mlir\;clang \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DPython3_EXECUTABLE="${PYTHON_BIN}" \
  -DCMAKE_MAKE_PROGRAM="${NINJA_BIN}"
run "${NINJA_BIN}" -C "${LLVM_ROOT}/build" -j"${BUILD_JOBS:-24}"

if [[ ! -d "${TENSORFLOW_ROOT}/.git" ]]; then
  run git clone --filter=blob:none \
    https://github.com/tensorflow/tensorflow.git "${TENSORFLOW_ROOT}"
fi
run git -C "${TENSORFLOW_ROOT}" checkout "${TENSORFLOW_COMMIT}"
run cmake -E chdir "${TENSORFLOW_ROOT}" env \
  PATH="${DEPS_ROOT}/bin:${PATH}" \
  PYTHON_BIN_PATH="${PYTHON_BIN}" \
  USE_DEFAULT_PYTHON_LIB_PATH=1 \
  TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_CLANG=0 TF_NEED_MPI=0 \
  TF_SET_ANDROID_WORKSPACE=0 CC_OPT_FLAGS=-Wno-sign-compare \
  "${PYTHON_BIN}" configure.py
run cmake -E chdir "${TENSORFLOW_ROOT}" "${BAZEL_ENV[@]}" \
  "${BAZEL_BIN}" "${BAZEL_STARTUP_ARGS[@]}" \
  --output_base="${BAZEL_OUTPUT_ROOT}" build \
  --jobs="${BUILD_JOBS:-24}" --local_resources=memory=65536 --config=opt \
  //tensorflow/compiler/mlir/lite:flatbuffer_translate \
  //tensorflow/compiler/mlir/lite:litert-opt

# Discard migrated FetchContent source overrides so an existing CMake cache
# cannot keep referring to dependency trees from another checkout or server.
run cmake \
  -U FETCHCONTENT_SOURCE_DIR_ABSL \
  -U FETCHCONTENT_SOURCE_DIR_PROTOBUF \
  -U absl_SOURCE_DIR \
  -U protobuf_SOURCE_DIR \
  -U utf8_range_SOURCE_DIR \
  -G Ninja -S "${REPO_ROOT}" -B "${BUILD_ROOT}" \
  -DFETCHCONTENT_BASE_DIR="${CMAKE_DEPS_ROOT}" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
  -DMLIR_DIR="${LLVM_ROOT}/build/lib/cmake/mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="${NINJA_BIN}" \
  -DONNX_MLIR_BUILD_STANDALONE=ON \
  -DONNX_MLIR_ENABLE_JAVA=OFF \
  -DONNX_MLIR_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE="${PYTHON_BIN}"
run "${NINJA_BIN}" -C "${BUILD_ROOT}" -j"${BUILD_JOBS:-24}" \
  onnx-mlir onnx-mlir-opt onnx-to-tflite

if [[ ! -e "${BUILD_ROOT}/bin" ]]; then
  run cmake -E create_symlink Release/bin "${BUILD_ROOT}/bin"
fi
run "${PYTHON_BIN}" "${REPO_ROOT}/utils/generate_test_models.py"
run "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/compiler/mlir/lite/flatbuffer_translate" \
  --help
run "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/compiler/mlir/lite/litert-opt" \
  --help
run "${BUILD_ROOT}/bin/onnx-to-tflite" \
  "${REPO_ROOT}/test/models/mlp.onnx" \
  -o "${BUILD_ROOT}/test/models/mlp.tflite" \
  --flatbuffer-translate \
  "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/compiler/mlir/lite/flatbuffer_translate" \
  --litert-opt \
  "${TENSORFLOW_ROOT}/bazel-bin/tensorflow/compiler/mlir/lite/litert-opt"
run env TF_CPP_MIN_LOG_LEVEL=3 "${PYTHON_BIN}" \
  "${REPO_ROOT}/utils/inspect_tflite.py" \
  "${BUILD_ROOT}/test/models/mlp.tflite" \
  --max-operators 3 \
  --forbid-op BATCH_MATMUL \
  --forbid-op ADD \
  --forbid-op RELU
run "${PYTHON_BIN}" "${REPO_ROOT}/test/e2e/run_compare.py" \
  --onnx "${REPO_ROOT}/test/models/mlp.onnx" \
  --tflite "${BUILD_ROOT}/test/models/mlp.tflite"
