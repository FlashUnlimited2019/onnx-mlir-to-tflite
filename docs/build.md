<!-- SPDX-License-Identifier: Apache-2.0 -->

# Building ONNX-MLIR TFLite

This document describes how to build the complete ONNX-to-TFLite toolchain
from source on Linux x86-64.

## What must be built

An end-to-end build contains three parts:

1. LLVM/MLIR, used to build ONNX-MLIR and the ONNX-to-TFL conversion pass;
2. this ONNX-MLIR repository, which provides `onnx-mlir`, `onnx-mlir-opt`, and
   the `onnx-to-tflite` driver;
3. TensorFlow/LiteRT, which provides `litert-opt` and
   `flatbuffer_translate`.

TensorFlow is therefore required for the complete conversion pipeline. It is
not linked into the ONNX-MLIR executables: the two toolchains communicate by
passing textual TFL Dialect MLIR between separate processes. Building only
this repository is sufficient to produce ONNX Dialect and TFL Dialect MLIR,
but it is not sufficient to export a TFLite FlatBuffer.

The TensorFlow tools have separate roles:

- `litert-opt` optimizes TFL Dialect MLIR. This stage is enabled by default but
  can be skipped with `--no-optimize-tfl`.
- `flatbuffer_translate` validates TFL Dialect MLIR and exports the final
  TFLite FlatBuffer. This tool is required.

## Requirements

The bootstrap script currently targets Linux x86-64. It requires:

- Git, CMake, curl, and a C++17 compiler;
- Python 3 and pip;
- network access to clone LLVM and TensorFlow and download Bazel dependencies;
- approximately 60 GB of free disk space;
- at least 16 GB of RAM, with more memory recommended for parallel builds.

The script installs pinned Ninja and Bazel executables inside the workspace;
they do not need to be installed globally.

## Python environment

Use an isolated Python environment. For example, this workspace can use the
existing Conda environment named `onnx`:

```bash
conda activate onnx
python -m pip install -r requirements.txt onnxruntime tensorflow
```

The `onnxruntime` and `tensorflow` Python packages are used by the numerical
smoke test. They are not used to perform the compiler conversion itself.

## Complete bootstrap build

From the repository root, run:

```bash
PYTHON_BIN="$(command -v python)" \
BUILD_JOBS=8 \
./scripts/bootstrap_and_build.sh
```

The script performs the following steps:

1. initializes this repository's Git submodules;
2. clones the pinned LLVM source revision when it is not already present;
3. builds LLVM, Clang, and MLIR with CMake and Ninja;
4. clones the pinned TensorFlow source revision when it is not already
   present;
5. builds `litert-opt` and `flatbuffer_translate` with Bazel;
6. builds the ONNX-MLIR importer, conversion pass, and conversion driver;
7. converts and validates a small model as an end-to-end smoke test.

By default, all downloaded sources and generated build state stay inside the
repository and are ignored by Git:

```text
onnx-mlir/
├── .bazel-output/              TensorFlow Bazel build state
├── .deps/                      pinned Ninja and Bazel executables
├── build/                      ONNX-MLIR build tree
├── llvm-project/               LLVM source and build tree
└── tensorflow/                 TensorFlow source and Bazel outputs
```

Keeping these directories below the repository root prevents the bootstrap
from discovering, modifying, or sharing an unrelated LLVM or TensorFlow
checkout in the parent directory.

The principal build products are:

```text
build/bin/onnx-mlir
build/bin/onnx-mlir-opt
build/bin/onnx-to-tflite
tensorflow/bazel-bin/tensorflow/compiler/mlir/lite/litert-opt
tensorflow/bazel-bin/tensorflow/compiler/mlir/lite/flatbuffer_translate
```

## Build configuration

The following environment variables can override the defaults:

| Variable | Purpose |
| --- | --- |
| `BUILD_JOBS` | Parallel LLVM, TensorFlow, and ONNX-MLIR build jobs |
| `PYTHON_BIN` | Python interpreter used for configuration and tests |
| `LLVM_ROOT` | LLVM source and build-tree location |
| `TENSORFLOW_ROOT` | TensorFlow source and build-output location |
| `BAZEL_OUTPUT_ROOT` | TensorFlow Bazel state and action-cache location |

For a memory-constrained machine, lower `BUILD_JOBS`. Existing source and
build trees are reused, so rerunning the script does not normally start every
step from an empty workspace.

## Reusing existing TensorFlow tools

If compatible `litert-opt` and `flatbuffer_translate` executables have already
been built elsewhere, the converter can use them without linking TensorFlow
into ONNX-MLIR:

```bash
./build/bin/onnx-to-tflite \
  /path/to/model.onnx \
  -o /path/to/model.tflite \
  --litert-opt /path/to/litert-opt \
  --flatbuffer-translate /path/to/flatbuffer_translate
```

Both tools must understand the TFL Dialect syntax emitted by this project.
Using the TensorFlow revision pinned by `scripts/bootstrap_and_build.sh` is the
supported configuration. If TFL optimization is disabled, `litert-opt` is not
required, but `flatbuffer_translate` remains mandatory.

## Verify the build

Run the MLIR regression tests with:

```bash
llvm-project/build/bin/llvm-lit \
  -sv build/test/mlir/conversion/onnx_to_tfl
```

Run a conversion and numerical comparison with:

```bash
./build/bin/onnx-to-tflite \
  test/models/mlp.onnx \
  -o build/test/models/mlp.tflite

python test/e2e/run_similarity.py \
  --onnx test/models/mlp.onnx \
  --tflite build/test/models/mlp.tflite
```
