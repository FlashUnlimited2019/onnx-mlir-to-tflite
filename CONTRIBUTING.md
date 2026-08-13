<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to ONNX-MLIR to TFLite

Thank you for your interest in improving ONNX-MLIR to TFLite. This repository
is an independent downstream project based on ONNX-MLIR; it is not an official
ONNX or ONNX-MLIR release.

Changes specific to ONNX-to-TFL lowering, TFLite export, or this project's
tooling and documentation should be proposed here. Changes that apply generally
to ONNX-MLIR and do not depend on the TFLite pipeline may be better submitted to
the [upstream ONNX-MLIR project](https://github.com/onnx/onnx-mlir).

## Before contributing

- Search this repository's issues and pull requests for related work.
- Open an issue before undertaking a large change or changing a public
  interface.
- Keep contributions focused. Avoid mixing unrelated refactoring with a
  feature or bug fix.
- Do not commit generated build trees, large model weights, proprietary models,
  credentials, or machine-specific paths.

All participation is subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Build the project

Follow the [build guide](docs/build.md) for prerequisites and a complete
repository-local LLVM/MLIR, ONNX-MLIR, and TensorFlow/LiteRT build. The standard
bootstrap command is:

```bash
PYTHON_BIN="$(command -v python)" \
BUILD_JOBS=8 \
./scripts/bootstrap_and_build.sh
```

The script builds the required toolchains and runs an end-to-end smoke test.
Build products and downloaded toolchains are ignored by Git.

## Development workflow

1. Fork the repository and create a branch from the latest `main`.
2. Implement the smallest complete change that addresses the issue.
3. Add focused regression tests and, when appropriate, numerical validation.
4. Run the relevant format, build, and test commands locally.
5. Update documentation when support, behavior, layout, or limitations change.
6. Open a pull request describing the motivation, implementation, limitations,
   and validation performed.

Pull requests should remain reviewable and should not contain generated build
output or unrelated formatting changes.

## Adding or extending an operator

ONNX-to-TFL conversion code is located in
`src/Conversion/ONNXToTFL/`. Contributions to operator support should:

- use MLIR Dialect Conversion and fully legalize the supported ONNX operation;
- preserve ONNX type, shape, broadcasting, axis, padding, and layout semantics;
- emit builtin TFLite operations or compositions of builtin operations;
- avoid Select TF, Flex, and custom-operator fallbacks;
- diagnose unsupported types, ranks, shapes, attributes, and dynamic forms;
- add focused MLIR tests under `test/mlir/conversion/onnx_to_tfl/`;
- add end-to-end numerical coverage when the change affects runtime behavior;
- update the [operator support matrix](docs/operator-support.md);
- update the [layout policy](docs/layout.md) or
  [known limitations](docs/known-limitations.md) when applicable.

Prefer small, deterministically generated test fixtures. Do not add large model
files solely for regression coverage.

## Formatting

Python files are checked with Black:

```bash
python -m black --check --exclude third_party .
```

C and C++ files under `src/` are checked with clang-format 9 using the
repository's `.clang-format` configuration. Format changed source files before
submitting:

```bash
clang-format -i path/to/changed_file.cpp
```

Also check patches for whitespace errors:

```bash
git diff --check
```

Avoid applying formatters to unrelated files.

## Testing

At minimum, build the affected targets and run the ONNX-to-TFL MLIR regression
suite:

```bash
cmake --build build --parallel 8

llvm-project/build/bin/llvm-lit \
  -sv build/test/mlir/conversion/onnx_to_tfl
```

For behavior visible in a generated FlatBuffer, convert a representative static
model and compare it with ONNX Runtime:

```bash
./build/bin/onnx-to-tflite \
  test/models/simple_conv.onnx \
  -o /tmp/simple_conv.tflite

python test/e2e/run_similarity.py \
  --onnx test/models/simple_conv.onnx \
  --tflite /tmp/simple_conv.tflite
```

Report the commands and results in the pull request. More extensive testing may
be required for layout-sensitive, high-rank, recurrent, attention, or large
model changes.

## Documentation

Keep public documentation general and reproducible. Avoid embedding private
paths, unpublished model names, migration notes, or one-off benchmark results.
Document supported configurations in the operator support matrix rather than
claiming unrestricted support for an entire operator or opset.

## Licensing

This project is distributed under the [Apache License 2.0](LICENSE). Unless you
explicitly state otherwise, contributions submitted to this repository are
provided under the same license, consistent with Section 5 of Apache-2.0.
