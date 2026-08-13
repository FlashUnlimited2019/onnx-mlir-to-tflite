<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Modified by FlashUnlimited2019 in 2026. -->

# ONNX-MLIR to TFLite

ONNX-MLIR TFLite is an experimental compiler pipeline that converts static
ONNX models directly to TensorFlow Lite FlatBuffers through MLIR. It is built
as a downstream extension of [ONNX-MLIR](https://github.com/onnx/onnx-mlir)
and reuses its ONNX importer, ONNX Dialect, shape inference, and
canonicalization infrastructure.

The project does not construct a TensorFlow graph, Keras model, or SavedModel.
Unsupported configurations fail conversion instead of falling back to Select
TF, Flex, or custom operators.

## Compiler pipeline

The default compilation path is:

```text
ONNX protobuf
    |
    |  onnx-mlir --EmitONNXIR
    v
ONNX Dialect MLIR
    |
    |  onnx-mlir-opt --convert-onnx-to-tfl
    v
TFL Dialect MLIR (unoptimized)
    |
    |  TensorFlow/LiteRT litert-opt
    |  optional; enabled by default
    v
TFL Dialect MLIR (optimized)
    |
    |  TensorFlow flatbuffer_translate
    v
TFLite FlatBuffer
```

Artifacts and intermediate IR appear as nodes; the executable responsible for
each transformation appears on the connecting arrow. With
`--no-optimize-tfl`, the `litert-opt` stage is bypassed and the unoptimized TFL
Dialect MLIR is passed directly to `flatbuffer_translate`.

The `onnx-to-tflite` driver orchestrates these stages, verifies intermediate
results, and publishes the output only after FlatBuffer validation succeeds.

The ONNX-to-TFL stage is implemented with MLIR Dialect Conversion and requires
the source ONNX operations to be fully legalized. TFL optimization remains in
the TFL Dialect and applies TensorFlow/LiteRT canonicalization, fusion, and
cleanup passes before FlatBuffer export.

ONNX-MLIR and TensorFlow are pinned to different LLVM/MLIR revisions. To avoid
mixing incompatible MLIR C++ ABIs in one process, the pipeline uses textual TFL
MLIR as the checked interface between the ONNX-MLIR tools and TensorFlow tools.
TensorFlow reparses and verifies that IR using its authoritative TFL Dialect
implementation.

## Key properties

- Static, ranked tensor compilation with FP32 as the primary activation type
  and selected integer and boolean tensor support.
- Layout-aware lowering: rank-4 activations use NHWC in TFLite, while other
  graph ranks retain their ONNX axis order.
- Compile-time conversion of convolution filters and broadcast parameters to
  the layouts expected by TFLite.
- Direct lowering to builtin TFLite operations without TensorFlow graph
  conversion or runtime fallback operators.
- Optional TensorFlow/LiteRT optimization, enabled by default and disabled with
  `--no-optimize-tfl` for debugging or A/B comparison.
- Automatic buffer-offset export for large models using common adjacent ONNX
  external-data layouts.
- Per-pass verification, FlatBuffer identifier checks, TensorFlow round-trip
  parsing, and atomic output publication.

See the [operator support matrix](docs/operator-support.md) for the exact
supported types, ranks, attributes, and opset revisions. Only configurations
listed there are claimed to be supported.

## Build

The complete pipeline requires building both the pinned LLVM/MLIR toolchain
and two TensorFlow/LiteRT tools: `litert-opt` for optional TFL optimization and
`flatbuffer_translate` for required FlatBuffer export. TensorFlow is invoked
as a separate toolchain and is not linked into the ONNX-MLIR executables.

For a complete bootstrap build from an activated Python environment:

```bash
python -m pip install -r requirements.txt onnxruntime tensorflow

PYTHON_BIN="$(command -v python)" \
BUILD_JOBS=8 \
./scripts/bootstrap_and_build.sh
```

The script clones pinned LLVM and TensorFlow revisions inside this repository,
builds all required tools, and runs an end-to-end smoke test. Generated source
and build trees are ignored by Git. See the [build guide](docs/build.md) for
prerequisites, repository layout, build products, configuration options, and
instructions for reusing existing TensorFlow tools.

## Convert a model

```bash
./build/bin/onnx-to-tflite \
  /path/to/model.onnx \
  -o /path/to/model.tflite
```

The driver automatically locates the other ONNX-MLIR and TensorFlow/LiteRT
tools in a bootstrapped workspace. Explicit paths can also be provided:

```bash
./build/bin/onnx-to-tflite \
  /path/to/model.onnx \
  -o /path/to/model.tflite \
  --onnx-mlir /path/to/onnx-mlir \
  --onnx-mlir-opt /path/to/onnx-mlir-opt \
  --litert-opt /path/to/litert-opt \
  --flatbuffer-translate /path/to/flatbuffer_translate
```

Useful diagnostic options include:

```text
--no-optimize-tfl          skip the optional TFL optimization stage
--dump-onnx-mlir           preserve the imported ONNX Dialect MLIR
--dump-tfl-mlir            preserve the final TFL Dialect MLIR
--keep-intermediate-files  preserve all intermediate files
--no-verify-each           disable per-pass MLIR verification
--use-buffer-offset        force large-model buffer-offset export
```

Run `./build/bin/onnx-to-tflite --help` for the complete command-line
interface.

## Layout contract

ONNX commonly represents rank-4 activations as NCHW, while TFLite uses NHWC.
This project exposes rank-4 TFLite graph inputs and outputs as NHWC and keeps
convolution and pooling regions in NHWC. It does not insert runtime transposes
solely to preserve an NCHW external ABI. Callers must therefore adapt rank-4
boundary tensors when comparing or integrating the generated model.

Rank-1, rank-2, rank-3, and rank-5-or-higher graph boundaries retain their
ONNX axis order. Local layout conversions required by individual operations,
including Conv3D fallback, are represented explicitly inside the graph.

See the [layout policy](docs/layout.md) for convolution filter layouts,
broadcast rules, rank-sensitive behavior, and Conv3D reduction details.

## Large models

For models whose constants would exceed ordinary FlatBuffer metadata limits,
the driver can use TensorFlow's buffer-offset format. Common adjacent external
data files such as `model.onnx.data` are detected automatically once the model
size reaches the configured threshold. The resulting `.tflite` remains a
single self-contained file, but its consumer must support TFLite buffer
offsets.

Use `--use-buffer-offset` to enable this mode explicitly for other external
data layouts.

## Verification and tests

Successful conversion includes all of the following checks:

1. MLIR verification after each conversion stage by default.
2. Rejection of unlegalized ONNX operations.
3. TFLite `TFL3` file-identifier validation.
4. FlatBuffer import back into MLIR through TensorFlow's parser.
5. Atomic publication only after validation succeeds.

Run the ONNX-to-TFL MLIR regression suite with:

```bash
llvm-project/build/bin/llvm-lit \
  -sv build/test/mlir/conversion/onnx_to_tfl
```

The end-to-end similarity utility handles the rank-4 boundary layout contract,
compares ONNX Runtime and TFLite results, and reports cosine similarity,
Euclidean distance, relative Euclidean distance, RMSE, and maximum absolute
error:

```bash
python test/e2e/run_similarity.py \
  --onnx test/models/mlp.onnx \
  --tflite build/test/models/mlp.tflite
```

## Current scope

The project focuses on statically shaped inference graphs. Dynamic shapes,
general control flow, quantization, sparse tensors, and arbitrary custom or
Flex operators are outside the current scope. Some complex operations are
supported only for constrained static configurations or through ONNX-MLIR
importer decomposition.

Review the [known limitations](docs/known-limitations.md) before relying on a
configuration that is not explicitly covered by the
[operator support matrix](docs/operator-support.md).

## Documentation

| Document | Description |
| --- | --- |
| [Build guide](docs/build.md) | Complete LLVM/MLIR, ONNX-MLIR, and TensorFlow/LiteRT build procedure |
| [Operator support](docs/operator-support.md) | Supported ONNX operations, opsets, types, ranks, attributes, and test coverage |
| [Architecture](docs/architecture.md) | Cross-version MLIR boundary, conversion design, optimization, and export details |
| [Layout policy](docs/layout.md) | Rank-sensitive layout contract and convolution lowering rules |
| [Known limitations](docs/known-limitations.md) | Unsupported configurations and implementation constraints |

The original ONNX-MLIR documentation remains available under `docs/` and at
[onnx.ai/onnx-mlir](https://onnx.ai/onnx-mlir/).

## Project lineage and license

This repository retains the history and compiler infrastructure of upstream
ONNX-MLIR and adds the ONNX-to-TFL conversion and TFLite export pipeline.

The upstream base is ONNX-MLIR commit
[`7cea64f8aa2fbc56f859df768211a95dc2c72ad6`](https://github.com/onnx/onnx-mlir/commit/7cea64f8aa2fbc56f859df768211a95dc2c72ad6).
All project-specific compiler, tooling, test, and documentation changes are
maintained independently from that revision.

See the repository [LICENSE](LICENSE) for licensing information.

This is an independent project and is not an official ONNX or ONNX-MLIR
release.
