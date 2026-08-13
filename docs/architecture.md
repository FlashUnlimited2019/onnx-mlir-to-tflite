<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

## Overview

This project extends ONNX-MLIR with a direct, static ONNX-to-TFLite compiler
pipeline. It reuses the upstream ONNX importer and ONNX Dialect infrastructure,
then lowers the imported program to the TensorFlow Lite Dialect before
serializing it as a TFLite FlatBuffer.

The default pipeline is:

```text
ONNX protobuf
    |
    |  onnx-mlir --EmitONNXIR
    v
ONNX Dialect MLIR
    |
    |  onnx-mlir-opt --convert-onnx-to-tfl
    v
TFL Dialect MLIR
    |
    |  TensorFlow/LiteRT litert-opt
    |  optional; enabled by default
    v
Optimized TFL Dialect MLIR
    |
    |  TensorFlow flatbuffer_translate
    v
TFLite FlatBuffer
```

The pipeline does not construct a TensorFlow graph, Keras model, or SavedModel.
It also does not fall back to Select TF, Flex, or custom operators.

## Tool and ABI boundary

ONNX-MLIR and TensorFlow are built from different LLVM/MLIR revisions. MLIR's
C++ APIs, registered type and operation identifiers, generated ODS classes,
and LLVM support libraries do not provide a stable cross-revision ABI. Linking
both compiler stacks into one process would therefore be unsafe.

The implementation keeps each stack in a separate executable and uses textual
TFL Dialect MLIR as the boundary between them. The ONNX-MLIR side produces the
textual IR; TensorFlow reparses it using its authoritative TFL Dialect classes,
verifiers, and FlatBuffer exporter. An incompatible syntax or verifier change
fails at this explicit boundary.

## Stage responsibilities

### ONNX import

The upstream ONNX-MLIR importer reads the ONNX protobuf and creates ONNX
Dialect MLIR. Existing shape inference, canonicalization, and importer
decomposition are reused where applicable.

### ONNX-to-TFL conversion

`onnx-mlir-opt --convert-onnx-to-tfl` uses MLIR Dialect Conversion with a type
converter, operation conversion patterns, and a full-conversion target. ONNX
operations are illegal after this stage, so an unsupported operation or
configuration causes conversion to fail instead of remaining in the output.

Lowering patterns are responsible for:

- selecting builtin TFL operations or equivalent builtin compositions;
- preserving ONNX broadcasting, axis, padding, and reduction semantics;
- adapting tensor and filter layouts to TFLite conventions;
- reducing supported high-rank static operations when required by TFLite;
- rejecting unsupported types, shapes, ranks, attributes, and dynamic forms.

The exact accepted configurations are documented in the
[operator support matrix](operator-support.md).

### TFL optimization

TensorFlow/LiteRT optimization is optional and enabled by default. The driver
runs the following passes after TensorFlow has parsed the textual TFL IR:

```text
tfl-optimize-batch-matmul
  -> tfl-optimize
  -> canonicalize
  -> cse
  -> symbol-dce
```

These passes perform TFL-specific folding and fusion, canonicalize the IR, and
remove redundant operations and symbols. `--no-optimize-tfl` bypasses this
stage and sends the unoptimized TFL IR directly to the exporter.

### FlatBuffer export

TensorFlow's `flatbuffer_translate` serializes the verified TFL module with
builtin TFLite operations enabled and Select TF and custom operations disabled.
Large constant buffers can use TFLite buffer offsets when the ordinary
FlatBuffer representation is insufficient.

## Layout contract

Rank-4 floating-point activations use NHWC in the generated TFLite graph.
Convolution filters and broadcast constants are transformed at compile time to
the layouts required by their target operations. Other ranks normally retain
their logical ONNX dimension order.

Some rank-sensitive operations temporarily restore logical ONNX order or
collapse compatible static dimensions, then convert the result back to its
required representation. Graph inputs and outputs follow the same rank-based
contract; no entry or exit transpose is inserted solely to expose an NCHW
runtime interface.

Detailed layout and convolution rules are defined in the
[layout policy](layout.md).

## Driver and artifact handling

The `onnx-to-tflite` driver discovers or accepts explicit paths to all required
executables, creates isolated intermediate files, propagates subprocess
failures, and publishes the destination atomically. Diagnostic options can
retain the imported ONNX IR, unoptimized TFL IR, optimized TFL IR, and other
intermediate artifacts.

## Verification

A successful conversion includes several independent checks:

1. MLIR verification during ONNX-to-TFL conversion.
2. Full legalization of all source ONNX operations.
3. TensorFlow parsing and verification of the generated TFL Dialect MLIR.
4. Validation of the `TFL3` FlatBuffer identifier.
5. Reverse translation of the FlatBuffer back to MLIR.
6. Atomic output publication only after all required checks succeed.

Regression tests cover lowering patterns, diagnostics, layouts, shapes, and
end-to-end numerical behavior. Test coverage categories and supported operator
configurations are recorded without coupling the architecture to particular
model artifacts.

## Scope

The compiler is designed primarily for statically shaped, ranked inference
graphs. Dynamic shapes, general control flow, arbitrary custom operators,
quantization, and other unsupported configurations remain outside the current
scope unless explicitly listed in the operator support matrix.

See [known limitations](known-limitations.md) for project-wide constraints and
the [build guide](build.md) for the pinned toolchains and repository-local
dependency layout.
