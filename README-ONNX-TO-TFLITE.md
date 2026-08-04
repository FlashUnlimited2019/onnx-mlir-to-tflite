# ONNX to TFLite MLIR prototype

This branch implements a minimal, direct compiler route:

```text
ONNX protobuf -> ONNX dialect MLIR -> TFL dialect MLIR -> TFLite FlatBuffer
```

It does not construct a TensorFlow graph, Keras model, or SavedModel. The
conversion pass uses MLIR Dialect Conversion. Because the pinned onnx-mlir and
TensorFlow revisions use incompatible LLVM/MLIR revisions, the TFL MLIR text is
the checked boundary between two ABI-consistent executables. See
`docs/architecture.md` for the version investigation.

## Build

The complete bootstrap builds pinned LLVM/MLIR and TensorFlow sources, the
onnx-mlir importer and pass, TensorFlow's real `flatbuffer_translate`, and then
runs the MLP smoke/numerical test:

```bash
./scripts/bootstrap_and_build.sh
```

The initial build is large. It requires Git, CMake, a C++17 compiler, Python 3,
curl, network access for fixed dependencies, roughly 60 GB free disk, and 16 GB
or more RAM. `BUILD_JOBS` can reduce build concurrency.

## Usage

```bash
./build/bin/onnx-to-tflite \
    test/models/mlp.onnx \
    -o /tmp/mlp.tflite

python test/e2e/run_compare.py \
    --onnx test/models/mlp.onnx \
    --tflite /tmp/mlp.tflite
```

If `flatbuffer_translate` is not at the bootstrapped sibling TensorFlow path,
pass it explicitly or set `FLATBUFFER_TRANSLATE`:

```bash
./build/bin/onnx-to-tflite model.onnx -o model.tflite \
  --flatbuffer-translate /path/to/flatbuffer_translate
```

Debug flags are `--dump-onnx-mlir`, `--dump-tfl-mlir`,
`--keep-intermediate-files`, and `--verify-each`/`--no-verify-each`. Dump files
are placed next to the output. Preserved intermediates use the directory
`<output>.intermediates`.

The equivalent explicit pipeline is:

```bash
./build/bin/onnx-mlir model.onnx --EmitONNXIR \
  --disable-conv-to-matmul -o /tmp/model
./build/bin/onnx-mlir-opt /tmp/model.onnx.mlir \
  --shape-inference --convert-onnx-to-tfl --canonicalize \
  -o /tmp/model.tfl.mlir
/path/to/flatbuffer_translate -mlir-to-tflite-flatbuffer \
  /tmp/model.tfl.mlir -o model.tflite \
  -emit-builtin-tflite-ops=true \
  -emit-select-tf-ops=false -emit-custom-ops=false
```

The driver additionally checks the `TFL3` identifier and round-trips the result
through `--tflite-flatbuffer-to-mlir`, which invokes TensorFlow's parser/schema
verification path.

## Validated in this checkout

The pinned TensorFlow exporter was built from source and its real `--help`
confirmed these translation names and flags. `add_relu.onnx`, `mlp.onnx`, and
`reshape_transpose.onnx` each passed export, TFL3/schema round-trip,
`tf.lite.Interpreter` allocation/invocation, and ONNX Runtime numerical
comparison with `rtol=1e-5`, `atol=1e-6`. `operator_sweep.onnx` additionally
validated Identity, Sub/Mul/Div, Sigmoid/Tanh, Gemm alpha/beta, and Concat
through the same end-to-end path. The generated fixtures pin opset 18, ONNX IR
11, and seed 20260804.

The filtered MLIR suite has nine passing files, including rank-3/rank-5 layout
preservation checks. The supplied ResNet imports
successfully but intentionally stops at its first `onnx.Conv`; no ResNet
FlatBuffer or numerical result is claimed.

## Scope

The first stage supports static, ranked FP32 graphs containing Constant,
Identity, Add/Sub/Mul/Div, Relu/Sigmoid/Tanh, MatMul, constrained Gemm, Reshape,
Transpose, Concat, and last-axis Softmax. Runtime inputs and outputs are FP32;
i32/i64 constants are permitted for shape operations.

Exactly rank-4 activations use NHWC when layout-sensitive lowering is added.
All other activation ranks retain ONNX order, explicitly including rank 3 and
rank 5. Conv and Pool lowering is not yet implemented, so the supplied ResNet
and `simple_conv.onnx` currently serve as honest unsupported-op diagnostics.
See `docs/operator-support.md`, `docs/layout.md`, and
`docs/known-limitations.md`.
