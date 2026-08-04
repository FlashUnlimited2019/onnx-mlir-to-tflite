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
onnx-mlir importer and pass, TensorFlow's real `litert-opt` and
`flatbuffer_translate`, and then runs the MLP smoke/numerical test:

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

The supplied ResNet uses an NCHW ONNX input and an NHWC TFLite input:

```bash
./build/bin/onnx-to-tflite \
    ../models/resnet34-v2-7-224.onnx \
    -o /tmp/resnet34.tflite

python test/e2e/run_similarity.py \
    --onnx ../models/resnet34-v2-7-224.onnx \
    --tflite /tmp/resnet34.tflite \
    --seed 20260804
```

The same conversion and comparison are packaged as:

```bash
./scripts/test_resnet34.sh
```

If the TensorFlow tools are not at the bootstrapped sibling path, pass them
explicitly or set `LITERT_OPT` and `FLATBUFFER_TRANSLATE`:

```bash
./build/bin/onnx-to-tflite model.onnx -o model.tflite \
  --litert-opt /path/to/litert-opt \
  --flatbuffer-translate /path/to/flatbuffer_translate
```

Debug flags are `--dump-onnx-mlir`, `--dump-tfl-mlir`,
`--keep-intermediate-files`, and `--verify-each`/`--no-verify-each`. Dump files
are placed next to the output. Preserved intermediates use the directory
`<output>.intermediates`; when optimization is enabled it contains both
`model.unoptimized.tfl.mlir` and the final `model.tfl.mlir`. TFL optimization
is on by default and can be disabled for debugging with `--no-optimize-tfl`.

The equivalent explicit pipeline is:

```bash
./build/bin/onnx-mlir model.onnx --EmitONNXIR \
  --disable-conv-to-matmul -o /tmp/model
./build/bin/onnx-mlir-opt /tmp/model.onnx.mlir \
  --shape-inference --convert-onnx-to-tfl --canonicalize \
  -o /tmp/model.unoptimized.tfl.mlir
/path/to/litert-opt /tmp/model.unoptimized.tfl.mlir \
  --tfl-optimize-batch-matmul --tfl-optimize \
  --canonicalize --cse --symbol-dce --verify-each=true \
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

The pinned TensorFlow tools were built from source and their real `--help`
confirmed these pass/translation names and flags. `add_relu.onnx`, `mlp.onnx`,
and `reshape_transpose.onnx` each passed export, TFL3/schema round-trip,
`tf.lite.Interpreter` allocation/invocation, and ONNX Runtime numerical
comparison with `rtol=1e-5`, `atol=1e-6`. `operator_sweep.onnx` additionally
validated Identity, Sub/Mul/Div, Sigmoid/Tanh, Gemm alpha/beta, and Concat
through the same end-to-end path. The generated fixtures pin opset 18, ONNX IR
11, and seed 20260804.

The filtered MLIR suite has ten passing files, including Conv/MaxPool/reduction,
rank-4 NHWC conversion, and rank-3/rank-5 preservation checks. The supplied
opset-7 ResNet-34 was exported to an 87,194,492-byte `TFL3` FlatBuffer, round-trip
parsed by TensorFlow, loaded and invoked by `tf.lite.Interpreter`, and compared
against ONNX Runtime using seed 20260804. The measured output metrics were:

```text
cosine similarity:           0.9999999999996326
Euclidean distance:          4.31770709728951e-05
relative Euclidean distance: 8.617660878502137e-07
RMSE:                        1.365378869690908e-06
max absolute error:          5.72204589844e-06
```

The TensorFlow-side MLIR pipeline reduced the ResNet runtime graph from 197 to
96 operators, tensors from 303 to 196, and constant tensors from 105 to 99.
All 37 Reshape operators were folded, all 34 standalone Relu operators were
fused, 29 of 33 explicit Pad operators were absorbed into equivalent Conv2D
padding, and the final BatchMatMul/Add became one FullyConnected operator.
Inspect any result with:

```bash
TF_CPP_MIN_LOG_LEVEL=3 python utils/inspect_tflite.py /tmp/resnet34.tflite
```

`simple_conv.onnx` also passed the layout-aware comparison, including the
OIHW→OHWI case where source and destination shapes are numerically identical.

## Scope

The prototype supports static, ranked FP32 graphs containing Constant,
Identity, Add/Sub/Mul/Div, Relu/Sigmoid/Tanh, MatMul, constrained Gemm, Reshape,
Transpose, Concat, last-axis Softmax, constrained Conv/MaxPool, and the spatial
ReduceMean form produced for GlobalAveragePool. Runtime inputs and outputs are
FP32; i32/i64 constants are permitted for shape operations.

Exactly rank-4 TFLite activations and graph boundaries use NHWC. All other
activation ranks retain ONNX order, explicitly including rank 3 and rank 5.
See `docs/operator-support.md`, `docs/layout.md`, and
`docs/known-limitations.md`.
