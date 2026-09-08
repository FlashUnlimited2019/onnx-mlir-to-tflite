<!-- SPDX-License-Identifier: Apache-2.0 -->

# Convert an ONNX Model

Use `./build/bin/onnx-to-tflite` after building from source. In an extracted
runtime package, use `./bin/onnx-to-tflite` instead. The driver locates the
other required tools automatically in either layout.

## Basic conversion

```bash
./build/bin/onnx-to-tflite \
  /path/to/model.onnx \
  -o /path/to/model.tflite
```

The output directory is created when necessary. The driver publishes the
output only after MLIR verification, TFLite identifier validation, and a
FlatBuffer-to-MLIR round trip succeed.

## Named inference signature

Newly converted models include a `serving_default` TFLite signature by default,
in both layout modes and with or without TFL optimization. Signature input and
output names come from the ONNX graph. When converting MLIR without ONNX name
attributes, the fallback names are `input0`, `input1`, ... and `output0`,
`output1`, ... .

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
print(interpreter.get_signature_list())
run = interpreter.get_signature_runner("serving_default")
outputs = run(**inputs)  # inputs: a dictionary of input names to NumPy arrays
```

The signature does not change tensor shapes, data types, or layouts. Supply
arrays in the converted model's layout; see [Layout selection](#layout-selection).
The usual `set_tensor` / `invoke` / `get_tensor` interface remains available.
Previously generated files need to be reconverted to include the signature.

## Command-line options

The syntax below distinguishes options that require a value from switches that
are used on their own. Text in angle brackets, such as `<path>`, must be
replaced with a value. Switches in the last table take no value.

### Required arguments

| Syntax | Description |
| --- | --- |
| `<input.onnx>` | Input ONNX model. This positional argument must name a `.onnx` file. |
| `-o <output.tflite>`<br>`--output <output.tflite>` | Destination for the generated `.tflite` file. |

### Options that require a value

| Syntax | Allowed value | Description |
| --- | --- | --- |
| `--layout-mode <mode>` | `legacy` (default)<br>`experimental` | Select the layout strategy. `legacy` uses the stable policy, with physical NHWC for rank-4 FP32 graph boundaries and activations. `experimental` enables conservative per-value layout propagation to reduce layout conversions; rank-4 boundary layouts outside convolutional regions may change. |
| `--layout-diagnostics <level>` | `none` (default)<br>`summary`<br>`values` | Control layout-analysis output. `none` prints nothing, `summary` prints aggregate statistics, and `values` prints the inferred layout of individual values and can produce substantial output. |
| `--onnx-mlir <path>` | Executable path | Use a specific `onnx-mlir` executable instead of automatic discovery. |
| `--onnx-mlir-opt <path>` | Executable path | Use a specific `onnx-mlir-opt` executable instead of automatic discovery. |
| `--litert-opt <path>` | Executable path | Use a specific TensorFlow/LiteRT `litert-opt` executable instead of automatic discovery. |
| `--flatbuffer-translate <path>` | Executable path | Use a specific TensorFlow `flatbuffer_translate` executable instead of automatic discovery. |

For example:

```bash
./build/bin/onnx-to-tflite \
  /path/to/model.onnx \
  -o /path/to/model.tflite \
  --layout-mode experimental \
  --onnx-mlir /path/to/onnx-mlir
```

### Switches that take no value

| Switch | Description |
| --- | --- |
| `--optimize-tfl` | Run TensorFlow/LiteRT TFL canonicalization and fusion. This is enabled by default. |
| `--no-optimize-tfl` | Export without the optional `litert-opt` optimization stage. |
| `--verify-each` | Verify MLIR after each pass. This is enabled by default. |
| `--no-verify-each` | Disable per-pass MLIR verification, primarily for debugging. |
| `--dump-onnx-mlir` | Save the imported ONNX Dialect MLIR beside the output as `<name>.onnx.mlir`. |
| `--dump-tfl-mlir` | Save the final TFL Dialect MLIR beside the output as `<name>.tfl.mlir`. |
| `--keep-intermediate-files` | Preserve all temporary artifacts in `<output>.intermediates`. The driver refuses to overwrite an existing directory. |
| `--use-buffer-offset` | Store large constants after the FlatBuffer metadata and reference them by file offset. |
| `--no-use-buffer-offset` | Disable buffer-offset export even when large-model mode would be selected automatically. |
| `-h`<br>`--help` | Display the current command-line help. |

## Layout selection

The default `legacy` mode retains the stable external layout contract used by
existing converted models. Use `experimental` only when evaluating per-value
layout propagation and validate the generated model against the source ONNX
model before deployment.

Convolution and pooling regions remain channels-last in both modes because
TFLite kernels require NHWC. In experimental mode, tensors outside those
regions may retain their logical ONNX axis order when doing so avoids layout
conversions.

Non-rank-4 graph inputs and outputs retain their ONNX axis order in both
modes, even when experimental mode uses a different layout internally.

## Large models

Buffer-offset export is selected automatically when a sufficiently large
model uses a common adjacent external-data file such as `model.onnx.data`.
Use `--use-buffer-offset` to force it for another supported external-data
layout. The resulting TFLite file is self-contained, but its consumer must
support TFLite buffer offsets.

## Tool discovery

The driver searches `PATH`, its own binary directory, a bootstrapped source
tree, and the corresponding runtime-package directories. Tool-path options
override automatic discovery. `LITERT_OPT` and `FLATBUFFER_TRANSLATE`
environment variables may also specify the TensorFlow/LiteRT tools.
