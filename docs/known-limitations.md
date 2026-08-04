# Known limitations

- Only static ranked tensors are accepted. Dynamic dimensions and unranked
  tensors fail conversion.
- Runtime activations must be FP32. Shape constants may be i32 or i64.
- Quantization, calibration, sparse tensors, custom ops, and Select TF/Flex ops
  are unsupported.
- ONNX control flow, sequences, maps, optionals, RNN/GRU/LSTM, and functions
  beyond the single graph entry point are unsupported.
- MatMul operands are restricted to ranks 2-3; rank 4 is rejected because its
  matrix axes conflict with the rank-4 spatial layout policy. ONNX rank-1
  MatMul promotion is not implemented.
- Softmax must use the last dimension. Older pre-opset-13 flattened Softmax
  semantics are not claimed.
- Rank-4 Softmax and arbitrary rank-4 Transpose are rejected. Rank-4 Concat is
  supported by axis remapping.
- Reshape requires a statically inferred output and rejects `allowzero=1`.
  Rank-4 layout crossing currently supports the ResNet case
  `[N,C,1,1] -> non-rank-4`; general rank-4 Reshape is rejected.
- The four-dimensional TFLite graph boundary and internal activation layout is
  NHWC. Callers must transpose ONNX NCHW rank-4 inputs; all other ranks keep
  ONNX order.
- Conv is limited to static 2D FP32, `group=1`, dilation `[1,1]`, and static
  nonnegative padding. Depthwise/grouped Conv and `SAME_LOWER` are unsupported.
- MaxPool is limited to static 2D FP32, dilation `[1,1]`, `ceil_mode=0`, and
  `storage_order=0`. AveragePool and standalone ONNX Pad are unsupported.
- GlobalAveragePool is supported only through the imported rank-4
  `ReduceMeanV13(axes=[2,3], keepdims=1)` form exercised by ResNet. Other
  rank-4 reductions are rejected.
- BatchNormalization has no direct lowering pattern; the onnx-mlir importer
  must fold it into Conv constants or decompose it into channel Mul/Add, as it
  does for the tested opset-7 ResNet.
- The bridge is coupled to both pinned MLIR revisions and to TensorFlow's TFL
  exporter ODS/schema. Updating either repository requires rerunning parser,
  FlatBuffer round-trip, interpreter-load, and numerical tests.
- The two-process design materializes constants in textual MLIR. For the tested
  ResNet, each textual MLIR/round-trip file is about 167 MiB and preserved
  intermediates total about 583 MiB, although the FlatBuffer is about 84 MiB.
