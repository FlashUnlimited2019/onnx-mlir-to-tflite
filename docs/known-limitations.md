# Known limitations

- Only static ranked tensors are accepted. Dynamic dimensions and unranked
  tensors fail conversion.
- Runtime activations must be FP32. Shape constants may be i32 or i64.
- Quantization, calibration, sparse tensors, custom ops, and Select TF/Flex ops
  are unsupported.
- ONNX control flow, sequences, maps, optionals, RNN/GRU/LSTM, and functions
  beyond the single graph entry point are unsupported.
- MatMul/Gemm operands are restricted to ranks 2 through 4. ONNX rank-1 MatMul
  promotion is not implemented.
- Softmax must use the last dimension. Older pre-opset-13 flattened Softmax
  semantics are not claimed.
- Reshape requires a statically inferred output and rejects `allowzero=1`.
- The four-dimensional activation layout contract is NHWC. Conv/Pool and the
  associated NCHW boundary conversion are documented but not implemented in
  the first-stage MLP prototype. Other ranks keep ONNX order.
- Conv, pooling, global pooling, flatten, and pad currently produce explicit
  unsupported-operation diagnostics. The supplied ResNet model is therefore a
  diagnostic/coverage probe, not a claimed successful conversion.
- The bridge is coupled to both pinned MLIR revisions and to TensorFlow's TFL
  exporter ODS/schema. Updating either repository requires rerunning parser,
  FlatBuffer round-trip, interpreter-load, and numerical tests.
- The two-process design materializes constants in textual MLIR and can be
  slow or large for production models.
