// SPDX-License-Identifier: Apache-2.0
// RUN: onnx-mlir-opt --split-input-file --verify-diagnostics %s

func.func @no_inputs() -> tensor<0xf32> {
  // expected-error @+1 {{requires at least one input}}
  %0 = "onnx.Concat"() {axis = 0 : si64} : () -> tensor<0xf32>
  return %0 : tensor<0xf32>
}

// -----

func.func @empty_vectors_invalid_axis(%a: tensor<0xf32>, %b: tensor<0xf32>) -> tensor<0xf32> {
  // expected-error @+1 {{onnx.Concat: 'axis' value is 1, accepted range is [-1, 0]}}
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64} : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}
