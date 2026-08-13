// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// ONNX divides alpha by size; TFLite expects the already-scaled alpha and a
// symmetric channel radius. Rank-4 input is physical NHWC in the TFL graph.
module {
  func.func @lrn(%input: tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32> {
    %result = "onnx.LRN"(%input) {alpha = 1.0e-4 : f32, beta = 0.75 : f32, bias = 1.0 : f32, size = 5 : si64} : (tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @lrn} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
// CHECK: "tfl.local_response_normalization"(%arg0) {alpha = 2.000000e-05 : f32, beta = 7.500000e-01 : f32, bias = 1.000000e+00 : f32, radius = 2 : i32} : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
// CHECK-NOT: onnx.
