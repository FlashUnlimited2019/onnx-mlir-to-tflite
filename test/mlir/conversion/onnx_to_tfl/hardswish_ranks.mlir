// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @hardswish_rank2(%arg0: tensor<2x12xf32>) -> tensor<2x12xf32> {
    %0 = "onnx.HardSwish"(%arg0) : (tensor<2x12xf32>) -> tensor<2x12xf32>
    return %0 : tensor<2x12xf32>
  }
  "onnx.EntryPoint"() {func = @hardswish_rank2} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x12xf32>) -> tensor<2x12xf32>
// CHECK: "tfl.hard_swish"(%arg0) : (tensor<2x12xf32>) -> tensor<2x12xf32>

// -----

module {
  func.func @hardswish_rank3(%arg0: tensor<2x4x6xf32>) -> tensor<2x4x6xf32> {
    %0 = "onnx.HardSwish"(%arg0) : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @hardswish_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
// CHECK: "tfl.hard_swish"(%arg0) : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>

// -----

module {
  func.func @hardswish_rank5(%arg0: tensor<1x3x2x4x4xf32>) -> tensor<1x3x2x4x4xf32> {
    %0 = "onnx.HardSwish"(%arg0) : (tensor<1x3x2x4x4xf32>) -> tensor<1x3x2x4x4xf32>
    return %0 : tensor<1x3x2x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @hardswish_rank5} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x2x4x4xf32>) -> tensor<1x3x2x4x4xf32>
// CHECK: "tfl.hard_swish"(%arg0) : (tensor<1x3x2x4x4xf32>) -> tensor<1x3x2x4x4xf32>
