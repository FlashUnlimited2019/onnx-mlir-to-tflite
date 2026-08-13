// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s -split-input-file | FileCheck %s

module {
  func.func @gelu_exact_rank4(%input: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %result = "onnx.Gelu"(%input) <{approximate = "none"}> : (tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @gelu_exact_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>)
// CHECK: "tfl.gelu"(%arg0) {approximate = false} : (tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @gelu_tanh_rank3(%input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %result = "onnx.Gelu"(%input) <{approximate = "tanh"}> : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @gelu_tanh_rank3} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>)
// CHECK: "tfl.gelu"(%arg0) {approximate = true} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @expand_rank3(%input: tensor<1x3x1xf32>) -> tensor<1x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 3, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Expand"(%input, %shape) : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @expand_rank3} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x1xf32>)
// CHECK: arith.constant dense<[1, 3, 4]> : tensor<3xi32>
// CHECK: "tfl.broadcast_to"(%arg0, {{.*}}) : (tensor<1x3x1xf32>, tensor<3xi32>) -> tensor<1x3x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @expand_into_rank4(%input: tensor<3x1x1xf32>) -> tensor<1x3x4x5xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 3, 4, 5]> : tensor<4xi64>} : () -> tensor<4xi64>
    %result = "onnx.Expand"(%input, %shape) : (tensor<3x1x1xf32>, tensor<4xi64>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @expand_into_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<3x1x1xf32>) -> tensor<1x4x5x3xf32>
// CHECK: "tfl.broadcast_to"(%arg0, {{.*}}) : (tensor<3x1x1xf32>, tensor<4xi32>) -> tensor<1x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x4x5xf32>, tensor<4xi32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.
