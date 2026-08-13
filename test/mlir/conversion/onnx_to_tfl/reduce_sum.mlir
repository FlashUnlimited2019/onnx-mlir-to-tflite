// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s -split-input-file | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>)
// CHECK: arith.constant dense<1> : tensor<1xi32>
// CHECK: arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.sum"{{.*}} {keep_dims = false} : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x3x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @main_graph(%input: tensor<1x384xi64>) -> tensor<1xf32> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384xi64>, tensor<1xi64>) -> tensor<1xi64>
    %cast = "onnx.Cast"(%result) <{saturate = 1 : si64, to = f32}> : (tensor<1xi64>) -> tensor<1xf32>
    return %cast : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x384xi64>) -> tensor<1xf32>
// CHECK: "tfl.sum"{{.*}} {keep_dims = false} : (tensor<1x384xi64>, tensor<1xi32>) -> tensor<1xi64>
// CHECK: "tfl.cast"{{.*}} : (tensor<1xi64>) -> tensor<1xf32>
// CHECK-NOT: onnx.
