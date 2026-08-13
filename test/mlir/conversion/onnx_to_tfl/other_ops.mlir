// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s -split-input-file | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: "tfl.transpose"

// -----

module {
  func.func @main_graph(%arg0: tensor<1x2xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x5xf32> {
    %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} : (tensor<1x2xf32>, tensor<1x3xf32>) -> tensor<1x5xf32>
    return %0 : tensor<1x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: "tfl.concatenation"
// CHECK-SAME: axis = 1 : i32

// -----

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: "tfl.softmax"
// CHECK-SAME: beta = 1.000000e+00 : f32
