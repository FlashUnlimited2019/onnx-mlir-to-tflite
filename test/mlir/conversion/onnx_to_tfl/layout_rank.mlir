// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s -split-input-file | FileCheck %s

module {
  func.func @rank3_graph(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @rank3_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>
// CHECK: "tfl.add"
// CHECK-NOT: "tfl.transpose"

// -----

module {
  func.func @rank4_graph(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @rank4_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>
// CHECK: "tfl.relu"(%arg0) : (tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: "tfl.transpose"

// -----

module {
  func.func @symmetric_shape_constant_graph() -> tensor<1x2x2x2xf32> {
    %0 = "onnx.Constant"() {value = dense<[[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]]> : tensor<1x2x2x2xf32>} : () -> tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @symmetric_shape_constant_graph} : () -> ()
}
// CHECK-LABEL: func.func @main()
// CHECK: arith.constant dense<{{.*0.000000e\+00, 4.000000e\+00.*1.000000e\+00, 5.000000e\+00.*2.000000e\+00, 6.000000e\+00.*3.000000e\+00, 7.000000e\+00.*}}> : tensor<1x2x2x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @rank5_graph(%arg0: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
    return %0 : tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @rank5_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>
// CHECK: "tfl.relu"
// CHECK-NOT: "tfl.transpose"
