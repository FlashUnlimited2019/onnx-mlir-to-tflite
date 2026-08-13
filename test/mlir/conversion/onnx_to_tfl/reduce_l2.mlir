// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --decompose-onnx --shape-inference --convert-onnx-to-tfl --canonicalize %s -split-input-file | FileCheck %s

// ReduceL2 is decomposed by onnx-mlir as Mul(x, x) -> ReduceSum -> Sqrt.
// Keep that representation because some target backends do not support
// TFLite SQUARE even though they support MUL.
module {
  func.func @main_graph(%input: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.ReduceL2"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
    return %result : tensor<2x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.mul"(%arg0, %arg0) {{.*fused_activation_function = "NONE".*}} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK: "tfl.sum"{{.*}} {keep_dims = true} : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<2x1x4xf32>
// CHECK: "tfl.sqrt"{{.*}} : (tensor<2x1x4xf32>) -> tensor<2x1x4xf32>
// CHECK-NOT: tfl.square
// CHECK-NOT: onnx.

// -----

// Cover a rank-4 NCHW input and keepdims=0. The reduction is performed in
// logical ONNX order, while the public input remains physical NHWC.
module {
  func.func @main_graph(%input: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>)
// CHECK: "tfl.mul"(%arg0, %arg0) {{.*}} : (tensor<1x3x4x2xf32>, tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: "tfl.transpose"{{.*}} : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.sum"{{.*}} {keep_dims = false} : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x3x4xf32>
// CHECK: "tfl.sqrt"{{.*}} : (tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK-NOT: tfl.square
// CHECK-NOT: onnx.
