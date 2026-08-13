// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// Gemma builds causal masks with ScatterElements whose indices select every
// element's own coordinate. The data tensor is consequently overwritten in
// full and the operation is exactly its updates operand.
module {
  func.func @scatter_identity_rank4(%data: tensor<1x1x3x4xf32>, %updates: tensor<1x1x3x4xf32>) -> tensor<1x1x3x4xf32> {
    %indices = "onnx.Constant"() {value = dense<[[[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]]]> : tensor<1x1x3x4xi64>} : () -> tensor<1x1x3x4xi64>
    %result = "onnx.ScatterElements"(%data, %indices, %updates) {axis = 2 : si64, reduction = "none"} : (tensor<1x1x3x4xf32>, tensor<1x1x3x4xi64>, tensor<1x1x3x4xf32>) -> tensor<1x1x3x4xf32>
    return %result : tensor<1x1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_identity_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x1xf32>, %arg1: tensor<1x3x4x1xf32>)
// CHECK-NOT: scatter
// CHECK: return %arg1

// -----

// Gemma residual connections can be recomposed as the optional full-shape
// bias of RMSLayerNormalization rather than a separate Add.
module {
  func.func @rms_norm_residual_bias(%input: tensor<1x4x8xf32>, %scale: tensor<8xf32>, %residual: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %result, %inv_std_dev = "onnx.RMSLayerNormalization"(%input, %scale, %residual) {axis = 2 : si64, epsilon = 1.0e-06 : f32, stash_type = 1 : si64} : (tensor<1x4x8xf32>, tensor<8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, none)
    return %result : tensor<1x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @rms_norm_residual_bias} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.mul"
// CHECK: "tfl.rsqrt"
// CHECK: "tfl.add"{{.*}}%arg2
// CHECK-NOT: onnx.

// -----

// Negative indices are also identity coordinates after ONNX normalization.
module {
  func.func @scatter_identity_negative(%data: tensor<2x3xf32>, %updates: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[0, 1, -1], [0, 1, -1]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
    %result = "onnx.ScatterElements"(%data, %indices, %updates) {axis = -1 : si64, reduction = "none"} : (tensor<2x3xf32>, tensor<2x3xi64>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_identity_negative} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>)
// CHECK-NOT: scatter
// CHECK: return %arg1
