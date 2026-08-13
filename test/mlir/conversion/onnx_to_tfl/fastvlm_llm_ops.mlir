// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// FastVLM's last-token selection computes an absolute-value score over the
// input embeddings before ReduceMax/Greater/Where/GatherElements.
module {
  func.func @abs_embedding(%input: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %result = "onnx.Abs"(%input) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %result : tensor<1x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @abs_embedding} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.abs"(%arg0) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
// CHECK-NOT: onnx.

// -----

// FastVLM constructs token positions with an i64 Where before reducing and
// gathering the final valid-token logits. Include scalar broadcasting.
module {
  func.func @where_i64(%data: tensor<1x4x4xf32>, %score: tensor<1x4xf32>, %positions: tensor<1x4xi64>) -> tensor<1x1x4xf32> {
    %zero = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %fallback = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %condition = "onnx.Greater"(%score, %zero) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xi1>
    %selected = "onnx.Where"(%condition, %positions, %fallback) : (tensor<1x4xi1>, tensor<1x4xi64>, tensor<i64>) -> tensor<1x4xi64>
    %maximum = "onnx.ReduceMaxV13"(%selected) <{axes = [1], keepdims = 0 : si64}> : (tensor<1x4xi64>) -> tensor<1xi64>
    %shape = "onnx.Constant"() {value = dense<[1, 1, 1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %reshaped = "onnx.Reshape"(%maximum, %shape) <{allowzero = 0 : si64}> : (tensor<1xi64>, tensor<3xi64>) -> tensor<1x1x1xi64>
    %expanded_shape = "onnx.Constant"() {value = dense<[1, 1, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %expanded = "onnx.Expand"(%reshaped, %expanded_shape) : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x1x4xi64>
    %result = "onnx.GatherElements"(%data, %expanded) <{axis = 1 : si64}> : (tensor<1x4x4xf32>, tensor<1x1x4xi64>) -> tensor<1x1x4xf32>
    return %result : tensor<1x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @where_i64} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.greater"
// CHECK: "tfl.select_v2"{{.*}} : (tensor<1x4xi1>, tensor<1x4xi64>, tensor<i64>) -> tensor<1x4xi64>
// CHECK: "tfl.reduce_max"{{.*}} : (tensor<1x4xi64>, tensor<1xi32>) -> tensor<1xi64>
// CHECK: "tfl.reshape"{{.*}} : (tensor<1xi64>, tensor<3xi32>) -> tensor<1x1x1xi64>
// CHECK: "tfl.broadcast_to"{{.*}} : (tensor<1x1x1xi64>, tensor<3xi32>) -> tensor<1x1x4xi64>
// CHECK: "tfl.cast"{{.*}} : (tensor<1x1x4xi64>) -> tensor<1x1x4xi32>
// CHECK: "tfl.concatenation"{{.*}} : (tensor<1x1x4x1xi32>, tensor<1x1x4x1xi32>, tensor<1x1x4x1xi32>) -> tensor<1x1x4x3xi32>
// CHECK: "tfl.gather_nd"{{.*}} : (tensor<1x4x4xf32>, tensor<1x1x4x3xi32>) -> tensor<1x1x4xf32>
// CHECK-NOT: onnx.
