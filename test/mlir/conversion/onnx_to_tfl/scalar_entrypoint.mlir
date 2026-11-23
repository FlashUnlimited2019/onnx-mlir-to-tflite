// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=legacy --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s | FileCheck %s

module {
  func.func @scalar_entry(%scalar: tensor<f32>, %index: tensor<i64>, %int32: tensor<i32>, %flag: tensor<i1>) -> (tensor<f32>, tensor<1x2x3xf32>, tensor<1xi64>, tensor<i32>, tensor<i1>) {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
    %vector_shape = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %expanded = "onnx.Expand"(%scalar, %shape) : (tensor<f32>, tensor<3xi64>) -> tensor<1x2x3xf32>
    %vector = "onnx.Reshape"(%index, %vector_shape) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %sum = "onnx.Add"(%scalar, %scalar) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %sum, %expanded, %vector, %int32, %flag : tensor<f32>, tensor<1x2x3xf32>, tensor<1xi64>, tensor<i32>, tensor<i1>
  }
  "onnx.EntryPoint"() {func = @scalar_entry} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<f32>
// CHECK-SAME: %arg1: tensor<i64>
// CHECK-SAME: %arg2: tensor<i32>
// CHECK-SAME: %arg3: tensor<i1>
// CHECK: "tfl.broadcast_to"(%arg0, {{.*}}) : (tensor<f32>, tensor<3xi32>) -> tensor<1x2x3xf32>
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<i64>, tensor<1xi32>) -> tensor<1xi64>
// CHECK: "tfl.add"(%arg0, %arg0) {{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: return {{.*}}, %arg2, %arg3 : tensor<f32>, tensor<1x2x3xf32>, tensor<1xi64>, tensor<i32>, tensor<i1>
