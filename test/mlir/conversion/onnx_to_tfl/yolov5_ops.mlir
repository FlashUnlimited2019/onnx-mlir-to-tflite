// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @leaky_relu(%arg0: tensor<1x8x4x6xf32>) -> tensor<1x8x4x6xf32> {
    %0 = "onnx.LeakyRelu"(%arg0) {alpha = 1.000000e-01 : f32} : (tensor<1x8x4x6xf32>) -> tensor<1x8x4x6xf32>
    return %0 : tensor<1x8x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @leaky_relu} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x6x8xf32>
// CHECK: "tfl.leaky_relu"(%arg0) {alpha = 1.000000e-01 : f32} : (tensor<1x4x6x8xf32>) -> tensor<1x4x6x8xf32>

// -----

module {
  func.func @leaky_relu_rank5(%arg0: tensor<1x3x2x4x5xf32>) -> tensor<1x3x2x4x5xf32> {
    %0 = "onnx.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<1x3x2x4x5xf32>) -> tensor<1x3x2x4x5xf32>
    return %0 : tensor<1x3x2x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @leaky_relu_rank5} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x2x4x5xf32>
// CHECK: "tfl.leaky_relu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<1x3x2x4x5xf32>) -> tensor<1x3x2x4x5xf32>

// -----

module {
  func.func @slice_step_two(%arg0: tensor<1x3x6x7xf32>) -> tensor<1x3x3x3xf32> {
    %starts = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
    %ends = "onnx.Constant"() {value = dense<[6, 7]> : tensor<2xi64>} : () -> tensor<2xi64>
    %axes = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %steps = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x3x6x7xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3x3x3xf32>
    return %0 : tensor<1x3x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @slice_step_two} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x6x7x3xf32>
// CHECK-DAG: arith.constant dense<[0, 0, 1, 0]> : tensor<4xi32>
// CHECK-DAG: arith.constant dense<[1, 6, 7, 3]> : tensor<4xi32>
// CHECK-DAG: arith.constant dense<[1, 2, 2, 1]> : tensor<4xi32>
// CHECK: "tfl.strided_slice"(%arg0, {{.*}}, {{.*}}, {{.*}}) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<1x6x7x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x3x3x3xf32>
