// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// BatchNormalization in the source model is decomposed by the importer to
// channel-broadcast Mul/Add. This file covers the new operations and rank
// transitions that surround those decomposed BatchNormalization branches.

module {
  func.func @reduce_rank5_to_rank4(%arg0: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[4]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x2x3x4x5xf32>, tensor<1xi64>) -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @reduce_rank5_to_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>) -> tensor<1x3x4x2xf32>
// CHECK: "tfl.mean"({{.*}}) {keep_dims = false} : (tensor<1x2x3x4x5xf32>, tensor<1xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x2x3x4xf32>, tensor<4xi32>) -> tensor<1x3x4x2xf32>

// -----

module {
  func.func @reduce_rank4_to_rank3(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3xf32> {
    %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.ReduceMean"(%arg0, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @reduce_rank4_to_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x2x3xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.mean"({{.*}}) {keep_dims = false} : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x2x3xf32>

// -----

module {
  func.func @gather_elements_rank3(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[[2, 0, 1], [1, -1, 0]]]> : tensor<1x2x3xi64>} : () -> tensor<1x2x3xi64>
    %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 2 : si64} : (tensor<1x2x3xf32>, tensor<1x2x3xi64>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_elements_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK: arith.constant {{.*}} : tensor<1x2x3x3xi32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<1x2x3xf32>, tensor<1x2x3x3xi32>) -> tensor<1x2x3xf32>

// -----

module {
  func.func @gather_elements_rank4(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<0> : tensor<1x2x2x3xi64>} : () -> tensor<1x2x2x3xi64>
    %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 3 : si64} : (tensor<1x2x2x3xf32>, tensor<1x2x2x3xi64>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_elements_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x2xf32>) -> tensor<1x2x3x2xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x2x3x2xf32>, tensor<4xi32>) -> tensor<1x2x2x3xf32>
// CHECK: "tfl.gather_nd"({{.*}}) : (tensor<1x2x2x3xf32>, tensor<1x2x2x3x4xi32>) -> tensor<1x2x2x3xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x2x2x3xf32>, tensor<4xi32>) -> tensor<1x2x3x2xf32>

// -----

module {
  func.func @matmul_rank3_output(%arg0: tensor<2x4x5xf32>, %arg1: tensor<5x3xf32>) -> tensor<2x4x3xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x4x5xf32>, tensor<5x3xf32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  "onnx.EntryPoint"() {func = @matmul_rank3_output} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5xf32>, %arg1: tensor<5x3xf32>) -> tensor<2x4x3xf32>
// CHECK: "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x4x5xf32>, tensor<5x3xf32>) -> tensor<2x4x3xf32>
