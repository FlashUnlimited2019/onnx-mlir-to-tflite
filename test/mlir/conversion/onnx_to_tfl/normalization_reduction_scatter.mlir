// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x3x4xf32>) -> (tensor<1x3x4xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) {
    %axes = onnx.Constant dense<[2]> : tensor<1xi64>
    %normalized = "onnx.LpNormalization"(%input) <{axis = 1 : si64, p = 2 : si64}> : (tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    %l1 = "onnx.ReduceL1"(%input, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<1x3xf32>
    %log_sum_exp = "onnx.ReduceLogSumExp"(%input, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<1x3xf32>
    %product = "onnx.ReduceProd"(%input, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<1x3xf32>
    %sum_square = "onnx.ReduceSumSquare"(%input, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<1x3xf32>
    return %normalized, %l1, %log_sum_exp, %product, %sum_square : tensor<1x3x4xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.equal"
// CHECK: "tfl.select_v2"
// CHECK: "tfl.div"
// CHECK: "tfl.abs"
// CHECK: "tfl.reduce_max"
// CHECK: "tfl.exp"
// CHECK: "tfl.log"
// CHECK: "tfl.reduce_prod"
// CHECK-NOT: tfl.square
// CHECK-NOT: onnx.

// -----

module {
  func.func @main_graph(%input: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %slope = onnx.Constant dense<2.000000e-01> : tensor<1x3x1x1xf32>
    %result = "onnx.PRelu"(%input, %slope) : (tensor<1x3x4x5xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>)
// CHECK: "tfl.reshape"{{.*}}tensor<1x1x3xf32>
// CHECK: "tfl.prelu"{{.*}} : (tensor<1x4x5x3xf32>, tensor<1x1x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @main_graph(%data: tensor<1x3x8xf32>, %indices: tensor<1x3x2xi64>, %updates: tensor<1x3x2xf32>) -> tensor<1x3x8xf32> {
    %result = "onnx.ScatterElements"(%data, %indices, %updates) <{axis = -1 : si64, reduction = "add"}> : (tensor<1x3x8xf32>, tensor<1x3x2xi64>, tensor<1x3x2xf32>) -> tensor<1x3x8xf32>
    return %result : tensor<1x3x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather_nd"
// CHECK: "tfl.scatter_nd"
// CHECK: "tfl.add"
// CHECK-NOT: onnx.
