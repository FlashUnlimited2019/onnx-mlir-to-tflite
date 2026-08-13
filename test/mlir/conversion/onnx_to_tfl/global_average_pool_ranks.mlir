// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// Importer form of rank-3 GlobalAveragePool: reduce the sole spatial axis.
module {
  func.func @global_average_pool_rank3(%input: tensor<2x4x9xf32>) -> tensor<2x4x1xf32> {
    %result = "onnx.ReduceMeanV13"(%input) <{axes = [2], keepdims = 1 : si64}> : (tensor<2x4x9xf32>) -> tensor<2x4x1xf32>
    return %result : tensor<2x4x1xf32>
  }
  "onnx.EntryPoint"() {func = @global_average_pool_rank3} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x9xf32>) -> tensor<2x4x1xf32>
// CHECK: arith.constant dense<2> : tensor<1xi32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = true} : (tensor<2x4x9xf32>, tensor<1xi32>) -> tensor<2x4x1xf32>
// CHECK-NOT: onnx.

// -----

// Importer form of rank-5 GlobalAveragePool: reduce D, H, and W.
module {
  func.func @global_average_pool_rank5(%input: tensor<1x3x2x4x5xf32>) -> tensor<1x3x1x1x1xf32> {
    %result = "onnx.ReduceMeanV13"(%input) <{axes = [2, 3, 4], keepdims = 1 : si64}> : (tensor<1x3x2x4x5xf32>) -> tensor<1x3x1x1x1xf32>
    return %result : tensor<1x3x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @global_average_pool_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x2x4x5xf32>) -> tensor<1x3x1x1x1xf32>
// CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = true} : (tensor<1x3x2x4x5xf32>, tensor<3xi32>) -> tensor<1x3x1x1x1xf32>
// CHECK-NOT: onnx.
