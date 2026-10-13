// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// A full-coordinate grid overwrites one complete channel. Recognize the
// source overwrite semantics and copy values without floating arithmetic.
module {
  func.func @scatter_nd_rank4(%data: tensor<1x2x2x3xf32>, %updates: tensor<1x1x2x3xf32>) -> tensor<1x2x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[[[[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 2]], [[0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2]]]]]> : tensor<1x1x2x3x4xi64>} : () -> tensor<1x1x2x3x4xi64>
    %result = "onnx.ScatterND"(%data, %indices, %updates) <{reduction = "none"}> : (tensor<1x2x2x3xf32>, tensor<1x1x2x3x4xi64>, tensor<1x1x2x3xf32>) -> tensor<1x2x2x3xf32>
    return %result : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_nd_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x2xf32>{{( \{[^}]*\})?}}, %arg1: tensor<1x2x3x1xf32>{{( \{[^}]*\})?}}) -> (tensor<1x2x3x2xf32>{{( \{[^}]*\})?}})
// CHECK: %[[PREFIX:.*]] = "tfl.slice"(%arg0, {{.*}}) : (tensor<1x2x3x2xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
// CHECK-NEXT: %[[RESULT:.*]] = "tfl.concatenation"(%[[PREFIX]], %arg1) {axis = 3 : i32, fused_activation_function = "NONE"}
// CHECK-NEXT: return %[[RESULT]]
// CHECK-NOT: onnx.

// -----

// The right-feature lookup creates a rank-6 intermediate from rank-5 data and
// a rank-2 constant index grid.
module {
  func.func @gather_rank5_to_rank6(%data: tensor<1x1x3x2x5xf32>) -> tensor<1x1x3x2x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[0, 1, 2], [2, 3, 4]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
    %result = "onnx.Gather"(%data, %indices) <{axis = 4 : si64}> : (tensor<1x1x3x2x5xf32>, tensor<2x3xi64>) -> tensor<1x1x3x2x2x3xf32>
    return %result : tensor<1x1x3x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_rank5_to_rank6} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x3x2x5xf32>{{( \{[^}]*\})?}}) -> (tensor<1x1x3x2x2x3xf32>{{( \{[^}]*\})?}})
// CHECK: arith.constant dense<{{.*}}> : tensor<2x3xi32>
// CHECK: "tfl.gather"(%arg0, {{.*}}) {axis = 4 : i32, batch_dims = 0 : i32} : (tensor<1x1x3x2x5xf32>, tensor<2x3xi32>) -> tensor<1x1x3x2x2x3xf32>
// CHECK-NOT: onnx.
