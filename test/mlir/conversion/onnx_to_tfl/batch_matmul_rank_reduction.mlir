// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @matching_rank5_batches(%lhs: tensor<1x2x3x4x5xf32>, %rhs: tensor<1x2x3x5x6xf32>) -> tensor<1x2x3x4x6xf32> {
    %result = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
    return %result : tensor<1x2x3x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @matching_rank5_batches} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x2x3x4x5xf32>, tensor<3xi32>) -> tensor<6x4x5xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x2x3x5x6xf32>, tensor<3xi32>) -> tensor<6x5x6xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<6x4x5xf32>, tensor<6x5x6xf32>) -> tensor<6x4x6xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<6x4x6xf32>, tensor<5xi32>) -> tensor<1x2x3x4x6xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @matching_rank6_batches(%lhs: tensor<4x1x6x2500x4x4xf32>, %rhs: tensor<4x1x6x2500x4x1xf32>) -> tensor<4x1x6x2500x4x1xf32> {
    %result = "onnx.MatMul"(%lhs, %rhs) : (tensor<4x1x6x2500x4x4xf32>, tensor<4x1x6x2500x4x1xf32>) -> tensor<4x1x6x2500x4x1xf32>
    return %result : tensor<4x1x6x2500x4x1xf32>
  }
  "onnx.EntryPoint"() {func = @matching_rank6_batches} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}} : (tensor<4x1x6x2500x4x4xf32>, tensor<3xi32>) -> tensor<60000x4x4xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<4x1x6x2500x4x1xf32>, tensor<3xi32>) -> tensor<60000x4x1xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<60000x4x4xf32>, tensor<60000x4x1xf32>) -> tensor<60000x4x1xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<60000x4x1xf32>, tensor<6xi32>) -> tensor<4x1x6x2500x4x1xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @fold_rhs_matrix_transpose(%lhs: tensor<1x2x3x4x5xf32>, %rhs: tensor<1x2x3x6x5xf32>) -> tensor<1x2x3x4x6xf32> {
    %first = "onnx.Transpose"(%rhs) <{perm = [0, 4, 2, 3, 1]}> : (tensor<1x2x3x6x5xf32>) -> tensor<1x5x3x6x2xf32>
    %second = "onnx.Transpose"(%first) <{perm = [0, 4, 2, 1, 3]}> : (tensor<1x5x3x6x2xf32>) -> tensor<1x2x3x5x6xf32>
    %result = "onnx.MatMul"(%lhs, %second) : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
    return %result : tensor<1x2x3x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @fold_rhs_matrix_transpose} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x2x3x6x5xf32>, tensor<3xi32>) -> tensor<6x6x5xf32>
// CHECK: "tfl.batch_matmul"{{.*}} {adj_x = false, adj_y = true} : (tensor<6x4x5xf32>, tensor<6x6x5xf32>) -> tensor<6x4x6xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @broadcast_rank5_batches(%lhs: tensor<1x2x3x4x5xf32>, %rhs: tensor<1x1x3x5x6xf32>) -> tensor<1x2x3x4x6xf32> {
    %result = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x3x4x5xf32>, tensor<1x1x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
    return %result : tensor<1x2x3x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @broadcast_rank5_batches} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x3x4x5xf32>, tensor<1x1x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
// CHECK-NOT: "tfl.reshape"
// CHECK-NOT: onnx.
