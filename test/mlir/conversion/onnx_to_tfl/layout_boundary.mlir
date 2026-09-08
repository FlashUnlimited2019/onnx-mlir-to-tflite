// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each --split-input-file %s | FileCheck %s

module {
  func.func @rank3_output(%input: tensor<1x2x3xf32>) -> tensor<1x3x2xf32> {
    %moved = "onnx.Transpose"(%input) <{perm = [0, 2, 1]}> : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
    %active = "onnx.Relu"(%moved) : (tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
    return %active : tensor<1x3x2xf32>
  }
  "onnx.EntryPoint"() {func = @rank3_output} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3xf32>
// CHECK-SAME: -> (tensor<1x3x2xf32>
// CHECK: %[[ACTIVE:.*]] = "tfl.relu"(%arg0) : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK: %[[RESULT:.*]] = "tfl.transpose"(%[[ACTIVE]], {{.*}}) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<1x3x2xf32>
// CHECK: return %[[RESULT]] : tensor<1x3x2xf32>

// -----

module {
  func.func @equal_extents(%input: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    %moved = "onnx.Transpose"(%input) <{perm = [2, 0, 1]}> : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    return %moved : tensor<2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @equal_extents} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x2x2xf32>
// CHECK: arith.constant dense<[2, 0, 1]> : tensor<3xi32>
// CHECK: %[[MOVED:.*]] = "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK: return %[[MOVED]]

// -----

module {
  func.func @rank3_input(%input: tensor<3x2x4xf32>, %weight: tensor<4x5xf32>) -> (tensor<2x3x5xf32>, tensor<3x2x4xf32>) {
    %moved = "onnx.Transpose"(%input) <{perm = [1, 0, 2]}> : (tensor<3x2x4xf32>) -> tensor<2x3x4xf32>
    %product = "onnx.MatMul"(%moved, %weight) : (tensor<2x3x4xf32>, tensor<4x5xf32>) -> tensor<2x3x5xf32>
    return %product, %input : tensor<2x3x5xf32>, tensor<3x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @rank3_input} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<3x2x4xf32>
// CHECK-SAME: -> (tensor<2x3x5xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<3x2x4xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
// CHECK: "tfl.batch_matmul"
// CHECK: return {{.*}}, %arg0 : tensor<2x3x5xf32>, tensor<3x2x4xf32>

// -----

module {
  func.func @rank6_singleton_boundary(%input: tensor<1x2x1x3x4x5xf32>) -> tensor<5x2x3x1x4x1xf32> {
    %moved = "onnx.Transpose"(%input) <{perm = [5, 1, 3, 0, 4, 2]}> : (tensor<1x2x1x3x4x5xf32>) -> tensor<5x2x3x1x4x1xf32>
    return %moved : tensor<5x2x3x1x4x1xf32>
  }
  "onnx.EntryPoint"() {func = @rank6_singleton_boundary} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x1x3x4x5xf32>
// CHECK-SAME: -> (tensor<5x2x3x1x4x1xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x1x3x4x5xf32>, tensor<5xi32>) -> tensor<2x1x3x4x5xf32>
// CHECK: arith.constant dense<[4, 0, 2, 3, 1]> : tensor<5xi32>
// CHECK: "tfl.transpose"{{.*}} : (tensor<2x1x3x4x5xf32>, tensor<5xi32>) -> tensor<5x2x3x4x1xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<5x2x3x1x4x1xf32>
