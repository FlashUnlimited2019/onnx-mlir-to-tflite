// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @tile_rank6(%input: tensor<1x1x6x1x4x4xf32>) -> tensor<4x1x6x2500x4x4xf32> {
    %repeats = "onnx.Constant"() {value = dense<[4, 1, 1, 2500, 1, 1]> : tensor<6xi64>} : () -> tensor<6xi64>
    %result = "onnx.Tile"(%input, %repeats) : (tensor<1x1x6x1x4x4xf32>, tensor<6xi64>) -> tensor<4x1x6x2500x4x4xf32>
    return %result : tensor<4x1x6x2500x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @tile_rank6} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x6x1x4x4xf32>)
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x1x6x1x4x4xf32>, tensor<5xi32>) -> tensor<1x6x1x4x4xf32>
// CHECK: "tfl.tile"{{.*}} : (tensor<1x6x1x4x4xf32>, tensor<5xi32>) -> tensor<4x6x2500x4x4xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<4x6x2500x4x4xf32>, tensor<6xi32>) -> tensor<4x1x6x2500x4x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @scatter_nd_rank6_indices(%data: tensor<1x1x1x1x1xf32>, %updates: tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32> {
    %indices = "onnx.Constant"() {value = dense<0> : tensor<1x1x1x1x1x5xi64>} : () -> tensor<1x1x1x1x1x5xi64>
    %result = "onnx.ScatterND"(%data, %indices, %updates) <{reduction = "none"}> : (tensor<1x1x1x1x1xf32>, tensor<1x1x1x1x1x5xi64>, tensor<1x1x1x1x1xf32>) -> tensor<1x1x1x1x1xf32>
    return %result : tensor<1x1x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_nd_rank6_indices} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x1x1x1xf32>, %arg1: tensor<1x1x1x1x1xf32>)
// CHECK: arith.constant dense<0> : tensor<1x5xi32>
// CHECK: %[[UPDATES:.*]] = "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x1x1x1x1xf32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<1x1x1x1x1xf32>, tensor<1x5xi32>) -> tensor<1xf32>
// CHECK: "tfl.scatter_nd"{{.*}} : (tensor<1x5xi32>, tensor<1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x1xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @gather_singleton_rank6(%input: tensor<2x2500x8x1x4x2xf32>) -> tensor<2x2500x8x4x2xf32> {
    %index = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %result = "onnx.Gather"(%input, %index) <{axis = 3 : si64}> : (tensor<2x2500x8x1x4x2xf32>, tensor<i64>) -> tensor<2x2500x8x4x2xf32>
    return %result : tensor<2x2500x8x4x2xf32>
  }
  "onnx.EntryPoint"() {func = @gather_singleton_rank6} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2500x8x1x4x2xf32>)
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x2500x8x1x4x2xf32>, tensor<5xi32>) -> tensor<2x2500x8x4x2xf32>
// CHECK-NOT: "tfl.gather"
// CHECK-NOT: onnx.

// -----

module {
  func.func @transpose_rank7(%input: tensor<1x2500x8x2x1x4x2xf32>) -> tensor<1x2x2500x8x1x4x2xf32> {
    %result = "onnx.Transpose"(%input) <{perm = [0, 3, 1, 2, 4, 5, 6]}> : (tensor<1x2500x8x2x1x4x2xf32>) -> tensor<1x2x2500x8x1x4x2xf32>
    return %result : tensor<1x2x2500x8x1x4x2xf32>
  }
  "onnx.EntryPoint"() {func = @transpose_rank7} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2500x8x2x1x4x2xf32>)
// CHECK: %[[REDUCED:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2500x8x2x1x4x2xf32>, tensor<5xi32>) -> tensor<2500x8x2x4x2xf32>
// CHECK: "tfl.transpose"(%[[REDUCED]], {{.*}}) : (tensor<2500x8x2x4x2xf32>, tensor<5xi32>) -> tensor<2x2500x8x4x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @add_rank7(%lhs: tensor<6x2500x1x1x1x4x2xf32>, %rhs: tensor<6x2500x8x1x2x4x2xf32>) -> tensor<6x2500x8x1x2x4x2xf32> {
    %result = "onnx.Add"(%lhs, %rhs) : (tensor<6x2500x1x1x1x4x2xf32>, tensor<6x2500x8x1x2x4x2xf32>) -> tensor<6x2500x8x1x2x4x2xf32>
    return %result : tensor<6x2500x8x1x2x4x2xf32>
  }
  "onnx.EntryPoint"() {func = @add_rank7} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<6x2500x1x1x1x4x2xf32>, %arg1: tensor<6x2500x8x1x2x4x2xf32>)
// CHECK: %[[LHS:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<6x2500x1x1x1x4x2xf32>, tensor<3xi32>) -> tensor<15000x1x8xf32>
// CHECK: %[[RHS:.*]] = "tfl.reshape"(%arg1, {{.*}}) : (tensor<6x2500x8x1x2x4x2xf32>, tensor<3xi32>) -> tensor<15000x16x8xf32>
// CHECK: "tfl.add"(%[[LHS]], %[[RHS]]) {{.*}} : (tensor<15000x1x8xf32>, tensor<15000x16x8xf32>) -> tensor<15000x16x8xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @transpose_bool(%lhs: tensor<6x1x2500xf32>, %rhs: tensor<6x1x2500xf32>) -> tensor<1x2500x6xf32> {
    %condition = "onnx.Less"(%lhs, %rhs) : (tensor<6x1x2500xf32>, tensor<6x1x2500xf32>) -> tensor<6x1x2500xi1>
    %transposed = "onnx.Transpose"(%condition) <{perm = [1, 2, 0]}> : (tensor<6x1x2500xi1>) -> tensor<1x2500x6xi1>
    %wide = "onnx.Cast"(%transposed) <{saturate = 1 : si64, to = i64}> : (tensor<1x2500x6xi1>) -> tensor<1x2500x6xi64>
    %result = "onnx.Cast"(%wide) <{saturate = 1 : si64, to = f32}> : (tensor<1x2500x6xi64>) -> tensor<1x2500x6xf32>
    return %result : tensor<1x2500x6xf32>
  }
  "onnx.EntryPoint"() {func = @transpose_bool} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<6x1x2500xf32>, %arg1: tensor<6x1x2500xf32>)
// CHECK: %[[COND:.*]] = "tfl.less"(%arg0, %arg1) : (tensor<6x1x2500xf32>, tensor<6x1x2500xf32>) -> tensor<6x1x2500xi1>
// CHECK: %[[TRANSPOSED:.*]] = "tfl.transpose"(%[[COND]], {{.*}}) : (tensor<6x1x2500xi1>, tensor<3xi32>) -> tensor<1x2500x6xi1>
// CHECK: "tfl.cast"(%[[TRANSPOSED]]) : (tensor<1x2500x6xi1>) -> tensor<1x2500x6xi64>
// CHECK-NOT: onnx.
