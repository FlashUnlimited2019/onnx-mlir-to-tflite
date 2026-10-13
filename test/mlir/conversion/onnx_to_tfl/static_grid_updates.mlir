// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl='layout-mode=experimental' --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @grid_channel_update(%data: tensor<2x3x2xf32>, %updates: tensor<6xf32>) -> tensor<2x3x2xf32> {
    %i = arith.constant dense<[[0, 0, 1], [0, 1, 1], [0, 2, 1], [1, 0, 1], [1, 1, 1], [1, 2, 1]]> : tensor<6x3xi64>
    %r = "onnx.ScatterND"(%data, %i, %updates) <{reduction = "none"}> : (tensor<2x3x2xf32>, tensor<6x3xi64>, tensor<6xf32>) -> tensor<2x3x2xf32>
    return %r : tensor<2x3x2xf32>
  }
  "onnx.EntryPoint"() {func = @grid_channel_update} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg1, {{.*}}) {{.*}} -> tensor<2x3x1xf32>
// CHECK: "tfl.slice"(%arg0, {{.*}}) {{.*}} -> tensor<2x3x1xf32>
// CHECK: "tfl.concatenation"{{.*}} {axis = 2 : i32, fused_activation_function = "NONE"}
// CHECK-NEXT: return

// -----

module {
  func.func @grid_full_update_trailing(%data: tensor<2x3x4xf32>, %updates: tensor<6x4xf32>) -> tensor<2x3x4xf32> {
    %i = arith.constant dense<[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]> : tensor<6x2xi64>
    %r = "onnx.ScatterND"(%data, %i, %updates) <{reduction = "none"}> : (tensor<2x3x4xf32>, tensor<6x2xi64>, tensor<6x4xf32>) -> tensor<2x3x4xf32>
    return %r : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @grid_full_update_trailing} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[R:.*]] = "tfl.reshape"(%arg1, {{.*}}) {{.*}} -> tensor<2x3x4xf32>
// CHECK-NEXT: return %[[R]]

// -----

module {
  func.func @grid_negative_gather(%data: tensor<2x3x2xf32>) -> tensor<2x2xf32> {
    %i = arith.constant dense<[[-2, -2], [-1, -2]]> : tensor<2x2xi64>
    %r = "onnx.GatherND"(%data, %i) <{batch_dims = 0 : si64}> : (tensor<2x3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }
  "onnx.EntryPoint"() {func = @grid_negative_gather} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.slice"(%arg0, {{.*}}) {{.*}} -> tensor<2x1x2xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x2xf32>
// CHECK-NEXT: return

// -----

module {
  func.func @grid_scrambled_update(%data: tensor<2x3xf32>, %updates: tensor<2xf32>) -> tensor<2x3xf32> {
    %i = arith.constant dense<[[1, 0], [0, 0]]> : tensor<2x2xi64>
    %r = "onnx.ScatterND"(%data, %i, %updates) <{reduction = "none"}> : (tensor<2x3xf32>, tensor<2x2xi64>, tensor<2xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @grid_scrambled_update} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"
// CHECK: "tfl.scatter_nd"

// -----

module {
  func.func @grid_reduce_add(%data: tensor<2x3xf32>, %updates: tensor<2xf32>) -> tensor<2x3xf32> {
    %i = arith.constant dense<[[0, 1], [1, 1]]> : tensor<2x2xi64>
    %r = "onnx.ScatterND"(%data, %i, %updates) <{reduction = "add"}> : (tensor<2x3xf32>, tensor<2x2xi64>, tensor<2xf32>) -> tensor<2x3xf32>
    return %r : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @grid_reduce_add} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.scatter_nd"

// -----

module {
  func.func @concat_middle_shared(%a: tensor<2x3xf32>, %b: tensor<4x3xf32>, %c: tensor<1x3xf32>) -> (tensor<7x3xf32>, tensor<2x3xf32>) {
    %v = "onnx.Concat"(%a, %b, %c) <{axis = -2 : si64}> : (tensor<2x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<7x3xf32>
    %start = arith.constant dense<3> : tensor<1xi64>
    %end = arith.constant dense<5> : tensor<1xi64>
    %axis = arith.constant dense<-2> : tensor<1xi64>
    %step = arith.constant dense<1> : tensor<1xi64>
    %r = "onnx.Slice"(%v, %start, %end, %axis, %step) : (tensor<7x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
    return %v, %r : tensor<7x3xf32>, tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @concat_middle_shared} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[C:.*]] = "tfl.concatenation"(%arg0, %arg1, %arg2)
// CHECK: %[[S:.*]] = "tfl.slice"(%arg1, {{.*}}) {{.*}} -> tensor<2x3xf32>
// CHECK-NEXT: return %[[C]], %[[S]]

// -----

// Duplicate Gather coordinates are legal and must retain their ordering.
module {
  func.func @constant_gather_duplicates(%data: tensor<2x3x4xf32>) -> tensor<3x4xf32> {
    %i = arith.constant dense<[[1, 2], [0, 0], [1, 2]]> : tensor<3x2xi32>
    %r = "tfl.gather_nd"(%data, %i) : (tensor<2x3x4xf32>, tensor<3x2xi32>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }
  "onnx.EntryPoint"() {func = @constant_gather_duplicates} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<[5, 0, 5]> : tensor<3xi32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) {{.*}} -> tensor<6x4xf32>
// CHECK: "tfl.gather"{{.*}} : (tensor<6x4xf32>, tensor<3xi32>) -> tensor<3x4xf32>
// CHECK-NEXT: return

// -----

// [0, 3] would have an in-range linear offset but an invalid second coordinate.
module {
  func.func @invalid_coordinate(%data: tensor<2x3xf32>) -> tensor<1xf32> {
    %i = arith.constant dense<[[0, 3]]> : tensor<1x2xi32>
    %r = "tfl.gather_nd"(%data, %i) : (tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @invalid_coordinate} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather"
// CHECK: "tfl.gather_nd"(%arg0, {{.*}})

// -----

module {
  func.func @negative_tfl_coordinate(%data: tensor<2x3xf32>) -> tensor<1xf32> {
    %i = arith.constant dense<[[1, -1]]> : tensor<1x2xi32>
    %r = "tfl.gather_nd"(%data, %i) : (tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @negative_tfl_coordinate} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather"
// CHECK: "tfl.gather_nd"(%arg0, {{.*}})

// -----

module {
  func.func @large_linear_dimension(%data: tensor<65536x65536xf32>) -> tensor<1xf32> {
    %i = arith.constant dense<[[0, 0]]> : tensor<1x2xi32>
    %r = "tfl.gather_nd"(%data, %i) : (tensor<65536x65536xf32>, tensor<1x2xi32>) -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @large_linear_dimension} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather"
// CHECK: "tfl.gather_nd"(%arg0, {{.*}})
