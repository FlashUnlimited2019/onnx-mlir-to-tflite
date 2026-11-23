// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// Physical i32 broadcasts are accepted without narrowing dynamic indices.
module {
  func.func @shared_rows(%x: tensor<2x7x5xf32>, %i: tensor<1x4x1xi32>) -> tensor<2x4x5xf32> {
    %s = arith.constant dense<[2, 4, 5]> : tensor<3xi32>
    %e = "tfl.broadcast_to"(%i, %s) : (tensor<1x4x1xi32>, tensor<3xi32>) -> tensor<2x4x5xi32>
    %y = "onnx.GatherElements"(%x, %e) {axis = 1 : si64} : (tensor<2x7x5xf32>, tensor<2x4x5xi32>) -> tensor<2x4x5xf32>
    return %y : tensor<2x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @shared_rows} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather_nd"
// CHECK: "tfl.less"{{.*}} -> tensor<4xi1>
// CHECK: "tfl.gather"{{.*}} {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<2x7x5xf32>, tensor<4xi32>) -> tensor<2x4x5xf32>
// CHECK-NOT: "tfl.gather_nd"
// CHECK: return

// -----

// Prevent overflow in the unselected Add branch of i32 index normalization.
module {
  func.func @large_axis(%x: tensor<1x1500000000x2xf32>, %i: tensor<1x1x1xi32>) -> tensor<1x1x2xf32> {
    %s = arith.constant dense<[1, 1, 2]> : tensor<3xi32>
    %e = "tfl.broadcast_to"(%i, %s) : (tensor<1x1x1xi32>, tensor<3xi32>) -> tensor<1x1x2xi32>
    %y = "onnx.GatherElements"(%x, %e) {axis = 1 : si64} : (tensor<1x1500000000x2xf32>, tensor<1x1x2xi32>) -> tensor<1x1x2xf32>
    return %y : tensor<1x1x2xf32>
  }
  "onnx.EntryPoint"() {func = @large_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[SAFE:.*]] = "tfl.minimum"{{.*}} -> tensor<1xi32>
// CHECK: "tfl.add"(%[[SAFE]],
// CHECK: "tfl.gather"{{.*}} {axis = 1 : i32, batch_dims = 0 : i32}
// CHECK: return

// -----

module {
  func.func @batched_rows(%x: tensor<2x7x5xi64>, %i: tensor<2x4x1xi64>) -> tensor<2x4x5xi64> {
    %s = onnx.Constant dense<[2, 4, 5]> : tensor<3xi64>
    %e = "onnx.Expand"(%i, %s) : (tensor<2x4x1xi64>, tensor<3xi64>) -> tensor<2x4x5xi64>
    %y = "onnx.GatherElements"(%x, %e) {axis = -2 : si64} : (tensor<2x7x5xi64>, tensor<2x4x5xi64>) -> tensor<2x4x5xi64>
    return %y : tensor<2x4x5xi64>
  }
  "onnx.EntryPoint"() {func = @batched_rows} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"{{.*}} {axis = 1 : i32, batch_dims = 1 : i32} : (tensor<2x7x5xi64>, tensor<2x4xi64>) -> tensor<2x4x5xi64>
// CHECK-NOT: "tfl.gather_nd"
// CHECK: return

// -----

// Partial trailing slices must retain element-wise indexing.
module {
  func.func @partial_rows(%x: tensor<2x7x5xf32>, %i: tensor<2x4x1xi64>) -> tensor<2x4x3xf32> {
    %s = onnx.Constant dense<[2, 4, 3]> : tensor<3xi64>
    %e = "onnx.Expand"(%i, %s) : (tensor<2x4x1xi64>, tensor<3xi64>) -> tensor<2x4x3xi64>
    %y = "onnx.GatherElements"(%x, %e) {axis = 1 : si64} : (tensor<2x7x5xf32>, tensor<2x4x3xi64>) -> tensor<2x4x3xf32>
    return %y : tensor<2x4x3xf32>
  }
  "onnx.EntryPoint"() {func = @partial_rows} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: return

// -----

module {
  func.func @mask_views(%x: tensor<5x1x3xf32>, %m: tensor<1x1x5xi1>, %z: tensor<f32>) -> tensor<1x5x3xf32> {
    %p = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %q = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %s = arith.constant dense<[1, 3, 5]> : tensor<3xi32>
    %t = "tfl.transpose"(%x, %p) : (tensor<5x1x3xf32>, tensor<3xi32>) -> tensor<1x3x5xf32>
    %b = "tfl.broadcast_to"(%m, %s) : (tensor<1x1x5xi1>, tensor<3xi32>) -> tensor<1x3x5xi1>
    %w = "tfl.select_v2"(%b, %z, %t) : (tensor<1x3x5xi1>, tensor<f32>, tensor<1x3x5xf32>) -> tensor<1x3x5xf32>
    %y = "tfl.transpose"(%w, %q) : (tensor<1x3x5xf32>, tensor<3xi32>) -> tensor<1x5x3xf32>
    return %y : tensor<1x5x3xf32>
  }
  "onnx.EntryPoint"() {func = @mask_views} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: "tfl.broadcast_to"
// CHECK: "tfl.select_v2"{{.*}} : (tensor<1x5x1xi1>, tensor<f32>, tensor<1x5x3xf32>) -> tensor<1x5x3xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// A full mask would require a new physical permutation; leave it unchanged.
module {
  func.func @mask_copy(%x: tensor<2x3x5xf32>, %m: tensor<2x5x3xi1>, %z: tensor<f32>) -> tensor<2x3x5xf32> {
    %p = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %t = "tfl.transpose"(%x, %p) : (tensor<2x3x5xf32>, tensor<3xi32>) -> tensor<2x5x3xf32>
    %w = "tfl.select_v2"(%m, %z, %t) : (tensor<2x5x3xi1>, tensor<f32>, tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
    %y = "tfl.transpose"(%w, %p) : (tensor<2x5x3xf32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    return %y : tensor<2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @mask_copy} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.select_v2"
// CHECK: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @singleton(%x: tensor<3x1x5xi64>) -> tensor<3x5xi64> {
    %i = arith.constant dense<0> : tensor<i32>
    %y = "tfl.gather"(%x, %i) {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<3x1x5xi64>, tensor<i32>) -> tensor<3x5xi64>
    return %y : tensor<3x5xi64>
  }
  "onnx.EntryPoint"() {func = @singleton} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather"
// CHECK: "tfl.reshape"{{.*}} -> tensor<3x5xi64>
// CHECK-NOT: "tfl.gather"
// CHECK: return

// -----

// Do not turn out-of-range or unchecked runtime indices into valid views.
module {
  func.func @singleton_guards(%x: tensor<3x1x5xi64>, %runtime: tensor<i32>) -> (tensor<3x5xi64>, tensor<3x5xi64>) {
    %i = arith.constant dense<1> : tensor<i32>
    %a = "tfl.gather"(%x, %i) {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<3x1x5xi64>, tensor<i32>) -> tensor<3x5xi64>
    %b = "tfl.gather"(%x, %runtime) {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<3x1x5xi64>, tensor<i32>) -> tensor<3x5xi64>
    return %a, %b : tensor<3x5xi64>, tensor<3x5xi64>
  }
  "onnx.EntryPoint"() {func = @singleton_guards} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"
// CHECK: "tfl.gather"
// CHECK: return

// -----

module {
  func.func @bounded_div(%b: tensor<i64>) -> tensor<4xi64> {
    %a = onnx.Constant dense<[0, 1, 31, 2047]> : tensor<4xi64>
    %one = onnx.Constant dense<1> : tensor<i64>
    %positive = "onnx.Max"(%b, %one) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %y = "onnx.Div"(%a, %positive) : (tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
    return %y : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @bounded_div} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<2048> : tensor<i64>
// CHECK: "tfl.minimum"{{.*}} -> tensor<i64>
// CHECK: "tfl.div"{{.*}} -> tensor<4xi32>
// CHECK: "tfl.cast"{{.*}} : (tensor<4xi32>) -> tensor<4xi64>
// CHECK-NOT: tensor<4xf32>
// CHECK: return

// -----

// A dynamic divisor without a proven positive lower bound keeps the exact
// general integer-division algorithm, including negative dividend handling.
module {
  func.func @unproven_div(%b: tensor<i64>) -> tensor<4xi64> {
    %a = onnx.Constant dense<[-1, 0, 31, 2047]> : tensor<4xi64>
    %y = "onnx.Div"(%a, %b) : (tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
    return %y : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @unproven_div} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.div"{{.*}} -> tensor<4xf32>
// CHECK: "tfl.mul"{{.*}} -> tensor<4xi64>
// CHECK: return
