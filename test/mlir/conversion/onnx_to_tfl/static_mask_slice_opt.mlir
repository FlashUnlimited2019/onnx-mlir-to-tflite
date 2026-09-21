// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @concat_windows(%cache: tensor<1x2x3xf32>, %current: tensor<1x2x5xf32>) -> (tensor<1x2x5xf32>, tensor<1x2x2xf32>, tensor<1x2x8xf32>) {
    %all = "tfl.concatenation"(%cache, %current) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xf32>, tensor<1x2x5xf32>) -> tensor<1x2x8xf32>
    %begin = arith.constant dense<[0, 0, 3]> : tensor<3xi32>
    %size = arith.constant dense<[1, 2, 5]> : tensor<3xi32>
    %tail = arith.constant dense<[0, 0, 6]> : tensor<3xi32>
    %rest = arith.constant dense<[1, 2, -1]> : tensor<3xi32>
    %a = "tfl.slice"(%all, %begin, %size) : (tensor<1x2x8xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x5xf32>
    %b = "tfl.slice"(%all, %tail, %rest) : (tensor<1x2x8xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x2xf32>
    return %a, %b, %all : tensor<1x2x5xf32>, tensor<1x2x2xf32>, tensor<1x2x8xf32>
  }
  "onnx.EntryPoint"() {func = @concat_windows} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[ALL:.*]] = "tfl.concatenation"(%arg0, %arg1)
// CHECK: %[[BEGIN:.*]] = arith.constant dense<[0, 0, 3]> : tensor<3xi32>
// CHECK: %[[TAIL:.*]] = "tfl.slice"(%arg1, %[[BEGIN]], {{.*}})
// CHECK-NOT: "tfl.slice"
// CHECK: return %arg1, %[[TAIL]], %[[ALL]]

// -----

module {
  func.func @constant_mask_suffix(%history: tensor<1x4xi1>, %current: tensor<1x3x2xf32>) -> tensor<1x3x2xf32> {
    %false = arith.constant dense<false> : tensor<1x4xi1>
    %all = "tfl.concatenation"(%history, %false) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x4xi1>, tensor<1x4xi1>) -> tensor<1x8xi1>
    %begin = arith.constant dense<[0, 0]> : tensor<2xi32>
    %end = arith.constant dense<[1, 8]> : tensor<2xi32>
    %step = arith.constant dense<[1, 2]> : tensor<2xi32>
    %down = "tfl.strided_slice"(%all, %begin, %end, %step) {begin_mask = 0 : i32, end_mask = 0 : i32, ellipsis_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<1x8xi1>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xi1>
    %tail = arith.constant dense<[0, 2]> : tensor<2xi32>
    %size = arith.constant dense<[1, 2]> : tensor<2xi32>
    %part = "tfl.slice"(%down, %tail, %size) : (tensor<1x4xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi1>
    %shape = arith.constant dense<[1, 1, 2]> : tensor<3xi32>
    %r = "tfl.reshape"(%part, %shape) : (tensor<1x2xi1>, tensor<3xi32>) -> tensor<1x1x2xi1>
    %expanded = arith.constant dense<[1, 3, 2]> : tensor<3xi32>
    %mask = "tfl.broadcast_to"(%r, %expanded) : (tensor<1x1x2xi1>, tensor<3xi32>) -> tensor<1x3x2xi1>
    %zero = arith.constant dense<0.0> : tensor<f32>
    %out = "tfl.select_v2"(%mask, %zero, %current) : (tensor<1x3x2xi1>, tensor<f32>, tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
    return %out : tensor<1x3x2xf32>
  }
  "onnx.EntryPoint"() {func = @constant_mask_suffix} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.
// CHECK: return %arg1

// -----

module {
  func.func @runtime_mask_is_preserved(%history: tensor<1x4xi1>, %current: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %false = arith.constant dense<false> : tensor<1x4xi1>
    %all = "tfl.concatenation"(%history, %false) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x4xi1>, tensor<1x4xi1>) -> tensor<1x8xi1>
    %begin = arith.constant dense<[0, 2]> : tensor<2xi32>
    %size = arith.constant dense<[1, 4]> : tensor<2xi32>
    %mask = "tfl.slice"(%all, %begin, %size) : (tensor<1x8xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xi1>
    %zero = arith.constant dense<0.0> : tensor<f32>
    %out = "tfl.select_v2"(%mask, %zero, %current) : (tensor<1x4xi1>, tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    return %out : tensor<1x4xf32>
  }
  "onnx.EntryPoint"() {func = @runtime_mask_is_preserved} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.concatenation"
// CHECK: "tfl.slice"
// CHECK: "tfl.select_v2"
// CHECK: return

// -----

module {
  func.func @selected_branch_requires_broadcast(%input: tensor<2xf32>) -> tensor<2xf32> {
    %condition = arith.constant dense<false> : tensor<2xi1>
    %scalar = arith.constant dense<7.0> : tensor<f32>
    %out = "tfl.select_v2"(%condition, %input, %scalar) : (tensor<2xi1>, tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
    return %out : tensor<2xf32>
  }
  "onnx.EntryPoint"() {func = @selected_branch_requires_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.select_v2"
// CHECK: return

// -----

module {
  func.func @concat_activation_is_preserved(%a: tensor<2xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
    %all = "tfl.concatenation"(%a, %b) {axis = 0 : i32, fused_activation_function = "RELU"} : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
    %start = arith.constant dense<2> : tensor<1xi32>
    %size = arith.constant dense<3> : tensor<1xi32>
    %out = "tfl.slice"(%all, %start, %size) : (tensor<5xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xf32>
    return %out : tensor<3xf32>
  }
  "onnx.EntryPoint"() {func = @concat_activation_is_preserved} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.concatenation"
// CHECK: "tfl.slice"
// CHECK: return
