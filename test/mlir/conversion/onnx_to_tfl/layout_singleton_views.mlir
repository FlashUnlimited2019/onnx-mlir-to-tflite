// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// Physical-layout fixtures cover singleton-only copies and conservative guards.
module {
  func.func @unit_transpose_views(%x: tensor<1x2x2xi32>, %y: tensor<1x2x2xi32>) -> tensor<2x2x2x1xi32> {
    %s = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi32>
    %p = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %a = "tfl.reshape"(%x, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %b = "tfl.reshape"(%y, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %c = "tfl.transpose"(%a, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %d = "tfl.transpose"(%b, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %e = "tfl.concatenation"(%c, %d) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x2x1xi32>, tensor<1x2x2x1xi32>) -> tensor<2x2x2x1xi32>
    return %e : tensor<2x2x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @unit_transpose_views} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.reshape"
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.concatenation"(%arg0, %arg1) {{.*}} -> tensor<2x2x2xi32>
// CHECK: "tfl.reshape"
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @real_transpose_views(%x: tensor<1x2x2xi32>, %y: tensor<1x2x2xi32>) -> tensor<2x2x2x1xi32> {
    %s = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi32>
    %p = arith.constant dense<[0, 3, 2, 1]> : tensor<4xi32>
    %a = "tfl.reshape"(%x, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %b = "tfl.reshape"(%y, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %c = "tfl.transpose"(%a, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %d = "tfl.transpose"(%b, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %e = "tfl.concatenation"(%c, %d) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x2x1xi32>, tensor<1x2x2x1xi32>) -> tensor<2x2x2x1xi32>
    return %e : tensor<2x2x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @real_transpose_views} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"
// CHECK: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @shared_unit_views(%x: tensor<1x2x2xi32>, %y: tensor<1x2x2xi32>) -> (tensor<2x2x2x1xi32>, tensor<1x1x2x2xi32>) {
    %s = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi32>
    %p = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %a = "tfl.reshape"(%x, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %b = "tfl.reshape"(%y, %s) : (tensor<1x2x2xi32>, tensor<4xi32>) -> tensor<1x1x2x2xi32>
    %c = "tfl.transpose"(%a, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %d = "tfl.transpose"(%b, %p) : (tensor<1x1x2x2xi32>, tensor<4xi32>) -> tensor<1x2x2x1xi32>
    %e = "tfl.concatenation"(%c, %d) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x2x1xi32>, tensor<1x2x2x1xi32>) -> tensor<2x2x2x1xi32>
    return %e, %a : tensor<2x2x2x1xi32>, tensor<1x1x2x2xi32>
  }
  "onnx.EntryPoint"() {func = @shared_unit_views} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"
// CHECK: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @lower_transpose(%x: tensor<1x2x3x4x5xi32>) -> tensor<2x4x5x3xi32> {
    %p = arith.constant dense<[1, 3, 4, 2, 0]> : tensor<5xi32>
    %s = arith.constant dense<[2, 4, 5, 3]> : tensor<4xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<1x2x3x4x5xi32>, tensor<5xi32>) -> tensor<2x4x5x3x1xi32>
    %b = "tfl.reshape"(%a, %s) : (tensor<2x4x5x3x1xi32>, tensor<4xi32>) -> tensor<2x4x5x3xi32>
    return %b : tensor<2x4x5x3xi32>
  }
  "onnx.EntryPoint"() {func = @lower_transpose} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x4x5xi32>, tensor<4xi32>) -> tensor<2x4x5x3xi32>
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @equal_axes(%x: tensor<1x2x2xi32>) -> tensor<2x2xi32> {
    %p = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
    %s = arith.constant dense<[2, 2]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<1x2x2xi32>, tensor<3xi32>) -> tensor<2x2x1xi32>
    %b = "tfl.reshape"(%a, %s) : (tensor<2x2x1xi32>, tensor<2xi32>) -> tensor<2x2xi32>
    return %b : tensor<2x2xi32>
  }
  "onnx.EntryPoint"() {func = @equal_axes} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @shared_transpose(%x: tensor<1x2x3x4x5xi32>) -> (tensor<2x4x5x3xi32>, tensor<2x4x5x3x1xi32>) {
    %p = arith.constant dense<[1, 3, 4, 2, 0]> : tensor<5xi32>
    %s = arith.constant dense<[2, 4, 5, 3]> : tensor<4xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<1x2x3x4x5xi32>, tensor<5xi32>) -> tensor<2x4x5x3x1xi32>
    %b = "tfl.reshape"(%a, %s) : (tensor<2x4x5x3x1xi32>, tensor<4xi32>) -> tensor<2x4x5x3xi32>
    return %b, %a : tensor<2x4x5x3xi32>, tensor<2x4x5x3x1xi32>
  }
  "onnx.EntryPoint"() {func = @shared_transpose} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"(%arg0,
// CHECK: "tfl.reshape"
// CHECK: return

// -----

module {
  func.func @nonunit_merge(%x: tensor<1x2x3x4x5xi32>) -> tensor<8x5x3xi32> {
    %p = arith.constant dense<[1, 3, 4, 2, 0]> : tensor<5xi32>
    %s = arith.constant dense<[8, 5, 3]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<1x2x3x4x5xi32>, tensor<5xi32>) -> tensor<2x4x5x3x1xi32>
    %b = "tfl.reshape"(%a, %s) : (tensor<2x4x5x3x1xi32>, tensor<3xi32>) -> tensor<8x5x3xi32>
    return %b : tensor<8x5x3xi32>
  }
  "onnx.EntryPoint"() {func = @nonunit_merge} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"(%arg0,
// CHECK: "tfl.reshape"
// CHECK: return

// -----

module {
  func.func @invalid_target(%x: tensor<1x2x3xi32>) -> tensor<3x2x1xi32> {
    %p = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
    %s = arith.constant dense<[3, 2, 2]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<1x2x3xi32>, tensor<3xi32>) -> tensor<3x2x1xi32>
    %b = "tfl.reshape"(%a, %s) : (tensor<3x2x1xi32>, tensor<3xi32>) -> tensor<3x2x1xi32>
    return %b : tensor<3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @invalid_target} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"(%arg0,
// CHECK: "tfl.reshape"
// CHECK: return

// -----

module {
  func.func @concat_unit_axis(%x0: tensor<1x3x2xi32>, %x1: tensor<1x3x2xi32>) -> tensor<2x3x2x1xi32> {
    %s0 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %s1 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x3x2x1xi32>, tensor<1x3x2x1xi32>) -> tensor<2x3x2x1xi32>
    return %b : tensor<2x3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @concat_unit_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.concatenation"(%arg0, %arg1)
// CHECK: "tfl.reshape"
// CHECK: return

// -----

module {
  func.func @concat_mapped_axis(%x0: tensor<2x3xi32>, %x1: tensor<2x5xi32>) -> tensor<1x2x8xi32> {
    %s0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<1x2x3xi32>
    %s1 = arith.constant dense<[1, 2, 5]> : tensor<3xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<2x5xi32>, tensor<3xi32>) -> tensor<1x2x5xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x5xi32>) -> tensor<1x2x8xi32>
    return %b : tensor<1x2x8xi32>
  }
  "onnx.EntryPoint"() {func = @concat_mapped_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.concatenation"(%arg0, %arg1)
// CHECK: "tfl.reshape"
// CHECK: return

// -----

module {
  func.func @concat_shared(%x0: tensor<1x3x2xi32>, %x1: tensor<1x3x2xi32>) -> (tensor<2x3x2x1xi32>, tensor<1x3x2x1xi32>) {
    %s0 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %s1 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x3x2x1xi32>, tensor<1x3x2x1xi32>) -> tensor<2x3x2x1xi32>
    return %b, %a0 : tensor<2x3x2x1xi32>, tensor<1x3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @concat_shared} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return
// -----

module {
  func.func @concat_new_axis(%x0: tensor<3x2xi32>, %x1: tensor<3x2xi32>) -> tensor<2x3x2xi32> {
    %s0 = arith.constant dense<[1, 3, 2]> : tensor<3xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<3x2xi32>, tensor<3xi32>) -> tensor<1x3x2xi32>
    %s1 = arith.constant dense<[1, 3, 2]> : tensor<3xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<3x2xi32>, tensor<3xi32>) -> tensor<1x3x2xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x3x2xi32>, tensor<1x3x2xi32>) -> tensor<2x3x2xi32>
    return %b : tensor<2x3x2xi32>
  }
  "onnx.EntryPoint"() {func = @concat_new_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @concat_merge(%x0: tensor<2x3xi32>, %x1: tensor<2x3xi32>) -> tensor<2x6xi32> {
    %s0 = arith.constant dense<[1, 6]> : tensor<2xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<1x6xi32>
    %s1 = arith.constant dense<[1, 6]> : tensor<2xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<1x6xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x6xi32>, tensor<1x6xi32>) -> tensor<2x6xi32>
    return %b : tensor<2x6xi32>
  }
  "onnx.EntryPoint"() {func = @concat_merge} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @concat_fused_activation(%x0: tensor<1x3x2xi32>, %x1: tensor<1x3x2xi32>) -> tensor<2x3x2x1xi32> {
    %s0 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %s1 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "RELU"} : (tensor<1x3x2x1xi32>, tensor<1x3x2x1xi32>) -> tensor<2x3x2x1xi32>
    return %b : tensor<2x3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @concat_fused_activation} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @concat_runtime_shape(%x0: tensor<1x3x2xi32>, %x1: tensor<1x3x2xi32>, %s: tensor<4xi32>) -> tensor<2x3x2x1xi32> {
    %a0 = "tfl.reshape"(%x0, %s) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %a1 = "tfl.reshape"(%x1, %s) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x3x2x1xi32>, tensor<1x3x2x1xi32>) -> tensor<2x3x2x1xi32>
    return %b : tensor<2x3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @concat_runtime_shape} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return

// -----

module {
  func.func @concat_invalid_result(%x0: tensor<1x3x2xi32>, %x1: tensor<1x3x2xi32>) -> tensor<3x3x2x1xi32> {
    %s0 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a0 = "tfl.reshape"(%x0, %s0) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %s1 = arith.constant dense<[1, 3, 2, 1]> : tensor<4xi32>
    %a1 = "tfl.reshape"(%x1, %s1) : (tensor<1x3x2xi32>, tensor<4xi32>) -> tensor<1x3x2x1xi32>
    %b = "tfl.concatenation"(%a0, %a1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x3x2x1xi32>, tensor<1x3x2x1xi32>) -> tensor<3x3x2x1xi32>
    return %b : tensor<3x3x2x1xi32>
  }
  "onnx.EntryPoint"() {func = @concat_invalid_result} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0,
// CHECK: "tfl.reshape"(%arg1,
// CHECK: "tfl.concatenation"
// CHECK: return
