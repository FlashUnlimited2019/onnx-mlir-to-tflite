// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// Right-aligned vector broadcasting must follow the physical permutation.
module {
  func.func @broadcast_vector(%x: tensor<2x3xf32>, %bias: tensor<2xf32>) -> tensor<2x3xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.sub"(%a, %bias) {fused_activation_function = "RELU6"} : (tensor<3x2xf32>, tensor<2xf32>) -> tensor<3x2xf32>
    %y = "tfl.transpose"(%b, %p) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @broadcast_vector} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: %[[S:.*]] = arith.constant dense<[2, 1]> : tensor<2xi32>
// CHECK: %[[V:.*]] = "tfl.reshape"(%arg1, %[[S]])
// CHECK: "tfl.sub"(%arg0, %[[V]]) {fused_activation_function = "RELU6"}
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// Repeated operands have multiple uses but only one producer to remove.
module {
  func.func @repeated_operand(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.mul"(%a, %a) {fused_activation_function = "NONE"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    %y = "tfl.transpose"(%b, %p) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @repeated_operand} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.mul"(%arg0, %arg0)
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// A shared computation must not be cloned to remove its output transpose.
module {
  func.func @shared_computation(%x: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<3x2xf32>) {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.relu"(%a) : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %y = "tfl.transpose"(%b, %p) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    return %y, %b : tensor<2x3xf32>, tensor<3x2xf32>
  }
  "onnx.EntryPoint"() {func = @shared_computation} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.relu"
// CHECK: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @unit_views(%x: tensor<2x1x3xf32>, %bias: tensor<3xf32>) -> tensor<2x1x3xf32> {
    %s = arith.constant dense<[2, 3]> : tensor<2xi32>
    %t = arith.constant dense<[2, 1, 3]> : tensor<3xi32>
    %a = "tfl.reshape"(%x, %s) : (tensor<2x1x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    %b = "tfl.add"(%a, %bias) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
    %y = "tfl.reshape"(%b, %t) : (tensor<2x3xf32>, tensor<3xi32>) -> tensor<2x1x3xf32>
    return %y : tensor<2x1x3xf32>
  }
  "onnx.EntryPoint"() {func = @unit_views} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[B:.*]] = "tfl.reshape"(%arg1,
// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.add"(%arg0, %[[B]])
// CHECK-NOT: "tfl.reshape"
// CHECK: return

// -----

// Remap both sides of asymmetric padding, preserving a nonzero fill value.
module {
  func.func @pad_axes(%x: tensor<2x3xf32>) -> tensor<3x6xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %pad = arith.constant dense<[[1, 2], [0, 1]]> : tensor<2x2xi32>
    %fill = arith.constant dense<-2.5> : tensor<f32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.padv2"(%a, %pad, %fill) : (tensor<3x2xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<6x3xf32>
    %y = "tfl.transpose"(%b, %p) : (tensor<6x3xf32>, tensor<2xi32>) -> tensor<3x6xf32>
    return %y : tensor<3x6xf32>
  }
  "onnx.EntryPoint"() {func = @pad_axes} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: %[[P:.*]] = arith.constant dense<{{\[\[0, 1\], \[1, 2\]\]}}> : tensor<2x2xi32>
// CHECK: "tfl.padv2"(%arg0, %[[P]],
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// Negative paddings are outside the supported Pad semantics; do not hide them.
module {
  func.func @negative_pad(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %pad = arith.constant dense<[[-1, 1], [0, 0]]> : tensor<2x2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.pad"(%a, %pad) : (tensor<3x2xf32>, tensor<2x2xi32>) -> tensor<3x2xf32>
    %y = "tfl.transpose"(%b, %p) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @negative_pad} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.pad"
// CHECK: "tfl.transpose"
// CHECK: return
