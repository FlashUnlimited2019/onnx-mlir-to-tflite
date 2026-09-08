// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @scaled_softmax(%input: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
    %p0 = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %p1 = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
    %p2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %c = arith.constant dense<0.125> : tensor<f32>
    %a = "tfl.transpose"(%input, %p0) : (tensor<2x3x5xf32>, tensor<3xi32>) -> tensor<5x2x3xf32>
    %b = "tfl.mul"(%c, %a) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<5x2x3xf32>) -> tensor<5x2x3xf32>
    %d = "tfl.transpose"(%b, %p1) : (tensor<5x2x3xf32>, tensor<3xi32>) -> tensor<3x2x5xf32>
    %s = "tfl.softmax"(%d) {beta = 2.0 : f32} : (tensor<3x2x5xf32>) -> tensor<3x2x5xf32>
    %r = "tfl.transpose"(%s, %p2) : (tensor<3x2x5xf32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    return %r : tensor<2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @scaled_softmax} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: %[[M:.*]] = "tfl.mul"({{.*}}, %arg0)
// CHECK-NEXT: %[[S:.*]] = "tfl.softmax"(%[[M]]) {beta = 2.000000e+00 : f32}
// CHECK-NOT: "tfl.transpose"
// CHECK: return %[[S]]

// -----

module {
  func.func @shared_scaled_softmax_input(%input: tensor<2x3x5xf32>) -> (tensor<2x3x5xf32>, tensor<3x2x5xf32>) {
    %p = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %c = arith.constant dense<0.125> : tensor<f32>
    %a = "tfl.transpose"(%input, %p) : (tensor<2x3x5xf32>, tensor<3xi32>) -> tensor<3x2x5xf32>
    %b = "tfl.mul"(%a, %c) {fused_activation_function = "NONE"} : (tensor<3x2x5xf32>, tensor<f32>) -> tensor<3x2x5xf32>
    %s = "tfl.softmax"(%b) {beta = 1.0 : f32} : (tensor<3x2x5xf32>) -> tensor<3x2x5xf32>
    %r = "tfl.transpose"(%s, %p) : (tensor<3x2x5xf32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    return %r, %a : tensor<2x3x5xf32>, tensor<3x2x5xf32>
  }
  "onnx.EntryPoint"() {func = @shared_scaled_softmax_input} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.mul"
// CHECK: "tfl.softmax"
// CHECK: "tfl.transpose"

// -----

module {
  func.func @changed_normalization_axis(%input: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
    %p = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %c = arith.constant dense<0.125> : tensor<f32>
    %a = "tfl.transpose"(%input, %p) : (tensor<2x3x5xf32>, tensor<3xi32>) -> tensor<2x5x3xf32>
    %b = "tfl.mul"(%a, %c) {fused_activation_function = "NONE"} : (tensor<2x5x3xf32>, tensor<f32>) -> tensor<2x5x3xf32>
    %s = "tfl.softmax"(%b) {beta = 1.0 : f32} : (tensor<2x5x3xf32>) -> tensor<2x5x3xf32>
    %r = "tfl.transpose"(%s, %p) : (tensor<2x5x3xf32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    return %r : tensor<2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @changed_normalization_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.mul"
// CHECK: "tfl.softmax"
// CHECK: "tfl.transpose"

// -----

module {
  func.func @non_scalar_scale(%input: tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
    %p = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %c = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5]> : tensor<5xf32>
    %a = "tfl.transpose"(%input, %p) : (tensor<2x3x5xf32>, tensor<3xi32>) -> tensor<3x2x5xf32>
    %b = "tfl.mul"(%a, %c) {fused_activation_function = "NONE"} : (tensor<3x2x5xf32>, tensor<5xf32>) -> tensor<3x2x5xf32>
    %s = "tfl.softmax"(%b) {beta = 1.0 : f32} : (tensor<3x2x5xf32>) -> tensor<3x2x5xf32>
    %r = "tfl.transpose"(%s, %p) : (tensor<3x2x5xf32>, tensor<3xi32>) -> tensor<2x3x5xf32>
    return %r : tensor<2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @non_scalar_scale} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.mul"
// CHECK: "tfl.softmax"
// CHECK: "tfl.transpose"

// -----

module {
  func.func @unequal_split_bias(%input: tensor<2x3x5xf32>) -> (tensor<2x3x2xf32>, tensor<2x3x3xf32>) {
    %sizes = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %bias0 = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %bias1 = "onnx.Constant"() {value = dense<[[[1.0, 2.0, 3.0]]]> : tensor<1x1x3xf32>} : () -> tensor<1x1x3xf32>
    %a, %b = "onnx.Split"(%input, %sizes) {axis = -1 : si64} : (tensor<2x3x5xf32>, tensor<2xi64>) -> (tensor<2x3x2xf32>, tensor<2x3x3xf32>)
    %c = "onnx.Add"(%bias0, %a) : (tensor<f32>, tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
    %d = "onnx.Add"(%b, %bias1) : (tensor<2x3x3xf32>, tensor<1x1x3xf32>) -> tensor<2x3x3xf32>
    return %c, %d : tensor<2x3x2xf32>, tensor<2x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @unequal_split_bias} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<[2.500000e-01, 2.500000e-01, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<5xf32>
// CHECK: "tfl.add"(%arg0,
// CHECK-NOT: "tfl.add"
// CHECK: "tfl.split_v"
// CHECK-NOT: "tfl.add"
// CHECK: return

// -----

module {
  func.func @shared_split_output(%input: tensor<2x3x4xf32>) -> (tensor<2x3x2xf32>, tensor<2x3x2xf32>, tensor<2x3x2xf32>) {
    %sizes = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %bias = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %a, %b = "onnx.Split"(%input, %sizes) {axis = 2 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> (tensor<2x3x2xf32>, tensor<2x3x2xf32>)
    %c = "onnx.Add"(%a, %bias) : (tensor<2x3x2xf32>, tensor<2xf32>) -> tensor<2x3x2xf32>
    %d = "onnx.Add"(%b, %bias) : (tensor<2x3x2xf32>, tensor<2xf32>) -> tensor<2x3x2xf32>
    return %a, %c, %d : tensor<2x3x2xf32>, tensor<2x3x2xf32>, tensor<2x3x2xf32>
  }
  "onnx.EntryPoint"() {func = @shared_split_output} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.split"
// CHECK: "tfl.add"
// CHECK: "tfl.add"

// -----

module {
  func.func @tanh_gelu_outer_half(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %half = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %one = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %coeff = "onnx.Constant"() {value = dense<0.044715> : tensor<f32>} : () -> tensor<f32>
    %sqrt = "onnx.Constant"() {value = dense<0.797884583> : tensor<f32>} : () -> tensor<f32>
    %square = "onnx.Mul"(%input, %input) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %cube = "onnx.Mul"(%square, %input) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %a = "onnx.Mul"(%coeff, %cube) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %b = "onnx.Add"(%a, %input) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %c = "onnx.Mul"(%sqrt, %b) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %d = "onnx.Tanh"(%c) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %e = "onnx.Add"(%one, %d) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %f = "onnx.Mul"(%e, %input) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %g = "onnx.Mul"(%half, %f) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %g : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @tanh_gelu_outer_half} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.mul"
// CHECK-NOT: "tfl.tanh"
// CHECK: "tfl.gelu"(%arg0) {approximate = true}
// CHECK-NOT: "tfl.mul"
// CHECK-NOT: "tfl.tanh"
// CHECK: return
