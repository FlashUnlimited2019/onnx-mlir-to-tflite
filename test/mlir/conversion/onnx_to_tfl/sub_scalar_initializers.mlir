// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @sub_scalar_lhs_rank2(%input: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %scalar = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    %result = "onnx.Sub"(%scalar, %input) : (tensor<f32>, tensor<1x16xf32>) -> tensor<1x16xf32>
    return %result : tensor<1x16xf32>
  }
  "onnx.EntryPoint"() {func = @sub_scalar_lhs_rank2} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x16xf32>)
// CHECK: %[[SCALAR:.*]] = arith.constant {{.*}} : tensor<f32>
// CHECK: "tfl.sub"(%[[SCALAR]], %arg0) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1x16xf32>) -> tensor<1x16xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @sub_scalar_rhs_rank1(%input: tensor<16xf32>) -> tensor<16xf32> {
    %scalar = "onnx.Constant"() {value = dense<-1.0> : tensor<f32>} : () -> tensor<f32>
    %result = "onnx.Sub"(%input, %scalar) : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
    return %result : tensor<16xf32>
  }
  "onnx.EntryPoint"() {func = @sub_scalar_rhs_rank1} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<16xf32>)
// CHECK: %[[SCALAR:.*]] = arith.constant {{.*}} : tensor<f32>
// CHECK: "tfl.sub"(%arg0, %[[SCALAR]]) {fused_activation_function = "NONE"} : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @sub_scalar_lhs_rank4(%input: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %scalar = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %result = "onnx.Sub"(%scalar, %input) : (tensor<f32>, tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @sub_scalar_lhs_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>)
// CHECK: %[[SCALAR:.*]] = arith.constant {{.*}} : tensor<f32>
// CHECK: "tfl.sub"(%[[SCALAR]], %arg0) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.
