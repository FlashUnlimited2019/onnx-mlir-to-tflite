// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @sub_constant_lhs_channel(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %weight = "onnx.Constant"() {value = dense<[[[[0.5]], [[1.5]], [[2.5]]]]> : tensor<1x3x1x1xf32>} : () -> tensor<1x3x1x1xf32>
    %0 = "onnx.Sub"(%weight, %arg0) : (tensor<1x3x1x1xf32>, tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @sub_constant_lhs_channel} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK: %[[WEIGHT:.*]] = arith.constant {{.*}} : tensor<1x1x1x3xf32>
// CHECK: "tfl.sub"(%[[WEIGHT]], %arg0) {fused_activation_function = "NONE"} : (tensor<1x1x1x3xf32>, tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>

// -----

module {
  func.func @sub_constant_rhs_spatial(%arg0: tensor<1x4x2x3xf32>) -> tensor<1x4x2x3xf32> {
    %weight = "onnx.Constant"() {value = dense<1.25> : tensor<1x1x2x3xf32>} : () -> tensor<1x1x2x3xf32>
    %0 = "onnx.Sub"(%arg0, %weight) : (tensor<1x4x2x3xf32>, tensor<1x1x2x3xf32>) -> tensor<1x4x2x3xf32>
    return %0 : tensor<1x4x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @sub_constant_rhs_spatial} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: %[[WEIGHT:.*]] = arith.constant {{.*}} : tensor<1x2x3x1xf32>
// CHECK: "tfl.sub"(%arg0, %[[WEIGHT]]) {fused_activation_function = "NONE"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x1xf32>) -> tensor<1x2x3x4xf32>

// -----

module {
  func.func @sub_constant_lhs_width(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %weight = "onnx.Constant"() {value = dense<[[[[0.0, 1.0, 2.0, 3.0, 4.0]]]]> : tensor<1x1x1x5xf32>} : () -> tensor<1x1x1x5xf32>
    %0 = "onnx.Sub"(%weight, %arg0) : (tensor<1x1x1x5xf32>, tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @sub_constant_lhs_width} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK: %[[WEIGHT:.*]] = arith.constant {{.*}} : tensor<1x1x5x1xf32>
// CHECK: "tfl.sub"(%[[WEIGHT]], %arg0) {fused_activation_function = "NONE"} : (tensor<1x1x5x1xf32>, tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
