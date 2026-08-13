// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @main_graph(%rank3: tensor<1x3x8xf32>, %rank4: tensor<1x3x4x5xf32>, %rank5: tensor<1x2x3x4x5xf32>) -> (tensor<1x3x8xf32>, tensor<1x3x4x5xf32>, tensor<1x2x3x4x5xf32>) {
    %exponent = onnx.Constant dense<1.500000e+00> : tensor<f32>
    %result3 = "onnx.Pow"(%rank3, %exponent) : (tensor<1x3x8xf32>, tensor<f32>) -> tensor<1x3x8xf32>
    %result4 = "onnx.Pow"(%rank4, %exponent) : (tensor<1x3x4x5xf32>, tensor<f32>) -> tensor<1x3x4x5xf32>
    %result5 = "onnx.Pow"(%rank5, %exponent) : (tensor<1x2x3x4x5xf32>, tensor<f32>) -> tensor<1x2x3x4x5xf32>
    return %result3, %result4, %result5 : tensor<1x3x8xf32>, tensor<1x3x4x5xf32>, tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.pow"{{.*}}tensor<1x3x8xf32>
// CHECK: "tfl.pow"{{.*}}tensor<1x4x5x3xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<120xf32>
// CHECK: "tfl.pow"{{.*}}tensor<120xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<1x2x3x4x5xf32>
// CHECK-NOT: onnx.
