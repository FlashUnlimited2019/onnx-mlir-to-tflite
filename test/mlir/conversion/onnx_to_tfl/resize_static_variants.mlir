// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @resize_linear_antialias(%input: tensor<1x2x2x4x6xf32>) -> tensor<1x2x1x3x5xf32> {
    %none = "onnx.NoValue"() : () -> none
    %sizes = arith.constant dense<[1, 3, 5]> : tensor<3xi64>
    %result = "onnx.Resize"(%input, %none, %none, %sizes) <{antialias = 1 : si64, axes = [2, 3, 4], coordinate_transformation_mode = "half_pixel", keep_aspect_ratio_policy = "stretch", mode = "linear"}> : (tensor<1x2x2x4x6xf32>, none, none, tensor<3xi64>) -> tensor<1x2x1x3x5xf32>
    return %result : tensor<1x2x1x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @resize_linear_antialias} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: "tfl.batch_matmul"
// CHECK: tensor<1x2x1x3x5xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @resize_cubic_antialias(%input: tensor<1x2x8x10xf32>) -> tensor<1x2x3x5xf32> {
    %none = "onnx.NoValue"() : () -> none
    %sizes = arith.constant dense<[3, 5]> : tensor<2xi64>
    %result = "onnx.Resize"(%input, %none, %none, %sizes) <{antialias = 1 : si64, axes = [2, 3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -5.000000e-01 : f32, exclude_outside = 1 : si64, keep_aspect_ratio_policy = "stretch", mode = "cubic"}> : (tensor<1x2x8x10xf32>, none, none, tensor<2xi64>) -> tensor<1x2x3x5xf32>
    return %result : tensor<1x2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @resize_cubic_antialias} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: "tfl.batch_matmul"
// CHECK: tensor<1x3x5x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @resize_crop_and_extrapolate(%input: tensor<1x2x4x6xf32>) -> tensor<1x2x3x5xf32> {
    %roi = arith.constant dense<[-2.500000e-01, 1.000000e-01, 1.250000e+00, 9.000000e-01]> : tensor<4xf32>
    %none = "onnx.NoValue"() : () -> none
    %sizes = arith.constant dense<[3, 5]> : tensor<2xi64>
    %result = "onnx.Resize"(%input, %roi, %none, %sizes) <{axes = [2, 3], coordinate_transformation_mode = "tf_crop_and_resize", extrapolation_value = 1.250000e-01 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear"}> : (tensor<1x2x4x6xf32>, tensor<4xf32>, none, tensor<2xi64>) -> tensor<1x2x3x5xf32>
    return %result : tensor<1x2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @resize_crop_and_extrapolate} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.add"
// CHECK: "tfl.batch_matmul"
// CHECK: tensor<1x3x5x2xf32>
// CHECK-NOT: onnx.
