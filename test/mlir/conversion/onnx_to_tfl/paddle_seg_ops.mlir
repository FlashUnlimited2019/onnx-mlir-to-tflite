// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @hard_swish(%arg0: tensor<1x4x3x5xf32>) -> tensor<1x4x3x5xf32> {
    %0 = "onnx.HardSwish"(%arg0) : (tensor<1x4x3x5xf32>) -> tensor<1x4x3x5xf32>
    return %0 : tensor<1x4x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @hard_swish} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x5x4xf32>
// CHECK: "tfl.hard_swish"(%arg0) : (tensor<1x3x5x4xf32>) -> tensor<1x3x5x4xf32>

// -----

module {
  func.func @hard_sigmoid(%arg0: tensor<1x4x3x5xf32>) -> tensor<1x4x3x5xf32> {
    %0 = "onnx.HardSigmoid"(%arg0) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<1x4x3x5xf32>) -> tensor<1x4x3x5xf32>
    return %0 : tensor<1x4x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @hard_sigmoid} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x5x4xf32>
// CHECK: arith.constant dense<2.000000e-01> : tensor<f32>
// CHECK: arith.constant dense<5.000000e-01> : tensor<f32>
// CHECK: "tfl.mul"(%arg0, {{.*}}) {fused_activation_function = "NONE"}
// CHECK: "tfl.add"({{.*}}) {fused_activation_function = "NONE"}
// CHECK: "tfl.maximum"
// CHECK: "tfl.minimum"

// -----

module {
  func.func @average_pool(%arg0: tensor<1x4x6x6xf32>) -> tensor<1x4x2x2xf32> {
    %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, count_include_pad = 0 : si64, kernel_shape = [3, 3], strides = [3, 3]} : (tensor<1x4x6x6xf32>) -> tensor<1x4x2x2xf32>
    return %0 : tensor<1x4x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @average_pool} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x6x6x4xf32>
// CHECK: "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 3 : i32} : (tensor<1x6x6x4xf32>) -> tensor<1x2x2x4xf32>

// -----

module {
  func.func @resize_bilinear(%arg0: tensor<1x3x2x2xf32>) -> tensor<1x3x6x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %sizes = "onnx.Constant"() {value = dense<[1, 3, 6, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
    %0 = "onnx.Resize"(%arg0, %none, %none, %sizes) {coordinate_transformation_mode = "half_pixel", mode = "linear"} : (tensor<1x3x2x2xf32>, none, none, tensor<4xi64>) -> tensor<1x3x6x6xf32>
    return %0 : tensor<1x3x6x6xf32>
  }
  "onnx.EntryPoint"() {func = @resize_bilinear} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x2x3xf32>
// CHECK: arith.constant dense<6> : tensor<2xi32>
// CHECK: "tfl.resize_bilinear"(%arg0, {{.*}}) {align_corners = false, half_pixel_centers = true} : (tensor<1x2x2x3xf32>, tensor<2xi32>) -> tensor<1x6x6x3xf32>

// -----

module {
  func.func @depthwise_conv(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
    %filter = "onnx.Constant"() {value = dense<1.0> : tensor<2x1x3x3xf32>} : () -> tensor<2x1x3x3xf32>
    %bias = "onnx.Constant"() {value = dense<0.0> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "onnx.Conv"(%arg0, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x2x4x4xf32>, tensor<2x1x3x3xf32>, tensor<2xf32>) -> tensor<1x2x4x4xf32>
    return %0 : tensor<1x2x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @depthwise_conv} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x4x2xf32>
// CHECK: arith.constant dense<1.000000e+00> : tensor<2x3x3x1xf32>
// CHECK: arith.constant dense<[3, 1, 2, 0]> : tensor<4xi32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x3x1xf32>, tensor<4xi32>) -> tensor<1x3x3x2xf32>
// CHECK: "tfl.depthwise_conv_2d"({{.*}}) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}

// -----

module {
  func.func @softmax_channels(%arg0: tensor<1x2x3x5xf32>) -> tensor<1x2x3x5xf32> {
    %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<1x2x3x5xf32>) -> tensor<1x2x3x5xf32>
    return %0 : tensor<1x2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @softmax_channels} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x5x2xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<1x3x5x2xf32>) -> tensor<1x3x5x2xf32>
// CHECK-NOT: "tfl.transpose"
