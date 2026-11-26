// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each --split-input-file %s | FileCheck %s --check-prefixes=CHECK,LEGACY
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each --split-input-file %s | FileCheck %s --check-prefixes=CHECK,EXPERIMENTAL

module {
  func.func @wide_1d(%x: tensor<1x6144x5xf32>, %w: tensor<6144x1x3xf32>, %b: tensor<6144xf32>) -> tensor<1x6144x3xf32> {
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 6144 : si64, kernel_shape = [3], strides = [1], dilations = [1]} : (tensor<1x6144x5xf32>, tensor<6144x1x3xf32>, tensor<6144xf32>) -> tensor<1x6144x3xf32>
    return %y : tensor<1x6144x3xf32>
  }
  "onnx.EntryPoint"() {func = @wide_1d} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: "tfl.conv_2d"{{.*}} : (tensor<1x5x1x6144xf32>, tensor<6144x3x1x1xf32>, tensor<6144xf32>) -> tensor<1x3x1x6144xf32>
// CHECK-NOT: "tfl.conv_2d"
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: return {{.*}} : tensor<1x6144x3xf32>

// -----

module {
  func.func @boundary_2d(%x: tensor<1x4832x5x4xf32>, %w: tensor<4832x1x3x2xf32>, %b: tensor<4832xf32>) -> tensor<1x4832x3x3xf32> {
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 4832 : si64, kernel_shape = [3, 2], strides = [1, 1], dilations = [1, 1]} : (tensor<1x4832x5x4xf32>, tensor<4832x1x3x2xf32>, tensor<4832xf32>) -> tensor<1x4832x3x3xf32>
    return %y : tensor<1x4832x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @boundary_2d} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.conv_2d"
// CHECK: "tfl.depthwise_conv_2d"{{.*}}depth_multiplier = 1 : i32
// CHECK-NOT: "tfl.conv_2d"
// LEGACY: return {{.*}} : tensor<1x3x3x4832xf32>
// EXPERIMENTAL: return {{.*}} : tensor<1x4832x3x3xf32>

// -----

module {
  func.func @above_boundary(%x: tensor<1x4833x5x4xf32>) -> tensor<1x4833x3x3xf32> {
    %w = onnx.Constant dense<1.0> : tensor<4833x1x3x2xf32>
    %b = onnx.Constant dense<0.0> : tensor<4833xf32>
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 4833 : si64, kernel_shape = [3, 2], strides = [1, 1], dilations = [1, 1]} : (tensor<1x4833x5x4xf32>, tensor<4833x1x3x2xf32>, tensor<4833xf32>) -> tensor<1x4833x3x3xf32>
    return %y : tensor<1x4833x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @above_boundary} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: "tfl.conv_2d"{{.*}} : (tensor<1x5x4x4833xf32>, tensor<4833x3x2x1xf32>, tensor<4833xf32>) -> tensor<1x3x3x4833xf32>
// CHECK-NOT: "tfl.conv_2d"
// CHECK-NOT: "tfl.depthwise_conv_2d"
// LEGACY: return {{.*}} : tensor<1x3x3x4833xf32>
// EXPERIMENTAL: return {{.*}} : tensor<1x4833x3x3xf32>

// -----

// The guard is on OUTPUT channels, including depth multipliers above one.
module {
  func.func @multiplier_two(%x: tensor<2x2417x5x4xf32>, %w: tensor<4834x1x3x2xf32>, %b: tensor<4834xf32>) -> tensor<2x4834x3x3xf32> {
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 2417 : si64, kernel_shape = [3, 2], strides = [1, 1], dilations = [1, 1]} : (tensor<2x2417x5x4xf32>, tensor<4834x1x3x2xf32>, tensor<4834xf32>) -> tensor<2x4834x3x3xf32>
    return %y : tensor<2x4834x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @multiplier_two} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: "tfl.conv_2d"{{.*}} : (tensor<2x5x4x2417xf32>, tensor<4834x3x2x1xf32>, tensor<4834xf32>) -> tensor<2x3x3x4834xf32>
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: return

// -----

// A single input channel is also depthwise, but the fallback has group=1.
module {
  func.func @single_input_channel(%x: tensor<1x1x5x4xf32>, %w: tensor<4833x1x3x2xf32>, %b: tensor<4833xf32>) -> tensor<1x4833x3x3xf32> {
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 1 : si64, kernel_shape = [3, 2], strides = [1, 1], dilations = [1, 1]} : (tensor<1x1x5x4xf32>, tensor<4833x1x3x2xf32>, tensor<4833xf32>) -> tensor<1x4833x3x3xf32>
    return %y : tensor<1x4833x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @single_input_channel} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: "tfl.conv_2d"{{.*}} : (tensor<1x5x4x1xf32>, tensor<4833x3x2x1xf32>, tensor<4833xf32>) -> tensor<1x3x3x4833xf32>
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK: return

// -----

module {
  func.func @reduced_3d(%x: tensor<1x6144x2x3x4xf32>, %w: tensor<6144x1x1x2x2xf32>, %b: tensor<6144xf32>) -> tensor<1x6144x2x2x3xf32> {
    %y = "onnx.Conv"(%x, %w, %b) {auto_pad = "VALID", group = 6144 : si64, kernel_shape = [1, 2, 2], strides = [1, 1, 1], dilations = [1, 1, 1]} : (tensor<1x6144x2x3x4xf32>, tensor<6144x1x1x2x2xf32>, tensor<6144xf32>) -> tensor<1x6144x2x2x3xf32>
    return %y : tensor<1x6144x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @reduced_3d} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK-NOT: "tfl.conv_3d"
// CHECK: "tfl.conv_2d"{{.*}} : (tensor<2x3x4x6144xf32>, tensor<6144x2x2x1xf32>, tensor<6144xf32>) -> tensor<2x2x3x6144xf32>
// CHECK-NOT: "tfl.conv_2d"
// CHECK-NOT: "tfl.depthwise_conv_2d"
// CHECK-NOT: "tfl.conv_3d"
// CHECK: return {{.*}} : tensor<1x6144x2x2x3xf32>
