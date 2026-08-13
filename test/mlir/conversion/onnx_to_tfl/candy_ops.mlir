// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @reflect_pad(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x6x7xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %value = "onnx.Constant"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%arg0, %pads, %value, %none) {mode = "reflect"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<1x3x6x7xf32>
    return %0 : tensor<1x3x6x7xf32>
  }
  "onnx.EntryPoint"() {func = @reflect_pad} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>
// CHECK: arith.constant dense<{{\[}}[0, 0], [1, 1], [1, 1], [0, 0]{{\]}}> : tensor<4x2xi32>
// CHECK: "tfl.mirror_pad"(%arg0, {{.*}}) {mode = #tfl<mirror_pad_attr REFLECT>} : (tensor<1x4x5x3xf32>, tensor<4x2xi32>) -> tensor<1x6x7x3xf32>

// -----

module {
  func.func @instance_normalization_form(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
    %bias = "onnx.Constant"() {value = dense<0.0> : tensor<2x1x1xf32>} : () -> tensor<2x1x1xf32>
    %y, %mean, %inv_std_dev = "onnx.LayerNormalization"(%arg0, %scale, %bias) {axis = 2 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64} : (tensor<1x2x3x4xf32>, tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> (tensor<1x2x3x4xf32>, none, none)
    return %y : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @instance_normalization_form} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>
// CHECK: arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = true} : (tensor<1x3x4x2xf32>, tensor<2xi32>) -> tensor<1x1x1x2xf32>
// CHECK: "tfl.sub"(%arg0, {{.*}}) {fused_activation_function = "NONE"}
// CHECK: "tfl.mul"
// CHECK: "tfl.mean"
// CHECK: "tfl.add"
// CHECK: "tfl.rsqrt"
// CHECK: "tfl.mul"
// CHECK: arith.constant dense<[1, 1, 2]> : tensor<3xi32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x1x1xf32>, tensor<3xi32>) -> tensor<1x1x2xf32>
// CHECK: "tfl.mul"
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x1x1xf32>, tensor<3xi32>) -> tensor<1x1x2xf32>
// CHECK: "tfl.add"({{.*}}) {fused_activation_function = "NONE"} : (tensor<1x3x4x2xf32>, tensor<1x1x2xf32>) -> tensor<1x3x4x2xf32>

// -----

module {
  func.func @resize_nearest_half_pixel(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x8x10xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x3x4x5xf32>, none, tensor<4xf32>, none) -> tensor<1x3x8x10xf32>
    return %0 : tensor<1x3x8x10xf32>
  }
  "onnx.EntryPoint"() {func = @resize_nearest_half_pixel} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0
// CHECK-COUNT-2: "tfl.gather"
// CHECK-NOT: "tfl.resize_nearest_neighbor"
