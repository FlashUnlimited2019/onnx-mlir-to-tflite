// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @squeeze_rank4(%arg0: tensor<1x64x1x1xf32>) -> tensor<1x64xf32> {
    %axes = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<1x64x1x1xf32>, tensor<2xi64>) -> tensor<1x64xf32>
    return %0 : tensor<1x64xf32>
  }
  "onnx.EntryPoint"() {func = @squeeze_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x1x64xf32>) -> tensor<1x64xf32>
// CHECK: arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x1x1x64xf32>, tensor<4xi32>) -> tensor<1x64x1x1xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x64x1x1xf32>, tensor<2xi32>) -> tensor<1x64xf32>

// -----

module {
  func.func @gather_scalar(%arg0: tensor<1x3xf32>) -> tensor<1xf32> {
    %index = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.Gather"(%arg0, %index) {axis = 1 : si64} : (tensor<1x3xf32>, tensor<i64>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @gather_scalar} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3xf32>) -> tensor<1xf32>
// CHECK: arith.constant dense<2> : tensor<i32>
// CHECK: "tfl.gather"(%arg0, {{.*}}) {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<1x3xf32>, tensor<i32>) -> tensor<1xf32>

// -----

module {
  func.func @clip_scalar_bounds(%arg0: tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32> {
    %min = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %max = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<1x2x1x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2x1x8xf32>
    return %0 : tensor<1x2x1x8xf32>
  }
  "onnx.EntryPoint"() {func = @clip_scalar_bounds} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x8x2xf32>) -> tensor<1x1x8x2xf32>
// CHECK: "tfl.maximum"(%arg0, {{.*}}) : (tensor<1x1x8x2xf32>, tensor<f32>) -> tensor<1x1x8x2xf32>
// CHECK: "tfl.minimum"({{.*}}) : (tensor<1x1x8x2xf32>, tensor<f32>) -> tensor<1x1x8x2xf32>

// -----

module {
  func.func @dilated_temporal_conv(%arg0: tensor<1x4x1x16xf32>, %arg1: tensor<6x4x1x3xf32>) -> tensor<1x6x1x16xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %arg1, %none) {auto_pad = "NOTSET", dilations = [1, 2], group = 1 : si64, kernel_shape = [1, 3], pads = [0, 2, 0, 2], strides = [1, 1]} : (tensor<1x4x1x16xf32>, tensor<6x4x1x3xf32>, none) -> tensor<1x6x1x16xf32>
    return %0 : tensor<1x6x1x16xf32>
  }
  "onnx.EntryPoint"() {func = @dilated_temporal_conv} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x16x4xf32>, %arg1: tensor<6x1x3x4xf32>) -> tensor<1x1x16x6xf32>
// CHECK: "tfl.pad"(%arg0, {{.*}}) : (tensor<1x1x16x4xf32>, tensor<4x2xi32>) -> tensor<1x1x20x4xf32>
// CHECK: "tfl.conv_2d"({{.*}}) {dilation_h_factor = 1 : i32, dilation_w_factor = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}
// CHECK-NOT: "tfl.space_to_batch_nd"
// CHECK-NOT: "tfl.batch_to_space_nd"

// -----

module {
  func.func @reduce_sum_v11(%arg0: tensor<1x64xf32>) -> tensor<1xf32> {
    %0 = "onnx.ReduceSumV11"(%arg0) {axes = [-1], keepdims = 0 : si64} : (tensor<1x64xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @reduce_sum_v11} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x64xf32>) -> tensor<1xf32>
// CHECK: arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.sum"(%arg0, {{.*}}) {keep_dims = false} : (tensor<1x64xf32>, tensor<1xi32>) -> tensor<1xf32>

// -----

module {
  func.func @layer_norm_rank2(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
    %bias = "onnx.Constant"() {value = dense<0.0> : tensor<3xf32>} : () -> tensor<3xf32>
    %y, %mean, %inv_std_dev = "onnx.LayerNormalization"(%arg0, %scale, %bias) {axis = 1 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64} : (tensor<1x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x3xf32>, none, none)
    return %y : tensor<1x3xf32>
  }
  "onnx.EntryPoint"() {func = @layer_norm_rank2} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = true} : (tensor<1x3xf32>, tensor<1xi32>) -> tensor<1x1xf32>
// CHECK: "tfl.sub"
// CHECK: "tfl.mul"
// CHECK: "tfl.mean"
// CHECK: "tfl.rsqrt"
// CHECK: "tfl.mul"
// CHECK: "tfl.mul"
// CHECK: "tfl.add"

// -----

module {
  func.func @layer_norm_rank3(%arg0: tensor<1x4x3xf32>) -> tensor<1x4x3xf32> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
    %bias = "onnx.Constant"() {value = dense<0.0> : tensor<3xf32>} : () -> tensor<3xf32>
    %y, %mean, %inv_std_dev = "onnx.LayerNormalization"(%arg0, %scale, %bias) {axis = -1 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64} : (tensor<1x4x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x4x3xf32>, none, none)
    return %y : tensor<1x4x3xf32>
  }
  "onnx.EntryPoint"() {func = @layer_norm_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
// CHECK: arith.constant dense<2> : tensor<1xi32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = true} : (tensor<1x4x3xf32>, tensor<1xi32>) -> tensor<1x4x1xf32>
// CHECK: "tfl.sub"
// CHECK: "tfl.mul"
// CHECK: "tfl.mean"
// CHECK: "tfl.rsqrt"
// CHECK: "tfl.mul"
// CHECK: "tfl.mul"
// CHECK: "tfl.add"

// -----

module {
  func.func @rms_layer_norm_rank3(%arg0: tensor<1x4x3xf32>) -> tensor<1x4x3xf32> {
    %scale = "onnx.Constant"() {value = dense<1.0> : tensor<3xf32>} : () -> tensor<3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %inv_std_dev = "onnx.RMSLayerNormalization"(%arg0, %scale, %none) {axis = 2 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64} : (tensor<1x4x3xf32>, tensor<3xf32>, none) -> (tensor<1x4x3xf32>, none)
    return %y : tensor<1x4x3xf32>
  }
  "onnx.EntryPoint"() {func = @rms_layer_norm_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
// CHECK: arith.constant dense<2> : tensor<1xi32>
// CHECK: "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<1x4x3xf32>, tensor<1x4x3xf32>) -> tensor<1x4x3xf32>
// CHECK: "tfl.mean"({{.*}}) {keep_dims = true} : (tensor<1x4x3xf32>, tensor<1xi32>) -> tensor<1x4x1xf32>
// CHECK: "tfl.add"
// CHECK: "tfl.rsqrt"
// CHECK: "tfl.mul"
// CHECK: "tfl.mul"
// CHECK-NOT: "tfl.sub"

// -----

module {
  func.func @neg_rank4(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = "onnx.Neg"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    return %0 : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @neg_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: "tfl.neg"(%arg0) : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>

// -----

module {
  func.func @average_pool_include_pad(%arg0: tensor<1x2x1x16xf32>) -> tensor<1x2x1x16xf32> {
    %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, count_include_pad = 1 : si64, kernel_shape = [1, 11], pads = [0, 5, 0, 5], strides = [1, 1]} : (tensor<1x2x1x16xf32>) -> tensor<1x2x1x16xf32>
    return %0 : tensor<1x2x1x16xf32>
  }
  "onnx.EntryPoint"() {func = @average_pool_include_pad} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x16x2xf32>) -> tensor<1x1x16x2xf32>
// CHECK: arith.constant dense<{{\[}}[0, 0], [0, 0], [5, 5], [0, 0]{{\]}}> : tensor<4x2xi32>
// CHECK: "tfl.pad"(%arg0, {{.*}}) : (tensor<1x1x16x2xf32>, tensor<4x2xi32>) -> tensor<1x1x26x2xf32>
// CHECK: "tfl.average_pool_2d"({{.*}}) {filter_height = 1 : i32, filter_width = 11 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}
