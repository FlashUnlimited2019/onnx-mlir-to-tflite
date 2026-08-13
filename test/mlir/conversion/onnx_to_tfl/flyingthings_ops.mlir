// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @argmin_rank4(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3xf32> {
    %0 = "onnx.ArgMin"(%arg0) <{axis = 3 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}> : (tensor<1x2x3x4xf32>) -> tensor<1x2x3xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x3xi64>) -> tensor<1x2x3xf32>
    return %1 : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @argmin_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x2x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.arg_min"({{.*}}) : (tensor<1x2x3x4xf32>, tensor<i32>) -> tensor<1x2x3xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x2x3xi64>) -> tensor<1x2x3xf32>

// -----

module {
  func.func @space_to_depth(%arg0: tensor<1x2x4x6xf32>) -> tensor<1x8x2x3xf32> {
    %0 = "onnx.SpaceToDepth"(%arg0) <{blocksize = 2 : si64}> : (tensor<1x2x4x6xf32>) -> tensor<1x8x2x3xf32>
    return %0 : tensor<1x8x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @space_to_depth} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x6x2xf32>) -> tensor<1x2x3x8xf32>
// CHECK: "tfl.space_to_depth"(%arg0) {block_size = 2 : i32} : (tensor<1x4x6x2xf32>) -> tensor<1x2x3x8xf32>

// -----

module {
  func.func @integer_warp_path(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) {
    %upper = "onnx.Constant"() {value = dense<7> : tensor<i32>} : () -> tensor<i32>
    %offset = "onnx.Constant"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %lower = "onnx.Constant"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %floored = "onnx.Floor"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %indices = "onnx.Cast"(%floored) <{saturate = 1 : si64, to = i32}> : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi32>
    %minimum = "onnx.Min"(%indices, %upper) : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
    %shifted = "onnx.Add"(%minimum, %offset) : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
    %maximum = "onnx.Max"(%shifted, %lower) : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
    %direct = "onnx.Cast"(%maximum) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xf32>
    %wide = "onnx.Cast"(%maximum) <{saturate = 1 : si64, to = i64}> : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi64>
    %roundtrip = "onnx.Cast"(%wide) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x3x4xi64>) -> tensor<1x2x3x4xf32>
    return %direct, %roundtrip : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @integer_warp_path} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> (tensor<1x3x4x2xf32>, tensor<1x3x4x2xf32>)
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi32>
// CHECK: "tfl.minimum"{{.*}} : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
// CHECK: "tfl.add"{{.*}} {fused_activation_function = "NONE"} : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
// CHECK: "tfl.maximum"{{.*}} : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<1x2x3x4xi32>
// CHECK: "tfl.cast"{{.*}} : (tensor<1x2x3x4xi32>) -> tensor<1x2x3x4xi64>
// CHECK-NOT: onnx.

// -----

module {
  func.func @reducible_rank5_upsample_and_pad(%arg0: tensor<1x3x2x3x1xf32>) -> tensor<1x3x7x13x1xf32> {
    %0 = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 2, 0, 3, 4, 0], strides = [2, 3, 1]}> : (tensor<1x3x2x3x1xf32>) -> tensor<1x3x7x13x1xf32>
    return %0 : tensor<1x3x7x13x1xf32>
  }
  "onnx.EntryPoint"() {func = @reducible_rank5_upsample_and_pad} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x2x3x1xf32>) -> tensor<1x3x7x13x1xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x3x2x3x1xf32>, tensor<4xi32>) -> tensor<1x3x2x3xf32>
// CHECK: "tfl.padv2"{{.*}} : (tensor<1x3x2x1x3xf32>, tensor<5x2xi32>, tensor<f32>) -> tensor<1x3x2x2x3xf32>
// CHECK: "tfl.padv2"{{.*}} : (tensor<1x3x4x3x1xf32>, tensor<5x2xi32>, tensor<f32>) -> tensor<1x3x4x3x3xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x3x7x13xf32>, tensor<5xi32>) -> tensor<1x3x7x13x1xf32>
// CHECK-NOT: tensor<1x3x2x3x1x1xf32>

// -----

module {
  func.func @rank6_slice_singleton_collapse(%arg0: tensor<1x2x3x4x1x5xf32>) -> tensor<1x2x3x2x1x5xf32> {
    %starts = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x2x3x4x1x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2x3x2x1x5xf32>
    return %0 : tensor<1x2x3x2x1x5xf32>
  }
  "onnx.EntryPoint"() {func = @rank6_slice_singleton_collapse} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x1x5xf32>) -> tensor<1x2x3x2x1x5xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3x4x1x5xf32>, tensor<5xi32>) -> tensor<2x3x4x1x5xf32>
// CHECK: "tfl.slice"{{.*}} : (tensor<2x3x4x1x5xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<2x3x2x1x5xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<2x3x2x1x5xf32>, tensor<6xi32>) -> tensor<1x2x3x2x1x5xf32>
