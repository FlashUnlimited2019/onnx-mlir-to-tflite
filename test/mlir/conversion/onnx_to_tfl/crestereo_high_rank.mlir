// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @collapse_rank6_expand(%arg0: tensor<1x4x2x3xf32>) -> tensor<1x4x5x2x3xf32> {
    %axes = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %expanded_shape = "onnx.Constant"() {value = dense<[1, 4, 1, 5, 2, 3]> : tensor<6xi64>} : () -> tensor<6xi64>
    %output_shape = "onnx.Constant"() {value = dense<[1, 4, 5, 2, 3]> : tensor<5xi64>} : () -> tensor<5xi64>
    %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<1x4x2x3xf32>, tensor<2xi64>) -> tensor<1x4x1x1x2x3xf32>
    %1 = "onnx.Expand"(%0, %expanded_shape) : (tensor<1x4x1x1x2x3xf32>, tensor<6xi64>) -> tensor<1x4x1x5x2x3xf32>
    %2 = "onnx.Reshape"(%1, %output_shape) <{allowzero = 0 : si64}> : (tensor<1x4x1x5x2x3xf32>, tensor<5xi64>) -> tensor<1x4x5x2x3xf32>
    return %2 : tensor<1x4x5x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @collapse_rank6_expand} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x4x5x2x3xf32>
// CHECK-NOT: tensor<1x4x1x1x2x3xf32>
// CHECK-NOT: tensor<1x4x1x5x2x3xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<1x4x1x2x3xf32>
// CHECK: "tfl.broadcast_to"{{.*}} -> tensor<1x4x5x2x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @lower_convex_upsample(%weights: tensor<1x36x2x3xf32>, %flow: tensor<1x2x2x3xf32>) -> tensor<1x2x4x6xf32> {
    %weight_shape = "onnx.Constant"() {value = dense<[1, 1, 9, 2, 2, 2, 3]> : tensor<7xi64>} : () -> tensor<7xi64>
    %weight_7d = "onnx.Reshape"(%weights, %weight_shape) <{allowzero = 0 : si64}> : (tensor<1x36x2x3xf32>, tensor<7xi64>) -> tensor<1x1x9x2x2x2x3xf32>
    %weight_moved = "onnx.Transpose"(%weight_7d) <{perm = [0, 1, 6, 3, 4, 5, 2]}> : (tensor<1x1x9x2x2x2x3xf32>) -> tensor<1x1x3x2x2x2x9xf32>
    %normalized = "onnx.Softmax"(%weight_moved) <{axis = 6 : si64}> : (tensor<1x1x3x2x2x2x9xf32>) -> tensor<1x1x3x2x2x2x9xf32>
    %weight_restored = "onnx.Transpose"(%normalized) <{perm = [0, 1, 6, 3, 4, 5, 2]}> : (tensor<1x1x3x2x2x2x9xf32>) -> tensor<1x1x9x2x2x2x3xf32>

    %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 1, 0, 0, 1, 1]> : tensor<8xi64>} : () -> tensor<8xi64>
    %zero = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %none = "onnx.NoValue"() : () -> none
    %padded = "onnx.Pad"(%flow, %pads, %zero, %none) <{mode = "constant"}> : (tensor<1x2x2x3xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x2x4x5xf32>
    %rows = "onnx.Constant"() {value = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>} : () -> tensor<3x2xi64>
    %columns = "onnx.Constant"() {value = dense<[[0, 1, 2], [1, 2, 3], [2, 3, 4]]> : tensor<3x3xi64>} : () -> tensor<3x3xi64>
    %row_patches = "onnx.Gather"(%padded, %rows) <{axis = 2 : si64}> : (tensor<1x2x4x5xf32>, tensor<3x2xi64>) -> tensor<1x2x3x2x5xf32>
    %patches = "onnx.Gather"(%row_patches, %columns) <{axis = 4 : si64}> : (tensor<1x2x3x2x5xf32>, tensor<3x3xi64>) -> tensor<1x2x3x2x3x3xf32>
    %patches_moved = "onnx.Transpose"(%patches) <{perm = [0, 1, 2, 4, 3, 5]}> : (tensor<1x2x3x2x3x3xf32>) -> tensor<1x2x3x3x2x3xf32>
    %patch_shape = "onnx.Constant"() {value = dense<[1, 2, 9, 1, 1, 2, 3]> : tensor<7xi64>} : () -> tensor<7xi64>
    %patch_7d = "onnx.Reshape"(%patches_moved, %patch_shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x3x2x3xf32>, tensor<7xi64>) -> tensor<1x2x9x1x1x2x3xf32>
    %product = "onnx.Mul"(%weight_restored, %patch_7d) : (tensor<1x1x9x2x2x2x3xf32>, tensor<1x2x9x1x1x2x3xf32>) -> tensor<1x2x9x2x2x2x3xf32>
    %summed = "onnx.ReduceSumV11"(%product) <{axes = [2], keepdims = 0 : si64}> : (tensor<1x2x9x2x2x2x3xf32>) -> tensor<1x2x2x2x2x3xf32>
    %output_moved = "onnx.Transpose"(%summed) <{perm = [0, 1, 4, 2, 5, 3]}> : (tensor<1x2x2x2x2x3xf32>) -> tensor<1x2x2x2x3x2xf32>
    %output_shape = "onnx.Constant"() {value = dense<[1, 2, 4, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
    %output = "onnx.Reshape"(%output_moved, %output_shape) <{allowzero = 0 : si64}> : (tensor<1x2x2x2x3x2xf32>, tensor<4xi64>) -> tensor<1x2x4x6xf32>
    return %output : tensor<1x2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @lower_convex_upsample} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x36xf32>, %arg1: tensor<1x2x3x2xf32>) -> tensor<1x4x6x2xf32>
// CHECK-NOT: tensor<1x1x9x2x2x2x3xf32>
// CHECK-NOT: tensor<1x2x9x2x2x2x3xf32>
// CHECK: "tfl.softmax"
// CHECK: "tfl.gather"{{.*}} tensor<9x6xi32>
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.depth_to_space"
// CHECK-NOT: onnx.
