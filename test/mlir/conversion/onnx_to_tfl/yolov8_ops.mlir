// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @slice_rank4(%arg0: tensor<1x8x4x5xf32>) -> tensor<1x4x4x5xf32> {
    %starts = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<6> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x8x4x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x4x5xf32>
    return %0 : tensor<1x4x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @slice_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x8xf32>
// CHECK: arith.constant dense<[0, 0, 0, 2]> : tensor<4xi32>
// CHECK: arith.constant dense<[1, 4, 5, 4]> : tensor<4xi32>
// CHECK: "tfl.slice"({{.*}}) : (tensor<1x4x5x8xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x4x5x4xf32>

// -----

module {
  func.func @slice_to_end(%arg0: tensor<1x8x5xf32>) -> tensor<1x6x5xf32> {
    %starts = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x8x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x6x5xf32>
    return %0 : tensor<1x6x5xf32>
  }
  "onnx.EntryPoint"() {func = @slice_to_end} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x8x5xf32>
// CHECK: arith.constant dense<[0, 2, 0]> : tensor<3xi32>
// CHECK: arith.constant dense<[1, 6, 5]> : tensor<3xi32>
// CHECK: "tfl.slice"

// -----

module {
  func.func @resize_nearest(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x8x10xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
    %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x3x4x5xf32>, none, tensor<4xf32>, none) -> tensor<1x3x8x10xf32>
    return %0 : tensor<1x3x8x10xf32>
  }
  "onnx.EntryPoint"() {func = @resize_nearest} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>
// CHECK: arith.constant dense<[8, 10]> : tensor<2xi32>
// CHECK: "tfl.resize_nearest_neighbor"(%arg0, {{.*}}) {align_corners = false, half_pixel_centers = false} : (tensor<1x4x5x3xf32>, tensor<2xi32>) -> tensor<1x8x10x3xf32>

// -----

module {
  func.func @reshape_to_rank4(%arg0: tensor<1x64x6xf32>) -> tensor<1x4x8x12xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 4, 8, 12]> : tensor<4xi64>} : () -> tensor<4xi64>
    %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x64x6xf32>, tensor<4xi64>) -> tensor<1x4x8x12xf32>
    return %0 : tensor<1x4x8x12xf32>
  }
  "onnx.EntryPoint"() {func = @reshape_to_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x64x6xf32>) -> tensor<1x8x12x4xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x64x6xf32>, tensor<4xi32>) -> tensor<1x4x8x12xf32>
// CHECK: arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x4x8x12xf32>, tensor<4xi32>) -> tensor<1x8x12x4xf32>

// -----

module {
  func.func @transpose_rank4(%arg0: tensor<1x4x8x12xf32>) -> tensor<1x12x4x8xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<1x4x8x12xf32>) -> tensor<1x12x4x8xf32>
    return %0 : tensor<1x12x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @transpose_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x8x12x4xf32>) -> tensor<1x4x8x12xf32>
// CHECK: arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x8x12x4xf32>, tensor<4xi32>) -> tensor<1x4x8x12xf32>

// -----

module {
  func.func @softmax_rank4(%arg0: tensor<1x12x4x16xf32>) -> tensor<1x12x4x16xf32> {
    %0 = "onnx.Softmax"(%arg0) {axis = 3 : si64} : (tensor<1x12x4x16xf32>) -> tensor<1x12x4x16xf32>
    return %0 : tensor<1x12x4x16xf32>
  }
  "onnx.EntryPoint"() {func = @softmax_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x16x12xf32>
// CHECK: arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x4x16x12xf32>, tensor<4xi32>) -> tensor<1x4x12x16xf32>
// CHECK: "tfl.softmax"({{.*}}) {beta = 1.000000e+00 : f32} : (tensor<1x4x12x16xf32>) -> tensor<1x4x12x16xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x4x12x16xf32>, tensor<4xi32>) -> tensor<1x4x16x12xf32>

// -----

module {
  func.func @reshape_from_rank4(%arg0: tensor<1x1x4x8xf32>) -> tensor<1x4x8xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 4, 8]> : tensor<3xi64>} : () -> tensor<3xi64>
    %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x1x4x8xf32>, tensor<3xi64>) -> tensor<1x4x8xf32>
    return %0 : tensor<1x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @reshape_from_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x8x1xf32>) -> tensor<1x4x8xf32>
// CHECK: arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x4x8x1xf32>, tensor<4xi32>) -> tensor<1x1x4x8xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x1x4x8xf32>, tensor<3xi32>) -> tensor<1x4x8xf32>

// -----

module {
  func.func @scalar_div(%arg0: tensor<1x2x8xf32>) -> tensor<1x2x8xf32> {
    %scalar = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    %0 = "onnx.Div"(%arg0, %scalar) : (tensor<1x2x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
    return %0 : tensor<1x2x8xf32>
  }
  "onnx.EntryPoint"() {func = @scalar_div} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x8xf32>
// CHECK: "tfl.div"(%arg0, {{.*}}) {fused_activation_function = "NONE"} : (tensor<1x2x8xf32>, tensor<f32>) -> tensor<1x2x8xf32>
