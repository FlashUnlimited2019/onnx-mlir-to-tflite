// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @unary_ops(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = "onnx.Elu"(%arg0) <{alpha = 1.000000e+00 : f32}> : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %1 = "onnx.Floor"(%0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    %2 = "onnx.Reciprocal"(%1) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
    return %2 : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @unary_ops} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: "tfl.elu"(%arg0) : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: "tfl.floor"({{.*}}) : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK: "tfl.div"({{.*}}) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @integer_coordinate_ops(%arg0: tensor<2x3xi64>) -> (tensor<2x3xf32>, tensor<1x2x3xf32>) {
    %scalar = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %less = "onnx.Less"(%arg0, %scalar) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi1>
    %greater = "onnx.Greater"(%arg0, %scalar) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi1>
    %less_f32 = "onnx.Cast"(%less) <{saturate = 1 : si64, to = f32}> : (tensor<2x3xi1>) -> tensor<2x3xf32>
    %greater_f32 = "onnx.Cast"(%greater) <{saturate = 1 : si64, to = f32}> : (tensor<2x3xi1>) -> tensor<2x3xf32>
    %comparison_sum = "onnx.Add"(%less_f32, %greater_f32) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %added = "onnx.Add"(%arg0, %scalar) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
    %multiplied = "onnx.Mul"(%added, %scalar) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
    %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %expanded = "onnx.Unsqueeze"(%multiplied, %axes) : (tensor<2x3xi64>, tensor<1xi64>) -> tensor<1x2x3xi64>
    %expanded_f32 = "onnx.Cast"(%expanded) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x3xi64>) -> tensor<1x2x3xf32>
    return %comparison_sum, %expanded_f32 : tensor<2x3xf32>, tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @integer_coordinate_ops} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x3xi64>) -> (tensor<2x3xf32>, tensor<1x2x3xf32>)
// CHECK: "tfl.less"(%arg0, {{.*}}) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi1>
// CHECK: "tfl.greater"(%arg0, {{.*}}) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi1>
// CHECK: "tfl.cast"
// CHECK: "tfl.cast"
// CHECK: "tfl.add"
// CHECK: "tfl.add"(%arg0, {{.*}}) {fused_activation_function = "NONE"} : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
// CHECK: "tfl.mul"({{.*}}) {fused_activation_function = "NONE"} : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
// CHECK: "tfl.reshape"{{.*}} : (tensor<2x3xi64>, tensor<3xi32>) -> tensor<1x2x3xi64>
// CHECK: "tfl.cast"{{.*}} : (tensor<1x2x3xi64>) -> tensor<1x2x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @depth_to_space_crd(%arg0: tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32> {
    %0 = "onnx.DepthToSpace"(%arg0) <{blocksize = 2 : si64, mode = "CRD"}> : (tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32>
    return %0 : tensor<1x2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @depth_to_space_crd} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x8xf32>) -> tensor<1x4x6x2xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3x8xf32>, tensor<3xi32>) -> tensor<6x2x4xf32>
// CHECK: arith.constant dense<[0, 2, 1]> : tensor<3xi32>
// CHECK: "tfl.transpose"{{.*}} : (tensor<6x2x4xf32>, tensor<3xi32>) -> tensor<6x4x2xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<6x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x8xf32>
// CHECK: "tfl.depth_to_space"{{.*}} : (tensor<1x2x3x8xf32>) -> tensor<1x4x6x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @depth_to_space_dcr(%arg0: tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32> {
    %0 = "onnx.DepthToSpace"(%arg0) <{blocksize = 2 : si64, mode = "DCR"}> : (tensor<1x8x2x3xf32>) -> tensor<1x2x4x6xf32>
    return %0 : tensor<1x2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @depth_to_space_dcr} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x8xf32>) -> tensor<1x4x6x2xf32>
// CHECK: "tfl.depth_to_space"(%arg0) {block_size = 2 : i32} : (tensor<1x2x3x8xf32>) -> tensor<1x4x6x2xf32>
// CHECK-NOT: onnx.
