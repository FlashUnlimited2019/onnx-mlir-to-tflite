// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @greater_equal_rank1(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    %0 = "onnx.GreaterOrEqual"(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<16xi1>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
  "onnx.EntryPoint"() {func = @greater_equal_rank1} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32>
// CHECK: "tfl.greater_equal"(%arg0, %arg1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<16xi1>) -> tensor<16xf32>

// -----

module {
  func.func @greater_equal_rank4_rank3(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %0 = "onnx.GreaterOrEqual"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
    return %1 : tensor<2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @greater_equal_rank4_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<2x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.greater_equal"({{.*}}, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<2x4x5x3xf32>

// -----

module {
  func.func @greater_equal_rank5_rank3(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
    %0 = "onnx.GreaterOrEqual"(%arg0, %arg1) : (tensor<1x2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x3x4x5xi1>) -> tensor<1x2x3x4x5xf32>
    return %1 : tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @greater_equal_rank5_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3x4x5xf32>, tensor<1xi32>) -> tensor<120xf32>
// CHECK: "tfl.broadcast_to"(%arg1, {{.*}}) : (tensor<3x4x5xf32>, tensor<5xi32>) -> tensor<1x2x3x4x5xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x2x3x4x5xf32>, tensor<1xi32>) -> tensor<120xf32>
// CHECK: "tfl.greater_equal"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xi1>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<120xi1>, tensor<5xi32>) -> tensor<1x2x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x2x3x4x5xi1>) -> tensor<1x2x3x4x5xf32>
