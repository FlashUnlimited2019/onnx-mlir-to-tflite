// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @mod_f32_rank4_rank2(%arg0: tensor<1x4x5x6xf32>, %arg1: tensor<5x6xf32>) -> tensor<1x4x5x6xf32> {
    %0 = "onnx.Mod"(%arg0, %arg1) <{fmod = 1 : si64}> : (tensor<1x4x5x6xf32>, tensor<5x6xf32>) -> tensor<1x4x5x6xf32>
    return %0 : tensor<1x4x5x6xf32>
  }
  "onnx.EntryPoint"() {func = @mod_f32_rank4_rank2} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x5x6x4xf32>, %arg1: tensor<5x6xf32>) -> tensor<1x5x6x4xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x5x6x4xf32>, tensor<4xi32>) -> tensor<1x4x5x6xf32>
// CHECK: "tfl.floor_mod"({{.*}}, %arg1) : (tensor<1x4x5x6xf32>, tensor<5x6xf32>) -> tensor<1x4x5x6xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x4x5x6xf32>, tensor<4xi32>) -> tensor<1x5x6x4xf32>

// -----

module {
  func.func @mod_f32_rank5_broadcast(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
    %0 = "onnx.Mod"(%arg0, %arg1) <{fmod = 1 : si64}> : (tensor<1x2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
    return %0 : tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @mod_f32_rank5_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
// CHECK: "tfl.broadcast_to"(%arg1, {{.*}}) : (tensor<3x4x5xf32>, tensor<5xi32>) -> tensor<1x2x3x4x5xf32>
// CHECK: "tfl.floor_mod"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<120xf32>, tensor<5xi32>) -> tensor<1x2x3x4x5xf32>

// -----

module {
  func.func @mod_i64_floor(%arg0: tensor<4x7xf32>) -> tensor<4x7xf32> {
    %rhs = "onnx.Constant"() {value = dense<[1, -2, 3, -4, 5, -6, 7]> : tensor<7xi64>} : () -> tensor<7xi64>
    %0 = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = i64}> : (tensor<4x7xf32>) -> tensor<4x7xi64>
    %1 = "onnx.Mod"(%0, %rhs) <{fmod = 0 : si64}> : (tensor<4x7xi64>, tensor<7xi64>) -> tensor<4x7xi64>
    %2 = "onnx.Cast"(%1) <{saturate = 1 : si64, to = f32}> : (tensor<4x7xi64>) -> tensor<4x7xf32>
    return %2 : tensor<4x7xf32>
  }
  "onnx.EntryPoint"() {func = @mod_i64_floor} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<4x7xf32>) -> tensor<4x7xf32>
// CHECK: "tfl.cast"(%arg0) : (tensor<4x7xf32>) -> tensor<4x7xi64>
// CHECK: "tfl.floor_mod"({{.*}}, {{.*}}) : (tensor<4x7xi64>, tensor<7xi64>) -> tensor<4x7xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<4x7xi64>) -> tensor<4x7xf32>

// -----

module {
  func.func @mod_i64_fmod_rank4(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %rhs = "onnx.Constant"() {value = dense<2> : tensor<3x4x5xi64>} : () -> tensor<3x4x5xi64>
    %0 = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = i64}> : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi64>
    %1 = "onnx.Mod"(%0, %rhs) <{fmod = 1 : si64}> : (tensor<2x3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<2x3x4x5xi64>
    %2 = "onnx.Cast"(%1) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x4x5xi64>) -> tensor<2x3x4x5xf32>
    return %2 : tensor<2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @mod_i64_fmod_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>) -> tensor<2x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xi64>
// CHECK: "tfl.floor_mod"({{.*}}, {{.*}}) : (tensor<2x3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<2x3x4x5xi64>
// CHECK: "tfl.less"({{.*}}, {{.*}}) : (tensor<2x3x4x5xi64>, tensor<i64>) -> tensor<2x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4x5xi64>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<2x4x5x3xf32>
