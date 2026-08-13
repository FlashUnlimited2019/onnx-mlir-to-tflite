// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @equal_rank4_rank3(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
    return %1 : tensor<2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @equal_rank4_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<2x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.equal"({{.*}}, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<2x4x5x3xf32>

// -----

module {
  func.func @equal_rank5_broadcast(%arg0: tensor<2x3x2x4x5xf32>, %arg1: tensor<3x2x4x5xf32>) -> tensor<2x3x2x4x5xf32> {
    %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x3x2x4x5xf32>, tensor<3x2x4x5xf32>) -> tensor<2x3x2x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x2x4x5xi1>) -> tensor<2x3x2x4x5xf32>
    return %1 : tensor<2x3x2x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @equal_rank5_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x2x4x5xf32>, %arg1: tensor<3x4x5x2xf32>) -> tensor<2x3x2x4x5xf32>
// CHECK: "tfl.transpose"(%arg1, {{.*}}) : (tensor<3x4x5x2xf32>, tensor<4xi32>) -> tensor<3x2x4x5xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x2x4x5xf32>, tensor<1xi32>) -> tensor<240xf32>
// CHECK: "tfl.broadcast_to"({{.*}}, {{.*}}) : (tensor<3x2x4x5xf32>, tensor<5xi32>) -> tensor<2x3x2x4x5xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x3x2x4x5xf32>, tensor<1xi32>) -> tensor<240xf32>
// CHECK: "tfl.equal"({{.*}}, {{.*}}) : (tensor<240xf32>, tensor<240xf32>) -> tensor<240xi1>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<240xi1>, tensor<5xi32>) -> tensor<2x3x2x4x5xi1>

// -----

module {
  func.func @sign_rank4(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.Sign"(%arg0) : (tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @sign_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK: "tfl.sign"(%arg0) : (tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>

// -----

module {
  func.func @fmod_rank5(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
    %0 = "onnx.Mod"(%arg0, %arg1) <{fmod = 1 : si64}> : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
    return %0 : tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @fmod_rank5} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3x4x5xf32>, tensor<1xi32>) -> tensor<120xf32>
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x2x3x4x5xf32>, tensor<1xi32>) -> tensor<120xf32>
// CHECK: "tfl.floor_mod"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
// CHECK: "tfl.not_equal"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<f32>) -> tensor<120xi1>
// CHECK: "tfl.less"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<f32>) -> tensor<120xi1>
// CHECK: "tfl.less"({{.*}}, {{.*}}) : (tensor<120xf32>, tensor<f32>) -> tensor<120xi1>
// CHECK: "tfl.not_equal"({{.*}}, {{.*}}) : (tensor<120xi1>, tensor<120xi1>) -> tensor<120xi1>
// CHECK: "tfl.sub"({{.*}}, {{.*}}) {fused_activation_function = "NONE"} : (tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
// CHECK: "tfl.select_v2"({{.*}}, {{.*}}, {{.*}}) : (tensor<120xi1>, tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
// CHECK: "tfl.select_v2"({{.*}}, {{.*}}, {{.*}}) : (tensor<120xi1>, tensor<120xf32>, tensor<120xf32>) -> tensor<120xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<120xf32>, tensor<5xi32>) -> tensor<1x2x3x4x5xf32>
