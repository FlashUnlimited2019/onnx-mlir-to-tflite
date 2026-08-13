// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @greater_rank4_rank3(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.Greater"(%arg0, %arg1) : (tensor<1x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<1x3x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x3x4x5xi1>) -> tensor<1x3x4x5xf32>
    return %1 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @greater_rank4_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<1x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x4x5x3xf32>, tensor<4xi32>) -> tensor<1x3x4x5xf32>
// CHECK: "tfl.greater"({{.*}}, %arg1) : (tensor<1x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<1x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x3x4x5xi1>) -> tensor<1x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x4x5xf32>, tensor<4xi32>) -> tensor<1x4x5x3xf32>

// -----

module {
  func.func @greater_rank4_rank2(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<2x3x4x5xf32> {
    %0 = "onnx.Greater"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<4x5xf32>) -> tensor<2x3x4x5xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
    return %1 : tensor<2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @greater_rank4_rank2} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<2x4x5x3xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.greater"({{.*}}, %arg1) : (tensor<2x3x4x5xf32>, tensor<4x5xf32>) -> tensor<2x3x4x5xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4x5xi1>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x3x4x5xf32>, tensor<4xi32>) -> tensor<2x4x5x3xf32>

// -----

module {
  func.func @greater_rank3_broadcast(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<2x3x4xf32> {
    %0 = "onnx.Greater"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<1x4xf32>) -> tensor<2x3x4xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x4xi1>) -> tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @greater_rank3_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<2x3x4xf32>
// CHECK: "tfl.greater"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<1x4xf32>) -> tensor<2x3x4xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x4xi1>) -> tensor<2x3x4xf32>

// -----

module {
  func.func @round_rank4(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.Round"(%arg0) : (tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @round_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK: "tfl.round"(%arg0) : (tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>

// -----

module {
  func.func @greater_rank5_broadcast(%arg0: tensor<2x3x2x3x4xf32>, %arg1: tensor<1x3x2x1x1xf32>) -> tensor<2x3x2x3x4xf32> {
    %0 = "onnx.Greater"(%arg0, %arg1) : (tensor<2x3x2x3x4xf32>, tensor<1x3x2x1x1xf32>) -> tensor<2x3x2x3x4xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3x2x3x4xi1>) -> tensor<2x3x2x3x4xf32>
    return %1 : tensor<2x3x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @greater_rank5_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x2x3x4xf32>, %arg1: tensor<1x3x2x1x1xf32>) -> tensor<2x3x2x3x4xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x2x3x4xf32>, tensor<1xi32>) -> tensor<144xf32>
// CHECK: "tfl.broadcast_to"(%arg1, {{.*}}) : (tensor<1x3x2x1x1xf32>, tensor<5xi32>) -> tensor<2x3x2x3x4xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x3x2x3x4xf32>, tensor<1xi32>) -> tensor<144xf32>
// CHECK: "tfl.greater"({{.*}}, {{.*}}) : (tensor<144xf32>, tensor<144xf32>) -> tensor<144xi1>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<144xi1>, tensor<5xi32>) -> tensor<2x3x2x3x4xi1>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3x2x3x4xi1>) -> tensor<2x3x2x3x4xf32>
