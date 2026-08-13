// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @argmax_rank2_keep_first(%arg0: tensor<2x3xf32>) -> tensor<1x3xf32> {
    %0 = "onnx.ArgMax"(%arg0) <{axis = 0 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}> : (tensor<2x3xf32>) -> tensor<1x3xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x3xi64>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
  "onnx.EntryPoint"() {func = @argmax_rank2_keep_first} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: "tfl.arg_max"({{.*}}) : (tensor<2x3xf32>, tensor<i32>) -> tensor<3xi64>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<3xi64>, tensor<2xi32>) -> tensor<1x3xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x3xi64>) -> tensor<1x3xf32>

// -----

module {
  func.func @argmax_rank3_nokeep_last(%arg0: tensor<2x3x4xf32>) -> tensor<2x3xf32> {
    %0 = "onnx.ArgMax"(%arg0) <{axis = -1 : si64, keepdims = 0 : si64, select_last_index = 1 : si64}> : (tensor<2x3x4xf32>) -> tensor<2x3xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3xi64>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @argmax_rank3_nokeep_last} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x3xf32>
// CHECK: arith.constant dense<2> : tensor<1xi32>
// CHECK: "tfl.reverse_v2"(%arg0, {{.*}}) : (tensor<2x3x4xf32>, tensor<1xi32>) -> tensor<2x3x4xf32>
// CHECK: "tfl.arg_max"({{.*}}) : (tensor<2x3x4xf32>, tensor<i32>) -> tensor<2x3xi64>
// CHECK: "tfl.sub"({{.*}}) {fused_activation_function = "NONE"} : (tensor<i64>, tensor<2x3xi64>) -> tensor<2x3xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x3xi64>) -> tensor<2x3xf32>

// -----

module {
  func.func @argmax_rank4_keep_last(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x1x4xf32> {
    %0 = "onnx.ArgMax"(%arg0) <{axis = 2 : si64, keepdims = 1 : si64, select_last_index = 1 : si64}> : (tensor<1x2x3x4xf32>) -> tensor<1x2x1x4xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x1x4xi64>) -> tensor<1x2x1x4xf32>
    return %1 : tensor<1x2x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @argmax_rank4_keep_last} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x1x4x2xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.reverse_v2"({{.*}}) : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.arg_max"({{.*}}) : (tensor<1x2x3x4xf32>, tensor<i32>) -> tensor<1x2x4xi64>
// CHECK: "tfl.sub"({{.*}}) {fused_activation_function = "NONE"} : (tensor<i64>, tensor<1x2x4xi64>) -> tensor<1x2x4xi64>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x2x4xi64>, tensor<4xi32>) -> tensor<1x2x1x4xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<1x2x1x4xi64>) -> tensor<1x2x1x4xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x2x1x4xf32>, tensor<4xi32>) -> tensor<1x1x4x2xf32>

// -----

module {
  func.func @argmax_rank5_to_rank4(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x4x5x6xf32> {
    %0 = "onnx.ArgMax"(%arg0) <{axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}> : (tensor<2x3x4x5x6xf32>) -> tensor<2x4x5x6xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x4x5x6xi64>) -> tensor<2x4x5x6xf32>
    return %1 : tensor<2x4x5x6xf32>
  }
  "onnx.EntryPoint"() {func = @argmax_rank5_to_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x5x6x4xf32>
// CHECK: "tfl.arg_max"({{.*}}) : (tensor<2x3x4x5x6xf32>, tensor<i32>) -> tensor<2x4x5x6xi64>
// CHECK: "tfl.cast"({{.*}}) : (tensor<2x4x5x6xi64>) -> tensor<2x4x5x6xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x4x5x6xf32>, tensor<4xi32>) -> tensor<2x5x6x4xf32>
