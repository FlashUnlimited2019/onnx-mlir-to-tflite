// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl="layout-mode=experimental" --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @less_equal_i64_broadcast(%lhs: tensor<1x1xi64>, %rhs: tensor<1x256xi64>) -> tensor<1x256xi1> {
    %0 = "onnx.LessOrEqual"(%lhs, %rhs) : (tensor<1x1xi64>, tensor<1x256xi64>) -> tensor<1x256xi1>
    return %0 : tensor<1x256xi1>
  }
  "onnx.EntryPoint"() {func = @less_equal_i64_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1xi64>{{( \{[^}]*\})?}}, %arg1: tensor<1x256xi64>{{( \{[^}]*\})?}}) -> (tensor<1x256xi1>{{( \{[^}]*\})?}})
// CHECK: "tfl.less_equal"(%arg0, %arg1) : (tensor<1x1xi64>, tensor<1x256xi64>) -> tensor<1x256xi1>

// -----

module {
  func.func @less_equal_rank4_rank3(%lhs: tensor<2x3x4x5xf32>, %rhs: tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1> {
    %0 = "onnx.LessOrEqual"(%lhs, %rhs) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>
    return %0 : tensor<2x3x4x5xi1>
  }
  "onnx.EntryPoint"() {func = @less_equal_rank4_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>{{( \{[^}]*\})?}}, %arg1: tensor<3x4x5xf32>{{( \{[^}]*\})?}}) -> (tensor<2x3x4x5xi1>{{( \{[^}]*\})?}})
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.less_equal"({{.*}}, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xi1>

// -----

module {
  func.func @less_equal_rank5_i64(%lhs: tensor<2x1x3x4x5xi64>, %rhs: tensor<1x3x1x5xi64>) -> tensor<2x1x3x4x5xi1> {
    %0 = "onnx.LessOrEqual"(%lhs, %rhs) : (tensor<2x1x3x4x5xi64>, tensor<1x3x1x5xi64>) -> tensor<2x1x3x4x5xi1>
    return %0 : tensor<2x1x3x4x5xi1>
  }
  "onnx.EntryPoint"() {func = @less_equal_rank5_i64} : () -> ()
}
// CHECK-LABEL: func.func @main(
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x1x3x4x5xi64>, tensor<1xi32>) -> tensor<120xi64>
// CHECK: "tfl.broadcast_to"(%arg1, {{.*}}) : (tensor<1x3x1x5xi64>, tensor<5xi32>) -> tensor<2x1x3x4x5xi64>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x1x3x4x5xi64>, tensor<1xi32>) -> tensor<120xi64>
// CHECK: "tfl.less_equal"({{.*}}) : (tensor<120xi64>, tensor<120xi64>) -> tensor<120xi1>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<120xi1>, tensor<5xi32>) -> tensor<2x1x3x4x5xi1>
