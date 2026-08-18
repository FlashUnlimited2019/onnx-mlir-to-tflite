// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @logical_broadcast(%arg0: tensor<1x2x3x4xi1>, %arg1: tensor<1x2x1x1xi1>) -> tensor<1x2x3x4xi1> {
    %0 = "onnx.And"(%arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x1x1xi1>) -> tensor<1x2x3x4xi1>
    return %0 : tensor<1x2x3x4xi1>
  }
  "onnx.EntryPoint"() {func = @logical_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xi1>, %arg1: tensor<1x2x1x1xi1>) -> tensor<1x2x3x4xi1>
// CHECK: "tfl.logical_and"(%arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x1x1xi1>) -> tensor<1x2x3x4xi1>

// -----

module {
  func.func @logical_tensor_ops(%arg0: tensor<1x2x3x4xi1>) -> tensor<1x96xi1> {
    %zero = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %shape = "onnx.Constant"() {value = dense<[1, 24]> : tensor<2xi64>} : () -> tensor<2xi64>
    %start = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %end = "onnx.Constant"() {value = dense<-9223372036854775807> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
    %step = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Squeeze"(%arg0, %zero) : (tensor<1x2x3x4xi1>, tensor<1xi64>) -> tensor<2x3x4xi1>
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xi1>) -> tensor<1x3x2x4xi1>
    %2 = "onnx.Reshape"(%1, %shape) {allowzero = 0 : si64} : (tensor<1x3x2x4xi1>, tensor<2xi64>) -> tensor<1x24xi1>
    %3 = "onnx.Not"(%arg0) : (tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
    %4 = "onnx.Slice"(%arg0, %start, %end, %axis, %step) : (tensor<1x2x3x4xi1>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2x3x4xi1>
    %5 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<2x3x4xi1>, tensor<2xi64>) -> tensor<1x24xi1>
    %6 = "onnx.Reshape"(%3, %shape) {allowzero = 0 : si64} : (tensor<1x2x3x4xi1>, tensor<2xi64>) -> tensor<1x24xi1>
    %7 = "onnx.Reshape"(%4, %shape) {allowzero = 0 : si64} : (tensor<1x2x3x4xi1>, tensor<2xi64>) -> tensor<1x24xi1>
    %8 = "onnx.Concat"(%5, %2, %6, %7) {axis = 1 : si64} : (tensor<1x24xi1>, tensor<1x24xi1>, tensor<1x24xi1>, tensor<1x24xi1>) -> tensor<1x96xi1>
    return %8 : tensor<1x96xi1>
  }
  "onnx.EntryPoint"() {func = @logical_tensor_ops} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xi1>) -> tensor<1x96xi1>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3x4xi1>, tensor<3xi32>) -> tensor<2x3x4xi1>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x2x3x4xi1>, tensor<4xi32>) -> tensor<1x3x2x4xi1>
// CHECK: "tfl.reshape"({{.*}}, {{.*}}) : (tensor<1x3x2x4xi1>, tensor<2xi32>) -> tensor<1x24xi1>
// CHECK: "tfl.logical_not"(%arg0) : (tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
// CHECK: "tfl.reverse_v2"(%arg0, {{.*}}) : (tensor<1x2x3x4xi1>, tensor<1xi32>) -> tensor<1x2x3x4xi1>
// CHECK: "tfl.concatenation"{{.*}} : (tensor<1x24xi1>, tensor<1x24xi1>, tensor<1x24xi1>, tensor<1x24xi1>) -> tensor<1x96xi1>

// -----

module {
  func.func @logical_rank5_broadcast(%arg0: tensor<2x3x2x3x4xi1>, %arg1: tensor<1x3x2x1x1xi1>) -> tensor<2x3x2x3x4xi1> {
    %0 = "onnx.And"(%arg0, %arg1) : (tensor<2x3x2x3x4xi1>, tensor<1x3x2x1x1xi1>) -> tensor<2x3x2x3x4xi1>
    return %0 : tensor<2x3x2x3x4xi1>
  }
  "onnx.EntryPoint"() {func = @logical_rank5_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x2x3x4xi1>, %arg1: tensor<1x3x2x1x1xi1>) -> tensor<2x3x2x3x4xi1>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x2x3x4xi1>, tensor<1xi32>) -> tensor<144xi1>
// CHECK: "tfl.broadcast_to"(%arg1, {{.*}}) : (tensor<1x3x2x1x1xi1>, tensor<5xi32>) -> tensor<2x3x2x3x4xi1>
// CHECK: "tfl.reshape"({{.*}}, {{.*}}) : (tensor<2x3x2x3x4xi1>, tensor<1xi32>) -> tensor<144xi1>
// CHECK: "tfl.logical_and"{{.*}} : (tensor<144xi1>, tensor<144xi1>) -> tensor<144xi1>
// CHECK: "tfl.reshape"{{.*}} : (tensor<144xi1>, tensor<5xi32>) -> tensor<2x3x2x3x4xi1>
