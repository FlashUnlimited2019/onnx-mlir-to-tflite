// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @flatten_rank4(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x24xf32> {
    %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<1x2x3x4xf32>) -> tensor<1x24xf32>
    return %0 : tensor<1x24xf32>
  }
  "onnx.EntryPoint"() {func = @flatten_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>
// CHECK: arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
// CHECK: arith.constant dense<[1, 24]> : tensor<2xi32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<1x24xf32>

// -----

module {
  func.func @flatten_rank3(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
    %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
    return %0 : tensor<6x4xf32>
  }
  "onnx.EntryPoint"() {func = @flatten_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK: arith.constant dense<[6, 4]> : tensor<2xi32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
