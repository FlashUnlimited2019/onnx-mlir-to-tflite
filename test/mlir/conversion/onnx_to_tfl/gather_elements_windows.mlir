// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=legacy --split-input-file --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --split-input-file --verify-each %s | FileCheck %s

module {
  func.func @sliding(%data: tensor<6x7xf32>) -> tensor<6x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[4, 5, 6], [2, 3, 4], [0, 1, 2], [4, 5, 6], [2, 3, 4], [0, 1, 2]]> : tensor<6x3xi64>} : () -> tensor<6x3xi64>
    %y = "onnx.GatherElements"(%data, %indices) {axis = 1 : si64} : (tensor<6x7xf32>, tensor<6x3xi64>) -> tensor<6x3xf32>
    return %y : tensor<6x3xf32>
  }
  "onnx.EntryPoint"() {func = @sliding} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x21xf32>
// CHECK: "tfl.slice"{{.*}} -> tensor<2x15xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<6x5xf32>
// CHECK: "tfl.slice"{{.*}} -> tensor<6x3xf32>
// CHECK-NOT: "tfl.gather_nd"
// CHECK: return

// -----

module {
  func.func @fixed_negative(%data: tensor<2x5xf32>) -> tensor<2x2xf32> {
    %indices = "onnx.Constant"() {value = dense<[[-3, -2], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %y = "onnx.GatherElements"(%data, %indices) {axis = -1 : si64} : (tensor<2x5xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
    return %y : tensor<2x2xf32>
  }
  "onnx.EntryPoint"() {func = @fixed_negative} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.gather_nd"
// CHECK: "tfl.slice"{{.*}} -> tensor<2x2xf32>
// CHECK-NOT: "tfl.gather_nd"
// CHECK: return

// -----

module {
  func.func @irregular(%data: tensor<3x5xf32>) -> tensor<3x3xf32> {
    %indices = "onnx.Constant"() {value = dense<[[2, 3, 4], [1, 3, 3], [0, 1, 2]]> : tensor<3x3xi64>} : () -> tensor<3x3xi64>
    %y = "onnx.GatherElements"(%data, %indices) {axis = 1 : si64} : (tensor<3x5xf32>, tensor<3x3xi64>) -> tensor<3x3xf32>
    return %y : tensor<3x3xf32>
  }
  "onnx.EntryPoint"() {func = @irregular} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"
// CHECK: return
