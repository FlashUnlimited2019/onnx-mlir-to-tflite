// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @maximum_first(%input: tensor<2x3xi64>) -> tensor<2xi64> {
    %y = "onnx.ArgMax"(%input) <{axis = -1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}> : (tensor<2x3xi64>) -> tensor<2xi64>
    return %y : tensor<2xi64>
  }
  "onnx.EntryPoint"() {func = @maximum_first} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.cast"{{.*}} : (tensor<2x3xi64>) -> tensor<2x3xi32>
// CHECK: "tfl.arg_max"{{.*}} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2xi64>
// CHECK-NOT: "tfl.cast"
// CHECK: return

// -----

module {
  func.func @minimum_last(%input: tensor<2x3x4xi64>) -> tensor<2x1x4xi64> {
    %y = "onnx.ArgMin"(%input) <{axis = 1 : si64, keepdims = 1 : si64, select_last_index = 1 : si64}> : (tensor<2x3x4xi64>) -> tensor<2x1x4xi64>
    return %y : tensor<2x1x4xi64>
  }
  "onnx.EntryPoint"() {func = @minimum_last} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.cast"{{.*}} : (tensor<2x3x4xi64>) -> tensor<2x3x4xi32>
// CHECK: "tfl.reverse_v2"{{.*}} -> tensor<2x3x4xi32>
// CHECK: "tfl.arg_min"{{.*}} -> tensor<2x4xi64>
// CHECK: "tfl.sub"{{.*}} -> tensor<2x4xi64>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x1x4xi64>
// CHECK: return

// -----

module {
  func.func @singleton(%input: tensor<2x1x3xi64>) -> tensor<2x3xi64> {
    %y = "onnx.ArgMax"(%input) <{axis = 1 : si64, keepdims = 0 : si64, select_last_index = 1 : si64}> : (tensor<2x1x3xi64>) -> tensor<2x3xi64>
    return %y : tensor<2x3xi64>
  }
  "onnx.EntryPoint"() {func = @singleton} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.cast"{{.*}} -> tensor<2x1x3xi32>
// CHECK: "tfl.reverse_v2"{{.*}} -> tensor<2x1x3xi32>
// CHECK: "tfl.arg_max"{{.*}} -> tensor<2x3xi64>
// CHECK: "tfl.sub"{{.*}} -> tensor<2x3xi64>
// CHECK: return
