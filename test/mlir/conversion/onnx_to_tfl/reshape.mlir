// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x8xf32>) -> tensor<2x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<1x8xf32>, tensor<2xi64>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<[2, 4]> : tensor<2xi32>
// CHECK: "tfl.reshape"
// CHECK-NOT: onnx.
