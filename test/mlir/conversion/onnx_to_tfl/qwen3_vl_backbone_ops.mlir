// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x3x8xf32>) -> (tensor<1x3x8xf32>, tensor<1x3x8xf32>) {
    %0 = "onnx.Cos"(%arg0) : (tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
    %1 = "onnx.Sin"(%arg0) : (tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
    return %0, %1 : tensor<1x3x8xf32>, tensor<1x3x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.cos"(%arg0) : (tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
// CHECK: "tfl.sin"(%arg0) : (tensor<1x3x8xf32>) -> tensor<1x3x8xf32>
// CHECK-NOT: onnx.
