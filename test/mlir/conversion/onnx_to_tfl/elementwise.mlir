// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "onnx.Identity"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "onnx.Sub"(%0, %arg1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %2 = "onnx.Mul"(%1, %arg1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %3 = "onnx.Div"(%2, %arg1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %4 = "onnx.Sigmoid"(%3) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %5 = "onnx.Tanh"(%4) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %5 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: onnx.Identity
// CHECK: "tfl.sub"(%arg0, %arg1) {fused_activation_function = "NONE"}
// CHECK: "tfl.mul"
// CHECK: "tfl.div"
// CHECK: "tfl.logistic"
// CHECK: "tfl.tanh"
// CHECK-NOT: onnx.
