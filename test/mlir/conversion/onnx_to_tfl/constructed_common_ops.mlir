// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %slope = onnx.Constant dense<[0.1, 0.2, 0.3, 0.4]> : tensor<4xf32>
    %axis = onnx.Constant dense<1> : tensor<i64>
    %zero = onnx.Constant dense<0.0> : tensor<f32>
    %0 = "onnx.PRelu"(%arg0, %slope) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %1 = "onnx.Softplus"(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "onnx.Softsign"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = "onnx.Selu"(%2) <{alpha = 1.67326 : f32, gamma = 1.0507 : f32}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %4 = "onnx.Elu"(%3) <{alpha = 1.1 : f32}> : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %5 = "onnx.Log"(%4) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %6 = "onnx.Erf"(%5) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %7 = "onnx.CumSum"(%6, %axis) <{exclusive = 0 : si64, reverse = 0 : si64}> : (tensor<2x4xf32>, tensor<i64>) -> tensor<2x4xf32>
    %8 = "onnx.Mean"(%6, %7, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %9 = "onnx.Greater"(%8, %zero) : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xi1>
    %10 = "onnx.And"(%9, %9) : (tensor<2x4xi1>, tensor<2x4xi1>) -> tensor<2x4xi1>
    %11 = "onnx.Or"(%10, %9) : (tensor<2x4xi1>, tensor<2x4xi1>) -> tensor<2x4xi1>
    %12 = "onnx.Xor"(%11, %10) : (tensor<2x4xi1>, tensor<2x4xi1>) -> tensor<2x4xi1>
    %13 = "onnx.Where"(%12, %8, %7) : (tensor<2x4xi1>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %13 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.prelu"
// CHECK: "tfl.log"
// CHECK: "tfl.cumsum"
// CHECK: "tfl.logical_and"
// CHECK: "tfl.logical_or"
// CHECK: "tfl.not_equal"
// CHECK: "tfl.select_v2"
// CHECK-NOT: onnx.
