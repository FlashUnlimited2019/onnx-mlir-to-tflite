// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @average_pool_1d_oversized(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x1xf32> {
    %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [5], pads = [1, 0], strides = [5]} : (tensor<1x2x3xf32>) -> tensor<1x2x1xf32>
    return %0 : tensor<1x2x1xf32>
  }
  "onnx.EntryPoint"() {func = @average_pool_1d_oversized} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3xf32>
// CHECK: "tfl.slice"
// CHECK: "tfl.mean"({{.*}}) {keep_dims = true} : (tensor<1x2x3xf32>, tensor<1xi32>) -> tensor<1x2x1xf32>

// -----

module {
  func.func @average_pool_2d_include_pad(%arg0: tensor<1x2x3x2xf32>) -> tensor<1x2x2x1xf32> {
    %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 1 : si64, count_include_pad = 1 : si64, kernel_shape = [2, 4], pads = [1, 1, 0, 0], strides = [2, 4]} : (tensor<1x2x3x2xf32>) -> tensor<1x2x2x1xf32>
    return %0 : tensor<1x2x2x1xf32>
  }
  "onnx.EntryPoint"() {func = @average_pool_2d_include_pad} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x2x2xf32>
// CHECK: "tfl.mean"
// CHECK: "tfl.mul"
// CHECK: "tfl.mean"
// CHECK: "tfl.mul"
// CHECK: "tfl.concatenation"({{.*}}) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x2x1x2xf32>

// -----

module {
  func.func @average_pool_3d(%arg0: tensor<1x1x2x2x2xf32>) -> tensor<1x1x1x1x1xf32> {
    %0 = "onnx.AveragePool"(%arg0) {ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [3, 2, 3], pads = [1, 0, 1, 0, 0, 0], strides = [3, 2, 3]} : (tensor<1x1x2x2x2xf32>) -> tensor<1x1x1x1x1xf32>
    return %0 : tensor<1x1x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @average_pool_3d} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x2x2x2xf32>
// CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi32>
// CHECK: "tfl.mean"({{.*}}) {keep_dims = true} : (tensor<1x1x2x2x2xf32>, tensor<3xi32>) -> tensor<1x1x1x1x1xf32>
