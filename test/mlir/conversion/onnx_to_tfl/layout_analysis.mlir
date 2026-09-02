// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-diagnostics=values' --verify-each %s 2>&1 | FileCheck %s

module {
  func.func @layout_diagnostics(%image: tensor<1x3x8x8xf32>, %filter: tensor<4x3x3x3xf32>, %lhs: tensor<1x2x4x8xf32>, %rhs: tensor<1x2x8x6xf32>) -> (tensor<1x4x6x6xf32>, tensor<1x2x4x6xf32>) {
    %none = "onnx.NoValue"() : () -> none
    %convolved = "onnx.Conv"(%image, %filter, %none) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x6x6xf32>
    %activated = "onnx.Relu"(%convolved) : (tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>
    %product = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x4x8xf32>, tensor<1x2x8x6xf32>) -> tensor<1x2x4x6xf32>
    return %activated, %product : tensor<1x4x6x6xf32>, tensor<1x2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @layout_diagnostics} : () -> ()
}

// CHECK: [onnx-to-tfl-layout] values=7 identity=4 channels-last=3 permuted=0 unknown=0 conflict=0
// CHECK: [onnx-to-tfl-layout] %arg0 rank=4 layout=channels-last
// CHECK: [onnx-to-tfl-layout] %arg1 rank=4 layout=identity
// CHECK: [onnx-to-tfl-layout] %1 rank=4 layout=channels-last
// CHECK: [onnx-to-tfl-layout] %2 rank=4 layout=channels-last
// CHECK: [onnx-to-tfl-layout] %arg2 rank=4 layout=identity
// CHECK: [onnx-to-tfl-layout] %arg3 rank=4 layout=identity
// CHECK: [onnx-to-tfl-layout] %3 rank=4 layout=identity
// CHECK: "tfl.conv_2d"
// CHECK: "tfl.batch_matmul"
// CHECK-NOT: onnx.
