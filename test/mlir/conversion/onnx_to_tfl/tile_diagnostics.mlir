// SPDX-License-Identifier: Apache-2.0

// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %s -split-input-file 2>&1 | FileCheck %s
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl="layout-mode=experimental" %s -split-input-file 2>&1 | FileCheck %s

module {
  func.func @runtime_repeats(%input: tensor<2xi64>, %repeats: tensor<1xi64>) -> tensor<4xi64> {
    %0 = "onnx.Tile"(%input, %repeats) : (tensor<2xi64>, tensor<1xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @runtime_repeats} : () -> ()
}
// CHECK: ONNXToTFL Tile requires one constant repeat per axis

// -----

module {
  func.func @irreducible_rank6(%input: tensor<2x2x2x2x2x2xi32>) -> tensor<4x4x4x4x4x4xi32> {
    %repeats = "onnx.Constant"() {value = dense<2> : tensor<6xi64>} : () -> tensor<6xi64>
    %0 = "onnx.Tile"(%input, %repeats) : (tensor<2x2x2x2x2x2xi32>, tensor<6xi64>) -> tensor<4x4x4x4x4x4xi32>
    return %0 : tensor<4x4x4x4x4x4xi32>
  }
  "onnx.EntryPoint"() {func = @irreducible_rank6} : () -> ()
}
// CHECK: high-rank Tile cannot be reduced to rank 5 by removing inert singleton axes
