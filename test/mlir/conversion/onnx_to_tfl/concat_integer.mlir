// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @integer_mask_concat(%x: tensor<1x1xi64>) -> tensor<1x5xi64> {
    %past = "onnx.Constant"() {value = dense<1> : tensor<1x4xi64>} : () -> tensor<1x4xi64>
    %y = "onnx.Concat"(%past, %x) {axis = -1 : si64} : (tensor<1x4xi64>, tensor<1x1xi64>) -> tensor<1x5xi64>
    return %y : tensor<1x5xi64>
  }
  "onnx.EntryPoint"() {func = @integer_mask_concat} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.cast"
// CHECK: "tfl.concatenation"({{.*}}) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x4xi64>, tensor<1x1xi64>) -> tensor<1x5xi64>
// CHECK-NOT: "tfl.cast"
// CHECK: return

// -----

// Integer rank-four data and axes must keep their ONNX order in both modes.
module {
  func.func @integer_rank4(%a: tensor<2x3x4x5xi32>, %b: tensor<2x2x4x5xi32>) -> tensor<2x5x4x5xi32> {
    %y = "onnx.Concat"(%a, %b) {axis = 1 : si64} : (tensor<2x3x4x5xi32>, tensor<2x2x4x5xi32>) -> tensor<2x5x4x5xi32>
    return %y : tensor<2x5x4x5xi32>
  }
  "onnx.EntryPoint"() {func = @integer_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.concatenation"(%arg0, %arg1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<2x3x4x5xi32>, tensor<2x2x4x5xi32>) -> tensor<2x5x4x5xi32>
// CHECK-NOT: "tfl.transpose"
// CHECK: return
