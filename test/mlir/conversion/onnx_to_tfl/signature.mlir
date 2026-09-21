// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each --split-input-file %s | FileCheck %s

module {
  func.func @main_graph(%x: tensor<2x3xf32> {onnx.name = "z/input:0"}, %bias: tensor<3xf32> {onnx.name = "a.bias"}) -> (tensor<2x3xf32> {onnx.name = "result:0"}, tensor<2x3xf32> {onnx.name = "alias"}) {
    %sum = "onnx.Add"(%x, %bias) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
    return %sum, %sum : tensor<2x3xf32>, tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: module attributes {
// CHECK-SAME: tf_saved_model.semantics
// CHECK-SAME: onnx_mlir.signature_order
// CHECK-SAME: z/input:0
// CHECK-SAME: a.bias
// CHECK-SAME: result:0
// CHECK-SAME: alias
// CHECK: func.func @main(
// CHECK-SAME: tf_saved_model.index_path = ["z/input:0"]
// CHECK-SAME: tf_saved_model.index_path = ["a.bias"]
// CHECK-SAME: tf_saved_model.index_path = ["result:0"]
// CHECK-SAME: tf_saved_model.index_path = ["alias"]
// CHECK-SAME: inputs = "z/input:0,a.bias", outputs = "result:0,alias"
// CHECK-SAME: tf_saved_model.exported_names = ["serving_default"]
// CHECK: %[[SUM:.*]] = "tfl.add"
// CHECK: return %[[SUM]], %[[SUM]]

// -----

module {
  func.func @main_graph(%x: tensor<2x3xf32>, %bias: tensor<3xf32>) -> tensor<2x3xf32> {
    %sum = "onnx.Add"(%x, %bias) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
    return %sum : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: module attributes {
// CHECK-SAME: tf_saved_model.semantics
// CHECK-SAME: onnx_mlir.signature_order
// CHECK-SAME: input0
// CHECK-SAME: input1
// CHECK-SAME: output0
// CHECK: func.func @main(
// CHECK-SAME: tf_saved_model.index_path = ["input0"]
// CHECK-SAME: tf_saved_model.index_path = ["input1"]
// CHECK-SAME: tf_saved_model.index_path = ["output0"]
// CHECK-SAME: inputs = "input0,input1", outputs = "output0"
// CHECK-SAME: tf_saved_model.exported_names = ["serving_default"]

// -----

module {
  func.func @main_graph() -> (tensor<2xf32> {onnx.name = "constant"}) {
    %value = "onnx.Constant"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    return %value : tensor<2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: module attributes {
// CHECK-SAME: tf_saved_model.semantics
// CHECK-SAME: onnx_mlir.signature_order
// CHECK-SAME: constant
// CHECK: func.func @main()
// CHECK-SAME: tf_saved_model.index_path = ["constant"]
// CHECK-SAME: inputs = "", outputs = "constant"
// CHECK-SAME: tf_saved_model.exported_names = ["serving_default"]
