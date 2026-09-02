// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @transposed_scatter_nd_slice_update(%data: tensor<1x2x3x5xf32>, %updates: tensor<1x2x3x2xf32>) -> tensor<1x2x3x5xf32> {
    %indices = "onnx.Constant"() {value = dense<[[1], [2]]> : tensor<2x1xi64>} : () -> tensor<2x1xi64>
    %data_moved = "onnx.Transpose"(%data) <{perm = [3, 1, 2, 0]}> : (tensor<1x2x3x5xf32>) -> tensor<5x2x3x1xf32>
    %updates_moved = "onnx.Transpose"(%updates) <{perm = [3, 1, 2, 0]}> : (tensor<1x2x3x2xf32>) -> tensor<2x2x3x1xf32>
    %updated = "onnx.ScatterND"(%data_moved, %indices, %updates_moved) <{reduction = "none"}> : (tensor<5x2x3x1xf32>, tensor<2x1xi64>, tensor<2x2x3x1xf32>) -> tensor<5x2x3x1xf32>
    %restored = "onnx.Transpose"(%updated) <{perm = [3, 1, 2, 0]}> : (tensor<5x2x3x1xf32>) -> tensor<1x2x3x5xf32>
    return %restored : tensor<1x2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @transposed_scatter_nd_slice_update} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: "tfl.slice"
// CHECK: "tfl.concatenation"
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: "tfl.gather_nd"
// CHECK-NOT: "tfl.scatter_nd"

// -----

module {
  func.func @scatter_elements_slice_update(%data: tensor<2x5xf32>, %updates: tensor<2x2xf32>) -> tensor<2x5xf32> {
    %indices = "onnx.Constant"() {value = dense<[[1, 2], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
    %updated = "onnx.ScatterElements"(%data, %indices, %updates) <{axis = 1 : si64, reduction = "none"}> : (tensor<2x5xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<2x5xf32>
    return %updated : tensor<2x5xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_elements_slice_update} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: "tfl.slice"
// CHECK: "tfl.concatenation"
// CHECK-NOT: "tfl.gather_nd"
// CHECK-NOT: "tfl.scatter_nd"

// -----

module {
  func.func @scatter_elements_irregular_fallback(%data: tensor<2x5xf32>, %updates: tensor<2x2xf32>) -> tensor<2x5xf32> {
    %indices = "onnx.Constant"() {value = dense<[[1, 3], [1, 3]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
    %updated = "onnx.ScatterElements"(%data, %indices, %updates) <{axis = 1 : si64, reduction = "none"}> : (tensor<2x5xf32>, tensor<2x2xi64>, tensor<2x2xf32>) -> tensor<2x5xf32>
    return %updated : tensor<2x5xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_elements_irregular_fallback} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.scatter_nd"
// CHECK-NOT: onnx.
