// SPDX-License-Identifier: Apache-2.0
// RUN: onnx-mlir-opt --canonicalize --verify-each --split-input-file %s | FileCheck %s

// Removing multiple empty operands must not shift the remaining erase indices.
func.func @empty_ends(%empty: tensor<1x2x3x0xf32>, %data: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %0 = "onnx.Concat"(%empty, %data, %empty) {axis = 3 : si64} : (tensor<1x2x3x0xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x0xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
// CHECK-LABEL: func.func @empty_ends(
// CHECK-SAME: %[[EMPTY:.*]]: tensor<1x2x3x0xf32>, %[[DATA:.*]]: tensor<1x2x3x4xf32>
// CHECK-NEXT: return %[[DATA]] : tensor<1x2x3x4xf32>

// -----

func.func @negative_axis_empty_ends(%empty: tensor<1x2x3x0xf32>, %data: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %0 = "onnx.Concat"(%empty, %data, %empty) {axis = -1 : si64} : (tensor<1x2x3x0xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x0xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}
// CHECK-LABEL: func.func @negative_axis_empty_ends(
// CHECK-SAME: %[[EMPTY:.*]]: tensor<1x2x3x0xf32>, %[[DATA:.*]]: tensor<1x2x3x4xf32>
// CHECK-NEXT: return %[[DATA]] : tensor<1x2x3x4xf32>

// -----

// Nonempty operands must retain their order, including dynamic dimensions.
func.func @interleaved_empty(%empty: tensor<?x0x4xf32>, %a: tensor<?x2x4xf32>, %b: tensor<?x3x4xf32>) -> tensor<?x5x4xf32> {
  %0 = "onnx.Concat"(%empty, %a, %empty, %b) {axis = 1 : si64} : (tensor<?x0x4xf32>, tensor<?x2x4xf32>, tensor<?x0x4xf32>, tensor<?x3x4xf32>) -> tensor<?x5x4xf32>
  return %0 : tensor<?x5x4xf32>
}
// CHECK-LABEL: func.func @interleaved_empty(
// CHECK-SAME: %[[EMPTY:.*]]: tensor<?x0x4xf32>, %[[A:.*]]: tensor<?x2x4xf32>, %[[B:.*]]: tensor<?x3x4xf32>
// CHECK-NEXT: %[[OUT:.*]] = "onnx.Concat"(%[[A]], %[[B]])
// CHECK-NEXT: return %[[OUT]] : tensor<?x5x4xf32>

// -----

func.func @empty_vectors(%empty: tensor<0xf32>, %data: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Concat"(%empty, %data, %empty) {axis = 0 : si64} : (tensor<0xf32>, tensor<4xf32>, tensor<0xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @empty_vectors(
// CHECK-SAME: %[[EMPTY:.*]]: tensor<0xf32>, %[[DATA:.*]]: tensor<4xf32>
// CHECK-NEXT: return %[[DATA]] : tensor<4xf32>

// -----

// Do not create an invalid Concat with no operands when every input is empty.
func.func @all_empty_vectors(%a: tensor<0xf32>, %b: tensor<0xf32>) -> tensor<0xf32> {
  %0 = "onnx.Concat"(%a, %b) {axis = 0 : si64} : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
  return %0 : tensor<0xf32>
}
// CHECK-LABEL: func.func @all_empty_vectors(
// CHECK-SAME: %[[A:.*]]: tensor<0xf32>, %[[B:.*]]: tensor<0xf32>
// CHECK-NEXT: %[[OUT:.*]] = "onnx.Concat"(%[[A]], %[[B]])
// CHECK-NEXT: return %[[OUT]] : tensor<0xf32>

// -----

func.func @all_empty_rank3(%a: tensor<2x0x4xf32>, %b: tensor<2x0x4xf32>) -> tensor<2x0x4xf32> {
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64} : (tensor<2x0x4xf32>, tensor<2x0x4xf32>) -> tensor<2x0x4xf32>
  return %0 : tensor<2x0x4xf32>
}
// CHECK-LABEL: func.func @all_empty_rank3(
// CHECK-SAME: %[[A:.*]]: tensor<2x0x4xf32>, %[[B:.*]]: tensor<2x0x4xf32>
// CHECK-NEXT: %[[OUT:.*]] = "onnx.Concat"(%[[A]], %[[B]])
// CHECK-NEXT: return %[[OUT]] : tensor<2x0x4xf32>

// -----

// An empty dimension outside the concatenation axis is not removable here.
func.func @empty_other_axis(%a: tensor<0x2xf32>, %b: tensor<0x3xf32>) -> tensor<0x5xf32> {
  %0 = "onnx.Concat"(%a, %b) {axis = 1 : si64} : (tensor<0x2xf32>, tensor<0x3xf32>) -> tensor<0x5xf32>
  return %0 : tensor<0x5xf32>
}
// CHECK-LABEL: func.func @empty_other_axis(
// CHECK-SAME: %[[A:.*]]: tensor<0x2xf32>, %[[B:.*]]: tensor<0x3xf32>
// CHECK-NEXT: %[[OUT:.*]] = "onnx.Concat"(%[[A]], %[[B]])
// CHECK-NEXT: return %[[OUT]] : tensor<0x5xf32>
