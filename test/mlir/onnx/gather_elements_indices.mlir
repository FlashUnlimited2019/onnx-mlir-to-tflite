// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid_bounds(%data: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %idx = "onnx.Constant"() {value = dense<[[-3, 2], [-1, 0]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %y = "onnx.GatherElements"(%data, %idx) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// -----

func.func @invalid_positive(%data: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %idx = "onnx.Constant"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // expected-error @+1 {{indices}}
  %y = "onnx.GatherElements"(%data, %idx) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}

// -----

func.func @invalid_negative(%data: tensor<2x3xf32>) -> tensor<2x2xf32> {
  %idx = "onnx.Constant"() {value = dense<[[0, 1], [2, -9223372036854775808]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  // expected-error @+1 {{indices}}
  %y = "onnx.GatherElements"(%data, %idx) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  return %y : tensor<2x2xf32>
}
