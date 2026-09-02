// SPDX-License-Identifier: Apache-2.0

// Copyright 2026 FlashUnlimited2019.

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s | FileCheck %s

module {
  func.func @expand_bool_rank4(%input: tensor<1x1x1x6xi1>) -> tensor<1x1x6x6xi1> {
    %shape = "onnx.Constant"() {value = dense<[1, 1, 6, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
    %result = "onnx.Expand"(%input, %shape) : (tensor<1x1x1x6xi1>, tensor<4xi64>) -> tensor<1x1x6x6xi1>
    return %result : tensor<1x1x6x6xi1>
  }
  "onnx.EntryPoint"() {func = @expand_bool_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x1x1x6xi1>) -> tensor<1x1x6x6xi1>
// CHECK: arith.constant dense<[1, 1, 6, 6]> : tensor<4xi32>
// CHECK: "tfl.broadcast_to"(%arg0, {{.*}}) : (tensor<1x1x1x6xi1>, tensor<4xi32>) -> tensor<1x1x6x6xi1>
// CHECK-NOT: onnx.
