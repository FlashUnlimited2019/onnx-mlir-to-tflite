// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s | FileCheck %s

// The selected count is genuinely dynamic. Only the complete padded consumer
// may be replaced; the output keeps every untouched i64 padding element.
module {
  func.func @padded_suppression(%boxes: tensor<2x5x4xf32>, %scores: tensor<2x3x5xf32>, %padding: tensor<12x3xi64>) -> tensor<12x3xi64> {
    %zero = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %one = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %classes = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
    %limit = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %axis0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %iou = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %threshold = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %selected = "onnx.NonMaxSuppression"(%boxes, %scores, %limit, %iou, %threshold) {center_point_box = 1 : si64} : (tensor<2x5x4xf32>, tensor<2x3x5xf32>, tensor<i64>, tensor<f32>, tensor<f32>) -> tensor<?x3xi64>
    %dim = "onnx.Dim"(%selected) {axis = 0 : si64} : (tensor<?x3xi64>) -> tensor<1xi64>
    %length = "onnx.Squeeze"(%dim, %axis0) : (tensor<1xi64>, tensor<1xi64>) -> tensor<i64>
    %positions = "onnx.Range"(%zero, %length, %one) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    %batch = "onnx.Gather"(%selected, %zero) {axis = 1 : si64} : (tensor<?x3xi64>, tensor<i64>) -> tensor<?xi64>
    %class = "onnx.Gather"(%selected, %one) {axis = 1 : si64} : (tensor<?x3xi64>, tensor<i64>) -> tensor<?xi64>
    %base = "onnx.Mul"(%batch, %classes) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %group = "onnx.Add"(%base, %class) : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
    %gi = "onnx.Unsqueeze"(%group, %axis1) : (tensor<?xi64>, tensor<1xi64>) -> tensor<?x1xi64>
    %gj = "onnx.Unsqueeze"(%group, %axis0) : (tensor<?xi64>, tensor<1xi64>) -> tensor<1x?xi64>
    %same = "onnx.Equal"(%gi, %gj) : (tensor<?x1xi64>, tensor<1x?xi64>) -> tensor<?x?xi1>
    %i = "onnx.Unsqueeze"(%positions, %axis1) : (tensor<?xi64>, tensor<1xi64>) -> tensor<?x1xi64>
    %j = "onnx.Unsqueeze"(%positions, %axis0) : (tensor<?xi64>, tensor<1xi64>) -> tensor<1x?xi64>
    %prefix = "onnx.LessOrEqual"(%j, %i) : (tensor<1x?xi64>, tensor<?x1xi64>) -> tensor<?x?xi1>
    %mask = "onnx.And"(%same, %prefix) : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
    %counts = "onnx.Cast"(%mask) {to = i64} : (tensor<?x?xi1>) -> tensor<?x?xi64>
    %count = "onnx.ReduceSumV11"(%counts) {axes = [1], keepdims = 0 : si64} : (tensor<?x?xi64>) -> tensor<?xi64>
    %ordinal = "onnx.Sub"(%count, %one) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %group_offset = "onnx.Mul"(%group, %limit) : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
    %offset = "onnx.Add"(%group_offset, %ordinal) : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
    %indices = "onnx.Unsqueeze"(%offset, %axis1) : (tensor<?xi64>, tensor<1xi64>) -> tensor<?x1xi64>
    %result = "onnx.ScatterND"(%padding, %indices, %selected) : (tensor<12x3xi64>, tensor<?x1xi64>, tensor<?x3xi64>) -> tensor<12x3xi64>
    return %result : tensor<12x3xi64>
  }
  "onnx.EntryPoint"() {func = @padded_suppression} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: "onnx.
// CHECK-NOT: tensor<?
// CHECK: "tfl.arg_max"
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.arg_max"
// CHECK: "tfl.select_v2"({{.*}}, %arg2) : (tensor<12x1xi1>, tensor<12x3xi64>, tensor<12x3xi64>) -> tensor<12x3xi64>
// CHECK: return {{.*}} : tensor<12x3xi64>
