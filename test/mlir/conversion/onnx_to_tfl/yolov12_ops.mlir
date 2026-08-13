// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x6x2x4xf32>, %rhs: tensor<1x2x4x3xf32>) -> tensor<1x2x2x3xf32> {
    %split_sizes = "onnx.Constant"() {value = dense<[2, 2, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %axes = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %part0, %part1, %part2 = "onnx.Split"(%input, %split_sizes) {axis = 1 : si64} : (tensor<1x6x2x4xf32>, tensor<3xi64>) -> (tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>)
    %scores = "onnx.MatMul"(%part0, %rhs) : (tensor<1x2x2x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x2x3xf32>
    %maximum = "onnx.ReduceMaxV13"(%scores) {axes = [-1], keepdims = 1 : si64} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x1xf32>
    %shifted = "onnx.Sub"(%scores, %maximum) : (tensor<1x2x2x3xf32>, tensor<1x2x2x1xf32>) -> tensor<1x2x2x3xf32>
    %exponent = "onnx.Exp"(%shifted) : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %sum = "onnx.ReduceSum"(%exponent, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x2x3xf32>, tensor<1xi64>) -> tensor<1x2x2x1xf32>
    %result = "onnx.Div"(%exponent, %sum) : (tensor<1x2x2x3xf32>, tensor<1x2x2x1xf32>) -> tensor<1x2x2x3xf32>
    return %result : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x6xf32>, %arg1: tensor<1x4x3x2xf32>) -> tensor<1x2x3x2xf32>
// CHECK: arith.constant dense<2> : tensor<3xi32>
// CHECK: arith.constant dense<3> : tensor<i32>
// CHECK: "tfl.split_v"(%arg0, {{.*}}, {{.*}}) {num_splits = 3 : i32} : (tensor<1x2x4x6xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<1x2x4x2xf32>, tensor<1x2x4x2xf32>, tensor<1x2x4x2xf32>)
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x2x4xf32>, tensor<1x2x4x3xf32>) -> tensor<1x2x2x3xf32>
// CHECK: "tfl.reduce_max"{{.*}} {keep_dims = true} : (tensor<1x2x2x3xf32>, tensor<1xi32>) -> tensor<1x2x2x1xf32>
// CHECK: "tfl.exp"
// CHECK: "tfl.sum"{{.*}} {keep_dims = true} : (tensor<1x2x2x3xf32>, tensor<1xi32>) -> tensor<1x2x2x1xf32>
// CHECK-NOT: onnx.
