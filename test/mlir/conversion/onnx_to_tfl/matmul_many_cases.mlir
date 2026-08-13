// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s | FileCheck %s

module {
  func.func @main_graph(
      %a3: tensor<6x4x5xf32>, %b3: tensor<1x5x7xf32>,
      %a4: tensor<1x2x12x5xf32>, %b4: tensor<1x1x5x8xf32>,
      %a5: tensor<1x2x3x4x5xf32>, %b5: tensor<1x1x3x5x6xf32>)
      -> tensor<1xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %shape = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %m3 = "onnx.MatMul"(%a3, %b3) : (tensor<6x4x5xf32>, tensor<1x5x7xf32>) -> tensor<6x4x7xf32>
    %m4 = "onnx.MatMul"(%a4, %b4) : (tensor<1x2x12x5xf32>, tensor<1x1x5x8xf32>) -> tensor<1x2x12x8xf32>
    %m5 = "onnx.MatMul"(%a5, %b5) : (tensor<1x2x3x4x5xf32>, tensor<1x1x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
    %s3 = "onnx.ReduceSum"(%m3, %none) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<6x4x7xf32>, none) -> tensor<f32>
    %s4 = "onnx.ReduceSum"(%m4, %none) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x12x8xf32>, none) -> tensor<f32>
    %s5 = "onnx.ReduceSum"(%m5, %none) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4x6xf32>, none) -> tensor<f32>
    %sum0 = "onnx.Add"(%s3, %s4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %sum1 = "onnx.Add"(%sum0, %s5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %result = "onnx.Reshape"(%sum1, %shape) {allowzero = 0 : si64} : (tensor<f32>, tensor<1xi64>) -> tensor<1xf32>
    return %result : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME: %arg0: tensor<6x4x5xf32>
// CHECK-SAME: %arg2: tensor<1x12x5x2xf32>
// CHECK-SAME: %arg4: tensor<1x2x3x4x5xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<6x4x5xf32>, tensor<1x5x7xf32>) -> tensor<6x4x7xf32>
// CHECK: "tfl.batch_matmul"({{.*}}) {{.*}} : (tensor<1x2x12x5xf32>, tensor<1x1x5x8xf32>) -> tensor<1x2x12x8xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x3x4x5xf32>, tensor<1x1x3x5x6xf32>) -> tensor<1x2x3x4x6xf32>
// CHECK-COUNT-3: "tfl.sum"
// CHECK: "tfl.add"{{.*}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<f32>, tensor<1xi32>) -> tensor<1xf32>
// CHECK-NOT: onnx.
