// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @forward(%x: tensor<3x1x2xf32>) -> (tensor<3x1x1x2xf32>, tensor<1x1x2xf32>, tensor<1x1x2xf32>) {
    %w = "onnx.Constant"() {value = dense<0.100000001> : tensor<1x8x2xf32>} : () -> tensor<1x8x2xf32>
    %r = "onnx.Constant"() {value = dense<0.0500000007> : tensor<1x8x2xf32>} : () -> tensor<1x8x2xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none) <{direction = "forward", hidden_size = 2 : si64, input_forget = 0 : si64, layout = 0 : si64}> : (tensor<3x1x2xf32>, tensor<1x8x2xf32>, tensor<1x8x2xf32>, none, none, none, none, none) -> (tensor<3x1x1x2xf32>, tensor<1x1x2xf32>, tensor<1x1x2xf32>)
    return %y, %yh, %yc : tensor<3x1x1x2xf32>, tensor<1x1x2xf32>, tensor<1x1x2xf32>
  }
  "onnx.EntryPoint"() {func = @forward} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<3x1x2xf32>) -> (tensor<3x1x2x1xf32>, tensor<1x1x2xf32>, tensor<1x1x2xf32>)
// CHECK: "tfl.split_v"{{.*}}{num_splits = 3 : i32}
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.add"
// CHECK: "tfl.add"
// CHECK: "tfl.split_v"{{.*}}{num_splits = 4 : i32}
// CHECK: "tfl.logistic"
// CHECK: "tfl.logistic"
// CHECK: "tfl.logistic"
// CHECK: "tfl.tanh"
// CHECK: "tfl.mul"
// CHECK: "tfl.add"
// CHECK: "tfl.tanh"
// CHECK: "tfl.mul"
// CHECK: "tfl.pack"
// CHECK: "tfl.pack"
// CHECK: "tfl.transpose"{{.*}}tensor<3x1x2x1xf32>
// CHECK: "tfl.pack"
// CHECK: "tfl.pack"
// CHECK-NOT: "tfl.while"
// CHECK-NOT: "tf.While"
// CHECK-NOT: TensorList
// CHECK-NOT: onnx.LSTM

// -----

module {
  func.func @bidirectional_lens_clip(%x: tensor<2x2x3xf32>, %initial_h: tensor<2x2x1xf32>, %initial_c: tensor<2x2x1xf32>) -> (tensor<2x2x2x1xf32>, tensor<2x2x1xf32>, tensor<2x2x1xf32>) {
    %w = "onnx.Constant"() {value = dense<0.125> : tensor<2x4x3xf32>} : () -> tensor<2x4x3xf32>
    %r = "onnx.Constant"() {value = dense<0.0625> : tensor<2x4x1xf32>} : () -> tensor<2x4x1xf32>
    %b = "onnx.Constant"() {value = dense<0.03125> : tensor<2x8xf32>} : () -> tensor<2x8xf32>
    %lens = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %b, %lens, %initial_h, %initial_c, %none) <{clip = 5.000000e-01 : f32, direction = "bidirectional", hidden_size = 1 : si64, input_forget = 0 : si64, layout = 0 : si64}> : (tensor<2x2x3xf32>, tensor<2x4x3xf32>, tensor<2x4x1xf32>, tensor<2x8xf32>, tensor<2xi32>, tensor<2x2x1xf32>, tensor<2x2x1xf32>, none) -> (tensor<2x2x2x1xf32>, tensor<2x2x1xf32>, tensor<2x2x1xf32>)
    return %y, %yh, %yc : tensor<2x2x2x1xf32>, tensor<2x2x1xf32>, tensor<2x2x1xf32>
  }
  "onnx.EntryPoint"() {func = @bidirectional_lens_clip} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x2x1xf32>, %arg2: tensor<2x2x1xf32>) -> (tensor<2x2x1x2xf32>, tensor<2x2x1xf32>, tensor<2x2x1xf32>)
// CHECK: "tfl.split_v"{{.*}}{num_splits = 2 : i32}
// CHECK: "tfl.split_v"{{.*}}{num_splits = 2 : i32}
// CHECK: "tfl.split_v"{{.*}}{num_splits = 2 : i32}
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.maximum"
// CHECK: "tfl.minimum"
// CHECK: "tfl.split_v"{{.*}}{num_splits = 4 : i32}
// CHECK: "tfl.logistic"
// CHECK: "tfl.tanh"
// CHECK: "tfl.tanh"
// CHECK: "tfl.pack"
// CHECK: "tfl.pack"
// CHECK: "tfl.pack"
// CHECK-NOT: "tfl.while"
// CHECK-NOT: "tf.While"
// CHECK-NOT: TensorList
// CHECK-NOT: onnx.LSTM
