// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @reverse_sequence(%input: tensor<2x1x2x1x2xf32>) -> tensor<2x1x2x1x2xf32> {
    %lens = arith.constant dense<2> : tensor<1xi64>
    %0 = "onnx.ReverseSequence"(%input, %lens) <{batch_axis = 1 : si64, time_axis = 0 : si64}> : (tensor<2x1x2x1x2xf32>, tensor<1xi64>) -> tensor<2x1x2x1x2xf32>
    return %0 : tensor<2x1x2x1x2xf32>
  }
  "onnx.EntryPoint"() {func = @reverse_sequence} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reverse_sequence"{{.*}}{batch_dim = 1 : i32, seq_dim = 0 : i32}
// CHECK-NOT: onnx.

// -----

module {
  func.func @left_shift(%input: tensor<4xi64>) -> tensor<4xf32> {
    %shift = "onnx.Constant"() <{value = dense<1> : tensor<ui32>}> : () -> tensor<ui32>
    %0 = "onnx.Cast"(%input) <{saturate = 1 : si64, to = ui32}> : (tensor<4xi64>) -> tensor<4xui32>
    %1 = "onnx.BitShift"(%0, %shift) <{direction = "LEFT"}> : (tensor<4xui32>, tensor<ui32>) -> tensor<4xui32>
    %2 = "onnx.Cast"(%1) <{saturate = 1 : si64, to = f32}> : (tensor<4xui32>) -> tensor<4xf32>
    return %2 : tensor<4xf32>
  }
  "onnx.EntryPoint"() {func = @left_shift} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.cast"{{.*}}tensor<4xui32>
// CHECK: "tfl.pseudo_const"{{.*}}dense<2> : tensor<ui32>
// CHECK: "tfl.mul"
// CHECK: "tfl.cast"{{.*}}tensor<4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @rnn(%input: tensor<2x1x2xf32>) -> (tensor<2x1x1x2xf32>, tensor<1x1x2xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x2x2xf32>
    %r = arith.constant dense<0.0625> : tensor<1x2x2xf32>
    %b = arith.constant dense<0.03125> : tensor<1x4xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.RNN"(%input, %w, %r, %b, %none, %none) <{activations = ["Tanh", "Tanh"], direction = "forward", hidden_size = 2 : si64, layout = 0 : si64}> : (tensor<2x1x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x4xf32>, none, none) -> (tensor<2x1x1x2xf32>, tensor<1x1x2xf32>)
    return %y, %yh : tensor<2x1x1x2xf32>, tensor<1x1x2xf32>
  }
  "onnx.EntryPoint"() {func = @rnn} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.split_v"{{.*}}{num_splits = 2 : i32}
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.add"
// CHECK: "tfl.tanh"
// CHECK: "tfl.pack"
// CHECK: "tfl.transpose"{{.*}}tensor<2x1x2x1xf32>
// CHECK-NOT: "tfl.while"
// CHECK-NOT: TensorList
// CHECK-NOT: onnx.

// -----

module {
  func.func @gru(%input: tensor<2x1x2xf32>) -> (tensor<2x1x1x2xf32>, tensor<1x1x2xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x6x2xf32>
    %r = arith.constant dense<0.0625> : tensor<1x6x2xf32>
    %b = arith.constant dense<0.03125> : tensor<1x12xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.GRU"(%input, %w, %r, %b, %none, %none) <{direction = "reverse", hidden_size = 2 : si64, layout = 0 : si64, linear_before_reset = 1 : si64}> : (tensor<2x1x2xf32>, tensor<1x6x2xf32>, tensor<1x6x2xf32>, tensor<1x12xf32>, none, none) -> (tensor<2x1x1x2xf32>, tensor<1x1x2xf32>)
    return %y, %yh : tensor<2x1x1x2xf32>, tensor<1x1x2xf32>
  }
  "onnx.EntryPoint"() {func = @gru} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.logistic"
// CHECK: "tfl.logistic"
// CHECK: "tfl.tanh"
// CHECK: "tfl.mul"
// CHECK: "tfl.pack"
// CHECK-NOT: "tfl.while"
// CHECK-NOT: TensorList
// CHECK-NOT: onnx.

// -----

module {
  func.func @roi_align_max(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x1x1xf32> {
    %rois = arith.constant dense<[[0.0, 0.0, 1.0, 1.0]]> : tensor<1x4xf32>
    %batch = arith.constant dense<0> : tensor<1xi64>
    %0 = "onnx.RoiAlign"(%input, %rois, %batch) <{coordinate_transformation_mode = "half_pixel", mode = "max", output_height = 1 : si64, output_width = 1 : si64, sampling_ratio = 2 : si64, spatial_scale = 1.0 : f32}> : (tensor<1x1x2x2xf32>, tensor<1x4xf32>, tensor<1xi64>) -> tensor<1x1x1x1xf32>
    return %0 : tensor<1x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @roi_align_max} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.mul"
// CHECK: "tfl.reduce_max"
// CHECK-NOT: onnx.

// -----

module {
  func.func @isinf_rank5(%input: tensor<1x2x1x2x2xf32>) -> tensor<1x2x1x2x2xf32> {
    %0 = "onnx.IsInf"(%input) <{detect_negative = 1 : si64, detect_positive = 1 : si64}> : (tensor<1x2x1x2x2xf32>) -> tensor<1x2x1x2x2xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x1x2x2xi1>) -> tensor<1x2x1x2x2xf32>
    return %1 : tensor<1x2x1x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @isinf_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<3.40282347E+38> : tensor<1x2x1x2x2xf32>
// CHECK: "tfl.greater"
// CHECK: "tfl.less"
// CHECK: "tfl.logical_or"
// CHECK-NOT: onnx.
