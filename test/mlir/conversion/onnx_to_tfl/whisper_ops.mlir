// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// Whisper's feature extractor uses a padded kernel-3 Conv1D with stride 2 to
// reduce 3000 mel frames to 1500 encoder positions.
module {
  func.func @whisper_conv_stride2(%input: tensor<1x80x9xf32>, %filter: tensor<384x80x3xf32>, %bias: tensor<384xf32>) -> tensor<1x384x5xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], pads = [1, 1], strides = [2]} : (tensor<1x80x9xf32>, tensor<384x80x3xf32>, tensor<384xf32>) -> tensor<1x384x5xf32>
    return %0 : tensor<1x384x5xf32>
  }
  "onnx.EntryPoint"() {func = @whisper_conv_stride2} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.pad"
// CHECK: "tfl.conv_2d"
// CHECK-SAME: stride_h = 2 : i32
// CHECK-SAME: stride_w = 1 : i32
// CHECK: "tfl.reshape"
// CHECK-NOT: onnx.

// -----

// Whisper attention uses rank-4 [batch, heads, sequence, channel] tensors.
// The layout-aware patterns restore logical ONNX order around both matrix
// multiplications and the last-axis Softmax.
module {
  func.func @whisper_attention(%query: tensor<1x2x5x4xf32>, %key_transposed: tensor<1x2x4x5xf32>, %value: tensor<1x2x5x4xf32>) -> tensor<1x2x5x4xf32> {
    %scores = "onnx.MatMul"(%query, %key_transposed) : (tensor<1x2x5x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x5x5xf32>
    %probabilities = "onnx.Softmax"(%scores) {axis = -1 : si64} : (tensor<1x2x5x5xf32>) -> tensor<1x2x5x5xf32>
    %context = "onnx.MatMul"(%probabilities, %value) : (tensor<1x2x5x5xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x5x4xf32>
    return %context : tensor<1x2x5x4xf32>
  }
  "onnx.EntryPoint"() {func = @whisper_attention} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x5x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x5x5xf32>
// CHECK: "tfl.softmax"{{.*}} : (tensor<1x5x2x5xf32>) -> tensor<1x5x2x5xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x5x5xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x5x4xf32>
// CHECK-NOT: onnx.

// -----

// Decoder cross-attention has different query and encoder sequence lengths
// (48 and 1500 in the real model). Keep a small asymmetric analogue so shape
// or layout changes cannot accidentally assume square attention matrices.
module {
  func.func @whisper_cross_attention(%query: tensor<1x2x3x4xf32>, %key_transposed: tensor<1x2x4x5xf32>, %value: tensor<1x2x5x4xf32>) -> tensor<1x2x3x4xf32> {
    %scores = "onnx.MatMul"(%query, %key_transposed) : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
    %probabilities = "onnx.Softmax"(%scores) {axis = -1 : si64} : (tensor<1x2x3x5xf32>) -> tensor<1x2x3x5xf32>
    %context = "onnx.MatMul"(%probabilities, %value) : (tensor<1x2x3x5xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x3x4xf32>
    return %context : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @whisper_cross_attention} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
// CHECK: "tfl.softmax"{{.*}} : (tensor<1x3x2x5xf32>) -> tensor<1x3x2x5xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x3x5xf32>, tensor<1x2x5x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK-NOT: onnx.
