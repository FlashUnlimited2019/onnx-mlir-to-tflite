// RUN: onnx-mlir-opt --recompose-onnx --split-input-file %s | FileCheck %s

// Complete filter concatenation changes the input-group association.
// CHECK-LABEL: func.func @parallel_grouped
// CHECK-COUNT-2: "onnx.Conv"
// CHECK-NOT: "onnx.Split"
// CHECK: return
func.func @parallel_grouped(%x: tensor<1x4x5x5xf32>, %a: tensor<6x2x3x3xf32>, %b: tensor<6x2x3x3xf32>, %bias: tensor<6xf32>) -> (tensor<1x6x3x3xf32>, tensor<1x6x3x3xf32>) {
  %y = "onnx.Conv"(%x, %a, %bias) {group = 2 : si64, pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1], kernel_shape = [3, 3], auto_pad = "NOTSET"} : (tensor<1x4x5x5xf32>, tensor<6x2x3x3xf32>, tensor<6xf32>) -> tensor<1x6x3x3xf32>
  %z = "onnx.Conv"(%x, %b, %bias) {group = 2 : si64, pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1], kernel_shape = [3, 3], auto_pad = "NOTSET"} : (tensor<1x4x5x5xf32>, tensor<6x2x3x3xf32>, tensor<6xf32>) -> tensor<1x6x3x3xf32>
  return %y, %z : tensor<1x6x3x3xf32>, tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: func.func @parallel_depthwise
// CHECK-COUNT-2: "onnx.Conv"
// CHECK: "onnx.Concat"
// CHECK: return
func.func @parallel_depthwise(%x: tensor<1x4x5x5xf32>, %a: tensor<8x1x3x3xf32>, %b: tensor<8x1x3x3xf32>, %bias: tensor<8xf32>) -> tensor<1x16x3x3xf32> {
  %y = "onnx.Conv"(%x, %a, %bias) {group = 4 : si64, pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1], kernel_shape = [3, 3], auto_pad = "NOTSET"} : (tensor<1x4x5x5xf32>, tensor<8x1x3x3xf32>, tensor<8xf32>) -> tensor<1x8x3x3xf32>
  %z = "onnx.Conv"(%x, %b, %bias) {group = 4 : si64, pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1], kernel_shape = [3, 3], auto_pad = "NOTSET"} : (tensor<1x4x5x5xf32>, tensor<8x1x3x3xf32>, tensor<8xf32>) -> tensor<1x8x3x3xf32>
  %out = "onnx.Concat"(%y, %z) {axis = 1 : si64} : (tensor<1x8x3x3xf32>, tensor<1x8x3x3xf32>) -> tensor<1x16x3x3xf32>
  return %out : tensor<1x16x3x3xf32>
}
