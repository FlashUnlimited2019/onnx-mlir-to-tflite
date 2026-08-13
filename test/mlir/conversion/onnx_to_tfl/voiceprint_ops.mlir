// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// Voiceprint attention tiles a singleton time axis to all 241 frames.
module {
  func.func @tile_rank3(%input: tensor<1x8x1xf32>) -> tensor<1x8x5xf32> {
    %repeats = "onnx.Constant"() {value = dense<[1, 1, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Tile"(%input, %repeats) : (tensor<1x8x1xf32>, tensor<3xi64>) -> tensor<1x8x5xf32>
    return %result : tensor<1x8x5xf32>
  }
  "onnx.EntryPoint"() {func = @tile_rank3} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<[1, 1, 5]> : tensor<3xi32>
// CHECK: "tfl.tile"(%arg0, {{.*}}) : (tensor<1x8x1xf32>, tensor<3xi32>) -> tensor<1x8x5xf32>
// CHECK-NOT: onnx.

// -----

// Rank-4 repeats follow logical NCHW axes but the TFL tensor is NHWC.
module {
  func.func @tile_rank4_layout(%input: tensor<1x2x3x4xf32>) -> tensor<1x4x3x8xf32> {
    %repeats = "onnx.Constant"() {value = dense<[1, 2, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    %result = "onnx.Tile"(%input, %repeats) : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x4x3x8xf32>
    return %result : tensor<1x4x3x8xf32>
  }
  "onnx.EntryPoint"() {func = @tile_rank4_layout} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>) -> tensor<1x3x8x4xf32>
// CHECK: arith.constant dense<[1, 1, 2, 2]> : tensor<4xi32>
// CHECK: "tfl.tile"(%arg0, {{.*}}) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x3x8x4xf32>
// CHECK-NOT: onnx.
