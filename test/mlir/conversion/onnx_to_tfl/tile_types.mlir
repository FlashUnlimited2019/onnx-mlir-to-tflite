// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl="layout-mode=experimental" --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @tile_i64(%input: tensor<64xi64>) -> tensor<256xi64> {
    %repeats = "onnx.Constant"() {value = dense<[4]> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Tile"(%input, %repeats) : (tensor<64xi64>, tensor<1xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
  "onnx.EntryPoint"() {func = @tile_i64} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<64xi64>{{( \{[^}]*\})?}}) -> (tensor<256xi64>{{( \{[^}]*\})?}})
// CHECK: "tfl.tile"(%arg0, {{.*}}) : (tensor<64xi64>, tensor<1xi32>) -> tensor<256xi64>

// -----

module {
  func.func @tile_rank4_types(%i64: tensor<1x2x3x4xi64>, %i32: tensor<1x2x3x4xi32>, %bool: tensor<1x2x3x4xi1>, %fp32: tensor<1x2x3x4xf32>) -> (tensor<1x4x9x16xi64>, tensor<1x4x9x16xi32>, tensor<1x4x9x16xi1>, tensor<1x4x9x16xf32>) {
    %repeats = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %0 = "onnx.Tile"(%i64, %repeats) : (tensor<1x2x3x4xi64>, tensor<4xi64>) -> tensor<1x4x9x16xi64>
    %1 = "onnx.Tile"(%i32, %repeats) : (tensor<1x2x3x4xi32>, tensor<4xi64>) -> tensor<1x4x9x16xi32>
    %2 = "onnx.Tile"(%bool, %repeats) : (tensor<1x2x3x4xi1>, tensor<4xi64>) -> tensor<1x4x9x16xi1>
    %3 = "onnx.Tile"(%fp32, %repeats) : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x4x9x16xf32>
    return %0, %1, %2, %3 : tensor<1x4x9x16xi64>, tensor<1x4x9x16xi32>, tensor<1x4x9x16xi1>, tensor<1x4x9x16xf32>
  }
  "onnx.EntryPoint"() {func = @tile_rank4_types} : () -> ()
}
// CHECK-LABEL: func.func @main(
// CHECK: %[[I64:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
// CHECK: "tfl.tile"(%arg0, %[[I64]]) : (tensor<1x2x3x4xi64>, tensor<4xi32>) -> tensor<1x4x9x16xi64>
// CHECK: %[[I32:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
// CHECK: "tfl.tile"(%arg1, %[[I32]]) : (tensor<1x2x3x4xi32>, tensor<4xi32>) -> tensor<1x4x9x16xi32>
// CHECK: %[[BOOL:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
// CHECK: "tfl.tile"(%arg2, %[[BOOL]]) : (tensor<1x2x3x4xi1>, tensor<4xi32>) -> tensor<1x4x9x16xi1>
// CHECK: %[[NHWC:.*]] = arith.constant dense<[1, 3, 4, 2]> : tensor<4xi32>
// CHECK: "tfl.tile"(%arg3, %[[NHWC]]) : (tensor<1x3x4x2xf32>, tensor<4xi32>) -> tensor<1x9x16x4xf32>

// -----

module {
  func.func @tile_rank6_i64(%input: tensor<1x2x1x3x4x1xi64>) -> tensor<1x4x1x3x8x1xi64> {
    %repeats = "onnx.Constant"() {value = dense<[1, 2, 1, 1, 2, 1]> : tensor<6xi64>} : () -> tensor<6xi64>
    %0 = "onnx.Tile"(%input, %repeats) : (tensor<1x2x1x3x4x1xi64>, tensor<6xi64>) -> tensor<1x4x1x3x8x1xi64>
    return %0 : tensor<1x4x1x3x8x1xi64>
  }
  "onnx.EntryPoint"() {func = @tile_rank6_i64} : () -> ()
}
// CHECK-LABEL: func.func @main(
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x1x3x4x1xi64>, tensor<5xi32>) -> tensor<2x1x3x4x1xi64>
// CHECK: "tfl.tile"({{.*}}) : (tensor<2x1x3x4x1xi64>, tensor<5xi32>) -> tensor<4x1x3x8x1xi64>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<4x1x3x8x1xi64>, tensor<6xi32>) -> tensor<1x4x1x3x8x1xi64>
