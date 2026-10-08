// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s --implicit-check-not='tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x}}'
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s --implicit-check-not='tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x}}'

// Unequal strides and asymmetric padding must preserve NHWC element order.
module {
  func.func @zero_insert_rank4(%x: tensor<2x3x4x5xf32>) -> tensor<2x3x8x16xf32> {
    %y = "onnx.UpsampleAndPad"(%x) {strides = [2, 3], pads = [1, 2, 0, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x8x16xf32>
    return %y : tensor<2x3x8x16xf32>
  }
  "onnx.EntryPoint"() {func = @zero_insert_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}(tensor<2x4x5x3xf32>, tensor<3xi32>) -> tensor<8x1x15xf32>
// CHECK: "tfl.padv2"{{.*}}(tensor<8x1x15xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<8x2x15xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x8x5x3xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<80x1x3xf32>
// CHECK: "tfl.padv2"{{.*}} -> tensor<80x3x3xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x8x15x3xf32>
// CHECK: "tfl.slice"{{.*}} -> tensor<2x7x13x3xf32>
// CHECK: "tfl.padv2"{{.*}} -> tensor<2x8x16x3xf32>
// CHECK: return

// -----

// Runtime negative indices and a smaller non-selected dimension. Coordinate
// columns must remain in ONNX logical axis order after flattening the prefix.
module {
  func.func @flat_coordinates(%x: tensor<2x3x4x5xf32>, %indices: tensor<2x2x3x5xi64>) -> tensor<2x2x3x5xf32> {
    %y = "onnx.GatherElements"(%x, %indices) {axis = -2 : si64} : (tensor<2x3x4x5xf32>, tensor<2x2x3x5xi64>) -> tensor<2x2x3x5xf32>
    return %y : tensor<2x2x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @flat_coordinates} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.cast"
// CHECK: "tfl.less"
// CHECK: "tfl.select_v2"
// CHECK: "tfl.reshape"{{.*}} -> tensor<60x1xi32>
// CHECK: "tfl.concatenation"{{.*}}{axis = 1 : i32, fused_activation_function = "NONE"}{{.*}} -> tensor<60x4xi32>
// CHECK: "tfl.gather_nd"{{.*}}tensor<60x4xi32>) -> tensor<60xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x2x3x5xf32>
// CHECK: return
