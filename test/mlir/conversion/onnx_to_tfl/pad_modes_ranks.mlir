// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x3xf32>) -> tensor<1x6xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 1, 0, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    %value = "onnx.Constant"() {value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%input, %pads, %value, %none) {mode = "constant"} : (tensor<1x3xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<1x6xf32>
    return %0 : tensor<1x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<{{\[}}[0, 0], [1, 2]{{\]}}> : tensor<2x2xi32>
// CHECK: "tfl.padv2"{{.*}}tensor<1x6xf32>
// CHECK-NOT: onnx.Pad

// -----

module {
  func.func @main_graph(%input: tensor<1x4xf32>) -> tensor<1x6xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 1, 0, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%input, %pads, %none, %none) {mode = "reflect"} : (tensor<1x4xf32>, tensor<4xi64>, none, none) -> tensor<1x6xf32>
    return %0 : tensor<1x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<{{\[}}[0, 0], [1, 1]{{\]}}> : tensor<2x2xi32>
// CHECK: "tfl.mirror_pad"{{.*}}mode = #tfl<mirror_pad_attr REFLECT>
// CHECK-NOT: onnx.Pad

// -----

module {
  func.func @main_graph(%input: tensor<1x2x3xf32>) -> tensor<1x3x6xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 1, 2, 0, 0, 1]> : tensor<6xi64>} : () -> tensor<6xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%input, %pads, %none, %none) {mode = "edge"} : (tensor<1x2x3xf32>, tensor<6xi64>, none, none) -> tensor<1x3x6xf32>
    return %0 : tensor<1x3x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.concatenation"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.concatenation"
// CHECK: return {{.*}} : tensor<1x3x6xf32>
// CHECK-NOT: onnx.Pad

// -----

module {
  func.func @main_graph(%input: tensor<1x2x3x4xf32>) -> tensor<1x2x7x10xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 2, 0, 0, 3, 4]> : tensor<8xi64>} : () -> tensor<8xi64>
    %value = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%input, %pads, %value, %none) {mode = "constant"} : (tensor<1x2x3x4xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x2x7x10xf32>
    return %0 : tensor<1x2x7x10xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x4x2xf32>
// CHECK: arith.constant dense<{{\[}}[0, 0], [1, 3], [2, 4], [0, 0]{{\]}}> : tensor<4x2xi32>
// CHECK: "tfl.padv2"{{.*}}tensor<1x7x10x2xf32>
// CHECK-NOT: onnx.Pad

// -----

module {
  func.func @main_graph(%input: tensor<1x1x2x3x4xf32>) -> tensor<1x1x4x4x7xf32> {
    %pads = "onnx.Constant"() {value = dense<[0, 0, 1, 0, 2, 0, 0, 1, 1, 1]> : tensor<10xi64>} : () -> tensor<10xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Pad"(%input, %pads, %none, %none) {mode = "edge"} : (tensor<1x1x2x3x4xf32>, tensor<10xi64>, none, none) -> tensor<1x1x4x4x7xf32>
    return %0 : tensor<1x1x4x4x7xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.concatenation"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.concatenation"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.slice"
// CHECK: "tfl.tile"
// CHECK: "tfl.concatenation"
// CHECK: return {{.*}} : tensor<1x1x4x4x7xf32>
// CHECK-NOT: onnx.Pad
