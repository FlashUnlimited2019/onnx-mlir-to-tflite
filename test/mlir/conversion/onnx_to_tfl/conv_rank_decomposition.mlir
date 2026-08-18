// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x3x8xf32>, %filter: tensor<4x3x3xf32>, %bias: tensor<4xf32>) -> tensor<1x4x8xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], pads = [1, 1], strides = [1]} : (tensor<1x3x8xf32>, tensor<4x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8xf32>
    return %0 : tensor<1x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.reshape"
// CHECK: "tfl.pad"
// CHECK: "tfl.conv_2d"
// CHECK: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<2x3x2x4x5xf32>, %filter: tensor<7x3x2x4x5xf32>, %bias: tensor<7xf32>) -> tensor<2x7x1x1x1xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 4, 5], pads = [0, 0, 0, 0, 0, 0], strides = [2, 4, 5]} : (tensor<2x3x2x4x5xf32>, tensor<7x3x2x4x5xf32>, tensor<7xf32>) -> tensor<2x7x1x1x1xf32>
    return %0 : tensor<2x7x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<2x6x4x5xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<2x4x5x6xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<7x6x4x5xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<7x4x5x6xf32>
// CHECK: "tfl.conv_2d"{{.*}}tensor<2x1x1x7xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<2x7x1x1x1xf32>
// CHECK-NOT: "tfl.conv_3d"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<4x2x6x8xf32>, %filter: tensor<3x2x2x3x4xf32>, %bias: tensor<3xf32>) -> tensor<1x8x3xf32> {
    %0 = "onnx.Transpose"(%input) {perm = [1, 0, 2, 3]} : (tensor<4x2x6x8xf32>) -> tensor<2x4x6x8xf32>
    %axes = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %1 = "onnx.Unsqueeze"(%0, %axes) : (tensor<2x4x6x8xf32>, tensor<1xi64>) -> tensor<1x2x4x6x8xf32>
    %2 = "onnx.Conv"(%1, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 3, 4], pads = [0, 0, 0, 0, 0, 0], strides = [2, 3, 4]} : (tensor<1x2x4x6x8xf32>, tensor<3x2x2x3x4xf32>, tensor<3xf32>) -> tensor<1x3x2x2x2xf32>
    %shape = "onnx.Constant"() {value = dense<[1, 3, 8]> : tensor<3xi64>} : () -> tensor<3xi64>
    %3 = "onnx.Reshape"(%2, %shape) {allowzero = 0 : si64} : (tensor<1x3x2x2x2xf32>, tensor<3xi64>) -> tensor<1x3x8xf32>
    %4 = "onnx.Transpose"(%3) {perm = [0, 2, 1]} : (tensor<1x3x8xf32>) -> tensor<1x8x3xf32>
    return %4 : tensor<1x8x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<4x6x8x2xf32>
// CHECK: %[[DEPTH_LAST:.*]] = "tfl.transpose"(%arg0, {{.*}}) : (tensor<4x6x8x2xf32>, tensor<4xi32>) -> tensor<6x8x4x2xf32>
// CHECK: %[[SPLIT:.*]] = "tfl.reshape"(%[[DEPTH_LAST]], {{.*}}) : (tensor<6x8x4x2xf32>, tensor<4xi32>) -> tensor<6x8x2x4xf32>
// CHECK: %[[PACKED:.*]] = "tfl.transpose"(%[[SPLIT]], {{.*}}) : (tensor<6x8x2x4xf32>, tensor<4xi32>) -> tensor<2x6x8x4xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<3x3x4x2x2xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<3x3x4x4xf32>
// CHECK: "tfl.conv_2d"(%[[PACKED]], {{.*}}){{.*}}tensor<2x2x2x3xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<1x8x3xf32>
// CHECK-NOT: "tfl.conv_3d"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x2x4x6x8xf32>, %filter: tensor<3x2x2x3x4xf32>, %bias: tensor<3xf32>) -> tensor<1x3x2x2x2xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [2, 3, 4], pads = [0, 0, 0, 0, 0, 0], strides = [2, 3, 4]} : (tensor<1x2x4x6x8xf32>, tensor<3x2x2x3x4xf32>, tensor<3xf32>) -> tensor<1x3x2x2x2xf32>
    return %0 : tensor<1x3x2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.conv_2d"{{.*}}tensor<2x2x2x3xf32>
// CHECK: "tfl.reshape"{{.*}}tensor<1x2x2x2x3xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<1x3x2x2x2xf32>
// CHECK-NOT: "tfl.conv_3d"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x4x8xf32>, %filter: tensor<6x2x5xf32>, %bias: tensor<6xf32>) -> tensor<1x6x8xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1], group = 2 : si64, kernel_shape = [5], pads = [2, 2], strides = [1]} : (tensor<1x4x8xf32>, tensor<6x2x5xf32>, tensor<6xf32>) -> tensor<1x6x8xf32>
    return %0 : tensor<1x6x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: "tfl.split_v"
// CHECK-COUNT-2: "tfl.conv_2d"
// CHECK: "tfl.concatenation"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x4x8x9xf32>, %filter: tensor<6x2x3x5xf32>, %bias: tensor<6xf32>) -> tensor<1x6x8x9xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [3, 5], pads = [1, 2, 1, 2], strides = [1, 1]} : (tensor<1x4x8x9xf32>, tensor<6x2x3x5xf32>, tensor<6xf32>) -> tensor<1x6x8x9xf32>
    return %0 : tensor<1x6x8x9xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: "tfl.split_v"
// CHECK-COUNT-2: "tfl.conv_2d"
// CHECK: "tfl.concatenation"
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x2x4x8x9xf32>, %filter: tensor<3x2x1x3x5xf32>, %bias: tensor<3xf32>) -> tensor<1x3x4x8x9xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [1, 3, 5], pads = [0, 1, 2, 0, 1, 2], strides = [1, 1, 1]} : (tensor<1x2x4x8x9xf32>, tensor<3x2x1x3x5xf32>, tensor<3xf32>) -> tensor<1x3x4x8x9xf32>
    return %0 : tensor<1x3x4x8x9xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<4x8x9x2xf32>
// CHECK: "tfl.conv_2d"{{.*}}tensor<4x8x9x3xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<1x3x4x8x9xf32>
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x2x8x4x5xf32>, %filter: tensor<3x2x3x1x1xf32>, %bias: tensor<3xf32>) -> tensor<1x3x8x4x5xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 1, 1], pads = [1, 0, 0, 1, 0, 0], strides = [1, 1, 1]} : (tensor<1x2x8x4x5xf32>, tensor<3x2x3x1x1xf32>, tensor<3xf32>) -> tensor<1x3x8x4x5xf32>
    return %0 : tensor<1x3x8x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<4x8x5x2xf32>
// CHECK: "tfl.conv_2d"{{.*}}tensor<4x8x5x3xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<1x3x8x4x5xf32>
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x2x8x4x6xf32>, %filter: tensor<4x2x3x1x2xf32>, %bias: tensor<4xf32>) -> tensor<1x4x8x4x6xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 1, 2], pads = [1, 0, 0, 1, 0, 1], strides = [1, 1, 1]} : (tensor<1x2x8x4x6xf32>, tensor<4x2x3x1x2xf32>, tensor<4xf32>) -> tensor<1x4x8x4x6xf32>
    return %0 : tensor<1x4x8x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<4x8x6x2xf32>
// CHECK: "tfl.conv_2d"{{.*}}tensor<4x8x6x4xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<1x4x8x4x6xf32>
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x2x5x6x7xf32>, %filter: tensor<3x2x3x3x3xf32>, %bias: tensor<3xf32>) -> tensor<1x3x5x6x7xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 3, 3], pads = [1, 1, 1, 1, 1, 1], strides = [1, 1, 1]} : (tensor<1x2x5x6x7xf32>, tensor<3x2x3x3x3xf32>, tensor<3xf32>) -> tensor<1x3x5x6x7xf32>
    return %0 : tensor<1x3x5x6x7xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"{{.*}}tensor<1x5x6x7x2xf32>
// CHECK: "tfl.pad"{{.*}}tensor<1x7x8x9x2xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<3x3x3x2x3xf32>
// CHECK: "tfl.conv_3d"{{.*}}tensor<1x5x6x7x3xf32>
// CHECK: "tfl.transpose"{{.*}}tensor<1x3x5x6x7xf32>
// CHECK-NOT: onnx.Conv

// -----

module {
  func.func @main_graph(%input: tensor<1x4x5x6x7xf32>, %filter: tensor<6x2x3x3x3xf32>, %bias: tensor<6xf32>) -> tensor<1x6x3x4x5xf32> {
    %0 = "onnx.Conv"(%input, %filter, %bias) {auto_pad = "VALID", dilations = [1, 1, 1], group = 2 : si64, kernel_shape = [3, 3, 3], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x4x5x6x7xf32>, tensor<6x2x3x3x3xf32>, tensor<6xf32>) -> tensor<1x6x3x4x5xf32>
    return %0 : tensor<1x6x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: "tfl.split_v"
// CHECK-COUNT-2: "tfl.conv_3d"
// CHECK: "tfl.concatenation"
// CHECK: "tfl.transpose"{{.*}}tensor<1x6x3x4x5xf32>
// CHECK-NOT: onnx.Conv
