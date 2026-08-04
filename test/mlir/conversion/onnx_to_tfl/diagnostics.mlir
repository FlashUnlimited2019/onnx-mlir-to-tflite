// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %s -split-input-file 2>&1 | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x4x8x8xf32>, %arg1: tensor<4x2x3x3xf32>) -> tensor<1x4x6x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%arg0, %arg1, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<4x2x3x3xf32>, none) -> tensor<1x4x6x6xf32>
    return %0 : tensor<1x4x6x6xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: unsupported Conv configuration: group=2 (only group=1 is supported)

// -----

module {
  func.func @main_graph(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x4xf32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: dynamic tensor shape is not supported by ONNXToTFL MVP

// -----

module {
  func.func @main_graph(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: ONNXToTFL MVP requires activation rank >= 1

// -----

module {
  func.func @main_graph(%arg0: tensor<1x8xf32>) -> tensor<2x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 1 : si64} : (tensor<1x8xf32>, tensor<2xi64>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: unsupported Reshape configuration: allowzero=1

// -----

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "onnx.Softmax"(%arg0) {axis = 0 : si64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: unsupported Softmax axis 0: TFLite MVP requires the last dimension

// -----

module {
  func.func @main_graph(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 1]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: unsupported Transpose permutation: not a permutation
