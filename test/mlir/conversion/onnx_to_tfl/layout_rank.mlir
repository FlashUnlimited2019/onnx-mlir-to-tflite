// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s -split-input-file | FileCheck %s

module {
  func.func @rank3_graph(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @rank3_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>
// CHECK: "tfl.add"
// CHECK-NOT: "tfl.transpose"

// -----

module {
  func.func @rank5_graph(%arg0: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
    return %0 : tensor<1x2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @rank5_graph} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>
// CHECK: "tfl.relu"
// CHECK-NOT: "tfl.transpose"
