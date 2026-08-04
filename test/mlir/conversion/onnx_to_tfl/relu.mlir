// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
    return %0 : tensor<1x8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.relu"(%arg0) : (tensor<1x8xf32>) -> tensor<1x8xf32>
// CHECK-NOT: onnx.
