// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false}
// CHECK-NOT: onnx.
