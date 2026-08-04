// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xf32> {onnx.name = "x"}, %arg1: tensor<4xf32> {onnx.name = "bias"}) -> (tensor<2x4xf32> {onnx.name = "y"}) {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: tf.entry_function
// CHECK: "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"}
// CHECK-NOT: onnx.
