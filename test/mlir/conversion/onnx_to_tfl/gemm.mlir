// RUN: onnx-mlir-opt --convert-onnx-to-tfl %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 5.000000e-01 : f32, beta = 2.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<2x4xf32>, tensor<4x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK: "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false}
// CHECK: arith.constant dense<5.000000e-01> : tensor<f32>
// CHECK: "tfl.mul"
// CHECK: arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK: "tfl.mul"
// CHECK: "tfl.add"
// CHECK-NOT: onnx.
