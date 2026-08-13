// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// Cover a rank-5 GatherElements whose indices/result are smaller than the data
// along the selected axis. The lowering materializes one complete GatherNd
// coordinate tuple for every result element.
module {
  func.func @gather_elements_rank5(%input: tensor<2x2x2x2x3xf32>) -> tensor<2x1x2x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<0> : tensor<2x1x2x2x3xi64>} : () -> tensor<2x1x2x2x3xi64>
    %result = "onnx.GatherElements"(%input, %indices) {axis = 1 : si64} : (tensor<2x2x2x2x3xf32>, tensor<2x1x2x2x3xi64>) -> tensor<2x1x2x2x3xf32>
    return %result : tensor<2x1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_elements_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2x2x2x3xf32>)
// CHECK: arith.constant {{.*}} : tensor<2x1x2x2x3x5xi32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<2x2x2x2x3xf32>, tensor<2x1x2x2x3x5xi32>) -> tensor<2x1x2x2x3xf32>
// CHECK-NOT: onnx.
