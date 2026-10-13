// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// Cover a rank-5 GatherElements whose indices/result are smaller than the data
// along the selected axis. Constant coordinate tuples are linearized into
// rank-one indices for a rank-two Gather.
module {
  func.func @gather_elements_rank5(%input: tensor<2x2x2x2x3xf32>) -> tensor<2x1x2x2x3xf32> {
    %indices = "onnx.Constant"() {value = dense<0> : tensor<2x1x2x2x3xi64>} : () -> tensor<2x1x2x2x3xi64>
    %result = "onnx.GatherElements"(%input, %indices) {axis = 1 : si64} : (tensor<2x2x2x2x3xf32>, tensor<2x1x2x2x3xi64>) -> tensor<2x1x2x2x3xf32>
    return %result : tensor<2x1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_elements_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2x2x2x3xf32>{{( \{[^}]*\})?}})
// CHECK: arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]> : tensor<24xi32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x2x2x2x3xf32>, tensor<2xi32>) -> tensor<48x1xf32>
// CHECK: "tfl.gather"{{.*}} : (tensor<48x1xf32>, tensor<24xi32>) -> tensor<24x1xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<24xf32>, tensor<5xi32>) -> tensor<2x1x2x2x3xf32>
// CHECK-NOT: onnx.
