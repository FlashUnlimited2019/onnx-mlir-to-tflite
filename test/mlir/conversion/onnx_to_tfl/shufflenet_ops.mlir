// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @spatial_mean_without_keepdims(%arg0: tensor<1x12x7x7xf32>) -> tensor<1x12xf32> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 0 : si64} : (tensor<1x12x7x7xf32>) -> tensor<1x12xf32>
    return %0 : tensor<1x12xf32>
  }
  "onnx.EntryPoint"() {func = @spatial_mean_without_keepdims} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x7x7x12xf32>
// CHECK: arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK: "tfl.mean"(%arg0, {{.*}}) {keep_dims = false} : (tensor<1x7x7x12xf32>, tensor<2xi32>) -> tensor<1x12xf32>
