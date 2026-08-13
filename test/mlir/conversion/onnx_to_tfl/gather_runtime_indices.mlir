// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// Runtime integer inputs are needed by embedding tables. Normalize possible
// negative ONNX indices before passing them to TFLite Gather.
module {
  func.func @gather_runtime_indices(%data: tensor<4x3xf32>, %indices: tensor<1x2xi64>) -> tensor<1x2x3xf32> {
    %result = "onnx.Gather"(%data, %indices) {axis = 0 : si64} : (tensor<4x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
    return %result : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @gather_runtime_indices} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<4x3xf32>, %arg1: tensor<1x2xi64>)
// CHECK: "tfl.less"(%arg1, {{.*}}) : (tensor<1x2xi64>, tensor<i64>) -> tensor<1x2xi1>
// CHECK: "tfl.add"(%arg1, {{.*}}) {{.*}} : (tensor<1x2xi64>, tensor<i64>) -> tensor<1x2xi64>
// CHECK: "tfl.select_v2"({{.*}}, {{.*}}, %arg1) : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK: "tfl.gather"(%arg0, {{.*}}) {axis = 0 : i32, batch_dims = 0 : i32} : (tensor<4x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
// CHECK-NOT: onnx.
