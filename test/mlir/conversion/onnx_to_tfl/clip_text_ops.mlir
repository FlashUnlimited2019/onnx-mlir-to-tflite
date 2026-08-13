// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// CLIP narrows runtime int64 token IDs before embedding Gather.
module {
  func.func @cast_token_ids(%input_ids: tensor<1x2xi64>, %embedding: tensor<8x4xf32>) -> (tensor<1x2x4xf32>, tensor<1xf32>) {
    %indices = "onnx.Cast"(%input_ids) <{saturate = 1 : si64, to = i32}> : (tensor<1x2xi64>) -> tensor<1x2xi32>
    %embedded = "onnx.Gather"(%embedding, %indices) <{axis = 0 : si64}> : (tensor<8x4xf32>, tensor<1x2xi32>) -> tensor<1x2x4xf32>
    %position = "onnx.ArgMax"(%indices) <{axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}> : (tensor<1x2xi32>) -> tensor<1xi64>
    %position_f32 = "onnx.Cast"(%position) <{saturate = 1 : si64, to = f32}> : (tensor<1xi64>) -> tensor<1xf32>
    return %embedded, %position_f32 : tensor<1x2x4xf32>, tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @cast_token_ids} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2xi64>, %arg1: tensor<8x4xf32>) -> (tensor<1x2x4xf32>, tensor<1xf32>)
// CHECK: %[[INDICES:.+]] = "tfl.cast"(%arg0) : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK: "tfl.select_v2"{{.*}}%[[INDICES]]
// CHECK: "tfl.gather"{{.*}} : (tensor<8x4xf32>, tensor<1x2xi32>) -> tensor<1x2x4xf32>
// CHECK: "tfl.arg_max"(%[[INDICES]], {{.*}}) : (tensor<1x2xi32>, tensor<i32>) -> tensor<1xi64>
// CHECK: "tfl.cast"{{.*}} : (tensor<1xi64>) -> tensor<1xf32>
// CHECK-NOT: onnx.
