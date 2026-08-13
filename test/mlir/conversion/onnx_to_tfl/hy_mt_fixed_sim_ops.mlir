// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// The full hy_mt graph converts its runtime int64 attention mask to bool
// before combining it with the static causal mask.
module {
  func.func @attention_mask_cast_gather_unsqueeze(%attention_mask: tensor<1x88xi64>, %reverse_indices: tensor<88xi64>) -> tensor<1x1x1x88xf32> {
    %mask = "onnx.Cast"(%attention_mask) <{saturate = 1 : si64, to = i1}> : (tensor<1x88xi64>) -> tensor<1x88xi1>
    %reversed = "onnx.Gather"(%mask, %reverse_indices) <{axis = 1 : si64}> : (tensor<1x88xi1>, tensor<88xi64>) -> tensor<1x88xi1>
    %axes = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %expanded = "onnx.Unsqueeze"(%reversed, %axes) : (tensor<1x88xi1>, tensor<2xi64>) -> tensor<1x1x1x88xi1>
    %result = "onnx.Cast"(%expanded) <{saturate = 1 : si64, to = f32}> : (tensor<1x1x1x88xi1>) -> tensor<1x1x1x88xf32>
    return %result : tensor<1x1x1x88xf32>
  }
  "onnx.EntryPoint"() {func = @attention_mask_cast_gather_unsqueeze} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x88xi64>, %arg1: tensor<88xi64>) -> tensor<1x1x88x1xf32>
// CHECK: %[[MASK:.+]] = "tfl.cast"(%arg0) : (tensor<1x88xi64>) -> tensor<1x88xi1>
// CHECK: %[[REVERSED:.+]] = "tfl.gather"(%[[MASK]], {{.*}}) {axis = 1 : i32, batch_dims = 0 : i32} : (tensor<1x88xi1>, tensor<88xi64>) -> tensor<1x88xi1>
// CHECK: %[[EXPANDED:.+]] = "tfl.reshape"(%[[REVERSED]], {{.*}}) : (tensor<1x88xi1>, tensor<4xi32>) -> tensor<1x1x1x88xi1>
// CHECK: %[[LOGICAL_RESULT:.+]] = "tfl.cast"(%[[EXPANDED]]) : (tensor<1x1x1x88xi1>) -> tensor<1x1x1x88xf32>
// CHECK: %[[RESULT:.+]] = "tfl.transpose"(%[[LOGICAL_RESULT]], {{.*}}) : (tensor<1x1x1x88xf32>, tensor<4xi32>) -> tensor<1x1x88x1xf32>
// CHECK: return %[[RESULT]] : tensor<1x1x88x1xf32>
// CHECK-NOT: onnx.
