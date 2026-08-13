// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @attention_family(%q: tensor<1x3x8xf32>, %k: tensor<1x3x4xf32>, %v: tensor<1x3x4xf32>, %packed: tensor<1x3x16xf32>) -> (tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>) {
    %none = "onnx.NoValue"() : () -> none
    %weights = onnx.Constant dense<1.250000e-01> : tensor<8x24xf32>
    %bias = onnx.Constant dense<0.000000e+00> : tensor<24xf32>
    %seqlens = onnx.Constant dense<[2]> : tensor<1xi32>
    %total = onnx.Constant dense<3> : tensor<i32>

    %standard, %pk, %pv, %qk = "onnx.Attention"(%q, %q, %q, %none, %none, %none, %none) <{is_causal = 1 : si64, kv_num_heads = 2 : si64, q_num_heads = 2 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}> : (tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, none, none, none, none) -> (tensor<1x3x8xf32>, none, none, none)
    %projected = "onnx.Custom"(%q, %weights, %bias) <{function_name = "Attention"}> {domain_name = "com.microsoft", num_heads = 2 : si64, unidirectional = 0 : si64} : (tensor<1x3x8xf32>, tensor<8x24xf32>, tensor<24xf32>) -> tensor<1x3x8xf32>
    %mha = "onnx.Custom"(%q, %q, %q, %bias) <{function_name = "MultiHeadAttention"}> {domain_name = "com.microsoft", num_heads = 2 : si64} : (tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<24xf32>) -> tensor<1x3x8xf32>
    %gqa = "onnx.Custom"(%q, %k, %v, %none, %none, %seqlens, %total) <{function_name = "GroupQueryAttention"}> {domain_name = "com.microsoft", kv_num_heads = 1 : si64, num_heads = 2 : si64} : (tensor<1x3x8xf32>, tensor<1x3x4xf32>, tensor<1x3x4xf32>, none, none, tensor<1xi32>, tensor<i32>) -> tensor<1x3x8xf32>
    %packed_gqa = "onnx.Custom"(%packed, %none, %none, %none, %none, %seqlens, %total) <{function_name = "GroupQueryAttention"}> {domain_name = "com.microsoft", kv_num_heads = 1 : si64, num_heads = 2 : si64} : (tensor<1x3x16xf32>, none, none, none, none, tensor<1xi32>, tensor<i32>) -> tensor<1x3x8xf32>
    return %standard, %projected, %mha, %gqa, %packed_gqa : tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>, tensor<1x3x8xf32>
  }
  "onnx.EntryPoint"() {func = @attention_family} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-3: "tfl.softmax"
// CHECK: "tfl.gather"
// CHECK: "tfl.softmax"
// CHECK: "tfl.slice"
// CHECK: "tfl.gather"
// CHECK: "tfl.softmax"
// CHECK-NOT: onnx.Attention
// CHECK-NOT: onnx.Custom
