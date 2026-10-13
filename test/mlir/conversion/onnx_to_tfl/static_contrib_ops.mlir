// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s --implicit-check-not=onnx.Custom
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s --implicit-check-not=onnx.Custom

module {
  func.func @rms(%x: tensor<2x3x4x8xf32>, %gamma: tensor<8xf32>) -> tensor<2x3x4x8xf32> {
    %y = "onnx.Custom"(%x, %gamma) <{function_name = "SimplifiedLayerNormalization"}> {domain_name = "", axis = -1 : si64, epsilon = 1.0e-5 : f32} : (tensor<2x3x4x8xf32>, tensor<8xf32>) -> tensor<2x3x4x8xf32>
    return %y : tensor<2x3x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @rms} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.mul"{{.*}}tensor<24x8xf32>
// CHECK: "tfl.mean"{{.*}}keep_dims = true
// CHECK: "tfl.rsqrt"{{.*}}tensor<24x1xf32>
// CHECK: return {{.*}} : tensor<2x4x8x3xf32>

// -----

module {
  func.func @skip_rms(%x: tensor<2x3x8xf32>, %skip: tensor<2x3x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> tensor<2x3x8xf32> {
    %y = "onnx.Custom"(%x, %skip, %gamma, %bias) <{function_name = "SkipSimplifiedLayerNormalization"}> {domain_name = "com.microsoft", epsilon = 1.0e-5 : f32} : (tensor<2x3x8xf32>, tensor<2x3x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x3x8xf32>
    return %y : tensor<2x3x8xf32>
  }
  "onnx.EntryPoint"() {func = @skip_rms} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.add"
// CHECK: "tfl.add"
// CHECK: "tfl.mean"
// CHECK: "tfl.rsqrt"
// CHECK: return {{.*}} : tensor<2x3x8xf32>

// -----

module {
  func.func @partial_interleaved_rope(%x: tensor<2x3x4x8xf32>, %cos: tensor<8x2xf32>, %sin: tensor<8x2xf32>) -> tensor<2x3x4x8xf32> {
    %position = onnx.Constant dense<1> : tensor<i64>
    %y = "onnx.Custom"(%x, %position, %cos, %sin) <{function_name = "RotaryEmbedding"}> {domain_name = "com.microsoft", num_heads = 3 : si64, rotary_embedding_dim = 4 : si64, interleaved = 1 : si64} : (tensor<2x3x4x8xf32>, tensor<i64>, tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<2x3x4x8xf32>
    return %y : tensor<2x3x4x8xf32>
  }
  "onnx.EntryPoint"() {func = @partial_interleaved_rope} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"
// CHECK: "tfl.sub"
// CHECK: "tfl.pack"{{.*}}tensor<8x3x2x2xf32>
// CHECK: "tfl.slice"{{.*}}tensor<8x3x4xf32>
// CHECK: "tfl.concatenation"
// CHECK: return {{.*}} : tensor<2x4x8x3xf32>

// -----

module {
  func.func @gqa_present(%q: tensor<2x3x16xf32>, %k: tensor<2x3x8xf32>, %v: tensor<2x3x8xf32>) -> (tensor<2x3x16xf32>, tensor<2x2x3x4xf32>, tensor<2x2x3x4xf32>) {
    %none = "onnx.NoValue"() : () -> none
    %lengths = onnx.Constant dense<2> : tensor<2xi32>
    %total = onnx.Constant dense<3> : tensor<i32>
    %y, %pk, %pv = "onnx.Custom"(%q, %k, %v, %none, %none, %lengths, %total) <{function_name = "GroupQueryAttention"}> {domain_name = "com.microsoft", num_heads = 4 : si64, kv_num_heads = 2 : si64} : (tensor<2x3x16xf32>, tensor<2x3x8xf32>, tensor<2x3x8xf32>, none, none, tensor<2xi32>, tensor<i32>) -> (tensor<2x3x16xf32>, tensor<2x2x3x4xf32>, tensor<2x2x3x4xf32>)
    return %y, %pk, %pv : tensor<2x3x16xf32>, tensor<2x2x3x4xf32>, tensor<2x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @gqa_present} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.softmax"
// CHECK: "tfl.batch_matmul"
// CHECK: return {{.*}} : tensor<2x3x16xf32>, tensor<2x3x4x2xf32>, tensor<2x3x4x2xf32>
