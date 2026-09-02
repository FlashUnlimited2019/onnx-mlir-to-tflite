// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @packed_split_projection(%input: tensor<1x2x12xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) {
    %sizes = "onnx.Constant"() {value = dense<[4, 4, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %bias0 = "onnx.Constant"() {value = dense<[0.0, 0.1, 0.2, 0.3]> : tensor<4xf32>} : () -> tensor<4xf32>
    %bias1 = "onnx.Constant"() {value = dense<[0.4, 0.5, 0.6, 0.7]> : tensor<4xf32>} : () -> tensor<4xf32>
    %bias2 = "onnx.Constant"() {value = dense<[0.8, 0.9, 1.0, 1.1]> : tensor<4xf32>} : () -> tensor<4xf32>
    %scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %shape = "onnx.Constant"() {value = dense<[1, 2, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    %q, %k, %v = "onnx.Split"(%input, %sizes) <{axis = 2 : si64}> : (tensor<1x2x12xf32>, tensor<3xi64>) -> (tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>)
    %qb = "onnx.Add"(%q, %bias0) : (tensor<1x2x4xf32>, tensor<4xf32>) -> tensor<1x2x4xf32>
    %kb = "onnx.Add"(%k, %bias1) : (tensor<1x2x4xf32>, tensor<4xf32>) -> tensor<1x2x4xf32>
    %vb = "onnx.Add"(%v, %bias2) : (tensor<1x2x4xf32>, tensor<4xf32>) -> tensor<1x2x4xf32>
    %qs = "onnx.Mul"(%qb, %scale) : (tensor<1x2x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %qr = "onnx.Reshape"(%qs, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
    %kr = "onnx.Reshape"(%kb, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
    %vr = "onnx.Reshape"(%vb, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
    %qt = "onnx.Transpose"(%qr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %kt = "onnx.Transpose"(%kr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %vt = "onnx.Transpose"(%vr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %qt, %kt, %vt : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @packed_split_projection} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.add"
// CHECK-NOT: "tfl.add"
// CHECK: "tfl.reshape"
// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.split_v"
// CHECK-NOT: "tfl.split_v"
// CHECK: "tfl.mul"
// CHECK-NOT: "tfl.mul"
// CHECK-NOT: onnx.

// -----

module {
  func.func @packed_separate_projection(%input: tensor<1x2x4xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) {
    %weight0 = "onnx.Constant"() {value = dense<0.125> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %weight1 = "onnx.Constant"() {value = dense<0.25> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
    %weight2 = "onnx.Constant"() {value = dense<0.5> : tensor<4x2xf32>} : () -> tensor<4x2xf32>
    %shape0 = "onnx.Constant"() {value = dense<[1, 2, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    %shape1 = "onnx.Constant"() {value = dense<[1, 2, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
    %q = "onnx.MatMul"(%input, %weight0) : (tensor<1x2x4xf32>, tensor<4x4xf32>) -> tensor<1x2x4xf32>
    %k = "onnx.MatMul"(%input, %weight1) : (tensor<1x2x4xf32>, tensor<4x2xf32>) -> tensor<1x2x2xf32>
    %v = "onnx.MatMul"(%input, %weight2) : (tensor<1x2x4xf32>, tensor<4x2xf32>) -> tensor<1x2x2xf32>
    %qr = "onnx.Reshape"(%q, %shape0) <{allowzero = 0 : si64}> : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
    %kr = "onnx.Reshape"(%k, %shape1) <{allowzero = 0 : si64}> : (tensor<1x2x2xf32>, tensor<4xi64>) -> tensor<1x2x1x2xf32>
    %vr = "onnx.Reshape"(%v, %shape1) <{allowzero = 0 : si64}> : (tensor<1x2x2xf32>, tensor<4xi64>) -> tensor<1x2x1x2xf32>
    %qt = "onnx.Transpose"(%qr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %kt = "onnx.Transpose"(%kr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x1x2xf32>) -> tensor<1x1x2x2xf32>
    %vt = "onnx.Transpose"(%vr) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x1x2xf32>) -> tensor<1x1x2x2xf32>
    return %qt, %kt, %vt : tensor<1x2x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @packed_separate_projection} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"
// CHECK-NOT: "tfl.batch_matmul"
// CHECK: "tfl.reshape"
// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.split_v"
// CHECK-NOT: "tfl.split_v"
// CHECK-NOT: onnx.

// -----

module {
  func.func @move_broadcast_add(%input: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %mask = "onnx.Constant"() {value = dense<[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]> : tensor<1x3x2xf32>} : () -> tensor<1x3x2xf32>
    %first = "onnx.Transpose"(%input) <{perm = [0, 2, 1]}> : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
    %sum = "onnx.Add"(%first, %mask) : (tensor<1x3x2xf32>, tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
    %result = "onnx.Transpose"(%sum) <{perm = [0, 2, 1]}> : (tensor<1x3x2xf32>) -> tensor<1x2x3xf32>
    return %result : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @move_broadcast_add} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<{{.*}}> : tensor<1x2x3xf32>
// CHECK: "tfl.add"
// CHECK-NOT: "tfl.add"
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @fold_singleton_batch_broadcast(%lhs: tensor<4x2x3xf32>, %rhs: tensor<1x3x5xf32>) -> tensor<4x2x5xf32> {
    %source_shape = "onnx.Constant"() {value = dense<[1, 1, 1, 3, 5]> : tensor<5xi64>} : () -> tensor<5xi64>
    %broadcast_shape = "onnx.Constant"() {value = dense<[1, 1, 4, 3, 5]> : tensor<5xi64>} : () -> tensor<5xi64>
    %flattened_shape = "onnx.Constant"() {value = dense<[4, 3, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
    %source = "onnx.Reshape"(%rhs, %source_shape) <{allowzero = 0 : si64}> : (tensor<1x3x5xf32>, tensor<5xi64>) -> tensor<1x1x1x3x5xf32>
    %broadcast = "onnx.Expand"(%source, %broadcast_shape) : (tensor<1x1x1x3x5xf32>, tensor<5xi64>) -> tensor<1x1x4x3x5xf32>
    %flattened = "onnx.Reshape"(%broadcast, %flattened_shape) <{allowzero = 0 : si64}> : (tensor<1x1x4x3x5xf32>, tensor<3xi64>) -> tensor<4x3x5xf32>
    %result = "onnx.MatMul"(%lhs, %flattened) : (tensor<4x2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
    return %result : tensor<4x2x5xf32>
  }
  "onnx.EntryPoint"() {func = @fold_singleton_batch_broadcast} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x3x5xf32>, tensor<3xi32>) -> tensor<1x3x5xf32>
// CHECK-NOT: "tfl.reshape"
// CHECK-NOT: "tfl.broadcast_to"
// CHECK: "tfl.batch_matmul"(%arg0, {{.*}}) {{.*}} : (tensor<4x2x3xf32>, tensor<1x3x5xf32>) -> tensor<4x2x5xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @fold_transpose_softmax_transpose(%input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %forward_perm = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %inverse_perm = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %moved = "tfl.transpose"(%input, %forward_perm) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<3x2x4xf32>
    %normalized = "tfl.softmax"(%moved) <{beta = 1.0 : f32}> : (tensor<3x2x4xf32>) -> tensor<3x2x4xf32>
    %result = "tfl.transpose"(%normalized, %inverse_perm) : (tensor<3x2x4xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @fold_transpose_softmax_transpose} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.softmax"(%arg0)
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @preserve_axis_moving_softmax_transposes(%input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %forward_perm = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %inverse_perm = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %moved = "tfl.transpose"(%input, %forward_perm) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<2x4x3xf32>
    %normalized = "tfl.softmax"(%moved) <{beta = 1.0 : f32}> : (tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
    %result = "tfl.transpose"(%normalized, %inverse_perm) : (tensor<2x4x3xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @preserve_axis_moving_softmax_transposes} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.softmax"
// CHECK: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @fold_transpose_gelu_transpose(%input: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %forward_perm = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %inverse_perm = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %moved = "tfl.transpose"(%input, %forward_perm) : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<1x3x2xf32>
    %activated = "tfl.gelu"(%moved) <{approximate = false}> : (tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
    %result = "tfl.transpose"(%activated, %inverse_perm) : (tensor<1x3x2xf32>, tensor<3xi32>) -> tensor<1x2x3xf32>
    return %result : tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @fold_transpose_gelu_transpose} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gelu"(%arg0)
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @collapse_head_layout_transpose_chain(%input: tensor<2x5x12xf32>) -> tensor<6x4x5xf32> {
    %input_shape = arith.constant dense<[2, 5, 3, 4]> : tensor<4xi32>
    %first_perm = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %second_perm = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
    %first_shape = arith.constant dense<[6, 5, 4]> : tensor<3xi32>
    %third_perm = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %fourth_perm = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %final_shape = arith.constant dense<[2, 3, 4, 5]> : tensor<4xi32>
    %output_shape = arith.constant dense<[6, 4, 5]> : tensor<3xi32>
    %split_heads = "tfl.reshape"(%input, %input_shape) : (tensor<2x5x12xf32>, tensor<4xi32>) -> tensor<2x5x3x4xf32>
    %width_first = "tfl.transpose"(%split_heads, %first_perm) : (tensor<2x5x3x4xf32>, tensor<4xi32>) -> tensor<2x5x4x3xf32>
    %heads_first = "tfl.transpose"(%width_first, %second_perm) : (tensor<2x5x4x3xf32>, tensor<4xi32>) -> tensor<2x3x5x4xf32>
    %flat_heads = "tfl.reshape"(%heads_first, %first_shape) : (tensor<2x3x5x4xf32>, tensor<3xi32>) -> tensor<6x5x4xf32>
    %sequence_first = "tfl.transpose"(%flat_heads, %third_perm) : (tensor<6x5x4xf32>, tensor<3xi32>) -> tensor<5x6x4xf32>
    %matrix_transposed = "tfl.transpose"(%sequence_first, %fourth_perm) : (tensor<5x6x4xf32>, tensor<3xi32>) -> tensor<6x4x5xf32>
    %joined_heads = "tfl.reshape"(%matrix_transposed, %final_shape) : (tensor<6x4x5xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
    %result = "tfl.reshape"(%joined_heads, %output_shape) : (tensor<2x3x4x5xf32>, tensor<3xi32>) -> tensor<6x4x5xf32>
    return %result : tensor<6x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @collapse_head_layout_transpose_chain} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x5x12xf32>, tensor<4xi32>) -> tensor<2x5x3x4xf32>
// CHECK: arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x5x3x4xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.reshape"({{.*}}) : (tensor<2x3x4x5xf32>, tensor<3xi32>) -> tensor<6x4x5xf32>
// CHECK-NOT: "tfl.reshape"
// CHECK-NOT: onnx.

// -----

module {
  func.func @preserve_noncanonical_head_layout_chain(%input: tensor<2x5x12xf32>) -> tensor<10x3x4xf32> {
    %input_shape = arith.constant dense<[2, 5, 3, 4]> : tensor<4xi32>
    %first_perm = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %first_shape = arith.constant dense<[6, 5, 4]> : tensor<3xi32>
    %second_perm = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
    %final_shape = arith.constant dense<[2, 5, 3, 4]> : tensor<4xi32>
    %output_shape = arith.constant dense<[10, 3, 4]> : tensor<3xi32>
    %split_heads = "tfl.reshape"(%input, %input_shape) : (tensor<2x5x12xf32>, tensor<4xi32>) -> tensor<2x5x3x4xf32>
    %heads_first = "tfl.transpose"(%split_heads, %first_perm) : (tensor<2x5x3x4xf32>, tensor<4xi32>) -> tensor<2x3x5x4xf32>
    %flat_heads = "tfl.reshape"(%heads_first, %first_shape) : (tensor<2x3x5x4xf32>, tensor<3xi32>) -> tensor<6x5x4xf32>
    %matrix_transposed = "tfl.transpose"(%flat_heads, %second_perm) : (tensor<6x5x4xf32>, tensor<3xi32>) -> tensor<5x6x4xf32>
    %result = "tfl.reshape"(%matrix_transposed, %final_shape) : (tensor<5x6x4xf32>, tensor<4xi32>) -> tensor<2x5x3x4xf32>
    %output = "tfl.reshape"(%result, %output_shape) : (tensor<2x5x3x4xf32>, tensor<3xi32>) -> tensor<10x3x4xf32>
    return %output : tensor<10x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @preserve_noncanonical_head_layout_chain} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK-NOT: onnx.
