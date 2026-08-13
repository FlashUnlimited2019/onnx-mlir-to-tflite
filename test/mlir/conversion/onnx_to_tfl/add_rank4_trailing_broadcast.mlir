// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// ONNX rank-1 broadcasting aligns to logical W. Under the physical NHWC
// layout the operand must therefore become [1,W,1], not remain [W].
module {
  func.func @add_rank4_rank1(%input: tensor<1x3x4x5xf32>, %width: tensor<5xf32>) -> tensor<1x3x4x5xf32> {
    %result = "onnx.Add"(%input, %width) : (tensor<1x3x4x5xf32>, tensor<5xf32>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @add_rank4_rank1} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>, %arg1: tensor<5xf32>)
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<5xf32>, tensor<3xi32>) -> tensor<1x5x1xf32>
// CHECK: "tfl.add"(%arg0, {{.*}}) {fused_activation_function = "NONE"} : (tensor<1x4x5x3xf32>, tensor<1x5x1xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.

// -----

// Rank-2 [H,W] similarly becomes physical [H,W,1]. Cover the lower-rank
// value on the left side as well, even though Add is commutative.
module {
  func.func @add_rank2_rank4(%spatial: tensor<4x5xf32>, %input: tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32> {
    %result = "onnx.Add"(%spatial, %input) : (tensor<4x5xf32>, tensor<1x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    return %result : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @add_rank2_rank4} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<4x5xf32>, %arg1: tensor<1x4x5x3xf32>)
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<4x5xf32>, tensor<3xi32>) -> tensor<4x5x1xf32>
// CHECK: "tfl.add"({{.*}}, %arg1) {fused_activation_function = "NONE"} : (tensor<4x5x1xf32>, tensor<1x4x5x3xf32>) -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.

// -----

// A general rank-3 CHW operand requires an actual CHW->HWC transpose because
// its channel and spatial dimensions can all contain data.
module {
  func.func @add_rank4_rank3(%input: tensor<2x3x4x5xf32>, %chw: tensor<3x4x5xf32>) -> tensor<2x3x4x5xf32> {
    %result = "onnx.Add"(%input, %chw) : (tensor<2x3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<2x3x4x5xf32>
    return %result : tensor<2x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @add_rank4_rank3} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>, %arg1: tensor<3x4x5xf32>)
// CHECK: "tfl.transpose"(%arg1, {{.*}}) : (tensor<3x4x5xf32>, tensor<3xi32>) -> tensor<4x5x3xf32>
// CHECK: "tfl.add"(%arg0, {{.*}}) {fused_activation_function = "NONE"} : (tensor<2x4x5x3xf32>, tensor<4x5x3xf32>) -> tensor<2x4x5x3xf32>
// CHECK-NOT: onnx.
