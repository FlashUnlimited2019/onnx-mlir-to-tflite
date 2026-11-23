// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// Non-splat weights and bias verify the exact output-channel permutation.
module {
  func.func @shuffle(%x: tensor<2x5x1xf32>) -> tensor<2x15x2xf32> {
    %is = arith.constant dense<[2, 1, 5, 1]> : tensor<4xi32>
    %os = arith.constant dense<[2, 15, 2]> : tensor<3xi32>
    %input = "tfl.reshape"(%x, %is) : (tensor<2x5x1xf32>, tensor<4xi32>) -> tensor<2x1x5x1xf32>
    %w = arith.constant dense<[[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]]> : tensor<6x1x1x1xf32>
    %b = arith.constant dense<[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]> : tensor<6xf32>
    %p = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
    %q = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %r = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %s1 = arith.constant dense<[2, 2, 3, 5]> : tensor<4xi32>
    %s2 = arith.constant dense<[2, 2, 1, 15]> : tensor<4xi32>
    %c = "tfl.conv_2d"(%input, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<2x1x5x1xf32>, tensor<6x1x1x1xf32>, tensor<6xf32>) -> tensor<2x1x5x6xf32>
    %t1 = "tfl.transpose"(%c, %p) : (tensor<2x1x5x6xf32>, tensor<4xi32>) -> tensor<2x6x1x5xf32>
    %a = "tfl.reshape"(%t1, %s1) : (tensor<2x6x1x5xf32>, tensor<4xi32>) -> tensor<2x2x3x5xf32>
    %t2 = "tfl.transpose"(%a, %q) : (tensor<2x2x3x5xf32>, tensor<4xi32>) -> tensor<2x2x5x3xf32>
    %d = "tfl.reshape"(%t2, %s2) : (tensor<2x2x5x3xf32>, tensor<4xi32>) -> tensor<2x2x1x15xf32>
    %out = "tfl.transpose"(%d, %r) : (tensor<2x2x1x15xf32>, tensor<4xi32>) -> tensor<2x1x15x2xf32>
    %result = "tfl.reshape"(%out, %os) : (tensor<2x1x15x2xf32>, tensor<3xi32>) -> tensor<2x15x2xf32>
    return %result : tensor<2x15x2xf32>
  }
  "onnx.EntryPoint"() {func = @shuffle} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[W:.*]] = arith.constant dense<{{.*}}0.000000e+00{{.*}}3.000000e+00{{.*}}1.000000e+00{{.*}}4.000000e+00{{.*}}2.000000e+00{{.*}}5.000000e+00{{.*}}>
// CHECK: %[[B:.*]] = arith.constant dense<[1.000000e+01, 1.300000e+01, 1.100000e+01, 1.400000e+01, 1.200000e+01, 1.500000e+01]>
// CHECK: %[[C:.*]] = "tfl.conv_2d"({{.*}}, %[[W]], %[[B]])
// CHECK-SAME: fused_activation_function = "RELU"
// CHECK-NOT: "tfl.transpose"
// CHECK: %[[R:.*]] = "tfl.reshape"(%[[C]],
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @partition_projection(%x: tensor<2x3x4xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>) {
    %w = arith.constant dense<0.125> : tensor<4x5xf32>
    %b = arith.constant dense<0.25> : tensor<5xf32>
    %i0 = arith.constant dense<0> : tensor<i64>
    %i1 = arith.constant dense<-1> : tensor<i64>
    %g0 = "onnx.Gather"(%x, %i0) {axis = 0 : si64} : (tensor<2x3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    %g1 = "onnx.Gather"(%x, %i1) {axis = 0 : si64} : (tensor<2x3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    %m0 = "onnx.MatMul"(%g0, %w) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    %m1 = "onnx.MatMul"(%g1, %w) : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
    %a0 = "onnx.Add"(%m0, %b) : (tensor<3x5xf32>, tensor<5xf32>) -> tensor<3x5xf32>
    %a1 = "onnx.Add"(%m1, %b) : (tensor<3x5xf32>, tensor<5xf32>) -> tensor<3x5xf32>
    %l0 = "onnx.LeakyRelu"(%a0) {alpha = 0.1 : f32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
    %l1 = "onnx.LeakyRelu"(%a1) {alpha = 0.1 : f32} : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %l0, %l1 : tensor<3x5xf32>, tensor<3x5xf32>
  }
  "onnx.EntryPoint"() {func = @partition_projection} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"
// CHECK-NOT: "tfl.batch_matmul"
// CHECK: "tfl.add"
// CHECK: "tfl.leaky_relu"
// CHECK-NOT: "tfl.leaky_relu"
// CHECK: "tfl.gather"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.batch_matmul"
// CHECK: return
