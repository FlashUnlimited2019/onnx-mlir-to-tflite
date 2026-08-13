// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

// Reverse logical W[255:127:-1]. Under NHWC this is physical axis 2. After
// ReverseV2 the selected interval is the ordinary forward slice [0:128].
module {
  func.func @slice_rank4_reverse(%input: tensor<1x1x256x256xf32>) -> tensor<1x1x256x128xf32> {
    %starts = "onnx.Constant"() {value = dense<[255]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[127]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Slice"(%input, %starts, %ends, %axes, %steps) : (tensor<1x1x256x256xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x256x128xf32>
    return %result : tensor<1x1x256x128xf32>
  }
  "onnx.EntryPoint"() {func = @slice_rank4_reverse} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x256x256x1xf32>)
// CHECK: %[[AXIS:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[REVERSED:.*]] = "tfl.reverse_v2"(%arg0, %[[AXIS]]) : (tensor<1x256x256x1xf32>, tensor<1xi32>) -> tensor<1x256x256x1xf32>
// CHECK: %[[BEGIN:.*]] = arith.constant dense<0> : tensor<4xi32>
// CHECK: %[[SIZE:.*]] = arith.constant dense<[1, 256, 128, 1]> : tensor<4xi32>
// CHECK: "tfl.slice"(%[[REVERSED]], %[[BEGIN]], %[[SIZE]]) : (tensor<1x256x256x1xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x256x128x1xf32>
// CHECK-NOT: onnx.

// -----

// INT64_MIN is ONNX's negative-step sentinel for slicing through element 0.
// Reverse first, then retain every second element using positive stride 2.
module {
  func.func @slice_rank2_reverse_to_start(%input: tensor<2x9xf32>) -> tensor<2x5xf32> {
    %starts = "onnx.Constant"() {value = dense<[8]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[-9223372036854775808]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %result = "onnx.Slice"(%input, %starts, %ends, %axes, %steps) : (tensor<2x9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x5xf32>
    return %result : tensor<2x5xf32>
  }
  "onnx.EntryPoint"() {func = @slice_rank2_reverse_to_start} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x9xf32>)
// CHECK: %[[REVERSED:.*]] = "tfl.reverse_v2"(%arg0, {{.*}}) : (tensor<2x9xf32>, tensor<1xi32>) -> tensor<2x9xf32>
// CHECK: "tfl.strided_slice"(%[[REVERSED]], {{.*}}) {{.*}} : (tensor<2x9xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x5xf32>
// CHECK-NOT: onnx.
