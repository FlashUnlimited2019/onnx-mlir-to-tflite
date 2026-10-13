// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @bounded_runtime_indices(%data: tensor<2x7xf32>) -> tensor<2x1xi32> {
    %axis = arith.constant dense<1> : tensor<i32>
    %shape = arith.constant dense<[2, 1]> : tensor<2xi32>
    %offset = arith.constant dense<[[0], [7]]> : tensor<2x1xi64>
    %zero = arith.constant dense<0> : tensor<i32>
    %extent = arith.constant dense<14> : tensor<i32>
    %a = "tfl.arg_max"(%data, %axis) : (tensor<2x7xf32>, tensor<i32>) -> tensor<2xi64>
    %b = "tfl.reshape"(%a, %shape) : (tensor<2xi64>, tensor<2xi32>) -> tensor<2x1xi64>
    %c = "tfl.add"(%b, %offset) {fused_activation_function = "NONE"} : (tensor<2x1xi64>, tensor<2x1xi64>) -> tensor<2x1xi64>
    %d = "tfl.cast"(%c) : (tensor<2x1xi64>) -> tensor<2x1xi32>
    %negative = "tfl.less"(%d, %zero) : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi1>
    %adjusted = "tfl.add"(%d, %extent) {fused_activation_function = "NONE"} : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
    %r = "tfl.select_v2"(%negative, %adjusted, %d) : (tensor<2x1xi1>, tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x1xi32>
    return %r : tensor<2x1xi32>
  }
  "onnx.EntryPoint"() {func = @bounded_runtime_indices} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.arg_max"
// CHECK: %[[CAST:.*]] = "tfl.cast"
// CHECK-NOT: "tfl.less"
// CHECK-NOT: "tfl.select_v2"
// CHECK: return %[[CAST]]

// -----

module {
  func.func @unsafe_ranges(%data: tensor<2x7xf32>, %unknown: tensor<2xi64>) -> (tensor<2xi1>, tensor<2xi1>, tensor<2xi1>, tensor<2xi1>, tensor<2xi1>) {
    %axis = arith.constant dense<-1> : tensor<i32>
    %zero = arith.constant dense<0> : tensor<i32>
    %zero64 = arith.constant dense<0> : tensor<i64>
    %max64 = arith.constant dense<9223372036854775807> : tensor<i64>
    %large = arith.constant dense<2147483648> : tensor<i64>
    %minus = arith.constant dense<-1> : tensor<i64>
    %max32 = arith.constant dense<2147483647> : tensor<i32>
    %a = "tfl.arg_min"(%data, %axis) : (tensor<2x7xf32>, tensor<i32>) -> tensor<2xi64>
    %b = "tfl.add"(%a, %large) {fused_activation_function = "NONE"} : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
    %c = "tfl.cast"(%b) : (tensor<2xi64>) -> tensor<2xi32>
    %narrow = "tfl.less"(%c, %zero) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
    %d = "tfl.add"(%a, %max64) {fused_activation_function = "NONE"} : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
    %overflow = "tfl.less"(%d, %zero64) : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
    %e = "tfl.add"(%a, %minus) {fused_activation_function = "NONE"} : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
    %offset = "tfl.less"(%e, %zero64) : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
    %runtime = "tfl.less"(%unknown, %zero64) : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
    %i32 = "tfl.cast"(%a) : (tensor<2xi64>) -> tensor<2xi32>
    %sum32 = "tfl.add"(%i32, %max32) {fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
    %overflow32 = "tfl.less"(%sum32, %zero) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
    return %narrow, %overflow, %offset, %runtime, %overflow32 : tensor<2xi1>, tensor<2xi1>, tensor<2xi1>, tensor<2xi1>, tensor<2xi1>
  }
  "onnx.EntryPoint"() {func = @unsafe_ranges} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-COUNT-5: "tfl.less"
// CHECK: return
