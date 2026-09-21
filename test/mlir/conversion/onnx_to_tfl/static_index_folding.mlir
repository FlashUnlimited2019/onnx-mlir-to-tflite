// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @integer_index_chain() -> tensor<4x3xi32> {
    %data = arith.constant dense<[-1, 0]> : tensor<2xi64>
    %repeats = arith.constant dense<[2]> : tensor<1xi32>
    %shape = arith.constant dense<[4, 1]> : tensor<2xi32>
    %columns = arith.constant dense<[0, 1, 2]> : tensor<3xi64>
    %zero = arith.constant dense<0> : tensor<i32>
    %extent = arith.constant dense<5> : tensor<i32>
    %a = "tfl.tile"(%data, %repeats) : (tensor<2xi64>, tensor<1xi32>) -> tensor<4xi64>
    %b = "tfl.reshape"(%a, %shape) : (tensor<4xi64>, tensor<2xi32>) -> tensor<4x1xi64>
    %c = "tfl.add"(%b, %columns) {fused_activation_function = "NONE"} : (tensor<4x1xi64>, tensor<3xi64>) -> tensor<4x3xi64>
    %d = "tfl.cast"(%c) : (tensor<4x3xi64>) -> tensor<4x3xi32>
    %negative = "tfl.less"(%d, %zero) : (tensor<4x3xi32>, tensor<i32>) -> tensor<4x3xi1>
    %adjusted = "tfl.add"(%d, %extent) {fused_activation_function = "NONE"} : (tensor<4x3xi32>, tensor<i32>) -> tensor<4x3xi32>
    %r = "tfl.select_v2"(%negative, %adjusted, %d) : (tensor<4x3xi1>, tensor<4x3xi32>, tensor<4x3xi32>) -> tensor<4x3xi32>
    return %r : tensor<4x3xi32>
  }
  "onnx.EntryPoint"() {func = @integer_index_chain} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.
// CHECK: %[[RESULT:.*]] = arith.constant dense<{{\[\[4, 0, 1\], \[0, 1, 2\], \[4, 0, 1\], \[0, 1, 2\]\]}}> : tensor<4x3xi32>
// CHECK-NOT: "tfl.
// CHECK: return %[[RESULT]]

// -----

module {
  func.func @boolean_cast_semantics() -> tensor<4xi64> {
    %data = arith.constant dense<[-2, 0, 2, 9223372036854775807]> : tensor<4xi64>
    %a = "tfl.cast"(%data) : (tensor<4xi64>) -> tensor<4xi1>
    %b = "tfl.cast"(%a) : (tensor<4xi1>) -> tensor<4xi64>
    return %b : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @boolean_cast_semantics} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<[1, 0, 1, 1]> : tensor<4xi64>
// CHECK-NOT: "tfl.cast"
// CHECK: return

// -----

module {
  func.func @overflow_and_narrowing() -> (tensor<1xi32>, tensor<1xi32>) {
    %a = arith.constant dense<2147483647> : tensor<1xi32>
    %one = arith.constant dense<1> : tensor<1xi32>
    %large = arith.constant dense<2147483648> : tensor<1xi64>
    %sum = "tfl.add"(%a, %one) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %cast = "tfl.cast"(%large) : (tensor<1xi64>) -> tensor<1xi32>
    return %sum, %cast : tensor<1xi32>, tensor<1xi32>
  }
  "onnx.EntryPoint"() {func = @overflow_and_narrowing} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.add"
// CHECK: "tfl.cast"
// CHECK: return

// -----

module {
  func.func @floating_arithmetic_is_unchanged() -> tensor<2xf32> {
    %a = arith.constant dense<[0.1, 0.2]> : tensor<2xf32>
    %b = arith.constant dense<0.3> : tensor<2xf32>
    %r = "tfl.add"(%a, %b) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
  "onnx.EntryPoint"() {func = @floating_arithmetic_is_unchanged} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.add"
// CHECK: return

// -----

module {
  func.func @runtime_tile(%data: tensor<2xi64>, %repeats: tensor<1xi32>) -> tensor<4xi64> {
    %r = "tfl.tile"(%data, %repeats) : (tensor<2xi64>, tensor<1xi32>) -> tensor<4xi64>
    return %r : tensor<4xi64>
  }
  "onnx.EntryPoint"() {func = @runtime_tile} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.tile"(%arg0, %arg1)
// CHECK: return

// -----

module {
  func.func @bounded_constant_expansion() -> tensor<1048577xi64> {
    %data = arith.constant dense<7> : tensor<1xi64>
    %repeats = arith.constant dense<1048577> : tensor<1xi32>
    %r = "tfl.tile"(%data, %repeats) : (tensor<1xi64>, tensor<1xi32>) -> tensor<1048577xi64>
    return %r : tensor<1048577xi64>
  }
  "onnx.EntryPoint"() {func = @bounded_constant_expansion} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.tile"
// CHECK: return

// -----

module {
  func.func @integer_concat_transpose() -> tensor<2x2xi64> {
    %a = arith.constant dense<[[1, 2]]> : tensor<1x2xi64>
    %b = arith.constant dense<[[3, 4]]> : tensor<1x2xi64>
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %c = "tfl.concatenation"(%a, %b) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<2x2xi64>
    %r = "tfl.transpose"(%c, %p) : (tensor<2x2xi64>, tensor<2xi32>) -> tensor<2x2xi64>
    return %r : tensor<2x2xi64>
  }
  "onnx.EntryPoint"() {func = @integer_concat_transpose} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<{{\[\[1, 3\], \[2, 4\]\]}}> : tensor<2x2xi64>
// CHECK-NOT: "tfl.
// CHECK: return
