// SPDX-License-Identifier: Apache-2.0

// RUN: split-file %s %t
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %t/zero.mlir 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental %t/zero.mlir 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %t/rank.mlir 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental %t/rank.mlir 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %t/bound.mlir 2>&1 | FileCheck %s --check-prefix=BOUND
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental %t/bound.mlir 2>&1 | FileCheck %s --check-prefix=BOUND

// ZERO: error: i64 Div requires nonzero divisors
// RANK: error: i64 Div requires nonempty static broadcast results of rank at most 5
// BOUND: error: ONNXToTFL Clip requires an omitted or scalar minimum with the input element type

//--- zero.mlir
module {
  func.func @zero_divisor(%x: tensor<i64>) -> tensor<i64> {
    %zero = onnx.Constant dense<0> : tensor<i64>
    %y = "onnx.Div"(%x, %zero) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %y : tensor<i64>
  }
  "onnx.EntryPoint"() {func = @zero_divisor} : () -> ()
}

//--- rank.mlir
module {
  func.func @unsupported_rank(%x: tensor<2x2x2x2x2x2xi64>, %d: tensor<i64>) -> tensor<2x2x2x2x2x2xi64> {
    %y = "onnx.Div"(%x, %d) : (tensor<2x2x2x2x2x2xi64>, tensor<i64>) -> tensor<2x2x2x2x2x2xi64>
    return %y : tensor<2x2x2x2x2x2xi64>
  }
  "onnx.EntryPoint"() {func = @unsupported_rank} : () -> ()
}

//--- bound.mlir
module {
  func.func @nonscalar_bound(%x: tensor<2xi64>, %lo: tensor<2xi64>, %hi: tensor<i64>) -> tensor<2xi64> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<2xi64>, tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
    return %y : tensor<2xi64>
  }
  "onnx.EntryPoint"() {func = @nonscalar_bound} : () -> ()
}
