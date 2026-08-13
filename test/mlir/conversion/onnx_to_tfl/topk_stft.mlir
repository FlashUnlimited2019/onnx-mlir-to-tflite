// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @topk(%arg0: tensor<2x4x6xf32>) -> tensor<2x2x6xf32> {
    %k = arith.constant dense<2> : tensor<1xi64>
    %values, %indices = "onnx.TopK"(%arg0, %k) <{axis = 1 : si64, largest = 0 : si64, sorted = 0 : si64}> : (tensor<2x4x6xf32>, tensor<1xi64>) -> (tensor<2x2x6xf32>, tensor<2x2x6xi64>)
    return %values : tensor<2x2x6xf32>
  }
  "onnx.EntryPoint"() {func = @topk} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.neg"
// CHECK: "tfl.topk_v2"
// CHECK: "tfl.cast"
// CHECK-NOT: "tfl.reverse_v2"
// CHECK-NOT: onnx.

// -----

module {
  func.func @stft(%arg0: tensor<1x16xf32>) -> tensor<1x5x5x2xf32> {
    %step = arith.constant dense<2> : tensor<i64>
    %window = arith.constant dense<1.0> : tensor<8xf32>
    %length = arith.constant dense<8> : tensor<i64>
    %result = "onnx.STFT"(%arg0, %step, %window, %length) <{onesided = 1 : si64}> : (tensor<1x16xf32>, tensor<i64>, tensor<8xf32>, tensor<i64>) -> tensor<1x5x5x2xf32>
    return %result : tensor<1x5x5x2xf32>
  }
  "onnx.EntryPoint"() {func = @stft} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.rfft2d"
// CHECK: "tfl.real"
// CHECK: "tfl.imag"
// CHECK: "tfl.pack"
// CHECK-NOT: onnx.

// -----

module {
  func.func @stft_non_power_of_two(%arg0: tensor<1x20xf32>) -> tensor<1x6x10x2xf32> {
    %step = arith.constant dense<2> : tensor<i64>
    %window = arith.constant dense<1.0> : tensor<10xf32>
    %length = arith.constant dense<10> : tensor<i64>
    %result = "onnx.STFT"(%arg0, %step, %window, %length) <{onesided = 0 : si64}> : (tensor<1x20xf32>, tensor<i64>, tensor<10xf32>, tensor<i64>) -> tensor<1x6x10x2xf32>
    return %result : tensor<1x6x10x2xf32>
  }
  "onnx.EntryPoint"() {func = @stft_non_power_of_two} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: "tfl.batch_matmul"
// CHECK: "tfl.pack"
// CHECK-NOT: "tfl.rfft2d"
// CHECK-NOT: onnx.
