// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// Precompute both input projections over T*B rows. Combine the update/reset
// recurrent projections and sigmoid, but keep the candidate reset separate.
module {
  func.func @packed_reset_after(%x: tensor<3x2x4xf32>, %h: tensor<1x2x3xf32>) -> (tensor<3x1x2x3xf32>, tensor<1x2x3xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x9x4xf32>
    %r = arith.constant dense<0.0625> : tensor<1x9x3xf32>
    %b = arith.constant dense<0.03125> : tensor<1x18xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.GRU"(%x, %w, %r, %b, %none, %h) <{direction = "reverse", hidden_size = 3 : si64, layout = 0 : si64, linear_before_reset = 1 : si64}> : (tensor<3x2x4xf32>, tensor<1x9x4xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, tensor<1x2x3xf32>) -> (tensor<3x1x2x3xf32>, tensor<1x2x3xf32>)
    return %y, %yh : tensor<3x1x2x3xf32>, tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @packed_reset_after} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"{{.*}}(tensor<6x4xf32>, tensor<6x4xf32>) -> tensor<6x6xf32>
// CHECK: "tfl.batch_matmul"{{.*}}(tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<6x3xf32>
// CHECK: "tfl.slice"{{.*}} -> tensor<2x6xf32>
// CHECK: "tfl.slice"{{.*}} -> tensor<2x3xf32>
// CHECK: "tfl.batch_matmul"{{.*}} -> tensor<2x6xf32>
// CHECK: "tfl.logistic"{{.*}} -> tensor<2x6xf32>
// CHECK: %[[ZR:.*]]:2 = "tfl.split"
// CHECK: %[[RH:.*]] = "tfl.batch_matmul"{{.*}} -> tensor<2x3xf32>
// CHECK: %[[RB:.*]] = "tfl.add"(%[[RH]],
// CHECK: "tfl.mul"(%[[ZR]]#1, %[[RB]])
// CHECK: "tfl.tanh"
// CHECK-NOT: "tfl.while"
// CHECK-NOT: onnx.

// -----

module {
  func.func @packed_reset_before(%x: tensor<3x2x4xf32>, %h: tensor<1x2x3xf32>) -> (tensor<3x1x2x3xf32>, tensor<1x2x3xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x9x4xf32>
    %r = arith.constant dense<0.0625> : tensor<1x9x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.GRU"(%x, %w, %r, %none, %none, %h) <{clip = 0.5 : f32, direction = "forward", hidden_size = 3 : si64, layout = 0 : si64, linear_before_reset = 0 : si64}> : (tensor<3x2x4xf32>, tensor<1x9x4xf32>, tensor<1x9x3xf32>, none, none, tensor<1x2x3xf32>) -> (tensor<3x1x2x3xf32>, tensor<1x2x3xf32>)
    return %y, %yh : tensor<3x1x2x3xf32>, tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @packed_reset_before} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.batch_matmul"{{.*}} -> tensor<6x6xf32>
// CHECK: "tfl.batch_matmul"{{.*}} -> tensor<6x3xf32>
// CHECK: "tfl.maximum"{{.*}} -> tensor<2x6xf32>
// CHECK: "tfl.minimum"{{.*}} -> tensor<2x6xf32>
// CHECK: "tfl.logistic"{{.*}} -> tensor<2x6xf32>
// CHECK: %[[ZR:.*]]:2 = "tfl.split"
// CHECK: %[[RESET:.*]] = "tfl.mul"(%[[ZR]]#1,
// CHECK: "tfl.batch_matmul"(%[[RESET]],
// CHECK: "tfl.maximum"{{.*}} -> tensor<2x3xf32>
// CHECK: "tfl.minimum"{{.*}} -> tensor<2x3xf32>
// CHECK: "tfl.tanh"
// CHECK-NOT: "tfl.while"
// CHECK-NOT: onnx.

// -----

// Short sequences keep the original lowering.
module {
  func.func @short_sequence(%x: tensor<2x2x4xf32>) -> (tensor<2x1x2x3xf32>, tensor<1x2x3xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x9x4xf32>
    %r = arith.constant dense<0.0625> : tensor<1x9x3xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.GRU"(%x, %w, %r, %none, %none, %none) <{direction = "forward", hidden_size = 3 : si64, layout = 0 : si64, linear_before_reset = 1 : si64}> : (tensor<2x2x4xf32>, tensor<1x9x4xf32>, tensor<1x9x3xf32>, none, none, none) -> (tensor<2x1x2x3xf32>, tensor<1x2x3xf32>)
    return %y, %yh : tensor<2x1x2x3xf32>, tensor<1x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @short_sequence} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.slice"
// CHECK-NOT: "tfl.split"
// CHECK: return

// -----

// The full-sequence projection exceeds the 4 MiB budget. Do not hoist it.
module {
  func.func @large_projection(%x: tensor<3x131072x1xf32>) -> (tensor<3x1x131072x1xf32>, tensor<1x131072x1xf32>) {
    %w = arith.constant dense<0.125> : tensor<1x3x1xf32>
    %r = arith.constant dense<0.0625> : tensor<1x3x1xf32>
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %yh = "onnx.GRU"(%x, %w, %r, %none, %none, %none) <{direction = "forward", hidden_size = 1 : si64, layout = 0 : si64, linear_before_reset = 1 : si64}> : (tensor<3x131072x1xf32>, tensor<1x3x1xf32>, tensor<1x3x1xf32>, none, none, none) -> (tensor<3x1x131072x1xf32>, tensor<1x131072x1xf32>)
    return %y, %yh : tensor<3x1x131072x1xf32>, tensor<1x131072x1xf32>
  }
  "onnx.EntryPoint"() {func = @large_projection} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.slice"
// CHECK-NOT: "tfl.split"
// CHECK: return
