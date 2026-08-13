// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --shape-inference %s | FileCheck %s

module {
  func.func @compress_shape(%arg0: tensor<8xf32>) -> tensor<?xf32> {
    %condition = onnx.Constant dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>
    %0 = "onnx.Compress"(%arg0, %condition) <{axis = 0 : si64}> : (tensor<8xf32>, tensor<8xi1>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK: func.func @compress_shape(%arg0: tensor<8xf32>) -> tensor<4xf32>
// CHECK: "onnx.Compress"{{.*}} -> tensor<4xf32>
