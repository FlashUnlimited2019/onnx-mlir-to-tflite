// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @rank2(%x: tensor<1x4xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>, %mean: tensor<4xf32>, %var: tensor<4xf32>) -> tensor<1x4xf32> {
    %0 = "onnx.BatchNormalizationInferenceMode"(%x, %scale, %bias, %mean, %var) <{epsilon = 1.000000e-05 : f32, momentum = 9.000000e-01 : f32}> : (tensor<1x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
  "onnx.EntryPoint"() {func = @rank2} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4xf32>
// CHECK: "tfl.sqrt"{{.*}} : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: "tfl.div"{{.*}} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK: "tfl.mul"{{.*}} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK: "tfl.add"{{.*}} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @rank3(%x: tensor<1x4x7xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>, %mean: tensor<4xf32>, %var: tensor<4xf32>) -> tensor<1x4x7xf32> {
    %0 = "onnx.BatchNormalizationInferenceMode"(%x, %scale, %bias, %mean, %var) <{epsilon = 1.000000e-05 : f32, momentum = 9.000000e-01 : f32}> : (tensor<1x4x7xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x7xf32>
    return %0 : tensor<1x4x7xf32>
  }
  "onnx.EntryPoint"() {func = @rank3} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x7xf32>
// CHECK-COUNT-2: "tfl.reshape"{{.*}} : (tensor<4xf32>, tensor<3xi32>) -> tensor<1x4x1xf32>
// CHECK: "tfl.mul"{{.*}} : (tensor<1x4x7xf32>, tensor<1x4x1xf32>) -> tensor<1x4x7xf32>
// CHECK: "tfl.add"{{.*}} : (tensor<1x4x7xf32>, tensor<1x4x1xf32>) -> tensor<1x4x7xf32>
// CHECK-NOT: onnx.
