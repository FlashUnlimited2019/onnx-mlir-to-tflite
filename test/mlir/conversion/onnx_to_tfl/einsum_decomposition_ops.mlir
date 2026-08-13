// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @where_rank2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %condition = onnx.Constant dense<[[true, false], [false, true]]> : tensor<2x2xi1>
    %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %0 = "onnx.Where"(%condition, %arg0, %zero) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  "onnx.EntryPoint"() {func = @where_rank2} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: arith.constant dense<{{.*true.*false.*}}> : tensor<2x2xi1>
// CHECK: "tfl.select_v2"({{.*}}, %arg0, {{.*}}) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>

// -----

module {
  func.func @where_rank4_logical(%arg0: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32> {
    %condition = onnx.Constant dense<[[[[true, false], [false, true]]]]> : tensor<1x1x2x2xi1>
    %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %0 = "onnx.Where"(%condition, %arg0, %zero) : (tensor<1x1x2x2xi1>, tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
    return %0 : tensor<1x3x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @where_rank4_logical} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
// CHECK: arith.constant dense<{{.*true.*false.*}}> : tensor<1x1x2x2xi1>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<1x2x2x3xf32>, tensor<4xi32>) -> tensor<1x3x2x2xf32>
// CHECK: "tfl.select_v2"({{.*}}) : (tensor<1x1x2x2xi1>, tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x2x2xf32>, tensor<4xi32>) -> tensor<1x2x2x3xf32>

// -----

module {
  func.func @unsqueeze_rank1(%arg0: tensor<4xf32>) -> tensor<4x1xf32> {
    %axes = onnx.Constant dense<1> : tensor<1xi64>
    %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<4xf32>, tensor<1xi64>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
  "onnx.EntryPoint"() {func = @unsqueeze_rank1} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<4xf32>) -> tensor<4x1xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<4xf32>, tensor<2xi32>) -> tensor<4x1xf32>

// -----

module {
  func.func @unsqueeze_into_rank4(%arg0: tensor<1x3x5xf32>) -> tensor<1x3x1x5xf32> {
    %axes = onnx.Constant dense<2> : tensor<1xi64>
    %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<1x3x5xf32>, tensor<1xi64>) -> tensor<1x3x1x5xf32>
    return %0 : tensor<1x3x1x5xf32>
  }
  "onnx.EntryPoint"() {func = @unsqueeze_into_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x3x5xf32>) -> tensor<1x1x5x3xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x3x5xf32>, tensor<4xi32>) -> tensor<1x3x1x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<1x3x1x5xf32>, tensor<4xi32>) -> tensor<1x1x5x3xf32>
