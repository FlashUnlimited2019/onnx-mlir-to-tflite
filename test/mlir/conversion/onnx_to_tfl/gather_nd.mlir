// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @gather_nd_negative(%arg0: tensor<5x6xf32>) -> tensor<3x6xf32> {
    %indices = "onnx.Constant"() {value = dense<[[-1], [-3], [0]]> : tensor<3x1xi64>} : () -> tensor<3x1xi64>
    %0 = "onnx.GatherND"(%arg0, %indices) <{batch_dims = 0 : si64}> : (tensor<5x6xf32>, tensor<3x1xi64>) -> tensor<3x6xf32>
    return %0 : tensor<3x6xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_negative} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<5x6xf32>) -> tensor<3x6xf32>
// CHECK: arith.constant {{.*}} : tensor<3x1xi32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<5x6xf32>, tensor<3x1xi32>) -> tensor<3x6xf32>

// -----

module {
  func.func @gather_nd_batch1(%arg0: tensor<2x3x4xf32>) -> tensor<2x2x4xf32> {
    %indices = "onnx.Constant"() {value = dense<[[[0], [2]], [[1], [0]]]> : tensor<2x2x1xi64>} : () -> tensor<2x2x1xi64>
    %0 = "onnx.GatherND"(%arg0, %indices) <{batch_dims = 1 : si64}> : (tensor<2x3x4xf32>, tensor<2x2x1xi64>) -> tensor<2x2x4xf32>
    return %0 : tensor<2x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_batch1} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>) -> tensor<2x2x4xf32>
// CHECK: arith.constant {{.*}} : tensor<2x2x2xi32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<2x3x4xf32>, tensor<2x2x2xi32>) -> tensor<2x2x4xf32>

// -----

module {
  func.func @gather_nd_rank4_data(%arg0: tensor<2x3x4x5xf32>) -> tensor<3x4x5xf32> {
    %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2], [0, 0]]> : tensor<3x2xi64>} : () -> tensor<3x2xi64>
    %0 = "onnx.GatherND"(%arg0, %indices) <{batch_dims = 0 : si64}> : (tensor<2x3x4x5xf32>, tensor<3x2xi64>) -> tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_rank4_data} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x4x5x3xf32>) -> tensor<3x4x5xf32>
// CHECK: "tfl.transpose"(%arg0, {{.*}}) : (tensor<2x4x5x3xf32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
// CHECK: "tfl.gather_nd"({{.*}}, {{.*}}) : (tensor<2x3x4x5xf32>, tensor<3x2xi32>) -> tensor<3x4x5xf32>

// -----

module {
  func.func @gather_nd_rank5_to_rank4(%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x2x4x5xf32> {
    %indices = "onnx.Constant"() {value = dense<[[[0, 1], [2, 0]], [[1, 0], [2, 1]]]> : tensor<2x2x2xi64>} : () -> tensor<2x2x2xi64>
    %0 = "onnx.GatherND"(%arg0, %indices) <{batch_dims = 1 : si64}> : (tensor<2x3x2x4x5xf32>, tensor<2x2x2xi64>) -> tensor<2x2x4x5xf32>
    return %0 : tensor<2x2x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_rank5_to_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x2x4x5xf32>) -> tensor<2x4x5x2xf32>
// CHECK: arith.constant {{.*}} : tensor<2x2x3xi32>
// CHECK: "tfl.gather_nd"(%arg0, {{.*}}) : (tensor<2x3x2x4x5xf32>, tensor<2x2x3xi32>) -> tensor<2x2x4x5xf32>
// CHECK: "tfl.transpose"({{.*}}) : (tensor<2x2x4x5xf32>, tensor<4xi32>) -> tensor<2x4x5x2xf32>

// -----

module {
  func.func @gather_nd_runtime_batch1(%data: tensor<2x3x4xf32>, %indices: tensor<2x2x1xi64>) -> tensor<2x2x4xf32> {
    %0 = "onnx.GatherND"(%data, %indices) <{batch_dims = 1 : si64}> : (tensor<2x3x4xf32>, tensor<2x2x1xi64>) -> tensor<2x2x4xf32>
    return %0 : tensor<2x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_runtime_batch1} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x2x1xi64>) -> tensor<2x2x4xf32>
// CHECK: "tfl.cast"(%arg1) : (tensor<2x2x1xi64>) -> tensor<2x2x1xi32>
// CHECK: "tfl.less"{{.*}} : (tensor<2x2x1xi32>, tensor<i32>) -> tensor<2x2x1xi1>
// CHECK: "tfl.add"{{.*}} : (tensor<2x2x1xi32>, tensor<1xi32>) -> tensor<2x2x1xi32>
// CHECK: "tfl.select_v2"{{.*}} : (tensor<2x2x1xi1>, tensor<2x2x1xi32>, tensor<2x2x1xi32>) -> tensor<2x2x1xi32>
// CHECK: "tfl.concatenation"{{.*}} {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<2x2x1xi32>, tensor<2x2x1xi32>) -> tensor<2x2x2xi32>
// CHECK: "tfl.gather_nd"{{.*}} : (tensor<2x3x4xf32>, tensor<2x2x2xi32>) -> tensor<2x2x4xf32>

// -----

module {
  func.func @gather_nd_runtime_rank5(%data: tensor<1x2x3x4x5xf32>, %indices: tensor<1x2x3x1x1xi64>) -> tensor<1x2x3x1x5xf32> {
    %0 = "onnx.GatherND"(%data, %indices) <{batch_dims = 3 : si64}> : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x1x1xi64>) -> tensor<1x2x3x1x5xf32>
    return %0 : tensor<1x2x3x1x5xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_runtime_rank5} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4x5xf32>, %arg1: tensor<1x2x3x1x1xi64>) -> tensor<1x2x3x1x5xf32>
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x2x3x1x1xi64>, tensor<1xi32>) -> tensor<6xi64>
// CHECK: "tfl.cast"{{.*}} : (tensor<6xi64>) -> tensor<6xi32>
// CHECK: "tfl.less"{{.*}} : (tensor<6xi32>, tensor<i32>) -> tensor<6xi1>
// CHECK: "tfl.select_v2"{{.*}} : (tensor<6xi1>, tensor<6xi32>, tensor<6xi32>) -> tensor<6xi32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<6xi32>, tensor<5xi32>) -> tensor<1x2x3x1x1xi32>
// CHECK: "tfl.concatenation"{{.*}} {axis = 4 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3x1x3xi32>, tensor<1x2x3x1x1xi32>) -> tensor<1x2x3x1x4xi32>
// CHECK: "tfl.gather_nd"{{.*}} : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x1x4xi32>) -> tensor<1x2x3x1x5xf32>

// -----

module {
  func.func @gather_nd_identity_reshape_i1(%input: tensor<1x4xi64>) -> tensor<1x1x1x4xf32> {
    %data = "onnx.Cast"(%input) <{saturate = 1 : si64, to = i1}> : (tensor<1x4xi64>) -> tensor<1x4xi1>
    %indices = "onnx.Constant"() {value = dense<[[[[[0, 0], [0, 1], [0, 2], [0, 3]]]]]> : tensor<1x1x1x4x2xi64>} : () -> tensor<1x1x1x4x2xi64>
    %0 = "onnx.GatherND"(%data, %indices) <{batch_dims = 0 : si64}> : (tensor<1x4xi1>, tensor<1x1x1x4x2xi64>) -> tensor<1x1x1x4xi1>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<1x1x1x4xi1>) -> tensor<1x1x1x4xf32>
    return %1 : tensor<1x1x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @gather_nd_identity_reshape_i1} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x4xi64>) -> tensor<1x1x4x1xf32>
// CHECK: "tfl.cast"(%arg0) : (tensor<1x4xi64>) -> tensor<1x4xi1>
// CHECK: "tfl.reshape"({{.*}}, {{.*}}) : (tensor<1x4xi1>, tensor<4xi32>) -> tensor<1x1x1x4xi1>
// CHECK: "tfl.cast"{{.*}} : (tensor<1x1x1x4xi1>) -> tensor<1x1x1x4xf32>
// CHECK: "tfl.transpose"{{.*}} : (tensor<1x1x1x4xf32>, tensor<4xi32>) -> tensor<1x1x4x1xf32>
// CHECK-NOT: "tfl.gather_nd"
