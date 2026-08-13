// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @legacy_math(%input: tensor<1x2x2x2x2xf32>) -> (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) {
    %one = arith.constant dense<1.0> : tensor<f32>
    %ceil = "onnx.Ceil"(%input) : (tensor<1x2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %positive = "onnx.Add"(%input, %one) : (tensor<1x2x2x2x2xf32>, tensor<f32>) -> tensor<1x2x2x2x2xf32>
    %acosh = "onnx.Acosh"(%positive) : (tensor<1x2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %hardmax = "onnx.Hardmax"(%input) <{axis = -1 : si64}> : (tensor<1x2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %condition = "onnx.Greater"(%input, %one) : (tensor<1x2x2x2x2xf32>, tensor<f32>) -> tensor<1x2x2x2x2xi1>
    %negated = "onnx.Not"(%condition) : (tensor<1x2x2x2x2xi1>) -> tensor<1x2x2x2x2xi1>
    %cast = "onnx.Cast"(%negated) <{saturate = 1 : si64, to = f32}> : (tensor<1x2x2x2x2xi1>) -> tensor<1x2x2x2x2xf32>
    return %ceil, %acosh, %hardmax, %cast : tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @legacy_math} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.ceil"
// CHECK: "tfl.sqrt"
// CHECK: "tfl.log"
// CHECK: "tfl.arg_max"
// CHECK: "tfl.one_hot"
// CHECK: "tfl.logical_not"
// CHECK-NOT: onnx.

// -----

module {
  func.func @lp_pool(%input: tensor<1x2x2x4x4xf32>) -> (tensor<1x2x2x2x2xf32>, tensor<1x2x1x1x1xf32>) {
    %local = "onnx.LpPool"(%input) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [1, 2, 2], p = 2 : si64, strides = [1, 2, 2]}> : (tensor<1x2x2x4x4xf32>) -> tensor<1x2x2x2x2xf32>
    %global = "onnx.GlobalLpPool"(%input) <{p = 2 : si64}> : (tensor<1x2x2x4x4xf32>) -> tensor<1x2x1x1x1xf32>
    return %local, %global : tensor<1x2x2x2x2xf32>, tensor<1x2x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @lp_pool} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.mul"
// CHECK: "tfl.slice"
// CHECK: "tfl.sum"
// CHECK: "tfl.sqrt"
// CHECK: "tfl.concatenation"
// CHECK: "tfl.sum"{{.*}} {keep_dims = true}
// CHECK-NOT: onnx.

// -----

module {
  func.func @indexed_pool_rank5(%input: tensor<1x1x2x4x4xf32>) -> tensor<1x1x2x4x4xf32> {
    %none = "onnx.NoValue"() : () -> none
    %values, %indices = "onnx.MaxPool"(%input) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [1, 2, 2], storage_order = 0 : si64, strides = [1, 2, 2]}> : (tensor<1x1x2x4x4xf32>) -> (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xi64>)
    %result = "onnx.MaxUnpool"(%values, %indices, %none) <{kernel_shape = [1, 2, 2], strides = [1, 2, 2]}> : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xi64>, none) -> tensor<1x1x2x4x4xf32>
    return %result : tensor<1x1x2x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @indexed_pool_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.strided_slice"
// CHECK: "tfl.greater"
// CHECK: "tfl.select_v2"
// CHECK: "tfl.scatter_nd"
// CHECK-NOT: onnx.

// -----

module {
  func.func @scatter_elements_rank5(%data: tensor<1x1x2x2x4xf32>, %indices: tensor<1x1x2x2x2xi64>, %updates: tensor<1x1x2x2x2xf32>) -> tensor<1x1x2x2x4xf32> {
    %result = "onnx.ScatterElements"(%data, %indices, %updates) <{axis = -1 : si64, reduction = "none"}> : (tensor<1x1x2x2x4xf32>, tensor<1x1x2x2x2xi64>, tensor<1x1x2x2x2xf32>) -> tensor<1x1x2x2x4xf32>
    return %result : tensor<1x1x2x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_elements_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<8xi32>
// CHECK: "tfl.less"{{.*}}tensor<8xi1>
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.scatter_nd"
// CHECK-NOT: onnx.

// -----

module {
  func.func @resize_static_ranks(%rank3: tensor<1x2x8xf32>, %rank4: tensor<1x2x4x4xf32>, %rank5: tensor<1x2x2x4x8xf32>) -> (tensor<1x2x4xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2x4xf32>) {
    %none = "onnx.NoValue"() : () -> none
    %size3 = arith.constant dense<[1, 2, 4]> : tensor<3xi64>
    %size4 = arith.constant dense<[1, 2, 2, 2]> : tensor<4xi64>
    %size5 = arith.constant dense<[1, 2, 2, 2, 4]> : tensor<5xi64>
    %out3 = "onnx.Resize"(%rank3, %none, %none, %size3) <{antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "round_prefer_floor"}> : (tensor<1x2x8xf32>, none, none, tensor<3xi64>) -> tensor<1x2x4xf32>
    %out4 = "onnx.Resize"(%rank4, %none, %none, %size4) <{antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "round_prefer_floor"}> : (tensor<1x2x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x2x2x2xf32>
    %out5 = "onnx.Resize"(%rank5, %none, %none, %size5) <{antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "round_prefer_floor"}> : (tensor<1x2x2x4x8xf32>, none, none, tensor<5xi64>) -> tensor<1x2x2x2x4xf32>
    return %out3, %out4, %out5 : tensor<1x2x4xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @resize_static_ranks} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-5: "tfl.gather"
// CHECK-NOT: "tfl.resize_nearest_neighbor"
// CHECK-NOT: onnx.
