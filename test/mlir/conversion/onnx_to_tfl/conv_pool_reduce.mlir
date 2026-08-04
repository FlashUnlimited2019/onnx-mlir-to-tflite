// RUN: onnx-mlir-opt --convert-onnx-to-tfl --canonicalize %s -split-input-file | FileCheck %s

module {
  func.func @conv_graph(%input: tensor<1x3x8x9xf32>, %filter: tensor<4x3x3x5xf32>) -> tensor<1x4x4x5xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Conv"(%input, %filter, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 5], pads = [1, 2, 1, 2], strides = [2, 2]} : (tensor<1x3x8x9xf32>, tensor<4x3x3x5xf32>, none) -> tensor<1x4x4x5xf32>
    return %0 : tensor<1x4x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @conv_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x8x9x3xf32>, %arg1: tensor<4x3x5x3xf32>)
// CHECK-DAG: arith.constant dense<{{.*}}1, 1{{.*}}2, 2{{.*}}> : tensor<4x2xi32>
// CHECK-DAG: arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: "tfl.pad"(%arg0, {{.*}}) : (tensor<1x8x9x3xf32>, tensor<4x2xi32>) -> tensor<1x10x13x3xf32>
// CHECK: "tfl.conv_2d"
// CHECK-SAME: padding = "VALID"
// CHECK-SAME: stride_h = 2 : i32
// CHECK-SAME: stride_w = 2 : i32
// CHECK-SAME: -> tensor<1x4x5x4xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @pool_graph(%input: tensor<1x3x8x9xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.MaxPoolSingleOut"(%input) {auto_pad = "NOTSET", ceil_mode = 0 : si64, dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x3x8x9xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @pool_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x8x9x3xf32>)
// CHECK: "tfl.padv2"(%arg0, {{.*}}) : (tensor<1x8x9x3xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x10x11x3xf32>
// CHECK: "tfl.max_pool_2d"
// CHECK-SAME: filter_height = 3 : i32
// CHECK-SAME: filter_width = 3 : i32
// CHECK-SAME: padding = "VALID"
// CHECK-SAME: -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @mean_graph(%input: tensor<1x3x4x5xf32>) -> tensor<1x3x1x1xf32> {
    %0 = "onnx.ReduceMeanV13"(%input) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x4x5xf32>) -> tensor<1x3x1x1xf32>
    return %0 : tensor<1x3x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @mean_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>)
// CHECK: arith.constant dense<[1, 2]> : tensor<2xi32>
// CHECK: "tfl.mean"
// CHECK-SAME: keep_dims = true
// CHECK-SAME: -> tensor<1x1x1x3xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @broadcast_graph(%input: tensor<1x3x4x5xf32>, %scale: tensor<3x1x1xf32>) -> tensor<1x3x4x5xf32> {
    %0 = "onnx.Mul"(%input, %scale) : (tensor<1x3x4x5xf32>, tensor<3x1x1xf32>) -> tensor<1x3x4x5xf32>
    return %0 : tensor<1x3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @broadcast_graph} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x3xf32>, %arg1: tensor<3x1x1xf32>)
// CHECK: arith.constant dense<[1, 1, 3]> : tensor<3xi32>
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<3x1x1xf32>, tensor<3xi32>) -> tensor<1x1x3xf32>
// CHECK: "tfl.mul"
// CHECK-SAME: -> tensor<1x4x5x3xf32>
// CHECK-NOT: onnx.
