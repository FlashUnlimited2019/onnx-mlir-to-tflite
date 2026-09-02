// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @identity_layout_elementwise_matmul(%lhs: tensor<1x2x4x8xf32>, %rhs: tensor<1x2x8x6xf32>) -> tensor<1x2x4x6xf32> {
    %bias = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %product = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x4x8xf32>, tensor<1x2x8x6xf32>) -> tensor<1x2x4x6xf32>
    %activated = "onnx.Relu"(%product) : (tensor<1x2x4x6xf32>) -> tensor<1x2x4x6xf32>
    %result = "onnx.Add"(%activated, %bias) : (tensor<1x2x4x6xf32>, tensor<f32>) -> tensor<1x2x4x6xf32>
    return %result : tensor<1x2x4x6xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_elementwise_matmul} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<1x2x8x6xf32>) -> tensor<1x2x4x6xf32>
// CHECK: "tfl.batch_matmul"{{.*}} : (tensor<1x2x4x8xf32>, tensor<1x2x8x6xf32>) -> tensor<1x2x4x6xf32>
// CHECK: "tfl.relu"{{.*}} : (tensor<1x2x4x6xf32>) -> tensor<1x2x4x6xf32>
// CHECK: "tfl.add"{{.*}} : (tensor<1x2x4x6xf32>, tensor<f32>) -> tensor<1x2x4x6xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @encoded_rank6_singleton_gather(%input: tensor<2x2500x8x1x4x2xf32>) -> tensor<2x8x2500x4x2xf32> {
    %index = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %result = "onnx.Gather"(%input, %index) <{axis = 3 : si64}> : (tensor<2x2500x8x1x4x2xf32>, tensor<i64>) -> tensor<2x2500x8x4x2xf32>
    %moved = "onnx.Transpose"(%result) <{perm = [0, 2, 1, 3, 4]}> : (tensor<2x2500x8x4x2xf32>) -> tensor<2x8x2500x4x2xf32>
    return %moved : tensor<2x8x2500x4x2xf32>
  }
  "onnx.EntryPoint"() {func = @encoded_rank6_singleton_gather} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<2x2500x8x1x4x2xf32>) -> tensor<2x2500x8x4x2xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x2500x8x1x4x2xf32>, tensor<5xi32>) -> tensor<2x2500x8x4x2xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: array<i64:
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_slice_concat(%input: tensor<1x2x4x6xf32>) -> tensor<1x4x2x6xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 4, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
    %starts = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %anchored = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x4x6xf32>, tensor<4xi64>) -> tensor<1x2x4x6xf32>
    %slice = "onnx.Slice"(%anchored, %starts, %ends, %axes, %steps) : (tensor<1x2x4x6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2x2x6xf32>
    %result = "onnx.Concat"(%slice, %slice) <{axis = 1 : si64}> : (tensor<1x2x2x6xf32>, tensor<1x2x2x6xf32>) -> tensor<1x4x2x6xf32>
    return %result : tensor<1x4x2x6xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_slice_concat} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x6xf32>) -> tensor<1x4x2x6xf32>
// CHECK: "tfl.slice"{{.*}} : (tensor<1x2x4x6xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x2x6xf32>
// CHECK: "tfl.concatenation"{{.*}} {axis = 1 : i32
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @permuted_rank3_slice(%input: tensor<1x64x384xf32>) -> tensor<1x384x20xf32> {
    %starts = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[20]> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[2]> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %transposed = "onnx.Transpose"(%input) <{perm = [0, 2, 1]}> : (tensor<1x64x384xf32>) -> tensor<1x384x64xf32>
    %result = "onnx.Slice"(%transposed, %starts, %ends, %axes, %steps) : (tensor<1x384x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x384x20xf32>
    return %result : tensor<1x384x20xf32>
  }
  "onnx.EntryPoint"() {func = @permuted_rank3_slice} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x64x384xf32>) -> tensor<1x20x384xf32>
// CHECK: arith.constant dense<[1, 20, 384]> : tensor<3xi32>
// CHECK: "tfl.slice"{{.*}} : (tensor<1x64x384xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x20x384xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_reduce(%input: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
    %anchored = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %result = "onnx.ReduceSum"(%anchored, %axes) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_reduce} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK: arith.constant dense<1> : tensor<1xi32>
// CHECK: "tfl.sum"{{.*}} {keep_dims = false} : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x3x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_gather(%input: tensor<1x2x4x6xf32>) -> tensor<1x2x6xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 4, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
    %index = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %anchored = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x4x6xf32>, tensor<4xi64>) -> tensor<1x2x4x6xf32>
    %result = "onnx.Gather"(%anchored, %index) <{axis = 2 : si64}> : (tensor<1x2x4x6xf32>, tensor<i64>) -> tensor<1x2x6xf32>
    return %result : tensor<1x2x6xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_gather} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x6xf32>) -> tensor<1x2x6xf32>
// CHECK: "tfl.gather"{{.*}} {axis = 2 : i32, batch_dims = 0 : i32} : (tensor<1x2x4x6xf32>, tensor<i32>) -> tensor<1x2x6xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @permuted_layout_transpose_softmax(%input: tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %anchored = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %moved = "onnx.Transpose"(%anchored) <{perm = [0, 2, 1, 3]}> : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
    %result = "onnx.Softmax"(%moved) <{axis = 1 : si64}> : (tensor<1x3x2x4xf32>) -> tensor<1x3x2x4xf32>
    return %result : tensor<1x3x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @permuted_layout_transpose_softmax} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
// CHECK: "tfl.softmax"
// CHECK-NOT: dense<[0, 2, 1, 3]>
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_gather_elements(%input: tensor<1x2x3x4xf32>, %indices: tensor<1x2x3x4xi64>) -> tensor<1x2x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %anchored = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %result = "onnx.GatherElements"(%anchored, %indices) <{axis = 2 : si64}> : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xi64>) -> tensor<1x2x3x4xf32>
    return %result : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_gather_elements} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xi64>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.gather_nd"{{.*}} -> tensor<1x2x3x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_scatter_elements(%data: tensor<1x2x3x4xf32>, %indices: tensor<1x2x2x4xi64>, %updates: tensor<1x2x2x4xf32>) -> tensor<1x2x3x4xf32> {
    %data_shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %updates_shape = "onnx.Constant"() {value = dense<[1, 2, 2, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %anchored_data = "onnx.Reshape"(%data, %data_shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %anchored_updates = "onnx.Reshape"(%updates, %updates_shape) <{allowzero = 0 : si64}> : (tensor<1x2x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x4xf32>
    %result = "onnx.ScatterElements"(%anchored_data, %indices, %anchored_updates) <{axis = 2 : si64, reduction = "none"}> : (tensor<1x2x3x4xf32>, tensor<1x2x2x4xi64>, tensor<1x2x2x4xf32>) -> tensor<1x2x3x4xf32>
    return %result : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_scatter_elements} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x2x4xi64>, %arg2: tensor<1x2x2x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.scatter_nd"
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_scatter_nd(%data: tensor<1x2x3x4xf32>, %updates: tensor<1x3x4xf32>) -> tensor<1x2x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %indices = "onnx.Constant"() {value = dense<[[0, 1]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
    %anchored = "onnx.Reshape"(%data, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %result = "onnx.ScatterND"(%anchored, %indices, %updates) <{reduction = "none"}> : (tensor<1x2x3x4xf32>, tensor<1x2xi64>, tensor<1x3x4xf32>) -> tensor<1x2x3x4xf32>
    return %result : tensor<1x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_scatter_nd} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK: "tfl.scatter_nd"
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_gather_nd(%data: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %indices = "onnx.Constant"() {value = dense<[[0, 1]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
    %anchored = "onnx.Reshape"(%data, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
    %result = "onnx.GatherND"(%anchored, %indices) <{batch_dims = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<1x2xi64>) -> tensor<1x3x4xf32>
    return %result : tensor<1x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_gather_nd} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x3x4xf32>
// CHECK: "tfl.gather_nd"{{.*}} -> tensor<1x3x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @channels_last_conv_pool_chain(%input: tensor<1x3x8x8xf32>, %filter: tensor<4x3x3x3xf32>) -> tensor<1x4x1x1xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %conv = "onnx.Conv"(%input, %filter, %none) <{auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x6x6xf32>
    %maximum = "onnx.MaxPoolSingleOut"(%conv) <{auto_pad = "VALID", ceil_mode = 0 : si64, dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x4x6x6xf32>) -> tensor<1x4x3x3xf32>
    %result = "onnx.AveragePool"(%maximum) <{auto_pad = "VALID", ceil_mode = 0 : si64, count_include_pad = 0 : si64, dilations = [1, 1], kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [3, 3]}> : (tensor<1x4x3x3xf32>) -> tensor<1x4x1x1xf32>
    return %result : tensor<1x4x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @channels_last_conv_pool_chain} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x8x8x3xf32>, %arg1: tensor<4x3x3x3xf32>) -> tensor<1x1x1x4xf32>
// CHECK: "tfl.conv_2d"{{.*}} -> tensor<1x6x6x4xf32>
// CHECK: "tfl.max_pool_2d"{{.*}} -> tensor<1x3x3x4xf32>
// CHECK: "tfl.average_pool_2d"{{.*}} -> tensor<1x1x1x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @identity_layout_reshape(%input: tensor<1x2x3x4xf32>) -> tensor<1x6x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 6, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
    %result = "onnx.Reshape"(%input, %shape) <{allowzero = 0 : si64}> : (tensor<1x2x3x4xf32>, tensor<3xi64>) -> tensor<1x6x4xf32>
    return %result : tensor<1x6x4xf32>
  }
  "onnx.EntryPoint"() {func = @identity_layout_reshape} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x6x4xf32>
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x2x3x4xf32>, tensor<3xi32>) -> tensor<1x6x4xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @singleton_reshape_layout_chain(%lhs: tensor<1x2x4x8xf32>, %rhs: tensor<1x2x8x8xf32>) -> tensor<1x8x2x1x4xf32> {
    %shape = "onnx.Constant"() {value = dense<[1, 8, 2, 1, 4]> : tensor<5xi64>} : () -> tensor<5xi64>
    %product = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>
    %moved = "onnx.Transpose"(%product) <{perm = [0, 3, 1, 2]}> : (tensor<1x2x4x8xf32>) -> tensor<1x8x2x4xf32>
    %result = "onnx.Reshape"(%moved, %shape) <{allowzero = 0 : si64}> : (tensor<1x8x2x4xf32>, tensor<5xi64>) -> tensor<1x8x2x1x4xf32>
    return %result : tensor<1x8x2x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @singleton_reshape_layout_chain} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<1x2x8x8xf32>) -> tensor<1x2x4x8x1xf32>
// CHECK: "tfl.batch_matmul"
// CHECK: "tfl.reshape"{{.*}} : (tensor<1x2x4x8xf32>, tensor<5xi32>) -> tensor<1x2x4x8x1xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.

// -----

module {
  func.func @singleton_axis_layout_chain(%lhs: tensor<1x2x4x8xf32>, %rhs: tensor<1x2x8x8xf32>) -> tensor<1x8x2x4xf32> {
    %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
    %product = "onnx.MatMul"(%lhs, %rhs) : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>
    %moved = "onnx.Transpose"(%product) <{perm = [0, 3, 1, 2]}> : (tensor<1x2x4x8xf32>) -> tensor<1x8x2x4xf32>
    %expanded = "onnx.Unsqueeze"(%moved, %axes) : (tensor<1x8x2x4xf32>, tensor<1xi64>) -> tensor<1x8x2x1x4xf32>
    %result = "onnx.Squeeze"(%expanded, %axes) : (tensor<1x8x2x1x4xf32>, tensor<1xi64>) -> tensor<1x8x2x4xf32>
    return %result : tensor<1x8x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @singleton_axis_layout_chain} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>
// CHECK: "tfl.batch_matmul"
// CHECK-COUNT-2: "tfl.reshape"
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: onnx.
