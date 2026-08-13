// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @mvn_mish(%arg0: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
    %0 = "onnx.MeanVarianceNormalization"(%arg0) <{axes = [0, 2]}> : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    %1 = "onnx.Mish"(%0) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %1 : tensor<1x2x4xf32>
  }
  "onnx.EntryPoint"() {func = @mvn_mish} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: "tfl.mean"
// CHECK: "tfl.sqrt"
// CHECK: "tfl.exp"
// CHECK: "tfl.tanh"
// CHECK-NOT: onnx.

// -----

module {
  func.func @scatter_reductions(%data: tensor<2x3xf32>, %updates: tensor<1x3xf32>) -> tensor<2x3xf32> {
    %indices = arith.constant dense<0> : tensor<1x1xi64>
    %0 = "onnx.ScatterND"(%data, %indices, %updates) <{reduction = "add"}> : (tensor<2x3xf32>, tensor<1x1xi64>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %1 = "onnx.ScatterND"(%0, %indices, %updates) <{reduction = "mul"}> : (tensor<2x3xf32>, tensor<1x1xi64>, tensor<1x3xf32>) -> tensor<2x3xf32>
    %2 = "onnx.ScatterND"(%1, %indices, %updates) <{reduction = "max"}> : (tensor<2x3xf32>, tensor<1x1xi64>, tensor<1x3xf32>) -> tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @scatter_reductions} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-DAG: "tfl.scatter_nd"
// CHECK-DAG: "tfl.mul"
// CHECK-DAG: "tfl.maximum"
// CHECK-NOT: onnx.

// -----

module {
  func.func @stft_rank3(%signal: tensor<1x16x1xf32>) -> tensor<1x5x5x2xf32> {
    %step = arith.constant dense<2> : tensor<i64>
    %window = arith.constant dense<1.0> : tensor<8xf32>
    %length = arith.constant dense<8> : tensor<i64>
    %0 = "onnx.STFT"(%signal, %step, %window, %length) <{onesided = 1 : si64}> : (tensor<1x16x1xf32>, tensor<i64>, tensor<8xf32>, tensor<i64>) -> tensor<1x5x5x2xf32>
    return %0 : tensor<1x5x5x2xf32>
  }
  "onnx.EntryPoint"() {func = @stft_rank3} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}}tensor<1x16xf32>
// CHECK: "tfl.rfft2d"
// CHECK-NOT: onnx.

// -----

module {
  func.func @dft(%input: tensor<1x4x2xf32>) -> tensor<1x4x2xf32> {
    %none = "onnx.NoValue"() : () -> none
    %axis = arith.constant dense<1> : tensor<i64>
    %0 = "onnx.DFT"(%input, %none, %axis) <{inverse = 0 : si64, onesided = 0 : si64}> : (tensor<1x4x2xf32>, none, tensor<i64>) -> tensor<1x4x2xf32>
    return %0 : tensor<1x4x2xf32>
  }
  "onnx.EntryPoint"() {func = @dft} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK-COUNT-4: "tfl.batch_matmul"
// CHECK: "tfl.pack"
// CHECK-NOT: onnx.

// -----

module {
  func.func @det(%input: tensor<1x2x2xf32>) -> tensor<1xf32> {
    %0 = "onnx.Det"(%input) : (tensor<1x2x2xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @det} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.div"
// CHECK: "tfl.mul"
// CHECK-NOT: onnx.

// -----

module {
  func.func @nll(%input: tensor<1x2x2xf32>) -> tensor<1xf32> {
    %labels = arith.constant dense<[[0, 1]]> : tensor<1x2xi64>
    %none = "onnx.NoValue"() : () -> none
    %0 = "onnx.NegativeLogLikelihoodLoss"(%input, %labels, %none) <{reduction = "mean"}> : (tensor<1x2x2xf32>, tensor<1x2xi64>, none) -> tensor<f32>
    %shape = arith.constant dense<1> : tensor<1xi64>
    %1 = "onnx.Reshape"(%0, %shape) <{allowzero = 0 : si64}> : (tensor<f32>, tensor<1xi64>) -> tensor<1xf32>
    return %1 : tensor<1xf32>
  }
  "onnx.EntryPoint"() {func = @nll} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.neg"
// CHECK: "tfl.mean"
// CHECK-NOT: onnx.

// -----

module {
  func.func @crop(%input: tensor<1x1x6xf32>) -> tensor<1x1x4xf32> {
    %shape = arith.constant dense<4> : tensor<1xi64>
    %0 = "onnx.CenterCropPad"(%input, %shape) <{axes = [2]}> : (tensor<1x1x6xf32>, tensor<1xi64>) -> tensor<1x1x4xf32>
    return %0 : tensor<1x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @crop} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.slice"
// CHECK-NOT: onnx.

// -----

module {
  func.func @col2im(%input: tensor<1x2x2xf32>) -> tensor<1x1x4xf32> {
    %image = arith.constant dense<4> : tensor<1xi64>
    %block = arith.constant dense<2> : tensor<1xi64>
    %0 = "onnx.Col2Im"(%input, %image, %block) <{strides = [2]}> : (tensor<1x2x2xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x4xf32>
    return %0 : tensor<1x1x4xf32>
  }
  "onnx.EntryPoint"() {func = @col2im} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather"
// CHECK-NOT: onnx.

// -----

module {
  func.func @affine_grid() -> tensor<1x2x2x2x3xf32> {
    %theta = arith.constant dense<[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]> : tensor<1x3x4xf32>
    %size = arith.constant dense<[1, 1, 2, 2, 2]> : tensor<5xi64>
    %0 = "onnx.AffineGrid"(%theta, %size) <{align_corners = 0 : si64}> : (tensor<1x3x4xf32>, tensor<5xi64>) -> tensor<1x2x2x2x3xf32>
    return %0 : tensor<1x2x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @affine_grid} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: arith.constant dense<
// CHECK-NOT: onnx.

// -----

module {
  func.func @deform_conv(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    %weight = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
    %offset = arith.constant dense<0.0> : tensor<1x2x2x2xf32>
    %bias = arith.constant dense<0.0> : tensor<1xf32>
    %mask = arith.constant dense<1.0> : tensor<1x1x2x2xf32>
    %0 = "onnx.DeformConv"(%input, %weight, %offset, %bias, %mask) <{group = 1 : si64, kernel_shape = [1, 1], offset_group = 1 : si64}> : (tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>, tensor<1x2x2x2xf32>, tensor<1xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    return %0 : tensor<1x1x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @deform_conv} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.sum"
// CHECK-NOT: onnx.

// -----

module {
  func.func @grid_sample_rank5(%input: tensor<1x1x2x2x2xf32>) -> tensor<1x1x1x1x1xf32> {
    %grid = arith.constant dense<0.0> : tensor<1x1x1x1x3xf32>
    %0 = "onnx.GridSample"(%input, %grid) <{align_corners = 0 : si64, mode = "nearest", padding_mode = "border"}> : (tensor<1x1x2x2x2xf32>, tensor<1x1x1x1x3xf32>) -> tensor<1x1x1x1x1xf32>
    return %0 : tensor<1x1x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @grid_sample_rank5} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK-NOT: onnx.

// -----

module {
  func.func @roi_align(%input: tensor<1x1x2x2xf32>) -> tensor<1x1x1x1xf32> {
    %rois = arith.constant dense<[[0.0, 0.0, 1.0, 1.0]]> : tensor<1x4xf32>
    %batch = arith.constant dense<0> : tensor<1xi64>
    %0 = "onnx.RoiAlign"(%input, %rois, %batch) <{coordinate_transformation_mode = "half_pixel", mode = "avg", output_height = 1 : si64, output_width = 1 : si64, sampling_ratio = 2 : si64, spatial_scale = 1.0 : f32}> : (tensor<1x1x2x2xf32>, tensor<1x4xf32>, tensor<1xi64>) -> tensor<1x1x1x1xf32>
    return %0 : tensor<1x1x1x1xf32>
  }
  "onnx.EntryPoint"() {func = @roi_align} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.gather_nd"
// CHECK: "tfl.sum"
// CHECK-NOT: onnx.

// -----

module {
  func.func @integer_one_hot(%indices: tensor<2xi64>) -> tensor<2x3xf32> {
    %depth = arith.constant dense<3> : tensor<1xi64>
    %values = arith.constant dense<[0, 1]> : tensor<2xi64>
    %0 = "onnx.OneHot"(%indices, %depth, %values) <{axis = 1 : si64}> : (tensor<2xi64>, tensor<1xi64>, tensor<2xi64>) -> tensor<2x3xi64>
    %1 = "onnx.Cast"(%0) <{saturate = 1 : si64, to = f32}> : (tensor<2x3xi64>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  "onnx.EntryPoint"() {func = @integer_one_hot} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.one_hot"
// CHECK: "tfl.cast"
// CHECK-NOT: onnx.
