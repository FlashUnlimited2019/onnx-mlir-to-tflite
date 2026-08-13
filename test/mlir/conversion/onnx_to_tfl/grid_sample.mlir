// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @bilinear(%x: tensor<1x2x4x5xf32>, %grid: tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32> {
    %0 = "onnx.GridSample"(%x, %grid) <{align_corners = 0 : si64, mode = "bilinear", padding_mode = "zeros"}> : (tensor<1x2x4x5xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @bilinear} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<1x3x2x2xf32>)
// CHECK: "tfl.transpose"(%arg1, {{.*}}) : (tensor<1x3x2x2xf32>, tensor<4xi32>) -> tensor<1x2x3x2xf32>
// CHECK-COUNT-4: "tfl.gather_nd"{{.*}} : (tensor<1x4x5x2xf32>, tensor<1x2x3x3xi32>) -> tensor<1x2x3x2xf32>
// CHECK: return {{.*}} : tensor<1x2x3x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @cubic(%x: tensor<1x2x4x5xf32>, %grid: tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32> {
    %0 = "onnx.GridSample"(%x, %grid) <{align_corners = 1 : si64, mode = "cubic", padding_mode = "border"}> : (tensor<1x2x4x5xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @cubic} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<1x3x2x2xf32>)
// CHECK-COUNT-16: "tfl.gather_nd"{{.*}} : (tensor<1x4x5x2xf32>, tensor<1x2x3x3xi32>) -> tensor<1x2x3x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @reflection(%x: tensor<1x2x4x5xf32>, %grid: tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32> {
    %0 = "onnx.GridSample"(%x, %grid) <{align_corners = 0 : si64, mode = "bilinear", padding_mode = "reflection"}> : (tensor<1x2x4x5xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @reflection} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<1x3x2x2xf32>)
// CHECK: "tfl.floor_mod"
// CHECK: "tfl.select_v2"
// CHECK-COUNT-4: "tfl.gather_nd"{{.*}} : (tensor<1x4x5x2xf32>, tensor<1x2x3x3xi32>) -> tensor<1x2x3x2xf32>
// CHECK-NOT: onnx.

// -----

module {
  func.func @nearest(%x: tensor<1x2x4x5xf32>, %grid: tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32> {
    %0 = "onnx.GridSample"(%x, %grid) <{align_corners = 1 : si64, mode = "nearest", padding_mode = "zeros"}> : (tensor<1x2x4x5xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @nearest} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<1x3x2x2xf32>)
// CHECK-COUNT-2: "tfl.round"
// CHECK-COUNT-1: "tfl.gather_nd"{{.*}} : (tensor<1x4x5x2xf32>, tensor<1x2x3x3xi32>) -> tensor<1x2x3x2xf32>
// CHECK-NOT: onnx.
