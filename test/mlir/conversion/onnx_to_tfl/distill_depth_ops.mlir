// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// Distill-Any-Depth uses align-corners bilinear interpolation throughout its
// feature-fusion decoder and final depth head.
module {
  func.func @resize_bilinear_align_corners(%input: tensor<1x64x19x19xf32>) -> tensor<1x64x37x37xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %sizes = "onnx.Constant"() {value = dense<[1, 64, 37, 37]> : tensor<4xi64>} : () -> tensor<4xi64>
    %result = "onnx.Resize"(%input, %none, %none, %sizes) <{antialias = 0 : si64, coordinate_transformation_mode = "align_corners", mode = "linear", nearest_mode = "floor"}> : (tensor<1x64x19x19xf32>, none, none, tensor<4xi64>) -> tensor<1x64x37x37xf32>
    return %result : tensor<1x64x37x37xf32>
  }
  "onnx.EntryPoint"() {func = @resize_bilinear_align_corners} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x19x19x64xf32>) -> tensor<1x37x37x64xf32>
// CHECK: arith.constant dense<37> : tensor<2xi32>
// CHECK: "tfl.resize_bilinear"(%arg0, {{.*}}) {align_corners = true, half_pixel_centers = false} : (tensor<1x19x19x64xf32>, tensor<2xi32>) -> tensor<1x37x37x64xf32>
// CHECK-NOT: onnx.
