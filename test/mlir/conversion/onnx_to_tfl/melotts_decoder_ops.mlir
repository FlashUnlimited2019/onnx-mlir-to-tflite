// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// ConvTranspose1D is importer-decomposed into UpsampleAndPad followed by a
// regular Conv1D. Preserve NCL order while inserting zeros along L.
module {
  func.func @upsample_and_pad_1d(%input: tensor<1x2x3xf32>) -> tensor<1x2x8xf32> {
    %result = "onnx.UpsampleAndPad"(%input) {strides = [2], pads = [1, 2]} : (tensor<1x2x3xf32>) -> tensor<1x2x8xf32>
    return %result : tensor<1x2x8xf32>
  }
  "onnx.EntryPoint"() {func = @upsample_and_pad_1d} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x8xf32>
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x3xf32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
// CHECK: "tfl.padv2"({{.*}}) : (tensor<1x2x3x1xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x2x3x2xf32>
// CHECK: "tfl.reshape"({{.*}}) : (tensor<1x2x3x2xf32>, tensor<3xi32>) -> tensor<1x2x6xf32>
// CHECK: "tfl.slice"({{.*}}) : (tensor<1x2x6xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x5xf32>
// CHECK: "tfl.padv2"({{.*}}) : (tensor<1x2x5xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<1x2x8xf32>
// CHECK-NOT: onnx.
