// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @upsample_and_pad(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x3x5x7xf32> {
    %0 = "onnx.UpsampleAndPad"(%arg0) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<1x3x2x3xf32>) -> tensor<1x3x5x7xf32>
    return %0 : tensor<1x3x5x7xf32>
  }
  "onnx.EntryPoint"() {func = @upsample_and_pad} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x3x3xf32>
// CHECK: "tfl.reshape"
// CHECK: "tfl.padv2"
// CHECK: "tfl.reshape"
// CHECK: "tfl.reshape"
// CHECK: "tfl.padv2"
// CHECK: "tfl.reshape"
// CHECK: "tfl.slice"
// CHECK: "tfl.padv2"
// CHECK: return {{.*}} : tensor<1x5x7x3xf32>
