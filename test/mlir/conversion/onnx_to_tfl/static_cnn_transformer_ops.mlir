// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

// ConvTranspose3D is importer-decomposed into UpsampleAndPad followed by
// Conv3D. A fully spatial rank-5 form cannot use the rank-reduction path, so
// lower its statically known zero insertion and padding to ScatterND.
module {
  func.func @full_rank5_upsample_and_pad(%arg0: tensor<1x2x2x2x2xf32>) -> tensor<1x2x5x5x5xf32> {
    %0 = "onnx.UpsampleAndPad"(%arg0) <{pads = [1, 1, 1, 1, 1, 1], strides = [2, 2, 2]}> : (tensor<1x2x2x2x2xf32>) -> tensor<1x2x5x5x5xf32>
    return %0 : tensor<1x2x5x5x5xf32>
  }
  "onnx.EntryPoint"() {func = @full_rank5_upsample_and_pad} : () -> ()
}

// CHECK-LABEL: func.func @main(%arg0: tensor<1x2x2x2x2xf32>) -> tensor<1x2x5x5x5xf32>
// CHECK: %[[INDICES:.*]] = arith.constant dense<
// CHECK-SAME: tensor<16x5xi32>
// CHECK: %[[UPDATES:.*]] = "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x2x2x2x2xf32>, tensor<1xi32>) -> tensor<16xf32>
// CHECK: %[[RESULT:.*]] = "tfl.scatter_nd"(%[[INDICES]], %[[UPDATES]], {{.*}}) : (tensor<16x5xi32>, tensor<16xf32>, tensor<5xi32>) -> tensor<1x2x5x5x5xf32>
// CHECK: return %[[RESULT]] : tensor<1x2x5x5x5xf32>
// CHECK-NOT: onnx.
