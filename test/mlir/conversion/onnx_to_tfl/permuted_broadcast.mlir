// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=legacy --verify-each %s | FileCheck %s --check-prefix=LEGACY

// The logical batch axis is physically last, and the lower-rank operand also
// has a nonidentity layout. The broadcast needs only a singleton-axis reshape,
// not a data transpose or an expanded tensor.
module {
  func.func @permuted_broadcast(%x: tensor<1x8x7x3xf32>, %b: tensor<1x8x7xf32>) -> tensor<3x1x7x8xf32> {
    %shape = onnx.Constant dense<[1, 8, 7, 3]> : tensor<4xi64>
    %anchor = "onnx.Reshape"(%x, %shape) <{allowzero = 0 : si64}> : (tensor<1x8x7x3xf32>, tensor<4xi64>) -> tensor<1x8x7x3xf32>
    %a = "onnx.Transpose"(%anchor) <{perm = [3, 0, 2, 1]}> : (tensor<1x8x7x3xf32>) -> tensor<3x1x7x8xf32>
    %c = "onnx.Transpose"(%b) <{perm = [0, 2, 1]}> : (tensor<1x8x7xf32>) -> tensor<1x7x8xf32>
    %y = "onnx.Mul"(%a, %c) : (tensor<3x1x7x8xf32>, tensor<1x7x8xf32>) -> tensor<3x1x7x8xf32>
    return %y : tensor<3x1x7x8xf32>
  }
  "onnx.EntryPoint"() {func = @permuted_broadcast} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"(%arg1, {{.*}}) : (tensor<1x8x7xf32>, tensor<3xi32>) -> tensor<8x7x1xf32>
// CHECK: "tfl.mul"{{.*}} : (tensor<1x8x7x3xf32>, tensor<8x7x1xf32>) -> tensor<1x8x7x3xf32>
// CHECK: "tfl.transpose"{{.*}} -> tensor<3x1x7x8xf32>
// CHECK-NOT: onnx.
// LEGACY-LABEL: func.func @main
// LEGACY: "tfl.mul"{{.*}} -> tensor<3x7x8x1xf32>
// LEGACY-NOT: onnx.
