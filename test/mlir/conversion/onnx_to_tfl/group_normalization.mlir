// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LEGACY --implicit-check-not=onnx.GroupNormalization --implicit-check-not='tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x}}'
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s --check-prefixes=CHECK,EXPERIMENTAL --implicit-check-not=onnx.GroupNormalization --implicit-check-not='tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x}}'

module {
  func.func @rank4(%x: tensor<2x12x3x5xf32>, %scale: tensor<12xf32>, %bias: tensor<12xf32>) -> tensor<2x12x3x5xf32> {
    %y = "onnx.GroupNormalization"(%x, %scale, %bias) {num_groups = 4 : si64, epsilon = 1.0e-5 : f32, stash_type = 1 : si64} : (tensor<2x12x3x5xf32>, tensor<12xf32>, tensor<12xf32>) -> tensor<2x12x3x5xf32>
    return %y : tensor<2x12x3x5xf32>
  }
  "onnx.EntryPoint"() {func = @rank4} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.mean"{{.*}}tensor<8x45xf32>
// CHECK: "tfl.sub"
// CHECK: "tfl.mul"
// CHECK: "tfl.mean"
// CHECK: "tfl.rsqrt"
// LEGACY: return {{.*}} : tensor<2x3x5x12xf32>
// EXPERIMENTAL: return {{.*}} : tensor<2x12x3x5xf32>

// -----

module {
  func.func @rank5(%x: tensor<2x12x2x3x4xf32>, %scale: tensor<12xf32>, %bias: tensor<12xf32>) -> tensor<2x12x2x3x4xf32> {
    %y = "onnx.GroupNormalization"(%x, %scale, %bias) {num_groups = 3 : si64, epsilon = 1.0e-3 : f32, stash_type = 1 : si64} : (tensor<2x12x2x3x4xf32>, tensor<12xf32>, tensor<12xf32>) -> tensor<2x12x2x3x4xf32>
    return %y : tensor<2x12x2x3x4xf32>
  }
  "onnx.EntryPoint"() {func = @rank5} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.mean"{{.*}}tensor<6x96xf32>
// CHECK: "tfl.rsqrt"
// CHECK: return {{.*}} : tensor<2x12x2x3x4xf32>

// -----

module {
  func.func @rank2(%x: tensor<2x6xf32>, %scale: tensor<6xf32>, %bias: tensor<6xf32>) -> tensor<2x6xf32> {
    %y = "onnx.GroupNormalization"(%x, %scale, %bias) {num_groups = 3 : si64, epsilon = 1.0e-5 : f32, stash_type = 1 : si64} : (tensor<2x6xf32>, tensor<6xf32>, tensor<6xf32>) -> tensor<2x6xf32>
    return %y : tensor<2x6xf32>
  }
  "onnx.EntryPoint"() {func = @rank2} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.mean"{{.*}}tensor<6x2xf32>
// CHECK: "tfl.rsqrt"
// CHECK: return {{.*}} : tensor<2x6xf32>
