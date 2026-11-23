// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' --verify-each %s -split-input-file | FileCheck %s

module {
  func.func @clip_scalar(%x: tensor<i64>, %lo: tensor<i64>, %hi: tensor<i64>) -> tensor<i64> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
    return %y : tensor<i64>
  }
  "onnx.EntryPoint"() {func = @clip_scalar} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.maximum"{{.*}} -> tensor<i64>
// CHECK: "tfl.minimum"{{.*}} -> tensor<i64>
// CHECK-NOT: "tfl.cast"
// CHECK: return

// -----

module {
  func.func @scalar_mod(%x: tensor<i64>, %y: tensor<i64>) -> tensor<i64> {
    %r = "onnx.Mod"(%x, %y) {fmod = 0 : si64} : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }
  "onnx.EntryPoint"() {func = @scalar_mod} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.floor_mod"{{.*}} -> tensor<i64>
// CHECK: return

// -----

module {
  func.func @sub_i64(%x: tensor<2x3x4x5xi64>, %y: tensor<i64>) -> tensor<2x3x4x5xi64> {
    %r = "onnx.Sub"(%x, %y) : (tensor<2x3x4x5xi64>, tensor<i64>) -> tensor<2x3x4x5xi64>
    return %r : tensor<2x3x4x5xi64>
  }
  "onnx.EntryPoint"() {func = @sub_i64} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.sub"{{.*}} -> tensor<2x3x4x5xi64>
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @integer_gather(%x: tensor<2x3x4x5xi64>, %idx: tensor<2x3x2x5xi64>) -> tensor<2x3x2x5xi64> {
    %r = "onnx.GatherElements"(%x, %idx) {axis = 2 : si64} : (tensor<2x3x4x5xi64>, tensor<2x3x2x5xi64>) -> tensor<2x3x2x5xi64>
    return %r : tensor<2x3x2x5xi64>
  }
  "onnx.EntryPoint"() {func = @integer_gather} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.gather_nd"{{.*}} -> tensor<60xi64>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x3x2x5xi64>
// CHECK-NOT: "tfl.transpose"
// CHECK: return
