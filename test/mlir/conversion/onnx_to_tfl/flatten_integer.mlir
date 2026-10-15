// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' --verify-each --split-input-file %s | FileCheck %s

module {
  func.func @flatten_i64(%x: tensor<1x7x1xi64>) -> tensor<7x1xi64> {
    %out = "onnx.Flatten"(%x) {axis = 2 : si64} : (tensor<1x7x1xi64>) -> tensor<7x1xi64>
    return %out : tensor<7x1xi64>
  }
  "onnx.EntryPoint"() {func = @flatten_i64} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<1x7x1xi64>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: "tfl.cast"
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<1x7x1xi64>, tensor<2xi32>) -> tensor<7x1xi64>
// CHECK-NEXT: return

// -----

module {
  func.func @flatten_i32_rank4(%x: tensor<2x3x4x5xi32>) -> tensor<24x5xi32> {
    %out = "onnx.Flatten"(%x) {axis = -1 : si64} : (tensor<2x3x4x5xi32>) -> tensor<24x5xi32>
    return %out : tensor<24x5xi32>
  }
  "onnx.EntryPoint"() {func = @flatten_i32_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4x5xi32>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: "tfl.cast"
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x4x5xi32>, tensor<2xi32>) -> tensor<24x5xi32>
// CHECK-NEXT: return

// -----

module {
  func.func @flatten_i64_rank4(%x: tensor<2x3x4x5xi64>) -> tensor<1x120xi64> {
    %out = "onnx.Flatten"(%x) {axis = 0 : si64} : (tensor<2x3x4x5xi64>) -> tensor<1x120xi64>
    return %out : tensor<1x120xi64>
  }
  "onnx.EntryPoint"() {func = @flatten_i64_rank4} : () -> ()
}
// CHECK-LABEL: func.func @main(%arg0: tensor<2x3x4x5xi64>
// CHECK-NOT: "tfl.transpose"
// CHECK-NOT: "tfl.cast"
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3x4x5xi64>, tensor<2xi32>) -> tensor<1x120xi64>
// CHECK-NEXT: return
