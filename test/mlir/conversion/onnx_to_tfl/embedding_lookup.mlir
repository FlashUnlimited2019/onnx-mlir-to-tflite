// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each --split-input-file %s | FileCheck %s --check-prefixes=CHECK,LEGACY
// RUN: onnx-mlir-opt --convert-onnx-to-tfl=layout-mode=experimental --verify-each --split-input-file %s | FileCheck %s --check-prefixes=CHECK,EXPERIMENTAL

// Inclusive width threshold, negative axis, and bounded narrowing. Constants
// are intentionally splats so shape/ownership checks do not inspect weights.
module {
  func.func @wide_table(%ids: tensor<2x3xi64>) -> tensor<2x3x128xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x128xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = -2 : si64} : (tensor<7x128xf32>, tensor<2x3xi64>) -> tensor<2x3x128xf32>
    return %result : tensor<2x3x128xf32>
  }
  "onnx.EntryPoint"() {func = @wide_table} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.reshape"{{.*}} -> tensor<6xi64>
// CHECK: arith.constant dense<7> : tensor<i64>
// CHECK: "tfl.minimum"{{.*}} -> tensor<6xi64>
// CHECK: arith.constant dense<-8> : tensor<i64>
// CHECK: "tfl.maximum"{{.*}} -> tensor<6xi64>
// CHECK: "tfl.cast"{{.*}} -> tensor<6xi32>
// CHECK: "tfl.less"{{.*}} -> tensor<6xi1>
// CHECK: "tfl.minimum"{{.*}} -> tensor<6xi32>
// CHECK: "tfl.add"{{.*}} -> tensor<6xi32>
// CHECK: "tfl.select_v2"{{.*}} -> tensor<6xi32>
// CHECK: "tfl.embedding_lookup"{{.*}} : (tensor<6xi32>, tensor<7x128xf32>) -> tensor<6x128xf32>
// CHECK: "tfl.reshape"{{.*}} -> tensor<2x3x128xf32>
// CHECK-NOT: "tfl.gather"

// -----

module {
  func.func @rank_four_result(%ids: tensor<2x3x4xi32>) -> tensor<2x3x4x129xf32> {
    %table = onnx.Constant dense<2.0> : tensor<7x129xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x129xf32>, tensor<2x3x4xi32>) -> tensor<2x3x4x129xf32>
    return %result : tensor<2x3x4x129xf32>
  }
  "onnx.EntryPoint"() {func = @rank_four_result} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.embedding_lookup"{{.*}} : (tensor<24xi32>, tensor<7x129xf32>) -> tensor<24x129xf32>
// LEGACY: return {{.*}} : tensor<2x4x129x3xf32>
// EXPERIMENTAL: return {{.*}} : tensor<2x3x4x129xf32>
// CHECK-NOT: "tfl.gather"

// -----

module {
  func.func @scalar_index(%ids: tensor<i32>) -> tensor<128xf32> {
    %table = onnx.Constant dense<2.0> : tensor<7x128xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x128xf32>, tensor<i32>) -> tensor<128xf32>
    return %result : tensor<128xf32>
  }
  "onnx.EntryPoint"() {func = @scalar_index} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.embedding_lookup"{{.*}} : (tensor<1xi32>, tensor<7x128xf32>) -> tensor<1x128xf32>
// CHECK: return {{.*}} : tensor<128xf32>

// -----

module {
  func.func @narrow_table(%ids: tensor<2xi32>) -> tensor<2x127xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x127xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x127xf32>, tensor<2xi32>) -> tensor<2x127xf32>
    return %result : tensor<2x127xf32>
  }
  "onnx.EntryPoint"() {func = @narrow_table} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.embedding_lookup"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.embedding_lookup"

// -----

module {
  func.func @nonnegative_indices(%ids: tensor<2xi64>) -> tensor<2x128xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x128xf32>
    %zero = onnx.Constant dense<0> : tensor<i64>
    %bounded = "onnx.Max"(%zero, %ids) : (tensor<i64>, tensor<2xi64>) -> tensor<2xi64>
    %result = "onnx.Gather"(%table, %bounded) {axis = 0 : si64} : (tensor<7x128xf32>, tensor<2xi64>) -> tensor<2x128xf32>
    return %result : tensor<2x128xf32>
  }
  "onnx.EntryPoint"() {func = @nonnegative_indices} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.maximum"
// CHECK: "tfl.minimum"{{.*}} -> tensor<2xi64>
// CHECK: "tfl.cast"{{.*}} -> tensor<2xi32>
// CHECK-NOT: "tfl.less"
// CHECK-NOT: "tfl.add"
// CHECK-NOT: "tfl.select_v2"
// CHECK: "tfl.embedding_lookup"
// CHECK: return

// -----

module {
  func.func @negative_bound(%ids: tensor<2xi32>) -> tensor<2x128xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x128xf32>
    %lower = onnx.Constant dense<-2> : tensor<i32>
    %bounded = "onnx.Max"(%ids, %lower) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
    %result = "onnx.Gather"(%table, %bounded) {axis = 0 : si64} : (tensor<7x128xf32>, tensor<2xi32>) -> tensor<2x128xf32>
    return %result : tensor<2x128xf32>
  }
  "onnx.EntryPoint"() {func = @negative_bound} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.maximum"
// CHECK: "tfl.less"
// CHECK: "tfl.add"
// CHECK: "tfl.select_v2"
// CHECK: "tfl.embedding_lookup"

// -----

module {
  func.func @runtime_table(%table: tensor<7x512xf32>, %ids: tensor<2xi32>) -> tensor<2x512xf32> {
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x512xf32>, tensor<2xi32>) -> tensor<2x512xf32>
    return %result : tensor<2x512xf32>
  }
  "onnx.EntryPoint"() {func = @runtime_table} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.embedding_lookup"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.embedding_lookup"

// -----

module {
  func.func @nonzero_axis(%ids: tensor<2xi32>) -> tensor<7x2xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x128xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = 1 : si64} : (tensor<7x128xf32>, tensor<2xi32>) -> tensor<7x2xf32>
    return %result : tensor<7x2xf32>
  }
  "onnx.EntryPoint"() {func = @nonzero_axis} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.embedding_lookup"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.embedding_lookup"

// -----

module {
  func.func @integer_table(%ids: tensor<2xi32>) -> tensor<2x128xi64> {
    %table = onnx.Constant dense<16777217> : tensor<7x128xi64>
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x128xi64>, tensor<2xi32>) -> tensor<2x128xi64>
    return %result : tensor<2x128xi64>
  }
  "onnx.EntryPoint"() {func = @integer_table} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.embedding_lookup"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.embedding_lookup"

// -----

module {
  func.func @empty_indices(%ids: tensor<0xi32>) -> tensor<0x128xf32> {
    %table = onnx.Constant dense<1.0> : tensor<7x128xf32>
    %result = "onnx.Gather"(%table, %ids) {axis = 0 : si64} : (tensor<7x128xf32>, tensor<0xi32>) -> tensor<0x128xf32>
    return %result : tensor<0x128xf32>
  }
  "onnx.EntryPoint"() {func = @empty_indices} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.embedding_lookup"
// CHECK: "tfl.gather"
// CHECK-NOT: "tfl.embedding_lookup"
