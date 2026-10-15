// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl --verify-each %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --allow-unregistered-dialect --convert-onnx-to-tfl=layout-mode=experimental --verify-each %s -split-input-file | FileCheck %s

// A shared first transpose stays alive, but the inverse branch bypasses it.
module {
  func.func @shared_inverse(%x: tensor<2x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<4x2x3xf32>) {
    %p = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %q = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<4x2x3xf32>
    %b = "tfl.transpose"(%a, %q) : (tensor<4x2x3xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
    return %b, %a : tensor<2x3x4xf32>, tensor<4x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @shared_inverse} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[T:.*]] = "tfl.transpose"(%arg0,
// CHECK-NOT: "tfl.transpose"
// CHECK: return %arg0, %[[T]]

// -----

module {
  func.func @tile_shared_input(%x: tensor<2x3x4xf32>) -> (tensor<2x9x8xf32>, tensor<4x2x3xf32>) {
    %p = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %q = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %r = arith.constant dense<[2, 1, 3]> : tensor<3xi64>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<4x2x3xf32>
    %b = "tfl.tile"(%a, %r) : (tensor<4x2x3xf32>, tensor<3xi64>) -> tensor<8x2x9xf32>
    %c = "tfl.transpose"(%b, %q) : (tensor<8x2x9xf32>, tensor<3xi32>) -> tensor<2x9x8xf32>
    return %c, %a : tensor<2x9x8xf32>, tensor<4x2x3xf32>
  }
  "onnx.EntryPoint"() {func = @tile_shared_input} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[T:.*]] = "tfl.transpose"(%arg0,
// CHECK: %[[R:.*]] = arith.constant dense<[1, 3, 2]> : tensor<3xi32>
// CHECK: %[[Y:.*]] = "tfl.tile"(%arg0, %[[R]])
// CHECK-NOT: "tfl.transpose"
// CHECK: return %[[Y]], %[[T]]

// -----

// Never duplicate a Tile whose untransposed result is also observable.
module {
  func.func @tile_shared_output(%x: tensor<2x3x4xi32>) -> (tensor<2x9x8xi32>, tensor<8x2x9xi32>) {
    %p = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %q = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %r = arith.constant dense<[2, 1, 3]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3x4xi32>, tensor<3xi32>) -> tensor<4x2x3xi32>
    %b = "tfl.tile"(%a, %r) : (tensor<4x2x3xi32>, tensor<3xi32>) -> tensor<8x2x9xi32>
    %c = "tfl.transpose"(%b, %q) : (tensor<8x2x9xi32>, tensor<3xi32>) -> tensor<2x9x8xi32>
    return %c, %b : tensor<2x9x8xi32>, tensor<8x2x9xi32>
  }
  "onnx.EntryPoint"() {func = @tile_shared_output} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.tile"
// CHECK: "tfl.transpose"
// CHECK: return

// -----

// Equal shapes are insufficient when the permutations are not inverses.
module {
  func.func @not_inverse(%x: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
    %p = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
    %b = "tfl.transpose"(%a, %p) : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
    return %b : tensor<2x2x2xf32>
  }
  "onnx.EntryPoint"() {func = @not_inverse} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[P:.*]] = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
// CHECK: "tfl.transpose"(%arg0, %[[P]])
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// The compatibility dialect does not verify parameter values. Cleanup must
// not erase malformed permutations before the authoritative consumer sees them.
module {
  func.func @invalid_permutation(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %p = arith.constant dense<[0, 0]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    %b = "tfl.transpose"(%a, %p) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    return %b : tensor<2x2xf32>
  }
  "onnx.EntryPoint"() {func = @invalid_permutation} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @runtime_repeats(%x: tensor<2x3xf32>, %r: tensor<2xi32>) -> tensor<2x6xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.tile"(%a, %r) : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<6x2xf32>
    %c = "tfl.transpose"(%b, %p) : (tensor<6x2xf32>, tensor<2xi32>) -> tensor<2x6xf32>
    return %c : tensor<2x6xf32>
  }
  "onnx.EntryPoint"() {func = @runtime_repeats} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.tile"
// CHECK: "tfl.transpose"
// CHECK: return

// -----

module {
  func.func @slice_inverse(%x: tensor<2x3x4xf32>) -> tensor<2x1x3xf32> {
    %p = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %q = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %begin = arith.constant dense<[1, 0, 1]> : tensor<3xi64>
    %size = arith.constant dense<[-1, 2, 1]> : tensor<3xi64>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<4x2x3xf32>
    %b = "tfl.slice"(%a, %begin, %size) : (tensor<4x2x3xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x2x1xf32>
    %c = "tfl.transpose"(%b, %q) : (tensor<3x2x1xf32>, tensor<3xi32>) -> tensor<2x1x3xf32>
    return %c : tensor<2x1x3xf32>
  }
  "onnx.EntryPoint"() {func = @slice_inverse} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: %[[B:.*]] = arith.constant dense<[0, 1, 1]> : tensor<3xi32>
// CHECK: %[[S:.*]] = arith.constant dense<[2, 1, 3]> : tensor<3xi32>
// CHECK: "tfl.slice"(%arg0, %[[B]], %[[S]])
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// -1 refers to the tail of the inner window, not of the original tensor.
module {
  func.func @compose_shared_slices(%x: tensor<8x10xf32>) -> (tensor<3x5xf32>, tensor<5x6xf32>) {
    %b0 = arith.constant dense<[1, 2]> : tensor<2xi32>
    %s0 = arith.constant dense<[5, 6]> : tensor<2xi32>
    %b1 = arith.constant dense<[2, 1]> : tensor<2xi32>
    %s1 = arith.constant dense<-1> : tensor<2xi32>
    %a = "tfl.slice"(%x, %b0, %s0) : (tensor<8x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5x6xf32>
    %b = "tfl.slice"(%a, %b1, %s1) : (tensor<5x6xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xf32>
    return %b, %a : tensor<3x5xf32>, tensor<5x6xf32>
  }
  "onnx.EntryPoint"() {func = @compose_shared_slices} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: %[[A:.*]] = "tfl.slice"(%arg0,
// CHECK: %[[B:.*]] = arith.constant dense<3> : tensor<2xi32>
// CHECK: %[[S:.*]] = arith.constant dense<[3, 5]> : tensor<2xi32>
// CHECK: %[[Y:.*]] = "tfl.slice"(%arg0, %[[B]], %[[S]])
// CHECK: return %[[Y]], %[[A]]

// -----

module {
  func.func @singleton_cancel(%x: tensor<2x3xf32>) -> tensor<2x3x1xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %shape = arith.constant dense<[3, 1, 2]> : tensor<3xi32>
    %q = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    %b = "tfl.reshape"(%a, %shape) : (tensor<3x2xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
    %c = "tfl.transpose"(%b, %q) : (tensor<3x1x2xf32>, tensor<3xi32>) -> tensor<2x3x1xf32>
    return %c : tensor<2x3x1xf32>
  }
  "onnx.EntryPoint"() {func = @singleton_cancel} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK-NOT: "tfl.transpose"
// CHECK: "tfl.reshape"(%arg0, {{.*}}) : (tensor<2x3xf32>, tensor<3xi32>) -> tensor<2x3x1xf32>
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// Equal extents do not make two distinct axes interchangeable.
module {
  func.func @equal_extent_axes(%x: tensor<2x2xf32>) -> tensor<2x2x1xf32> {
    %p = arith.constant dense<[1, 0]> : tensor<2xi32>
    %shape = arith.constant dense<[1, 2, 2]> : tensor<3xi32>
    %q = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    %b = "tfl.reshape"(%a, %shape) : (tensor<2x2xf32>, tensor<3xi32>) -> tensor<1x2x2xf32>
    %c = "tfl.transpose"(%b, %q) : (tensor<1x2x2xf32>, tensor<3xi32>) -> tensor<2x2x1xf32>
    return %c : tensor<2x2x1xf32>
  }
  "onnx.EntryPoint"() {func = @equal_extent_axes} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"(%arg0,
// CHECK: "tfl.reshape"
// CHECK-NOT: "tfl.transpose"
// CHECK: return

// -----

// Splitting or merging non-singleton dimensions is deliberately not handled.
module {
  func.func @nontrivial_reshape(%x: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
    %p = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
    %shape = arith.constant dense<[4, 2, 3]> : tensor<3xi32>
    %a = "tfl.transpose"(%x, %p) : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<2x4x3xf32>
    %b = "tfl.reshape"(%a, %shape) : (tensor<2x4x3xf32>, tensor<3xi32>) -> tensor<4x2x3xf32>
    %c = "tfl.transpose"(%b, %p) : (tensor<4x2x3xf32>, tensor<3xi32>) -> tensor<4x3x2xf32>
    return %c : tensor<4x3x2xf32>
  }
  "onnx.EntryPoint"() {func = @nontrivial_reshape} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.transpose"
// CHECK: "tfl.reshape"
// CHECK: "tfl.transpose"
// CHECK: return
