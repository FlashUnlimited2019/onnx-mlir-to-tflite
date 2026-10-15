// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' --verify-each %s | FileCheck %s

module {
  func.func @bounded_div(%x: tensor<2x3xi32>) -> tensor<2x3xi64> {
    %wide = "onnx.Cast"(%x) {to = i64} : (tensor<2x3xi32>) -> tensor<2x3xi64>
    %divisor = onnx.Constant dense<3> : tensor<i64>
    %q = "onnx.Div"(%wide, %divisor) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
    return %q : tensor<2x3xi64>
  }
  "onnx.EntryPoint"() {func = @bounded_div} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.div"{{.*}} -> tensor<2x3xi32>
// CHECK: "tfl.cast"{{.*}} : (tensor<2x3xi32>) -> tensor<2x3xi64>
// CHECK: return
