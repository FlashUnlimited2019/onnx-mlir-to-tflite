// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' --verify-each %s | FileCheck %s

module {
  func.func @unbounded_div(%x: tensor<2x3xi64>) -> tensor<2x3xi64> {
    %divisor = onnx.Constant dense<3> : tensor<i64>
    %q = "onnx.Div"(%x, %divisor) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
    return %q : tensor<2x3xi64>
  }
  "onnx.EntryPoint"() {func = @unbounded_div} : () -> ()
}
// CHECK-LABEL: func.func @main
// CHECK: "tfl.minimum"
// CHECK: "tfl.div"{{.*}} -> tensor<2x3xf32>
// CHECK: "tfl.mul"{{.*}} -> tensor<2x3xi64>
// CHECK: "tfl.sub"{{.*}} -> tensor<2x3xi64>
// CHECK: "tfl.less_equal"
// CHECK: "tfl.select_v2"{{.*}} -> tensor<2x3xi64>
// CHECK: return
