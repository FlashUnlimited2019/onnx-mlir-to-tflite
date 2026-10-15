// RUN: not onnx-mlir-opt --convert-onnx-to-tfl %s 2>&1 | FileCheck %s
// RUN: not onnx-mlir-opt --convert-onnx-to-tfl='layout-mode=experimental' %s 2>&1 | FileCheck %s

module {
  func.func @unbounded_div(%x: tensor<2x3xi64>) -> tensor<2x3xi64> {
    %divisor = onnx.Constant dense<3> : tensor<i64>
    %q = "onnx.Div"(%x, %divisor) : (tensor<2x3xi64>, tensor<i64>) -> tensor<2x3xi64>
    return %q : tensor<2x3xi64>
  }
  "onnx.EntryPoint"() {func = @unbounded_div} : () -> ()
}
// CHECK: i64 Div requires an i32-representable dividend
