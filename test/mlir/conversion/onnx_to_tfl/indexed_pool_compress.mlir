// SPDX-License-Identifier: Apache-2.0

// RUN: onnx-mlir-opt --convert-onnx-to-tfl --verify-each %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x4xf32>) -> tensor<1x2xf32> {
    %none = "onnx.NoValue"() : () -> none
    %shape = onnx.Constant dense<[1, 2, 2]> : tensor<3xi64>
    %condition = onnx.Constant dense<[true, false]> : tensor<2xi1>
    %axis = onnx.Constant dense<-1> : tensor<i64>
    %values, %indices = "onnx.MaxPool"(%arg0) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2], storage_order = 0 : si64, strides = [2]}> : (tensor<1x1x4xf32>) -> (tensor<1x1x2xf32>, tensor<1x1x2xi64>)
    %unpooled = "onnx.MaxUnpool"(%values, %indices, %none) <{kernel_shape = [2], strides = [2]}> : (tensor<1x1x2xf32>, tensor<1x1x2xi64>, none) -> tensor<1x1x4xf32>
    %matrix = "onnx.Reshape"(%unpooled, %shape) <{allowzero = 0 : si64}> : (tensor<1x1x4xf32>, tensor<3xi64>) -> tensor<1x2x2xf32>
    %triangle = "onnx.Trilu"(%matrix, %none) <{upper = 0 : si64}> : (tensor<1x2x2xf32>, none) -> tensor<1x2x2xf32>
    %compressed = "onnx.Compress"(%triangle, %condition) <{axis = -1 : si64}> : (tensor<1x2x2xf32>, tensor<2xi1>) -> tensor<1x2x1xf32>
    %summed = "onnx.CumSum"(%compressed, %axis) <{exclusive = 0 : si64, reverse = 0 : si64}> : (tensor<1x2x1xf32>, tensor<i64>) -> tensor<1x2x1xf32>
    %output = "onnx.Flatten"(%summed) <{axis = 1 : si64}> : (tensor<1x2x1xf32>) -> tensor<1x2xf32>
    return %output : tensor<1x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK-LABEL: func.func @main
// CHECK: "tfl.strided_slice"
// CHECK: "tfl.scatter_nd"
// CHECK: "tfl.gather"
// CHECK: "tfl.cumsum"
// CHECK-NOT: onnx.
