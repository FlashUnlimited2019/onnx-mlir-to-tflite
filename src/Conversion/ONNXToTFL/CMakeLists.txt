# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXToTFL
  ConvertONNXToTFL.cpp
  ONNXToTFLLegalizeUtils.cpp

  Math/Elementwise.cpp
  Math/MatMulGemm.cpp
  Math/ReduceMean.cpp
  Math/Softmax.cpp
  NN/Conv.cpp
  NN/MaxPool.cpp
  Tensor/Concat.cpp
  Tensor/Constant.cpp
  Tensor/Reshape.cpp
  Tensor/Transpose.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRTransforms
  )
