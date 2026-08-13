# SPDX-License-Identifier: Apache-2.0
# Modified by FlashUnlimited2019 in 2026.

add_subdirectory(ONNXConversionCommon)
add_subdirectory(ONNXToKrnl)
add_subdirectory(KrnlToLLVM)
add_subdirectory(KrnlToAffine)
add_subdirectory(KrnlSeqToMemref)
add_subdirectory(ONNXToTOSA)
add_subdirectory(ONNXToTFL)
add_subdirectory(ONNXToLinalg)

if (ONNX_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(ONNXToStablehlo)
endif()
