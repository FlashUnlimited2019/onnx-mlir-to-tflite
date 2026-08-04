# SPDX-License-Identifier: Apache-2.0

set(ONNX_TO_TFLITE_OUTPUT
  "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin/onnx-to-tflite")

add_custom_command(
  OUTPUT "${ONNX_TO_TFLITE_OUTPUT}"
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx-to-tflite.py"
    "${ONNX_TO_TFLITE_OUTPUT}"
  COMMAND chmod +x "${ONNX_TO_TFLITE_OUTPUT}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx-to-tflite.py"
  VERBATIM)

add_custom_target(onnx-to-tflite ALL DEPENDS "${ONNX_TO_TFLITE_OUTPUT}")
