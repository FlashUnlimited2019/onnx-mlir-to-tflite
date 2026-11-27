# SPDX-License-Identifier: Apache-2.0

set(check_cmd ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check.py)

set(source_doc_excludes
  .git
  .deps
  .bazel-output
  llvm-project
  tensorflow
  third_party
  docs/_build
  docs-onnx-mlir/doc_check/test
  docs-onnx-mlir/_build
  docs-onnx-mlir/_site
  docs-onnx-mlir/doxygen_html
  )
# Match ignored in-tree build directories, including inactive builds whose
# generated checks may still reference a different machine or toolchain.
file(GLOB source_build_dirs LIST_DIRECTORIES true CONFIGURE_DEPENDS
  "${ONNX_MLIR_SRC_ROOT}/build*"
  "${ONNX_MLIR_SRC_ROOT}/cmake-*"
  )
foreach(build_dir IN LISTS source_build_dirs)
  if (IS_DIRECTORY "${build_dir}")
    list(APPEND source_doc_excludes "${build_dir}")
  endif()
endforeach()
# Check generated documentation separately with the build directory as its root.
if (NOT CMAKE_BINARY_DIR STREQUAL ONNX_MLIR_SRC_ROOT)
  list(APPEND source_doc_excludes "${CMAKE_BINARY_DIR}")
endif()
list(REMOVE_DUPLICATES source_doc_excludes)

add_custom_target(check-doc
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_scanning.py
  COMMAND ${check_cmd}
    ${ONNX_MLIR_SRC_ROOT}
    --exclude_dirs ${source_doc_excludes}
  COMMAND ${check_cmd} ${CMAKE_BINARY_DIR} --exclude_dirs third_party _deps
  VERBATIM
  )

set_target_properties(check-doc PROPERTIES FOLDER "Docs")
# Exclude the target from the default VS build
set_target_properties(check-doc PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
