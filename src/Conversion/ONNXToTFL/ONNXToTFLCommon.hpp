/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ONNXToTFLCommon.hpp - ONNX to TensorFlow Lite lowering -*- C++ -*-===//

#ifndef ONNX_MLIR_ONNX_TO_TFL_COMMON_H
#define ONNX_MLIR_ONNX_TO_TFL_COMMON_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToTFL/ONNXToTFLLegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

namespace onnx_mlir {

// This dialect reserves the `tfl` namespace in the producer built against
// onnx-mlir's LLVM revision. It intentionally has no ODS operation classes.
// TensorFlow's independently-built parser supplies the authoritative TFL ODS
// definitions and verifies every operation before FlatBuffer export.
class TFLCompatibilityDialect final : public mlir::Dialect {
public:
  explicit TFLCompatibilityDialect(mlir::MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
            mlir::TypeID::get<TFLCompatibilityDialect>()) {
    allowUnknownOperations();
  }

  static llvm::StringRef getDialectNamespace() { return "tfl"; }
};

void populateLoweringONNXElementwiseOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXMatMulGemmOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSoftmaxOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXConcatOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXConstantOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXReshapeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXTransposeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);

} // namespace onnx_mlir

#endif
