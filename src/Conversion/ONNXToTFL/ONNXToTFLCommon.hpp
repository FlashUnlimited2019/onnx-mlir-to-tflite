/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

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

void populateONNXToTFLPreprocessingPatterns(mlir::RewritePatternSet &);

void populateLoweringONNXElementwiseOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXAdditionalElementwiseOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXAdditionalMathOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXUncommonMathOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXArgMaxOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXComparisonOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXLegacyMathOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXMatMulGemmOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSoftmaxOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXTopKOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXConvOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXGridSampleOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXStaticSamplingOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXMaxPoolOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXIndexedMaxPoolOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXAveragePoolOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXAttentionOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXBatchNormalizationOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXLayerNormalizationOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXLRNOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXLpPoolOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXLSTMOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXStaticRecurrentOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSTFTOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXReduceMeanOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXReduceMaxOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXReduceSumOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXConcatOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXConstantOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXCastOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXFlattenOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXExpandOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXGatherOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXGatherElementsOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXScatterElementsOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXScatterNDOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXGatherNDOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXReshapeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXTransposeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXUnsqueezeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSliceOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSplitOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSequenceOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSqueezeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXStaticUncommonTensorOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXTileOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXResizeOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXDepthToSpaceOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXSpaceToDepthOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXPadOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXUpsampleAndPadOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXWhereOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXCompressTriluOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);
void populateLoweringONNXIm2ColOpToTFLPatterns(
    mlir::RewritePatternSet &, mlir::TypeConverter &);

} // namespace onnx_mlir

#endif
