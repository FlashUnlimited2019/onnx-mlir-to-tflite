/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_MLIR_ONNX_TO_TFL_LEGALIZE_UTILS_H
#define ONNX_MLIR_ONNX_TO_TFL_LEGALIZE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {

mlir::LogicalResult validateStaticF32Tensor(
    mlir::Operation *op, mlir::Type type, llvm::StringRef role);

mlir::Operation *createTFLOperation(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::StringRef name, mlir::TypeRange resultTypes,
    mlir::ValueRange operands,
    llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

mlir::Value createI32ShapeConstant(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> values);

mlir::Value createF32ScalarTensorConstant(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, float value);

// ONNX image tensors use NCHW. The MVP keeps every non-rank-4 tensor in its
// original order and maps rank-4 f32 tensors to NHWC. For Conv filters this
// same permutation maps ONNX OIHW to the OHWI order required by TFL Conv2D.
mlir::Type convertRank4NCHWToNHWCType(mlir::Type type);

mlir::FailureOr<mlir::DenseElementsAttr> transposeRank4NCHWToNHWC(
    mlir::DenseElementsAttr input);

int64_t mapNCHWAxisToNHWC(int64_t axis);

int64_t normalizeAxis(int64_t axis, int64_t rank);

mlir::NamedAttribute getFusedActivationNone(mlir::Builder &builder);

} // namespace onnx_mlir

#endif
