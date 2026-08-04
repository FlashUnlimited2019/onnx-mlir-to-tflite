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

int64_t normalizeAxis(int64_t axis, int64_t rank);

mlir::NamedAttribute getFusedActivationNone(mlir::Builder &builder);

} // namespace onnx_mlir

#endif
