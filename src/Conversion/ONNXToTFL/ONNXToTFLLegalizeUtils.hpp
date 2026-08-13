/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#ifndef ONNX_MLIR_ONNX_TO_TFL_LEGALIZE_UTILS_H
#define ONNX_MLIR_ONNX_TO_TFL_LEGALIZE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnx_mlir {

mlir::LogicalResult validateStaticF32Tensor(
    mlir::Operation *op, mlir::Type type, llvm::StringRef role);

mlir::LogicalResult validateStaticF32TensorOrScalar(
    mlir::Operation *op, mlir::Type type, llvm::StringRef role);

mlir::Operation *createTFLOperation(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::StringRef name, mlir::TypeRange resultTypes,
    mlir::ValueRange operands,
    llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

mlir::Value createI32ShapeConstant(mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, llvm::ArrayRef<int64_t> values);

mlir::Value createI32ScalarTensorConstant(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    int32_t value);

mlir::Value createF32ScalarTensorConstant(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc, float value);

// Reads a one-dimensional integer tensor from either an ONNX Constant or an
// arith.constant. This accepts both i32 and i64 so conversion patterns do not
// depend on the order in which constants and their users are rewritten.
mlir::FailureOr<llvm::SmallVector<int64_t>> getConstantIntValues(
    mlir::Value value);

// Reads a dense FP32 tensor from either an ONNX Constant or arith.constant.
mlir::FailureOr<llvm::SmallVector<float>> getConstantF32Values(
    mlir::Value value);

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
