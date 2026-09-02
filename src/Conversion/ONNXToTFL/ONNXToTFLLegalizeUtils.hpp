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

// Converts a tensor carrying the internal per-value layout encoding to its
// physical TFL type. Unencoded rank-4 FP32 tensors retain the legacy NHWC
// policy, so this helper is also safe in the default mode.
mlir::Type convertONNXToTFLPhysicalType(mlir::Type type);

llvm::SmallVector<int64_t> getONNXToTFLPhysicalToLogicalLayout(
    mlir::RankedTensorType type);

mlir::RankedTensorType stripONNXToTFLLayoutEncoding(
    mlir::RankedTensorType type);

mlir::FailureOr<mlir::Value> convertTFLValueLayout(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::Value value, mlir::RankedTensorType logicalType,
    llvm::ArrayRef<int64_t> sourcePhysicalToLogical,
    llvm::ArrayRef<int64_t> targetPhysicalToLogical);

mlir::FailureOr<mlir::DenseElementsAttr> transposeRank4NCHWToNHWC(
    mlir::DenseElementsAttr input);

int64_t mapNCHWAxisToNHWC(int64_t axis);

int64_t mapONNXToTFLLogicalAxis(
    mlir::RankedTensorType type, int64_t logicalAxis);

llvm::FailureOr<llvm::SmallVector<int64_t>> computeONNXToTFLReducedLayout(
    llvm::ArrayRef<int64_t> inputLayout,
    llvm::ArrayRef<int64_t> reducedLogicalAxes, bool keepDims);

// Computes a zero-copy output layout for a reshape that only inserts or
// removes singleton dimensions. Fails when non-singleton dimensions are
// merged, split, reordered, or otherwise changed.
llvm::FailureOr<llvm::SmallVector<int64_t>>
computeONNXToTFLSingletonReshapeLayout(llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> outputShape,
    llvm::ArrayRef<int64_t> inputPhysicalToLogical);

mlir::FailureOr<mlir::Value> createTFLLayoutAwareReduction(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    llvm::StringRef tflName, mlir::Value input,
    mlir::RankedTensorType sourceInputType,
    mlir::RankedTensorType sourceResultType,
    llvm::ArrayRef<int64_t> logicalAxes, bool keepDims);

int64_t normalizeAxis(int64_t axis, int64_t rank);

mlir::NamedAttribute getFusedActivationNone(mlir::Builder &builder);

} // namespace onnx_mlir

#endif
