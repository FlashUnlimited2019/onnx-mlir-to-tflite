/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class CeilLowering final : public OpConversionPattern<ONNXCeilOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXCeilOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType ||
        failed(validateStaticF32TensorOrScalar(op, inputType, "Ceil input")) ||
        failed(validateStaticF32TensorOrScalar(op, resultType, "Ceil result")) ||
        inputType != resultType)
      return op.emitError("ONNXToTFL Ceil requires matching static f32 "
                          "input/result tensors"),
             failure();

    Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.ceil",
        TypeRange{physicalResultType}, ValueRange{adaptor.getX()})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class NotLowering final : public OpConversionPattern<ONNXNotOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXNotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        !inputType.getElementType().isInteger(1) || inputType != resultType)
      return op.emitError("ONNXToTFL Not requires matching static boolean "
                          "input/result tensors"),
             failure();

    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.logical_not",
        TypeRange{resultType}, ValueRange{adaptor.getX()})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class HardmaxLowering final : public OpConversionPattern<ONNXHardmaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXHardmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || inputType.getRank() < 1 ||
        inputType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "Hardmax input")) ||
        failed(validateStaticF32Tensor(op, resultType, "Hardmax result")) ||
        inputType != resultType)
      return op.emitError("ONNXToTFL Hardmax requires matching static rank-1 "
                          "through rank-5 f32 input/result tensors"),
             failure();

    int64_t rank = inputType.getRank();
    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError("Hardmax axis is out of range"), failure();
    int64_t depth = inputType.getShape()[axis];
    if (depth <= 0 || depth > std::numeric_limits<int32_t>::max())
      return op.emitError("Hardmax depth is outside the supported static "
                          "int32 range"),
             failure();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    if (rank == 4) {
      Value permutation = createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2});
      input = createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{inputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    SmallVector<int64_t> indicesShape(inputType.getShape());
    indicesShape.erase(indicesShape.begin() + axis);
    auto indicesType =
        RankedTensorType::get(indicesShape, rewriter.getI32Type());
    Value axisValue = createI32ScalarTensorConstant(
        rewriter, loc, static_cast<int32_t>(axis));
    Value indices = createTFLOperation(rewriter, loc, "tfl.arg_max",
        TypeRange{indicesType}, ValueRange{input, axisValue})
                        ->getResult(0);
    Value depthValue = createI32ScalarTensorConstant(
        rewriter, loc, static_cast<int32_t>(depth));
    Value one = createF32ScalarTensorConstant(rewriter, loc, 1.0f);
    Value zero = createF32ScalarTensorConstant(rewriter, loc, 0.0f);
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "axis", rewriter.getI32IntegerAttr(static_cast<int32_t>(axis)))};
    Value result = createTFLOperation(rewriter, loc, "tfl.one_hot",
        TypeRange{resultType},
        ValueRange{indices, depthValue, one, zero}, attributes)
                       ->getResult(0);
    if (rank == 4) {
      Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
      Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
      result = createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXLegacyMathOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CeilLowering, NotLowering, HardmaxLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
