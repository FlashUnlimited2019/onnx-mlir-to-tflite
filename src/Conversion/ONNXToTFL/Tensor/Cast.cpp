/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class CastLowering final : public OpConversionPattern<ONNXCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool integerToF32 =
        sourceInputType && sourceResultType &&
        (sourceInputType.getElementType().isInteger(1) ||
            sourceInputType.getElementType().isUnsignedInteger(32) ||
            sourceInputType.getElementType().isSignlessInteger(32) ||
            sourceInputType.getElementType().isSignlessInteger(64)) &&
        sourceResultType.getElementType().isF32();
    bool f32ToInteger =
        sourceInputType && sourceResultType &&
        sourceInputType.getElementType().isF32() &&
        (sourceResultType.getElementType().isSignlessInteger(32) ||
            sourceResultType.getElementType().isSignlessInteger(64));
    bool integerToInteger =
        sourceInputType && sourceResultType &&
        (sourceInputType.getElementType().isInteger(1) ||
            sourceInputType.getElementType().isUnsignedInteger(32) ||
            sourceInputType.getElementType().isSignlessInteger(32) ||
            sourceInputType.getElementType().isSignlessInteger(64)) &&
        (sourceResultType.getElementType().isInteger(1) ||
            sourceResultType.getElementType().isUnsignedInteger(32) ||
            sourceResultType.getElementType().isSignlessInteger(32) ||
            sourceResultType.getElementType().isSignlessInteger(64)) &&
        sourceInputType.getElementType() != sourceResultType.getElementType();
    if (!sourceInputType || !sourceResultType ||
        !sourceInputType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() ||
        sourceInputType.getShape() != sourceResultType.getShape() ||
        (!integerToF32 && !f32ToInteger && !integerToInteger)) {
      op.emitError(
          "ONNXToTFL Cast currently supports static bool/i32/i64/uint32 to "
          "f32, "
          "bool/i32/i64/uint32 integer interchange, f32 to i32/i64, with "
          "unchanged shape");
      return failure();
    }

    // FP32 rank-4 inputs are physical NHWC, while integer rank-4 tensors stay
    // in logical NCHW. Restore the input before an f32-to-i64 cast.
    if (f32ToInteger && sourceInputType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
        TypeRange{sourceResultType}, ValueRange{input})
                       ->getResult(0);
    if (!integerToF32 || sourceResultType.getRank() != 4) {
      rewriter.replaceOp(op, result);
      return success();
    }

    // Integer rank-4 inputs are logical NCHW. Enter the physical NHWC
    // representation after converting them to FP32.
    Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    Operation *physicalResult =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{physicalResultType}, ValueRange{result, permutation});
    rewriter.replaceOp(op, physicalResult->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXCastOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CastLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
