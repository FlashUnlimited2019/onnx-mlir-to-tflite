/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {
namespace {

class WhereLowering final : public OpConversionPattern<ONNXWhereOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXWhereOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto conditionType =
        dyn_cast<RankedTensorType>(op.getCondition().getType());
    auto sourceXType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto sourceYType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool supportedValueType =
        sourceXType && sourceYType && sourceResultType &&
        sourceXType.hasStaticShape() && sourceYType.hasStaticShape() &&
        sourceResultType.hasStaticShape() &&
        (sourceResultType.getElementType().isF32() ||
            sourceResultType.getElementType().isSignlessInteger(64)) &&
        sourceXType.getElementType() == sourceResultType.getElementType() &&
        sourceYType.getElementType() == sourceResultType.getElementType();
    if (!conditionType || !conditionType.hasStaticShape() ||
        !conditionType.getElementType().isInteger(1) || !sourceXType ||
        !sourceYType || !sourceResultType || !supportedValueType) {
      op.emitError("ONNXToTFL Where requires a static boolean condition and "
                   "static same-type FP32 or i64 values/result");
      return failure();
    }

    int64_t resultRank = std::max({conditionType.getRank(),
        sourceXType.getRank(), sourceYType.getRank()});
    if (resultRank > 5 || sourceResultType.getRank() != resultRank) {
      op.emitError("ONNXToTFL Where supports result ranks up to 5");
      return failure();
    }

    SmallVector<int64_t> expectedShape(resultRank, 1);
    auto mergeShape = [&](RankedTensorType type) -> LogicalResult {
      for (int64_t offset = 0; offset < type.getRank(); ++offset) {
        int64_t sourceAxis = type.getRank() - 1 - offset;
        int64_t resultAxis = resultRank - 1 - offset;
        int64_t dimension = type.getShape()[sourceAxis];
        int64_t &expected = expectedShape[resultAxis];
        if (expected != 1 && dimension != 1 && expected != dimension)
          return failure();
        expected = std::max(expected, dimension);
      }
      return success();
    };
    if (failed(mergeShape(conditionType)) || failed(mergeShape(sourceXType)) ||
        failed(mergeShape(sourceYType)) ||
        !llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError("Where operands/result are not broadcast-compatible");
      return failure();
    }

    auto restoreLogicalRank4 = [&](Value value,
                                   RankedTensorType sourceType) -> Value {
      if (!sourceResultType.getElementType().isF32() ||
          sourceType.getRank() != 4)
        return value;
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      return createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{value, permutation})
          ->getResult(0);
    };

    Value x = restoreLogicalRank4(adaptor.getX(), sourceXType);
    Value y = restoreLogicalRank4(adaptor.getY(), sourceYType);
    Value logicalResult = createTFLOperation(rewriter, op.getLoc(),
        "tfl.select_v2", TypeRange{sourceResultType},
        ValueRange{adaptor.getCondition(), x, y})
                              ->getResult(0);
    if (!sourceResultType.getElementType().isF32() ||
        sourceResultType.getRank() != 4) {
      rewriter.replaceOp(op, logicalResult);
      return success();
    }

    Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    Operation *physicalResult = createTFLOperation(rewriter, op.getLoc(),
        "tfl.transpose", TypeRange{physicalResultType},
        ValueRange{logicalResult, permutation});
    rewriter.replaceOp(op, physicalResult->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXWhereOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<WhereLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
