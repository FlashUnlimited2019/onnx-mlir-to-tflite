/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class UnsqueezeLowering final : public OpConversionPattern<ONNXUnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXUnsqueezeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool supportedElementType =
        sourceDataType && sourceResultType &&
        (sourceDataType.getElementType().isF32() ||
            sourceDataType.getElementType().isInteger(1) ||
            sourceDataType.getElementType().isSignlessInteger(64)) &&
        sourceResultType.getElementType() == sourceDataType.getElementType();
    if (!sourceDataType || !sourceResultType ||
        !sourceDataType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || !supportedElementType) {
      op.emitError(
          "ONNXToTFL Unsqueeze requires static same-type FP32, bool, or i64 "
          "data and result tensors");
      return failure();
    }

    FailureOr<SmallVector<int64_t>> axes = getConstantIntValues(op.getAxes());
    if (failed(axes) ||
        static_cast<int64_t>(axes->size()) !=
            sourceResultType.getRank() - sourceDataType.getRank()) {
      op.emitError("ONNXToTFL Unsqueeze requires constant axes matching the "
                   "static rank increase");
      return failure();
    }

    SmallVector<bool> insertedAxes(sourceResultType.getRank(), false);
    for (int64_t axis : *axes) {
      if (axis < 0)
        axis += sourceResultType.getRank();
      if (axis < 0 || axis >= sourceResultType.getRank() ||
          insertedAxes[axis]) {
        op.emitError("Unsqueeze axes are invalid or duplicated");
        return failure();
      }
      insertedAxes[axis] = true;
    }

    SmallVector<int64_t> expectedShape;
    expectedShape.reserve(sourceResultType.getRank());
    int64_t sourceAxis = 0;
    for (int64_t resultAxis = 0; resultAxis < sourceResultType.getRank();
         ++resultAxis) {
      if (insertedAxes[resultAxis])
        expectedShape.push_back(1);
      else
        expectedShape.push_back(sourceDataType.getShape()[sourceAxis++]);
    }
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError("Unsqueeze result shape does not match data and axes");
      return failure();
    }

    Value data = adaptor.getData();
    bool isF32 = sourceDataType.getElementType().isF32();
    if (isF32 && sourceDataType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceDataType}, ValueRange{data, permutation})
                 ->getResult(0);
    }

    Value shape = createI32ShapeConstant(
        rewriter, op.getLoc(), sourceResultType.getShape());
    Value logicalResult = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reshape", TypeRange{sourceResultType}, ValueRange{data, shape})
                              ->getResult(0);
    if (!isF32 || sourceResultType.getRank() != 4) {
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

void populateLoweringONNXUnsqueezeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<UnsqueezeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
