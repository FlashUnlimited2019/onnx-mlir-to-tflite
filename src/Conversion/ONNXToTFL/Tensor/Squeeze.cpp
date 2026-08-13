/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class SqueezeLowering final : public OpConversionPattern<ONNXSqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSqueezeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceDataType || !sourceResultType ||
        failed(validateStaticF32Tensor(
            op, adaptor.getData().getType(), "Squeeze data")) ||
        failed(validateStaticF32TensorOrScalar(
            op, sourceResultType, "Squeeze result")))
      return failure();

    int64_t dataRank = sourceDataType.getRank();
    SmallVector<bool> removedAxes(dataRank, false);
    if (isa<NoneType>(op.getAxes().getType())) {
      for (int64_t axis = 0; axis < dataRank; ++axis)
        removedAxes[axis] = sourceDataType.getShape()[axis] == 1;
    } else {
      FailureOr<SmallVector<int64_t>> axes = getConstantIntValues(op.getAxes());
      if (failed(axes))
        return op.emitError("ONNXToTFL Squeeze requires constant axes"),
               failure();
      for (int64_t rawAxis : *axes) {
        int64_t axis = normalizeAxis(rawAxis, dataRank);
        if (axis < 0 || axis >= dataRank || removedAxes[axis] ||
            sourceDataType.getShape()[axis] != 1) {
          op.emitError("Squeeze axes must be unique singleton dimensions");
          return failure();
        }
        removedAxes[axis] = true;
      }
    }

    SmallVector<int64_t> expectedShape;
    for (int64_t axis = 0; axis < dataRank; ++axis)
      if (!removedAxes[axis])
        expectedShape.push_back(sourceDataType.getShape()[axis]);
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError("Squeeze result shape does not match data and axes");
      return failure();
    }

    Value data = adaptor.getData();
    if (dataRank == 4) {
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
    if (sourceResultType.getRank() != 4) {
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

void populateLoweringONNXSqueezeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SqueezeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
