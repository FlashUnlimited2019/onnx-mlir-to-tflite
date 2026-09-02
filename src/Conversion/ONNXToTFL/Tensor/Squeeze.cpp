/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <numeric>

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
    bool supported =
        sourceDataType && sourceResultType && sourceDataType.hasStaticShape() &&
        sourceResultType.hasStaticShape() &&
        sourceDataType.getElementType() == sourceResultType.getElementType() &&
        (sourceDataType.getElementType().isF32() ||
            sourceDataType.getElementType().isInteger(1));
    if (!supported)
      return op.emitError("ONNXToTFL Squeeze requires matching static FP32 "
                          "or boolean data/result tensors"),
             failure();

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
    bool hasPerValueLayout =
        isa_and_nonnull<DenseI64ArrayAttr>(sourceDataType.getEncoding()) ||
        isa_and_nonnull<DenseI64ArrayAttr>(sourceResultType.getEncoding());
    if (hasPerValueLayout) {
      auto logicalDataType = stripONNXToTFLLayoutEncoding(sourceDataType);
      auto logicalResultType = stripONNXToTFLLayoutEncoding(sourceResultType);
      SmallVector<int64_t> dataLayout =
          getONNXToTFLPhysicalToLogicalLayout(sourceDataType);
      SmallVector<int64_t> resultLayout =
          getONNXToTFLPhysicalToLogicalLayout(sourceResultType);
      FailureOr<SmallVector<int64_t>> zeroCopyResultLayout =
          computeONNXToTFLSingletonReshapeLayout(logicalDataType.getShape(),
              logicalResultType.getShape(), dataLayout);
      if (succeeded(zeroCopyResultLayout) &&
          *zeroCopyResultLayout == resultLayout) {
        auto physicalResultType = cast<RankedTensorType>(
            this->getTypeConverter()->convertType(sourceResultType));
        Value physicalShape = createI32ShapeConstant(
            rewriter, op.getLoc(), physicalResultType.getShape());
        Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{physicalResultType}, ValueRange{data, physicalShape})
                           ->getResult(0);
        rewriter.replaceOp(op, result);
        return success();
      }

      SmallVector<int64_t> dataIdentity(sourceDataType.getRank());
      SmallVector<int64_t> resultIdentity(sourceResultType.getRank());
      std::iota(dataIdentity.begin(), dataIdentity.end(), 0);
      std::iota(resultIdentity.begin(), resultIdentity.end(), 0);
      FailureOr<Value> logicalData = convertTFLValueLayout(rewriter,
          op.getLoc(), data, logicalDataType, dataLayout, dataIdentity);
      if (failed(logicalData))
        return failure();
      Value logicalShape = createI32ShapeConstant(
          rewriter, op.getLoc(), logicalResultType.getShape());
      Value logicalResult = createTFLOperation(rewriter, op.getLoc(),
          "tfl.reshape", TypeRange{logicalResultType},
          ValueRange{*logicalData, logicalShape})
                                ->getResult(0);
      FailureOr<Value> physicalResult =
          convertTFLValueLayout(rewriter, op.getLoc(), logicalResult,
              logicalResultType, resultIdentity, resultLayout);
      if (failed(physicalResult) ||
          (*physicalResult).getType() !=
              this->getTypeConverter()->convertType(sourceResultType))
        return failure();
      rewriter.replaceOp(op, *physicalResult);
      return success();
    }

    bool isF32 = sourceDataType.getElementType().isF32();
    if (isF32 && dataRank == 4) {
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

void populateLoweringONNXSqueezeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SqueezeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
