/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class BatchNormalizationLowering final
    : public OpConversionPattern<ONNXBatchNormalizationInferenceModeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXBatchNormalizationInferenceModeOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || inputType.getRank() < 2 ||
        inputType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "BatchNorm input")) ||
        failed(validateStaticF32Tensor(op, resultType, "BatchNorm result")) ||
        inputType != resultType)
      return op.emitError("ONNXToTFL BatchNormalization requires matching "
                          "static rank-2 through rank-5 FP32 input/result"),
             failure();

    int64_t channels = inputType.getShape()[1];
    auto validateParameter = [&](Value value, StringRef name) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != 1 || type.getShape()[0] != channels ||
          failed(validateStaticF32Tensor(op, type, name)))
        return failure();
      return success();
    };
    if (failed(validateParameter(op.getScale(), "BatchNorm scale")) ||
        failed(validateParameter(op.getB(), "BatchNorm bias")) ||
        failed(validateParameter(op.getMean(), "BatchNorm mean")) ||
        failed(validateParameter(op.getVar(), "BatchNorm variance")))
      return op.emitError("ONNXToTFL BatchNormalization parameters must be "
                          "rank-1 FP32 tensors matching the channel count"),
             failure();

    Location loc = op.getLoc();
    auto parameterType =
        RankedTensorType::get({channels}, rewriter.getF32Type());
    Value epsilon = createF32ScalarTensorConstant(
        rewriter, loc, op.getEpsilon().convertToFloat());
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Value varianceWithEpsilon =
        createTFLOperation(rewriter, loc, "tfl.add", TypeRange{parameterType},
            ValueRange{adaptor.getVar(), epsilon}, fusedNone)
            ->getResult(0);
    Value standardDeviation = createTFLOperation(rewriter, loc, "tfl.sqrt",
        TypeRange{parameterType}, ValueRange{varianceWithEpsilon})
                                  ->getResult(0);
    Value factor =
        createTFLOperation(rewriter, loc, "tfl.div", TypeRange{parameterType},
            ValueRange{adaptor.getScale(), standardDeviation}, fusedNone)
            ->getResult(0);
    Value scaledMean =
        createTFLOperation(rewriter, loc, "tfl.mul", TypeRange{parameterType},
            ValueRange{adaptor.getMean(), factor}, fusedNone)
            ->getResult(0);
    Value offset =
        createTFLOperation(rewriter, loc, "tfl.sub", TypeRange{parameterType},
            ValueRange{adaptor.getB(), scaledMean}, fusedNone)
            ->getResult(0);

    // Rank-2 NC and physical rank-4 NHWC tensors already have channel as the
    // trailing broadcast dimension. Logical rank-3/rank-5 tensors retain
    // channel at axis 1 and need explicit singleton dimensions.
    int64_t rank = inputType.getRank();
    if (rank == 3 || rank == 5) {
      SmallVector<int64_t> broadcastShape(rank, 1);
      broadcastShape[1] = channels;
      auto broadcastType =
          RankedTensorType::get(broadcastShape, rewriter.getF32Type());
      Value shape = createI32ShapeConstant(rewriter, loc, broadcastShape);
      factor = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{broadcastType}, ValueRange{factor, shape})
                   ->getResult(0);
      offset = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{broadcastType}, ValueRange{offset, shape})
                   ->getResult(0);
    }

    Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
    Value scaled = createTFLOperation(rewriter, loc, "tfl.mul",
        TypeRange{physicalResultType}, ValueRange{adaptor.getX(), factor},
        fusedNone)
                       ->getResult(0);
    Value result = createTFLOperation(rewriter, loc, "tfl.add",
        TypeRange{physicalResultType}, ValueRange{scaled, offset}, fusedNone)
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXBatchNormalizationOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<BatchNormalizationLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
