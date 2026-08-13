/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class LRNLowering final : public OpConversionPattern<ONNXLRNOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLRNOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || inputType.getRank() != 4 ||
        resultType != inputType ||
        failed(validateStaticF32Tensor(op, inputType, "LRN input")) ||
        failed(validateStaticF32Tensor(op, resultType, "LRN result"))) {
      op.emitError("ONNXToTFL LRN requires equal static rank-4 FP32 "
                   "input/result tensors");
      return failure();
    }

    int64_t size = op.getSize();
    if (size <= 0 || size % 2 == 0) {
      op.emitError("ONNXToTFL LRN requires a positive odd size");
      return failure();
    }

    float alpha = op.getAlpha().convertToFloat() / static_cast<float>(size);
    float beta = op.getBeta().convertToFloat();
    float bias = op.getBias().convertToFloat();
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("radius",
            rewriter.getI32IntegerAttr(static_cast<int32_t>((size - 1) / 2))),
        rewriter.getNamedAttr("bias", rewriter.getF32FloatAttr(bias)),
        rewriter.getNamedAttr("alpha", rewriter.getF32FloatAttr(alpha)),
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(beta))};
    Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
    Operation *result = createTFLOperation(rewriter, op.getLoc(),
        "tfl.local_response_normalization", TypeRange{physicalResultType},
        ValueRange{adaptor.getX()}, attributes);
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXLRNOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<LRNLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
