/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class SpaceToDepthLowering final
    : public OpConversionPattern<ONNXSpaceToDepthOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSpaceToDepthOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType ||
        failed(validateStaticF32Tensor(op, inputType, "SpaceToDepth input")) ||
        failed(
            validateStaticF32Tensor(op, outputType, "SpaceToDepth output")) ||
        inputType.getRank() != 4 || outputType.getRank() != 4)
      return op.emitError(
                 "ONNXToTFL SpaceToDepth requires static rank-4 FP32 tensors"),
             failure();

    int64_t blockSize = op.getBlocksize();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (blockSize <= 0 || inputShape[2] % blockSize != 0 ||
        inputShape[3] % blockSize != 0 ||
        outputShape != ArrayRef<int64_t>{inputShape[0],
                           inputShape[1] * blockSize * blockSize,
                           inputShape[2] / blockSize,
                           inputShape[3] / blockSize})
      return op.emitError("invalid SpaceToDepth block size or result shape"),
             failure();

    // Both ONNX SpaceToDepth and the TFLite NHWC builtin flatten channels in
    // [block-height, block-width, input-channel] (DCR) order. The bridge's
    // physical NHWC representation can therefore use the builtin directly.
    Type physicalOutputType = convertRank4NCHWToNHWCType(outputType);
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "block_size", rewriter.getI32IntegerAttr(blockSize))};
    Operation *spaceToDepth = createTFLOperation(rewriter, op.getLoc(),
        "tfl.space_to_depth", TypeRange{physicalOutputType},
        ValueRange{adaptor.getInput()}, attributes);
    rewriter.replaceOp(op, spaceToDepth->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXSpaceToDepthOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SpaceToDepthLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
