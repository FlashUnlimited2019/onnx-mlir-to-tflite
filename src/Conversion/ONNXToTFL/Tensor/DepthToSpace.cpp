/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class DepthToSpaceLowering final
    : public OpConversionPattern<ONNXDepthToSpaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXDepthToSpaceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType ||
        failed(validateStaticF32Tensor(op, inputType, "DepthToSpace input")) ||
        failed(
            validateStaticF32Tensor(op, outputType, "DepthToSpace output")) ||
        inputType.getRank() != 4 || outputType.getRank() != 4)
      return op.emitError(
                 "ONNXToTFL DepthToSpace requires static rank-4 FP32 tensors"),
             failure();

    int64_t blockSize = op.getBlocksize();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (blockSize <= 0 || inputShape[1] % (blockSize * blockSize) != 0 ||
        outputShape != ArrayRef<int64_t>{inputShape[0],
                           inputShape[1] / (blockSize * blockSize),
                           inputShape[2] * blockSize,
                           inputShape[3] * blockSize})
      return op.emitError("invalid DepthToSpace block size or result shape"),
             failure();

    Type physicalOutputType = convertRank4NCHWToNHWCType(outputType);
    if (op.getMode() == "DCR") {
      SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
          "block_size", rewriter.getI32IntegerAttr(blockSize))};
      Operation *depthToSpace = createTFLOperation(rewriter, op.getLoc(),
          "tfl.depth_to_space", TypeRange{physicalOutputType},
          ValueRange{adaptor.getInput()}, attributes);
      rewriter.replaceOp(op, depthToSpace->getResults());
      return success();
    }
    if (op.getMode() != "CRD")
      return op.emitError()
                 << "unsupported DepthToSpace mode: " << op.getMode(),
             failure();

    // TFLite's builtin uses ONNX DCR ordering. Convert CRD's flattened channel
    // order [C,r*r] to [r*r,C], then use the builtin. Collapse N/H/W while
    // reordering so the temporary tensors stay rank 3 rather than rank 6:
    // [N,H,W,C*r*r] -> [N*H*W,C,r*r] -> [N*H*W,r*r,C]
    //                   -> [N,H,W,r*r*C] -> DepthToSpace(DCR).
    int64_t outputChannels = outputShape[1];
    int64_t positions = inputShape[0] * inputShape[2] * inputShape[3];
    int64_t blockElements = blockSize * blockSize;
    auto reshapeType = RankedTensorType::get(
        {positions, outputChannels, blockElements}, rewriter.getF32Type());
    Value reshapeShape =
        createI32ShapeConstant(rewriter, op.getLoc(), reshapeType.getShape());
    Value reshaped = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{reshapeType}, ValueRange{adaptor.getInput(), reshapeShape})
                         ->getResult(0);
    auto transposeType = RankedTensorType::get(
        {positions, blockElements, outputChannels}, rewriter.getF32Type());
    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 1});
    Value transposed =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{transposeType}, ValueRange{reshaped, permutation})
            ->getResult(0);
    auto physicalInputType =
        cast<RankedTensorType>(adaptor.getInput().getType());
    Value inputShapeValue = createI32ShapeConstant(
        rewriter, op.getLoc(), physicalInputType.getShape());
    Value reorderedInput = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reshape", TypeRange{physicalInputType},
        ValueRange{transposed, inputShapeValue})
                               ->getResult(0);
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "block_size", rewriter.getI32IntegerAttr(blockSize))};
    Operation *depthToSpace = createTFLOperation(rewriter, op.getLoc(),
        "tfl.depth_to_space", TypeRange{physicalOutputType},
        ValueRange{reorderedInput}, attributes);
    rewriter.replaceOp(op, depthToSpace->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXDepthToSpaceOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<DepthToSpaceLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
