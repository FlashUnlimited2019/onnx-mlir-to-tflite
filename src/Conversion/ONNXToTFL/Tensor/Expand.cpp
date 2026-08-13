/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {
namespace {

class ExpandLowering final : public OpConversionPattern<ONNXExpandOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXExpandOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool supportedElementType =
        sourceInputType && sourceResultType &&
        (sourceInputType.getElementType().isF32() ||
            sourceInputType.getElementType().isSignlessInteger(64)) &&
        sourceResultType.getElementType() == sourceInputType.getElementType();
    if (!sourceInputType || !sourceResultType ||
        !sourceInputType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || !supportedElementType) {
      op.emitError("ONNXToTFL Expand requires static same-type FP32 or i64 "
                   "input/result tensors");
      return failure();
    }
    if (sourceInputType.getRank() > 5 || sourceResultType.getRank() > 5) {
      op.emitError("ONNXToTFL Expand supports ranks up to 5");
      return failure();
    }

    FailureOr<SmallVector<int64_t>> requested =
        getConstantIntValues(op.getShape());
    if (failed(requested)) {
      op.emitError("ONNXToTFL Expand requires a constant shape");
      return failure();
    }
    int64_t outputRank =
        std::max<int64_t>(sourceInputType.getRank(), requested->size());
    if (sourceResultType.getRank() != outputRank) {
      op.emitError("Expand result rank does not match input and shape");
      return failure();
    }

    SmallVector<int64_t> expectedShape(outputRank, 1);
    for (int64_t offset = 0; offset < outputRank; ++offset) {
      int64_t inputAxis = sourceInputType.getRank() - 1 - offset;
      int64_t shapeAxis = requested->size() - 1 - offset;
      int64_t inputDim =
          inputAxis >= 0 ? sourceInputType.getShape()[inputAxis] : int64_t{1};
      int64_t requestedDim =
          shapeAxis >= 0 ? (*requested)[shapeAxis] : int64_t{1};
      if (requestedDim < 0 ||
          (inputDim != 1 && requestedDim != 1 && inputDim != requestedDim)) {
        op.emitError("Expand input and requested shape are not compatible");
        return failure();
      }
      expectedShape[outputRank - 1 - offset] = std::max(inputDim, requestedDim);
    }
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError("Expand result shape does not match broadcast semantics");
      return failure();
    }

    // Express broadcasting in logical ONNX order. Rank-4 activations are
    // physically NHWC at this boundary, so restore NCHW before BroadcastTo
    // and convert a rank-4 result back to NHWC afterwards.
    Value input = adaptor.getInput();
    bool isF32 = sourceInputType.getElementType().isF32();
    if (isF32 && sourceInputType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    Value shape = createI32ShapeConstant(
        rewriter, op.getLoc(), sourceResultType.getShape());
    Value logicalResult =
        createTFLOperation(rewriter, op.getLoc(), "tfl.broadcast_to",
            TypeRange{sourceResultType}, ValueRange{input, shape})
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

void populateLoweringONNXExpandOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ExpandLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
