/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ConcatLowering final : public OpConversionPattern<ONNXConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getOperands();
    if (inputs.empty()) {
      op.emitError("Concat requires at least one input");
      return failure();
    }
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        (!resultType.getElementType().isF32() &&
            !resultType.getElementType().isInteger(1)))
      return op.emitError(
                 "Concat requires a static FP32 or boolean result tensor"),
             failure();
    for (Type type : op->getOperandTypes()) {
      auto inputType = dyn_cast<RankedTensorType>(type);
      if (!inputType || !inputType.hasStaticShape() ||
          inputType.getElementType() != resultType.getElementType() ||
          inputType.getRank() != resultType.getRank())
        return op.emitError("Concat requires matching static FP32 or boolean "
                            "input/result tensors"),
               failure();
    }

    int64_t rank = resultType.getRank();
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr) {
      op.emitError("Concat requires an axis attribute");
      return failure();
    }
    int64_t rawAxis = axisAttr.getValue().getSExtValue();
    int64_t axis = normalizeAxis(rawAxis, rank);
    if (axis < 0 || axis >= rank) {
      op.emitError() << "unsupported Concat axis " << rawAxis;
      return failure();
    }
    if (rank == 4 && resultType.getElementType().isF32())
      axis = mapNCHWAxisToNHWC(axis);

    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
        getFusedActivationNone(rewriter)};
    Operation *newOp =
        createTFLOperation(rewriter, op.getLoc(), "tfl.concatenation",
            TypeRange{convertRank4NCHWToNHWCType(op->getResult(0).getType())},
            inputs, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXConcatOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConcatLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
