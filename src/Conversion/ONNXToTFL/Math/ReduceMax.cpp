/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ReduceMaxLowering final : public OpConversionPattern<ONNXReduceMaxV13Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXReduceMaxV13Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getData();
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool supportedElementType =
        sourceInputType && sourceResultType &&
        (sourceInputType.getElementType().isF32() ||
            sourceInputType.getElementType().isSignlessInteger(64)) &&
        sourceResultType.getElementType() == sourceInputType.getElementType();
    if (!sourceInputType || !sourceResultType ||
        !sourceInputType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || !supportedElementType) {
      op.emitError("ONNXToTFL ReduceMax requires static same-type FP32 or "
                   "i64 input/result tensors");
      return failure();
    }

    int64_t rank = sourceInputType.getRank();
    SmallVector<int64_t> axes;
    SmallVector<bool> seen(rank, false);
    if (auto attr = op->getAttrOfType<ArrayAttr>("axes")) {
      for (Attribute element : attr) {
        int64_t raw = cast<IntegerAttr>(element).getValue().getSExtValue();
        int64_t axis = normalizeAxis(raw, rank);
        if (axis < 0 || axis >= rank || seen[axis])
          return op.emitError() << "invalid ReduceMax axis " << raw, failure();
        seen[axis] = true;
        axes.push_back(axis);
      }
    } else {
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
    }

    bool keepDims = true;
    if (auto attr = op->getAttrOfType<IntegerAttr>("keepdims"))
      keepDims = attr.getValue().getSExtValue() != 0;

    // Reduction axes are logical ONNX axes. Restore NCHW order while reducing
    // rank-4 values, then convert a rank-4 result back to physical NHWC.
    if (rank == 4 && sourceInputType.getElementType().isF32()) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    Operation *reduce = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reduce_max", TypeRange{sourceResultType},
        ValueRange{input, axisValue}, attributes);
    Value result = reduce->getResult(0);
    if (!sourceResultType.getElementType().isF32() ||
        sourceResultType.getRank() != 4) {
      rewriter.replaceOp(op, result);
      return success();
    }

    Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    Operation *physicalResult =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{physicalResultType}, ValueRange{result, permutation});
    rewriter.replaceOp(op, physicalResult->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXReduceMaxOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReduceMaxLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
