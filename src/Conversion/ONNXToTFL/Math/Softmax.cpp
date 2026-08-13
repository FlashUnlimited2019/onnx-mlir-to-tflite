/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class SoftmaxLowering final : public OpConversionPattern<ONNXSoftmaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    int64_t axis = -1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getValue().getSExtValue();
    axis = normalizeAxis(axis, rank);
    if (axis < 0 || axis >= rank) {
      op.emitError() << "unsupported Softmax axis " << axis << " for rank "
                     << rank;
      return failure();
    }
    if (rank == 4)
      axis = mapNCHWAxisToNHWC(axis);

    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(1.0f))};
    Type resultType = convertRank4NCHWToNHWCType(op->getResult(0).getType());
    if (axis == rank - 1) {
      Operation *newOp = createTFLOperation(rewriter, op.getLoc(),
          "tfl.softmax", TypeRange{resultType}, ValueRange{input}, attributes);
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }

    // TFLite Softmax operates on the last physical dimension. Move the ONNX
    // axis there, apply Softmax, and restore the physical tensor order.
    SmallVector<int64_t> permutation;
    permutation.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        permutation.push_back(i);
    permutation.push_back(axis);
    SmallVector<int64_t> inverse(rank);
    SmallVector<int64_t> permutedShape(rank);
    for (int64_t i = 0; i < rank; ++i) {
      inverse[permutation[i]] = i;
      permutedShape[i] = inputType.getShape()[permutation[i]];
    }
    auto permutedType =
        RankedTensorType::get(permutedShape, inputType.getElementType());
    Value forwardPermutation =
        createI32ShapeConstant(rewriter, op.getLoc(), permutation);
    Operation *forward =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{permutedType}, ValueRange{input, forwardPermutation});
    Operation *softmax =
        createTFLOperation(rewriter, op.getLoc(), "tfl.softmax",
            TypeRange{permutedType}, forward->getResults(), attributes);
    Value inversePermutation =
        createI32ShapeConstant(rewriter, op.getLoc(), inverse);
    Operation *back = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
        TypeRange{resultType},
        ValueRange{softmax->getResult(0), inversePermutation});
    rewriter.replaceOp(op, back->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SoftmaxLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
