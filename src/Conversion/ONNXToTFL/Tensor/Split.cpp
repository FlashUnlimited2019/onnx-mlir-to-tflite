/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class SplitLowering final : public OpConversionPattern<ONNXSplitOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSplitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!sourceInputType ||
        failed(validateStaticF32Tensor(op, input.getType(), "Split input")))
      return failure();
    if (op->getNumResults() == 0 ||
        op->getNumResults() > std::numeric_limits<int32_t>::max())
      return op.emitError("invalid Split result count"), failure();

    int64_t rank = sourceInputType.getRank();
    int64_t rawAxis = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      rawAxis = attr.getValue().getSExtValue();
    int64_t axis = normalizeAxis(rawAxis, rank);
    if (axis < 0 || axis >= rank)
      return op.emitError() << "invalid Split axis " << rawAxis, failure();

    SmallVector<int64_t> splitSizes;
    if (!isa<NoneType>(op.getSplit().getType())) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getSplit());
      if (failed(values))
        return op.emitError("ONNXToTFL Split requires constant split sizes"),
               failure();
      splitSizes = std::move(*values);
    } else {
      int64_t numOutputs = static_cast<int64_t>(op->getNumResults());
      if (auto attr = op->getAttrOfType<IntegerAttr>("num_outputs"))
        numOutputs = attr.getValue().getSExtValue();
      if (numOutputs != static_cast<int64_t>(op->getNumResults()) ||
          numOutputs <= 0)
        return op.emitError("Split num_outputs does not match result count"),
               failure();
      int64_t dimension = sourceInputType.getShape()[axis];
      int64_t chunk = (dimension + numOutputs - 1) / numOutputs;
      splitSizes.assign(numOutputs, chunk);
      splitSizes.back() = dimension - chunk * (numOutputs - 1);
    }

    if (splitSizes.size() != op->getNumResults())
      return op.emitError("Split sizes do not match result count"), failure();
    int64_t total = 0;
    for (int64_t size : splitSizes) {
      if (size < 0 || size > std::numeric_limits<int32_t>::max())
        return op.emitError("Split size is outside the TFLite int32 range"),
               failure();
      total += size;
    }
    if (total != sourceInputType.getShape()[axis])
      return op.emitError("Split sizes do not cover the input dimension"),
             failure();

    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      auto sourceResultType = dyn_cast<RankedTensorType>(result.getType());
      if (!sourceResultType ||
          failed(validateStaticF32Tensor(op, sourceResultType, "Split result")))
        return failure();
      ArrayRef<int64_t> inputShape = sourceInputType.getShape();
      ArrayRef<int64_t> resultShape = sourceResultType.getShape();
      if (resultShape.size() != inputShape.size())
        return op.emitError("Split result rank does not match input"),
               failure();
      for (int64_t dimension = 0; dimension < rank; ++dimension) {
        int64_t expected =
            dimension == axis ? splitSizes[index] : inputShape[dimension];
        if (resultShape[dimension] != expected)
          return op.emitError("Split result shape does not match split sizes"),
                 failure();
      }
      resultTypes.push_back(convertRank4NCHWToNHWCType(sourceResultType));
    }

    int64_t physicalAxis = rank == 4 ? mapNCHWAxisToNHWC(axis) : axis;
    Value sizes = createI32ShapeConstant(rewriter, op.getLoc(), splitSizes);
    Value splitDim = createI32ScalarTensorConstant(
        rewriter, op.getLoc(), static_cast<int32_t>(physicalAxis));
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "num_splits", rewriter.getI32IntegerAttr(op->getNumResults()))};
    Operation *split = createTFLOperation(rewriter, op.getLoc(), "tfl.split_v",
        TypeRange{resultTypes}, ValueRange{input, sizes, splitDim}, attributes);
    rewriter.replaceOp(op, split->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXSplitOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SplitLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
