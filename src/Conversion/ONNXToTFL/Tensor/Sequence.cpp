/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ReverseSequenceLowering final
    : public OpConversionPattern<ONNXReverseSequenceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXReverseSequenceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto lensType = dyn_cast<RankedTensorType>(op.getSequenceLens().getType());
    if (!inputType || !inputType.hasStaticShape() || !lensType ||
        !lensType.hasStaticShape() || lensType.getRank() != 1 ||
        (!inputType.getElementType().isF32() &&
            !inputType.getElementType().isSignlessInteger(32) &&
            !inputType.getElementType().isSignlessInteger(64)) ||
        (!lensType.getElementType().isSignlessInteger(32) &&
            !lensType.getElementType().isSignlessInteger(64)))
      return op.emitError(
                 "ONNXToTFL ReverseSequence requires static f32/i32/i64 "
                 "input and rank-1 i32/i64 sequence_lens"),
             failure();
    int64_t rank = inputType.getRank();
    int64_t batchAxis = normalizeAxis(op.getBatchAxis(), rank);
    int64_t timeAxis = normalizeAxis(op.getTimeAxis(), rank);
    if (batchAxis < 0 || timeAxis < 0 || batchAxis >= rank ||
        timeAxis >= rank || batchAxis == timeAxis ||
        lensType.getShape()[0] != inputType.getShape()[batchAxis])
      return op.emitError("ReverseSequence axes or sequence_lens are invalid"),
             failure();
    if (rank == 4 && inputType.getElementType().isF32()) {
      constexpr int64_t logicalToPhysical[] = {0, 3, 1, 2};
      batchAxis = logicalToPhysical[batchAxis];
      timeAxis = logicalToPhysical[timeAxis];
    }
    auto resultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(op.getY().getType()));
    Operation *result = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reverse_sequence", TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getSequenceLens()},
        {rewriter.getNamedAttr("seq_dim", rewriter.getI32IntegerAttr(timeAxis)),
            rewriter.getNamedAttr(
                "batch_dim", rewriter.getI32IntegerAttr(batchAxis))});
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

class BitShiftLowering final : public OpConversionPattern<ONNXBitShiftOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXBitShiftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto xType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto yType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getZ().getType());
    if (!xType || !yType || !resultType || !xType.hasStaticShape() ||
        !yType.hasStaticShape() || !resultType.hasStaticShape() ||
        !xType.getElementType().isUnsignedInteger(32) ||
        xType.getElementType() != yType.getElementType() ||
        xType.getElementType() != resultType.getElementType())
      return op.emitError(
                 "ONNXToTFL BitShift currently requires static uint32 tensors"),
             failure();
    if (op.getDirection() == "RIGHT") {
      if (xType.getShape() != yType.getShape())
        return op.emitError(
                   "TFLite right_shift currently requires equal input shapes"),
               failure();
      Operation *result = createTFLOperation(rewriter, op.getLoc(),
          "tfl.right_shift", TypeRange{resultType},
          ValueRange{adaptor.getX(), adaptor.getY()});
      rewriter.replaceOp(op, result->getResults());
      return success();
    }
    if (op.getDirection() != "LEFT")
      return op.emitError("BitShift direction must be LEFT or RIGHT"),
             failure();
    FailureOr<SmallVector<int64_t>> shifts = getConstantIntValues(op.getY());
    if (failed(shifts))
      return op.emitError("TFLite has no builtin left shift; shift amounts "
                          "must be constant"),
             failure();
    SmallVector<APInt> factors;
    factors.reserve(shifts->size());
    for (int64_t shift : *shifts) {
      if (shift < 0 || shift >= 32)
        return op.emitError("uint32 left shift amount must be in [0, 31]"),
               failure();
      factors.emplace_back(32, uint64_t{1} << shift, false);
    }
    auto factorAttr = DenseIntElementsAttr::get(yType, factors);
    Value factor = createTFLOperation(rewriter, op.getLoc(), "tfl.pseudo_const",
        TypeRange{yType}, ValueRange{},
        {rewriter.getNamedAttr("value", factorAttr)})
                       ->getResult(0);
    Operation *result = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{resultType}, ValueRange{adaptor.getX(), factor},
        {getFusedActivationNone(rewriter)});
    rewriter.replaceOp(op, result->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXSequenceOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReverseSequenceLowering, BitShiftLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
