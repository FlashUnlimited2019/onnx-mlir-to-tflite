/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <type_traits>

using namespace mlir;

namespace onnx_mlir {
namespace {

template <typename ONNXOp>
class ReduceSumLowering final : public OpConversionPattern<ONNXOp> {
public:
  using OpConversionPattern<ONNXOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    auto sourceInputType =
        dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceInputType || !sourceResultType ||
        !sourceInputType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || sourceInputType.getRank() < 1 ||
        sourceResultType.getElementType() != sourceInputType.getElementType())
      return op.emitError("ONNXToTFL ReduceSum requires matching static "
                          "ranked input/result tensors"),
             failure();
    Type elementType = sourceInputType.getElementType();
    bool isF32 = elementType.isF32();
    bool isInteger =
        elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
    if (!isF32 && !isInteger)
      return op.emitError("ONNXToTFL ReduceSum supports FP32, i32, or i64"),
             failure();

    int64_t rank = sourceInputType.getRank();
    SmallVector<int64_t> axes;
    bool axesAbsent = false;
    bool noopWithEmptyAxes = false;
    if constexpr (std::is_same_v<ONNXOp, ONNXReduceSumOp>) {
      axesAbsent = isa<NoneType>(op.getAxes().getType());
      if (!axesAbsent) {
        FailureOr<SmallVector<int64_t>> values =
            getConstantIntValues(op.getAxes());
        if (failed(values))
          return op.emitError("ONNXToTFL ReduceSum requires constant axes"),
                 failure();
        axes = std::move(*values);
      }
      if (auto attr =
              op->template getAttrOfType<IntegerAttr>("noop_with_empty_axes"))
        noopWithEmptyAxes = attr.getValue().getSExtValue() != 0;
    } else {
      if (auto attr = op->template getAttrOfType<ArrayAttr>("axes")) {
        for (Attribute element : attr)
          axes.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
      } else {
        axesAbsent = true;
      }
    }

    if ((axesAbsent || axes.empty()) && noopWithEmptyAxes) {
      rewriter.replaceOp(op, input);
      return success();
    }
    if (axesAbsent || axes.empty())
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);

    SmallVector<bool> seen(rank, false);
    for (int64_t &raw : axes) {
      int64_t axis = normalizeAxis(raw, rank);
      if (axis < 0 || axis >= rank || seen[axis])
        return op.emitError() << "invalid ReduceSum axis " << raw, failure();
      seen[axis] = true;
      raw = axis;
    }

    bool keepDims = true;
    if (auto attr = op->template getAttrOfType<IntegerAttr>("keepdims"))
      keepDims = attr.getValue().getSExtValue() != 0;

    // As with MatMul, restore logical NCHW order before a rank-4 reduction.
    // This also preserves ONNX's output dimension order when keepdims is false
    // and the result is no longer rank 4.
    if (isF32 && rank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    Operation *sum = createTFLOperation(rewriter, op.getLoc(), "tfl.sum",
        TypeRange{sourceResultType}, ValueRange{input, axisValue}, attributes);
    Value result = sum->getResult(0);
    if (!isF32 || sourceResultType.getRank() != 4) {
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

void populateLoweringONNXReduceSumOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReduceSumLowering<ONNXReduceSumOp>,
      ReduceSumLowering<ONNXReduceSumV11Op>>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
