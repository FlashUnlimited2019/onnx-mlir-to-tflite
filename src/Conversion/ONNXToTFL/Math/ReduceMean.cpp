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
class ReduceMeanLowering final : public OpConversionPattern<ONNXOp> {
public:
  using OpConversionPattern<ONNXOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    auto sourceInputType =
        dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto sourceResultType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!sourceInputType || !sourceResultType ||
        failed(validateStaticF32TensorOrScalar(
            op, input.getType(), "ReduceMean input")) ||
        failed(validateStaticF32TensorOrScalar(
            op, sourceResultType, "ReduceMean result")))
      return failure();

    int64_t rank = sourceInputType.getRank();
    SmallVector<int64_t> axes;
    bool axesAbsent = false;
    bool noopWithEmptyAxes = false;
    if constexpr (std::is_same_v<ONNXOp, ONNXReduceMeanOp>) {
      axesAbsent = isa<NoneType>(op->getOperand(1).getType());
      if (!axesAbsent) {
        FailureOr<SmallVector<int64_t>> values =
            getConstantIntValues(op->getOperand(1));
        if (failed(values))
          return op.emitError("ONNXToTFL ReduceMean requires constant axes"),
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

    // Reducing the only value of a rank-0 tensor is an identity. Avoid
    // emitting a backend reduction with an empty axis tensor.
    if (rank == 0) {
      if (!axesAbsent && !axes.empty())
        return op.emitError("rank-0 ReduceMean cannot have an explicit axis"),
               failure();
      rewriter.replaceOp(op, input);
      return success();
    }

    if ((axesAbsent || axes.empty()) && noopWithEmptyAxes) {
      rewriter.replaceOp(op, input);
      return success();
    }
    if (axesAbsent || axes.empty()) {
      axes.clear();
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
    }

    SmallVector<bool> seen(rank, false);
    for (int64_t &raw : axes) {
      int64_t axis = normalizeAxis(raw, rank);
      if (axis < 0 || axis >= rank || seen[axis])
        return op.emitError() << "invalid ReduceMean axis " << raw, failure();
      seen[axis] = true;
      raw = axis;
    }

    bool keepDims = true;
    if (auto attr = op->template getAttrOfType<IntegerAttr>("keepdims"))
      keepDims = attr.getValue().getSExtValue() != 0;

    // Preserve the compact and well-optimized spatial rank-4 path used by
    // GlobalAveragePool and classification heads. Both the reduction axes and
    // result dimension order are already correct in physical NHWC here.
    SmallVector<int64_t> sortedAxes(axes);
    llvm::sort(sortedAxes);
    bool directSpatialRank4 =
        rank == 4 && sortedAxes == SmallVector<int64_t>{2, 3};
    if (directSpatialRank4) {
      for (int64_t &axis : axes)
        axis = mapNCHWAxisToNHWC(axis);
      Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
      Type resultType = convertRank4NCHWToNHWCType(sourceResultType);
      Operation *mean = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
          TypeRange{resultType}, ValueRange{input, axisValue}, attributes);
      rewriter.replaceOp(op, mean->getResults());
      return success();
    }

    // Every other reduction is expressed in logical ONNX dimension order.
    // Rank-4 inputs are physically NHWC, so restore NCHW first. Likewise, a
    // rank-4 result produced from any source rank is transposed back to NHWC.
    if (rank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
        TypeRange{sourceResultType}, ValueRange{input, axisValue}, attributes)
                       ->getResult(0);
    if (sourceResultType.getRank() != 4) {
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

void populateLoweringONNXReduceMeanOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReduceMeanLowering<ONNXReduceMeanOp>,
      ReduceMeanLowering<ONNXReduceMeanV13Op>>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
