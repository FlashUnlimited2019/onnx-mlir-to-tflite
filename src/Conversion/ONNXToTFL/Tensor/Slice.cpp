/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class SliceLowering final : public OpConversionPattern<ONNXSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    if (failed(validateStaticF32Tensor(op, data.getType(), "data")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    auto sourceType = cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t rank = sourceType.getRank();
    FailureOr<SmallVector<int64_t>> starts =
        getConstantIntValues(op.getStarts());
    FailureOr<SmallVector<int64_t>> ends = getConstantIntValues(op.getEnds());
    if (failed(starts) || failed(ends)) {
      op.emitError("ONNXToTFL Slice requires constant starts and ends");
      return failure();
    }

    SmallVector<int64_t> axes;
    if (isa<NoneType>(op.getAxes().getType())) {
      for (int64_t i = 0, e = starts->size(); i < e; ++i)
        axes.push_back(i);
    } else {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getAxes());
      if (failed(values)) {
        op.emitError("ONNXToTFL Slice requires constant axes");
        return failure();
      }
      axes = std::move(*values);
    }
    SmallVector<int64_t> steps;
    if (isa<NoneType>(op.getSteps().getType()))
      steps.assign(starts->size(), 1);
    else {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getSteps());
      if (failed(values)) {
        op.emitError("ONNXToTFL Slice requires constant steps");
        return failure();
      }
      steps = std::move(*values);
    }
    if (starts->size() != ends->size() || starts->size() != axes.size() ||
        starts->size() != steps.size()) {
      op.emitError("invalid Slice parameters: vector lengths differ");
      return failure();
    }

    SmallVector<int64_t> begin(rank, 0);
    SmallVector<int64_t> end(sourceType.getShape());
    SmallVector<int64_t> size(sourceType.getShape());
    SmallVector<int64_t> stride(rank, 1);
    SmallVector<int64_t> reverseAxes;
    SmallVector<bool> seen(rank, false);
    bool hasStridedStep = false;
    for (auto [startValue, endValue, axisValue, step] :
        llvm::zip(*starts, *ends, axes, steps)) {
      int64_t axis = normalizeAxis(axisValue, rank);
      if (axis < 0 || axis >= rank || seen[axis]) {
        op.emitError("invalid Slice axes");
        return failure();
      }
      seen[axis] = true;
      if (step == 0 || step == std::numeric_limits<int64_t>::min()) {
        op.emitError("ONNXToTFL Slice requires a nonzero representable step");
        return failure();
      }
      int64_t dimension = sourceType.getShape()[axis];
      if (step > 0) {
        auto adjustAndClamp = [dimension](int64_t value) {
          if (value < 0) {
            if (value < -dimension)
              return int64_t{0};
            value += dimension;
          }
          return std::clamp(value, int64_t{0}, dimension);
        };
        int64_t start = adjustAndClamp(startValue);
        int64_t stop = adjustAndClamp(endValue);
        begin[axis] = start;
        end[axis] = stop;
        stride[axis] = step;
        size[axis] = stop > start ? (stop - start + step - 1) / step : 0;
        hasStridedStep |= step != 1;
        continue;
      }

      // ONNX clamps negative-step bounds to [-1, dimension - 1]. Reverse
      // that axis first, then express the same selection with a positive
      // stride. This avoids backend-dependent negative StridedSlice bounds,
      // particularly the INT64_MIN sentinel used to include element zero.
      auto adjustAndClampNegative = [dimension](int64_t value) {
        if (value < 0) {
          if (value < -dimension)
            return int64_t{-1};
          value += dimension;
        }
        return std::clamp(value, int64_t{-1}, dimension - 1);
      };
      int64_t start = adjustAndClampNegative(startValue);
      int64_t stop = adjustAndClampNegative(endValue);
      int64_t positiveStep = -step;
      reverseAxes.push_back(axis);
      begin[axis] = dimension - 1 - start;
      end[axis] = dimension - 1 - stop;
      stride[axis] = positiveStep;
      size[axis] =
          start > stop ? (start - stop + positiveStep - 1) / positiveStep : 0;
      hasStridedStep |= positiveStep != 1;
    }

    // TFLite Slice only prepares inputs through rank 5. Higher-rank ONNX
    // slices often contain layout-only singleton axes (for example
    // [N,H,W,K,1,C]). Remove axes that are unchanged singleton dimensions,
    // perform the real slice at rank <= 5, and restore the declared result
    // shape with Reshape. This changes no element order and avoids a high-rank
    // Slice kernel in the FlatBuffer.
    SmallVector<int64_t> collapsedAxes;
    SmallVector<int64_t> compactInputShape(sourceType.getShape());
    while (rank > 5) {
      std::optional<int64_t> collapsedAxis;
      for (int64_t axis = 0; axis < rank; ++axis) {
        if (compactInputShape[axis] == 1 && begin[axis] == 0 &&
            size[axis] == 1 && stride[axis] == 1 &&
            !llvm::is_contained(reverseAxes, axis)) {
          collapsedAxis = axis;
          break;
        }
      }
      if (!collapsedAxis) {
        op.emitError("ONNXToTFL Slice rank above 5 requires unchanged "
                     "singleton axes for rank reduction");
        return failure();
      }
      int64_t axis = *collapsedAxis;
      collapsedAxes.push_back(axis);
      begin.erase(begin.begin() + axis);
      end.erase(end.begin() + axis);
      size.erase(size.begin() + axis);
      stride.erase(stride.begin() + axis);
      compactInputShape.erase(compactInputShape.begin() + axis);
      for (int64_t &reverseAxis : reverseAxes)
        if (reverseAxis > axis)
          --reverseAxis;
      --rank;
    }
    if (!collapsedAxes.empty()) {
      auto compactInputType =
          RankedTensorType::get(compactInputShape, rewriter.getF32Type());
      Value compactShape =
          createI32ShapeConstant(rewriter, op.getLoc(), compactInputShape);
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{compactInputType}, ValueRange{data, compactShape})
                 ->getResult(0);
    }

    if (rank == 4) {
      SmallVector<int64_t> physicalBegin(4), physicalEnd(4), physicalSize(4),
          physicalStride(4);
      for (int64_t logicalAxis = 0; logicalAxis < 4; ++logicalAxis) {
        int64_t physicalAxis = mapNCHWAxisToNHWC(logicalAxis);
        physicalBegin[physicalAxis] = begin[logicalAxis];
        physicalEnd[physicalAxis] = end[logicalAxis];
        physicalSize[physicalAxis] = size[logicalAxis];
        physicalStride[physicalAxis] = stride[logicalAxis];
      }
      begin = std::move(physicalBegin);
      end = std::move(physicalEnd);
      size = std::move(physicalSize);
      stride = std::move(physicalStride);
      for (int64_t &axis : reverseAxes)
        axis = mapNCHWAxisToNHWC(axis);
    }
    auto finalResultType = cast<RankedTensorType>(
        convertRank4NCHWToNHWCType(op->getResult(0).getType()));
    auto resultType = finalResultType;
    if (!collapsedAxes.empty()) {
      SmallVector<int64_t> compactResultShape(
          cast<RankedTensorType>(op->getResult(0).getType()).getShape());
      for (int64_t axis : collapsedAxes)
        compactResultShape.erase(compactResultShape.begin() + axis);
      resultType =
          RankedTensorType::get(compactResultShape, rewriter.getF32Type());
    }
    if (!llvm::equal(size, resultType.getShape())) {
      op.emitError("Slice parameters do not match the inferred result shape");
      return failure();
    }
    auto fitsI32 = [](int64_t value) {
      return value >= std::numeric_limits<int32_t>::min() &&
             value <= std::numeric_limits<int32_t>::max();
    };
    if (!llvm::all_of(begin, fitsI32) || !llvm::all_of(end, fitsI32) ||
        !llvm::all_of(size, fitsI32) || !llvm::all_of(stride, fitsI32)) {
      op.emitError("Slice parameters exceed TFLite int32 range");
      return failure();
    }

    for (int64_t reverseAxis : reverseAxes) {
      Value axesValue =
          createI32ShapeConstant(rewriter, op.getLoc(), {reverseAxis});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.reverse_v2",
          TypeRange{data.getType()}, ValueRange{data, axesValue})
                 ->getResult(0);
    }

    Value beginValue = createI32ShapeConstant(rewriter, op.getLoc(), begin);
    if (hasStridedStep) {
      Value endValue = createI32ShapeConstant(rewriter, op.getLoc(), end);
      Value strideValue = createI32ShapeConstant(rewriter, op.getLoc(), stride);
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("begin_mask", rewriter.getI32IntegerAttr(0)),
          rewriter.getNamedAttr("ellipsis_mask", rewriter.getI32IntegerAttr(0)),
          rewriter.getNamedAttr("end_mask", rewriter.getI32IntegerAttr(0)),
          rewriter.getNamedAttr("new_axis_mask", rewriter.getI32IntegerAttr(0)),
          rewriter.getNamedAttr("offset", rewriter.getBoolAttr(false)),
          rewriter.getNamedAttr(
              "shrink_axis_mask", rewriter.getI32IntegerAttr(0))};
      Value result = createTFLOperation(rewriter, op.getLoc(),
          "tfl.strided_slice", TypeRange{resultType},
          ValueRange{data, beginValue, endValue, strideValue}, attributes)
                         ->getResult(0);
      if (!collapsedAxes.empty()) {
        Value restoredShape = createI32ShapeConstant(
            rewriter, op.getLoc(), finalResultType.getShape());
        result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{finalResultType}, ValueRange{result, restoredShape})
                     ->getResult(0);
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    Value sizeValue = createI32ShapeConstant(rewriter, op.getLoc(), size);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.slice",
        TypeRange{resultType}, ValueRange{data, beginValue, sizeValue})
                       ->getResult(0);
    if (!collapsedAxes.empty()) {
      Value restoredShape = createI32ShapeConstant(
          rewriter, op.getLoc(), finalResultType.getShape());
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{finalResultType}, ValueRange{result, restoredShape})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXSliceOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SliceLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
