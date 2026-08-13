/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

Value createPaddingMatrix(Location loc, ArrayRef<int64_t> pairs,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> values;
  values.reserve(pairs.size());
  for (int64_t value : pairs)
    values.push_back(static_cast<int32_t>(value));
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(pairs.size() / 2), 2}, rewriter.getI32Type());
  return arith::ConstantOp::create(rewriter, loc, type,
      DenseIntElementsAttr::get(type, ArrayRef<int32_t>(values)));
}

SmallVector<int64_t> getI64Values(ArrayAttr attr) {
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute item : attr)
    values.push_back(cast<IntegerAttr>(item).getInt());
  return values;
}

class UpsampleAndPadLowering final
    : public OpConversionPattern<ONNXUpsampleAndPadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXUpsampleAndPadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || inputType.getRank() < 3 ||
        inputType.getRank() > 5 ||
        resultType.getRank() != inputType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "input")) ||
        failed(validateStaticF32Tensor(op, resultType, "result"))) {
      op.emitError("ONNXToTFL UpsampleAndPad supports static rank-3 through "
                   "rank-5 f32 tensors only");
      return failure();
    }

    int64_t spatialRank = inputType.getRank() - 2;
    SmallVector<int64_t> strides(spatialRank, 1);
    if (std::optional<ArrayAttr> attr = op.getStrides())
      strides = getI64Values(*attr);
    SmallVector<int64_t> pads(2 * spatialRank, 0);
    if (std::optional<ArrayAttr> attr = op.getPads())
      pads = getI64Values(*attr);
    if (static_cast<int64_t>(strides.size()) != spatialRank ||
        static_cast<int64_t>(pads.size()) != 2 * spatialRank) {
      op.emitError("ONNXToTFL UpsampleAndPad stride/pad counts must match "
                   "the spatial rank");
      return failure();
    }
    auto fitsI32 = [](int64_t value) {
      return value >= 0 && value <= std::numeric_limits<int32_t>::max();
    };
    if (!llvm::all_of(strides,
            [&](int64_t value) { return value > 0 && fitsI32(value); }) ||
        !llvm::all_of(pads, fitsI32)) {
      op.emitError("ONNXToTFL UpsampleAndPad requires positive int32 "
                   "strides and non-negative int32 pads");
      return failure();
    }

    // A full ConvTranspose3D decomposition has no removable singleton axis.
    // Build its zero-inserted and padded rank-5 result directly: flatten the
    // input values, map every static NCDHW coordinate to the corresponding
    // padded/strided output coordinate, and let ScatterND supply zeros for all
    // other positions. This avoids rank-6 reshape/pad temporaries, which are
    // poorly supported by TFLite backends.
    if (inputType.getRank() == 5) {
      std::optional<int64_t> collapsedSpatialAxis;
      for (int64_t axis = 0; axis < 3; ++axis) {
        if (inputType.getShape()[2 + axis] == 1 &&
            resultType.getShape()[2 + axis] == 1 && strides[axis] == 1 &&
            pads[axis] == 0 && pads[axis + 3] == 0) {
          collapsedSpatialAxis = axis;
          break;
        }
      }
      if (!collapsedSpatialAxis) {
        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> resultShape = resultType.getShape();
        if (inputShape[0] != resultShape[0] || inputShape[1] != resultShape[1])
          return op.emitError("rank-5 UpsampleAndPad must preserve N and C"),
                 failure();
        for (int64_t axis = 0; axis < 3; ++axis) {
          int64_t expected = (inputShape[2 + axis] - 1) * strides[axis] + 1 +
                             pads[axis] + pads[axis + 3];
          if (expected != resultShape[2 + axis] || !fitsI32(expected))
            return op.emitError(
                       "rank-5 UpsampleAndPad attributes do not match the "
                       "inferred result shape"),
                   failure();
        }

        int64_t elements = inputType.getNumElements();
        if (!fitsI32(elements))
          return op.emitError(
                     "rank-5 UpsampleAndPad input is too large for static "
                     "ScatterND coordinates"),
                 failure();
        SmallVector<int32_t> coordinates;
        coordinates.reserve(elements * 5);
        for (int64_t n = 0; n < inputShape[0]; ++n)
          for (int64_t c = 0; c < inputShape[1]; ++c)
            for (int64_t d = 0; d < inputShape[2]; ++d)
              for (int64_t h = 0; h < inputShape[3]; ++h)
                for (int64_t w = 0; w < inputShape[4]; ++w)
                  coordinates.append(
                      {static_cast<int32_t>(n), static_cast<int32_t>(c),
                          static_cast<int32_t>(pads[0] + d * strides[0]),
                          static_cast<int32_t>(pads[1] + h * strides[1]),
                          static_cast<int32_t>(pads[2] + w * strides[2])});

        Location loc = op.getLoc();
        auto indicesType =
            RankedTensorType::get({elements, 5}, rewriter.getI32Type());
        Value indices = arith::ConstantOp::create(rewriter, loc, indicesType,
            DenseIntElementsAttr::get(indicesType, coordinates));
        auto updatesType =
            RankedTensorType::get({elements}, rewriter.getF32Type());
        Value updatesShape = createI32ShapeConstant(rewriter, loc, {elements});
        Value updates = createTFLOperation(rewriter, loc, "tfl.reshape",
            TypeRange{updatesType}, ValueRange{adaptor.getX(), updatesShape})
                            ->getResult(0);
        Value outputShape = createI32ShapeConstant(rewriter, loc, resultShape);
        Operation *result = createTFLOperation(rewriter, loc, "tfl.scatter_nd",
            TypeRange{resultType}, ValueRange{indices, updates, outputShape});
        rewriter.replaceOp(op, result->getResults());
        return success();
      }
    }

    auto physicalInputType =
        dyn_cast<RankedTensorType>(adaptor.getX().getType());
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    Value current = adaptor.getX();
    bool restoreRank5 = false;
    SmallVector<int64_t> spatialAxes = inputType.getRank() == 3
                                           ? SmallVector<int64_t>{2}
                                           : SmallVector<int64_t>{1, 2};

    // A decomposed ConvTranspose3D frequently carries an inert singleton
    // spatial axis, for example NCHW1 with strides [4,4,1]. Remove that axis
    // while inserting zeros and padding so all temporary tensors stay rank 5
    // or lower, then restore the rank-5 type for the following reducible
    // Conv3D.
    if (inputType.getRank() == 5) {
      std::optional<int64_t> collapsedSpatialAxis;
      for (int64_t axis = 0; axis < 3; ++axis) {
        if (inputType.getShape()[2 + axis] == 1 &&
            resultType.getShape()[2 + axis] == 1 && strides[axis] == 1 &&
            pads[axis] == 0 && pads[axis + 3] == 0) {
          collapsedSpatialAxis = axis;
          break;
        }
      }
      if (!collapsedSpatialAxis) {
        op.emitError("rank-5 UpsampleAndPad requires an inert singleton "
                     "spatial axis for rank reduction");
        return failure();
      }

      int64_t removedAxis = 2 + *collapsedSpatialAxis;
      SmallVector<int64_t> compactInputShape(inputType.getShape());
      SmallVector<int64_t> compactResultShape(resultType.getShape());
      compactInputShape.erase(compactInputShape.begin() + removedAxis);
      compactResultShape.erase(compactResultShape.begin() + removedAxis);
      auto compactInputType =
          RankedTensorType::get(compactInputShape, rewriter.getF32Type());
      auto compactResultType =
          RankedTensorType::get(compactResultShape, rewriter.getF32Type());
      Value compactShape =
          createI32ShapeConstant(rewriter, op.getLoc(), compactInputShape);
      current = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{compactInputType}, ValueRange{current, compactShape})
                    ->getResult(0);
      physicalInputType = compactInputType;
      physicalResultType = compactResultType;

      SmallVector<int64_t> compactStrides;
      SmallVector<int64_t> compactPads;
      for (int64_t axis = 0; axis < 3; ++axis)
        if (axis != *collapsedSpatialAxis)
          compactStrides.push_back(strides[axis]);
      for (int64_t axis = 0; axis < 3; ++axis)
        if (axis != *collapsedSpatialAxis)
          compactPads.push_back(pads[axis]);
      for (int64_t axis = 0; axis < 3; ++axis)
        if (axis != *collapsedSpatialAxis)
          compactPads.push_back(pads[axis + 3]);
      strides = std::move(compactStrides);
      pads = std::move(compactPads);
      spatialRank = 2;
      spatialAxes = {2, 3};
      restoreRank5 = true;
    }
    if (!physicalInputType || physicalInputType.getRank() != spatialRank + 2 ||
        llvm::any_of(spatialAxes, [&](int64_t axis) {
          return physicalInputType.getShape()[axis] <= 0;
        })) {
      op.emitError("ONNXToTFL UpsampleAndPad requires positive static spatial "
                   "dimensions");
      return failure();
    }

    SmallVector<int64_t> inputShape(physicalInputType.getShape());
    SmallVector<int64_t> upsampledShape(inputShape);
    SmallVector<int64_t> expectedShape(inputShape);
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t axis = spatialAxes[i];
      int64_t upsampled = (inputShape[axis] - 1) * strides[i] + 1;
      upsampledShape[axis] = upsampled;
      expectedShape[axis] = upsampled + pads[i] + pads[i + spatialRank];
    }
    if (!llvm::equal(expectedShape, physicalResultType.getShape()) ||
        !llvm::all_of(expectedShape, fitsI32)) {
      op.emitError("UpsampleAndPad attributes do not match the inferred result "
                   "shape or exceed the TFLite int32 shape range");
      return failure();
    }

    Location loc = op.getLoc();
    Value zero = createF32ScalarTensorConstant(rewriter, loc, 0.0f);
    SmallVector<int64_t> currentShape(inputShape);

    // Insert zeros after every element of a spatial axis by temporarily
    // splitting that axis into [dimension, 1], padding the singleton axis to
    // the stride, and merging the two axes again. The trailing inserted zeros
    // are cropped after both spatial axes have been expanded.
    auto expandAxis = [&](int64_t axis, int64_t stride) {
      if (stride == 1)
        return;
      SmallVector<int64_t> splitShape(currentShape);
      splitShape.insert(splitShape.begin() + axis + 1, 1);
      auto splitType = RankedTensorType::get(splitShape, rewriter.getF32Type());
      Value splitShapeValue = createI32ShapeConstant(rewriter, loc, splitShape);
      current = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{splitType}, ValueRange{current, splitShapeValue})
                    ->getResult(0);

      SmallVector<int64_t> pairs(2 * splitShape.size(), 0);
      pairs[2 * (axis + 1) + 1] = stride - 1;
      SmallVector<int64_t> paddedShape(splitShape);
      paddedShape[axis + 1] = stride;
      auto paddedType =
          RankedTensorType::get(paddedShape, rewriter.getF32Type());
      Value padding = createPaddingMatrix(loc, pairs, rewriter);
      current = createTFLOperation(rewriter, loc, "tfl.padv2",
          TypeRange{paddedType}, ValueRange{current, padding, zero})
                    ->getResult(0);

      currentShape[axis] *= stride;
      auto expandedType =
          RankedTensorType::get(currentShape, rewriter.getF32Type());
      Value expandedShapeValue =
          createI32ShapeConstant(rewriter, loc, currentShape);
      current = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{expandedType}, ValueRange{current, expandedShapeValue})
                    ->getResult(0);
    };
    for (int64_t i = 0; i < spatialRank; ++i)
      expandAxis(spatialAxes[i], strides[i]);

    if (!llvm::equal(currentShape, upsampledShape)) {
      SmallVector<int64_t> zeros(currentShape.size(), 0);
      Value begin = createI32ShapeConstant(rewriter, loc, zeros);
      Value size = createI32ShapeConstant(rewriter, loc, upsampledShape);
      auto upsampledType =
          RankedTensorType::get(upsampledShape, rewriter.getF32Type());
      current = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{upsampledType}, ValueRange{current, begin, size})
                    ->getResult(0);
      currentShape = upsampledShape;
    }

    SmallVector<int64_t> physicalPads(2 * physicalInputType.getRank(), 0);
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t axis = spatialAxes[i];
      physicalPads[2 * axis] = pads[i];
      physicalPads[2 * axis + 1] = pads[i + spatialRank];
    }
    if (llvm::any_of(physicalPads, [](int64_t value) { return value != 0; })) {
      Value padding = createPaddingMatrix(loc, physicalPads, rewriter);
      current = createTFLOperation(rewriter, loc, "tfl.padv2",
          TypeRange{physicalResultType}, ValueRange{current, padding, zero})
                    ->getResult(0);
    }

    if (restoreRank5) {
      Value restoredShape =
          createI32ShapeConstant(rewriter, loc, resultType.getShape());
      current = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{resultType}, ValueRange{current, restoredShape})
                    ->getResult(0);
    }

    rewriter.replaceOp(op, current);
    return success();
  }
};

} // namespace

void populateLoweringONNXUpsampleAndPadOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<UpsampleAndPadLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
