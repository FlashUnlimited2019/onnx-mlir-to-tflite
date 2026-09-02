/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "src/Dialect/ONNX/DialectBuilder.hpp"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getPermutation(ONNXTransposeOp op) {
  SmallVector<int64_t> permutation;
  if (auto array = op->getAttrOfType<ArrayAttr>("perm")) {
    for (Attribute element : array)
      permutation.push_back(cast<IntegerAttr>(element).getInt());
  } else if (auto dense = op->getAttrOfType<DenseIntElementsAttr>("perm")) {
    for (APInt element : dense.getValues<APInt>())
      permutation.push_back(element.getSExtValue());
  }
  return permutation;
}

bool isValidPermutation(ArrayRef<int64_t> permutation, int64_t rank) {
  if (static_cast<int64_t>(permutation.size()) != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int64_t dimension : permutation) {
    if (dimension < 0 || dimension >= rank || seen[dimension])
      return false;
    seen[dimension] = true;
  }
  return true;
}

bool areInversePermutations(ArrayRef<int64_t> first, ArrayRef<int64_t> second) {
  if (first.size() != second.size())
    return false;
  for (int64_t dimension = 0; dimension < static_cast<int64_t>(first.size());
       ++dimension) {
    if (second[first[dimension]] != dimension)
      return false;
  }
  return true;
}

RankedTensorType getSliceType(
    RankedTensorType sourceType, int64_t axis, int64_t extent) {
  SmallVector<int64_t> shape(sourceType.getShape());
  shape[axis] = extent;
  return RankedTensorType::get(
      shape, sourceType.getElementType(), sourceType.getEncoding());
}

Value createStaticSlice(PatternRewriter &rewriter, Location loc, Value data,
    RankedTensorType dataType, int64_t axis, int64_t start, int64_t end) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  Value starts = create.onnx.constantInt64({start});
  Value ends = create.onnx.constantInt64({end});
  Value axes = create.onnx.constantInt64({axis});
  Value steps = create.onnx.constantInt64({1});
  return ONNXSliceOp::create(rewriter, loc,
      getSliceType(dataType, axis, end - start), data, starts, ends, axes,
      steps);
}

FailureOr<Value> createStaticSliceUpdate(PatternRewriter &rewriter,
    Location loc, Value data, Value updates, RankedTensorType dataType,
    RankedTensorType updatesType, RankedTensorType resultType, int64_t axis,
    int64_t start) {
  int64_t updateExtent = updatesType.getShape()[axis];
  int64_t end = start + updateExtent;
  int64_t dataExtent = dataType.getShape()[axis];
  if (start < 0 || end > dataExtent || resultType != dataType)
    return failure();

  for (int64_t dimension = 0; dimension < dataType.getRank(); ++dimension) {
    if (dimension != axis &&
        dataType.getShape()[dimension] != updatesType.getShape()[dimension])
      return failure();
  }

  SmallVector<Value> pieces;
  if (start > 0)
    pieces.push_back(
        createStaticSlice(rewriter, loc, data, dataType, axis, 0, start));
  pieces.push_back(updates);
  if (end < dataExtent)
    pieces.push_back(createStaticSlice(
        rewriter, loc, data, dataType, axis, end, dataExtent));

  if (pieces.size() == 1)
    return pieces.front();
  return ONNXConcatOp::create(
      rewriter, loc, resultType, pieces, rewriter.getI64IntegerAttr(axis))
      .getResult();
}

// Recognize a static contiguous update that was exported by moving the
// updated dimension to the front, applying ScatterND, and restoring the
// original dimension order. Rebuilding it as Slice/Concat removes all three
// layout conversions and avoids the general GatherND/ScatterND expansion.
class TransposedScatterNDSliceUpdatePattern final
    : public OpRewritePattern<ONNXTransposeOp> {
public:
  TransposedScatterNDSliceUpdatePattern(MLIRContext *context)
      : OpRewritePattern<ONNXTransposeOp>(context, PatternBenefit(10)) {}

  LogicalResult matchAndRewrite(
      ONNXTransposeOp op, PatternRewriter &rewriter) const override {
    auto scatter = op.getData().getDefiningOp<ONNXScatterNDOp>();
    if (!scatter || !scatter.getResult().hasOneUse() ||
        scatter.getReduction() != "none")
      return failure();

    auto dataTranspose = scatter.getData().getDefiningOp<ONNXTransposeOp>();
    auto updatesTranspose =
        scatter.getUpdates().getDefiningOp<ONNXTransposeOp>();
    if (!dataTranspose || !updatesTranspose ||
        !dataTranspose.getResult().hasOneUse() ||
        !updatesTranspose.getResult().hasOneUse())
      return failure();

    auto dataType =
        dyn_cast<RankedTensorType>(dataTranspose.getData().getType());
    auto updatesType =
        dyn_cast<RankedTensorType>(updatesTranspose.getData().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto transposedDataType =
        dyn_cast<RankedTensorType>(dataTranspose.getResult().getType());
    auto transposedUpdatesType =
        dyn_cast<RankedTensorType>(updatesTranspose.getResult().getType());
    auto scatterType =
        dyn_cast<RankedTensorType>(scatter.getResult().getType());
    auto indicesType =
        dyn_cast<RankedTensorType>(scatter.getIndices().getType());
    if (!dataType || !updatesType || !resultType || !transposedDataType ||
        !transposedUpdatesType || !scatterType || !indicesType ||
        !dataType.hasStaticShape() || !updatesType.hasStaticShape() ||
        !resultType.hasStaticShape() || !transposedDataType.hasStaticShape() ||
        !transposedUpdatesType.hasStaticShape() ||
        !scatterType.hasStaticShape() || !indicesType.hasStaticShape() ||
        dataType.getRank() < 1 || dataType.getRank() > 5 ||
        updatesType.getRank() != dataType.getRank() || resultType != dataType ||
        scatterType != transposedDataType ||
        dataType.getElementType() != updatesType.getElementType() ||
        !dataType.getElementType().isF32() || indicesType.getRank() != 2 ||
        indicesType.getShape()[1] != 1 ||
        (!indicesType.getElementType().isSignlessInteger(32) &&
            !indicesType.getElementType().isSignlessInteger(64)))
      return failure();

    SmallVector<int64_t> dataPermutation = getPermutation(dataTranspose);
    SmallVector<int64_t> updatesPermutation = getPermutation(updatesTranspose);
    SmallVector<int64_t> resultPermutation = getPermutation(op);
    int64_t rank = dataType.getRank();
    if (!isValidPermutation(dataPermutation, rank) ||
        updatesPermutation != dataPermutation ||
        !isValidPermutation(resultPermutation, rank) ||
        !areInversePermutations(dataPermutation, resultPermutation))
      return failure();

    int64_t axis = dataPermutation.front();
    int64_t updateCount = indicesType.getShape()[0];
    if (updateCount != updatesType.getShape()[axis] ||
        transposedUpdatesType.getShape().front() != updateCount)
      return failure();
    for (int64_t outputDimension = 0; outputDimension < rank;
         ++outputDimension) {
      int64_t inputDimension = dataPermutation[outputDimension];
      if (transposedDataType.getShape()[outputDimension] !=
              dataType.getShape()[inputDimension] ||
          transposedUpdatesType.getShape()[outputDimension] !=
              updatesType.getShape()[inputDimension])
        return failure();
    }

    FailureOr<SmallVector<int64_t>> indices =
        getConstantIntValues(scatter.getIndices());
    if (failed(indices) ||
        static_cast<int64_t>(indices->size()) != updateCount ||
        updateCount <= 0)
      return failure();
    int64_t dataExtent = dataType.getShape()[axis];
    int64_t start = (*indices)[0];
    if (start < 0)
      start += dataExtent;
    for (int64_t offset = 0; offset < updateCount; ++offset) {
      int64_t index = (*indices)[offset];
      if (index < 0)
        index += dataExtent;
      if (index != start + offset)
        return failure();
    }

    FailureOr<Value> replacement = createStaticSliceUpdate(rewriter,
        op.getLoc(), dataTranspose.getData(), updatesTranspose.getData(),
        dataType, updatesType, resultType, axis, start);
    if (failed(replacement))
      return failure();

    rewriter.replaceOp(op, *replacement);
    rewriter.eraseOp(scatter);
    rewriter.eraseOp(dataTranspose);
    rewriter.eraseOp(updatesTranspose);
    return success();
  }
};

// ScatterElements is also a slice update when every non-axis dimension covers
// the input and every index is the same contiguous axis interval. Restricting
// the rewrite to constant, statically proven indices keeps all irregular and
// reduction cases on the established general lowering.
class ScatterElementsSliceUpdatePattern final
    : public OpRewritePattern<ONNXScatterElementsOp> {
public:
  ScatterElementsSliceUpdatePattern(MLIRContext *context)
      : OpRewritePattern<ONNXScatterElementsOp>(context, PatternBenefit(5)) {}

  LogicalResult matchAndRewrite(
      ONNXScatterElementsOp op, PatternRewriter &rewriter) const override {
    if (op.getReduction() != "none")
      return failure();
    auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto updatesType = dyn_cast<RankedTensorType>(op.getUpdates().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!dataType || !indicesType || !updatesType || !resultType ||
        !dataType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !updatesType.hasStaticShape() || !resultType.hasStaticShape() ||
        dataType.getRank() < 1 || dataType.getRank() > 5 ||
        indicesType.getRank() != dataType.getRank() ||
        updatesType.getRank() != dataType.getRank() || resultType != dataType ||
        indicesType.getShape() != updatesType.getShape() ||
        dataType.getElementType() != updatesType.getElementType() ||
        !dataType.getElementType().isF32() ||
        (!indicesType.getElementType().isSignlessInteger(32) &&
            !indicesType.getElementType().isSignlessInteger(64)))
      return failure();

    int64_t rank = dataType.getRank();
    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return failure();
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dimension != axis &&
          dataType.getShape()[dimension] != updatesType.getShape()[dimension])
        return failure();
    }

    FailureOr<SmallVector<int64_t>> indices =
        getConstantIntValues(op.getIndices());
    int64_t updateCount = updatesType.getNumElements();
    if (failed(indices) ||
        static_cast<int64_t>(indices->size()) != updateCount ||
        updateCount <= 0)
      return failure();

    ArrayRef<int64_t> updateShape = updatesType.getShape();
    int64_t dataExtent = dataType.getShape()[axis];
    int64_t axisStride = 1;
    for (int64_t dimension = axis + 1; dimension < rank; ++dimension)
      axisStride *= updateShape[dimension];
    int64_t firstIndex = (*indices)[0];
    if (firstIndex < 0)
      firstIndex += dataExtent;
    int64_t start = firstIndex;
    for (int64_t flatIndex = 0; flatIndex < updateCount; ++flatIndex) {
      int64_t axisPosition = (flatIndex / axisStride) % updateShape[axis];
      int64_t index = (*indices)[flatIndex];
      if (index < 0)
        index += dataExtent;
      if (index != start + axisPosition)
        return failure();
    }

    FailureOr<Value> replacement =
        createStaticSliceUpdate(rewriter, op.getLoc(), op.getData(),
            op.getUpdates(), dataType, updatesType, resultType, axis, start);
    if (failed(replacement))
      return failure();
    rewriter.replaceOp(op, *replacement);
    return success();
  }
};

} // namespace

void populateONNXToTFLStaticUpdatePreprocessingPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TransposedScatterNDSliceUpdatePattern,
      ScatterElementsSliceUpdatePattern>(patterns.getContext());
}

} // namespace onnx_mlir
