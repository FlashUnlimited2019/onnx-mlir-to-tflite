/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "src/Dialect/ONNX/DialectBuilder.hpp"

#include <algorithm>
#include <functional>

using namespace mlir;

namespace onnx_mlir {
namespace {

// A static prefix Slice of a Concat only needs the operands intersecting the
// prefix. Rebuild that short prefix before dialect conversion so large
// diagnostic graphs do not retain unrelated recurrent/pooling branches whose
// values provably cannot reach the model output.
class PruneStaticConcatPrefixSlicePattern final
    : public OpRewritePattern<ONNXSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXSliceOp op, PatternRewriter &rewriter) const override {
    auto concat = op.getData().getDefiningOp<ONNXConcatOp>();
    auto inputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getType());
    if (!concat || !inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || !concat.getResult().hasOneUse())
      return failure();

    FailureOr<SmallVector<int64_t>> starts =
        getConstantIntValues(op.getStarts());
    FailureOr<SmallVector<int64_t>> ends = getConstantIntValues(op.getEnds());
    FailureOr<SmallVector<int64_t>> axes = getConstantIntValues(op.getAxes());
    FailureOr<SmallVector<int64_t>> steps = getConstantIntValues(op.getSteps());
    if (failed(starts) || failed(ends) || failed(axes) || failed(steps) ||
        starts->size() != 1 || ends->size() != 1 || axes->size() != 1 ||
        steps->size() != 1 || (*starts)[0] != 0 || (*steps)[0] != 1)
      return failure();

    int64_t rank = inputType.getRank();
    int64_t axis = normalizeAxis((*axes)[0], rank);
    if (axis < 0 || axis >= rank || axis != concat.getAxis())
      return failure();
    int64_t prefix =
        std::clamp((*ends)[0], int64_t{0}, inputType.getShape()[axis]);
    if (prefix <= 0 || outputType.getShape()[axis] != prefix)
      return failure();

    SmallVector<Value> prefixOperands;
    int64_t remaining = prefix;
    bool slicedPartialOperand = false;
    for (Value operand : concat.getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || !operandType.hasStaticShape() ||
          operandType.getRank() != rank)
        return failure();
      int64_t extent = operandType.getShape()[axis];
      if (remaining >= extent) {
        prefixOperands.push_back(operand);
        remaining -= extent;
      } else if (remaining > 0) {
        SmallVector<int64_t> partialShape(operandType.getShape());
        partialShape[axis] = remaining;
        auto partialType =
            RankedTensorType::get(partialShape, operandType.getElementType());
        MultiDialectBuilder<OnnxBuilder> create(rewriter, op.getLoc());
        Value partialEnd = create.onnx.constantInt64({remaining});
        Value partial = ONNXSliceOp::create(rewriter, op.getLoc(), partialType,
            operand, op.getStarts(), partialEnd, op.getAxes(), op.getSteps());
        prefixOperands.push_back(partial);
        slicedPartialOperand = true;
        remaining = 0;
      }
      if (remaining == 0)
        break;
    }
    if (remaining != 0 || prefixOperands.empty() ||
        (!slicedPartialOperand &&
            prefixOperands.size() == concat.getNumOperands()))
      return failure();

    Value replacement;
    if (prefixOperands.size() == 1 &&
        prefixOperands.front().getType() == outputType) {
      replacement = prefixOperands.front();
    } else {
      replacement = ONNXConcatOp::create(
          rewriter, op.getLoc(), outputType, prefixOperands, axis);
    }
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

SmallVector<int64_t> getIntegerArray(Operation *op, StringRef name) {
  SmallVector<int64_t> values;
  if (auto array = op->getAttrOfType<ArrayAttr>(name)) {
    for (Attribute element : array)
      values.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  } else if (auto dense = op->getAttrOfType<DenseIntElementsAttr>(name)) {
    for (APInt element : dense.getValues<APInt>())
      values.push_back(element.getSExtValue());
  }
  return values;
}

bool hasIntegerArray(
    Operation *op, StringRef name, ArrayRef<int64_t> expected) {
  return getIntegerArray(op, name) == expected;
}

bool hasSingleResultUse(Operation *op) {
  return op && op->getNumResults() == 1 && op->getResult(0).hasOneUse();
}

bool hasStaticF32Shape(Value value, ArrayRef<int64_t> expected) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getElementType().isF32() &&
         type.getShape() == expected;
}

// Eliminate a rank-6 broadcast whose only extra dimension is a singleton:
//
//   [B,C,H,W] -> Unsqueeze [B,C,1,1,H,W]
//     -> Expand [B,C,1,K,H,W] -> Reshape [B,C,K,H,W]
//
// becomes a rank-5 Reshape followed by a rank-5 Expand. The matching is shape
// based, so equivalent static graphs are handled without relying on node names.
class CollapseSingletonExpandReshapePattern final
    : public OpRewritePattern<ONNXReshapeOp> {
public:
  CollapseSingletonExpandReshapePattern(MLIRContext *context)
      : OpRewritePattern<ONNXReshapeOp>(context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(
      ONNXReshapeOp op, PatternRewriter &rewriter) const override {
    auto expand = op.getData().getDefiningOp<ONNXExpandOp>();
    if (!hasSingleResultUse(expand))
      return failure();
    auto unsqueeze = expand.getInput().getDefiningOp<ONNXUnsqueezeOp>();
    if (!hasSingleResultUse(unsqueeze))
      return failure();

    auto unsqueezedType =
        dyn_cast<RankedTensorType>(unsqueeze.getResult().getType());
    auto expandedType =
        dyn_cast<RankedTensorType>(expand.getResult().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto sourceType = dyn_cast<RankedTensorType>(unsqueeze.getData().getType());
    if (!unsqueezedType || !expandedType || !resultType || !sourceType ||
        !unsqueezedType.hasStaticShape() || !expandedType.hasStaticShape() ||
        !resultType.hasStaticShape() || !sourceType.hasStaticShape() ||
        expandedType.getRank() <= 5 || resultType.getRank() > 5 ||
        unsqueezedType.getRank() != expandedType.getRank() ||
        resultType.getRank() >= expandedType.getRank() ||
        unsqueezedType.getElementType() != resultType.getElementType() ||
        expandedType.getElementType() != resultType.getElementType() ||
        sourceType.getElementType() != resultType.getElementType())
      return failure();

    SmallVector<bool> removed(expandedType.getRank(), false);
    SmallVector<int64_t> reducedExpandedShape(expandedType.getShape());
    SmallVector<int64_t> reducedUnsqueezedShape(unsqueezedType.getShape());
    std::function<bool(int64_t, int64_t)> selectRemovedAxes =
        [&](int64_t expandedAxis, int64_t resultAxis) {
          if (expandedAxis == expandedType.getRank())
            return resultAxis == resultType.getRank();

          // Prefer preserving dimensions that already align. Backtracking is
          // needed for ambiguous unit dimensions such as [1,1,H,W].
          if (resultAxis < resultType.getRank() &&
              expandedType.getShape()[expandedAxis] ==
                  resultType.getShape()[resultAxis] &&
              selectRemovedAxes(expandedAxis + 1, resultAxis + 1))
            return true;
          if (unsqueezedType.getShape()[expandedAxis] == 1 &&
              expandedType.getShape()[expandedAxis] == 1) {
            removed[expandedAxis] = true;
            if (selectRemovedAxes(expandedAxis + 1, resultAxis))
              return true;
            removed[expandedAxis] = false;
          }
          return false;
        };
    if (!selectRemovedAxes(0, 0))
      return failure();

    SmallVector<int64_t> compactExpandedShape;
    SmallVector<int64_t> compactInputShape;
    for (int64_t axis = 0; axis < expandedType.getRank(); ++axis) {
      if (removed[axis])
        continue;
      compactExpandedShape.push_back(reducedExpandedShape[axis]);
      compactInputShape.push_back(reducedUnsqueezedShape[axis]);
    }
    if (compactExpandedShape != resultType.getShape())
      return failure();

    int64_t sourceElements = sourceType.getNumElements();
    int64_t compactInputElements = 1;
    for (int64_t dimension : compactInputShape)
      compactInputElements *= dimension;
    if (sourceElements != compactInputElements)
      return failure();
    for (auto [inputDimension, outputDimension] :
        llvm::zip(compactInputShape, compactExpandedShape))
      if (inputDimension != 1 && inputDimension != outputDimension)
        return failure();

    Location loc = op.getLoc();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    Value compactShape = create.onnx.constantInt64(compactInputShape);
    auto compactType =
        RankedTensorType::get(compactInputShape, resultType.getElementType());
    Value compact =
        create.onnx.reshape(compactType, unsqueeze.getData(), compactShape);
    Value outputShape = create.onnx.constantInt64(resultType.getShape());
    Value replacement = create.onnx.expand(resultType, compact, outputShape);

    rewriter.replaceOp(op, replacement);
    rewriter.eraseOp(expand);
    rewriter.eraseOp(unsqueeze);
    return success();
  }
};

// Lower the static convex-upsampling pattern used by CREStereo. The exported
// graph expresses it with rank-6/rank-7 tensors. Rebuild the same computation
// with rank <= 4 before TFL conversion:
//
//   weights: [B,K*R*R,H,W] -> Softmax(K)
//   patches: Pad(flow) -> flattened K-neighbour Gather
//   weighted sum: batched MatMul
//   output: [B,C*R*R,H,W] -> CRD DepthToSpace
class LowerConvexUpsamplePattern final
    : public OpRewritePattern<ONNXReshapeOp> {
public:
  LowerConvexUpsamplePattern(MLIRContext *context)
      : OpRewritePattern<ONNXReshapeOp>(context, PatternBenefit(10)) {}

  LogicalResult matchAndRewrite(
      ONNXReshapeOp op, PatternRewriter &rewriter) const override {
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!outputType || !outputType.hasStaticShape() ||
        !outputType.getElementType().isF32() || outputType.getRank() != 4)
      return failure();

    auto outputTranspose = op.getData().getDefiningOp<ONNXTransposeOp>();
    if (!hasSingleResultUse(outputTranspose) ||
        !hasIntegerArray(outputTranspose, "perm", {0, 1, 4, 2, 5, 3}))
      return failure();
    auto reduce =
        outputTranspose->getOperand(0).getDefiningOp<ONNXReduceSumV11Op>();
    if (!hasSingleResultUse(reduce) || !hasIntegerArray(reduce, "axes", {2}))
      return failure();
    auto keepDims = reduce->getAttrOfType<IntegerAttr>("keepdims");
    if (!keepDims || keepDims.getValue().getSExtValue() != 0)
      return failure();
    auto product = reduce->getOperand(0).getDefiningOp<ONNXMulOp>();
    if (!hasSingleResultUse(product))
      return failure();

    ONNXTransposeOp normalizedWeights;
    ONNXReshapeOp patchReshape;
    for (Value operand : product->getOperands()) {
      if (auto transpose = operand.getDefiningOp<ONNXTransposeOp>())
        normalizedWeights = transpose;
      else if (auto reshape = operand.getDefiningOp<ONNXReshapeOp>())
        patchReshape = reshape;
    }
    if (!hasSingleResultUse(normalizedWeights) ||
        !hasSingleResultUse(patchReshape) ||
        !hasIntegerArray(normalizedWeights, "perm", {0, 1, 6, 3, 4, 5, 2}))
      return failure();

    auto softmax =
        normalizedWeights->getOperand(0).getDefiningOp<ONNXSoftmaxOp>();
    if (!hasSingleResultUse(softmax))
      return failure();
    auto softmaxAxis = softmax->getAttrOfType<IntegerAttr>("axis");
    if (!softmaxAxis || softmaxAxis.getValue().getSExtValue() != 6)
      return failure();
    auto moveKernel = softmax->getOperand(0).getDefiningOp<ONNXTransposeOp>();
    if (!hasSingleResultUse(moveKernel) ||
        !hasIntegerArray(moveKernel, "perm", {0, 1, 6, 3, 4, 5, 2}))
      return failure();
    auto weightReshape =
        moveKernel->getOperand(0).getDefiningOp<ONNXReshapeOp>();
    if (!hasSingleResultUse(weightReshape))
      return failure();

    auto patchTranspose =
        patchReshape->getOperand(0).getDefiningOp<ONNXTransposeOp>();
    if (!hasSingleResultUse(patchTranspose) ||
        !hasIntegerArray(patchTranspose, "perm", {0, 1, 2, 4, 3, 5}))
      return failure();
    auto gatherColumns =
        patchTranspose->getOperand(0).getDefiningOp<ONNXGatherOp>();
    if (!hasSingleResultUse(gatherColumns))
      return failure();
    auto gatherRows =
        gatherColumns->getOperand(0).getDefiningOp<ONNXGatherOp>();
    if (!hasSingleResultUse(gatherRows))
      return failure();
    auto paddedFlow = gatherRows->getOperand(0).getDefiningOp<ONNXPadOp>();
    auto rowAxis = gatherRows->getAttrOfType<IntegerAttr>("axis");
    auto columnAxis = gatherColumns->getAttrOfType<IntegerAttr>("axis");
    if (!paddedFlow || !rowAxis || !columnAxis ||
        rowAxis.getValue().getSExtValue() != 2 ||
        columnAxis.getValue().getSExtValue() != 4)
      return failure();

    auto weightInputType =
        dyn_cast<RankedTensorType>(weightReshape.getData().getType());
    auto weightShapeType =
        dyn_cast<RankedTensorType>(weightReshape.getResult().getType());
    auto paddedType =
        dyn_cast<RankedTensorType>(paddedFlow.getResult().getType());
    auto rowsType =
        dyn_cast<RankedTensorType>(gatherRows.getIndices().getType());
    auto columnsType =
        dyn_cast<RankedTensorType>(gatherColumns.getIndices().getType());
    if (!weightInputType || !weightShapeType || !paddedType || !rowsType ||
        !columnsType || !weightInputType.hasStaticShape() ||
        !weightShapeType.hasStaticShape() || !paddedType.hasStaticShape() ||
        !rowsType.hasStaticShape() || !columnsType.hasStaticShape() ||
        weightInputType.getRank() != 4 || weightShapeType.getRank() != 7 ||
        paddedType.getRank() != 4 || rowsType.getRank() != 2 ||
        columnsType.getRank() != 2)
      return failure();

    ArrayRef<int64_t> weightInputShape = weightInputType.getShape();
    ArrayRef<int64_t> weightShape = weightShapeType.getShape();
    ArrayRef<int64_t> paddedShape = paddedType.getShape();
    int64_t batch = weightShape[0];
    int64_t kernelElements = weightShape[2];
    int64_t blockSize = weightShape[3];
    int64_t height = weightShape[5];
    int64_t width = weightShape[6];
    int64_t channels = paddedShape[1];
    int64_t paddedWidth = paddedShape[3];
    int64_t kernelHeight = rowsType.getShape()[0];
    int64_t kernelWidth = columnsType.getShape()[0];
    int64_t spatialElements = height * width;
    int64_t blockElements = blockSize * blockSize;
    if (weightShape[1] != 1 || weightShape[4] != blockSize ||
        weightInputShape !=
            ArrayRef<int64_t>(
                {batch, kernelElements * blockElements, height, width}) ||
        paddedShape[0] != batch || rowsType.getShape()[1] != height ||
        columnsType.getShape()[1] != width ||
        kernelElements != kernelHeight * kernelWidth ||
        outputType.getShape() != ArrayRef<int64_t>({batch, channels,
                                     height * blockSize, width * blockSize}))
      return failure();

    if (!hasStaticF32Shape(moveKernel.getResult(),
            {batch, 1, width, blockSize, blockSize, height, kernelElements}) ||
        !hasStaticF32Shape(softmax.getResult(),
            {batch, 1, width, blockSize, blockSize, height, kernelElements}) ||
        !hasStaticF32Shape(normalizedWeights.getResult(),
            {batch, 1, kernelElements, blockSize, blockSize, height, width}) ||
        !hasStaticF32Shape(gatherRows.getResult(),
            {batch, channels, kernelHeight, height, paddedWidth}) ||
        !hasStaticF32Shape(gatherColumns.getResult(),
            {batch, channels, kernelHeight, height, kernelWidth, width}) ||
        !hasStaticF32Shape(patchTranspose.getResult(),
            {batch, channels, kernelHeight, kernelWidth, height, width}) ||
        !hasStaticF32Shape(patchReshape.getResult(),
            {batch, channels, kernelElements, 1, 1, height, width}) ||
        !hasStaticF32Shape(
            product.getResult(), {batch, channels, kernelElements, blockSize,
                                     blockSize, height, width}) ||
        !hasStaticF32Shape(reduce.getResult(),
            {batch, channels, blockSize, blockSize, height, width}) ||
        !hasStaticF32Shape(outputTranspose.getResult(),
            {batch, channels, height, blockSize, width, blockSize}))
      return failure();

    FailureOr<SmallVector<int64_t>> rowIndices =
        getConstantIntValues(gatherRows.getIndices());
    FailureOr<SmallVector<int64_t>> columnIndices =
        getConstantIntValues(gatherColumns.getIndices());
    if (failed(rowIndices) || failed(columnIndices) ||
        static_cast<int64_t>(rowIndices->size()) != kernelHeight * height ||
        static_cast<int64_t>(columnIndices->size()) != kernelWidth * width)
      return failure();

    SmallVector<int64_t> flattenedIndices;
    flattenedIndices.reserve(kernelElements * spatialElements);
    for (int64_t kernelY = 0; kernelY < kernelHeight; ++kernelY) {
      for (int64_t kernelX = 0; kernelX < kernelWidth; ++kernelX) {
        for (int64_t y = 0; y < height; ++y) {
          int64_t row = (*rowIndices)[kernelY * height + y];
          if (row < 0 || row >= paddedShape[2])
            return failure();
          for (int64_t x = 0; x < width; ++x) {
            int64_t column = (*columnIndices)[kernelX * width + x];
            if (column < 0 || column >= paddedWidth)
              return failure();
            flattenedIndices.push_back(row * paddedWidth + column);
          }
        }
      }
    }

    Location loc = op.getLoc();
    Type elementType = outputType.getElementType();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    auto tensorType = [&](ArrayRef<int64_t> shape) {
      return RankedTensorType::get(shape, elementType);
    };
    auto transpose = [&](Value input, ArrayRef<int64_t> shape,
                         ArrayRef<int64_t> permutation) {
      return create.onnx.transpose(
          tensorType(shape), input, rewriter.getI64ArrayAttr(permutation));
    };

    Value weight4DShape = create.onnx.constantInt64(
        {batch, kernelElements, blockElements, spatialElements});
    Value weight4D = create.onnx.reshape(
        tensorType({batch, kernelElements, blockElements, spatialElements}),
        weightReshape.getData(), weight4DShape);
    Value softmaxInput = transpose(weight4D,
        {batch, blockElements, spatialElements, kernelElements}, {0, 2, 3, 1});
    Value softmaxOutput = ONNXSoftmaxOp::create(
        rewriter, loc, softmaxInput.getType(), softmaxInput, /*axis=*/3)
                              .getOutput();
    Value matrixWeights = transpose(softmaxOutput,
        {batch, spatialElements, blockElements, kernelElements}, {0, 2, 1, 3});

    Value flatFlowShape = create.onnx.constantInt64(
        {batch, channels, paddedShape[2] * paddedWidth});
    Value flatFlow = create.onnx.reshape(
        tensorType({batch, channels, paddedShape[2] * paddedWidth}),
        paddedFlow.getResult(), flatFlowShape);
    auto indicesType = RankedTensorType::get(
        {kernelElements, spatialElements}, rewriter.getI64Type());
    auto indicesAttribute =
        DenseIntElementsAttr::get(indicesType, flattenedIndices);
    Value flatIndices = create.onnx.constant(indicesAttribute);
    Value patches = ONNXGatherOp::create(rewriter, loc,
        tensorType({batch, channels, kernelElements, spatialElements}),
        flatFlow, flatIndices, /*axis=*/2)
                        .getOutput();
    Value matrixPatches = transpose(patches,
        {batch, spatialElements, kernelElements, channels}, {0, 3, 2, 1});
    Value weighted = create.onnx.matmul(
        tensorType({batch, spatialElements, blockElements, channels}),
        matrixWeights, matrixPatches, /*useGemm=*/false);
    Value combined = transpose(weighted,
        {batch, channels, blockElements, spatialElements}, {0, 3, 2, 1});

    Value packedShape = create.onnx.constantInt64(
        {batch, channels * blockElements, height, width});
    Value packed = create.onnx.reshape(
        tensorType({batch, channels * blockElements, height, width}), combined,
        packedShape);
    Value replacement = ONNXDepthToSpaceOp::create(rewriter, loc, outputType,
        packed, rewriter.getI64IntegerAttr(blockSize),
        rewriter.getStringAttr("CRD"))
                            .getOutput();

    rewriter.replaceOp(op, replacement);
    rewriter.eraseOp(outputTranspose);
    rewriter.eraseOp(reduce);
    rewriter.eraseOp(product);
    rewriter.eraseOp(normalizedWeights);
    rewriter.eraseOp(softmax);
    rewriter.eraseOp(moveKernel);
    rewriter.eraseOp(weightReshape);
    rewriter.eraseOp(patchReshape);
    rewriter.eraseOp(patchTranspose);
    rewriter.eraseOp(gatherColumns);
    rewriter.eraseOp(gatherRows);
    return success();
  }
};

} // namespace

void populateONNXToTFLPreprocessingPatterns(RewritePatternSet &patterns) {
  patterns.add<PruneStaticConcatPrefixSlicePattern,
      CollapseSingletonExpandReshapePattern, LowerConvexUpsamplePattern>(
      patterns.getContext());
}

} // namespace onnx_mlir
