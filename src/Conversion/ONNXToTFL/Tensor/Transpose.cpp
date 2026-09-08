/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class TransposeLowering final : public OpConversionPattern<ONNXTransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXTransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getOperands()[0];
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        inputType.getElementType() != resultType.getElementType() ||
        (!inputType.getElementType().isF32() &&
            !inputType.getElementType().isInteger(1) &&
            !inputType.getElementType().isSignlessInteger(32) &&
            !inputType.getElementType().isSignlessInteger(64)))
      return op.emitError("ONNXToTFL Transpose requires matching static FP32, "
                          "bool, i32, or i64 tensors"),
             failure();
    int64_t rank = inputType.getRank();
    SmallVector<int64_t> permutation;
    if (auto perm = op->getAttrOfType<ArrayAttr>("perm")) {
      for (Attribute value : perm)
        permutation.push_back(
            cast<IntegerAttr>(value).getValue().getSExtValue());
    } else if (auto perm = op->getAttrOfType<DenseIntElementsAttr>("perm")) {
      for (APInt value : perm.getValues<APInt>())
        permutation.push_back(value.getSExtValue());
    } else {
      for (int64_t i = rank; i > 0; --i)
        permutation.push_back(i - 1);
    }
    if (static_cast<int64_t>(permutation.size()) != rank) {
      op.emitError("unsupported Transpose permutation: rank mismatch");
      return failure();
    }
    SmallVector<bool> seen(rank, false);
    for (int64_t axis : permutation) {
      if (axis < 0 || axis >= rank || seen[axis]) {
        op.emitError("unsupported Transpose permutation: not a permutation");
        return failure();
      }
      seen[axis] = true;
    }

    SmallVector<int64_t> inputLayout =
        getONNXToTFLPhysicalToLogicalLayout(inputType);
    SmallVector<int64_t> resultLayout =
        getONNXToTFLPhysicalToLogicalLayout(resultType);
    SmallVector<int64_t> inverseInput(rank);
    for (int64_t physicalAxis = 0; physicalAxis < rank; ++physicalAxis)
      inverseInput[inputLayout[physicalAxis]] = physicalAxis;
    SmallVector<int64_t> physicalPermutation(rank);
    for (int64_t physicalAxis = 0; physicalAxis < rank; ++physicalAxis) {
      int64_t logicalOutputAxis = resultLayout[physicalAxis];
      physicalPermutation[physicalAxis] =
          inverseInput[permutation[logicalOutputAxis]];
    }
    bool physicalIdentity = true;
    for (int64_t axis = 0; axis < rank; ++axis)
      physicalIdentity &= physicalPermutation[axis] == axis;
    Type convertedResultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (physicalIdentity && data.getType() == convertedResultType) {
      rewriter.replaceOp(op, data);
      return success();
    }

    // TFLite backends commonly cap Transpose at rank 5. High-rank attention
    // layouts often carry inert singleton axes; remove enough of them before
    // the real permutation and restore the logical result with a Reshape.
    if (rank > 5) {
      // Layout-aware boundary conversions may have a logical identity perm
      // but a non-identity physical perm. Reduce the actual TFL tensor axes.
      auto physicalInputType = cast<RankedTensorType>(data.getType());
      auto physicalResultType = cast<RankedTensorType>(convertedResultType);
      SmallVector<bool> dropped(rank, false);
      int64_t axesToDrop = rank - 5;
      for (int64_t axis = 0; axis < rank && axesToDrop > 0; ++axis) {
        if (physicalInputType.getShape()[axis] != 1)
          continue;
        dropped[axis] = true;
        --axesToDrop;
      }
      if (axesToDrop != 0)
        return op.emitError("high-rank Transpose cannot be reduced to rank 5 "
                            "by removing singleton axes"),
               failure();

      SmallVector<int64_t> oldToReduced(rank, -1);
      SmallVector<int64_t> reducedInputShape;
      for (int64_t axis = 0; axis < rank; ++axis) {
        if (dropped[axis])
          continue;
        oldToReduced[axis] = reducedInputShape.size();
        reducedInputShape.push_back(physicalInputType.getShape()[axis]);
      }
      SmallVector<int64_t> reducedOutputShape;
      SmallVector<int64_t> reducedPermutation;
      for (int64_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
        int64_t inputAxis = physicalPermutation[outputAxis];
        if (dropped[inputAxis])
          continue;
        reducedOutputShape.push_back(physicalResultType.getShape()[outputAxis]);
        reducedPermutation.push_back(oldToReduced[inputAxis]);
      }

      auto reducedInputType =
          RankedTensorType::get(reducedInputShape, inputType.getElementType());
      auto reducedOutputType = RankedTensorType::get(
          reducedOutputShape, resultType.getElementType());
      Value reducedInputShapeValue =
          createI32ShapeConstant(rewriter, op.getLoc(), reducedInputShape);
      Value reducedInput = createTFLOperation(rewriter, op.getLoc(),
          "tfl.reshape", TypeRange{reducedInputType},
          ValueRange{data, reducedInputShapeValue})
                               ->getResult(0);
      Value reducedPermutationValue =
          createI32ShapeConstant(rewriter, op.getLoc(), reducedPermutation);
      Value reducedResult = createTFLOperation(rewriter, op.getLoc(),
          "tfl.transpose", TypeRange{reducedOutputType},
          ValueRange{reducedInput, reducedPermutationValue})
                                ->getResult(0);
      Value restoredShape = createI32ShapeConstant(
          rewriter, op.getLoc(), physicalResultType.getShape());
      Value restored = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{physicalResultType},
          ValueRange{reducedResult, restoredShape})
                           ->getResult(0);
      rewriter.replaceOp(op, restored);
      return success();
    }

    permutation = std::move(physicalPermutation);

    Value perm = createI32ShapeConstant(rewriter, op.getLoc(), permutation);
    Operation *newOp =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{convertedResultType}, ValueRange{data, perm});
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXTransposeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<TransposeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
