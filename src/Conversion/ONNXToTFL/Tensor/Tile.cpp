/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class TileLowering final : public OpConversionPattern<ONNXTileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceInputType || !sourceResultType ||
        failed(validateStaticF32Tensor(
            op, adaptor.getInput().getType(), "Tile input")) ||
        failed(validateStaticF32Tensor(op, sourceResultType, "Tile result")))
      return failure();

    int64_t rank = sourceInputType.getRank();
    if (rank < 1 || rank > 8 || sourceResultType.getRank() != rank) {
      op.emitError("ONNXToTFL Tile supports equal-rank static FP32 tensors "
                   "of rank 1 through 8");
      return failure();
    }

    FailureOr<SmallVector<int64_t>> repeats =
        getConstantIntValues(op.getRepeats());
    if (failed(repeats) || static_cast<int64_t>(repeats->size()) != rank) {
      op.emitError("ONNXToTFL Tile requires one constant repeat per axis");
      return failure();
    }
    for (int64_t axis = 0; axis < rank; ++axis) {
      int64_t repeat = (*repeats)[axis];
      int64_t inputDimension = sourceInputType.getShape()[axis];
      int64_t resultDimension = sourceResultType.getShape()[axis];
      if (repeat <= 0 || repeat > std::numeric_limits<int32_t>::max() ||
          inputDimension <= 0 ||
          inputDimension > std::numeric_limits<int64_t>::max() / repeat ||
          inputDimension * repeat != resultDimension) {
        op.emitError("Tile repeats do not match the static result shape");
        return failure();
      }
    }

    Value input = adaptor.getInput();
    Type resultType;
    SmallVector<int64_t> physicalRepeats(*repeats);
    if (rank > 5) {
      SmallVector<int64_t> reducedInputShape;
      SmallVector<int64_t> reducedResultShape;
      SmallVector<int64_t> reducedRepeats;
      int64_t axesToDrop = rank - 5;
      for (int64_t axis = 0; axis < rank; ++axis) {
        bool inertSingleton = sourceInputType.getShape()[axis] == 1 &&
                              sourceResultType.getShape()[axis] == 1 &&
                              (*repeats)[axis] == 1;
        if (inertSingleton && axesToDrop > 0) {
          --axesToDrop;
          continue;
        }
        reducedInputShape.push_back(sourceInputType.getShape()[axis]);
        reducedResultShape.push_back(sourceResultType.getShape()[axis]);
        reducedRepeats.push_back((*repeats)[axis]);
      }
      if (reducedInputShape.size() > 5) {
        op.emitError("high-rank Tile cannot be reduced to rank 5 by removing "
                     "inert singleton axes");
        return failure();
      }
      auto reducedInputType =
          RankedTensorType::get(reducedInputShape, rewriter.getF32Type());
      auto reducedResultType =
          RankedTensorType::get(reducedResultShape, rewriter.getF32Type());
      Value reducedInputShapeValue =
          createI32ShapeConstant(rewriter, op.getLoc(), reducedInputShape);
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{reducedInputType},
          ValueRange{input, reducedInputShapeValue})
                  ->getResult(0);
      physicalRepeats = std::move(reducedRepeats);
      resultType = reducedResultType;
    } else {
      resultType = convertRank4NCHWToNHWCType(sourceResultType);
    }
    if (rank == 4)
      physicalRepeats = {
          (*repeats)[0], (*repeats)[2], (*repeats)[3], (*repeats)[1]};
    Value repeatValue =
        createI32ShapeConstant(rewriter, op.getLoc(), physicalRepeats);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.tile",
        TypeRange{resultType}, ValueRange{input, repeatValue})
                       ->getResult(0);
    if (rank > 5) {
      Value restoredShape = createI32ShapeConstant(
          rewriter, op.getLoc(), sourceResultType.getShape());
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{sourceResultType}, ValueRange{result, restoredShape})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXTileOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<TileLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
