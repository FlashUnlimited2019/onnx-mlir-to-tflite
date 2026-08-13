/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>

using namespace mlir;

namespace onnx_mlir {
namespace {

template <typename ONNXOp>
class ComparisonLowering final : public OpConversionPattern<ONNXOp> {
public:
  ComparisonLowering(TypeConverter &typeConverter, MLIRContext *context,
      StringRef tflName, StringRef sourceName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName),
        sourceName(sourceName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceLhsType = dyn_cast<RankedTensorType>(op.getA().getType());
    auto sourceRhsType = dyn_cast<RankedTensorType>(op.getB().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool operandsSupported =
        sourceLhsType && sourceRhsType && sourceLhsType.hasStaticShape() &&
        sourceRhsType.hasStaticShape() &&
        sourceLhsType.getElementType() == sourceRhsType.getElementType() &&
        (sourceLhsType.getElementType().isF32() ||
            sourceLhsType.getElementType().isSignlessInteger(64));
    if (!operandsSupported || !sourceResultType ||
        !sourceResultType.hasStaticShape() ||
        !sourceResultType.getElementType().isInteger(1)) {
      op.emitError() << "ONNXToTFL " << sourceName
                     << " requires matching static FP32 or i64 operands and "
                        "a static boolean result";
      return failure();
    }
    if (sourceLhsType.getRank() > 5 || sourceRhsType.getRank() > 5 ||
        sourceResultType.getRank() > 5) {
      op.emitError() << "ONNXToTFL " << sourceName
                     << " supports operand/result ranks up to 5";
      return failure();
    }

    int64_t resultRank =
        std::max(sourceLhsType.getRank(), sourceRhsType.getRank());
    SmallVector<int64_t> expectedShape(resultRank, 1);
    for (int64_t offset = 0; offset < resultRank; ++offset) {
      int64_t lhsAxis = sourceLhsType.getRank() - 1 - offset;
      int64_t rhsAxis = sourceRhsType.getRank() - 1 - offset;
      int64_t lhsDim =
          lhsAxis >= 0 ? sourceLhsType.getShape()[lhsAxis] : int64_t{1};
      int64_t rhsDim =
          rhsAxis >= 0 ? sourceRhsType.getShape()[rhsAxis] : int64_t{1};
      if (lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim) {
        op.emitError() << sourceName
                       << " operands are not broadcast-compatible";
        return failure();
      }
      expectedShape[resultRank - 1 - offset] = std::max(lhsDim, rhsDim);
    }
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError() << sourceName
                     << " result shape does not match operand broadcasting";
      return failure();
    }

    // TFL broadcasting is also trailing-dimension based. Restore physical
    // rank-4 NHWC FP32 operands to their logical ONNX NCHW form so rank-3 and
    // rank-2 operands broadcast against exactly the same axes as in ONNX.
    auto restoreLogicalRank4 = [&](Value value,
                                   RankedTensorType sourceType) -> Value {
      if (sourceType.getRank() != 4 || !sourceType.getElementType().isF32())
        return value;
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      return createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{value, permutation})
          ->getResult(0);
    };

    Value lhs = restoreLogicalRank4(adaptor.getA(), sourceLhsType);
    Value rhs = restoreLogicalRank4(adaptor.getB(), sourceRhsType);

    // TFLite's broadcast comparison kernel dispatches through a fixed 4D
    // reference implementation and aborts at runtime for rank-5 broadcasting.
    // Materialize the static broadcast with BroadcastTo, flatten both operands,
    // compare in rank 1, then restore the logical boolean result shape.
    if (resultRank > 4) {
      int64_t elementCount = sourceResultType.getNumElements();
      auto flattenOperand = [&](Value value,
                                RankedTensorType sourceType) -> Value {
        SmallVector<int64_t> alignedShape(resultRank, 1);
        std::copy(sourceType.getShape().begin(), sourceType.getShape().end(),
            alignedShape.end() - sourceType.getRank());
        if (!llvm::equal(alignedShape, expectedShape)) {
          auto broadcastType =
              RankedTensorType::get(expectedShape, sourceType.getElementType());
          Value shape =
              createI32ShapeConstant(rewriter, op.getLoc(), expectedShape);
          value = createTFLOperation(rewriter, op.getLoc(), "tfl.broadcast_to",
              TypeRange{broadcastType}, ValueRange{value, shape})
                      ->getResult(0);
        }

        auto flatType =
            RankedTensorType::get({elementCount}, sourceType.getElementType());
        Value flatShape =
            createI32ShapeConstant(rewriter, op.getLoc(), {elementCount});
        return createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{flatType}, ValueRange{value, flatShape})
            ->getResult(0);
      };

      lhs = flattenOperand(lhs, sourceLhsType);
      rhs = flattenOperand(rhs, sourceRhsType);
      auto flatResultType = RankedTensorType::get(
          {elementCount}, sourceResultType.getElementType());
      Value flatResult = createTFLOperation(rewriter, op.getLoc(), tflName,
          TypeRange{flatResultType}, ValueRange{lhs, rhs})
                             ->getResult(0);
      Value resultShape = createI32ShapeConstant(
          rewriter, op.getLoc(), sourceResultType.getShape());
      Operation *result =
          createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
              TypeRange{sourceResultType}, ValueRange{flatResult, resultShape});
      rewriter.replaceOp(op, result->getResults());
      return success();
    }

    Operation *greater = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{sourceResultType}, ValueRange{lhs, rhs});
    rewriter.replaceOp(op, greater->getResults());
    return success();
  }

private:
  std::string tflName;
  std::string sourceName;
};

} // namespace

void populateLoweringONNXComparisonOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ComparisonLowering<ONNXEqualOp>>(
      typeConverter, context, "tfl.equal", "Equal");
  patterns.add<ComparisonLowering<ONNXGreaterOp>>(
      typeConverter, context, "tfl.greater", "Greater");
  patterns.add<ComparisonLowering<ONNXGreaterOrEqualOp>>(
      typeConverter, context, "tfl.greater_equal", "GreaterOrEqual");
  patterns.add<ComparisonLowering<ONNXLessOp>>(
      typeConverter, context, "tfl.less", "Less");
}

} // namespace onnx_mlir
