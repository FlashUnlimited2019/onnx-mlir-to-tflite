/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class ReshapeLowering final : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getOperands()[0];
    auto sourceDataType =
        dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto sourceResultType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    bool supportedElementType =
        sourceDataType && sourceResultType &&
        (sourceDataType.getElementType().isF32() ||
            sourceDataType.getElementType().isSignlessInteger(64)) &&
        sourceResultType.getElementType() == sourceDataType.getElementType();
    if (!sourceDataType || !sourceResultType ||
        !sourceDataType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || sourceResultType.getRank() < 1 ||
        !supportedElementType) {
      op.emitError("ONNXToTFL Reshape requires static same-type FP32 or i64 "
                   "data/result and result rank >= 1");
      return failure();
    }

    if (auto allowZero = op->getAttrOfType<IntegerAttr>("allowzero");
        allowZero && allowZero.getValue().getSExtValue() != 0) {
      op.emitError("unsupported Reshape configuration: allowzero=1");
      return failure();
    }

    // The inferred static result shape is authoritative. Materializing it
    // avoids ONNX's zero/-1 shape encoding leaking into TFL Reshape semantics.
    bool isF32 = sourceDataType.getElementType().isF32();
    auto resultType = isF32 ? cast<RankedTensorType>(
                                  convertRank4NCHWToNHWCType(sourceResultType))
                            : sourceResultType;
    if (llvm::any_of(resultType.getShape(), [](int64_t dimension) {
          return dimension > std::numeric_limits<int32_t>::max();
        })) {
      op.emitError(
          "unsupported Reshape result: dimension exceeds TFLite int32 shape");
      return failure();
    }
    bool sourceIsRank4 = isF32 && sourceDataType.getRank() == 4;
    bool resultIsRank4 = isF32 && sourceResultType.getRank() == 4;

    // [N,C,1,1] and its NHWC representation [N,1,1,C] have the same flat
    // element order. Keep this common GlobalAveragePool/ResNet path compact.
    bool rank4FlattenFastPath = sourceIsRank4 && !resultIsRank4 &&
                                sourceDataType.getShape()[2] == 1 &&
                                sourceDataType.getShape()[3] == 1;
    if (!sourceIsRank4 || rank4FlattenFastPath) {
      RankedTensorType reshapeType =
          resultIsRank4 ? sourceResultType : resultType;
      Value shape =
          createI32ShapeConstant(rewriter, op.getLoc(), reshapeType.getShape());
      Operation *reshape = createTFLOperation(rewriter, op.getLoc(),
          "tfl.reshape", TypeRange{reshapeType}, ValueRange{data, shape});
      if (!resultIsRank4) {
        rewriter.replaceOp(op, reshape->getResults());
        return success();
      }

      // The reshape produced logical NCHW order; transpose it to the physical
      // NHWC representation used by rank-4 TFL tensors.
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      Operation *transpose = createTFLOperation(rewriter, op.getLoc(),
          "tfl.transpose", TypeRange{resultType},
          ValueRange{reshape->getResult(0), permutation});
      rewriter.replaceOp(op, transpose->getResults());
      return success();
    }

    // Convert the physical NHWC input back to the logical NCHW element order
    // before applying ONNX's row-major reshape semantics.
    Value toNCHW = createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
    Operation *logicalInput = createTFLOperation(rewriter, op.getLoc(),
        "tfl.transpose", TypeRange{sourceDataType}, ValueRange{data, toNCHW});
    Value logicalShape = createI32ShapeConstant(
        rewriter, op.getLoc(), sourceResultType.getShape());
    Operation *reshape = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reshape", TypeRange{sourceResultType},
        ValueRange{logicalInput->getResult(0), logicalShape});
    if (!resultIsRank4) {
      rewriter.replaceOp(op, reshape->getResults());
      return success();
    }

    Value toNHWC = createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    Operation *physicalResult =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{resultType}, ValueRange{reshape->getResult(0), toNHWC});
    rewriter.replaceOp(op, physicalResult->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReshapeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
