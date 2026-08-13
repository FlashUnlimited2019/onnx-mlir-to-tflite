/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class FlattenLowering final : public OpConversionPattern<ONNXFlattenOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXFlattenOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || resultType.getRank() != 2 ||
        failed(validateStaticF32Tensor(op, inputType, "Flatten input")) ||
        failed(validateStaticF32Tensor(op, resultType, "Flatten result"))) {
      op.emitError(
          "ONNXToTFL Flatten supports static f32 tensors and rank-2 results");
      return failure();
    }
    int64_t rank = inputType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis > rank) {
      op.emitError() << "unsupported Flatten axis " << op.getAxis()
                     << " for rank " << rank;
      return failure();
    }
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i)
      outer *= inputType.getShape()[i];
    for (int64_t i = axis; i < rank; ++i)
      inner *= inputType.getShape()[i];
    if (resultType.getShape() != ArrayRef<int64_t>({outer, inner})) {
      op.emitError("Flatten axis does not match the inferred result shape");
      return failure();
    }

    Value input = adaptor.getInput();
    if (rank == 4) {
      // Rank-4 activations are physically NHWC. ONNX Flatten observes NCHW
      // row-major order, so restore logical order before flattening.
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{inputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }
    Value shape =
        createI32ShapeConstant(rewriter, op.getLoc(), resultType.getShape());
    Operation *flatten = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reshape", TypeRange{resultType}, ValueRange{input, shape});
    rewriter.replaceOp(op, flatten->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXFlattenOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<FlattenLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
