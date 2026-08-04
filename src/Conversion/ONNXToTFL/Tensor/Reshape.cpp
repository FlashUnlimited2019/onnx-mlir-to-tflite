/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
    if (failed(validateStaticF32Tensor(op, data.getType(), "data")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    if (auto allowZero = op->getAttrOfType<IntegerAttr>("allowzero");
        allowZero && allowZero.getValue().getSExtValue() != 0) {
      op.emitError("unsupported Reshape configuration: allowzero=1");
      return failure();
    }

    // The inferred static result shape is authoritative. Materializing it
    // avoids ONNX's zero/-1 shape encoding leaking into TFL Reshape semantics.
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    if (llvm::any_of(resultType.getShape(), [](int64_t dimension) {
          return dimension > std::numeric_limits<int32_t>::max();
        })) {
      op.emitError(
          "unsupported Reshape result: dimension exceeds TFLite int32 shape");
      return failure();
    }
    Value shape =
        createI32ShapeConstant(rewriter, op.getLoc(), resultType.getShape());
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{resultType}, ValueRange{data, shape});
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReshapeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
