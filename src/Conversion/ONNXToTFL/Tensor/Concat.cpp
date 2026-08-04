/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ConcatLowering final : public OpConversionPattern<ONNXConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = adaptor.getOperands();
    if (inputs.empty()) {
      op.emitError("Concat requires at least one input");
      return failure();
    }
    for (auto [index, input] : llvm::enumerate(inputs)) {
      if (failed(validateStaticF32Tensor(
              op, input.getType(), (Twine("input ") + Twine(index)).str())))
        return failure();
    }
    if (failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    int64_t rank = cast<RankedTensorType>(inputs[0].getType()).getRank();
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr) {
      op.emitError("Concat requires an axis attribute");
      return failure();
    }
    int64_t rawAxis = axisAttr.getValue().getSExtValue();
    int64_t axis = normalizeAxis(rawAxis, rank);
    if (axis < 0 || axis >= rank) {
      op.emitError() << "unsupported Concat axis " << rawAxis;
      return failure();
    }
    if (rank == 4)
      axis = mapNCHWAxisToNHWC(axis);

    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
        getFusedActivationNone(rewriter)};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(),
        "tfl.concatenation",
        TypeRange{
            this->getTypeConverter()->convertType(op->getResult(0).getType())},
        inputs, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXConcatOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConcatLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
