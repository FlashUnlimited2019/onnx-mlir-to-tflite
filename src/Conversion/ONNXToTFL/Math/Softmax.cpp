/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class SoftmaxLowering final : public OpConversionPattern<ONNXSoftmaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXSoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    int64_t rank = cast<RankedTensorType>(input.getType()).getRank();
    if (rank == 4) {
      op.emitError("rank-4 Softmax is not supported with NCHW-to-NHWC layout "
                   "conversion");
      return failure();
    }
    int64_t axis = -1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = attr.getValue().getSExtValue();
    axis = normalizeAxis(axis, rank);
    if (axis != rank - 1) {
      return op.emitError() << "unsupported Softmax axis " << axis
                            << ": TFLite MVP requires the last dimension",
             failure();
    }

    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(1.0f))};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.softmax",
        TypeRange{
            this->getTypeConverter()->convertType(op->getResult(0).getType())},
        adaptor.getOperands(), attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXSoftmaxOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<SoftmaxLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
