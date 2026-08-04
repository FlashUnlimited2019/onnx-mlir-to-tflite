/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ReduceMeanLowering final
    : public OpConversionPattern<ONNXReduceMeanV13Op> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXReduceMeanV13Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType ||
        failed(validateStaticF32Tensor(op, inputType, "ReduceMean input")) ||
        failed(validateStaticF32Tensor(op, resultType, "ReduceMean result")))
      return failure();

    int64_t rank = inputType.getRank();
    SmallVector<int64_t> axes;
    if (auto attr = op->getAttrOfType<ArrayAttr>("axes")) {
      for (Attribute element : attr) {
        int64_t raw = cast<IntegerAttr>(element).getValue().getSExtValue();
        int64_t axis = normalizeAxis(raw, rank);
        if (axis < 0 || axis >= rank)
          return op.emitError() << "unsupported ReduceMean axis " << raw,
                 failure();
        axes.push_back(axis);
      }
    } else {
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
    }

    bool keepDims = true;
    if (auto attr = op->getAttrOfType<IntegerAttr>("keepdims"))
      keepDims = attr.getValue().getSExtValue() != 0;
    if (rank == 4) {
      SmallVector<int64_t> sortedAxes(axes);
      llvm::sort(sortedAxes);
      if (!keepDims || sortedAxes != SmallVector<int64_t>{2, 3}) {
        op.emitError("unsupported rank-4 ReduceMean under NHWC layout: MVP "
                     "requires axes=[2,3] and keepdims=1");
        return failure();
      }
      for (int64_t &axis : axes)
        axis = mapNCHWAxisToNHWC(axis);
    }

    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    Type convertedResultType =
        this->getTypeConverter()->convertType(resultType);
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
        TypeRange{convertedResultType},
        ValueRange{adaptor.getOperands()[0], axisValue}, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXReduceMeanOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ReduceMeanLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
