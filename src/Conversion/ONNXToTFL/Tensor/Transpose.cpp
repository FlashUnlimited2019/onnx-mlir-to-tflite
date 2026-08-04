/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
    if (failed(validateStaticF32Tensor(op, data.getType(), "data")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    int64_t rank = cast<RankedTensorType>(data.getType()).getRank();
    if (rank == 4) {
      op.emitError("rank-4 Transpose is not supported with NCHW-to-NHWC "
                   "layout conversion");
      return failure();
    }
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

    Value perm = createI32ShapeConstant(rewriter, op.getLoc(), permutation);
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(),
        "tfl.transpose",
        TypeRange{
            this->getTypeConverter()->convertType(op->getResult(0).getType())},
        ValueRange{data, perm});
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
