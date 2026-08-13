/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class ConstantLowering final : public OpConversionPattern<ONNXConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::optional<Attribute> sparse = adaptor.getSparseValue();
    if (sparse.has_value()) {
      op.emitError("sparse tensor is not supported by ONNXToTFL MVP");
      return failure();
    }
    std::optional<Attribute> value = adaptor.getValue();
    if (!value.has_value() || !isa<ElementsAttr>(*value)) {
      op.emitError("ONNX Constant must contain a dense tensor value");
      return failure();
    }

    Attribute constantValue = *value;
    if (auto disposable = dyn_cast<DisposableElementsAttr>(constantValue))
      constantValue = disposable.toDenseElementsAttr();
    if (!isa<DenseElementsAttr>(constantValue)) {
      op.emitError(
          "ONNXToTFL MVP only supports dense numeric Constant attributes");
      return failure();
    }

    Type sourceResultType = op->getResult(0).getType();
    auto tensorType = dyn_cast<RankedTensorType>(sourceResultType);
    if (!tensorType || !tensorType.hasStaticShape()) {
      op.emitError(
          "dynamic tensor shape is not supported by ONNXToTFL MVP (Constant)");
      return failure();
    }
    Type elementType = tensorType.getElementType();
    if (!elementType.isF32() && !elementType.isInteger(1) &&
        !elementType.isUnsignedInteger(32) &&
        !elementType.isSignlessInteger(32) &&
        !elementType.isSignlessInteger(64)) {
      op.emitError() << "unsupported Constant element type: " << elementType;
      return failure();
    }

    Type resultType = convertRank4NCHWToNHWCType(sourceResultType);
    // Transpose by semantic layout, not by type inequality: OIHW [O,3,3,3]
    // and OHWI [O,3,3,3] have identical shapes but different element order.
    if (tensorType.getRank() == 4 && elementType.isF32()) {
      FailureOr<DenseElementsAttr> transposed =
          transposeRank4NCHWToNHWC(cast<DenseElementsAttr>(constantValue));
      if (failed(transposed)) {
        op.emitError("failed to transpose rank-4 Constant from NCHW/OIHW to "
                     "NHWC/OHWI");
        return failure();
      }
      constantValue = *transposed;
    }

    if (elementType.isUnsignedInteger(32)) {
      Operation *constant = createTFLOperation(rewriter, op.getLoc(),
          "tfl.pseudo_const", TypeRange{resultType}, ValueRange{},
          {rewriter.getNamedAttr("value", constantValue)});
      rewriter.replaceOp(op, constant->getResults());
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, resultType, cast<TypedAttr>(constantValue));
    }
    return success();
  }
};

} // namespace

void populateLoweringONNXConstantOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConstantLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
