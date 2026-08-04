/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

    Type resultType = op->getResult(0).getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType || !tensorType.hasStaticShape()) {
      op.emitError(
          "dynamic tensor shape is not supported by ONNXToTFL MVP (Constant)");
      return failure();
    }
    Type elementType = tensorType.getElementType();
    if (!elementType.isF32() && !elementType.isSignlessInteger(32) &&
        !elementType.isSignlessInteger(64)) {
      op.emitError() << "unsupported Constant element type: " << elementType;
      return failure();
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, resultType, cast<TypedAttr>(constantValue));
    return success();
  }
};

} // namespace

void populateLoweringONNXConstantOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConstantLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
