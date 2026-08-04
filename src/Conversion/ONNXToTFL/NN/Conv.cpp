/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getI64ArrayOr(
    Operation *op, StringRef name, ArrayRef<int64_t> defaultValues) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaultValues);
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute element : attr)
    values.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  return values;
}

class ConvLowering final : public OpConversionPattern<ONNXConvOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 3)
      return op.emitError("Conv requires input, filter, and bias/none"),
             failure();
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto filterType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !filterType || !resultType || inputType.getRank() != 4 ||
        filterType.getRank() != 4 || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "Conv input")) ||
        failed(validateStaticF32Tensor(op, filterType, "Conv filter")) ||
        failed(validateStaticF32Tensor(op, resultType, "Conv result"))) {
      op.emitError("ONNXToTFL Conv supports static rank-4 f32 tensors only");
      return failure();
    }

    int64_t group = 1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("group"))
      group = attr.getValue().getSExtValue();
    if (group != 1)
      return op.emitError() << "unsupported Conv configuration: group=" << group
                            << " (only group=1 is supported)",
             failure();

    SmallVector<int64_t> kernel =
        getI64ArrayOr(op, "kernel_shape", filterType.getShape().take_back(2));
    SmallVector<int64_t> dilations = getI64ArrayOr(op, "dilations", {1, 1});
    SmallVector<int64_t> strides = getI64ArrayOr(op, "strides", {1, 1});
    SmallVector<int64_t> pads = getI64ArrayOr(op, "pads", {0, 0, 0, 0});
    if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2)
      return op.emitError("unsupported Conv configuration: only 2D Conv is "
                          "supported"),
             failure();
    if (llvm::any_of(dilations, [](int64_t value) { return value != 1; }))
      return op.emitError("unsupported Conv configuration: dilation must be "
                          "[1,1]"),
             failure();
    if (llvm::any_of(strides, [](int64_t value) { return value <= 0; }))
      return op.emitError("unsupported Conv configuration: strides must be "
                          "positive"),
             failure();

    StringRef autoPad = "NOTSET";
    if (auto attr = op->getAttrOfType<StringAttr>("auto_pad"))
      autoPad = attr.getValue();
    StringRef padding;
    Value input = adaptor.getOperands()[0];
    if (autoPad == "VALID")
      padding = "VALID";
    else if (autoPad == "SAME_UPPER")
      padding = "SAME";
    else if (autoPad == "NOTSET") {
      if (pads.size() != 4 ||
          llvm::any_of(pads, [](int64_t value) { return value < 0; }))
        return op.emitError(
                   "unsupported Conv padding: expected four non-negative "
                   "explicit values"),
               failure();
      padding = "VALID";
      if (llvm::any_of(pads, [](int64_t value) { return value != 0; })) {
        // TFL SAME padding can choose a different top/left split from explicit
        // ONNX padding when stride > 1. Materialize ONNX's exact pads and run
        // the convolution as VALID.
        auto convertedInputType = cast<RankedTensorType>(input.getType());
        ArrayRef<int64_t> shape = convertedInputType.getShape();
        auto paddedType =
            RankedTensorType::get({shape[0], shape[1] + pads[0] + pads[2],
                                      shape[2] + pads[1] + pads[3], shape[3]},
                convertedInputType.getElementType());
        auto paddingType = RankedTensorType::get({4, 2}, rewriter.getI32Type());
        SmallVector<int32_t> paddingValues{0, 0, static_cast<int32_t>(pads[0]),
            static_cast<int32_t>(pads[2]), static_cast<int32_t>(pads[1]),
            static_cast<int32_t>(pads[3]), 0, 0};
        Value paddingValue =
            arith::ConstantOp::create(rewriter, op.getLoc(), paddingType,
                DenseIntElementsAttr::get(
                    paddingType, ArrayRef<int32_t>(paddingValues)));
        input = createTFLOperation(rewriter, op.getLoc(), "tfl.pad",
            TypeRange{paddedType}, ValueRange{input, paddingValue})
                    ->getResult(0);
      }
    } else
      return op.emitError() << "unsupported Conv auto_pad=" << autoPad,
             failure();

    Value bias = adaptor.getOperands()[2];
    if (isa<NoneType>(bias.getType())) {
      auto zeroBiasType = RankedTensorType::get(
          {filterType.getShape()[0]}, rewriter.getF32Type());
      auto zeroBiasValue = DenseElementsAttr::get(zeroBiasType, 0.0f);
      bias = arith::ConstantOp::create(
          rewriter, op.getLoc(), zeroBiasType, zeroBiasValue);
    } else {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || biasType.getRank() != 1 ||
          biasType.getShape()[0] != filterType.getShape()[0] ||
          failed(validateStaticF32Tensor(op, biasType, "Conv bias"))) {
        op.emitError("unsupported Conv bias: expected rank-1 f32 tensor with "
                     "one value per output channel");
        return failure();
      }
    }

    Type convertedResultType =
        this->getTypeConverter()->convertType(resultType);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr(
            "dilation_h_factor", rewriter.getI32IntegerAttr(dilations[0])),
        rewriter.getNamedAttr(
            "dilation_w_factor", rewriter.getI32IntegerAttr(dilations[1])),
        getFusedActivationNone(rewriter),
        rewriter.getNamedAttr("padding", rewriter.getStringAttr(padding)),
        rewriter.getNamedAttr(
            "stride_h", rewriter.getI32IntegerAttr(strides[0])),
        rewriter.getNamedAttr(
            "stride_w", rewriter.getI32IntegerAttr(strides[1]))};
    SmallVector<Value> operands{input, adaptor.getOperands()[1], bias};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.conv_2d",
        TypeRange{convertedResultType}, operands, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXConvOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConvLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
