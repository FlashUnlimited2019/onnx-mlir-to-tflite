/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getPoolArrayOr(
    Operation *op, StringRef name, ArrayRef<int64_t> defaultValues) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaultValues);
  SmallVector<int64_t> values;
  for (Attribute element : attr)
    values.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  return values;
}

class MaxPoolLowering final
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType || inputType.getRank() != 4 ||
        resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "MaxPool input")) ||
        failed(validateStaticF32Tensor(op, resultType, "MaxPool result"))) {
      op.emitError("ONNXToTFL MaxPool supports static rank-4 f32 tensors only");
      return failure();
    }
    auto getInteger = [&](StringRef name, int64_t defaultValue) {
      if (auto attr = op->getAttrOfType<IntegerAttr>(name))
        return attr.getValue().getSExtValue();
      return defaultValue;
    };
    if (getInteger("ceil_mode", 0) != 0)
      return op.emitError("unsupported MaxPool configuration: ceil_mode=1"),
             failure();
    if (getInteger("storage_order", 0) != 0)
      return op.emitError(
                 "unsupported MaxPool configuration: storage_order!=0"),
             failure();

    SmallVector<int64_t> kernel = getPoolArrayOr(op, "kernel_shape", {});
    SmallVector<int64_t> strides = getPoolArrayOr(op, "strides", {1, 1});
    SmallVector<int64_t> pads = getPoolArrayOr(op, "pads", {0, 0, 0, 0});
    SmallVector<int64_t> dilations = getPoolArrayOr(op, "dilations", {1, 1});
    if (kernel.size() != 2 || strides.size() != 2 || pads.size() != 4 ||
        dilations.size() != 2)
      return op.emitError("unsupported MaxPool configuration: only 2D pooling "
                          "is supported"),
             failure();
    if (llvm::any_of(dilations, [](int64_t value) { return value != 1; }))
      return op.emitError(
                 "unsupported MaxPool configuration: dilation must be [1,1]"),
             failure();

    StringRef autoPad = "NOTSET";
    if (auto attr = op->getAttrOfType<StringAttr>("auto_pad"))
      autoPad = attr.getValue();
    StringRef padding;
    Value input = adaptor.getOperands()[0];
    if (autoPad == "VALID") {
      padding = "VALID";
    } else if (autoPad == "SAME_UPPER") {
      padding = "SAME";
    } else if (autoPad == "NOTSET") {
      if (llvm::any_of(pads, [](int64_t value) { return value < 0; }))
        return op.emitError(
                   "unsupported MaxPool padding: values must be non-negative"),
               failure();
      padding = "VALID";
      if (llvm::any_of(pads, [](int64_t value) { return value != 0; })) {
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
        Value padValue = createF32ScalarTensorConstant(
            rewriter, op.getLoc(), std::numeric_limits<float>::lowest());
        input = createTFLOperation(rewriter, op.getLoc(), "tfl.padv2",
            TypeRange{paddedType}, ValueRange{input, paddingValue, padValue})
                    ->getResult(0);
      }
    } else {
      return op.emitError() << "unsupported MaxPool auto_pad=" << autoPad,
             failure();
    }

    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("padding", rewriter.getStringAttr(padding)),
        rewriter.getNamedAttr(
            "stride_w", rewriter.getI32IntegerAttr(strides[1])),
        rewriter.getNamedAttr(
            "stride_h", rewriter.getI32IntegerAttr(strides[0])),
        rewriter.getNamedAttr(
            "filter_width", rewriter.getI32IntegerAttr(kernel[1])),
        rewriter.getNamedAttr(
            "filter_height", rewriter.getI32IntegerAttr(kernel[0])),
        getFusedActivationNone(rewriter)};
    Type convertedResultType = convertRank4NCHWToNHWCType(resultType);
    Operation *newOp =
        createTFLOperation(rewriter, op.getLoc(), "tfl.max_pool_2d",
            TypeRange{convertedResultType}, ValueRange{input}, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXMaxPoolOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MaxPoolLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
