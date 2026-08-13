/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getArray(
    Operation *op, StringRef name, ArrayRef<int64_t> defaults = {}) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaults);
  SmallVector<int64_t> values;
  for (Attribute value : attr)
    values.push_back(cast<IntegerAttr>(value).getValue().getSExtValue());
  return values;
}

class Im2ColLowering final : public OpConversionPattern<ONNXIm2ColOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXIm2ColOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getRank() != 4 ||
        resultType.getRank() != 3 || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || op.getAutoPad() != "NOTSET")
      return op.emitError("ONNXToTFL Im2Col requires static rank-4 NCHW f32 "
                          "input and rank-3 output"),
             failure();
    SmallVector<int64_t> kernel = getArray(op, "kernel_shape");
    SmallVector<int64_t> pads = getArray(op, "pads", {0, 0, 0, 0});
    SmallVector<int64_t> strides = getArray(op, "strides", {1, 1});
    SmallVector<int64_t> dilations = getArray(op, "dilations", {1, 1});
    if (kernel.size() != 2 || pads.size() != 4 || strides != ArrayRef<int64_t>{1, 1} ||
        dilations != ArrayRef<int64_t>{1, 1})
      return op.emitError("ONNXToTFL Im2Col currently requires 2D unit "
                          "strides and dilations"),
             failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t outputHeight =
        inputShape[2] + pads[0] + pads[2] - kernel[0] + 1;
    int64_t outputWidth =
        inputShape[3] + pads[1] + pads[3] - kernel[1] + 1;
    if (resultType.getShape()[0] != inputShape[0] ||
        resultType.getShape()[1] != inputShape[1] * kernel[0] * kernel[1] ||
        resultType.getShape()[2] != outputHeight * outputWidth)
      return op.emitError("Im2Col result shape does not match attributes"),
             failure();

    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
    Value logicalInput = createTFLOperation(rewriter, op.getLoc(),
        "tfl.transpose", TypeRange{inputType},
        ValueRange{adaptor.getX(), permutation})
                             ->getResult(0);
    auto paddingType = RankedTensorType::get({4, 2}, rewriter.getI32Type());
    SmallVector<int32_t> paddingValues{0, 0, 0, 0,
        static_cast<int32_t>(pads[0]), static_cast<int32_t>(pads[2]),
        static_cast<int32_t>(pads[1]), static_cast<int32_t>(pads[3])};
    Value padding = arith::ConstantOp::create(rewriter, op.getLoc(),
        paddingType, DenseIntElementsAttr::get(paddingType, paddingValues));
    auto paddedType = RankedTensorType::get(
        {inputShape[0], inputShape[1], inputShape[2] + pads[0] + pads[2],
            inputShape[3] + pads[1] + pads[3]},
        rewriter.getF32Type());
    Value padded = createTFLOperation(rewriter, op.getLoc(), "tfl.pad",
        TypeRange{paddedType}, ValueRange{logicalInput, padding})
                       ->getResult(0);

    auto patchType = RankedTensorType::get(
        {inputShape[0], inputShape[1], outputHeight, outputWidth},
        rewriter.getF32Type());
    auto expandedPatchType = RankedTensorType::get(
        {inputShape[0], inputShape[1], 1, outputHeight, outputWidth},
        rewriter.getF32Type());
    SmallVector<Value> patches;
    for (int64_t y = 0; y < kernel[0]; ++y) {
      for (int64_t x = 0; x < kernel[1]; ++x) {
        Value patch = createTFLOperation(rewriter, op.getLoc(), "tfl.slice",
            TypeRange{patchType},
            ValueRange{padded,
                createI32ShapeConstant(rewriter, op.getLoc(), {0, 0, y, x}),
                createI32ShapeConstant(rewriter, op.getLoc(),
                    {inputShape[0], inputShape[1], outputHeight,
                        outputWidth})})
                          ->getResult(0);
        Value expandedShape = createI32ShapeConstant(
            rewriter, op.getLoc(), expandedPatchType.getShape());
        patches.push_back(createTFLOperation(rewriter, op.getLoc(),
            "tfl.reshape", TypeRange{expandedPatchType},
            ValueRange{patch, expandedShape})
                              ->getResult(0));
      }
    }
    auto stackedType = RankedTensorType::get(
        {inputShape[0], inputShape[1], kernel[0] * kernel[1], outputHeight,
            outputWidth},
        rewriter.getF32Type());
    SmallVector<NamedAttribute> concatAttributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(2)),
        getFusedActivationNone(rewriter)};
    Value stacked = createTFLOperation(rewriter, op.getLoc(),
        "tfl.concatenation", TypeRange{stackedType}, patches, concatAttributes)
                            ->getResult(0);
    Value outputShape = createI32ShapeConstant(
        rewriter, op.getLoc(), resultType.getShape());
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{resultType}, ValueRange{stacked, outputShape})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXIm2ColOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<Im2ColLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
