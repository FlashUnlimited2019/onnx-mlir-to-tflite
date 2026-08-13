/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

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

FailureOr<Value> getOrCreateConvBias(Operation *op, Value bias,
    int64_t outputChannels, ConversionPatternRewriter &rewriter) {
  if (isa<NoneType>(bias.getType())) {
    auto type = RankedTensorType::get({outputChannels}, rewriter.getF32Type());
    auto value = DenseElementsAttr::get(type, 0.0f);
    return arith::ConstantOp::create(rewriter, op->getLoc(), type, value)
        .getResult();
  }
  auto type = dyn_cast<RankedTensorType>(bias.getType());
  if (!type || type.getRank() != 1 || type.getShape()[0] != outputChannels ||
      failed(validateStaticF32Tensor(op, type, "Conv bias"))) {
    op->emitError("unsupported Conv bias: expected rank-1 f32 tensor with one "
                  "value per output channel");
    return failure();
  }
  return bias;
}

FailureOr<Value> padConv2DInput(Operation *op, Value input,
    ArrayRef<int64_t> pads, ConversionPatternRewriter &rewriter) {
  if (pads.size() != 4 ||
      llvm::any_of(pads, [](int64_t value) { return value < 0; })) {
    op->emitError(
        "unsupported Conv padding: expected four non-negative 2D values");
    return failure();
  }
  if (llvm::all_of(pads, [](int64_t value) { return value == 0; }))
    return input;

  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> shape = inputType.getShape();
  auto paddedType =
      RankedTensorType::get({shape[0], shape[1] + pads[0] + pads[2],
                                shape[2] + pads[1] + pads[3], shape[3]},
          inputType.getElementType());
  auto paddingType = RankedTensorType::get({4, 2}, rewriter.getI32Type());
  SmallVector<int32_t> values{0, 0, static_cast<int32_t>(pads[0]),
      static_cast<int32_t>(pads[2]), static_cast<int32_t>(pads[1]),
      static_cast<int32_t>(pads[3]), 0, 0};
  Value padding = arith::ConstantOp::create(rewriter, op->getLoc(), paddingType,
      DenseIntElementsAttr::get(paddingType, ArrayRef<int32_t>(values)));
  return createTFLOperation(rewriter, op->getLoc(), "tfl.pad",
      TypeRange{paddedType}, ValueRange{input, padding})
      ->getResult(0);
}

SmallVector<NamedAttribute> getConv2DAttributes(Builder &builder,
    int64_t dilationH, int64_t dilationW, StringRef padding, int64_t strideH,
    int64_t strideW) {
  return {builder.getNamedAttr(
              "dilation_h_factor", builder.getI32IntegerAttr(dilationH)),
      builder.getNamedAttr(
          "dilation_w_factor", builder.getI32IntegerAttr(dilationW)),
      getFusedActivationNone(builder),
      builder.getNamedAttr("padding", builder.getStringAttr(padding)),
      builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(strideH)),
      builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(strideW))};
}

SmallVector<NamedAttribute> getConv3DAttributes(Builder &builder,
    ArrayRef<int64_t> dilations, StringRef padding, ArrayRef<int64_t> strides) {
  return {builder.getNamedAttr(
              "dilation_d_factor", builder.getI32IntegerAttr(dilations[0])),
      builder.getNamedAttr(
          "dilation_h_factor", builder.getI32IntegerAttr(dilations[1])),
      builder.getNamedAttr(
          "dilation_w_factor", builder.getI32IntegerAttr(dilations[2])),
      getFusedActivationNone(builder),
      builder.getNamedAttr("padding", builder.getStringAttr(padding)),
      builder.getNamedAttr("stride_d", builder.getI32IntegerAttr(strides[0])),
      builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(strides[1])),
      builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(strides[2]))};
}

FailureOr<Value> padConv3DInput(Operation *op, Value input,
    ArrayRef<int64_t> pads, ConversionPatternRewriter &rewriter) {
  if (pads.size() != 6 ||
      llvm::any_of(pads, [](int64_t value) { return value < 0; })) {
    op->emitError(
        "unsupported Conv3D padding: expected six non-negative values");
    return failure();
  }
  if (llvm::all_of(pads, [](int64_t value) { return value == 0; }))
    return input;

  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> shape = inputType.getShape();
  auto paddedType = RankedTensorType::get(
      {shape[0], shape[1] + pads[0] + pads[3], shape[2] + pads[1] + pads[4],
          shape[3] + pads[2] + pads[5], shape[4]},
      inputType.getElementType());
  auto paddingType = RankedTensorType::get({5, 2}, rewriter.getI32Type());
  SmallVector<int32_t> values{0, 0, static_cast<int32_t>(pads[0]),
      static_cast<int32_t>(pads[3]), static_cast<int32_t>(pads[1]),
      static_cast<int32_t>(pads[4]), static_cast<int32_t>(pads[2]),
      static_cast<int32_t>(pads[5]), 0, 0};
  Value padding = arith::ConstantOp::create(rewriter, op->getLoc(), paddingType,
      DenseIntElementsAttr::get(paddingType, ArrayRef<int32_t>(values)));
  return createTFLOperation(rewriter, op->getLoc(), "tfl.pad",
      TypeRange{paddedType}, ValueRange{input, padding})
      ->getResult(0);
}

FailureOr<Value> createGroupedConv2D(Operation *op, Value input, Value filter,
    Value bias, RankedTensorType resultType, int64_t group,
    ArrayRef<NamedAttribute> convAttributes,
    ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  int64_t inputChannels = inputType.getShape()[3];
  int64_t outputChannels = resultType.getShape()[3];
  if (group <= 1 || inputChannels % group != 0 || outputChannels % group != 0 ||
      filterType.getShape()[0] != outputChannels ||
      filterType.getShape()[3] != inputChannels / group ||
      biasType.getShape()[0] != outputChannels) {
    op->emitError("invalid grouped Conv2D decomposition shapes");
    return failure();
  }

  int64_t inputChannelsPerGroup = inputChannels / group;
  int64_t outputChannelsPerGroup = outputChannels / group;
  SmallVector<int64_t> inputSplits(group, inputChannelsPerGroup);
  SmallVector<int64_t> outputSplits(group, outputChannelsPerGroup);
  Value inputSplitSizes =
      createI32ShapeConstant(rewriter, op->getLoc(), inputSplits);
  Value outputSplitSizes =
      createI32ShapeConstant(rewriter, op->getLoc(), outputSplits);
  Value inputAxis = createI32ScalarTensorConstant(rewriter, op->getLoc(), 3);
  Value outputAxis = createI32ScalarTensorConstant(rewriter, op->getLoc(), 0);
  SmallVector<NamedAttribute> splitAttributes{
      rewriter.getNamedAttr("num_splits", rewriter.getI32IntegerAttr(group))};

  SmallVector<Type> inputPartTypes;
  SmallVector<Type> filterPartTypes;
  SmallVector<Type> biasPartTypes;
  SmallVector<Type> resultPartTypes;
  for (int64_t index = 0; index < group; ++index) {
    inputPartTypes.push_back(RankedTensorType::get(
        {inputType.getShape()[0], inputType.getShape()[1],
            inputType.getShape()[2], inputChannelsPerGroup},
        inputType.getElementType()));
    filterPartTypes.push_back(RankedTensorType::get(
        {outputChannelsPerGroup, filterType.getShape()[1],
            filterType.getShape()[2], filterType.getShape()[3]},
        filterType.getElementType()));
    biasPartTypes.push_back(RankedTensorType::get(
        {outputChannelsPerGroup}, biasType.getElementType()));
    resultPartTypes.push_back(RankedTensorType::get(
        {resultType.getShape()[0], resultType.getShape()[1],
            resultType.getShape()[2], outputChannelsPerGroup},
        resultType.getElementType()));
  }
  Operation *inputParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{inputPartTypes},
      ValueRange{input, inputSplitSizes, inputAxis}, splitAttributes);
  Operation *filterParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{filterPartTypes},
      ValueRange{filter, outputSplitSizes, outputAxis}, splitAttributes);
  Operation *biasParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{biasPartTypes},
      ValueRange{bias, outputSplitSizes, outputAxis}, splitAttributes);

  SmallVector<Value> results;
  for (int64_t index = 0; index < group; ++index) {
    Operation *conv = createTFLOperation(rewriter, op->getLoc(), "tfl.conv_2d",
        TypeRange{resultPartTypes[index]},
        ValueRange{inputParts->getResult(index), filterParts->getResult(index),
            biasParts->getResult(index)},
        convAttributes);
    results.push_back(conv->getResult(0));
  }
  SmallVector<NamedAttribute> concatAttributes{
      rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(3)),
      getFusedActivationNone(rewriter)};
  return createTFLOperation(rewriter, op->getLoc(), "tfl.concatenation",
      TypeRange{resultType}, results, concatAttributes)
      ->getResult(0);
}

FailureOr<Value> createGroupedConv3D(Operation *op, Value input, Value filter,
    Value bias, RankedTensorType resultType, int64_t group,
    ArrayRef<NamedAttribute> convAttributes,
    ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  int64_t inputChannels = inputType.getShape()[4];
  int64_t outputChannels = resultType.getShape()[4];
  if (group <= 1 || inputChannels % group != 0 || outputChannels % group != 0 ||
      filterType.getShape()[3] != inputChannels / group ||
      filterType.getShape()[4] != outputChannels ||
      biasType.getShape()[0] != outputChannels) {
    op->emitError("invalid grouped Conv3D decomposition shapes");
    return failure();
  }

  int64_t inputChannelsPerGroup = inputChannels / group;
  int64_t outputChannelsPerGroup = outputChannels / group;
  SmallVector<int64_t> inputSplits(group, inputChannelsPerGroup);
  SmallVector<int64_t> outputSplits(group, outputChannelsPerGroup);
  Value inputSplitSizes =
      createI32ShapeConstant(rewriter, op->getLoc(), inputSplits);
  Value outputSplitSizes =
      createI32ShapeConstant(rewriter, op->getLoc(), outputSplits);
  Value inputAxis = createI32ScalarTensorConstant(rewriter, op->getLoc(), 4);
  Value filterAxis = createI32ScalarTensorConstant(rewriter, op->getLoc(), 4);
  Value biasAxis = createI32ScalarTensorConstant(rewriter, op->getLoc(), 0);
  SmallVector<NamedAttribute> splitAttributes{
      rewriter.getNamedAttr("num_splits", rewriter.getI32IntegerAttr(group))};

  SmallVector<Type> inputPartTypes;
  SmallVector<Type> filterPartTypes;
  SmallVector<Type> biasPartTypes;
  SmallVector<Type> resultPartTypes;
  for (int64_t index = 0; index < group; ++index) {
    inputPartTypes.push_back(RankedTensorType::get(
        {inputType.getShape()[0], inputType.getShape()[1],
            inputType.getShape()[2], inputType.getShape()[3],
            inputChannelsPerGroup},
        inputType.getElementType()));
    filterPartTypes.push_back(RankedTensorType::get(
        {filterType.getShape()[0], filterType.getShape()[1],
            filterType.getShape()[2], filterType.getShape()[3],
            outputChannelsPerGroup},
        filterType.getElementType()));
    biasPartTypes.push_back(RankedTensorType::get(
        {outputChannelsPerGroup}, biasType.getElementType()));
    resultPartTypes.push_back(RankedTensorType::get(
        {resultType.getShape()[0], resultType.getShape()[1],
            resultType.getShape()[2], resultType.getShape()[3],
            outputChannelsPerGroup},
        resultType.getElementType()));
  }
  Operation *inputParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{inputPartTypes},
      ValueRange{input, inputSplitSizes, inputAxis}, splitAttributes);
  Operation *filterParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{filterPartTypes},
      ValueRange{filter, outputSplitSizes, filterAxis}, splitAttributes);
  Operation *biasParts = createTFLOperation(rewriter, op->getLoc(),
      "tfl.split_v", TypeRange{biasPartTypes},
      ValueRange{bias, outputSplitSizes, biasAxis}, splitAttributes);

  SmallVector<Value> results;
  for (int64_t index = 0; index < group; ++index) {
    Operation *conv = createTFLOperation(rewriter, op->getLoc(), "tfl.conv_3d",
        TypeRange{resultPartTypes[index]},
        ValueRange{inputParts->getResult(index), filterParts->getResult(index),
            biasParts->getResult(index)},
        convAttributes);
    results.push_back(conv->getResult(0));
  }
  SmallVector<NamedAttribute> concatAttributes{
      rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(4)),
      getFusedActivationNone(rewriter)};
  return createTFLOperation(rewriter, op->getLoc(), "tfl.concatenation",
      TypeRange{resultType}, results, concatAttributes)
      ->getResult(0);
}

LogicalResult lowerConv1D(ONNXConvOp op, ONNXConvOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
  auto filterType = dyn_cast<RankedTensorType>(op.getW().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!inputType || !filterType || !resultType || inputType.getRank() != 3 ||
      filterType.getRank() != 3 || resultType.getRank() != 3 ||
      failed(validateStaticF32Tensor(op, inputType, "Conv1D input")) ||
      failed(validateStaticF32Tensor(op, filterType, "Conv1D filter")) ||
      failed(validateStaticF32Tensor(op, resultType, "Conv1D result")))
    return op.emitError("Conv1D requires static rank-3 f32 tensors"), failure();

  int64_t group = op.getGroup();
  int64_t inputChannels = inputType.getShape()[1];
  int64_t outputChannels = filterType.getShape()[0];
  if (group <= 0 || inputChannels % group != 0 || outputChannels % group != 0 ||
      filterType.getShape()[1] != inputChannels / group ||
      resultType.getShape()[0] != inputType.getShape()[0] ||
      resultType.getShape()[1] != outputChannels)
    return op.emitError("invalid grouped Conv1D channel dimensions"), failure();
  bool isDepthwise = group == inputChannels && filterType.getShape()[1] == 1;
  bool isGrouped = group != 1 && !isDepthwise;
  int64_t depthMultiplier = 1;
  if (isDepthwise) {
    depthMultiplier = outputChannels / inputChannels;
  }

  SmallVector<int64_t> kernel =
      getI64ArrayOr(op, "kernel_shape", filterType.getShape().take_back(1));
  SmallVector<int64_t> dilations = getI64ArrayOr(op, "dilations", {1});
  SmallVector<int64_t> strides = getI64ArrayOr(op, "strides", {1});
  SmallVector<int64_t> pads = getI64ArrayOr(op, "pads", {0, 0});
  if (kernel.size() != 1 || kernel[0] != filterType.getShape()[2] ||
      dilations.size() != 1 || strides.size() != 1 || dilations[0] <= 0 ||
      strides[0] <= 0)
    return op.emitError("unsupported Conv1D kernel/stride/dilation"), failure();

  StringRef autoPad = op.getAutoPad();
  StringRef padding = "VALID";
  SmallVector<int64_t> pads2D{0, 0, 0, 0};
  if (autoPad == "SAME_UPPER")
    padding = "SAME";
  else if (autoPad == "NOTSET") {
    if (pads.size() != 2 ||
        llvm::any_of(pads, [](int64_t value) { return value < 0; }))
      return op.emitError("unsupported Conv1D explicit padding"), failure();
    pads2D = {pads[0], 0, pads[1], 0};
  } else if (autoPad != "VALID")
    return op.emitError() << "unsupported Conv1D auto_pad=" << autoPad,
           failure();

  int64_t n = inputType.getShape()[0];
  int64_t length = inputType.getShape()[2];
  Value inputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 1});
  auto nlcType =
      RankedTensorType::get({n, length, inputChannels}, rewriter.getF32Type());
  Value nlc = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{nlcType}, ValueRange{adaptor.getX(), inputPermutation})
                  ->getResult(0);
  auto input4DType = RankedTensorType::get(
      {n, length, 1, inputChannels}, rewriter.getF32Type());
  Value inputShape =
      createI32ShapeConstant(rewriter, op.getLoc(), input4DType.getShape());
  Value input4D = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{input4DType}, ValueRange{nlc, inputShape})
                      ->getResult(0);
  if (padding == "VALID") {
    FailureOr<Value> padded = padConv2DInput(op, input4D, pads2D, rewriter);
    if (failed(padded))
      return failure();
    input4D = *padded;
  }

  int64_t k = kernel[0];
  Value filterPermutation;
  RankedTensorType transposedFilterType;
  RankedTensorType filter4DType;
  if (!isDepthwise) {
    filterPermutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 1});
    int64_t inputChannelsPerGroup = filterType.getShape()[1];
    transposedFilterType = RankedTensorType::get(
        {outputChannels, k, inputChannelsPerGroup}, rewriter.getF32Type());
    filter4DType = RankedTensorType::get(
        {outputChannels, k, 1, inputChannelsPerGroup}, rewriter.getF32Type());
  } else {
    filterPermutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {1, 2, 0});
    transposedFilterType =
        RankedTensorType::get({1, k, outputChannels}, rewriter.getF32Type());
    filter4DType =
        RankedTensorType::get({1, k, 1, outputChannels}, rewriter.getF32Type());
  }
  Value transposedFilter = createTFLOperation(rewriter, op.getLoc(),
      "tfl.transpose", TypeRange{transposedFilterType},
      ValueRange{adaptor.getW(), filterPermutation})
                               ->getResult(0);
  Value filterShape =
      createI32ShapeConstant(rewriter, op.getLoc(), filter4DType.getShape());
  Value filter4D = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{filter4DType}, ValueRange{transposedFilter, filterShape})
                       ->getResult(0);

  FailureOr<Value> bias =
      getOrCreateConvBias(op, adaptor.getB(), outputChannels, rewriter);
  if (failed(bias))
    return failure();
  SmallVector<NamedAttribute> attributes =
      getConv2DAttributes(rewriter, dilations[0], 1, padding, strides[0], 1);
  StringRef opName = "tfl.conv_2d";
  if (isDepthwise) {
    attributes.push_back(rewriter.getNamedAttr(
        "depth_multiplier", rewriter.getI32IntegerAttr(depthMultiplier)));
    opName = "tfl.depthwise_conv_2d";
  }
  int64_t outputLength = resultType.getShape()[2];
  auto convType = RankedTensorType::get(
      {n, outputLength, 1, outputChannels}, rewriter.getF32Type());
  Value conv;
  if (isGrouped) {
    FailureOr<Value> grouped = createGroupedConv2D(
        op, input4D, filter4D, *bias, convType, group, attributes, rewriter);
    if (failed(grouped))
      return failure();
    conv = *grouped;
  } else {
    conv = createTFLOperation(rewriter, op.getLoc(), opName,
        TypeRange{convType}, ValueRange{input4D, filter4D, *bias}, attributes)
               ->getResult(0);
  }
  auto nlmType = RankedTensorType::get(
      {n, outputLength, outputChannels}, rewriter.getF32Type());
  Value outputShape =
      createI32ShapeConstant(rewriter, op.getLoc(), nlmType.getShape());
  Value nlm = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{nlmType}, ValueRange{conv, outputShape})
                  ->getResult(0);
  Value outputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 1});
  Value output = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{resultType}, ValueRange{nlm, outputPermutation})
                     ->getResult(0);
  rewriter.replaceOp(op, output);
  return success();
}

LogicalResult lowerDirectConv3D(ONNXConvOp op, ONNXConvOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(op.getX().getType());
  auto filterType = cast<RankedTensorType>(op.getW().getType());
  auto resultType = cast<RankedTensorType>(op.getY().getType());
  int64_t group = op.getGroup();
  int64_t inputChannels = inputType.getShape()[1];
  int64_t outputChannels = filterType.getShape()[0];
  if (group <= 0 || inputChannels % group != 0 || outputChannels % group != 0 ||
      filterType.getShape()[1] != inputChannels / group ||
      resultType.getShape()[0] != inputType.getShape()[0] ||
      resultType.getShape()[1] != outputChannels)
    return op.emitError("invalid grouped Conv3D channel dimensions"), failure();

  SmallVector<int64_t> kernel =
      getI64ArrayOr(op, "kernel_shape", filterType.getShape().take_back(3));
  SmallVector<int64_t> dilations = getI64ArrayOr(op, "dilations", {1, 1, 1});
  SmallVector<int64_t> strides = getI64ArrayOr(op, "strides", {1, 1, 1});
  SmallVector<int64_t> pads = getI64ArrayOr(op, "pads", {0, 0, 0, 0, 0, 0});
  if (kernel.size() != 3 || dilations.size() != 3 || strides.size() != 3 ||
      !llvm::equal(kernel, filterType.getShape().take_back(3)) ||
      llvm::any_of(dilations, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(strides, [](int64_t value) { return value <= 0; }))
    return op.emitError("unsupported Conv3D kernel/stride/dilation"), failure();

  StringRef autoPad = op.getAutoPad();
  StringRef padding = "VALID";
  bool materializePadding = false;
  if (autoPad == "SAME_UPPER")
    padding = "SAME";
  else if (autoPad == "NOTSET") {
    if (pads.size() != 6 ||
        llvm::any_of(pads, [](int64_t value) { return value < 0; }))
      return op.emitError("unsupported Conv3D explicit padding"), failure();
    materializePadding = true;
  } else if (autoPad == "SAME_LOWER") {
    pads.assign(6, 0);
    for (int64_t axis = 0; axis < 3; ++axis) {
      int64_t inputExtent = inputType.getShape()[2 + axis];
      int64_t outputExtent = resultType.getShape()[2 + axis];
      int64_t effectiveKernel = dilations[axis] * (kernel[axis] - 1) + 1;
      int64_t totalPadding = std::max<int64_t>(
          (outputExtent - 1) * strides[axis] + effectiveKernel - inputExtent,
          0);
      pads[axis + 3] = totalPadding / 2;
      pads[axis] = totalPadding - pads[axis + 3];
    }
    materializePadding = true;
  } else if (autoPad != "VALID")
    return op.emitError() << "unsupported Conv3D auto_pad=" << autoPad,
           failure();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  auto ndhwcInputType =
      RankedTensorType::get({inputShape[0], inputShape[2], inputShape[3],
                                inputShape[4], inputChannels},
          rewriter.getF32Type());
  Value inputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 4, 1});
  Value input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{ndhwcInputType}, ValueRange{adaptor.getX(), inputPermutation})
                    ->getResult(0);
  if (materializePadding) {
    FailureOr<Value> padded = padConv3DInput(op, input, pads, rewriter);
    if (failed(padded))
      return failure();
    input = *padded;
  }

  int64_t inputChannelsPerGroup = filterType.getShape()[1];
  auto dhwioFilterType = RankedTensorType::get(
      {kernel[0], kernel[1], kernel[2], inputChannelsPerGroup, outputChannels},
      rewriter.getF32Type());
  Value filterPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {2, 3, 4, 1, 0});
  Value filter = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{dhwioFilterType}, ValueRange{adaptor.getW(), filterPermutation})
                     ->getResult(0);
  FailureOr<Value> bias =
      getOrCreateConvBias(op, adaptor.getB(), outputChannels, rewriter);
  if (failed(bias))
    return failure();

  ArrayRef<int64_t> resultShape = resultType.getShape();
  auto ndhwcResultType =
      RankedTensorType::get({resultShape[0], resultShape[2], resultShape[3],
                                resultShape[4], outputChannels},
          rewriter.getF32Type());
  SmallVector<NamedAttribute> attributes =
      getConv3DAttributes(rewriter, dilations, padding, strides);
  Value conv;
  if (group == 1) {
    conv = createTFLOperation(rewriter, op.getLoc(), "tfl.conv_3d",
        TypeRange{ndhwcResultType}, ValueRange{input, filter, *bias},
        attributes)
               ->getResult(0);
  } else {
    FailureOr<Value> grouped = createGroupedConv3D(
        op, input, filter, *bias, ndhwcResultType, group, attributes, rewriter);
    if (failed(grouped))
      return failure();
    conv = *grouped;
  }

  Value outputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 4, 1, 2, 3});
  Value output = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{resultType}, ValueRange{conv, outputPermutation})
                     ->getResult(0);
  rewriter.replaceOp(op, output);
  return success();
}

LogicalResult lowerFullDepthConv3DAsConv2D(ONNXConvOp op,
    ONNXConvOpAdaptor adaptor, RankedTensorType inputType,
    RankedTensorType filterType, RankedTensorType resultType,
    ArrayRef<int64_t> kernel, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
    ConversionPatternRewriter &rewriter) {
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  int64_t n = inputShape[0];
  int64_t inputChannels = inputShape[1];
  int64_t inputDepth = inputShape[2];
  int64_t inputHeight = inputShape[3];
  int64_t inputWidth = inputShape[4];
  int64_t outputChannels = filterType.getShape()[0];
  int64_t mergedChannels = inputChannels * inputDepth;

  // NCDHW is contiguous in C,D order, so a full-depth convolution with one
  // depth output can merge C and D without changing the dot-product order.
  // Convert the resulting N(CH)HW tensor to the NHWC layout expected by
  // TFLite Conv2D.
  auto nchwInputType = RankedTensorType::get(
      {n, mergedChannels, inputHeight, inputWidth}, rewriter.getF32Type());
  Value nchwInputShape =
      createI32ShapeConstant(rewriter, op.getLoc(), nchwInputType.getShape());
  Value nchwInput = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{nchwInputType}, ValueRange{adaptor.getX(), nchwInputShape})
                        ->getResult(0);
  auto nhwcInputType = RankedTensorType::get(
      {n, inputHeight, inputWidth, mergedChannels}, rewriter.getF32Type());
  Value inputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
  Value input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{nhwcInputType}, ValueRange{nchwInput, inputPermutation})
                    ->getResult(0);

  StringRef padding = op.getAutoPad() == "SAME_UPPER" ? "SAME" : "VALID";
  if (op.getAutoPad() == "NOTSET") {
    SmallVector<int64_t> pads2D{pads[1], pads[2], pads[4], pads[5]};
    FailureOr<Value> padded = padConv2DInput(op, input, pads2D, rewriter);
    if (failed(padded))
      return failure();
    input = *padded;
  }

  // OIDHW is likewise contiguous in I,D order. Merge those dimensions before
  // converting the ordinary OIHW filter to TFLite OHWI.
  auto oihwFilterType = RankedTensorType::get(
      {outputChannels, mergedChannels, kernel[1], kernel[2]},
      rewriter.getF32Type());
  Value oihwFilterShape =
      createI32ShapeConstant(rewriter, op.getLoc(), oihwFilterType.getShape());
  Value oihwFilter = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{oihwFilterType}, ValueRange{adaptor.getW(), oihwFilterShape})
                         ->getResult(0);
  auto ohwiFilterType = RankedTensorType::get(
      {outputChannels, kernel[1], kernel[2], mergedChannels},
      rewriter.getF32Type());
  Value filterPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
  Value filter = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{ohwiFilterType}, ValueRange{oihwFilter, filterPermutation})
                     ->getResult(0);

  FailureOr<Value> bias =
      getOrCreateConvBias(op, adaptor.getB(), outputChannels, rewriter);
  if (failed(bias))
    return failure();
  SmallVector<NamedAttribute> attributes = getConv2DAttributes(
      rewriter, dilations[1], dilations[2], padding, strides[1], strides[2]);
  auto nhwcResultType =
      RankedTensorType::get({n, resultShape[3], resultShape[4], outputChannels},
          rewriter.getF32Type());
  Value conv = createTFLOperation(rewriter, op.getLoc(), "tfl.conv_2d",
      TypeRange{nhwcResultType}, ValueRange{input, filter, *bias}, attributes)
                   ->getResult(0);

  auto nchwResultType =
      RankedTensorType::get({n, outputChannels, resultShape[3], resultShape[4]},
          rewriter.getF32Type());
  Value outputPermutation =
      createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
  Value nchwResult = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{nchwResultType}, ValueRange{conv, outputPermutation})
                         ->getResult(0);
  Value outputShape =
      createI32ShapeConstant(rewriter, op.getLoc(), resultShape);
  Value output = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{resultType}, ValueRange{nchwResult, outputShape})
                     ->getResult(0);
  rewriter.replaceOp(op, output);
  return success();
}

LogicalResult lowerConv3D(ONNXConvOp op, ONNXConvOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
  auto filterType = dyn_cast<RankedTensorType>(op.getW().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
  if (!inputType || !filterType || !resultType || inputType.getRank() != 5 ||
      filterType.getRank() != 5 || resultType.getRank() != 5 ||
      failed(validateStaticF32Tensor(op, inputType, "Conv3D input")) ||
      failed(validateStaticF32Tensor(op, filterType, "Conv3D filter")) ||
      failed(validateStaticF32Tensor(op, resultType, "Conv3D result")))
    return op.emitError("Conv3D requires static rank-5 f32 tensors"), failure();

  int64_t group = op.getGroup();
  int64_t inputChannels = inputType.getShape()[1];
  int64_t outputChannels = filterType.getShape()[0];
  if (group <= 0 || inputChannels % group != 0 || outputChannels % group != 0 ||
      filterType.getShape()[1] != inputChannels / group ||
      resultType.getShape()[0] != inputType.getShape()[0] ||
      resultType.getShape()[1] != outputChannels)
    return op.emitError("invalid grouped Conv3D channel dimensions"), failure();
  bool isDepthwise = group == inputChannels && filterType.getShape()[1] == 1;
  bool isGrouped = group != 1 && !isDepthwise;
  int64_t depthMultiplier = 1;
  if (isDepthwise) {
    depthMultiplier = outputChannels / inputChannels;
  }

  SmallVector<int64_t> kernel =
      getI64ArrayOr(op, "kernel_shape", filterType.getShape().take_back(3));
  SmallVector<int64_t> dilations = getI64ArrayOr(op, "dilations", {1, 1, 1});
  SmallVector<int64_t> strides = getI64ArrayOr(op, "strides", {1, 1, 1});
  SmallVector<int64_t> pads = getI64ArrayOr(op, "pads", {0, 0, 0, 0, 0, 0});
  if (kernel.size() != 3 || dilations.size() != 3 || strides.size() != 3 ||
      !llvm::equal(kernel, filterType.getShape().take_back(3)) ||
      llvm::any_of(dilations, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(strides, [](int64_t value) { return value <= 0; }))
    return op.emitError("unsupported Conv3D kernel/stride/dilation"), failure();

  StringRef autoPad = op.getAutoPad();
  if (autoPad == "NOTSET") {
    if (pads.size() != 6 ||
        llvm::any_of(pads, [](int64_t value) { return value < 0; }))
      return op.emitError("unsupported Conv3D explicit padding"), failure();
  } else if (autoPad == "VALID")
    pads.assign(6, 0);
  else if (autoPad == "SAME_LOWER")
    return lowerDirectConv3D(op, adaptor, rewriter);
  else if (autoPad != "SAME_UPPER")
    return op.emitError() << "unsupported Conv3D auto_pad=" << autoPad,
           failure();

  // A patch embedding commonly uses a Conv3D whose depth kernel covers the
  // entire input and produces exactly one depth position. For group=1 this is
  // exactly a Conv2D after merging C and D into the input-channel dimension.
  bool depthPaddingIsZero =
      autoPad != "NOTSET" || (pads[0] == 0 && pads[3] == 0);
  if (group == 1 && resultType.getShape()[2] == 1 && dilations[0] == 1 &&
      kernel[0] == inputType.getShape()[2] && depthPaddingIsZero)
    return lowerFullDepthConv3DAsConv2D(op, adaptor, inputType, filterType,
        resultType, kernel, dilations, strides, pads, rewriter);

  // Prefer an exact rank reduction. Additional Conv3D-to-lower-rank rules can
  // be inserted here; the direct Conv3D path below remains the final fallback.
  std::optional<int64_t> collapsedAxis;
  for (int64_t axis = 0; axis < 3; ++axis) {
    bool paddingPreservesAxis =
        autoPad != "NOTSET" || (pads[axis] == 0 && pads[axis + 3] == 0);
    if (kernel[axis] == 1 && strides[axis] == 1 && paddingPreservesAxis) {
      collapsedAxis = axis;
      break;
    }
  }
  if (!collapsedAxis)
    return lowerDirectConv3D(op, adaptor, rewriter);

  int64_t a = *collapsedAxis;
  SmallVector<int64_t> remaining;
  for (int64_t axis = 0; axis < 3; ++axis)
    if (axis != a)
      remaining.push_back(axis);
  int64_t b = remaining[0];
  int64_t c = remaining[1];
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape[2 + a] != inputShape[2 + a])
    return lowerDirectConv3D(op, adaptor, rewriter);

  int64_t n = inputShape[0];
  int64_t extentA = inputShape[2 + a];
  int64_t extentB = inputShape[2 + b];
  int64_t extentC = inputShape[2 + c];
  SmallVector<int64_t> inputPermutation{0, 2 + a, 2 + b, 2 + c, 1};
  auto orderedInputType = RankedTensorType::get(
      {n, extentA, extentB, extentC, inputChannels}, rewriter.getF32Type());
  Value inputPerm =
      createI32ShapeConstant(rewriter, op.getLoc(), inputPermutation);
  Value orderedInput =
      createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{orderedInputType}, ValueRange{adaptor.getX(), inputPerm})
          ->getResult(0);
  auto input4DType = RankedTensorType::get(
      {n * extentA, extentB, extentC, inputChannels}, rewriter.getF32Type());
  Value input4DShape =
      createI32ShapeConstant(rewriter, op.getLoc(), input4DType.getShape());
  Value input4D = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{input4DType}, ValueRange{orderedInput, input4DShape})
                      ->getResult(0);

  StringRef padding = autoPad == "SAME_UPPER" ? "SAME" : "VALID";
  if (autoPad == "NOTSET") {
    SmallVector<int64_t> pads2D{pads[b], pads[c], pads[b + 3], pads[c + 3]};
    FailureOr<Value> padded = padConv2DInput(op, input4D, pads2D, rewriter);
    if (failed(padded))
      return failure();
    input4D = *padded;
  }

  int64_t kernelB = kernel[b];
  int64_t kernelC = kernel[c];
  SmallVector<int64_t> filterPermutation;
  RankedTensorType orderedFilterType;
  RankedTensorType filter4DType;
  if (!isDepthwise) {
    filterPermutation = {0, 2 + b, 2 + c, 1, 2 + a};
    int64_t inputChannelsPerGroup = filterType.getShape()[1];
    orderedFilterType = RankedTensorType::get(
        {outputChannels, kernelB, kernelC, inputChannelsPerGroup, 1},
        rewriter.getF32Type());
    filter4DType = RankedTensorType::get(
        {outputChannels, kernelB, kernelC, inputChannelsPerGroup},
        rewriter.getF32Type());
  } else {
    filterPermutation = {1, 2 + b, 2 + c, 0, 2 + a};
    orderedFilterType = RankedTensorType::get(
        {1, kernelB, kernelC, outputChannels, 1}, rewriter.getF32Type());
    filter4DType = RankedTensorType::get(
        {1, kernelB, kernelC, outputChannels}, rewriter.getF32Type());
  }
  Value filterPerm =
      createI32ShapeConstant(rewriter, op.getLoc(), filterPermutation);
  Value orderedFilter =
      createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{orderedFilterType}, ValueRange{adaptor.getW(), filterPerm})
          ->getResult(0);
  Value filterShape =
      createI32ShapeConstant(rewriter, op.getLoc(), filter4DType.getShape());
  Value filter4D = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{filter4DType}, ValueRange{orderedFilter, filterShape})
                       ->getResult(0);

  FailureOr<Value> bias =
      getOrCreateConvBias(op, adaptor.getB(), outputChannels, rewriter);
  if (failed(bias))
    return failure();
  SmallVector<NamedAttribute> attributes = getConv2DAttributes(
      rewriter, dilations[b], dilations[c], padding, strides[b], strides[c]);
  StringRef opName = "tfl.conv_2d";
  if (isDepthwise) {
    attributes.push_back(rewriter.getNamedAttr(
        "depth_multiplier", rewriter.getI32IntegerAttr(depthMultiplier)));
    opName = "tfl.depthwise_conv_2d";
  }
  int64_t outputB = resultShape[2 + b];
  int64_t outputC = resultShape[2 + c];
  auto convType = RankedTensorType::get(
      {n * extentA, outputB, outputC, outputChannels}, rewriter.getF32Type());
  Value conv;
  if (isGrouped) {
    FailureOr<Value> grouped = createGroupedConv2D(
        op, input4D, filter4D, *bias, convType, group, attributes, rewriter);
    if (failed(grouped))
      return failure();
    conv = *grouped;
  } else {
    conv = createTFLOperation(rewriter, op.getLoc(), opName,
        TypeRange{convType}, ValueRange{input4D, filter4D, *bias}, attributes)
               ->getResult(0);
  }

  auto orderedOutputType = RankedTensorType::get(
      {n, extentA, outputB, outputC, outputChannels}, rewriter.getF32Type());
  Value orderedOutputShape = createI32ShapeConstant(
      rewriter, op.getLoc(), orderedOutputType.getShape());
  Value orderedOutput = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{orderedOutputType}, ValueRange{conv, orderedOutputShape})
                            ->getResult(0);
  SmallVector<int64_t> outputPermutation{0, 4};
  for (int64_t spatialAxis = 0; spatialAxis < 3; ++spatialAxis) {
    auto position = llvm::find(inputPermutation, 2 + spatialAxis);
    outputPermutation.push_back(
        std::distance(inputPermutation.begin(), position));
  }
  Value outputPerm =
      createI32ShapeConstant(rewriter, op.getLoc(), outputPermutation);
  Value output = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
      TypeRange{resultType}, ValueRange{orderedOutput, outputPerm})
                     ->getResult(0);
  rewriter.replaceOp(op, output);
  return success();
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
    if (inputType && inputType.getRank() == 3)
      return lowerConv1D(op, adaptor, rewriter);
    if (inputType && inputType.getRank() == 5)
      return lowerConv3D(op, adaptor, rewriter);
    if (!inputType || !filterType || !resultType || inputType.getRank() != 4 ||
        filterType.getRank() != 4 || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "Conv input")) ||
        failed(validateStaticF32Tensor(op, filterType, "Conv filter")) ||
        failed(validateStaticF32Tensor(op, resultType, "Conv result"))) {
      op.emitError("ONNXToTFL Conv supports static rank-3/rank-4 f32 tensors "
                   "and reducible static rank-5 f32 tensors only");
      return failure();
    }

    int64_t group = 1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("group"))
      group = attr.getValue().getSExtValue();
    int64_t inputChannels = inputType.getShape()[1];
    int64_t outputChannels = filterType.getShape()[0];
    if (group <= 0 || inputChannels % group != 0 ||
        outputChannels % group != 0 ||
        filterType.getShape()[1] != inputChannels / group)
      return op.emitError("invalid grouped Conv2D channel dimensions"),
             failure();
    bool isDepthwise = group == inputChannels && filterType.getShape()[1] == 1;
    bool isGrouped = group != 1 && !isDepthwise;
    int64_t depthMultiplier = 1;
    if (isDepthwise) {
      if (resultType.getShape()[1] != outputChannels) {
        return op.emitError()
                   << "unsupported grouped Conv: expected depthwise layout "
                      "with group=input_channels and filter [M,1,kH,kW]",
               failure();
      }
      depthMultiplier = outputChannels / inputChannels;
    }

    SmallVector<int64_t> kernel =
        getI64ArrayOr(op, "kernel_shape", filterType.getShape().take_back(2));
    SmallVector<int64_t> dilations = getI64ArrayOr(op, "dilations", {1, 1});
    SmallVector<int64_t> strides = getI64ArrayOr(op, "strides", {1, 1});
    SmallVector<int64_t> pads = getI64ArrayOr(op, "pads", {0, 0, 0, 0});
    if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2)
      return op.emitError("unsupported Conv configuration: only 2D Conv is "
                          "supported"),
             failure();
    if (llvm::any_of(dilations, [](int64_t value) { return value <= 0; }))
      return op.emitError("unsupported Conv configuration: dilations must be "
                          "positive"),
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

    Type convertedResultType = convertRank4NCHWToNHWCType(resultType);
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
    Value filter = adaptor.getOperands()[1];
    StringRef tflOpName = "tfl.conv_2d";
    if (isDepthwise) {
      // Generic rank-4 Constant lowering maps ONNX OIHW to OHWI. TFL
      // depthwise filters instead use [1,H,W,input_channels*multiplier].
      // Convert OHWI [M,H,W,1] to [1,H,W,M]; TensorFlow folds this transpose
      // when the filter is constant.
      auto convertedFilterType = cast<RankedTensorType>(filter.getType());
      ArrayRef<int64_t> filterShape = convertedFilterType.getShape();
      auto depthwiseFilterType = RankedTensorType::get(
          {1, filterShape[1], filterShape[2], filterShape[0]},
          convertedFilterType.getElementType());
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {3, 1, 2, 0});
      filter = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{depthwiseFilterType}, ValueRange{filter, permutation})
                   ->getResult(0);
      attributes.push_back(rewriter.getNamedAttr(
          "depth_multiplier", rewriter.getI32IntegerAttr(depthMultiplier)));
      tflOpName = "tfl.depthwise_conv_2d";
    }
    if (isGrouped) {
      FailureOr<Value> grouped = createGroupedConv2D(op, input, filter, bias,
          cast<RankedTensorType>(convertedResultType), group, attributes,
          rewriter);
      if (failed(grouped))
        return failure();
      rewriter.replaceOp(op, *grouped);
    } else {
      SmallVector<Value> operands{input, filter, bias};
      Operation *newOp = createTFLOperation(rewriter, op.getLoc(), tflOpName,
          TypeRange{convertedResultType}, operands, attributes);
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};

} // namespace

void populateLoweringONNXConvOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConvLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
