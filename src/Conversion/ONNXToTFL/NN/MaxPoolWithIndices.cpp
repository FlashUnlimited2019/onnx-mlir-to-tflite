/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getIntegerArray(
    Operation *op, StringRef name, ArrayRef<int64_t> defaults = {}) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaults);
  SmallVector<int64_t> result;
  for (Attribute value : attr)
    result.push_back(cast<IntegerAttr>(value).getValue().getSExtValue());
  return result;
}

class StaticNon2DMaxPoolLowering final
    : public OpConversionPattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getRank() < 3 ||
        inputType.getRank() > 5 || inputType.getRank() == 4 ||
        !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return failure();
    if (op.getCeilMode() != 0 || op.getStorageOrder() != 0 ||
        op.getAutoPad() != "NOTSET")
      return op.emitError("non-2D MaxPool requires floor mode, storage_order "
                          "0, and auto_pad NOTSET"),
             failure();
    int64_t spatialRank = inputType.getRank() - 2;
    SmallVector<int64_t> kernel = getIntegerArray(op, "kernel_shape");
    SmallVector<int64_t> strides =
        getIntegerArray(op, "strides", SmallVector<int64_t>(spatialRank, 1));
    SmallVector<int64_t> pads =
        getIntegerArray(op, "pads", SmallVector<int64_t>(2 * spatialRank, 0));
    SmallVector<int64_t> dilations = getIntegerArray(
        op, "dilations", SmallVector<int64_t>(spatialRank, 1));
    if (kernel.size() != static_cast<size_t>(spatialRank) ||
        strides.size() != static_cast<size_t>(spatialRank) ||
        pads.size() != static_cast<size_t>(2 * spatialRank) ||
        dilations.size() != static_cast<size_t>(spatialRank) ||
        llvm::any_of(pads, [](int64_t value) { return value != 0; }) ||
        llvm::any_of(dilations, [](int64_t value) { return value != 1; }))
      return op.emitError("non-2D MaxPool currently requires zero padding and "
                          "unit dilation"),
             failure();

    int64_t candidates = 1;
    for (int64_t extent : kernel)
      candidates *= extent;
    SmallVector<NamedAttribute> sliceAttributes{
        rewriter.getNamedAttr("begin_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("ellipsis_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("end_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("new_axis_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("offset", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr(
            "shrink_axis_mask", rewriter.getI32IntegerAttr(0))};
    Value result;
    for (int64_t candidate = 0; candidate < candidates; ++candidate) {
      int64_t code = candidate;
      SmallVector<int64_t> offset(spatialRank);
      for (int64_t axis = spatialRank - 1; axis >= 0; --axis) {
        offset[axis] = code % kernel[axis];
        code /= kernel[axis];
      }
      SmallVector<int64_t> begin(inputType.getRank(), 0);
      SmallVector<int64_t> end(inputType.getShape());
      SmallVector<int64_t> stride(inputType.getRank(), 1);
      for (int64_t axis = 0; axis < spatialRank; ++axis) {
        begin[axis + 2] = offset[axis];
        stride[axis + 2] = strides[axis];
        end[axis + 2] = offset[axis] +
                        (resultType.getShape()[axis + 2] - 1) * strides[axis] +
                        1;
      }
      Value candidateValue = createTFLOperation(rewriter, op.getLoc(),
          "tfl.strided_slice", TypeRange{resultType},
          ValueRange{adaptor.getX(),
              createI32ShapeConstant(rewriter, op.getLoc(), begin),
              createI32ShapeConstant(rewriter, op.getLoc(), end),
              createI32ShapeConstant(rewriter, op.getLoc(), stride)},
          sliceAttributes)
                                 ->getResult(0);
      if (!result) {
        result = candidateValue;
      } else {
        result = createTFLOperation(rewriter, op.getLoc(), "tfl.maximum",
            TypeRange{resultType}, ValueRange{result, candidateValue})
                     ->getResult(0);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MaxPoolWithIndicesLowering final
    : public OpConversionPattern<ONNXMaxPoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMaxPoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto valueType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    if (!inputType || !valueType || !indicesType ||
        !inputType.hasStaticShape() || !valueType.hasStaticShape() ||
        !indicesType.hasStaticShape() || inputType.getRank() < 3 ||
        inputType.getRank() > 5 || !inputType.getElementType().isF32() ||
        !valueType.getElementType().isF32() ||
        !indicesType.getElementType().isSignlessInteger(64))
      return op.emitError("ONNXToTFL indexed MaxPool supports static rank-3/4/5 "
                          "f32 tensors with i64 indices"),
             failure();
    if (op.getCeilMode() != 0 || op.getStorageOrder() != 0 ||
        op.getAutoPad() != "NOTSET")
      return op.emitError("indexed MaxPool requires floor mode, storage_order "
                          "0, and auto_pad NOTSET"),
             failure();
    int64_t spatialRank = inputType.getRank() - 2;
    SmallVector<int64_t> kernel =
        getIntegerArray(op, "kernel_shape");
    SmallVector<int64_t> strides =
        getIntegerArray(op, "strides", SmallVector<int64_t>(spatialRank, 1));
    SmallVector<int64_t> pads =
        getIntegerArray(op, "pads", SmallVector<int64_t>(2 * spatialRank, 0));
    SmallVector<int64_t> dilations = getIntegerArray(
        op, "dilations", SmallVector<int64_t>(spatialRank, 1));
    if (kernel.size() != static_cast<size_t>(spatialRank) ||
        strides.size() != static_cast<size_t>(spatialRank) ||
        pads.size() != static_cast<size_t>(2 * spatialRank) ||
        dilations.size() != static_cast<size_t>(spatialRank) ||
        llvm::any_of(pads, [](int64_t value) { return value != 0; }) ||
        llvm::any_of(dilations, [](int64_t value) { return value != 1; }))
      return op.emitError("indexed MaxPool currently requires zero padding "
                          "and unit dilation"),
             failure();

    Value input = adaptor.getX();
    if (inputType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{inputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    int64_t candidates = 1;
    for (int64_t extent : kernel)
      candidates *= extent;
    Value bestValue;
    Value bestIndex;
    SmallVector<NamedAttribute> sliceAttributes{
        rewriter.getNamedAttr("begin_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("ellipsis_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("end_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("new_axis_mask", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("offset", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr(
            "shrink_axis_mask", rewriter.getI32IntegerAttr(0))};
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = valueType.getShape();
    for (int64_t candidate = 0; candidate < candidates; ++candidate) {
      int64_t code = candidate;
      SmallVector<int64_t> offset(spatialRank);
      for (int64_t axis = spatialRank - 1; axis >= 0; --axis) {
        offset[axis] = code % kernel[axis];
        code /= kernel[axis];
      }
      SmallVector<int64_t> begin(inputType.getRank(), 0);
      SmallVector<int64_t> end(inputShape);
      SmallVector<int64_t> stride(inputType.getRank(), 1);
      for (int64_t axis = 0; axis < spatialRank; ++axis) {
        begin[axis + 2] = offset[axis];
        stride[axis + 2] = strides[axis];
        end[axis + 2] = offset[axis] +
                        (outputShape[axis + 2] - 1) * strides[axis] + 1;
      }
      Value beginValue = createI32ShapeConstant(rewriter, op.getLoc(), begin);
      Value endValue = createI32ShapeConstant(rewriter, op.getLoc(), end);
      Value strideValue =
          createI32ShapeConstant(rewriter, op.getLoc(), stride);
      Value candidateValue = createTFLOperation(rewriter, op.getLoc(),
          "tfl.strided_slice", TypeRange{valueType},
          ValueRange{input, beginValue, endValue, strideValue},
          sliceAttributes)
                                 ->getResult(0);

      SmallVector<int64_t> flatIndices;
      flatIndices.reserve(valueType.getNumElements());
      for (int64_t linear = 0; linear < valueType.getNumElements(); ++linear) {
        int64_t remainder = linear;
        SmallVector<int64_t> coordinates(valueType.getRank());
        for (int64_t axis = valueType.getRank() - 1; axis >= 0; --axis) {
          coordinates[axis] = remainder % outputShape[axis];
          remainder /= outputShape[axis];
        }
        int64_t inputLinear = 0;
        for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
          int64_t coordinate = coordinates[axis];
          if (axis >= 2)
            coordinate = coordinate * strides[axis - 2] + offset[axis - 2];
          inputLinear = inputLinear * inputShape[axis] + coordinate;
        }
        flatIndices.push_back(inputLinear);
      }
      Value candidateIndex = arith::ConstantOp::create(rewriter, op.getLoc(),
          indicesType, DenseIntElementsAttr::get(indicesType, flatIndices));
      if (!bestValue) {
        bestValue = candidateValue;
        bestIndex = candidateIndex;
        continue;
      }
      auto conditionType = RankedTensorType::get(
          valueType.getShape(), rewriter.getI1Type());
      Value better = createTFLOperation(rewriter, op.getLoc(), "tfl.greater",
          TypeRange{conditionType}, ValueRange{candidateValue, bestValue})
                         ->getResult(0);
      bestValue = createTFLOperation(rewriter, op.getLoc(), "tfl.select_v2",
          TypeRange{valueType}, ValueRange{better, candidateValue, bestValue})
                      ->getResult(0);
      bestIndex = createTFLOperation(rewriter, op.getLoc(), "tfl.select_v2",
          TypeRange{indicesType}, ValueRange{better, candidateIndex, bestIndex})
                      ->getResult(0);
    }
    if (valueType.getRank() == 4) {
      Type physicalValueType = convertRank4NCHWToNHWCType(valueType);
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      bestValue = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalValueType}, ValueRange{bestValue, permutation})
                      ->getResult(0);
    }
    rewriter.replaceOp(op, ValueRange{bestValue, bestIndex});
    return success();
  }
};

class MaxUnpoolLowering final
    : public OpConversionPattern<ONNXMaxUnpoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMaxUnpoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto valueType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getI().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!valueType || !indicesType || !resultType ||
        !valueType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !resultType.hasStaticShape() || valueType.getRank() < 3 ||
        valueType.getRank() > 5 || !valueType.getElementType().isF32() ||
        !indicesType.getElementType().isSignlessInteger(64) ||
        !resultType.getElementType().isF32() ||
        !isa<NoneType>(op.getOutputShape().getType()))
      return op.emitError("ONNXToTFL MaxUnpool supports static rank-3/4/5 f32 "
                          "values, i64 indices, and inferred output shape"),
             failure();
    int64_t updateCount = valueType.getNumElements();
    int64_t outputCount = resultType.getNumElements();
    Value values = adaptor.getX();
    if (valueType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      values = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{valueType}, ValueRange{values, permutation})
                   ->getResult(0);
    }
    auto flatValuesType =
        RankedTensorType::get({updateCount}, rewriter.getF32Type());
    auto reshapedIndicesType =
        RankedTensorType::get({updateCount, 1}, rewriter.getI64Type());
    auto scatterIndicesType =
        RankedTensorType::get({updateCount, 1}, rewriter.getI32Type());
    Value flatShape =
        createI32ShapeConstant(rewriter, op.getLoc(), {updateCount});
    Value scatterIndexShape =
        createI32ShapeConstant(rewriter, op.getLoc(), {updateCount, 1});
    Value flatValues = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{flatValuesType}, ValueRange{values, flatShape})
                           ->getResult(0);
    Value reshapedIndices = createTFLOperation(rewriter, op.getLoc(),
        "tfl.reshape", TypeRange{reshapedIndicesType},
        ValueRange{adaptor.getI(), scatterIndexShape})
                               ->getResult(0);
    Value scatterIndices = createTFLOperation(rewriter, op.getLoc(),
        "tfl.cast", TypeRange{scatterIndicesType},
        ValueRange{reshapedIndices})
                               ->getResult(0);
    Value outputShape =
        createI32ShapeConstant(rewriter, op.getLoc(), {outputCount});
    auto flatOutputType =
        RankedTensorType::get({outputCount}, rewriter.getF32Type());
    Value flatOutput = createTFLOperation(rewriter, op.getLoc(),
        "tfl.scatter_nd", TypeRange{flatOutputType},
        ValueRange{scatterIndices, flatValues, outputShape})
                           ->getResult(0);
    Value logicalShape = createI32ShapeConstant(
        rewriter, op.getLoc(), resultType.getShape());
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{resultType}, ValueRange{flatOutput, logicalShape})
                       ->getResult(0);
    if (resultType.getRank() == 4) {
      Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXIndexedMaxPoolOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<StaticNon2DMaxPoolLowering, MaxPoolWithIndicesLowering,
      MaxUnpoolLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
