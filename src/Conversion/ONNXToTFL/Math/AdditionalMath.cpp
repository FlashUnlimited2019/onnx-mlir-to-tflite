/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

Value restoreLogicalRank4(ConversionPatternRewriter &rewriter, Location loc,
    Value value, RankedTensorType sourceType) {
  if (sourceType.getRank() != 4 || !sourceType.getElementType().isF32())
    return value;
  Value permutation = createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2});
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{sourceType}, ValueRange{value, permutation})
      ->getResult(0);
}

Value makePhysicalRank4(ConversionPatternRewriter &rewriter, Location loc,
    Value value, RankedTensorType sourceType) {
  if (sourceType.getRank() != 4 || !sourceType.getElementType().isF32())
    return value;
  Type physicalType = convertRank4NCHWToNHWCType(sourceType);
  Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{physicalType}, ValueRange{value, permutation})
      ->getResult(0);
}

class CumSumLowering final : public OpConversionPattern<ONNXCumSumOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXCumSumOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    FailureOr<SmallVector<int64_t>> axes = getConstantIntValues(op.getAxis());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        inputType.getElementType() != resultType.getElementType() ||
        (!inputType.getElementType().isF32() &&
            !inputType.getElementType().isSignlessInteger(32) &&
            !inputType.getElementType().isSignlessInteger(64)) ||
        failed(axes) || axes->size() != 1)
      return op.emitError("ONNXToTFL CumSum requires a static f32/i32/i64 "
                          "tensor and one constant axis"),
             failure();
    int64_t axis = normalizeAxis((*axes)[0], inputType.getRank());
    if (axis < 0 || axis >= inputType.getRank())
      return op.emitError("CumSum axis is out of range"), failure();
    if (inputType.getRank() == 4 && inputType.getElementType().isF32())
      axis = mapNCHWAxisToNHWC(axis);
    Value axisValue = createI32ScalarTensorConstant(
        rewriter, op.getLoc(), static_cast<int32_t>(axis));
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr(
            "exclusive", rewriter.getBoolAttr(op.getExclusive() != 0)),
        rewriter.getNamedAttr(
            "reverse", rewriter.getBoolAttr(op.getReverse() != 0))};
    Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.cumsum",
        TypeRange{physicalResultType}, ValueRange{adaptor.getX(), axisValue},
        attributes)
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class OneHotLowering final : public OpConversionPattern<ONNXOneHotOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXOneHotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    FailureOr<SmallVector<int64_t>> depth = getConstantIntValues(op.getDepth());
    Type valueElementType;
    if (auto valuesType = dyn_cast<RankedTensorType>(op.getValues().getType()))
      valueElementType = valuesType.getElementType();
    bool floatingValues = valueElementType && valueElementType.isF32();
    bool integerValues =
        valueElementType && (valueElementType.isSignlessInteger(32) ||
                                valueElementType.isSignlessInteger(64));
    if (!indicesType || !resultType || !indicesType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        (!indicesType.getElementType().isSignlessInteger(32) &&
            !indicesType.getElementType().isSignlessInteger(64)) ||
        resultType.getElementType() != valueElementType || failed(depth) ||
        depth->size() != 1 || (!floatingValues && !integerValues))
      return op.emitError("ONNXToTFL OneHot requires static integer indices, "
                          "constant depth and two constant f32/i32/i64 "
                          "values"),
             failure();
    Value depthValue = createI32ScalarTensorConstant(
        rewriter, op.getLoc(), static_cast<int32_t>((*depth)[0]));
    Value offValue;
    Value onValue;
    if (floatingValues) {
      FailureOr<SmallVector<float>> values =
          getConstantF32Values(op.getValues());
      if (failed(values) || values->size() != 2)
        return op.emitError("OneHot values must be a two-element constant"),
               failure();
      offValue =
          createF32ScalarTensorConstant(rewriter, op.getLoc(), (*values)[0]);
      onValue =
          createF32ScalarTensorConstant(rewriter, op.getLoc(), (*values)[1]);
    } else {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getValues());
      if (failed(values) || values->size() != 2)
        return op.emitError("OneHot values must be a two-element constant"),
               failure();
      auto scalarType = RankedTensorType::get({}, valueElementType);
      if (valueElementType.isSignlessInteger(64)) {
        offValue = arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
            DenseIntElementsAttr::get(
                scalarType, ArrayRef<int64_t>{(*values)[0]}));
        onValue = arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
            DenseIntElementsAttr::get(
                scalarType, ArrayRef<int64_t>{(*values)[1]}));
      } else {
        offValue = arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
            DenseIntElementsAttr::get(scalarType,
                ArrayRef<int32_t>{static_cast<int32_t>((*values)[0])}));
        onValue = arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
            DenseIntElementsAttr::get(scalarType,
                ArrayRef<int32_t>{static_cast<int32_t>((*values)[1])}));
      }
    }
    int64_t axis = normalizeAxis(op.getAxis(), indicesType.getRank() + 1);
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "axis", rewriter.getI32IntegerAttr(static_cast<int32_t>(axis)))};
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.one_hot",
        TypeRange{resultType},
        ValueRange{adaptor.getIndices(), depthValue, onValue, offValue},
        attributes)
                       ->getResult(0);
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, op.getLoc(), result, resultType));
    return success();
  }
};

class LpNormalizationLowering final
    : public OpConversionPattern<ONNXLpNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXLpNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        inputType != resultType || inputType.getRank() < 1)
      return op.emitError(
                 "ONNXToTFL LpNormalization requires matching static f32 "
                 "input/result tensors"),
             failure();
    if (op.getP() != 1 && op.getP() != 2)
      return op.emitError("ONNXToTFL LpNormalization supports p=1 or p=2"),
             failure();

    int64_t rank = inputType.getRank();
    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError("LpNormalization axis is out of range"), failure();

    SmallVector<int64_t> normShape(inputType.getShape());
    normShape[axis] = 1;
    auto logicalNormType =
        RankedTensorType::get(normShape, inputType.getElementType());
    auto physicalInputType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(inputType));
    auto physicalNormType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(logicalNormType));
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    if (rank == 4)
      axis = mapNCHWAxisToNHWC(axis);

    Location loc = op.getLoc();
    Value magnitude;
    if (op.getP() == 1) {
      magnitude = createTFLOperation(rewriter, loc, "tfl.abs",
          TypeRange{physicalInputType}, ValueRange{adaptor.getInput()})
                      ->getResult(0);
    } else {
      SmallVector<NamedAttribute> mulAttributes{rewriter.getNamedAttr(
          "fused_activation_function", rewriter.getStringAttr("NONE"))};
      magnitude = createTFLOperation(rewriter, loc, "tfl.mul",
          TypeRange{physicalInputType},
          ValueRange{adaptor.getInput(), adaptor.getInput()}, mulAttributes)
                      ->getResult(0);
    }
    Value axisValue = createI32ShapeConstant(rewriter, loc, {axis});
    SmallVector<NamedAttribute> sumAttributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    Value norm = createTFLOperation(rewriter, loc, "tfl.sum",
        TypeRange{physicalNormType}, ValueRange{magnitude, axisValue},
        sumAttributes)
                     ->getResult(0);
    if (op.getP() == 2)
      norm = createTFLOperation(rewriter, loc, "tfl.sqrt",
          TypeRange{physicalNormType}, ValueRange{norm})
                 ->getResult(0);

    Value zero = arith::ConstantOp::create(rewriter, loc, physicalNormType,
        DenseElementsAttr::get(physicalNormType, 0.0f));
    Value one = arith::ConstantOp::create(rewriter, loc, physicalNormType,
        DenseElementsAttr::get(physicalNormType, 1.0f));
    auto conditionType = RankedTensorType::get(
        physicalNormType.getShape(), rewriter.getI1Type());
    Value normIsZero = createTFLOperation(rewriter, loc, "tfl.equal",
        TypeRange{conditionType}, ValueRange{norm, zero})
                           ->getResult(0);
    Value safeNorm = createTFLOperation(rewriter, loc, "tfl.select_v2",
        TypeRange{physicalNormType}, ValueRange{normIsZero, one, norm})
                         ->getResult(0);
    SmallVector<NamedAttribute> divAttributes{rewriter.getNamedAttr(
        "fused_activation_function", rewriter.getStringAttr("NONE"))};
    Value normalized = createTFLOperation(rewriter, loc, "tfl.div",
        TypeRange{physicalResultType}, ValueRange{adaptor.getInput(), safeNorm},
        divAttributes)
                           ->getResult(0);
    rewriter.replaceOp(op, normalized);
    return success();
  }
};

enum class StaticReductionKind { L1, LogSumExp, Prod, SumSquare };

template <typename ONNXOp, StaticReductionKind kind>
class StaticReductionLowering final : public OpConversionPattern<ONNXOp> {
public:
  using OpConversionPattern<ONNXOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXOp::Adaptor;

  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getReduced().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || inputType.getRank() < 1)
      return op.emitError(
                 "ONNXToTFL reduction requires static ranked f32 tensors"),
             failure();

    bool axesAbsent = isa<NoneType>(op.getAxes().getType());
    SmallVector<int64_t> axes;
    if (!axesAbsent) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getAxes());
      if (failed(values))
        return op.emitError("ONNXToTFL reduction requires constant axes"),
               failure();
      axes = std::move(*values);
    }
    if ((axesAbsent || axes.empty()) && op.getNoopWithEmptyAxes() != 0) {
      rewriter.replaceOp(op, adaptor.getData());
      return success();
    }
    int64_t rank = inputType.getRank();
    if (axesAbsent || axes.empty())
      for (int64_t axis = 0; axis < rank; ++axis)
        axes.push_back(axis);
    SmallVector<bool> seen(rank, false);
    for (int64_t &rawAxis : axes) {
      int64_t axis = normalizeAxis(rawAxis, rank);
      if (axis < 0 || axis >= rank || seen[axis])
        return op.emitError() << "invalid reduction axis " << rawAxis,
               failure();
      seen[axis] = true;
      rawAxis = axis;
    }

    Location loc = op.getLoc();
    bool keepDims = op.getKeepdims() != 0;
    Value input =
        restoreLogicalRank4(rewriter, loc, adaptor.getData(), inputType);
    Value axisValue = createI32ShapeConstant(rewriter, loc, axes);
    SmallVector<NamedAttribute> resultReductionAttributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    SmallVector<NamedAttribute> binaryAttributes{rewriter.getNamedAttr(
        "fused_activation_function", rewriter.getStringAttr("NONE"))};

    Value result;
    if constexpr (kind == StaticReductionKind::L1) {
      Value magnitude = createTFLOperation(
          rewriter, loc, "tfl.abs", TypeRange{inputType}, ValueRange{input})
                            ->getResult(0);
      result =
          createTFLOperation(rewriter, loc, "tfl.sum", TypeRange{resultType},
              ValueRange{magnitude, axisValue}, resultReductionAttributes)
              ->getResult(0);
    } else if constexpr (kind == StaticReductionKind::Prod) {
      result = createTFLOperation(rewriter, loc, "tfl.reduce_prod",
          TypeRange{resultType}, ValueRange{input, axisValue},
          resultReductionAttributes)
                   ->getResult(0);
    } else if constexpr (kind == StaticReductionKind::SumSquare) {
      Value squared = createTFLOperation(rewriter, loc, "tfl.mul",
          TypeRange{inputType}, ValueRange{input, input}, binaryAttributes)
                          ->getResult(0);
      result =
          createTFLOperation(rewriter, loc, "tfl.sum", TypeRange{resultType},
              ValueRange{squared, axisValue}, resultReductionAttributes)
              ->getResult(0);
    } else {
      SmallVector<int64_t> keepShape(inputType.getShape());
      for (int64_t axis : axes)
        keepShape[axis] = 1;
      auto keepType =
          RankedTensorType::get(keepShape, inputType.getElementType());
      SmallVector<NamedAttribute> keepReductionAttributes{
          rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
      Value maximum = createTFLOperation(rewriter, loc, "tfl.reduce_max",
          TypeRange{keepType}, ValueRange{input, axisValue},
          keepReductionAttributes)
                          ->getResult(0);
      Value centered = createTFLOperation(rewriter, loc, "tfl.sub",
          TypeRange{inputType}, ValueRange{input, maximum}, binaryAttributes)
                           ->getResult(0);
      Value exponentiated = createTFLOperation(
          rewriter, loc, "tfl.exp", TypeRange{inputType}, ValueRange{centered})
                                ->getResult(0);
      Value sum =
          createTFLOperation(rewriter, loc, "tfl.sum", TypeRange{keepType},
              ValueRange{exponentiated, axisValue}, keepReductionAttributes)
              ->getResult(0);
      Value logarithm = createTFLOperation(
          rewriter, loc, "tfl.log", TypeRange{keepType}, ValueRange{sum})
                            ->getResult(0);
      Value stabilized = createTFLOperation(rewriter, loc, "tfl.add",
          TypeRange{keepType}, ValueRange{logarithm, maximum}, binaryAttributes)
                             ->getResult(0);
      if (keepDims) {
        result = stabilized;
      } else {
        Value resultShape =
            createI32ShapeConstant(rewriter, loc, resultType.getShape());
        result = createTFLOperation(rewriter, loc, "tfl.reshape",
            TypeRange{resultType}, ValueRange{stabilized, resultShape})
                     ->getResult(0);
      }
    }

    result = makePhysicalRank4(rewriter, loc, result, resultType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ONNXOp>
class ReduceAttributeAxesLowering final : public OpConversionPattern<ONNXOp> {
public:
  ReduceAttributeAxesLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}
  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return op.emitError("ONNXToTFL reduction requires static f32 tensors"),
             failure();
    SmallVector<int64_t> axes;
    if (auto attr = op->template getAttrOfType<ArrayAttr>("axes")) {
      for (Attribute element : attr)
        axes.push_back(
            normalizeAxis(cast<IntegerAttr>(element).getValue().getSExtValue(),
                inputType.getRank()));
    } else {
      for (int64_t axis = 0; axis < inputType.getRank(); ++axis)
        axes.push_back(axis);
    }
    bool keepDims = op->template getAttrOfType<IntegerAttr>("keepdims")
                        .getValue()
                        .getSExtValue() != 0;
    Value input = restoreLogicalRank4(
        rewriter, op.getLoc(), adaptor.getOperands().front(), inputType);
    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
    Value result = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, ValueRange{input, axisValue}, attributes)
                       ->getResult(0);
    result = makePhysicalRank4(rewriter, op.getLoc(), result, resultType);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  std::string tflName;
};

class ReduceMaxInputAxesLowering final
    : public OpConversionPattern<ONNXReduceMaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReduceMaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getReduced().getType());
    FailureOr<SmallVector<int64_t>> sourceAxes =
        getConstantIntValues(op.getAxes());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || failed(sourceAxes))
      return op.emitError("ONNXToTFL ReduceMax requires static f32 tensors "
                          "and constant axes"),
             failure();
    SmallVector<int64_t> axes;
    for (int64_t axis : *sourceAxes)
      axes.push_back(normalizeAxis(axis, inputType.getRank()));
    if (axes.empty() && op.getNoopWithEmptyAxes() != 0) {
      rewriter.replaceOp(op, adaptor.getData());
      return success();
    }
    if (axes.empty())
      for (int64_t axis = 0; axis < inputType.getRank(); ++axis)
        axes.push_back(axis);
    Value input = restoreLogicalRank4(
        rewriter, op.getLoc(), adaptor.getData(), inputType);
    Value axisValue = createI32ShapeConstant(rewriter, op.getLoc(), axes);
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "keep_dims", rewriter.getBoolAttr(op.getKeepdims() != 0))};
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.reduce_max",
        TypeRange{resultType}, ValueRange{input, axisValue}, attributes)
                       ->getResult(0);
    result = makePhysicalRank4(rewriter, op.getLoc(), result, resultType);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class IsNaNLowering final : public OpConversionPattern<ONNXIsNaNOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXIsNaNOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isInteger(1))
      return failure();
    Value input =
        restoreLogicalRank4(rewriter, op.getLoc(), adaptor.getX(), inputType);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.not_equal",
        TypeRange{resultType}, ValueRange{input, input})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class IsInfLowering final : public OpConversionPattern<ONNXIsInfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXIsInfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isInteger(1))
      return failure();
    Value input =
        restoreLogicalRank4(rewriter, op.getLoc(), adaptor.getX(), inputType);
    auto comparisonConstant = [&](float value) -> Value {
      if (inputType.getRank() <= 4)
        return createF32ScalarTensorConstant(rewriter, op.getLoc(), value);
      return arith::ConstantOp::create(rewriter, op.getLoc(), inputType,
          DenseElementsAttr::get(inputType, value));
    };
    // The TFLite comparison kernels implement scalar broadcasting only up to
    // rank 4. Materialize rank-5 constants to keep this static path builtin
    // and avoid a runtime abort in BroadcastComparison4DSlow.
    Value maximum = comparisonConstant(std::numeric_limits<float>::max());
    Value minimum = comparisonConstant(-std::numeric_limits<float>::max());
    Value positive = createTFLOperation(rewriter, op.getLoc(), "tfl.greater",
        TypeRange{resultType}, ValueRange{input, maximum})
                         ->getResult(0);
    Value negative = createTFLOperation(rewriter, op.getLoc(), "tfl.less",
        TypeRange{resultType}, ValueRange{input, minimum})
                         ->getResult(0);
    Value result;
    if (op.getDetectNegative() != 0 && op.getDetectPositive() != 0)
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.logical_or",
          TypeRange{resultType}, ValueRange{negative, positive})
                   ->getResult(0);
    else
      result = op.getDetectNegative() != 0 ? negative : positive;
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXAdditionalMathOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CumSumLowering, OneHotLowering, LpNormalizationLowering,
      ReduceMaxInputAxesLowering, IsNaNLowering, IsInfLowering>(
      typeConverter, context);
  patterns.add<StaticReductionLowering<ONNXReduceL1Op, StaticReductionKind::L1>,
      StaticReductionLowering<ONNXReduceLogSumExpOp,
          StaticReductionKind::LogSumExp>,
      StaticReductionLowering<ONNXReduceProdOp, StaticReductionKind::Prod>,
      StaticReductionLowering<ONNXReduceSumSquareOp,
          StaticReductionKind::SumSquare>>(typeConverter, context);
  patterns.add<ReduceAttributeAxesLowering<ONNXReduceMinV13Op>>(
      typeConverter, context, "tfl.reduce_min");
}

} // namespace onnx_mlir
