/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

Value unary(ConversionPatternRewriter &rewriter, Location loc, StringRef name,
    Type type, Value input) {
  return createTFLOperation(
      rewriter, loc, name, TypeRange{type}, ValueRange{input})
      ->getResult(0);
}

Value binary(ConversionPatternRewriter &rewriter, Location loc, StringRef name,
    Type type, Value lhs, Value rhs, bool hasFusedActivation = true) {
  SmallVector<NamedAttribute> attributes;
  if (hasFusedActivation)
    attributes.push_back(getFusedActivationNone(rewriter));
  return createTFLOperation(
      rewriter, loc, name, TypeRange{type}, ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

enum class CompositeKind {
  Celu,
  Selu,
  Softplus,
  Softsign,
  Shrink,
  ThresholdedRelu,
  Asin,
  Acos,
  Atan,
  Atanh,
  Sinh,
  Cosh,
  Asinh,
  Acosh,
  Erf,
};

template <typename ONNXOp>
class CompositeUnaryLowering final : public OpConversionPattern<ONNXOp> {
public:
  CompositeUnaryLowering(
      TypeConverter &typeConverter, MLIRContext *context, CompositeKind kind)
      : OpConversionPattern<ONNXOp>(typeConverter, context), kind(kind) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands().front();
    auto sourceResultType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!sourceResultType ||
        failed(validateStaticF32TensorOrScalar(
            op, op->getOperand(0).getType(), "input")) ||
        failed(validateStaticF32TensorOrScalar(op, sourceResultType, "result")))
      return failure();
    auto resultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(sourceResultType));
    Location loc = op.getLoc();
    auto scalar = [&](float value) {
      return createF32ScalarTensorConstant(rewriter, loc, value);
    };
    auto comparisonConstant = [&](float value) -> Value {
      Value scalarValue = scalar(value);
      if (resultType.getRank() <= 4)
        return scalarValue;
      Value shape =
          createI32ShapeConstant(rewriter, loc, resultType.getShape());
      return createTFLOperation(rewriter, loc, "tfl.fill",
          TypeRange{resultType}, ValueRange{shape, scalarValue})
          ->getResult(0);
    };
    auto add = [&](Value lhs, Value rhs) {
      return binary(rewriter, loc, "tfl.add", resultType, lhs, rhs);
    };
    auto sub = [&](Value lhs, Value rhs) {
      return binary(rewriter, loc, "tfl.sub", resultType, lhs, rhs);
    };
    auto mul = [&](Value lhs, Value rhs) {
      return binary(rewriter, loc, "tfl.mul", resultType, lhs, rhs);
    };
    auto div = [&](Value lhs, Value rhs) {
      return binary(rewriter, loc, "tfl.div", resultType, lhs, rhs);
    };

    Value zero = scalar(0.0f);
    Value one = scalar(1.0f);
    Value result;
    switch (kind) {
    case CompositeKind::Celu: {
      float alpha =
          op->template getAttrOfType<FloatAttr>("alpha").getValueAsDouble();
      Value alphaValue = scalar(alpha);
      Value positive = binary(rewriter, loc, "tfl.maximum", resultType, input,
          zero, /*hasFusedActivation=*/false);
      Value normalized = div(input, alphaValue);
      Value negative = mul(alphaValue,
          sub(unary(rewriter, loc, "tfl.exp", resultType, normalized), one));
      negative = binary(rewriter, loc, "tfl.minimum", resultType, negative,
          zero, /*hasFusedActivation=*/false);
      result = add(positive, negative);
      break;
    }
    case CompositeKind::Selu: {
      float alpha =
          op->template getAttrOfType<FloatAttr>("alpha").getValueAsDouble();
      float gamma =
          op->template getAttrOfType<FloatAttr>("gamma").getValueAsDouble();
      Value positive = binary(rewriter, loc, "tfl.maximum", resultType, input,
          zero, /*hasFusedActivation=*/false);
      Value negative = mul(scalar(alpha),
          sub(unary(rewriter, loc, "tfl.exp", resultType, input), one));
      negative = binary(rewriter, loc, "tfl.minimum", resultType, negative,
          zero, /*hasFusedActivation=*/false);
      result = mul(scalar(gamma), add(positive, negative));
      break;
    }
    case CompositeKind::Softplus: {
      // max(x, 0) + log(1 + exp(-abs(x))) avoids overflow for large x.
      Value positive = binary(rewriter, loc, "tfl.maximum", resultType, input,
          zero, /*hasFusedActivation=*/false);
      Value absolute = unary(rewriter, loc, "tfl.abs", resultType, input);
      Value negative = unary(rewriter, loc, "tfl.neg", resultType, absolute);
      Value tail = unary(rewriter, loc, "tfl.log", resultType,
          add(one, unary(rewriter, loc, "tfl.exp", resultType, negative)));
      result = add(positive, tail);
      break;
    }
    case CompositeKind::Softsign:
      result = div(
          input, add(one, unary(rewriter, loc, "tfl.abs", resultType, input)));
      break;
    case CompositeKind::Shrink: {
      float bias =
          op->template getAttrOfType<FloatAttr>("bias").getValueAsDouble();
      float lambda =
          op->template getAttrOfType<FloatAttr>("lambd").getValueAsDouble();
      auto conditionType =
          RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
      Value above = createTFLOperation(rewriter, loc, "tfl.greater",
          TypeRange{conditionType},
          ValueRange{input, comparisonConstant(lambda)})
                        ->getResult(0);
      Value below = createTFLOperation(rewriter, loc, "tfl.less",
          TypeRange{conditionType},
          ValueRange{input, comparisonConstant(-lambda)})
                        ->getResult(0);
      Value positive = sub(input, scalar(bias));
      Value negative = add(input, scalar(bias));
      Value lowerSelected = createTFLOperation(rewriter, loc, "tfl.select_v2",
          TypeRange{resultType}, ValueRange{below, negative, zero})
                                ->getResult(0);
      result = createTFLOperation(rewriter, loc, "tfl.select_v2",
          TypeRange{resultType}, ValueRange{above, positive, lowerSelected})
                   ->getResult(0);
      break;
    }
    case CompositeKind::ThresholdedRelu: {
      float alpha =
          op->template getAttrOfType<FloatAttr>("alpha").getValueAsDouble();
      auto conditionType =
          RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
      Value condition = createTFLOperation(rewriter, loc, "tfl.greater",
          TypeRange{conditionType},
          ValueRange{input, comparisonConstant(alpha)})
                            ->getResult(0);
      result = createTFLOperation(rewriter, loc, "tfl.select_v2",
          TypeRange{resultType}, ValueRange{condition, input, zero})
                   ->getResult(0);
      break;
    }
    case CompositeKind::Atan: {
      Value oneLike = add(mul(input, zero), one);
      result = createTFLOperation(rewriter, loc, "tfl.atan2",
          TypeRange{resultType}, ValueRange{input, oneLike})
                   ->getResult(0);
      break;
    }
    case CompositeKind::Asin:
    case CompositeKind::Acos: {
      Value square = mul(input, input);
      Value radicand = binary(rewriter, loc, "tfl.maximum", resultType,
          sub(one, square), zero, /*hasFusedActivation=*/false);
      Value root = unary(rewriter, loc, "tfl.sqrt", resultType, radicand);
      Value lhs = kind == CompositeKind::Asin ? input : root;
      Value rhs = kind == CompositeKind::Asin ? root : input;
      result = createTFLOperation(rewriter, loc, "tfl.atan2",
          TypeRange{resultType}, ValueRange{lhs, rhs})
                   ->getResult(0);
      break;
    }
    case CompositeKind::Atanh:
      result = mul(scalar(0.5f), unary(rewriter, loc, "tfl.log", resultType,
                                     div(add(one, input), sub(one, input))));
      break;
    case CompositeKind::Sinh:
    case CompositeKind::Cosh: {
      Value positive = unary(rewriter, loc, "tfl.exp", resultType, input);
      Value negative = unary(rewriter, loc, "tfl.exp", resultType,
          unary(rewriter, loc, "tfl.neg", resultType, input));
      Value combined = kind == CompositeKind::Sinh ? sub(positive, negative)
                                                   : add(positive, negative);
      result = mul(scalar(0.5f), combined);
      break;
    }
    case CompositeKind::Asinh:
      result = unary(rewriter, loc, "tfl.log", resultType,
          add(input, unary(rewriter, loc, "tfl.sqrt", resultType,
                         add(mul(input, input), one))));
      break;
    case CompositeKind::Acosh: {
      Value radicand = binary(rewriter, loc, "tfl.maximum", resultType,
          sub(mul(input, input), one), zero,
          /*hasFusedActivation=*/false);
      result = unary(rewriter, loc, "tfl.log", resultType,
          add(input, unary(rewriter, loc, "tfl.sqrt", resultType, radicand)));
      break;
    }
    case CompositeKind::Erf: {
      // Numerical Recipes' erfc approximation. Its maximum absolute error is
      // about 1.2e-7 for f32 and it uses only portable TFL builtins.
      Value absolute = unary(rewriter, loc, "tfl.abs", resultType, input);
      Value t = div(one, add(one, mul(scalar(0.5f), absolute)));
      constexpr float coefficients[] = {0.17087277f, -0.82215223f, 1.48851587f,
          -1.13520398f, 0.27886807f, -0.18628806f, 0.09678418f, 0.37409196f,
          1.00002368f};
      Value polynomial = scalar(coefficients[0]);
      for (float coefficient : llvm::drop_begin(coefficients))
        polynomial = add(scalar(coefficient), mul(t, polynomial));
      Value exponent = add(sub(unary(rewriter, loc, "tfl.neg", resultType,
                                   mul(absolute, absolute)),
                               scalar(1.26551223f)),
          mul(t, polynomial));
      Value tau = mul(t, unary(rewriter, loc, "tfl.exp", resultType, exponent));
      Value magnitude = sub(one, tau);
      Value sign = unary(rewriter, loc, "tfl.sign", resultType, input);
      result = mul(sign, magnitude);
      break;
    }
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  CompositeKind kind;
};

class PReluLowering final : public OpConversionPattern<ONNXPReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXPReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto slopeType = dyn_cast<RankedTensorType>(op.getSlope().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !slopeType || !resultType ||
        failed(validateStaticF32Tensor(op, inputType, "input")) ||
        failed(validateStaticF32Tensor(op, slopeType, "slope")) ||
        failed(validateStaticF32Tensor(op, resultType, "result")))
      return failure();
    Value slope = adaptor.getSlope();
    if (resultType.getRank() == 4 &&
        (slopeType.getRank() == 3 || slopeType.getRank() == 4)) {
      ArrayRef<int64_t> shape = slopeType.getShape();
      int64_t channelCount;
      if (slopeType.getRank() == 3) {
        if (shape[1] != 1 || shape[2] != 1)
          return op.emitError("rank-4 PRelu requires a [C,1,1] or "
                              "[1,C,1,1] slope"),
                 failure();
        channelCount = shape[0];
      } else {
        if (shape[0] != 1 || shape[2] != 1 || shape[3] != 1)
          return op.emitError("rank-4 PRelu requires a [C,1,1] or "
                              "[1,C,1,1] slope"),
                 failure();
        channelCount = shape[1];
      }
      auto physicalSlopeType =
          RankedTensorType::get({1, 1, channelCount}, rewriter.getF32Type());
      Value physicalShape =
          createI32ShapeConstant(rewriter, op.getLoc(), {1, 1, channelCount});
      slope = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{physicalSlopeType}, ValueRange{slope, physicalShape})
                  ->getResult(0);
    } else if (resultType.getRank() != 4) {
      int64_t inputRank = inputType.getRank();
      int64_t slopeRank = slopeType.getRank();
      if (inputRank < 2 || slopeRank > inputRank)
        return op.emitError(
                   "PRelu requires an input rank of at least two and a slope "
                   "rank no greater than the input rank"),
               failure();

      SmallVector<int64_t> expandedSlopeShape(inputRank, 1);
      std::copy(slopeType.getShape().begin(), slopeType.getShape().end(),
          expandedSlopeShape.end() - slopeRank);
      for (int64_t axis = 0; axis < inputRank; ++axis) {
        int64_t slopeDimension = expandedSlopeShape[axis];
        int64_t inputDimension = inputType.getShape()[axis];
        if (slopeDimension != 1 && slopeDimension != inputDimension)
          return op.emitError("PRelu slope is not broadcast-compatible with "
                              "the input"),
                 failure();
      }
      if (expandedSlopeShape.front() != 1)
        return op.emitError(
                   "TFL PRelu does not support a batch-dependent slope"),
               failure();

      SmallVector<int64_t> normalizedSlopeShape(
          llvm::drop_begin(expandedSlopeShape));
      if (slopeType.getShape() != ArrayRef<int64_t>(normalizedSlopeShape)) {
        auto normalizedSlopeType =
            RankedTensorType::get(normalizedSlopeShape, rewriter.getF32Type());
        Value normalizedShape =
            createI32ShapeConstant(rewriter, op.getLoc(), normalizedSlopeShape);
        slope = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{normalizedSlopeType}, ValueRange{slope, normalizedShape})
                    ->getResult(0);
      }
    }
    Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.prelu",
        TypeRange{physicalResultType}, ValueRange{adaptor.getX(), slope})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MeanLowering final : public OpConversionPattern<ONNXMeanOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMeanOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType || adaptor.getOperands().empty() ||
        failed(validateStaticF32Tensor(op, resultType, "result")))
      return failure();
    Type physicalType = convertRank4NCHWToNHWCType(resultType);
    Value sum = adaptor.getOperands().front();
    for (Value operand : llvm::drop_begin(adaptor.getOperands()))
      sum =
          binary(rewriter, op.getLoc(), "tfl.add", physicalType, sum, operand);
    Value count = createF32ScalarTensorConstant(rewriter, op.getLoc(),
        static_cast<float>(adaptor.getOperands().size()));
    Value result =
        binary(rewriter, op.getLoc(), "tfl.div", physicalType, sum, count);
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ONNXOp>
class LogicalBinaryLowering final : public OpConversionPattern<ONNXOp> {
public:
  LogicalBinaryLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}
  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        !resultType.getElementType().isInteger(1))
      return op.emitError("ONNXToTFL logical operation requires a static "
                          "boolean result"),
             failure();
    Value result = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, adaptor.getOperands())
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  std::string tflName;
};

} // namespace

void populateLoweringONNXAdditionalElementwiseOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<PReluLowering, MeanLowering>(typeConverter, context);
  patterns.add<LogicalBinaryLowering<ONNXAndOp>>(
      typeConverter, context, "tfl.logical_and");
  patterns.add<LogicalBinaryLowering<ONNXOrOp>>(
      typeConverter, context, "tfl.logical_or");
  patterns.add<LogicalBinaryLowering<ONNXXorOp>>(
      typeConverter, context, "tfl.not_equal");
  patterns.add<CompositeUnaryLowering<ONNXCeluOp>>(
      typeConverter, context, CompositeKind::Celu);
  patterns.add<CompositeUnaryLowering<ONNXSeluOp>>(
      typeConverter, context, CompositeKind::Selu);
  patterns.add<CompositeUnaryLowering<ONNXSoftplusOp>>(
      typeConverter, context, CompositeKind::Softplus);
  patterns.add<CompositeUnaryLowering<ONNXSoftsignOp>>(
      typeConverter, context, CompositeKind::Softsign);
  patterns.add<CompositeUnaryLowering<ONNXShrinkOp>>(
      typeConverter, context, CompositeKind::Shrink);
  patterns.add<CompositeUnaryLowering<ONNXThresholdedReluOp>>(
      typeConverter, context, CompositeKind::ThresholdedRelu);
  patterns.add<CompositeUnaryLowering<ONNXAsinOp>>(
      typeConverter, context, CompositeKind::Asin);
  patterns.add<CompositeUnaryLowering<ONNXAcosOp>>(
      typeConverter, context, CompositeKind::Acos);
  patterns.add<CompositeUnaryLowering<ONNXAtanOp>>(
      typeConverter, context, CompositeKind::Atan);
  patterns.add<CompositeUnaryLowering<ONNXAtanhOp>>(
      typeConverter, context, CompositeKind::Atanh);
  patterns.add<CompositeUnaryLowering<ONNXSinhOp>>(
      typeConverter, context, CompositeKind::Sinh);
  patterns.add<CompositeUnaryLowering<ONNXCoshOp>>(
      typeConverter, context, CompositeKind::Cosh);
  patterns.add<CompositeUnaryLowering<ONNXAsinhOp>>(
      typeConverter, context, CompositeKind::Asinh);
  patterns.add<CompositeUnaryLowering<ONNXAcoshOp>>(
      typeConverter, context, CompositeKind::Acosh);
  patterns.add<CompositeUnaryLowering<ONNXErfOp>>(
      typeConverter, context, CompositeKind::Erf);
}

} // namespace onnx_mlir
