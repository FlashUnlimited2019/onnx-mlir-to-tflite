/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<Value> adaptRank4ElementwiseBroadcast(Operation *sourceOp,
    unsigned operandIndex, Value convertedOperand, Type convertedResultType,
    ConversionPatternRewriter &rewriter) {
  auto sourceResultType =
      dyn_cast<RankedTensorType>(sourceOp->getResult(0).getType());
  auto sourceOperandType =
      dyn_cast<RankedTensorType>(sourceOp->getOperand(operandIndex).getType());
  auto resultType = dyn_cast<RankedTensorType>(convertedResultType);
  if (!sourceResultType || sourceResultType.getRank() != 4 ||
      !sourceOperandType)
    return convertedOperand;

  int64_t operandRank = sourceOperandType.getRank();
  if (operandRank == 0 || operandRank == 4)
    return convertedOperand;
  if (operandRank < 0 || operandRank > 3 || !resultType) {
    sourceOp->emitError()
        << "unsupported rank-4 elementwise broadcast: operand "
        << sourceOperandType << ", result " << sourceResultType;
    return failure();
  }

  SmallVector<int64_t> resultLayout =
      getONNXToTFLPhysicalToLogicalLayout(sourceResultType);
  SmallVector<int64_t> operandLayout =
      getONNXToTFLPhysicalToLogicalLayout(sourceOperandType);
  SmallVector<int64_t> logicalShape(4, 1);
  std::copy(sourceOperandType.getShape().begin(),
      sourceOperandType.getShape().end(), logicalShape.end() - operandRank);
  for (int64_t axis = 0; axis < 4; ++axis) {
    int64_t operandDim = logicalShape[axis];
    int64_t resultDim = sourceResultType.getShape()[axis];
    if (operandDim != 1 && operandDim != resultDim) {
      sourceOp->emitError()
          << "rank-4 elementwise operands are not broadcast-compatible in "
             "logical ONNX axis order: operand "
          << sourceOperandType << ", result " << sourceResultType;
      return failure();
    }
  }

  bool operandIsIdentity =
      llvm::all_of(llvm::enumerate(operandLayout), [](auto axis) {
        return static_cast<int64_t>(axis.index()) == axis.value();
      });
  if (operandIsIdentity && resultLayout == SmallVector<int64_t>({0, 1, 2, 3}))
    return convertedOperand;

  // Broadcast aligns logical suffix axes, not physical suffix axes. For a
  // permuted result or operand, order the operand's existing axes as they
  // occur in the result, then insert the missing singleton axes. This also
  // handles layouts that move the logical batch axis away from the front.
  if (!operandIsIdentity ||
      resultLayout != SmallVector<int64_t>({0, 2, 3, 1})) {
    SmallVector<int64_t> targetOperandLayout;
    SmallVector<int64_t> broadcastShape;
    for (int64_t axis : resultLayout) {
      broadcastShape.push_back(logicalShape[axis]);
      if (axis >= 4 - operandRank)
        targetOperandLayout.push_back(axis - (4 - operandRank));
    }
    auto nonSingletonAxes = [&](ArrayRef<int64_t> layout) {
      SmallVector<int64_t> axes;
      for (int64_t axis : layout)
        if (sourceOperandType.getShape()[axis] != 1)
          axes.push_back(axis);
      return axes;
    };
    Value value = convertedOperand;
    if (nonSingletonAxes(operandLayout) !=
        nonSingletonAxes(targetOperandLayout)) {
      FailureOr<Value> reordered =
          convertTFLValueLayout(rewriter, sourceOp->getLoc(), value,
              stripONNXToTFLLayoutEncoding(sourceOperandType), operandLayout,
              targetOperandLayout);
      if (failed(reordered))
        return failure();
      value = *reordered;
    }
    // Leading singleton axes are implicit in TFLite broadcasting. Never
    // remove interior singleton axes: they locate the logical suffix axes.
    while (broadcastShape.size() > 1 && broadcastShape.front() == 1)
      broadcastShape.erase(broadcastShape.begin());
    auto broadcastType = RankedTensorType::get(
        broadcastShape, sourceOperandType.getElementType());
    if (value.getType() == broadcastType)
      return value;
    Value shape =
        createI32ShapeConstant(rewriter, sourceOp->getLoc(), broadcastShape);
    return createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.reshape",
        TypeRange{broadcastType}, ValueRange{value, shape})
        ->getResult(0);
  }

  // A lower-rank ONNX operand aligns with the logical C/H/W suffix. TFLite
  // broadcasts against the physical H/W/C suffix, so represent it as HWC.
  // Rank 1/2 only move singleton dimensions and can use Reshape. General CHW
  // rank-3 data needs a real CHW->HWC transpose; retain the cheaper Reshape
  // for the common [C,1,1] channel-parameter and [1,H,W] cases.
  auto broadcastType =
      RankedTensorType::get({logicalShape[2], logicalShape[3], logicalShape[1]},
          sourceOperandType.getElementType());
  if (operandRank == 3 && logicalShape[1] != 1 &&
      (logicalShape[2] != 1 || logicalShape[3] != 1)) {
    Value permutation =
        createI32ShapeConstant(rewriter, sourceOp->getLoc(), {1, 2, 0});
    return createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.transpose",
        TypeRange{broadcastType}, ValueRange{convertedOperand, permutation})
        ->getResult(0);
  }

  Value shapeValue = createI32ShapeConstant(
      rewriter, sourceOp->getLoc(), broadcastType.getShape());
  return createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.reshape",
      TypeRange{broadcastType}, ValueRange{convertedOperand, shapeValue})
      ->getResult(0);
}

struct I64Range {
  int64_t lower = std::numeric_limits<int64_t>::min();
  int64_t upper = std::numeric_limits<int64_t>::max();
};

// A deliberately small range analysis: never assume bounds for graph inputs,
// infer through overflowing arithmetic, or reinterpret floating-point casts.
// Limit recursion and constant inspection independently of graph size.
I64Range knownI64Range(Value value, unsigned depth = 0) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() ||
      !type.getElementType().isSignlessInteger(64) || depth >= 8)
    return {};
  if (type.getNumElements() > 0 && type.getNumElements() <= 65536) {
    auto values = getConstantIntValues(value);
    if (succeeded(values) && !values->empty()) {
      auto bounds = std::minmax_element(values->begin(), values->end());
      return {*bounds.first, *bounds.second};
    }
  }
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getNumResults() != 1)
    return {};
  StringRef name = producer->getName().getStringRef();
  if (name == "tfl.cast" && producer->getNumOperands() == 1) {
    auto input = dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
    if (input && input.getElementType().isSignlessInteger(32))
      return {std::numeric_limits<int32_t>::min(),
          std::numeric_limits<int32_t>::max()};
  }
  if ((name == "tfl.reshape" || name == "tfl.transpose" ||
          name == "tfl.broadcast_to") &&
      producer->getNumOperands() == 2)
    return knownI64Range(producer->getOperand(0), depth + 1);
  if ((name == "tfl.maximum" || name == "tfl.minimum") &&
      producer->getNumOperands() == 2) {
    auto a = knownI64Range(producer->getOperand(0), depth + 1);
    auto b = knownI64Range(producer->getOperand(1), depth + 1);
    if (name == "tfl.maximum")
      return {std::max(a.lower, b.lower), std::max(a.upper, b.upper)};
    return {std::min(a.lower, b.lower), std::min(a.upper, b.upper)};
  }
  return {};
}

FailureOr<Value> lowerGeneralI64Div(Operation *op, Value lhs, Value rhs,
    RankedTensorType resultType, ConversionPatternRewriter &rewriter) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  int64_t rank = std::max(lhsType.getRank(), rhsType.getRank());
  if (rank > 5 || rank != resultType.getRank() ||
      llvm::any_of(resultType.getShape(), [](int64_t d) { return d <= 0; }))
    return op->emitError("i64 Div requires nonempty static broadcast results "
                         "of rank at most 5"),
           failure();
  for (int64_t offset = 0; offset < rank; ++offset) {
    int64_t a = offset < lhsType.getRank()
                    ? lhsType.getDimSize(lhsType.getRank() - 1 - offset)
                    : 1;
    int64_t b = offset < rhsType.getRank()
                    ? rhsType.getDimSize(rhsType.getRank() - 1 - offset)
                    : 1;
    if ((a != b && a != 1 && b != 1) ||
        resultType.getDimSize(rank - 1 - offset) != std::max(a, b))
      return op->emitError("i64 Div result does not match broadcasting"),
             failure();
  }
  auto divisors = getConstantIntValues(rhs);
  if (succeeded(divisors) && llvm::is_contained(*divisors, int64_t{0}))
    return op->emitError("i64 Div requires nonzero divisors"), failure();

  Location loc = op->getLoc();
  auto emit = [&](StringRef name, Type type, ValueRange operands) -> Value {
    SmallVector<NamedAttribute> attrs;
    if (name == "tfl.add" || name == "tfl.sub" || name == "tfl.mul" ||
        name == "tfl.div")
      attrs.push_back(getFusedActivationNone(rewriter));
    return createTFLOperation(
        rewriter, loc, name, TypeRange{type}, operands, attrs)
        ->getResult(0);
  };
  I64Range dividendRange = knownI64Range(lhs);
  I64Range divisorRange = knownI64Range(rhs);
  if (dividendRange.lower >= 0 &&
      dividendRange.upper < std::numeric_limits<int32_t>::max() &&
      divisorRange.lower >= 1) {
    // For 0<=a<=A and b>=1, a/b == a/min(b,A+1). Clamping the divisor
    // preserves the zero quotient above A and makes both i32 casts exact.
    // Neither a user-provided length hint nor unchecked i64 narrowing is used.
    int64_t cap = dividendRange.upper + 1;
    if (divisorRange.upper > cap) {
      auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
      Value limit = arith::ConstantOp::create(rewriter, loc, scalarType,
          DenseIntElementsAttr::get(scalarType, ArrayRef<int64_t>{cap}));
      rhs = emit("tfl.minimum", rhsType, {rhs, limit});
    }
    auto narrow = [&](Value value) {
      auto type = cast<RankedTensorType>(value.getType());
      return emit("tfl.cast",
          RankedTensorType::get(type.getShape(), rewriter.getI32Type()),
          {value});
    };
    Value quotient = emit("tfl.div",
        RankedTensorType::get(resultType.getShape(), rewriter.getI32Type()),
        {narrow(lhs), narrow(rhs)});
    return emit("tfl.cast", resultType, {quotient});
  }
  RankedTensorType calculationType = resultType;
  // Comparison kernels broadcast at most four dimensions. Materialize the
  // rank-five broadcast once, then do the elementwise calculation in rank one.
  if (rank == 5) {
    if (resultType.getNumElements() > std::numeric_limits<int32_t>::max())
      return op->emitError("i64 Div flattened shape exceeds the i32 range"),
             failure();
    calculationType = RankedTensorType::get(
        {resultType.getNumElements()}, rewriter.getI64Type());
    auto flatten = [&](Value value) {
      if (value.getType() != resultType)
        value = emit("tfl.broadcast_to", resultType,
            {value,
                createI32ShapeConstant(rewriter, loc, resultType.getShape())});
      return emit("tfl.reshape", calculationType,
          {value, createI32ShapeConstant(
                      rewriter, loc, calculationType.getShape())});
    };
    lhs = flatten(lhs);
    rhs = flatten(rhs);
    lhsType = rhsType = calculationType;
  }
  auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
  auto scalar = [&](int64_t value) -> Value {
    return arith::ConstantOp::create(rewriter, loc, scalarType,
        DenseIntElementsAttr::get(scalarType, ArrayRef<int64_t>{value}));
  };
  Value zero = scalar(0), one = scalar(1);
  auto negativeMagnitude = [&](Value value) {
    Type type = value.getType();
    Value positive = emit("tfl.maximum", type, {value, zero});
    Value negative = emit("tfl.sub", type, {zero, positive});
    return emit("tfl.minimum", type, {value, negative});
  };
  Value remainder = negativeMagnitude(lhs);
  Value divisor = negativeMagnitude(rhs);
  auto floatType =
      RankedTensorType::get(calculationType.getShape(), rewriter.getF32Type());
  auto boolType =
      RankedTensorType::get(calculationType.getShape(), rewriter.getI1Type());
  Value divisorFloat = emit("tfl.cast",
      RankedTensorType::get(rhsType.getShape(), rewriter.getF32Type()),
      {divisor});
  Value scale = createF32ScalarTensorConstant(rewriter, loc, 1.0f - 0x1p-12f);
  Value quotient = zero;
  // Do not use a plain float quotient or narrow arbitrary i64 inputs. For
  // nonzero divisors, round-to-nearest FP32 estimates scaled by (1-2^-12)
  // underestimate the remaining magnitude quotient. Their relative deficit
  // is less than 2^-11, including casts/division/multiplication rounding.
  // Each integer residual update therefore reduces q to < q*2^-11+1.
  // Starting at q<=2^63, six updates leave q<2; one integer comparison gives
  // the exact final unit. Negative magnitudes preserve INT64_MIN without abs
  // overflow. As for integer division generally, INT64_MIN/-1 is excluded.
  for (int i = 0; i < 6; ++i) {
    auto remainderType = cast<RankedTensorType>(remainder.getType());
    Value asFloat = emit("tfl.cast",
        RankedTensorType::get(remainderType.getShape(), rewriter.getF32Type()),
        {remainder});
    Value estimate = emit("tfl.div", floatType, {asFloat, divisorFloat});
    estimate = emit("tfl.mul", floatType, {estimate, scale});
    Value part = emit("tfl.cast", calculationType, {estimate});
    Value product = emit("tfl.mul", calculationType, {part, divisor});
    remainder = emit("tfl.sub", calculationType, {remainder, product});
    quotient = emit("tfl.sub", calculationType, {quotient, part});
  }
  Value extra = emit("tfl.less_equal", boolType, {remainder, divisor});
  Value delta = emit("tfl.select_v2", calculationType, {extra, one, zero});
  quotient = emit("tfl.sub", calculationType, {quotient, delta});
  Value lhsNegative = emit("tfl.less",
      RankedTensorType::get(lhsType.getShape(), rewriter.getI1Type()),
      {lhs, zero});
  Value rhsNegative = emit("tfl.less",
      RankedTensorType::get(rhsType.getShape(), rewriter.getI1Type()),
      {rhs, zero});
  Value negative = emit("tfl.not_equal", boolType, {lhsNegative, rhsNegative});
  // Select before negating so the valid negative INT64_MIN quotient is never
  // negated even speculatively in an unselected branch.
  Value toNegate =
      emit("tfl.select_v2", calculationType, {negative, zero, quotient});
  Value positive = emit("tfl.sub", calculationType, {zero, toNegate});
  Value result =
      emit("tfl.select_v2", calculationType, {negative, quotient, positive});
  if (rank == 5)
    result = emit("tfl.reshape", resultType,
        {result, createI32ShapeConstant(rewriter, loc, resultType.getShape())});
  return result;
}

template <typename ONNXOp>
class BinaryElementwiseLowering final : public OpConversionPattern<ONNXOp> {
public:
  BinaryElementwiseLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto lhsType =
        dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    auto rhsType =
        dyn_cast<RankedTensorType>(adaptor.getOperands()[1].getType());
    auto sourceResultType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    bool supportsI32 = tflName == "tfl.add";
    bool supportsI64 = tflName == "tfl.add" || tflName == "tfl.sub" ||
                       tflName == "tfl.mul" || tflName == "tfl.div";
    bool isF32 = lhsType && lhsType.getElementType().isF32();
    bool isI32 = lhsType && lhsType.getElementType().isSignlessInteger(32);
    bool isI64 = lhsType && lhsType.getElementType().isSignlessInteger(64);
    if (!lhsType || !rhsType || !sourceResultType ||
        !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() ||
        rhsType.getElementType() != lhsType.getElementType() ||
        sourceResultType.getElementType() != lhsType.getElementType() ||
        (!isF32 && !(isI32 && supportsI32) && !(isI64 && supportsI64))) {
      op.emitError() << "ONNXToTFL " << op->getName()
                     << " requires matching static FP32 operands/results, "
                        "i32 for Add, or i64 for Add/Sub/Mul/Div";
      return failure();
    }

    if (isI64 && tflName == "tfl.div") {
      // TFLite's integer Div kernel accepts i32, not i64. Narrow only when
      // the dividend is proven representable and the divisor is a positive
      // i32 constant. Positivity also excludes INT32_MIN / -1 overflow.
      Value lhs = adaptor.getOperands()[0];
      Value rhs = adaptor.getOperands()[1];
      auto fitsI32 = [](Value value) {
        auto constants = getConstantIntValues(value);
        if (succeeded(constants))
          return llvm::all_of(*constants, [](int64_t v) {
            return v >= std::numeric_limits<int32_t>::min() &&
                   v <= std::numeric_limits<int32_t>::max();
          });
        Operation *producer = value.getDefiningOp();
        if (!producer || producer->getName().getStringRef() != "tfl.cast" ||
            producer->getNumOperands() != 1)
          return false;
        auto source =
            dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
        return source && source.getElementType().isSignlessInteger(32);
      };
      auto divisors = getConstantIntValues(rhs);
      if (sourceResultType.getRank() > 5 || !fitsI32(lhs) || failed(divisors) ||
          divisors->empty() || !llvm::all_of(*divisors, [](int64_t v) {
            return v > 0 && v <= std::numeric_limits<int32_t>::max();
          })) {
        auto result = lowerGeneralI64Div(op, lhs, rhs,
            cast<RankedTensorType>(
                this->getTypeConverter()->convertType(sourceResultType)),
            rewriter);
        if (failed(result))
          return failure();
        rewriter.replaceOp(op, *result);
        return success();
      }
      auto narrow = [&](Value value, RankedTensorType type) {
        return createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
            TypeRange{
                RankedTensorType::get(type.getShape(), rewriter.getI32Type())},
            ValueRange{value})
            ->getResult(0);
      };
      auto quotientType = RankedTensorType::get(
          sourceResultType.getShape(), rewriter.getI32Type());
      Value quotient = createTFLOperation(rewriter, op.getLoc(), "tfl.div",
          TypeRange{quotientType},
          ValueRange{narrow(lhs, lhsType), narrow(rhs, rhsType)},
          {getFusedActivationNone(rewriter)})
                           ->getResult(0);
      Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
          TypeRange{sourceResultType}, ValueRange{quotient})
                         ->getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    SmallVector<Value> operands(adaptor.getOperands());
    bool isPow = tflName == "tfl.pow";
    int64_t maximumBroadcastRank = isPow ? 4 : 5;
    if (sourceResultType.getRank() > maximumBroadcastRank) {
      int64_t resultRank = sourceResultType.getRank();
      SmallVector<int64_t> lhsShape(resultRank, 1);
      SmallVector<int64_t> rhsShape(resultRank, 1);
      std::copy(lhsType.getShape().begin(), lhsType.getShape().end(),
          lhsShape.end() - lhsType.getRank());
      std::copy(rhsType.getShape().begin(), rhsType.getShape().end(),
          rhsShape.end() - rhsType.getRank());

      SmallVector<int64_t> reducedResultShape;
      SmallVector<int64_t> reducedLhsShape;
      SmallVector<int64_t> reducedRhsShape;
      SmallVector<std::pair<bool, bool>> broadcastKinds;
      for (int64_t axis = 0; axis < resultRank; ++axis) {
        int64_t resultDimension = sourceResultType.getShape()[axis];
        int64_t lhsDimension = lhsShape[axis];
        int64_t rhsDimension = rhsShape[axis];
        if ((lhsDimension != 1 && lhsDimension != resultDimension) ||
            (rhsDimension != 1 && rhsDimension != resultDimension))
          return op.emitError("high-rank elementwise operands are not "
                              "broadcast-compatible"),
                 failure();
        if (resultDimension == 1)
          continue;
        std::pair<bool, bool> kind{lhsDimension == 1, rhsDimension == 1};
        if (!broadcastKinds.empty() && broadcastKinds.back() == kind) {
          reducedResultShape.back() *= resultDimension;
          reducedLhsShape.back() *= lhsDimension;
          reducedRhsShape.back() *= rhsDimension;
          continue;
        }
        broadcastKinds.push_back(kind);
        reducedResultShape.push_back(resultDimension);
        reducedLhsShape.push_back(lhsDimension);
        reducedRhsShape.push_back(rhsDimension);
      }
      if (reducedResultShape.empty()) {
        reducedResultShape.push_back(1);
        reducedLhsShape.push_back(1);
        reducedRhsShape.push_back(1);
      }
      if (reducedResultShape.size() > 4)
        return op.emitError("high-rank elementwise broadcast cannot be "
                            "collapsed safely to rank 4"),
               failure();

      auto reshapeOperand = [&](Value value, RankedTensorType sourceType,
                                ArrayRef<int64_t> shape) -> Value {
        if (isF32 && sourceType.getRank() == 4) {
          Value permutation =
              createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
          value = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
              TypeRange{sourceType}, ValueRange{value, permutation})
                      ->getResult(0);
        }
        auto reducedType =
            RankedTensorType::get(shape, sourceType.getElementType());
        Value reducedShape =
            createI32ShapeConstant(rewriter, op.getLoc(), shape);
        return createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{reducedType}, ValueRange{value, reducedShape})
            ->getResult(0);
      };
      operands[0] = reshapeOperand(operands[0], lhsType, reducedLhsShape);
      operands[1] = reshapeOperand(operands[1], rhsType, reducedRhsShape);
      auto reducedResultType = RankedTensorType::get(
          reducedResultShape, sourceResultType.getElementType());
      SmallVector<NamedAttribute> attributes;
      if (!isPow)
        attributes.push_back(getFusedActivationNone(rewriter));
      Value reducedResult = createTFLOperation(rewriter, op.getLoc(), tflName,
          TypeRange{reducedResultType}, operands, attributes)
                                ->getResult(0);
      Value restoredShape = createI32ShapeConstant(
          rewriter, op.getLoc(), sourceResultType.getShape());
      Value restored = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{sourceResultType}, ValueRange{reducedResult, restoredShape})
                           ->getResult(0);
      rewriter.replaceOp(op, restored);
      return success();
    }
    if (isF32) {
      for (unsigned i = 0; i < operands.size(); ++i) {
        FailureOr<Value> adapted = adaptRank4ElementwiseBroadcast(
            op, i, operands[i], resultType, rewriter);
        if (failed(adapted))
          return failure();
        operands[i] = *adapted;
      }
    }
    SmallVector<NamedAttribute> attributes;
    if (!isPow)
      attributes.push_back(getFusedActivationNone(rewriter));
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, operands, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  std::string tflName;
};

FailureOr<SmallVector<int64_t>> computeBroadcastShape(
    ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  size_t rank = std::max(lhs.size(), rhs.size());
  SmallVector<int64_t> result(rank, 1);
  for (size_t offset = 0; offset < rank; ++offset) {
    int64_t lhsDim = offset < lhs.size() ? lhs[lhs.size() - 1 - offset] : 1;
    int64_t rhsDim = offset < rhs.size() ? rhs[rhs.size() - 1 - offset] : 1;
    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
      return failure();
    result[rank - 1 - offset] = std::max(lhsDim, rhsDim);
  }
  return result;
}

template <typename ONNXOp>
class VariadicMinMaxLowering final : public OpConversionPattern<ONNXOp> {
public:
  VariadicMinMaxLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceResultType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!sourceResultType || !sourceResultType.hasStaticShape() ||
        adaptor.getOperands().empty())
      return op.emitError("ONNXToTFL Min/Max requires static tensor operands"),
             failure();

    Type elementType = sourceResultType.getElementType();
    if (!elementType.isF32() && !elementType.isSignlessInteger(32) &&
        !elementType.isSignlessInteger(64))
      return op.emitError("ONNXToTFL Min/Max supports FP32, i32, and i64"),
             failure();

    Type convertedResultType =
        this->getTypeConverter()->convertType(sourceResultType);
    SmallVector<Value> operands(adaptor.getOperands());
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto operandType =
          dyn_cast<RankedTensorType>(op->getOperand(i).getType());
      if (!operandType || !operandType.hasStaticShape() ||
          operandType.getElementType() != elementType)
        return op.emitError(
                   "ONNXToTFL Min/Max requires matching static operand types"),
               failure();
      if (elementType.isF32()) {
        FailureOr<Value> adapted = adaptRank4ElementwiseBroadcast(
            op, i, operands[i], convertedResultType, rewriter);
        if (failed(adapted))
          return failure();
        operands[i] = *adapted;
      }
    }

    Value result = operands.front();
    for (Value operand : llvm::drop_begin(operands)) {
      auto lhsType = cast<RankedTensorType>(result.getType());
      auto rhsType = cast<RankedTensorType>(operand.getType());
      FailureOr<SmallVector<int64_t>> resultShape =
          computeBroadcastShape(lhsType.getShape(), rhsType.getShape());
      if (failed(resultShape))
        return op.emitError("ONNXToTFL Min/Max operands are not broadcastable"),
               failure();
      auto resultType = RankedTensorType::get(*resultShape, elementType);
      result = createTFLOperation(rewriter, op.getLoc(), tflName,
          TypeRange{resultType}, ValueRange{result, operand})
                   ->getResult(0);
    }
    if (result.getType() != convertedResultType)
      return op.emitError("ONNXToTFL Min/Max broadcast result shape does not "
                          "match ONNX"),
             failure();
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  std::string tflName;
};

template <typename ONNXOp>
class UnaryElementwiseLowering final : public OpConversionPattern<ONNXOp> {
public:
  UnaryElementwiseLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32TensorOrScalar(
            op, adaptor.getOperands()[0].getType(), "operand")) ||
        failed(validateStaticF32TensorOrScalar(
            op, op->getResult(0).getType(), "result")))
      return failure();

    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, adaptor.getOperands());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  std::string tflName;
};

class IdentityLowering final : public OpConversionPattern<ONNXIdentityOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXIdentityOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32Tensor(
            op, adaptor.getOperands()[0].getType(), "operand")))
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

class GeluLowering final : public OpConversionPattern<ONNXGeluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32TensorOrScalar(
            op, adaptor.getX().getType(), "Gelu input")) ||
        failed(validateStaticF32TensorOrScalar(
            op, op.getY().getType(), "Gelu result")))
      return failure();

    StringRef approximate = op.getApproximate();
    if (approximate != "none" && approximate != "tanh") {
      op.emitError() << "ONNXToTFL Gelu supports approximate='none' or "
                        "approximate='tanh', but got '"
                     << approximate << "'";
      return failure();
    }

    Type resultType =
        this->getTypeConverter()->convertType(op.getY().getType());
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "approximate", rewriter.getBoolAttr(approximate == "tanh"))};
    Operation *gelu = createTFLOperation(rewriter, op.getLoc(), "tfl.gelu",
        TypeRange{resultType}, ValueRange{adaptor.getX()}, attributes);
    rewriter.replaceOp(op, gelu->getResults());
    return success();
  }
};

class EluLowering final : public OpConversionPattern<ONNXEluOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXEluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32TensorOrScalar(
            op, adaptor.getX().getType(), "Elu input")) ||
        failed(validateStaticF32TensorOrScalar(
            op, op.getY().getType(), "Elu result")))
      return failure();
    Type resultType =
        this->getTypeConverter()->convertType(op.getY().getType());
    float alpha = op.getAlpha().convertToFloat();
    if (alpha == 1.0f) {
      Operation *elu = createTFLOperation(rewriter, op.getLoc(), "tfl.elu",
          TypeRange{resultType}, ValueRange{adaptor.getX()});
      rewriter.replaceOp(op, elu->getResults());
      return success();
    }
    Value zero = createF32ScalarTensorConstant(rewriter, op.getLoc(), 0.0f);
    Value one = createF32ScalarTensorConstant(rewriter, op.getLoc(), 1.0f);
    Value alphaValue =
        createF32ScalarTensorConstant(rewriter, op.getLoc(), alpha);
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Value positive = createTFLOperation(rewriter, op.getLoc(), "tfl.maximum",
        TypeRange{resultType}, ValueRange{adaptor.getX(), zero})
                         ->getResult(0);
    Value exponent = createTFLOperation(rewriter, op.getLoc(), "tfl.exp",
        TypeRange{resultType}, ValueRange{adaptor.getX()})
                         ->getResult(0);
    Value shifted = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
        TypeRange{resultType}, ValueRange{exponent, one}, fusedNone)
                        ->getResult(0);
    Value scaled = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{resultType}, ValueRange{shifted, alphaValue}, fusedNone)
                       ->getResult(0);
    Value negative = createTFLOperation(rewriter, op.getLoc(), "tfl.minimum",
        TypeRange{resultType}, ValueRange{scaled, zero})
                         ->getResult(0);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
        TypeRange{resultType}, ValueRange{positive, negative}, fusedNone)
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ReciprocalLowering final : public OpConversionPattern<ONNXReciprocalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXReciprocalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32TensorOrScalar(
            op, adaptor.getX().getType(), "Reciprocal input")) ||
        failed(validateStaticF32TensorOrScalar(
            op, op.getY().getType(), "Reciprocal result")))
      return failure();
    Type resultType =
        this->getTypeConverter()->convertType(op.getY().getType());
    Value one = createF32ScalarTensorConstant(rewriter, op.getLoc(), 1.0f);
    SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
    Operation *division = createTFLOperation(rewriter, op.getLoc(), "tfl.div",
        TypeRange{resultType}, ValueRange{one, adaptor.getX()}, attributes);
    rewriter.replaceOp(op, division->getResults());
    return success();
  }
};

class ModLowering final : public OpConversionPattern<ONNXModOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXModOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto sourceLhsType = dyn_cast<RankedTensorType>(op.getA().getType());
    auto sourceRhsType = dyn_cast<RankedTensorType>(op.getB().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceLhsType || !sourceRhsType || !sourceResultType ||
        !sourceLhsType.hasStaticShape() || !sourceRhsType.hasStaticShape() ||
        !sourceResultType.hasStaticShape()) {
      op.emitError("ONNXToTFL Mod requires static ranked tensors");
      return failure();
    }

    Type elementType = sourceResultType.getElementType();
    bool isF32 = elementType.isF32();
    bool isI64 = elementType.isSignlessInteger(64);
    if ((!isF32 && !isI64) || sourceLhsType.getElementType() != elementType ||
        sourceRhsType.getElementType() != elementType ||
        sourceLhsType.getRank() > 5 || sourceRhsType.getRank() > 5 ||
        sourceResultType.getRank() > 5 || (isF32 && op.getFmod() != 1) ||
        (isI64 && op.getFmod() != 0 && op.getFmod() != 1)) {
      op.emitError("ONNXToTFL Mod supports static FP32 fmod=1 and i64 "
                   "fmod=0/1 operands/results of ranks 0 through 5");
      return failure();
    }

    int64_t resultRank =
        std::max(sourceLhsType.getRank(), sourceRhsType.getRank());
    SmallVector<int64_t> expectedShape(resultRank, 1);
    for (int64_t offset = 0; offset < resultRank; ++offset) {
      int64_t lhsAxis = sourceLhsType.getRank() - 1 - offset;
      int64_t rhsAxis = sourceRhsType.getRank() - 1 - offset;
      int64_t lhsDim =
          lhsAxis >= 0 ? sourceLhsType.getShape()[lhsAxis] : int64_t{1};
      int64_t rhsDim =
          rhsAxis >= 0 ? sourceRhsType.getShape()[rhsAxis] : int64_t{1};
      if (lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim) {
        op.emitError("Mod operands are not broadcast-compatible");
        return failure();
      }
      expectedShape[resultRank - 1 - offset] = std::max(lhsDim, rhsDim);
    }
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError("Mod result shape does not match operand broadcasting");
      return failure();
    }

    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();
    auto restoreLogicalRank4 = [&](Value value,
                                   RankedTensorType sourceType) -> Value {
      if (!isF32 || sourceType.getRank() != 4)
        return value;
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      return createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{value, permutation})
          ->getResult(0);
    };
    lhs = restoreLogicalRank4(lhs, sourceLhsType);
    rhs = restoreLogicalRank4(rhs, sourceRhsType);

    auto calculationType = sourceResultType;
    bool flattened = resultRank == 5;
    if (flattened) {
      calculationType = RankedTensorType::get(
          {sourceResultType.getNumElements()}, elementType);
      auto broadcastAndFlatten = [&](Value value,
                                     RankedTensorType sourceType) -> Value {
        SmallVector<int64_t> alignedShape(resultRank, 1);
        std::copy(sourceType.getShape().begin(), sourceType.getShape().end(),
            alignedShape.end() - sourceType.getRank());
        if (!llvm::equal(alignedShape, expectedShape)) {
          auto broadcastType =
              RankedTensorType::get(expectedShape, elementType);
          Value broadcastShape =
              createI32ShapeConstant(rewriter, op.getLoc(), expectedShape);
          value = createTFLOperation(rewriter, op.getLoc(), "tfl.broadcast_to",
              TypeRange{broadcastType}, ValueRange{value, broadcastShape})
                      ->getResult(0);
        }
        Value flatShape = createI32ShapeConstant(
            rewriter, op.getLoc(), calculationType.getShape());
        return createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{calculationType}, ValueRange{value, flatShape})
            ->getResult(0);
      };
      lhs = broadcastAndFlatten(lhs, sourceLhsType);
      rhs = broadcastAndFlatten(rhs, sourceRhsType);
    }

    Value floorMod = createTFLOperation(rewriter, op.getLoc(), "tfl.floor_mod",
        TypeRange{calculationType}, ValueRange{lhs, rhs})
                         ->getResult(0);
    Value result = floorMod;
    if (op.getFmod() == 1) {
      // TFL FloorMod follows the divisor's sign, whereas ONNX fmod follows
      // the dividend's sign. Subtract the divisor for nonzero remainders whose
      // sign differs from lhs. Less comparisons work for both f32 and i64.
      Value zero;
      if (isF32) {
        zero = createF32ScalarTensorConstant(rewriter, op.getLoc(), 0.0f);
      } else {
        auto scalarType = RankedTensorType::get({}, rewriter.getI64Type());
        zero = arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
            DenseIntElementsAttr::get(
                scalarType, ArrayRef<int64_t>{int64_t{0}}));
      }
      auto boolType = RankedTensorType::get(
          calculationType.getShape(), rewriter.getI1Type());
      auto lhsType = cast<RankedTensorType>(lhs.getType());
      auto lhsBoolType =
          RankedTensorType::get(lhsType.getShape(), rewriter.getI1Type());
      Value nonzero = createTFLOperation(rewriter, op.getLoc(), "tfl.not_equal",
          TypeRange{boolType}, ValueRange{floorMod, zero})
                          ->getResult(0);
      Value lhsNegative = createTFLOperation(rewriter, op.getLoc(), "tfl.less",
          TypeRange{lhsBoolType}, ValueRange{lhs, zero})
                              ->getResult(0);
      Value remainderNegative = createTFLOperation(rewriter, op.getLoc(),
          "tfl.less", TypeRange{boolType}, ValueRange{floorMod, zero})
                                    ->getResult(0);
      Value signDiff =
          createTFLOperation(rewriter, op.getLoc(), "tfl.not_equal",
              TypeRange{boolType}, ValueRange{lhsNegative, remainderNegative})
              ->getResult(0);
      SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
      Value corrected = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
          TypeRange{calculationType}, ValueRange{floorMod, rhs}, fusedNone)
                            ->getResult(0);
      Value signedRemainder = createTFLOperation(rewriter, op.getLoc(),
          "tfl.select_v2", TypeRange{calculationType},
          ValueRange{signDiff, corrected, floorMod})
                                  ->getResult(0);
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.select_v2",
          TypeRange{calculationType},
          ValueRange{nonzero, signedRemainder, floorMod})
                   ->getResult(0);
    }

    if (flattened) {
      Value resultShape = createI32ShapeConstant(
          rewriter, op.getLoc(), sourceResultType.getShape());
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{sourceResultType}, ValueRange{result, resultShape})
                   ->getResult(0);
    }
    if (isF32 && resultRank == 4) {
      Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
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

class ClipLowering final : public OpConversionPattern<ONNXClipOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXClipOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    auto sourceType = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        sourceType.getShape() != resultType.getShape() ||
        sourceType.getElementType() != resultType.getElementType() ||
        resultType.getRank() > 5 ||
        (!resultType.getElementType().isF32() &&
            !resultType.getElementType().isSignlessInteger(32) &&
            !resultType.getElementType().isSignlessInteger(64)))
      return op.emitError("ONNXToTFL Clip requires matching static FP32, i32, "
                          "or i64 input/result shapes of rank at most 5"),
             failure();

    auto validateBound = [&](Value source, Value converted,
                             StringRef role) -> LogicalResult {
      if (isa<NoneType>(source.getType()))
        return success();
      auto type = dyn_cast<RankedTensorType>(source.getType());
      auto convertedType = dyn_cast<RankedTensorType>(converted.getType());
      if (!type || type.getRank() != 0 || !convertedType ||
          convertedType.getRank() != 0 ||
          type.getElementType() != resultType.getElementType() ||
          convertedType.getElementType() != resultType.getElementType())
        return op.emitError() << "ONNXToTFL Clip requires an omitted or scalar "
                              << role << " with the input element type",
               failure();
      return success();
    };
    if (failed(validateBound(op.getMin(), adaptor.getMin(), "minimum")) ||
        failed(validateBound(op.getMax(), adaptor.getMax(), "maximum")))
      return failure();

    Type physicalResultType = getTypeConverter()->convertType(resultType);
    if (input.getType() != physicalResultType)
      return op.emitError("Clip input/result physical layouts must match"),
             failure();
    Value result = input;
    if (!isa<NoneType>(op.getMin().getType()))
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.maximum",
          TypeRange{physicalResultType}, ValueRange{result, adaptor.getMin()})
                   ->getResult(0);
    if (!isa<NoneType>(op.getMax().getType()))
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.minimum",
          TypeRange{physicalResultType}, ValueRange{result, adaptor.getMax()})
                   ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class HardSwishLowering final : public OpConversionPattern<ONNXHardSwishOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXHardSwishOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getX();
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    Operation *hardSwish = createTFLOperation(rewriter, op.getLoc(),
        "tfl.hard_swish", TypeRange{resultType}, ValueRange{input});
    rewriter.replaceOp(op, hardSwish->getResults());
    return success();
  }
};

class HardSigmoidLowering final
    : public OpConversionPattern<ONNXHardSigmoidOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXHardSigmoidOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getX();
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    float alpha = op.getAlpha().convertToFloat();
    float beta = op.getBeta().convertToFloat();
    Value alphaValue =
        createF32ScalarTensorConstant(rewriter, op.getLoc(), alpha);
    Value betaValue =
        createF32ScalarTensorConstant(rewriter, op.getLoc(), beta);
    Value zero = createF32ScalarTensorConstant(rewriter, op.getLoc(), 0.0f);
    Value one = createF32ScalarTensorConstant(rewriter, op.getLoc(), 1.0f);
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Operation *scaled = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{resultType}, ValueRange{input, alphaValue}, fusedNone);
    Operation *shifted = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
        TypeRange{resultType}, ValueRange{scaled->getResult(0), betaValue},
        fusedNone);
    Operation *lowerClamped =
        createTFLOperation(rewriter, op.getLoc(), "tfl.maximum",
            TypeRange{resultType}, ValueRange{shifted->getResult(0), zero});
    Operation *upperClamped =
        createTFLOperation(rewriter, op.getLoc(), "tfl.minimum",
            TypeRange{resultType}, ValueRange{lowerClamped->getResult(0), one});
    rewriter.replaceOp(op, upperClamped->getResults());
    return success();
  }
};

class LeakyReluLowering final : public OpConversionPattern<ONNXLeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXLeakyReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getX();
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
        "alpha", rewriter.getF32FloatAttr(op.getAlpha().convertToFloat()))};
    Operation *leakyRelu = createTFLOperation(rewriter, op.getLoc(),
        "tfl.leaky_relu", TypeRange{resultType}, ValueRange{input}, attributes);
    rewriter.replaceOp(op, leakyRelu->getResults());
    return success();
  }
};

class ReturnLowering final : public OpConversionPattern<ONNXReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class NoValueLowering final : public OpConversionPattern<ONNXNoneOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXNoneOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("value", rewriter.getUnitAttr())};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.no_value",
        TypeRange{this->getTypeConverter()->convertType(op.getType())}, {},
        attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<IdentityLowering, GeluLowering, EluLowering, ReciprocalLowering,
      ModLowering, ClipLowering, ReturnLowering, NoValueLowering>(
      typeConverter, context);
  patterns.add<HardSwishLowering, HardSigmoidLowering, LeakyReluLowering>(
      typeConverter, context);
  patterns.add<BinaryElementwiseLowering<ONNXAddOp>>(
      typeConverter, context, "tfl.add");
  patterns.add<BinaryElementwiseLowering<ONNXSubOp>>(
      typeConverter, context, "tfl.sub");
  patterns.add<BinaryElementwiseLowering<ONNXMulOp>>(
      typeConverter, context, "tfl.mul");
  patterns.add<BinaryElementwiseLowering<ONNXDivOp>>(
      typeConverter, context, "tfl.div");
  patterns.add<BinaryElementwiseLowering<ONNXPowOp>>(
      typeConverter, context, "tfl.pow");
  patterns.add<VariadicMinMaxLowering<ONNXMinOp>>(
      typeConverter, context, "tfl.minimum");
  patterns.add<VariadicMinMaxLowering<ONNXMaxOp>>(
      typeConverter, context, "tfl.maximum");
  patterns.add<UnaryElementwiseLowering<ONNXReluOp>>(
      typeConverter, context, "tfl.relu");
  patterns.add<UnaryElementwiseLowering<ONNXSigmoidOp>>(
      typeConverter, context, "tfl.logistic");
  patterns.add<UnaryElementwiseLowering<ONNXTanhOp>>(
      typeConverter, context, "tfl.tanh");
  patterns.add<UnaryElementwiseLowering<ONNXExpOp>>(
      typeConverter, context, "tfl.exp");
  patterns.add<UnaryElementwiseLowering<ONNXLogOp>>(
      typeConverter, context, "tfl.log");
  patterns.add<UnaryElementwiseLowering<ONNXSqrtOp>>(
      typeConverter, context, "tfl.sqrt");
  patterns.add<UnaryElementwiseLowering<ONNXCosOp>>(
      typeConverter, context, "tfl.cos");
  patterns.add<UnaryElementwiseLowering<ONNXSinOp>>(
      typeConverter, context, "tfl.sin");
  patterns.add<UnaryElementwiseLowering<ONNXAbsOp>>(
      typeConverter, context, "tfl.abs");
  patterns.add<UnaryElementwiseLowering<ONNXNegOp>>(
      typeConverter, context, "tfl.neg");
  patterns.add<UnaryElementwiseLowering<ONNXRoundOp>>(
      typeConverter, context, "tfl.round");
  patterns.add<UnaryElementwiseLowering<ONNXSignOp>>(
      typeConverter, context, "tfl.sign");
  patterns.add<UnaryElementwiseLowering<ONNXFloorOp>>(
      typeConverter, context, "tfl.floor");
}

} // namespace onnx_mlir
