/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;

namespace onnx_mlir {
namespace {

std::optional<int64_t> integerAttr(
    Operation *op, StringRef name, int64_t fallback) {
  Attribute attr = op->getAttr(name);
  if (!attr)
    return fallback;
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer || integer.getValue().getBitWidth() > 64)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

std::optional<float> floatAttr(Operation *op, StringRef name, float fallback) {
  Attribute attr = op->getAttr(name);
  if (!attr)
    return fallback;
  auto value = dyn_cast<FloatAttr>(attr);
  if (!value || !std::isfinite(value.getValueAsDouble()))
    return std::nullopt;
  float result = static_cast<float>(value.getValueAsDouble());
  return std::isfinite(result) ? std::optional<float>(result) : std::nullopt;
}

RankedTensorType staticF32(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32() ||
      tensor.getRank() > 5)
    return {};
  int64_t count = 1;
  for (int64_t dim : tensor.getShape()) {
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max() / count)
      return {};
    count *= dim;
  }
  return tensor;
}

// Calculate in logical order at explicit layout boundaries. Row-wise work is
// kept rank two or three, independently of the source activation rank.
class Builder {
public:
  Builder(ConversionPatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  RankedTensorType type(ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, rewriter.getF32Type());
  }
  Value op(StringRef name, Type type, ValueRange inputs,
      ArrayRef<NamedAttribute> attrs = {}) {
    return createTFLOperation(
        rewriter, loc, name, TypeRange{type}, inputs, attrs)
        ->getResult(0);
  }
  Value binary(StringRef name, Value a, Value b, Type type) {
    return op(name, type, {a, b}, {getFusedActivationNone(rewriter)});
  }
  Value reshape(Value value, ArrayRef<int64_t> shape) {
    if (value.getType() == type(shape))
      return value;
    return op("tfl.reshape", type(shape),
        {value, createI32ShapeConstant(rewriter, loc, shape)});
  }
  Value transpose(
      Value value, ArrayRef<int64_t> shape, ArrayRef<int64_t> permutation) {
    return op("tfl.transpose", type(shape),
        {value, createI32ShapeConstant(rewriter, loc, permutation)});
  }
  Value gather(Value value, ArrayRef<int64_t> shape, ArrayRef<int64_t> indices,
      int64_t axis) {
    return op("tfl.gather", type(shape),
        {value, createI32ShapeConstant(rewriter, loc, indices)},
        {rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
            rewriter.getNamedAttr(
                "batch_dims", rewriter.getI32IntegerAttr(0))});
  }
  Value concat(ValueRange inputs, ArrayRef<int64_t> shape, int64_t axis) {
    return op("tfl.concatenation", type(shape), inputs,
        {rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
            getFusedActivationNone(rewriter)});
  }
  FailureOr<Value> layout(
      Value value, RankedTensorType logicalType, bool toLogical) {
    SmallVector<int64_t> identity(logicalType.getRank());
    std::iota(identity.begin(), identity.end(), 0);
    auto physical = getONNXToTFLPhysicalToLogicalLayout(logicalType);
    return convertTFLValueLayout(rewriter, loc, value, logicalType,
        toLogical ? physical : identity, toLogical ? identity : physical);
  }

  ConversionPatternRewriter &rewriter;
  Location loc;
};

class ContribLowering final : public OpConversionPattern<ONNXCustomOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXCustomOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!domain)
      return failure();
    StringRef name = op.getFunctionName();
    if (domain.getValue().empty() && name == "SimplifiedLayerNormalization")
      return normalize(op, adaptor.getOperands(), rewriter, false);
    if (domain.getValue() != "com.microsoft")
      return failure();
    if (name == "SkipSimplifiedLayerNormalization")
      return normalize(op, adaptor.getOperands(), rewriter, true);
    if (name == "RotaryEmbedding")
      return rotary(op, adaptor.getOperands(), rewriter);
    return failure();
  }

private:
  LogicalResult normalize(ONNXCustomOp op, ValueRange operands,
      ConversionPatternRewriter &rewriter, bool skip) const {
    auto reject = [&]() {
      return op.emitError("static contrib RMS normalization requires FP32 "
                          "rank-2 through rank-5 last-axis data (rank-2/3 "
                          "for skip), matching skip shape, rank-1 scale/bias, "
                          "finite nonnegative epsilon, stash_type=1, and "
                          "the primary output only"),
             failure();
    };
    if (op.getNumResults() != 1 || (!skip && operands.size() != 2) ||
        (skip && operands.size() != 3 && operands.size() != 4))
      return reject();
    auto inputType = staticF32(op->getOperand(0).getType());
    unsigned scaleIndex = skip ? 2 : 1;
    auto scaleType = staticF32(op->getOperand(scaleIndex).getType());
    auto outputType = staticF32(op.getResult(0).getType());
    auto epsilon = floatAttr(op, "epsilon", skip ? 1e-12f : 1e-5f);
    auto axis = integerAttr(op, "axis", -1);
    auto stash = integerAttr(op, "stash_type", 1);
    if (!inputType || !scaleType || !outputType || !epsilon || *epsilon < 0 ||
        !axis || !stash || *stash != 1 || inputType.getRank() < 2 ||
        (skip && inputType.getRank() > 3) ||
        stripONNXToTFLLayoutEncoding(inputType) !=
            stripONNXToTFLLayoutEncoding(outputType) ||
        (*axis != -1 && *axis != inputType.getRank() - 1) ||
        scaleType.getRank() != 1 ||
        scaleType.getShape()[0] != inputType.getShape().back())
      return reject();
    bool bias = skip && operands.size() == 4 &&
                !isa<NoneType>(op->getOperand(3).getType());
    if (skip && op->getOperand(1).getType() != inputType)
      return reject();
    if (bias && op->getOperand(3).getType() != scaleType)
      return reject();

    Builder create(rewriter, op.getLoc());
    auto logical = create.layout(operands[0], inputType, true);
    if (failed(logical))
      return failure();
    Value input = *logical;
    if (skip) {
      auto other = create.layout(operands[1], inputType, true);
      if (failed(other))
        return failure();
      input = create.binary("tfl.add", input, *other, input.getType());
      if (bias)
        input = create.binary("tfl.add", input, operands[3], input.getType());
    }
    int64_t width = inputType.getShape().back();
    int64_t rows = inputType.getNumElements() / width;
    Value matrix = create.reshape(input, {rows, width});
    Value squared = create.binary("tfl.mul", matrix, matrix, matrix.getType());
    Value mean = create.op("tfl.mean", create.type({rows, 1}),
        {squared, createI32ShapeConstant(rewriter, op.getLoc(), {1})},
        {rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))});
    Value eps = createF32ScalarTensorConstant(rewriter, op.getLoc(), *epsilon);
    Value variance = create.binary("tfl.add", mean, eps, mean.getType());
    Value inverse = create.op("tfl.rsqrt", mean.getType(), {variance});
    Value normalized =
        create.binary("tfl.mul", matrix, inverse, matrix.getType());
    Value scaled = create.binary(
        "tfl.mul", normalized, operands[scaleIndex], matrix.getType());
    Value result = create.reshape(scaled, inputType.getShape());
    auto physical = create.layout(result, outputType, false);
    if (failed(physical))
      return failure();
    rewriter.replaceOp(op, *physical);
    return success();
  }

  LogicalResult rotary(ONNXCustomOp op, ValueRange operands,
      ConversionPatternRewriter &rewriter) const {
    auto reject = [&]() {
      return op.emitError("static com.microsoft RotaryEmbedding requires "
                          "rank-3 BSH or rank-4 BNSH FP32 input, matching "
                          "FP32 caches of width rotary_dim/2, constant "
                          "in-range i64 positions, consistent heads, an even "
                          "rotary dimension, interleaved=0/1, scale=1, and "
                          "is_packed_batching=0"),
             failure();
    };
    if (operands.size() != 4 || op.getNumResults() != 1)
      return reject();
    auto inputType = staticF32(op->getOperand(0).getType());
    auto outputType = staticF32(op.getResult(0).getType());
    auto cacheType = staticF32(op->getOperand(2).getType());
    auto positionsType =
        dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto headsAttr = integerAttr(op, "num_heads", 0);
    auto dimensionAttr = integerAttr(op, "rotary_embedding_dim", 0);
    auto interleaved = integerAttr(op, "interleaved", 0);
    auto packed = integerAttr(op, "is_packed_batching", 0);
    auto scale = floatAttr(op, "scale", 1.0f);
    if (!inputType || !outputType || !cacheType || !positionsType ||
        !positionsType.hasStaticShape() ||
        !positionsType.getElementType().isSignlessInteger(64) || !headsAttr ||
        *headsAttr < 0 || !dimensionAttr || *dimensionAttr < 0 ||
        !interleaved || (*interleaved != 0 && *interleaved != 1) || !packed ||
        *packed != 0 || !scale || *scale != 1.0f ||
        (inputType.getRank() != 3 && inputType.getRank() != 4) ||
        stripONNXToTFLLayoutEncoding(inputType) !=
            stripONNXToTFLLayoutEncoding(outputType) ||
        cacheType.getRank() != 2 || op->getOperand(3).getType() != cacheType)
      return reject();
    auto shape = inputType.getShape();
    int64_t batch = shape[0];
    int64_t sequence = shape[inputType.getRank() == 3 ? 1 : 2];
    int64_t hidden = inputType.getRank() == 3 ? shape[2] : shape[1] * shape[3];
    int64_t rotaryDim = *dimensionAttr;
    if (rotaryDim && *headsAttr == 0)
      return reject();
    int64_t headSize =
        rotaryDim ? hidden / *headsAttr : cacheType.getShape()[1] * 2;
    if (headSize <= 0 || hidden % headSize != 0)
      return reject();
    int64_t heads = hidden / headSize;
    if ((*headsAttr && *headsAttr != heads) ||
        (inputType.getRank() == 4 &&
            (shape[1] != heads || shape[3] != headSize)))
      return reject();
    if (!rotaryDim)
      rotaryDim = headSize;
    if (rotaryDim <= 0 || rotaryDim > headSize || rotaryDim % 2 ||
        cacheType.getShape()[1] != rotaryDim / 2 ||
        batch * sequence > 1024 * 1024)
      return reject();
    bool offset =
        positionsType.getRank() == 0 ||
        (positionsType.getRank() == 1 && positionsType.getShape()[0] == 1);
    if (!offset &&
        (positionsType.getRank() != 2 || positionsType.getShape()[0] != batch ||
            positionsType.getShape()[1] != sequence))
      return reject();
    auto positions = getConstantIntValues(op->getOperand(1));
    if (failed(positions) || static_cast<int64_t>(positions->size()) !=
                                 (offset ? 1 : batch * sequence))
      return reject();
    SmallVector<int64_t> indices;
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t s = 0; s < sequence; ++s) {
        int64_t position = (*positions)[offset ? 0 : b * sequence + s];
        if (position < 0 || position >= cacheType.getShape()[0] ||
            (offset && s >= cacheType.getShape()[0] - position))
          return reject();
        indices.push_back(position + (offset ? s : 0));
      }

    Builder create(rewriter, op.getLoc());
    auto logical = create.layout(operands[0], inputType, true);
    if (failed(logical))
      return failure();
    Value bsnh = *logical;
    if (inputType.getRank() == 4)
      bsnh = create.transpose(
          bsnh, {batch, sequence, heads, headSize}, {0, 2, 1, 3});
    int64_t tokens = batch * sequence, half = rotaryDim / 2;
    Value data = create.reshape(bsnh, {tokens, heads, headSize});
    SmallVector<int64_t> first, second;
    for (int64_t i = 0; i < half; ++i) {
      first.push_back(*interleaved ? 2 * i : i);
      second.push_back(*interleaved ? 2 * i + 1 : i + half);
    }
    SmallVector<int64_t> partShape{tokens, heads, half};
    Value a = create.gather(data, partShape, first, 2);
    Value b = create.gather(data, partShape, second, 2);
    Value cos =
        create.reshape(create.gather(operands[2], {tokens, half}, indices, 0),
            {tokens, 1, half});
    Value sin =
        create.reshape(create.gather(operands[3], {tokens, half}, indices, 0),
            {tokens, 1, half});
    auto partType = create.type(partShape);
    Value ac = create.binary("tfl.mul", a, cos, partType);
    Value bs = create.binary("tfl.mul", b, sin, partType);
    Value as = create.binary("tfl.mul", a, sin, partType);
    Value bc = create.binary("tfl.mul", b, cos, partType);
    Value real = create.binary("tfl.sub", ac, bs, partType);
    Value imag = create.binary("tfl.add", as, bc, partType);
    Value rotated;
    if (*interleaved) {
      Value pairs = create.op("tfl.pack", create.type({tokens, heads, half, 2}),
          {real, imag},
          {rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(3)),
              rewriter.getNamedAttr(
                  "values_count", rewriter.getI32IntegerAttr(2))});
      rotated = create.reshape(pairs, {tokens, heads, rotaryDim});
    } else {
      rotated = create.concat({real, imag}, {tokens, heads, rotaryDim}, 2);
    }
    if (rotaryDim < headSize) {
      Value tail = create.op("tfl.slice",
          create.type({tokens, heads, headSize - rotaryDim}),
          {data,
              createI32ShapeConstant(rewriter, op.getLoc(), {0, 0, rotaryDim}),
              createI32ShapeConstant(rewriter, op.getLoc(),
                  {tokens, heads, headSize - rotaryDim})});
      rotated = create.concat({rotated, tail}, {tokens, heads, headSize}, 2);
    }
    Value result;
    if (inputType.getRank() == 3)
      result = create.reshape(rotated, shape);
    else
      result = create.transpose(
          create.reshape(rotated, {batch, sequence, heads, headSize}), shape,
          {0, 2, 1, 3});
    auto physical = create.layout(result, outputType, false);
    if (failed(physical))
      return failure();
    rewriter.replaceOp(op, *physical);
    return success();
  }
};

} // namespace

void populateLoweringONNXContribOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ContribLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
