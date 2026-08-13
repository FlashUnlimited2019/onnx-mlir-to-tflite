/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

Value f32Constant(Location loc, ArrayRef<int64_t> shape, ArrayRef<float> values,
    ConversionPatternRewriter &rewriter) {
  auto type = RankedTensorType::get(shape, rewriter.getF32Type());
  return arith::ConstantOp::create(
      rewriter, loc, type, DenseFPElementsAttr::get(type, values));
}

Value f32Splat(Location loc, ArrayRef<int64_t> shape, float value,
    ConversionPatternRewriter &rewriter) {
  auto type = RankedTensorType::get(shape, rewriter.getF32Type());
  return arith::ConstantOp::create(
      rewriter, loc, type, DenseElementsAttr::get(type, value));
}

Value binary(Location loc, StringRef name, Value lhs, Value rhs, Type type,
    ConversionPatternRewriter &rewriter) {
  return createTFLOperation(rewriter, loc, name, TypeRange{type},
      ValueRange{lhs, rhs}, {getFusedActivationNone(rewriter)})
      ->getResult(0);
}

Value unary(Location loc, StringRef name, Value input, Type type,
    ConversionPatternRewriter &rewriter) {
  return createTFLOperation(
      rewriter, loc, name, TypeRange{type}, ValueRange{input})
      ->getResult(0);
}

Value matmul(Location loc, Value lhs, Value rhs, Type type,
    ConversionPatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(true))};
  return createTFLOperation(rewriter, loc, "tfl.batch_matmul", TypeRange{type},
      ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

Value reshape(Location loc, Value input, ArrayRef<int64_t> shape,
    ConversionPatternRewriter &rewriter) {
  auto type = RankedTensorType::get(
      shape, cast<RankedTensorType>(input.getType()).getElementType());
  Value shapeValue = createI32ShapeConstant(rewriter, loc, shape);
  return createTFLOperation(rewriter, loc, "tfl.reshape", TypeRange{type},
      ValueRange{input, shapeValue})
      ->getResult(0);
}

SmallVector<Value> split(Location loc, Value input, int64_t axis,
    ArrayRef<int64_t> sizes, ArrayRef<Type> types,
    ConversionPatternRewriter &rewriter) {
  Value sizeValue = createI32ShapeConstant(rewriter, loc, sizes);
  Value axisValue = createI32ScalarTensorConstant(rewriter, loc, axis);
  Operation *operation = createTFLOperation(rewriter, loc, "tfl.split_v",
      TypeRange(types), ValueRange{input, sizeValue, axisValue},
      {rewriter.getNamedAttr(
          "num_splits", rewriter.getI32IntegerAttr(sizes.size()))});
  return SmallVector<Value>(operation->result_begin(), operation->result_end());
}

Value pack(Location loc, ValueRange values, int64_t axis, RankedTensorType type,
    ConversionPatternRewriter &rewriter) {
  return createTFLOperation(rewriter, loc, "tfl.pack", TypeRange{type}, values,
      {rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
          rewriter.getNamedAttr(
              "values_count", rewriter.getI32IntegerAttr(values.size()))})
      ->getResult(0);
}

Value clip(Location loc, Value input, float limit,
    ConversionPatternRewriter &rewriter) {
  if (limit == 0.0f)
    return input;
  Value lower = createF32ScalarTensorConstant(rewriter, loc, -limit);
  Value upper = createF32ScalarTensorConstant(rewriter, loc, limit);
  Value bounded = createTFLOperation(rewriter, loc, "tfl.maximum",
      TypeRange{input.getType()}, ValueRange{input, lower})
                      ->getResult(0);
  return createTFLOperation(rewriter, loc, "tfl.minimum",
      TypeRange{input.getType()}, ValueRange{bounded, upper})
      ->getResult(0);
}

SmallVector<Value> splitTimesteps(Location loc, Value input,
    int64_t sequenceLength, int64_t batchSize, int64_t inputSize,
    ConversionPatternRewriter &rewriter) {
  auto partType =
      RankedTensorType::get({1, batchSize, inputSize}, rewriter.getF32Type());
  SmallVector<Type> types(sequenceLength, partType);
  SmallVector<int64_t> sizes(sequenceLength, 1);
  SmallVector<Value> parts = split(loc, input, 0, sizes, types, rewriter);
  for (Value &part : parts)
    part = reshape(loc, part, {batchSize, inputSize}, rewriter);
  return parts;
}

SmallVector<Value> initialStates(Location loc, Value initial,
    int64_t directions, int64_t batchSize, int64_t hiddenSize,
    ConversionPatternRewriter &rewriter) {
  if (isa<NoneType>(initial.getType())) {
    Value zero = f32Splat(loc, {batchSize, hiddenSize}, 0.0f, rewriter);
    return SmallVector<Value>(directions, zero);
  }
  auto partType =
      RankedTensorType::get({1, batchSize, hiddenSize}, rewriter.getF32Type());
  SmallVector<Type> types(directions, partType);
  SmallVector<int64_t> sizes(directions, 1);
  SmallVector<Value> parts = split(loc, initial, 0, sizes, types, rewriter);
  for (Value &part : parts)
    part = reshape(loc, part, {batchSize, hiddenSize}, rewriter);
  return parts;
}

Value physicalSequence(Location loc, ValueRange directions,
    RankedTensorType logicalType, ConversionPatternRewriter &rewriter) {
  Value logical = pack(loc, directions, 1, logicalType, rewriter);
  auto physicalType =
      cast<RankedTensorType>(convertRank4NCHWToNHWCType(logicalType));
  Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{physicalType}, ValueRange{logical, permutation})
      ->getResult(0);
}

LogicalResult validateCommon(Operation *op, Value x, Value w, Value r, Value b,
    Value sequenceLens, Value initial, int64_t gates, StringRef direction,
    std::optional<int64_t> hiddenSizeAttribute, int64_t layout,
    RankedTensorType &xType, RankedTensorType &wType, RankedTensorType &rType,
    int64_t &directions, int64_t &sequenceLength, int64_t &batchSize,
    int64_t &inputSize, int64_t &hiddenSize) {
  if (failed(validateStaticF32Tensor(op, x.getType(), "recurrent X")) ||
      failed(validateStaticF32Tensor(op, w.getType(), "recurrent W")) ||
      failed(validateStaticF32Tensor(op, r.getType(), "recurrent R")))
    return failure();
  xType = cast<RankedTensorType>(x.getType());
  wType = cast<RankedTensorType>(w.getType());
  rType = cast<RankedTensorType>(r.getType());
  if (xType.getRank() != 3 || wType.getRank() != 3 || rType.getRank() != 3 ||
      layout != 0)
    return op->emitError("ONNXToTFL static recurrent lowering requires rank-3 "
                         "X/W/R and layout=0");
  if (direction != "forward" && direction != "reverse" &&
      direction != "bidirectional")
    return op->emitError("unsupported recurrent direction");
  directions = direction == "bidirectional" ? 2 : 1;
  sequenceLength = xType.getShape()[0];
  batchSize = xType.getShape()[1];
  inputSize = xType.getShape()[2];
  if (!hiddenSizeAttribute || *hiddenSizeAttribute <= 0)
    return op->emitError("static recurrent lowering requires hidden_size");
  hiddenSize = *hiddenSizeAttribute;
  if (sequenceLength <= 0 || batchSize <= 0 || inputSize <= 0 ||
      wType.getShape() !=
          ArrayRef<int64_t>{directions, gates * hiddenSize, inputSize} ||
      rType.getShape() !=
          ArrayRef<int64_t>{directions, gates * hiddenSize, hiddenSize})
    return op->emitError("inconsistent static recurrent tensor shapes");
  if (!isa<NoneType>(sequenceLens.getType()))
    return op->emitError(
        "static RNN/GRU lowering currently requires omitted sequence_lens");
  if (!isa<NoneType>(initial.getType())) {
    auto type = dyn_cast<RankedTensorType>(initial.getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getShape() != ArrayRef<int64_t>{directions, batchSize, hiddenSize})
      return op->emitError("inconsistent recurrent initial_h shape");
  }
  if (!isa<NoneType>(b.getType())) {
    auto type = dyn_cast<RankedTensorType>(b.getType());
    if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
        type.getShape() !=
            ArrayRef<int64_t>{directions, 2 * gates * hiddenSize})
      return op->emitError("inconsistent recurrent bias shape");
  }
  return success();
}

class RNNLowering final : public OpConversionPattern<ONNXRNNOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXRNNOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType xType, wType, rType;
    int64_t directions, sequenceLength, batchSize, inputSize, hiddenSize;
    if (failed(validateCommon(op, adaptor.getX(), adaptor.getW(),
            adaptor.getR(), adaptor.getB(), adaptor.getSequenceLens(),
            adaptor.getInitialH(), 1, op.getDirection(), op.getHiddenSize(),
            op.getLayout(), xType, wType, rType, directions, sequenceLength,
            batchSize, inputSize, hiddenSize)))
      return failure();
    auto yType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto yhType = dyn_cast<RankedTensorType>(op.getYH().getType());
    if (!yType || !yhType ||
        yType.getShape() != ArrayRef<int64_t>{sequenceLength, directions,
                                batchSize, hiddenSize} ||
        yhType.getShape() !=
            ArrayRef<int64_t>{directions, batchSize, hiddenSize})
      return op.emitError("inconsistent RNN output shapes"), failure();
    FailureOr<SmallVector<float>> wValues = getConstantF32Values(op.getW());
    FailureOr<SmallVector<float>> rValues = getConstantF32Values(op.getR());
    if (failed(wValues) || failed(rValues))
      return op.emitError("RNN W/R must be constant FP32 tensors"), failure();
    SmallVector<float> bValues;
    if (!isa<NoneType>(op.getB().getType())) {
      FailureOr<SmallVector<float>> values = getConstantF32Values(op.getB());
      if (failed(values))
        return op.emitError("RNN B must be a constant FP32 tensor"), failure();
      bValues = std::move(*values);
    }
    ArrayAttr activations = op.getActivations();
    if (activations.empty() ||
        (cast<StringAttr>(activations[0]).getValue() != "Tanh" &&
            cast<StringAttr>(activations[0]).getValue() != "Relu") ||
        op.getActivationAlpha().has_value() ||
        op.getActivationBeta().has_value())
      return op.emitError("static RNN supports only Tanh or Relu activation"),
             failure();
    float clipLimit = 0.0f;
    if (std::optional<APFloat> value = op.getClip()) {
      clipLimit = value->convertToFloat();
      if (clipLimit <= 0.0f)
        return op.emitError("RNN clip must be positive"), failure();
    }

    Location loc = op.getLoc();
    auto stateType =
        RankedTensorType::get({batchSize, hiddenSize}, rewriter.getF32Type());
    SmallVector<Value> timesteps = splitTimesteps(
        loc, adaptor.getX(), sequenceLength, batchSize, inputSize, rewriter);
    SmallVector<Value> states = initialStates(loc, adaptor.getInitialH(),
        directions, batchSize, hiddenSize, rewriter);
    SmallVector<Value> sequences;
    SmallVector<Value> finals;
    for (int64_t d = 0; d < directions; ++d) {
      int64_t wOffset = d * hiddenSize * inputSize;
      int64_t rOffset = d * hiddenSize * hiddenSize;
      Value w = f32Constant(loc, {hiddenSize, inputSize},
          ArrayRef<float>(*wValues).slice(wOffset, hiddenSize * inputSize),
          rewriter);
      Value r = f32Constant(loc, {hiddenSize, hiddenSize},
          ArrayRef<float>(*rValues).slice(rOffset, hiddenSize * hiddenSize),
          rewriter);
      SmallVector<float> bias(hiddenSize, 0.0f);
      if (!bValues.empty())
        for (int64_t i = 0; i < hiddenSize; ++i)
          bias[i] = bValues[d * 2 * hiddenSize + i] +
                    bValues[d * 2 * hiddenSize + hiddenSize + i];
      Value b = f32Constant(loc, {hiddenSize}, bias, rewriter);
      Value hidden = states[d];
      SmallVector<Value> outputs(sequenceLength);
      bool reverse = op.getDirection() == "reverse" ||
                     (op.getDirection() == "bidirectional" && d == 1);
      StringRef activation = cast<StringAttr>(
          activations[std::min<int64_t>(d, activations.size() - 1)])
                                 .getValue();
      for (int64_t iteration = 0; iteration < sequenceLength; ++iteration) {
        int64_t t = reverse ? sequenceLength - 1 - iteration : iteration;
        Value affine = binary(loc, "tfl.add",
            matmul(loc, timesteps[t], w, stateType, rewriter),
            matmul(loc, hidden, r, stateType, rewriter), stateType, rewriter);
        affine = binary(loc, "tfl.add", affine, b, stateType, rewriter);
        affine = clip(loc, affine, clipLimit, rewriter);
        hidden = unary(loc, activation == "Relu" ? "tfl.relu" : "tfl.tanh",
            affine, stateType, rewriter);
        outputs[t] = hidden;
      }
      sequences.push_back(pack(loc, outputs, 0,
          RankedTensorType::get(
              {sequenceLength, batchSize, hiddenSize}, rewriter.getF32Type()),
          rewriter));
      finals.push_back(hidden);
    }
    rewriter.replaceOp(
        op, ValueRange{physicalSequence(loc, sequences, yType, rewriter),
                pack(loc, finals, 0, yhType, rewriter)});
    return success();
  }
};

class GRULowering final : public OpConversionPattern<ONNXGRUOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGRUOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType xType, wType, rType;
    int64_t directions, sequenceLength, batchSize, inputSize, hiddenSize;
    if (failed(validateCommon(op, adaptor.getX(), adaptor.getW(),
            adaptor.getR(), adaptor.getB(), adaptor.getSequenceLens(),
            adaptor.getInitialH(), 3, op.getDirection(), op.getHiddenSize(),
            op.getLayout(), xType, wType, rType, directions, sequenceLength,
            batchSize, inputSize, hiddenSize)))
      return failure();
    auto yType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto yhType = dyn_cast<RankedTensorType>(op.getYH().getType());
    if (!yType || !yhType ||
        yType.getShape() != ArrayRef<int64_t>{sequenceLength, directions,
                                batchSize, hiddenSize} ||
        yhType.getShape() !=
            ArrayRef<int64_t>{directions, batchSize, hiddenSize})
      return op.emitError("inconsistent GRU output shapes"), failure();
    if (op.getActivationAlpha().has_value() ||
        op.getActivationBeta().has_value() || op.getActivations().has_value())
      return op.emitError(
                 "static GRU supports only default Sigmoid/Tanh activations"),
             failure();
    FailureOr<SmallVector<float>> wValues = getConstantF32Values(op.getW());
    FailureOr<SmallVector<float>> rValues = getConstantF32Values(op.getR());
    if (failed(wValues) || failed(rValues))
      return op.emitError("GRU W/R must be constant FP32 tensors"), failure();
    SmallVector<float> bValues;
    if (!isa<NoneType>(op.getB().getType())) {
      FailureOr<SmallVector<float>> values = getConstantF32Values(op.getB());
      if (failed(values))
        return op.emitError("GRU B must be a constant FP32 tensor"), failure();
      bValues = std::move(*values);
    }
    float clipLimit = 0.0f;
    if (std::optional<APFloat> value = op.getClip()) {
      clipLimit = value->convertToFloat();
      if (clipLimit <= 0.0f)
        return op.emitError("GRU clip must be positive"), failure();
    }

    Location loc = op.getLoc();
    auto stateType =
        RankedTensorType::get({batchSize, hiddenSize}, rewriter.getF32Type());
    SmallVector<Value> timesteps = splitTimesteps(
        loc, adaptor.getX(), sequenceLength, batchSize, inputSize, rewriter);
    SmallVector<Value> states = initialStates(loc, adaptor.getInitialH(),
        directions, batchSize, hiddenSize, rewriter);
    SmallVector<Value> sequences;
    SmallVector<Value> finals;
    for (int64_t d = 0; d < directions; ++d) {
      SmallVector<Value> ws, rs, wbs, rbs;
      for (int64_t gate = 0; gate < 3; ++gate) {
        int64_t wOffset = (d * 3 * hiddenSize + gate * hiddenSize) * inputSize;
        int64_t rOffset = (d * 3 * hiddenSize + gate * hiddenSize) * hiddenSize;
        ws.push_back(f32Constant(loc, {hiddenSize, inputSize},
            ArrayRef<float>(*wValues).slice(wOffset, hiddenSize * inputSize),
            rewriter));
        rs.push_back(f32Constant(loc, {hiddenSize, hiddenSize},
            ArrayRef<float>(*rValues).slice(rOffset, hiddenSize * hiddenSize),
            rewriter));
        SmallVector<float> wb(hiddenSize, 0.0f), rb(hiddenSize, 0.0f);
        if (!bValues.empty())
          for (int64_t i = 0; i < hiddenSize; ++i) {
            wb[i] = bValues[d * 6 * hiddenSize + gate * hiddenSize + i];
            rb[i] = bValues[d * 6 * hiddenSize + 3 * hiddenSize +
                            gate * hiddenSize + i];
          }
        wbs.push_back(f32Constant(loc, {hiddenSize}, wb, rewriter));
        rbs.push_back(f32Constant(loc, {hiddenSize}, rb, rewriter));
      }
      Value hidden = states[d];
      SmallVector<Value> outputs(sequenceLength);
      bool reverse = op.getDirection() == "reverse" ||
                     (op.getDirection() == "bidirectional" && d == 1);
      Value one = createF32ScalarTensorConstant(rewriter, loc, 1.0f);
      for (int64_t iteration = 0; iteration < sequenceLength; ++iteration) {
        int64_t t = reverse ? sequenceLength - 1 - iteration : iteration;
        auto gateAffine = [&](int64_t gate) {
          Value xProjection = binary(loc, "tfl.add",
              matmul(loc, timesteps[t], ws[gate], stateType, rewriter),
              wbs[gate], stateType, rewriter);
          Value hProjection = binary(loc, "tfl.add",
              matmul(loc, hidden, rs[gate], stateType, rewriter), rbs[gate],
              stateType, rewriter);
          return clip(loc,
              binary(loc, "tfl.add", xProjection, hProjection, stateType,
                  rewriter),
              clipLimit, rewriter);
        };
        Value update =
            unary(loc, "tfl.logistic", gateAffine(0), stateType, rewriter);
        Value reset =
            unary(loc, "tfl.logistic", gateAffine(1), stateType, rewriter);
        Value candidateInput = binary(loc, "tfl.add",
            matmul(loc, timesteps[t], ws[2], stateType, rewriter), wbs[2],
            stateType, rewriter);
        Value candidateRecurrent;
        if (op.getLinearBeforeReset() != 0) {
          Value recurrent = binary(loc, "tfl.add",
              matmul(loc, hidden, rs[2], stateType, rewriter), rbs[2],
              stateType, rewriter);
          candidateRecurrent =
              binary(loc, "tfl.mul", reset, recurrent, stateType, rewriter);
        } else {
          Value gatedHidden =
              binary(loc, "tfl.mul", reset, hidden, stateType, rewriter);
          candidateRecurrent = binary(loc, "tfl.add",
              matmul(loc, gatedHidden, rs[2], stateType, rewriter), rbs[2],
              stateType, rewriter);
        }
        Value candidateAffine = clip(loc,
            binary(loc, "tfl.add", candidateInput, candidateRecurrent,
                stateType, rewriter),
            clipLimit, rewriter);
        Value candidate =
            unary(loc, "tfl.tanh", candidateAffine, stateType, rewriter);
        Value inverseUpdate =
            binary(loc, "tfl.sub", one, update, stateType, rewriter);
        hidden = binary(loc, "tfl.add",
            binary(
                loc, "tfl.mul", inverseUpdate, candidate, stateType, rewriter),
            binary(loc, "tfl.mul", update, hidden, stateType, rewriter),
            stateType, rewriter);
        outputs[t] = hidden;
      }
      sequences.push_back(pack(loc, outputs, 0,
          RankedTensorType::get(
              {sequenceLength, batchSize, hiddenSize}, rewriter.getF32Type()),
          rewriter));
      finals.push_back(hidden);
    }
    rewriter.replaceOp(
        op, ValueRange{physicalSequence(loc, sequences, yType, rewriter),
                pack(loc, finals, 0, yhType, rewriter)});
    return success();
  }
};

} // namespace

void populateLoweringONNXStaticRecurrentOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<RNNLowering, GRULowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
