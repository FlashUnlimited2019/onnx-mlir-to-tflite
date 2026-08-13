/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<DenseElementsAttr> getDenseElements(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<ONNXConstantOp>())
    attribute = constant.getValueAttr();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  if (auto disposable = dyn_cast_or_null<DisposableElementsAttr>(attribute))
    attribute = disposable.toDenseElementsAttr();
  auto elements = dyn_cast_or_null<DenseElementsAttr>(attribute);
  if (!elements)
    return failure();
  return elements;
}

FailureOr<SmallVector<float>> getConstantF32Values(Value value) {
  FailureOr<DenseElementsAttr> elements = getDenseElements(value);
  if (failed(elements) || !elements->getElementType().isF32())
    return failure();
  SmallVector<float> values;
  values.reserve(elements->size());
  for (APFloat item : elements->getValues<APFloat>())
    values.push_back(item.convertToFloat());
  return values;
}

Value createF32Constant(Location loc, ArrayRef<int64_t> shape,
    ArrayRef<float> values, ConversionPatternRewriter &rewriter) {
  auto type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto attribute = DenseFPElementsAttr::get(type, values);
  return arith::ConstantOp::create(rewriter, loc, type, attribute);
}

Value createF32Splat(Location loc, ArrayRef<int64_t> shape, float value,
    ConversionPatternRewriter &rewriter) {
  auto type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto attribute = DenseElementsAttr::get(type, value);
  return arith::ConstantOp::create(rewriter, loc, type, attribute);
}

Value createBinary(StringRef name, Location loc, Value lhs, Value rhs,
    Type resultType, ConversionPatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
  return createTFLOperation(rewriter, loc, name, TypeRange{resultType},
      ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

Value createUnary(StringRef name, Location loc, Value input, Type resultType,
    ConversionPatternRewriter &rewriter) {
  return createTFLOperation(
      rewriter, loc, name, TypeRange{resultType}, ValueRange{input})
      ->getResult(0);
}

Value createBatchMatMul(Location loc, Value lhs, Value rhs, Type resultType,
    ConversionPatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(true))};
  return createTFLOperation(rewriter, loc, "tfl.batch_matmul",
      TypeRange{resultType}, ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

SmallVector<Value> createSplitV(Location loc, Value input, int64_t axis,
    ArrayRef<int64_t> splitSizes, ArrayRef<Type> resultTypes,
    ConversionPatternRewriter &rewriter) {
  Value sizes = createI32ShapeConstant(rewriter, loc, splitSizes);
  Value axisValue =
      createI32ScalarTensorConstant(rewriter, loc, static_cast<int32_t>(axis));
  SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
      "num_splits", rewriter.getI32IntegerAttr(splitSizes.size()))};
  Operation *split = createTFLOperation(rewriter, loc, "tfl.split_v",
      TypeRange{resultTypes}, ValueRange{input, sizes, axisValue}, attributes);
  return SmallVector<Value>(split->result_begin(), split->result_end());
}

Value createReshape(Location loc, Value input, ArrayRef<int64_t> shape,
    ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = RankedTensorType::get(shape, inputType.getElementType());
  Value shapeValue = createI32ShapeConstant(rewriter, loc, shape);
  return createTFLOperation(rewriter, loc, "tfl.reshape", TypeRange{resultType},
      ValueRange{input, shapeValue})
      ->getResult(0);
}

Value createPack(Location loc, ValueRange inputs, int64_t axis,
    RankedTensorType resultType, ConversionPatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
      rewriter.getNamedAttr(
          "values_count", rewriter.getI32IntegerAttr(inputs.size()))};
  return createTFLOperation(
      rewriter, loc, "tfl.pack", TypeRange{resultType}, inputs, attributes)
      ->getResult(0);
}

Value clipActivationInput(Location loc, Value input, float clip,
    ConversionPatternRewriter &rewriter) {
  if (clip <= 0.0f)
    return input;
  Value lower = createF32ScalarTensorConstant(rewriter, loc, -clip);
  Value upper = createF32ScalarTensorConstant(rewriter, loc, clip);
  Value bounded = createTFLOperation(rewriter, loc, "tfl.maximum",
      TypeRange{input.getType()}, ValueRange{input, lower})
                      ->getResult(0);
  return createTFLOperation(rewriter, loc, "tfl.minimum",
      TypeRange{input.getType()}, ValueRange{bounded, upper})
      ->getResult(0);
}

struct DirectionParameters {
  Value inputWeights;
  Value recurrentWeights;
  Value bias;
};

struct DirectionResult {
  Value sequence;
  Value finalHidden;
  Value finalCell;
};

class LSTMLowering final : public OpConversionPattern<ONNXLSTMOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLSTMOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getX();
    Value w = adaptor.getW();
    Value r = adaptor.getR();
    Value b = adaptor.getB();
    Value sequenceLens = adaptor.getSequenceLens();
    Value initialH = adaptor.getInitialH();
    Value initialC = adaptor.getInitialC();
    Value peepholes = adaptor.getP();

    if (failed(validateStaticF32Tensor(op, x.getType(), "LSTM X")) ||
        failed(validateStaticF32Tensor(op, w.getType(), "LSTM W")) ||
        failed(validateStaticF32Tensor(op, r.getType(), "LSTM R")))
      return failure();
    auto xType = cast<RankedTensorType>(x.getType());
    auto wType = cast<RankedTensorType>(w.getType());
    auto rType = cast<RankedTensorType>(r.getType());
    if (xType.getRank() != 3 || wType.getRank() != 3 || rType.getRank() != 3)
      return op.emitError(
                 "unsupported LSTM shapes: X, W, and R must have rank 3"),
             failure();
    if (op.getLayout() != 0)
      return op.emitError(
                 "unsupported LSTM layout: only layout=0 is supported"),
             failure();
    if (op.getInputForget() != 0)
      return op.emitError("unsupported LSTM input_forget: coupled gates are "
                          "not supported"),
             failure();
    if (!isa<NoneType>(peepholes.getType()))
      return op.emitError("unsupported LSTM peephole input P"), failure();
    if (op.getActivationAlpha().has_value() ||
        op.getActivationBeta().has_value() || op.getActivations().has_value())
      return op.emitError(
                 "unsupported LSTM activations: only default Sigmoid/Tanh/Tanh "
                 "activations are supported"),
             failure();

    StringRef direction = op.getDirection();
    if (direction != "forward" && direction != "reverse" &&
        direction != "bidirectional")
      return op.emitError() << "unsupported LSTM direction: " << direction,
             failure();
    int64_t numDirections = direction == "bidirectional" ? 2 : 1;
    int64_t sequenceLength = xType.getShape()[0];
    int64_t batchSize = xType.getShape()[1];
    int64_t inputSize = xType.getShape()[2];
    std::optional<int64_t> hiddenSizeAttribute = op.getHiddenSize();
    if (!hiddenSizeAttribute.has_value() || *hiddenSizeAttribute <= 0)
      return op.emitError("LSTM requires a positive hidden_size attribute"),
             failure();
    int64_t hiddenSize = *hiddenSizeAttribute;
    if (sequenceLength <= 0 || batchSize <= 0 || inputSize <= 0)
      return op.emitError("LSTM requires non-empty static dimensions"),
             failure();
    if (wType.getShape() !=
            ArrayRef<int64_t>{numDirections, 4 * hiddenSize, inputSize} ||
        rType.getShape() !=
            ArrayRef<int64_t>{numDirections, 4 * hiddenSize, hiddenSize})
      return op.emitError("invalid LSTM W/R shape for the selected direction, "
                          "input size, and hidden size"),
             failure();

    auto yType = dyn_cast<RankedTensorType>(op.getY().getType());
    auto yhType = dyn_cast<RankedTensorType>(op.getYH().getType());
    auto ycType = dyn_cast<RankedTensorType>(op.getYC().getType());
    if (!yType || !yhType || !ycType || !yType.hasStaticShape() ||
        !yhType.hasStaticShape() || !ycType.hasStaticShape() ||
        !yType.getElementType().isF32() || !yhType.getElementType().isF32() ||
        !ycType.getElementType().isF32() ||
        yType.getShape() != ArrayRef<int64_t>{sequenceLength, numDirections,
                                batchSize, hiddenSize} ||
        yhType.getShape() !=
            ArrayRef<int64_t>{numDirections, batchSize, hiddenSize} ||
        ycType.getShape() !=
            ArrayRef<int64_t>{numDirections, batchSize, hiddenSize})
      return op.emitError("unsupported or inconsistent LSTM output shapes"),
             failure();

    FailureOr<SmallVector<float>> wValues = getConstantF32Values(w);
    FailureOr<SmallVector<float>> rValues = getConstantF32Values(r);
    if (failed(wValues) || failed(rValues))
      return op.emitError("LSTM W and R must be constant f32 tensors"),
             failure();

    SmallVector<float> bValues;
    if (!isa<NoneType>(b.getType())) {
      auto bType = dyn_cast<RankedTensorType>(b.getType());
      FailureOr<SmallVector<float>> values = getConstantF32Values(b);
      if (!bType || !bType.hasStaticShape() ||
          bType.getShape() !=
              ArrayRef<int64_t>{numDirections, 8 * hiddenSize} ||
          failed(values))
        return op.emitError("LSTM B must be a constant f32 tensor with shape "
                            "[num_directions, 8*hidden_size]"),
               failure();
      bValues = std::move(*values);
    }

    SmallVector<int64_t> lens(batchSize, sequenceLength);
    bool hasVariableSequenceLengths = false;
    if (!isa<NoneType>(sequenceLens.getType())) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(sequenceLens);
      if (failed(values) || static_cast<int64_t>(values->size()) != batchSize)
        return op.emitError(
                   "LSTM sequence_lens must be a constant integer tensor with "
                   "one value per batch"),
               failure();
      for (auto [index, value] : llvm::enumerate(*values)) {
        if (value < 0 || value > sequenceLength)
          return op.emitError("LSTM sequence_lens value is out of range"),
                 failure();
        lens[index] = value;
        hasVariableSequenceLengths |= value != sequenceLength;
      }
    }

    if (failed(validateInitialState(
            op, initialH, numDirections, batchSize, hiddenSize, "initial_h")) ||
        failed(validateInitialState(
            op, initialC, numDirections, batchSize, hiddenSize, "initial_c")))
      return failure();

    SmallVector<Type> xPartTypes(
        sequenceLength, RankedTensorType::get(
                            {1, batchSize, inputSize}, rewriter.getF32Type()));
    SmallVector<int64_t> xSplitSizes(sequenceLength, 1);
    SmallVector<Value> xParts =
        createSplitV(loc, x, 0, xSplitSizes, xPartTypes, rewriter);
    SmallVector<Value> timesteps;
    timesteps.reserve(sequenceLength);
    for (Value part : xParts)
      timesteps.push_back(
          createReshape(loc, part, {batchSize, inputSize}, rewriter));

    SmallVector<Value> initialHiddenParts = splitInitialState(
        loc, initialH, numDirections, batchSize, hiddenSize, rewriter);
    SmallVector<Value> initialCellParts = splitInitialState(
        loc, initialC, numDirections, batchSize, hiddenSize, rewriter);

    SmallVector<DirectionParameters> parameters;
    parameters.reserve(numDirections);
    for (int64_t d = 0; d < numDirections; ++d) {
      DirectionParameters parameter;
      int64_t wOffset = d * 4 * hiddenSize * inputSize;
      parameter.inputWeights = createF32Constant(loc,
          {4 * hiddenSize, inputSize},
          ArrayRef<float>(*wValues).slice(wOffset, 4 * hiddenSize * inputSize),
          rewriter);
      int64_t rOffset = d * 4 * hiddenSize * hiddenSize;
      parameter.recurrentWeights = createF32Constant(loc,
          {4 * hiddenSize, hiddenSize},
          ArrayRef<float>(*rValues).slice(rOffset, 4 * hiddenSize * hiddenSize),
          rewriter);
      SmallVector<float> combinedBias(4 * hiddenSize, 0.0f);
      if (!bValues.empty()) {
        int64_t bOffset = d * 8 * hiddenSize;
        for (int64_t i = 0; i < 4 * hiddenSize; ++i)
          combinedBias[i] =
              bValues[bOffset + i] + bValues[bOffset + 4 * hiddenSize + i];
      }
      parameter.bias =
          createF32Constant(loc, {4 * hiddenSize}, combinedBias, rewriter);
      parameters.push_back(parameter);
    }

    float clip = 0.0f;
    if (std::optional<APFloat> clipAttribute = op.getClip()) {
      clip = clipAttribute->convertToFloat();
      if (clip <= 0.0f)
        return op.emitError("LSTM clip must be positive"), failure();
    }

    SmallVector<DirectionResult> directionResults;
    directionResults.reserve(numDirections);
    for (int64_t d = 0; d < numDirections; ++d) {
      bool reverse =
          direction == "reverse" || (direction == "bidirectional" && d == 1);
      directionResults.push_back(
          lowerDirection(op, timesteps, parameters[d], initialHiddenParts[d],
              initialCellParts[d], lens, hasVariableSequenceLengths, reverse,
              clip, batchSize, hiddenSize, rewriter));
    }

    SmallVector<Value> sequences;
    SmallVector<Value> finalHidden;
    SmallVector<Value> finalCell;
    for (const DirectionResult &result : directionResults) {
      sequences.push_back(result.sequence);
      finalHidden.push_back(result.finalHidden);
      finalCell.push_back(result.finalCell);
    }

    Value logicalY = createPack(loc, sequences, 1, yType, rewriter);
    auto physicalYType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(yType));
    Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
    Value physicalY = createTFLOperation(rewriter, loc, "tfl.transpose",
        TypeRange{physicalYType}, ValueRange{logicalY, permutation})
                          ->getResult(0);
    Value yH = createPack(loc, finalHidden, 0, yhType, rewriter);
    Value yC = createPack(loc, finalCell, 0, ycType, rewriter);
    rewriter.replaceOp(op, ValueRange{physicalY, yH, yC});
    return success();
  }

private:
  LogicalResult validateInitialState(ONNXLSTMOp op, Value state,
      int64_t numDirections, int64_t batchSize, int64_t hiddenSize,
      StringRef name) const {
    if (isa<NoneType>(state.getType()))
      return success();
    if (failed(validateStaticF32Tensor(op, state.getType(), name)))
      return failure();
    auto type = cast<RankedTensorType>(state.getType());
    if (type.getShape() !=
        ArrayRef<int64_t>{numDirections, batchSize, hiddenSize})
      return op.emitError() << "invalid LSTM " << name << " shape", failure();
    return success();
  }

  SmallVector<Value> splitInitialState(Location loc, Value state,
      int64_t numDirections, int64_t batchSize, int64_t hiddenSize,
      ConversionPatternRewriter &rewriter) const {
    if (isa<NoneType>(state.getType())) {
      Value zero = createF32Splat(loc, {batchSize, hiddenSize}, 0.0f, rewriter);
      return SmallVector<Value>(numDirections, zero);
    }
    SmallVector<Type> partTypes(
        numDirections, RankedTensorType::get(
                           {1, batchSize, hiddenSize}, rewriter.getF32Type()));
    SmallVector<int64_t> splitSizes(numDirections, 1);
    SmallVector<Value> parts =
        createSplitV(loc, state, 0, splitSizes, partTypes, rewriter);
    SmallVector<Value> result;
    result.reserve(numDirections);
    for (Value part : parts)
      result.push_back(
          createReshape(loc, part, {batchSize, hiddenSize}, rewriter));
    return result;
  }

  DirectionResult lowerDirection(ONNXLSTMOp op, ArrayRef<Value> timesteps,
      const DirectionParameters &parameters, Value initialHidden,
      Value initialCell, ArrayRef<int64_t> lens,
      bool hasVariableSequenceLengths, bool reverse, float clip,
      int64_t batchSize, int64_t hiddenSize,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto stateType =
        RankedTensorType::get({batchSize, hiddenSize}, rewriter.getF32Type());
    auto gatesType = RankedTensorType::get(
        {batchSize, 4 * hiddenSize}, rewriter.getF32Type());
    Value hidden = initialHidden;
    Value cell = initialCell;
    SmallVector<Value> outputs(timesteps.size());

    for (int64_t iteration = 0, end = static_cast<int64_t>(timesteps.size());
        iteration < end; ++iteration) {
      int64_t timestep = reverse ? end - 1 - iteration : iteration;
      Value inputProjection = createBatchMatMul(loc, timesteps[timestep],
          parameters.inputWeights, gatesType, rewriter);
      Value recurrentProjection = createBatchMatMul(
          loc, hidden, parameters.recurrentWeights, gatesType, rewriter);
      Value gates = createBinary("tfl.add", loc, inputProjection,
          recurrentProjection, gatesType, rewriter);
      gates = createBinary(
          "tfl.add", loc, gates, parameters.bias, gatesType, rewriter);
      gates = clipActivationInput(loc, gates, clip, rewriter);

      SmallVector<Type> gateTypes(4, stateType);
      SmallVector<int64_t> gateSizes(4, hiddenSize);
      SmallVector<Value> gateValues =
          createSplitV(loc, gates, 1, gateSizes, gateTypes, rewriter);
      Value inputGate =
          createUnary("tfl.logistic", loc, gateValues[0], stateType, rewriter);
      Value outputGate =
          createUnary("tfl.logistic", loc, gateValues[1], stateType, rewriter);
      Value forgetGate =
          createUnary("tfl.logistic", loc, gateValues[2], stateType, rewriter);
      Value cellGate =
          createUnary("tfl.tanh", loc, gateValues[3], stateType, rewriter);

      Value retainedCell =
          createBinary("tfl.mul", loc, forgetGate, cell, stateType, rewriter);
      Value candidateCell = createBinary(
          "tfl.mul", loc, inputGate, cellGate, stateType, rewriter);
      Value nextCell = createBinary(
          "tfl.add", loc, retainedCell, candidateCell, stateType, rewriter);
      // ONNX applies `clip` to the affine gate inputs. The cell state passed
      // to the final h activation is not clipped (matching ONNX Runtime).
      Value activatedCell =
          createUnary("tfl.tanh", loc, nextCell, stateType, rewriter);
      Value nextHidden = createBinary(
          "tfl.mul", loc, outputGate, activatedCell, stateType, rewriter);

      Value output = nextHidden;
      if (hasVariableSequenceLengths) {
        SmallVector<float> maskValues;
        maskValues.reserve(batchSize);
        for (int64_t batch = 0; batch < batchSize; ++batch)
          maskValues.push_back(timestep < lens[batch] ? 1.0f : 0.0f);
        Value mask =
            createF32Constant(loc, {batchSize, 1}, maskValues, rewriter);
        SmallVector<float> inverseValues;
        inverseValues.reserve(batchSize);
        for (float value : maskValues)
          inverseValues.push_back(1.0f - value);
        Value inverseMask =
            createF32Constant(loc, {batchSize, 1}, inverseValues, rewriter);
        Value maskedNextHidden =
            createBinary("tfl.mul", loc, mask, nextHidden, stateType, rewriter);
        Value maskedOldHidden = createBinary(
            "tfl.mul", loc, inverseMask, hidden, stateType, rewriter);
        hidden = createBinary("tfl.add", loc, maskedNextHidden, maskedOldHidden,
            stateType, rewriter);
        Value maskedNextCell =
            createBinary("tfl.mul", loc, mask, nextCell, stateType, rewriter);
        Value maskedOldCell = createBinary(
            "tfl.mul", loc, inverseMask, cell, stateType, rewriter);
        cell = createBinary(
            "tfl.add", loc, maskedNextCell, maskedOldCell, stateType, rewriter);
        output = maskedNextHidden;
      } else {
        hidden = nextHidden;
        cell = nextCell;
      }
      outputs[timestep] = output;
    }

    auto sequenceType = RankedTensorType::get(
        {static_cast<int64_t>(timesteps.size()), batchSize, hiddenSize},
        rewriter.getF32Type());
    Value sequence = createPack(loc, outputs, 0, sequenceType, rewriter);
    return {sequence, hidden, cell};
  }
};

} // namespace

void populateLoweringONNXLSTMOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<LSTMLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
