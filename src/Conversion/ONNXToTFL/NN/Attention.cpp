/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <cmath>
#include <optional>

using namespace mlir;

namespace onnx_mlir {
namespace {

int64_t getIntegerAttributeOr(
    Operation *op, StringRef name, int64_t defaultValue) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getValue().getSExtValue();
  return defaultValue;
}

std::optional<float> getOptionalFloatAttribute(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return static_cast<float>(attr.getValueAsDouble());
  return std::nullopt;
}

bool isAbsentOperand(Operation *op, unsigned index) {
  return index >= op->getNumOperands() ||
         isa<NoneType>(op->getOperand(index).getType());
}

FailureOr<RankedTensorType> getStaticF32Type(
    Operation *op, Value value, StringRef role) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32()) {
    op->emitError() << "static Attention requires a ranked static FP32 " << role
                    << " tensor, got " << value.getType();
    return failure();
  }
  return type;
}

class TFLAttentionBuilder {
public:
  TFLAttentionBuilder(Operation *sourceOp, ConversionPatternRewriter &rewriter)
      : rewriter(rewriter), loc(sourceOp->getLoc()) {}

  RankedTensorType f32Tensor(ArrayRef<int64_t> shape) const {
    return RankedTensorType::get(shape, rewriter.getF32Type());
  }

  Value f32Constant(ArrayRef<int64_t> shape, ArrayRef<float> values) {
    auto type = f32Tensor(shape);
    return arith::ConstantOp::create(
        rewriter, loc, type, DenseFPElementsAttr::get(type, values));
  }

  Value i32Constant(ArrayRef<int64_t> shape, ArrayRef<int32_t> values) {
    auto type = RankedTensorType::get(shape, rewriter.getI32Type());
    return arith::ConstantOp::create(
        rewriter, loc, type, DenseIntElementsAttr::get(type, values));
  }

  Value reshape(Value input, ArrayRef<int64_t> shape) {
    Value shapeValue = createI32ShapeConstant(rewriter, loc, shape);
    return createTFLOperation(rewriter, loc, "tfl.reshape",
        TypeRange{f32Tensor(shape)}, ValueRange{input, shapeValue})
        ->getResult(0);
  }

  Value transpose(Value input, ArrayRef<int64_t> resultShape,
      ArrayRef<int64_t> permutation) {
    Value perm = createI32ShapeConstant(rewriter, loc, permutation);
    return createTFLOperation(rewriter, loc, "tfl.transpose",
        TypeRange{f32Tensor(resultShape)}, ValueRange{input, perm})
        ->getResult(0);
  }

  Value batchMatMul(Value lhs, Value rhs, ArrayRef<int64_t> resultShape,
      bool adjointRhs = false) {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(adjointRhs))};
    return createTFLOperation(rewriter, loc, "tfl.batch_matmul",
        TypeRange{f32Tensor(resultShape)}, ValueRange{lhs, rhs}, attributes)
        ->getResult(0);
  }

  Value binary(StringRef name, Value lhs, Value rhs, Type resultType) {
    SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
    return createTFLOperation(rewriter, loc, name, TypeRange{resultType},
        ValueRange{lhs, rhs}, attributes)
        ->getResult(0);
  }

  Value slice(Value input, ArrayRef<int64_t> begin, ArrayRef<int64_t> size) {
    Value beginValue = createI32ShapeConstant(rewriter, loc, begin);
    Value sizeValue = createI32ShapeConstant(rewriter, loc, size);
    return createTFLOperation(rewriter, loc, "tfl.slice",
        TypeRange{f32Tensor(size)}, ValueRange{input, beginValue, sizeValue})
        ->getResult(0);
  }

  Value gatherHeads(Value input, int64_t batch, int64_t sourceHeads,
      int64_t targetHeads, int64_t sequence, int64_t headSize) {
    SmallVector<int32_t> indices;
    indices.reserve(batch * targetHeads);
    int64_t repeats = targetHeads / sourceHeads;
    for (int64_t b = 0; b < batch; ++b)
      for (int64_t head = 0; head < targetHeads; ++head)
        indices.push_back(
            static_cast<int32_t>(b * sourceHeads + head / repeats));
    Value indexValue =
        i32Constant({batch * targetHeads}, ArrayRef<int32_t>(indices));
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
    return createTFLOperation(rewriter, loc, "tfl.gather",
        TypeRange{f32Tensor({batch * targetHeads, sequence, headSize})},
        ValueRange{input, indexValue}, attributes)
        ->getResult(0);
  }

  Value softmax(Value input, ArrayRef<int64_t> shape) {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(1.0f))};
    return createTFLOperation(rewriter, loc, "tfl.softmax",
        TypeRange{f32Tensor(shape)}, ValueRange{input}, attributes)
        ->getResult(0);
  }

private:
  ConversionPatternRewriter &rewriter;
  Location loc;
};

struct FlattenedHeads {
  Value value;
  int64_t batch;
  int64_t heads;
  int64_t sequence;
  int64_t headSize;
};

FailureOr<FlattenedHeads> flattenHeads(Operation *op, Value physicalValue,
    RankedTensorType logicalType, int64_t requestedHeads,
    TFLAttentionBuilder &create) {
  ArrayRef<int64_t> shape = logicalType.getShape();
  if (logicalType.getRank() != 3 && logicalType.getRank() != 4)
    return op->emitError(
               "static Attention supports rank-3 BSH or rank-4 BNSH data"),
           failure();

  int64_t batch = shape[0];
  int64_t heads;
  int64_t sequence;
  int64_t headSize;
  Value logicalBNSH;
  if (logicalType.getRank() == 3) {
    heads = requestedHeads;
    sequence = shape[1];
    if (heads <= 0 || shape[2] % heads != 0)
      return op->emitError(
                 "rank-3 Attention hidden size must divide by head count"),
             failure();
    headSize = shape[2] / heads;
    Value bsnh =
        create.reshape(physicalValue, {batch, sequence, heads, headSize});
    logicalBNSH = create.transpose(
        bsnh, {batch, heads, sequence, headSize}, {0, 2, 1, 3});
  } else {
    heads = shape[1];
    sequence = shape[2];
    headSize = shape[3];
    if (requestedHeads > 0 && requestedHeads != heads)
      return op->emitError(
                 "rank-4 Attention head attribute disagrees with BNSH shape"),
             failure();
    // Every rank-4 FP32 value arrives through the bridge's physical NHWC
    // representation [B,S,H,N]. Restore logical BNSH before flattening heads.
    logicalBNSH = create.transpose(
        physicalValue, {batch, heads, sequence, headSize}, {0, 3, 1, 2});
  }

  return FlattenedHeads{
      create.reshape(logicalBNSH, {batch * heads, sequence, headSize}), batch,
      heads, sequence, headSize};
}

FailureOr<Value> materializeAttentionBias(Operation *op, Value sourceMask,
    int64_t batch, int64_t heads, int64_t querySequence, int64_t keySequence,
    bool causal, float maskFilterValue, TFLAttentionBuilder &create) {
  SmallVector<int64_t> sourceShape;
  SmallVector<float> sourceValues;
  bool integerMask = false;
  SmallVector<int64_t> integerValues;
  if (sourceMask) {
    auto type = dyn_cast<RankedTensorType>(sourceMask.getType());
    if (!type || !type.hasStaticShape() || type.getRank() > 4)
      return op->emitError(
                 "static Attention mask/bias must have static rank at most 4"),
             failure();
    sourceShape.assign(type.getShape().begin(), type.getShape().end());
    if (type.getElementType().isF32()) {
      FailureOr<SmallVector<float>> values = getConstantF32Values(sourceMask);
      if (failed(values))
        return op->emitError(
                   "static Attention requires a constant additive mask/bias"),
               failure();
      sourceValues = std::move(*values);
    } else if (type.getElementType().isInteger()) {
      FailureOr<SmallVector<int64_t>> values = getConstantIntValues(sourceMask);
      if (failed(values))
        return op->emitError(
                   "static Attention requires a constant boolean mask"),
               failure();
      integerMask = true;
      integerValues = std::move(*values);
    } else {
      return op->emitError(
                 "static Attention mask must be FP32 additive or integer "
                 "boolean data"),
             failure();
    }
  }

  int64_t targetShape[] = {batch, heads, querySequence, keySequence};
  if (sourceMask) {
    int64_t leading = 4 - sourceShape.size();
    for (int64_t i = 0; i < static_cast<int64_t>(sourceShape.size()); ++i) {
      int64_t target = targetShape[leading + i];
      if (sourceShape[i] != 1 && sourceShape[i] != target)
        return op->emitError(
                   "static Attention mask/bias is not broadcast-compatible "
                   "with [B,N,Sq,Sk]"),
               failure();
    }
  }
  if (causal && keySequence < querySequence)
    return op->emitError(
               "static causal Attention requires key sequence >= query "
               "sequence"),
           failure();

  auto sourceOffset = [&](ArrayRef<int64_t> coordinates) {
    int64_t leading = 4 - sourceShape.size();
    int64_t offset = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(sourceShape.size()); ++i) {
      int64_t coordinate = sourceShape[i] == 1 ? 0 : coordinates[leading + i];
      offset = offset * sourceShape[i] + coordinate;
    }
    return offset;
  };

  SmallVector<float> result;
  result.reserve(batch * heads * querySequence * keySequence);
  int64_t causalOffset = keySequence - querySequence;
  for (int64_t b = 0; b < batch; ++b)
    for (int64_t head = 0; head < heads; ++head)
      for (int64_t query = 0; query < querySequence; ++query)
        for (int64_t key = 0; key < keySequence; ++key) {
          float value = 0.0f;
          if (sourceMask) {
            int64_t offset = sourceOffset({b, head, query, key});
            value = integerMask
                        ? (integerValues[offset] != 0 ? 0.0f : maskFilterValue)
                        : sourceValues[offset];
          }
          if (causal && key > causalOffset + query)
            value += maskFilterValue;
          result.push_back(value);
        }
  return create.f32Constant(
      {batch * heads, querySequence, keySequence}, result);
}

struct StaticAttentionInputs {
  Value query;
  Value key;
  Value value;
  RankedTensorType queryType;
  RankedTensorType keyType;
  RankedTensorType valueType;
  RankedTensorType outputType;
  int64_t queryHeads;
  int64_t kvHeads;
  std::optional<float> scale;
  bool causal = false;
  float maskFilterValue = -10000.0f;
  Value additiveMask;
};

FailureOr<Value> lowerStaticAttention(Operation *op,
    const StaticAttentionInputs &inputs, ConversionPatternRewriter &rewriter) {
  TFLAttentionBuilder create(op, rewriter);
  FailureOr<FlattenedHeads> query = flattenHeads(
      op, inputs.query, inputs.queryType, inputs.queryHeads, create);
  FailureOr<FlattenedHeads> key =
      flattenHeads(op, inputs.key, inputs.keyType, inputs.kvHeads, create);
  FailureOr<FlattenedHeads> value =
      flattenHeads(op, inputs.value, inputs.valueType, inputs.kvHeads, create);
  if (failed(query) || failed(key) || failed(value))
    return failure();
  if (query->batch != key->batch || query->batch != value->batch ||
      key->sequence != value->sequence || query->headSize != key->headSize ||
      query->heads <= 0 || key->heads <= 0 || query->heads % key->heads != 0 ||
      key->heads != value->heads)
    return op->emitError(
               "incompatible static Attention batch/head/sequence shapes"),
           failure();

  if (key->heads != query->heads) {
    key->value = create.gatherHeads(key->value, key->batch, key->heads,
        query->heads, key->sequence, key->headSize);
    value->value = create.gatherHeads(value->value, value->batch, value->heads,
        query->heads, value->sequence, value->headSize);
    key->heads = query->heads;
    value->heads = query->heads;
  }

  int64_t flattenedBatch = query->batch * query->heads;
  SmallVector<int64_t> logitsShape{
      flattenedBatch, query->sequence, key->sequence};
  Value logits = create.batchMatMul(
      query->value, key->value, logitsShape, /*adjointRhs=*/true);
  float scale = inputs.scale.value_or(
      1.0f / std::sqrt(static_cast<float>(query->headSize)));
  if (!std::isfinite(scale))
    return op->emitError("static Attention scale must be finite"), failure();
  if (scale != 1.0f) {
    Value scaleValue =
        createF32ScalarTensorConstant(rewriter, op->getLoc(), scale);
    logits = create.binary(
        "tfl.mul", logits, scaleValue, create.f32Tensor(logitsShape));
  }

  if (inputs.additiveMask || inputs.causal) {
    FailureOr<Value> bias = materializeAttentionBias(op, inputs.additiveMask,
        query->batch, query->heads, query->sequence, key->sequence,
        inputs.causal, inputs.maskFilterValue, create);
    if (failed(bias))
      return failure();
    logits =
        create.binary("tfl.add", logits, *bias, create.f32Tensor(logitsShape));
  }
  Value probabilities = create.softmax(logits, logitsShape);
  SmallVector<int64_t> contextShape{
      flattenedBatch, query->sequence, value->headSize};
  Value context = create.batchMatMul(
      probabilities, value->value, contextShape, /*adjointRhs=*/false);

  ArrayRef<int64_t> outputShape = inputs.outputType.getShape();
  Value logicalBNSH = create.reshape(
      context, {query->batch, query->heads, query->sequence, value->headSize});
  if (inputs.outputType.getRank() == 3) {
    SmallVector<int64_t> expected{
        query->batch, query->sequence, query->heads * value->headSize};
    if (!llvm::equal(outputShape, expected))
      return op->emitError("rank-3 Attention output must have B,S,N*Hv shape"),
             failure();
    Value logicalBSNH = create.transpose(logicalBNSH,
        {query->batch, query->sequence, query->heads, value->headSize},
        {0, 2, 1, 3});
    return create.reshape(logicalBSNH, expected);
  }
  if (inputs.outputType.getRank() == 4) {
    SmallVector<int64_t> expected{
        query->batch, query->heads, query->sequence, value->headSize};
    if (!llvm::equal(outputShape, expected))
      return op->emitError("rank-4 Attention output must have B,N,S,Hv shape"),
             failure();
    // Return the physical NHWC representation [B,S,Hv,N].
    return create.transpose(logicalBNSH,
        {query->batch, query->sequence, value->headSize, query->heads},
        {0, 2, 3, 1});
  }
  return op->emitError("static Attention output must have rank 3 or 4"),
         failure();
}

FailureOr<Value> addBiasSlice(Operation *op, Value input,
    RankedTensorType inputType, ArrayRef<float> allBias, int64_t offset,
    TFLAttentionBuilder &create) {
  int64_t width = inputType.getShape().back();
  if (offset < 0 || offset + width > static_cast<int64_t>(allBias.size()))
    return op->emitError("Attention bias slice is out of range"), failure();
  SmallVector<float> values(
      allBias.begin() + offset, allBias.begin() + offset + width);
  Value bias = create.f32Constant({width}, values);
  return create.binary("tfl.add", input, bias, inputType);
}

FailureOr<Value> projectAttentionInput(Operation *op, Value input,
    RankedTensorType inputType, ArrayRef<float> weights,
    ArrayRef<int64_t> weightShape, ArrayRef<float> bias, int64_t offset,
    int64_t outputWidth, TFLAttentionBuilder &create) {
  if (inputType.getRank() != 3 || weightShape.size() != 2 ||
      weightShape[0] != inputType.getShape()[2] || offset < 0 ||
      outputWidth <= 0 || offset + outputWidth > weightShape[1] ||
      static_cast<int64_t>(weights.size()) != weightShape[0] * weightShape[1])
    return op->emitError("invalid static com.microsoft::Attention projection"),
           failure();

  SmallVector<float> slicedWeights;
  slicedWeights.reserve(weightShape[0] * outputWidth);
  for (int64_t row = 0; row < weightShape[0]; ++row)
    for (int64_t column = 0; column < outputWidth; ++column)
      slicedWeights.push_back(weights[row * weightShape[1] + offset + column]);
  Value weight =
      create.f32Constant({weightShape[0], outputWidth}, slicedWeights);
  SmallVector<int64_t> outputShape{
      inputType.getShape()[0], inputType.getShape()[1], outputWidth};
  Value projected = create.batchMatMul(input, weight, outputShape);
  auto outputType = create.f32Tensor(outputShape);
  return addBiasSlice(op, projected, outputType, bias, offset, create);
}

class AttentionLowering final : public OpConversionPattern<ONNXAttentionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAttentionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isAbsentOperand(op, 4) || !isAbsentOperand(op, 5) ||
        !isAbsentOperand(op, 6) || !isa<NoneType>(op.getResult(1).getType()) ||
        !isa<NoneType>(op.getResult(2).getType()) ||
        !isa<NoneType>(op.getResult(3).getType()) ||
        getIntegerAttributeOr(op, "qk_matmul_output_mode", 0) != 0 ||
        getIntegerAttributeOr(op, "softmax_precision", 0) != 0 ||
        getOptionalFloatAttribute(op, "softcap").value_or(0.0f) != 0.0f)
      return op.emitError(
                 "static ai.onnx::Attention currently requires no KV cache, "
                 "nonpad length, qk output, softcap, or precision override"),
             failure();

    FailureOr<RankedTensorType> queryType =
        getStaticF32Type(op, op.getQ(), "query");
    FailureOr<RankedTensorType> keyType =
        getStaticF32Type(op, op.getK(), "key");
    FailureOr<RankedTensorType> valueType =
        getStaticF32Type(op, op.getV(), "value");
    FailureOr<RankedTensorType> outputType =
        getStaticF32Type(op, op.getY(), "output");
    if (failed(queryType) || failed(keyType) || failed(valueType) ||
        failed(outputType))
      return failure();

    int64_t queryHeads = getIntegerAttributeOr(op, "q_num_heads",
        (*queryType).getRank() == 4 ? (*queryType).getShape()[1] : -1);
    int64_t kvHeads = getIntegerAttributeOr(op, "kv_num_heads",
        (*keyType).getRank() == 4 ? (*keyType).getShape()[1] : queryHeads);
    Value mask = isAbsentOperand(op, 3) ? Value() : op.getAttnMask();
    StaticAttentionInputs inputs{adaptor.getQ(), adaptor.getK(), adaptor.getV(),
        *queryType, *keyType, *valueType, *outputType, queryHeads, kvHeads,
        getOptionalFloatAttribute(op, "scale"),
        getIntegerAttributeOr(op, "is_causal", 0) != 0, -10000.0f, mask};
    FailureOr<Value> result = lowerStaticAttention(op, inputs, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(
        op, ValueRange{*result, adaptor.getPastKey(), adaptor.getPastValue(),
                adaptor.getNonpadKvSeqlen()});
    return success();
  }
};

class MicrosoftAttentionLowering final
    : public OpConversionPattern<ONNXCustomOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXCustomOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    if (!domain || domain.getValue() != "com.microsoft")
      return failure();
    StringRef name = op.getFunctionName();
    if (name != "Attention" && name != "MultiHeadAttention" &&
        name != "GroupQueryAttention")
      return failure();
    if (op.getNumResults() != 1)
      return op.emitError(
                 "static com.microsoft Attention lowering currently supports "
                 "the primary output only"),
             failure();

    if (name == "Attention")
      return lowerProjectedAttention(op, adaptor.getOperands(), rewriter);
    if (name == "MultiHeadAttention")
      return lowerMultiHeadAttention(op, adaptor.getOperands(), rewriter);
    return lowerGroupQueryAttention(op, adaptor.getOperands(), rewriter);
  }

private:
  LogicalResult lowerProjectedAttention(ONNXCustomOp op, ValueRange operands,
      ConversionPatternRewriter &rewriter) const {
    if (op.getNumOperands() < 3 || !isAbsentOperand(op, 3) ||
        !isAbsentOperand(op, 4) || !isAbsentOperand(op, 6) ||
        getIntegerAttributeOr(op, "do_rotary", 0) != 0 ||
        getIntegerAttributeOr(op, "past_present_share_buffer", 0) != 0)
      return op.emitError(
                 "static com.microsoft::Attention requires no mask_index, KV "
                 "cache, shared cache buffer, or rotary embedding"),
             failure();

    FailureOr<RankedTensorType> inputType =
        getStaticF32Type(op, op->getOperand(0), "input");
    FailureOr<RankedTensorType> weightType =
        getStaticF32Type(op, op->getOperand(1), "projection weights");
    FailureOr<RankedTensorType> biasType =
        getStaticF32Type(op, op->getOperand(2), "projection bias");
    FailureOr<RankedTensorType> outputType =
        getStaticF32Type(op, op.getResult(0), "output");
    if (failed(inputType) || failed(weightType) || failed(biasType) ||
        failed(outputType) || (*inputType).getRank() != 3 ||
        (*weightType).getRank() != 2 || (*biasType).getRank() != 1)
      return failure();

    FailureOr<SmallVector<float>> weights =
        getConstantF32Values(op->getOperand(1));
    FailureOr<SmallVector<float>> bias =
        getConstantF32Values(op->getOperand(2));
    if (failed(weights) || failed(bias))
      return op.emitError(
                 "static com.microsoft::Attention requires constant FP32 "
                 "projection weights and bias"),
             failure();

    int64_t totalWidth = (*weightType).getShape()[1];
    SmallVector<int64_t> widths;
    if (auto attr = op->getAttrOfType<ArrayAttr>("qkv_hidden_sizes")) {
      for (Attribute item : attr)
        widths.push_back(cast<IntegerAttr>(item).getInt());
    } else if (totalWidth % 3 == 0) {
      widths.assign(3, totalWidth / 3);
    }
    if (widths.size() != 3 || widths[0] + widths[1] + widths[2] != totalWidth ||
        static_cast<int64_t>(bias->size()) != totalWidth)
      return op.emitError("invalid com.microsoft::Attention QKV widths"),
             failure();

    TFLAttentionBuilder create(op, rewriter);
    ArrayRef<int64_t> weightShape = (*weightType).getShape();
    FailureOr<Value> query = projectAttentionInput(op, operands[0], *inputType,
        *weights, weightShape, *bias, 0, widths[0], create);
    FailureOr<Value> key = projectAttentionInput(op, operands[0], *inputType,
        *weights, weightShape, *bias, widths[0], widths[1], create);
    FailureOr<Value> value = projectAttentionInput(op, operands[0], *inputType,
        *weights, weightShape, *bias, widths[0] + widths[1], widths[2], create);
    if (failed(query) || failed(key) || failed(value))
      return failure();

    int64_t batch = (*inputType).getShape()[0];
    int64_t sequence = (*inputType).getShape()[1];
    auto queryType = create.f32Tensor({batch, sequence, widths[0]});
    auto keyType = create.f32Tensor({batch, sequence, widths[1]});
    auto valueType = create.f32Tensor({batch, sequence, widths[2]});
    int64_t heads = getIntegerAttributeOr(op, "num_heads", -1);
    Value mask = isAbsentOperand(op, 5) ? Value() : op->getOperand(5);
    StaticAttentionInputs inputs{*query, *key, *value, queryType, keyType,
        valueType, *outputType, heads, heads,
        getOptionalFloatAttribute(op, "scale"),
        getIntegerAttributeOr(op, "unidirectional", 0) != 0,
        getOptionalFloatAttribute(op, "mask_filter_value").value_or(-10000.0f),
        mask};
    FailureOr<Value> result = lowerStaticAttention(op, inputs, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }

  LogicalResult lowerMultiHeadAttention(ONNXCustomOp op, ValueRange operands,
      ConversionPatternRewriter &rewriter) const {
    if (op.getNumOperands() < 3 || isAbsentOperand(op, 1) ||
        isAbsentOperand(op, 2) || !isAbsentOperand(op, 4) ||
        !isAbsentOperand(op, 6) || !isAbsentOperand(op, 7) ||
        !isAbsentOperand(op, 8) || !isAbsentOperand(op, 9))
      return op.emitError(
                 "static com.microsoft::MultiHeadAttention requires separate "
                 "Q/K/V and no key-padding mask or KV cache"),
             failure();

    FailureOr<RankedTensorType> queryType =
        getStaticF32Type(op, op->getOperand(0), "query");
    FailureOr<RankedTensorType> keyType =
        getStaticF32Type(op, op->getOperand(1), "key");
    FailureOr<RankedTensorType> valueType =
        getStaticF32Type(op, op->getOperand(2), "value");
    FailureOr<RankedTensorType> outputType =
        getStaticF32Type(op, op.getResult(0), "output");
    if (failed(queryType) || failed(keyType) || failed(valueType) ||
        failed(outputType) || (*queryType).getRank() != 3 ||
        (*keyType).getRank() != 3 || (*valueType).getRank() != 3)
      return failure();

    Value query = operands[0];
    Value key = operands[1];
    Value value = operands[2];
    if (!isAbsentOperand(op, 3)) {
      FailureOr<SmallVector<float>> bias =
          getConstantF32Values(op->getOperand(3));
      int64_t queryWidth = (*queryType).getShape()[2];
      int64_t keyWidth = (*keyType).getShape()[2];
      int64_t valueWidth = (*valueType).getShape()[2];
      if (failed(bias) || static_cast<int64_t>(bias->size()) !=
                              queryWidth + keyWidth + valueWidth)
        return op.emitError(
                   "static MultiHeadAttention bias must contain Q/K/V bias"),
               failure();
      TFLAttentionBuilder create(op, rewriter);
      FailureOr<Value> biasedQuery =
          addBiasSlice(op, query, *queryType, *bias, 0, create);
      FailureOr<Value> biasedKey =
          addBiasSlice(op, key, *keyType, *bias, queryWidth, create);
      FailureOr<Value> biasedValue = addBiasSlice(
          op, value, *valueType, *bias, queryWidth + keyWidth, create);
      if (failed(biasedQuery) || failed(biasedKey) || failed(biasedValue))
        return failure();
      query = *biasedQuery;
      key = *biasedKey;
      value = *biasedValue;
    }

    int64_t heads = getIntegerAttributeOr(op, "num_heads", -1);
    Value mask = isAbsentOperand(op, 5) ? Value() : op->getOperand(5);
    StaticAttentionInputs inputs{query, key, value, *queryType, *keyType,
        *valueType, *outputType, heads, heads,
        getOptionalFloatAttribute(op, "scale"),
        getIntegerAttributeOr(op, "unidirectional", 0) != 0,
        getOptionalFloatAttribute(op, "mask_filter_value").value_or(-10000.0f),
        mask};
    FailureOr<Value> result = lowerStaticAttention(op, inputs, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }

  LogicalResult lowerGroupQueryAttention(ONNXCustomOp op, ValueRange operands,
      ConversionPatternRewriter &rewriter) const {
    if (op.getNumOperands() < 7 || !isAbsentOperand(op, 3) ||
        !isAbsentOperand(op, 4) || isAbsentOperand(op, 5) ||
        isAbsentOperand(op, 6) || !isAbsentOperand(op, 7) ||
        !isAbsentOperand(op, 8) || !isAbsentOperand(op, 9) ||
        !isAbsentOperand(op, 10) || !isAbsentOperand(op, 11) ||
        getIntegerAttributeOr(op, "do_rotary", 0) != 0 ||
        getIntegerAttributeOr(op, "local_window_size", -1) != -1 ||
        getIntegerAttributeOr(op, "smooth_softmax", 0) != 0 ||
        getIntegerAttributeOr(op, "qk_output", 0) != 0 ||
        getOptionalFloatAttribute(op, "softcap").value_or(0.0f) != 0.0f)
      return op.emitError(
                 "static GroupQueryAttention currently requires no cache, "
                 "rotary, local window, attention bias, head sink, softcap, "
                 "smooth softmax, or qk output"),
             failure();

    FailureOr<RankedTensorType> packedOrQueryType =
        getStaticF32Type(op, op->getOperand(0), "query");
    FailureOr<RankedTensorType> outputType =
        getStaticF32Type(op, op.getResult(0), "output");
    if (failed(packedOrQueryType) || failed(outputType) ||
        (*packedOrQueryType).getRank() != 3)
      return failure();

    int64_t queryHeads = getIntegerAttributeOr(op, "num_heads", -1);
    int64_t kvHeads = getIntegerAttributeOr(op, "kv_num_heads", -1);
    if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
      return op.emitError("invalid GroupQueryAttention head counts"), failure();

    Value query = operands[0];
    Value key;
    Value value;
    RankedTensorType queryType = *packedOrQueryType;
    RankedTensorType keyType;
    RankedTensorType valueType;
    bool packed = isAbsentOperand(op, 1) && isAbsentOperand(op, 2);
    if (packed) {
      int64_t packedWidth = queryType.getShape()[2];
      int64_t headDivisor = queryHeads + 2 * kvHeads;
      if (packedWidth % headDivisor != 0)
        return op.emitError(
                   "packed GroupQueryAttention width does not match heads"),
               failure();
      int64_t headSize = packedWidth / headDivisor;
      int64_t queryWidth = queryHeads * headSize;
      int64_t kvWidth = kvHeads * headSize;
      int64_t batch = queryType.getShape()[0];
      int64_t sequence = queryType.getShape()[1];
      TFLAttentionBuilder create(op, rewriter);
      query =
          create.slice(operands[0], {0, 0, 0}, {batch, sequence, queryWidth});
      key = create.slice(
          operands[0], {0, 0, queryWidth}, {batch, sequence, kvWidth});
      value = create.slice(operands[0], {0, 0, queryWidth + kvWidth},
          {batch, sequence, kvWidth});
      queryType = create.f32Tensor({batch, sequence, queryWidth});
      keyType = create.f32Tensor({batch, sequence, kvWidth});
      valueType = keyType;
    } else {
      if (isAbsentOperand(op, 1) || isAbsentOperand(op, 2))
        return op.emitError(
                   "GroupQueryAttention key and value must both be present"),
               failure();
      FailureOr<RankedTensorType> foundKeyType =
          getStaticF32Type(op, op->getOperand(1), "key");
      FailureOr<RankedTensorType> foundValueType =
          getStaticF32Type(op, op->getOperand(2), "value");
      if (failed(foundKeyType) || failed(foundValueType) ||
          (*foundKeyType).getRank() != 3 || (*foundValueType).getRank() != 3)
        return failure();
      key = operands[1];
      value = operands[2];
      keyType = *foundKeyType;
      valueType = *foundValueType;
    }

    FailureOr<SmallVector<int64_t>> sequenceLengths =
        getConstantIntValues(op->getOperand(5));
    FailureOr<SmallVector<int64_t>> totalSequence =
        getConstantIntValues(op->getOperand(6));
    int64_t keySequence = keyType.getShape()[1];
    if (failed(sequenceLengths) || failed(totalSequence) ||
        totalSequence->size() != 1 || (*totalSequence)[0] != keySequence ||
        static_cast<int64_t>(sequenceLengths->size()) !=
            queryType.getShape()[0] ||
        llvm::any_of(*sequenceLengths,
            [&](int64_t value) { return value != keySequence - 1; }))
      return op.emitError(
                 "static GroupQueryAttention requires constant full-length "
                 "seqlens_k and total_sequence_length"),
             failure();

    StaticAttentionInputs inputs{query, key, value, queryType, keyType,
        valueType, *outputType, queryHeads, kvHeads,
        getOptionalFloatAttribute(op, "scale"), /*causal=*/true, -10000.0f,
        Value()};
    FailureOr<Value> result = lowerStaticAttention(op, inputs, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

} // namespace

void populateLoweringONNXAttentionOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AttentionLowering, MicrosoftAttentionLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
