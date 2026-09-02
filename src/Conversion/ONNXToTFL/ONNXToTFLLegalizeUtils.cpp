/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLLegalizeUtils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <cstring>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {

LogicalResult validateStaticF32Tensor(
    Operation *op, Type type, StringRef role) {
  if (failed(validateStaticF32TensorOrScalar(op, type, role)))
    return failure();
  auto tensorType = cast<RankedTensorType>(type);
  if (tensorType.getRank() < 1) {
    return op->emitError() << "ONNXToTFL MVP requires activation rank >= 1 ("
                           << role << ")";
  }
  return success();
}

LogicalResult validateStaticF32TensorOrScalar(
    Operation *op, Type type, StringRef role) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return op->emitError()
           << "dynamic or unranked tensor shape is not supported by "
              "ONNXToTFL MVP ("
           << role << ")";
  }
  if (!tensorType.hasStaticShape()) {
    return op->emitError()
           << "dynamic tensor shape is not supported by ONNXToTFL MVP (" << role
           << ": " << type << ")";
  }
  if (!tensorType.getElementType().isF32()) {
    return op->emitError()
           << "ONNXToTFL MVP only supports f32 activation tensors (" << role
           << ": " << type << ")";
  }
  return success();
}

Operation *createTFLOperation(ConversionPatternRewriter &rewriter, Location loc,
    StringRef name, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attributes) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(attributes);
  return rewriter.create(state);
}

Value createI32ShapeConstant(ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<int64_t> values) {
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI32Type());
  SmallVector<int32_t> narrowed;
  narrowed.reserve(values.size());
  for (int64_t value : values)
    narrowed.push_back(static_cast<int32_t>(value));
  auto attr = DenseIntElementsAttr::get(type, ArrayRef<int32_t>(narrowed));
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

Value createI32ScalarTensorConstant(
    ConversionPatternRewriter &rewriter, Location loc, int32_t value) {
  auto type = RankedTensorType::get({}, rewriter.getI32Type());
  auto attr = DenseIntElementsAttr::get(type, ArrayRef<int32_t>{value});
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

Value createF32ScalarTensorConstant(
    ConversionPatternRewriter &rewriter, Location loc, float value) {
  auto type = RankedTensorType::get({}, rewriter.getF32Type());
  auto attr = DenseFPElementsAttr::get(type, ArrayRef<float>{value});
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

FailureOr<SmallVector<int64_t>> getConstantIntValues(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<ONNXConstantOp>())
    attribute = constant.getValueAttr();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  auto elements = dyn_cast_or_null<ElementsAttr>(attribute);
  if (!elements || !elements.getElementType().isIntOrIndex())
    return failure();

  SmallVector<int64_t> values;
  values.reserve(elements.size());
  for (APInt item : elements.getValues<APInt>())
    values.push_back(item.getSExtValue());
  return values;
}

FailureOr<SmallVector<float>> getConstantF32Values(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<ONNXConstantOp>())
    attribute = constant.getValueAttr();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  auto elements = dyn_cast_or_null<ElementsAttr>(attribute);
  if (!elements || !elements.getElementType().isF32())
    return failure();

  SmallVector<float> values;
  values.reserve(elements.size());
  for (APFloat item : elements.getValues<APFloat>())
    values.push_back(item.convertToFloat());
  return values;
}

Type convertRank4NCHWToNHWCType(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked || ranked.getRank() != 4 || !ranked.getElementType().isF32())
    return type;
  ArrayRef<int64_t> shape = ranked.getShape();
  return RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[1]}, ranked.getElementType());
}

SmallVector<int64_t> getONNXToTFLPhysicalToLogicalLayout(
    RankedTensorType type) {
  int64_t rank = type.getRank();
  if (auto encoded = dyn_cast_or_null<DenseI64ArrayAttr>(type.getEncoding())) {
    SmallVector<int64_t> layout(encoded.asArrayRef());
    if (static_cast<int64_t>(layout.size()) == rank)
      return layout;
  }
  if (rank == 4 && type.getElementType().isF32())
    return {0, 2, 3, 1};
  SmallVector<int64_t> identity(rank);
  std::iota(identity.begin(), identity.end(), 0);
  return identity;
}

RankedTensorType stripONNXToTFLLayoutEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

Type convertONNXToTFLPhysicalType(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked)
    return type;
  SmallVector<int64_t> layout = getONNXToTFLPhysicalToLogicalLayout(ranked);
  SmallVector<int64_t> physicalShape;
  physicalShape.reserve(ranked.getRank());
  for (int64_t logicalDimension : layout)
    physicalShape.push_back(ranked.getShape()[logicalDimension]);
  return RankedTensorType::get(physicalShape, ranked.getElementType());
}

FailureOr<Value> convertTFLValueLayout(ConversionPatternRewriter &rewriter,
    Location loc, Value value, RankedTensorType logicalType,
    ArrayRef<int64_t> sourcePhysicalToLogical,
    ArrayRef<int64_t> targetPhysicalToLogical) {
  int64_t rank = logicalType.getRank();
  if (static_cast<int64_t>(sourcePhysicalToLogical.size()) != rank ||
      static_cast<int64_t>(targetPhysicalToLogical.size()) != rank)
    return failure();
  if (sourcePhysicalToLogical == targetPhysicalToLogical)
    return value;

  SmallVector<int64_t> inverseSource(rank, -1);
  for (int64_t physicalDimension = 0; physicalDimension < rank;
       ++physicalDimension) {
    int64_t logicalDimension = sourcePhysicalToLogical[physicalDimension];
    if (logicalDimension < 0 || logicalDimension >= rank ||
        inverseSource[logicalDimension] != -1)
      return failure();
    inverseSource[logicalDimension] = physicalDimension;
  }
  SmallVector<int64_t> transposePermutation;
  SmallVector<int64_t> targetShape;
  transposePermutation.reserve(rank);
  targetShape.reserve(rank);
  SmallVector<bool> seenTarget(rank, false);
  for (int64_t logicalDimension : targetPhysicalToLogical) {
    if (logicalDimension < 0 || logicalDimension >= rank ||
        seenTarget[logicalDimension])
      return failure();
    seenTarget[logicalDimension] = true;
    transposePermutation.push_back(inverseSource[logicalDimension]);
    targetShape.push_back(logicalType.getShape()[logicalDimension]);
  }
  Value permutation =
      createI32ShapeConstant(rewriter, loc, transposePermutation);
  auto targetType =
      RankedTensorType::get(targetShape, logicalType.getElementType());
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{targetType}, ValueRange{value, permutation})
      ->getResult(0);
}

FailureOr<DenseElementsAttr> transposeRank4NCHWToNHWC(DenseElementsAttr input) {
  auto oldType = dyn_cast<RankedTensorType>(input.getType());
  if (!oldType || oldType.getRank() != 4)
    return failure();
  auto newType = cast<RankedTensorType>(convertRank4NCHWToNHWCType(oldType));
  if (input.isSplat())
    return input.reshape(newType);

  ArrayRef<char> rawData = input.getRawData();
  if (!rawData.data() || oldType.getElementTypeBitWidth() % 8 != 0)
    return failure();

  // This is the standard row-major dense-attribute transpose algorithm used
  // by MLIR's TOSA transpose reduction. Permutation [0, 2, 3, 1] maps NCHW
  // activations to NHWC and OIHW Conv filters to OHWI.
  constexpr int64_t permutation[] = {0, 2, 3, 1};
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape = newType.getShape();
  auto calculateStrides = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t> strides(shape.size());
    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
  };
  SmallVector<int64_t> inputStrides = calculateStrides(oldShape);
  SmallVector<int64_t> outputStrides = calculateStrides(newShape);
  size_t elementSize = oldType.getElementTypeBitWidth() / 8;
  int64_t numElements = oldType.getNumElements();
  SmallVector<char> outputBuffer(numElements * elementSize);

  for (int64_t destination = 0; destination < numElements; ++destination) {
    int64_t remaining = destination;
    int64_t source = 0;
    for (int64_t dimension = 0; dimension < 4; ++dimension) {
      int64_t coordinate = remaining / outputStrides[dimension];
      remaining %= outputStrides[dimension];
      source += coordinate * inputStrides[permutation[dimension]];
    }
    std::memcpy(outputBuffer.data() + destination * elementSize,
        rawData.data() + source * elementSize, elementSize);
  }
  return DenseElementsAttr::getFromRawBuffer(newType, outputBuffer);
}

int64_t mapNCHWAxisToNHWC(int64_t axis) {
  constexpr int64_t mapping[] = {0, 3, 1, 2};
  return axis >= 0 && axis < 4 ? mapping[axis] : axis;
}

int64_t mapONNXToTFLLogicalAxis(RankedTensorType type, int64_t logicalAxis) {
  SmallVector<int64_t> layout = getONNXToTFLPhysicalToLogicalLayout(type);
  auto found = llvm::find(layout, logicalAxis);
  return found == layout.end() ? -1 : std::distance(layout.begin(), found);
}

FailureOr<SmallVector<int64_t>> computeONNXToTFLReducedLayout(
    ArrayRef<int64_t> inputLayout, ArrayRef<int64_t> reducedLogicalAxes,
    bool keepDims) {
  int64_t rank = inputLayout.size();
  if (keepDims)
    return SmallVector<int64_t>(inputLayout);
  SmallVector<bool> reduced(rank, false);
  for (int64_t axis : reducedLogicalAxes) {
    if (axis < 0 || axis >= rank || reduced[axis])
      return failure();
    reduced[axis] = true;
  }
  SmallVector<int64_t> logicalInputToOutput(rank, -1);
  int64_t outputAxis = 0;
  for (int64_t inputAxis = 0; inputAxis < rank; ++inputAxis)
    if (!reduced[inputAxis])
      logicalInputToOutput[inputAxis] = outputAxis++;
  SmallVector<int64_t> outputLayout;
  outputLayout.reserve(outputAxis);
  for (int64_t logicalInputAxis : inputLayout)
    if (!reduced[logicalInputAxis])
      outputLayout.push_back(logicalInputToOutput[logicalInputAxis]);
  return outputLayout;
}

FailureOr<SmallVector<int64_t>> computeONNXToTFLSingletonReshapeLayout(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
    ArrayRef<int64_t> inputPhysicalToLogical) {
  int64_t inputRank = inputShape.size();
  int64_t outputRank = outputShape.size();
  if (static_cast<int64_t>(inputPhysicalToLogical.size()) != inputRank)
    return failure();

  SmallVector<bool> seenInput(inputRank, false);
  for (int64_t logicalAxis : inputPhysicalToLogical) {
    if (logicalAxis < 0 || logicalAxis >= inputRank || seenInput[logicalAxis])
      return failure();
    seenInput[logicalAxis] = true;
  }

  SmallVector<int64_t> inputNonSingletonAxes;
  SmallVector<int64_t> outputNonSingletonAxes;
  SmallVector<int64_t> outputSingletonAxes;
  for (int64_t axis = 0; axis < inputRank; ++axis)
    if (inputShape[axis] != 1)
      inputNonSingletonAxes.push_back(axis);
  for (int64_t axis = 0; axis < outputRank; ++axis) {
    if (outputShape[axis] == 1)
      outputSingletonAxes.push_back(axis);
    else
      outputNonSingletonAxes.push_back(axis);
  }
  if (inputNonSingletonAxes.size() != outputNonSingletonAxes.size())
    return failure();

  SmallVector<int64_t> inputToOutput(inputRank, -1);
  for (auto [inputAxis, outputAxis] :
      llvm::zip(inputNonSingletonAxes, outputNonSingletonAxes)) {
    if (inputShape[inputAxis] != outputShape[outputAxis])
      return failure();
    inputToOutput[inputAxis] = outputAxis;
  }

  SmallVector<int64_t> outputLayout;
  outputLayout.reserve(outputRank);
  int64_t nextOutputSingleton = 0;
  for (int64_t inputAxis : inputPhysicalToLogical) {
    if (inputShape[inputAxis] != 1) {
      outputLayout.push_back(inputToOutput[inputAxis]);
      continue;
    }
    if (nextOutputSingleton < static_cast<int64_t>(outputSingletonAxes.size()))
      outputLayout.push_back(outputSingletonAxes[nextOutputSingleton++]);
  }
  while (nextOutputSingleton < static_cast<int64_t>(outputSingletonAxes.size()))
    outputLayout.push_back(outputSingletonAxes[nextOutputSingleton++]);
  if (static_cast<int64_t>(outputLayout.size()) != outputRank)
    return failure();
  return outputLayout;
}

FailureOr<Value> createTFLLayoutAwareReduction(
    ConversionPatternRewriter &rewriter, Location loc, StringRef tflName,
    Value input, RankedTensorType sourceInputType,
    RankedTensorType sourceResultType, ArrayRef<int64_t> logicalAxes,
    bool keepDims) {
  SmallVector<int64_t> inputLayout =
      getONNXToTFLPhysicalToLogicalLayout(sourceInputType);
  SmallVector<int64_t> physicalAxes;
  physicalAxes.reserve(logicalAxes.size());
  for (int64_t logicalAxis : logicalAxes) {
    auto found = llvm::find(inputLayout, logicalAxis);
    if (found == inputLayout.end())
      return failure();
    physicalAxes.push_back(std::distance(inputLayout.begin(), found));
  }
  FailureOr<SmallVector<int64_t>> naturalResultLayout =
      computeONNXToTFLReducedLayout(inputLayout, logicalAxes, keepDims);
  if (failed(naturalResultLayout) ||
      static_cast<int64_t>(naturalResultLayout->size()) !=
          sourceResultType.getRank())
    return failure();

  auto logicalResultType = stripONNXToTFLLayoutEncoding(sourceResultType);
  auto naturalEncoding =
      DenseI64ArrayAttr::get(rewriter.getContext(), *naturalResultLayout);
  auto naturalEncodedType = RankedTensorType::get(logicalResultType.getShape(),
      logicalResultType.getElementType(), naturalEncoding);
  Type naturalPhysicalType = convertONNXToTFLPhysicalType(naturalEncodedType);
  Value axisValue = createI32ShapeConstant(rewriter, loc, physicalAxes);
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(keepDims))};
  Value result = createTFLOperation(rewriter, loc, tflName,
      TypeRange{naturalPhysicalType}, ValueRange{input, axisValue}, attributes)
                     ->getResult(0);

  SmallVector<int64_t> targetLayout =
      getONNXToTFLPhysicalToLogicalLayout(sourceResultType);
  FailureOr<Value> physicalResult = convertTFLValueLayout(rewriter, loc, result,
      logicalResultType, *naturalResultLayout, targetLayout);
  if (failed(physicalResult) ||
      (*physicalResult).getType() !=
          convertONNXToTFLPhysicalType(sourceResultType))
    return failure();
  return *physicalResult;
}

int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

NamedAttribute getFusedActivationNone(Builder &builder) {
  return builder.getNamedAttr(
      "fused_activation_function", builder.getStringAttr("NONE"));
}

} // namespace onnx_mlir
