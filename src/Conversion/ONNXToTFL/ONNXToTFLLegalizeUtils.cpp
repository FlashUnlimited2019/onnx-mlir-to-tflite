/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLLegalizeUtils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cstring>

using namespace mlir;

namespace onnx_mlir {

LogicalResult validateStaticF32Tensor(
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
  if (tensorType.getRank() < 1) {
    return op->emitError() << "ONNXToTFL MVP requires activation rank >= 1 ("
                           << role << ")";
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

Value createF32ScalarTensorConstant(
    ConversionPatternRewriter &rewriter, Location loc, float value) {
  auto type = RankedTensorType::get({}, rewriter.getF32Type());
  auto attr = DenseFPElementsAttr::get(type, ArrayRef<float>{value});
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

Type convertRank4NCHWToNHWCType(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked || ranked.getRank() != 4 || !ranked.getElementType().isF32())
    return type;
  ArrayRef<int64_t> shape = ranked.getShape();
  return RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[1]}, ranked.getElementType());
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

int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

NamedAttribute getFusedActivationNone(Builder &builder) {
  return builder.getNamedAttr(
      "fused_activation_function", builder.getStringAttr("NONE"));
}

} // namespace onnx_mlir
