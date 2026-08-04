/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLLegalizeUtils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

int64_t normalizeAxis(int64_t axis, int64_t rank) {
  return axis < 0 ? axis + rank : axis;
}

NamedAttribute getFusedActivationNone(Builder &builder) {
  return builder.getNamedAttr(
      "fused_activation_function", builder.getStringAttr("NONE"));
}

} // namespace onnx_mlir
