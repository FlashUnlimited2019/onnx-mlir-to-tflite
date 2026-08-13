/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<SmallVector<int64_t>> getPhysicalPadPairs(
    ONNXPadOp op, int64_t rank) {
  if (!isa<NoneType>(op.getAxes().getType())) {
    op.emitError("unsupported Pad configuration: axes must be absent");
    return failure();
  }
  FailureOr<SmallVector<int64_t>> pads = getConstantIntValues(op.getPads());
  if (failed(pads) || static_cast<int64_t>(pads->size()) != 2 * rank ||
      llvm::any_of(*pads, [](int64_t value) {
        return value < 0 || value > std::numeric_limits<int32_t>::max();
      })) {
    op.emitError() << "ONNXToTFL Pad requires " << 2 * rank
                   << " constant non-negative pad values in the int32 range";
    return failure();
  }

  SmallVector<int64_t> physicalPairs(2 * rank);
  for (int64_t logicalAxis = 0; logicalAxis < rank; ++logicalAxis) {
    int64_t physicalAxis =
        rank == 4 ? mapNCHWAxisToNHWC(logicalAxis) : logicalAxis;
    physicalPairs[2 * physicalAxis] = (*pads)[logicalAxis];
    physicalPairs[2 * physicalAxis + 1] = (*pads)[logicalAxis + rank];
  }
  return physicalPairs;
}

Value createPaddingMatrix(Location loc, ArrayRef<int64_t> pairs,
    ConversionPatternRewriter &rewriter) {
  SmallVector<int32_t> values;
  values.reserve(pairs.size());
  for (int64_t value : pairs)
    values.push_back(static_cast<int32_t>(value));
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(pairs.size() / 2), 2}, rewriter.getI32Type());
  return arith::ConstantOp::create(rewriter, loc, type,
      DenseIntElementsAttr::get(type, ArrayRef<int32_t>(values)));
}

FailureOr<Value> getScalarPadValue(
    ONNXPadOp op, Value value, ConversionPatternRewriter &rewriter) {
  auto scalarType = RankedTensorType::get({}, rewriter.getF32Type());
  if (isa<NoneType>(value.getType()))
    return arith::ConstantOp::create(rewriter, op.getLoc(), scalarType,
        DenseElementsAttr::get(scalarType, 0.0f))
        .getResult();

  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getNumElements() != 1) {
    op.emitError("constant Pad value must be a scalar f32 tensor");
    return failure();
  }
  if (type.getRank() == 0)
    return value;

  Value shape = createI32ShapeConstant(rewriter, op.getLoc(), {});
  return createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
      TypeRange{scalarType}, ValueRange{value, shape})
      ->getResult(0);
}

FailureOr<Value> lowerEdgePad(ONNXPadOp op, Value input,
    RankedTensorType resultType, ArrayRef<int64_t> pairs,
    ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> currentShape(inputType.getShape());
  Value current = input;

  for (int64_t axis = 0; axis < rank; ++axis) {
    int64_t before = pairs[2 * axis];
    int64_t after = pairs[2 * axis + 1];
    if (before == 0 && after == 0)
      continue;

    SmallVector<Value> concatenands;
    auto createRepeatedEdge = [&](bool first, int64_t count) -> Value {
      SmallVector<int64_t> begin(rank, 0);
      SmallVector<int64_t> size(currentShape);
      if (!first)
        begin[axis] = currentShape[axis] - 1;
      size[axis] = 1;
      SmallVector<int64_t> sliceShape(currentShape);
      sliceShape[axis] = 1;
      auto sliceType =
          RankedTensorType::get(sliceShape, inputType.getElementType());
      Value beginValue = createI32ShapeConstant(rewriter, op.getLoc(), begin);
      Value sizeValue = createI32ShapeConstant(rewriter, op.getLoc(), size);
      Value slice = createTFLOperation(rewriter, op.getLoc(), "tfl.slice",
          TypeRange{sliceType}, ValueRange{current, beginValue, sizeValue})
                        ->getResult(0);
      SmallVector<int64_t> multiples(rank, 1);
      multiples[axis] = count;
      SmallVector<int64_t> repeatedShape(sliceShape);
      repeatedShape[axis] = count;
      auto repeatedType =
          RankedTensorType::get(repeatedShape, inputType.getElementType());
      Value multiplesValue =
          createI32ShapeConstant(rewriter, op.getLoc(), multiples);
      return createTFLOperation(rewriter, op.getLoc(), "tfl.tile",
          TypeRange{repeatedType}, ValueRange{slice, multiplesValue})
          ->getResult(0);
    };

    if (before != 0)
      concatenands.push_back(createRepeatedEdge(/*first=*/true, before));
    concatenands.push_back(current);
    if (after != 0)
      concatenands.push_back(createRepeatedEdge(/*first=*/false, after));

    currentShape[axis] += before + after;
    auto nextType =
        RankedTensorType::get(currentShape, inputType.getElementType());
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
        getFusedActivationNone(rewriter)};
    current = createTFLOperation(rewriter, op.getLoc(), "tfl.concatenation",
        TypeRange{nextType}, concatenands, attributes)
                  ->getResult(0);
  }

  if (!llvm::equal(currentShape, resultType.getShape())) {
    op.emitError("Pad values do not match the inferred result shape");
    return failure();
  }
  return current;
}

class PadLowering final : public OpConversionPattern<ONNXPadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXPadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !resultType || inputType.getRank() < 1 ||
        inputType.getRank() > 5 ||
        resultType.getRank() != inputType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "Pad input")) ||
        failed(validateStaticF32Tensor(op, resultType, "Pad result"))) {
      op.emitError("ONNXToTFL Pad supports static rank-1 through rank-5 f32 "
                   "tensors only");
      return failure();
    }
    int64_t rank = inputType.getRank();
    FailureOr<SmallVector<int64_t>> pairs = getPhysicalPadPairs(op, rank);
    if (failed(pairs))
      return failure();

    auto physicalInputType =
        cast<RankedTensorType>(adaptor.getData().getType());
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    for (int64_t axis = 0; axis < rank; ++axis) {
      int64_t expected = physicalInputType.getShape()[axis] +
                         (*pairs)[2 * axis] + (*pairs)[2 * axis + 1];
      if (physicalResultType.getShape()[axis] != expected) {
        op.emitError("Pad values do not match the inferred result shape");
        return failure();
      }
    }

    StringRef mode = op.getMode();
    if (mode == "constant") {
      Value padding = createPaddingMatrix(op.getLoc(), *pairs, rewriter);
      FailureOr<Value> value =
          getScalarPadValue(op, adaptor.getConstantValue(), rewriter);
      if (failed(value))
        return failure();
      Operation *pad = createTFLOperation(rewriter, op.getLoc(), "tfl.padv2",
          TypeRange{physicalResultType},
          ValueRange{adaptor.getData(), padding, *value});
      rewriter.replaceOp(op, pad->getResults());
      return success();
    }
    if (mode == "reflect") {
      for (int64_t axis = 0; axis < rank; ++axis) {
        if ((*pairs)[2 * axis] >= physicalInputType.getShape()[axis] ||
            (*pairs)[2 * axis + 1] >= physicalInputType.getShape()[axis]) {
          op.emitError("reflect Pad values must be smaller than their input "
                       "dimensions");
          return failure();
        }
      }
      Value padding = createPaddingMatrix(op.getLoc(), *pairs, rewriter);
      Attribute mirrorMode = OpaqueAttr::get(rewriter.getStringAttr("tfl"),
          "mirror_pad_attr REFLECT", rewriter.getNoneType());
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("mode", mirrorMode)};
      Operation *pad = createTFLOperation(rewriter, op.getLoc(),
          "tfl.mirror_pad", TypeRange{physicalResultType},
          ValueRange{adaptor.getData(), padding}, attributes);
      rewriter.replaceOp(op, pad->getResults());
      return success();
    }
    if (mode == "edge") {
      FailureOr<Value> padded = lowerEdgePad(
          op, adaptor.getData(), physicalResultType, *pairs, rewriter);
      if (failed(padded))
        return failure();
      rewriter.replaceOp(op, *padded);
      return success();
    }
    op.emitError() << "unsupported Pad mode=" << mode
                   << ": expected constant, reflect, or edge";
    return failure();
  }
};

} // namespace

void populateLoweringONNXPadOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<PadLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
