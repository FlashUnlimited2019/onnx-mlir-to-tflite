/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class TopKLowering final : public OpConversionPattern<ONNXTopKOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXTopKOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto valueType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto indexType = dyn_cast<RankedTensorType>(op->getResult(1).getType());
    if (!inputType || !valueType || !indexType || !inputType.hasStaticShape() ||
        !valueType.hasStaticShape() || !indexType.hasStaticShape() ||
        !inputType.getElementType().isF32() ||
        !valueType.getElementType().isF32() ||
        !indexType.getElementType().isSignlessInteger(64) ||
        valueType.getShape() != indexType.getShape())
      return op.emitError("ONNXToTFL TopK requires a static FP32 input/value "
                          "and matching static i64 indices"),
             failure();

    int64_t rank = inputType.getRank();
    if (rank < 1 || rank > 5)
      return op.emitError("ONNXToTFL TopK supports ranks 1 through 5"),
             failure();
    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError("ONNXToTFL TopK axis is out of range"), failure();
    if ((op.getLargest() != 0 && op.getLargest() != 1) ||
        (op.getSorted() != 0 && op.getSorted() != 1))
      return op.emitError("ONNXToTFL TopK largest and sorted must be 0 or 1"),
             failure();

    FailureOr<SmallVector<int64_t>> kValues = getConstantIntValues(op.getK());
    if (failed(kValues) || kValues->size() != 1)
      return op.emitError("ONNXToTFL TopK requires a constant scalar K"),
             failure();
    int64_t k = kValues->front();
    if (k <= 0 || k > inputType.getShape()[axis] ||
        k > std::numeric_limits<int32_t>::max())
      return op.emitError("ONNXToTFL TopK K is out of range"), failure();

    SmallVector<int64_t> expectedShape(inputType.getShape());
    expectedShape[axis] = k;
    if (!llvm::equal(expectedShape, valueType.getShape()))
      return op.emitError("ONNXToTFL TopK result shape is inconsistent"),
             failure();

    Location loc = op.getLoc();
    Value input = adaptor.getX();
    auto transpose = [&](Value value, ArrayRef<int64_t> permutation,
                         ArrayRef<int64_t> shape, Type elementType) -> Value {
      Value permutationValue =
          createI32ShapeConstant(rewriter, loc, permutation);
      auto resultType = RankedTensorType::get(shape, elementType);
      return createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{resultType}, ValueRange{value, permutationValue})
          ->getResult(0);
    };

    // Rank-4 FP32 values are physical NHWC at the bridge boundary. TopK axes
    // are logical ONNX axes, so perform the operation in logical order.
    if (rank == 4)
      input = transpose(input, {0, 3, 1, 2}, inputType.getShape(),
          inputType.getElementType());

    SmallVector<int64_t> permutation;
    for (int64_t dimension = 0; dimension < rank; ++dimension)
      if (dimension != axis)
        permutation.push_back(dimension);
    permutation.push_back(axis);

    SmallVector<int64_t> permutedInputShape;
    for (int64_t dimension : permutation)
      permutedInputShape.push_back(inputType.getShape()[dimension]);
    if (axis != rank - 1)
      input = transpose(
          input, permutation, permutedInputShape, inputType.getElementType());

    auto permutedInputType =
        RankedTensorType::get(permutedInputShape, rewriter.getF32Type());
    if (op.getLargest() == 0)
      input = createTFLOperation(rewriter, loc, "tfl.neg",
          TypeRange{permutedInputType}, ValueRange{input})
                  ->getResult(0);

    SmallVector<int64_t> permutedResultShape(permutedInputShape);
    permutedResultShape.back() = k;
    auto permutedValueType =
        RankedTensorType::get(permutedResultShape, rewriter.getF32Type());
    auto permutedI32Type =
        RankedTensorType::get(permutedResultShape, rewriter.getI32Type());
    Value kValue =
        createI32ScalarTensorConstant(rewriter, loc, static_cast<int32_t>(k));
    Operation *topK = createTFLOperation(rewriter, loc, "tfl.topk_v2",
        TypeRange{permutedValueType, permutedI32Type},
        ValueRange{input, kValue});
    Value values = topK->getResult(0);
    Value indices = topK->getResult(1);
    if (op.getLargest() == 0)
      values = createTFLOperation(rewriter, loc, "tfl.neg",
          TypeRange{permutedValueType}, ValueRange{values})
                   ->getResult(0);

    // TFLite TopKV2 always sorts its results. ONNX explicitly leaves the
    // order undefined when sorted=0, so retaining this deterministic native
    // order is conformant and avoids reproducing a backend-specific heap
    // layout.

    if (axis != rank - 1) {
      SmallVector<int64_t> inversePermutation(rank);
      for (auto [position, dimension] : llvm::enumerate(permutation))
        inversePermutation[dimension] = position;
      values = transpose(values, inversePermutation, valueType.getShape(),
          rewriter.getF32Type());
      indices = transpose(indices, inversePermutation, indexType.getShape(),
          rewriter.getI32Type());
    }

    Value indicesI64 = createTFLOperation(
        rewriter, loc, "tfl.cast", TypeRange{indexType}, ValueRange{indices})
                           ->getResult(0);
    if (rank == 4)
      values = transpose(values, {0, 2, 3, 1},
          cast<RankedTensorType>(convertRank4NCHWToNHWCType(valueType))
              .getShape(),
          rewriter.getF32Type());

    rewriter.replaceOp(op, ValueRange{values, indicesI64});
    return success();
  }
};

} // namespace

void populateLoweringONNXTopKOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<TopKLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
