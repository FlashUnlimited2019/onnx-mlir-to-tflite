/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class CompressLowering final : public OpConversionPattern<ONNXCompressOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXCompressOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    auto conditionConstant =
        op.getCondition().getDefiningOp<ONNXConstantOp>();
    auto condition = conditionConstant
                         ? dyn_cast_or_null<ElementsAttr>(
                               conditionConstant.getValueAttr())
                         : ElementsAttr();
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || !condition)
      return op.emitError("ONNXToTFL Compress requires static f32 tensors and "
                          "a constant condition"),
             failure();
    if (!op.getAxis().has_value())
      return op.emitError("ONNXToTFL Compress currently requires axis"),
             failure();
    int64_t axis = normalizeAxis(*op.getAxis(), inputType.getRank());
    if (axis < 0 || axis >= inputType.getRank())
      return op.emitError("Compress axis is out of range"), failure();
    SmallVector<int32_t> indices;
    int64_t limit = inputType.getShape()[axis];
    int64_t index = 0;
    for (APInt selected : condition.getValues<APInt>()) {
      if (index >= limit)
        break;
      if (!selected.isZero())
        indices.push_back(static_cast<int32_t>(index));
      ++index;
    }
    if (static_cast<int64_t>(indices.size()) != resultType.getShape()[axis])
      return op.emitError("Compress constant condition disagrees with result "
                          "shape"),
             failure();

    Value input = adaptor.getInput();
    if (inputType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{inputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }
    auto indicesType = RankedTensorType::get(
        {static_cast<int64_t>(indices.size())}, rewriter.getI32Type());
    Value indicesValue = arith::ConstantOp::create(rewriter, op.getLoc(),
        indicesType, DenseIntElementsAttr::get(indicesType, indices));
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr(
            "axis", rewriter.getI32IntegerAttr(static_cast<int32_t>(axis))),
        rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.gather",
        TypeRange{resultType}, ValueRange{input, indicesValue}, attributes)
                       ->getResult(0);
    if (resultType.getRank() == 4) {
      Type physicalType = convertRank4NCHWToNHWCType(resultType);
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TriluLowering final : public OpConversionPattern<ONNXTriluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXTriluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !resultType || !inputType.hasStaticShape() ||
        !resultType.hasStaticShape() || inputType.getRank() < 2 ||
        !inputType.getElementType().isF32() ||
        !resultType.getElementType().isF32())
      return op.emitError("ONNXToTFL Trilu requires a static rank >= 2 f32 "
                          "tensor"),
             failure();
    int64_t diagonal = 0;
    if (!isa<NoneType>(op.getK().getType())) {
      FailureOr<SmallVector<int64_t>> values = getConstantIntValues(op.getK());
      if (failed(values) || values->size() != 1)
        return op.emitError("ONNXToTFL Trilu requires constant k"), failure();
      diagonal = (*values)[0];
    }

    Value input = adaptor.getInput();
    if (inputType.getRank() == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{inputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }
    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t rows = shape[shape.size() - 2];
    int64_t columns = shape.back();
    int64_t matrices = inputType.getNumElements() / (rows * columns);
    SmallVector<float> mask;
    mask.reserve(inputType.getNumElements());
    bool upper = op.getUpper() != 0;
    for (int64_t matrix = 0; matrix < matrices; ++matrix)
      for (int64_t row = 0; row < rows; ++row)
        for (int64_t column = 0; column < columns; ++column) {
          bool keep = upper ? column - row >= diagonal
                            : column - row <= diagonal;
          mask.push_back(keep ? 1.0f : 0.0f);
        }
    Value maskValue = arith::ConstantOp::create(rewriter, op.getLoc(),
        inputType, DenseFPElementsAttr::get(inputType, mask));
    SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{resultType}, ValueRange{input, maskValue}, attributes)
                       ->getResult(0);
    if (resultType.getRank() == 4) {
      Type physicalType = convertRank4NCHWToNHWCType(resultType);
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXCompressTriluOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CompressLowering, TriluLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
