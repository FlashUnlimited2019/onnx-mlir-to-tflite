/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

// An identity ScatterElements overwrites every data element with the update at
// the same logical coordinate. Forwarding updates is both exact and avoids
// materializing a large rank+1 coordinate tensor for transformer attention
// masks.
class ScatterElementsIdentityLowering final
    : public OpConversionPattern<ONNXScatterElementsOp> {
public:
  ScatterElementsIdentityLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<ONNXScatterElementsOp>(
            typeConverter, context, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(ONNXScatterElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto updatesType = dyn_cast<RankedTensorType>(op.getUpdates().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!dataType || !indicesType || !updatesType || !resultType ||
        !dataType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !updatesType.hasStaticShape() || !resultType.hasStaticShape() ||
        failed(validateStaticF32Tensor(
            op, adaptor.getData().getType(), "ScatterElements data")) ||
        failed(validateStaticF32Tensor(
            op, adaptor.getUpdates().getType(), "ScatterElements updates")) ||
        failed(
            validateStaticF32Tensor(op, resultType, "ScatterElements result")))
      return failure();

    int64_t rank = dataType.getRank();
    if (rank < 1 || rank > 5 || indicesType.getRank() != rank ||
        updatesType.getRank() != rank || resultType.getRank() != rank ||
        !indicesType.getElementType().isIntOrIndex() ||
        dataType.getShape() != indicesType.getShape() ||
        dataType.getShape() != updatesType.getShape() ||
        dataType.getShape() != resultType.getShape()) {
      return failure();
    }
    if (op.getReduction() != "none") {
      return failure();
    }

    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return failure();

    FailureOr<SmallVector<int64_t>> indexValues =
        getConstantIntValues(op.getIndices());
    if (failed(indexValues))
      return failure();

    ArrayRef<int64_t> shape = dataType.getShape();
    int64_t elementCount = dataType.getNumElements();
    if (static_cast<int64_t>(indexValues->size()) != elementCount) {
      return failure();
    }

    SmallVector<int64_t> position(rank);
    for (int64_t flatIndex = 0; flatIndex < elementCount; ++flatIndex) {
      int64_t remainder = flatIndex;
      for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
        position[dimension] = remainder % shape[dimension];
        remainder /= shape[dimension];
      }
      int64_t index = (*indexValues)[flatIndex];
      if (index < 0)
        index += shape[axis];
      if (index != position[axis]) {
        return failure();
      }
    }

    rewriter.replaceOp(op, adaptor.getUpdates());
    return success();
  }
};

// Lower general static-shape ScatterElements to logical GatherNd/ScatterNd.
// The runtime indices may vary; only ranks and extents are compile-time data.
// ONNX leaves duplicate-index ordering unspecified for reduction="none", so
// ScatterNd's additive accumulation is valid for conforming unique-index uses.
class ScatterElementsLowering final
    : public OpConversionPattern<ONNXScatterElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXScatterElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto updatesType = dyn_cast<RankedTensorType>(op.getUpdates().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!dataType || !indicesType || !updatesType || !resultType ||
        !dataType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !updatesType.hasStaticShape() || !resultType.hasStaticShape() ||
        failed(validateStaticF32Tensor(
            op, adaptor.getData().getType(), "ScatterElements data")) ||
        failed(validateStaticF32Tensor(
            op, adaptor.getUpdates().getType(), "ScatterElements updates")) ||
        failed(
            validateStaticF32Tensor(op, resultType, "ScatterElements result")))
      return failure();

    int64_t rank = dataType.getRank();
    if (rank < 1 || rank > 5 || indicesType.getRank() != rank ||
        updatesType.getRank() != rank || resultType.getRank() != rank ||
        (!indicesType.getElementType().isSignlessInteger(32) &&
            !indicesType.getElementType().isSignlessInteger(64)) ||
        indicesType.getShape() != updatesType.getShape() ||
        dataType.getShape() != resultType.getShape() ||
        dataType.getElementType() != updatesType.getElementType())
      return op.emitError("ONNXToTFL ScatterElements requires static "
                          "equal-rank rank-1 through rank-5 f32 tensors and "
                          "i32/i64 indices"),
             failure();
    if (op.getReduction() != "none")
      return op.emitError("ONNXToTFL general ScatterElements currently "
                          "supports reduction=none only"),
             failure();

    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError() << "invalid ScatterElements axis " << op.getAxis(),
             failure();
    ArrayRef<int64_t> dataShape = dataType.getShape();
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dataShape[dimension] <= 0 || indicesShape[dimension] <= 0 ||
          dataShape[dimension] > std::numeric_limits<int32_t>::max() ||
          indicesShape[dimension] > std::numeric_limits<int32_t>::max() ||
          (dimension != axis && indicesShape[dimension] > dataShape[dimension]))
        return op.emitError("unsupported ScatterElements static dimensions"),
               failure();
    }
    int64_t updateCount = indicesType.getNumElements();
    if (updateCount > std::numeric_limits<int32_t>::max() / rank)
      return op.emitError("ScatterElements coordinate tensor is too large"),
             failure();

    Location loc = op.getLoc();
    auto i32IndicesType =
        RankedTensorType::get(indicesShape, rewriter.getI32Type());
    Value runtimeIndices = adaptor.getIndices();
    if (indicesType.getElementType().isSignlessInteger(64))
      runtimeIndices = createTFLOperation(rewriter, loc, "tfl.cast",
          TypeRange{i32IndicesType}, ValueRange{runtimeIndices})
                           ->getResult(0);

    Value comparisonIndices = runtimeIndices;
    auto comparisonType = i32IndicesType;
    if (rank > 4) {
      comparisonType =
          RankedTensorType::get({updateCount}, rewriter.getI32Type());
      Value flatShape = createI32ShapeConstant(rewriter, loc, {updateCount});
      comparisonIndices = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{comparisonType}, ValueRange{runtimeIndices, flatShape})
                              ->getResult(0);
    }
    Value zeroIndex = createI32ScalarTensorConstant(rewriter, loc, 0);
    Value axisExtent = createI32ScalarTensorConstant(
        rewriter, loc, static_cast<int32_t>(dataShape[axis]));
    auto negativeType =
        RankedTensorType::get(comparisonType.getShape(), rewriter.getI1Type());
    Value negative = createTFLOperation(rewriter, loc, "tfl.less",
        TypeRange{negativeType}, ValueRange{comparisonIndices, zeroIndex})
                         ->getResult(0);
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Value wrapped =
        createTFLOperation(rewriter, loc, "tfl.add", TypeRange{comparisonType},
            ValueRange{comparisonIndices, axisExtent}, fusedNone)
            ->getResult(0);
    comparisonIndices = createTFLOperation(rewriter, loc, "tfl.select_v2",
        TypeRange{comparisonType},
        ValueRange{negative, wrapped, comparisonIndices})
                            ->getResult(0);
    if (rank > 4) {
      Value logicalShape = createI32ShapeConstant(rewriter, loc, indicesShape);
      runtimeIndices = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{i32IndicesType},
          ValueRange{comparisonIndices, logicalShape})
                           ->getResult(0);
    } else {
      runtimeIndices = comparisonIndices;
    }

    SmallVector<int64_t> componentShape(indicesShape);
    componentShape.push_back(1);
    auto componentType =
        RankedTensorType::get(componentShape, rewriter.getI32Type());
    Value componentShapeValue =
        createI32ShapeConstant(rewriter, loc, componentShape);
    Value runtimeComponent = createTFLOperation(rewriter, loc, "tfl.reshape",
        TypeRange{componentType},
        ValueRange{runtimeIndices, componentShapeValue})
                                 ->getResult(0);

    SmallVector<Value> components;
    components.reserve(rank);
    SmallVector<int64_t> position(rank);
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dimension == axis) {
        components.push_back(runtimeComponent);
        continue;
      }
      SmallVector<int32_t> values;
      values.reserve(updateCount);
      for (int64_t flatIndex = 0; flatIndex < updateCount; ++flatIndex) {
        int64_t remainder = flatIndex;
        for (int64_t coordinate = rank - 1; coordinate >= 0; --coordinate) {
          position[coordinate] = remainder % indicesShape[coordinate];
          remainder /= indicesShape[coordinate];
        }
        values.push_back(static_cast<int32_t>(position[dimension]));
      }
      components.push_back(arith::ConstantOp::create(rewriter, loc,
          componentType,
          DenseIntElementsAttr::get(componentType, ArrayRef<int32_t>(values))));
    }
    SmallVector<int64_t> coordinateShape(indicesShape);
    coordinateShape.push_back(rank);
    auto coordinateType =
        RankedTensorType::get(coordinateShape, rewriter.getI32Type());
    SmallVector<NamedAttribute> concatenateAttributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(rank)),
        getFusedActivationNone(rewriter)};
    Value coordinates = createTFLOperation(rewriter, loc, "tfl.concatenation",
        TypeRange{coordinateType}, components, concatenateAttributes)
                            ->getResult(0);

    auto restoreLogicalRank4 = [&](Value value,
                                   RankedTensorType sourceType) -> Value {
      if (sourceType.getRank() != 4)
        return value;
      Value permutation = createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2});
      return createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{sourceType}, ValueRange{value, permutation})
          ->getResult(0);
    };
    Value data = restoreLogicalRank4(adaptor.getData(), dataType);
    Value updates = restoreLogicalRank4(adaptor.getUpdates(), updatesType);
    Value oldValues = createTFLOperation(rewriter, loc, "tfl.gather_nd",
        TypeRange{updatesType}, ValueRange{data, coordinates})
                          ->getResult(0);
    Value deltas = createTFLOperation(rewriter, loc, "tfl.sub",
        TypeRange{updatesType}, ValueRange{updates, oldValues}, fusedNone)
                       ->getResult(0);
    Value shape = createI32ShapeConstant(rewriter, loc, dataShape);
    Value sparseDeltas = createTFLOperation(rewriter, loc, "tfl.scatter_nd",
        TypeRange{dataType}, ValueRange{coordinates, deltas, shape})
                             ->getResult(0);
    Value result = createTFLOperation(rewriter, loc, "tfl.add",
        TypeRange{resultType}, ValueRange{data, sparseDeltas}, fusedNone)
                       ->getResult(0);
    if (rank == 4) {
      Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
      Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
      result = createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXScatterElementsOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ScatterElementsIdentityLowering, ScatterElementsLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
