/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

// This is an operator-selection policy, not a definition of an embedding.
// Keep narrow tables on the general Gather path.
constexpr int64_t kMinEmbeddingLookupWidth = 128;

ElementsAttr getConstantElements(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<ONNXConstantOp>())
    attribute = constant.getValueAttr();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  return dyn_cast_or_null<ElementsAttr>(attribute);
}

bool isKnownNonNegativeIndex(Value value) {
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getName().getStringRef() != "tfl.maximum" ||
      producer->getNumOperands() != 2)
    return false;
  // Integer Maximum with a non-negative scalar bound proves this property
  // without relying on names or input-value assumptions. In particular this
  // recognizes a lowered Clip while leaving its padding mask untouched.
  for (Value operand : producer->getOperands()) {
    auto bound = getConstantElements(operand);
    if (bound && bound.size() == 1 && bound.getElementType().isIntOrIndex() &&
        !(*bound.getValues<APInt>().begin()).isNegative())
      return true;
  }
  return false;
}

bool canUseEmbeddingLookup(Value data, Value indices, RankedTensorType dataType,
    RankedTensorType indicesType, int64_t axis) {
  constexpr int64_t limit = std::numeric_limits<int32_t>::max();
  if (axis != 0 || dataType.getRank() != 2 ||
      !dataType.getElementType().isF32() || dataType.getDimSize(0) <= 0 ||
      dataType.getDimSize(0) > limit ||
      dataType.getDimSize(1) < kMinEmbeddingLookupWidth ||
      dataType.getDimSize(1) > limit ||
      getONNXToTFLPhysicalToLogicalLayout(dataType) !=
          SmallVector<int64_t>{0, 1})
    return false;

  // Inspect only metadata: do not materialize or duplicate large weight
  // buffers. Runtime tables, including block arguments, must remain Gather.
  auto table = getConstantElements(data);
  if (!table || !table.getElementType().isF32() ||
      cast<ShapedType>(table.getType()).getShape() != dataType.getShape() ||
      getConstantElements(indices))
    return false;

  // Leave empty and oversized cases on the existing path. Both the flat
  // lookup count and its result must fit the runtime's shape arithmetic.
  int64_t count = 1;
  for (int64_t dimension : indicesType.getShape()) {
    if (dimension <= 0 || count > limit / dimension)
      return false;
    count *= dimension;
  }
  return count <= limit / dataType.getDimSize(1);
}

LogicalResult lowerEmbeddingLookup(ONNXGatherOp op, Value data, Value indices,
    RankedTensorType dataType, RankedTensorType indicesType,
    RankedTensorType resultType, ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto physicalIndicesType = cast<RankedTensorType>(indices.getType());
  int64_t count = indicesType.getNumElements();
  int64_t rows = dataType.getDimSize(0);
  int64_t width = dataType.getDimSize(1);
  bool nonNegative = isKnownNonNegativeIndex(indices);
  auto reshape = [&](Value value, ArrayRef<int64_t> shape) -> Value {
    auto type = RankedTensorType::get(
        shape, cast<RankedTensorType>(value.getType()).getElementType());
    if (value.getType() == type)
      return value;
    return createTFLOperation(rewriter, loc, "tfl.reshape", TypeRange{type},
        {value, createI32ShapeConstant(rewriter, loc, shape)})
        ->getResult(0);
  };
  auto scalar = [&](Type elementType, int64_t value) -> Value {
    auto type = RankedTensorType::get({}, elementType);
    return arith::ConstantOp::create(rewriter, loc, type,
        DenseIntElementsAttr::get(
            type, APInt(elementType.getIntOrFloatBitWidth(), value)));
  };

  // All index arithmetic is rank one, even for high-rank index tensors.
  indices = reshape(indices, {count});
  if (indicesType.getElementType().isSignlessInteger(64)) {
    auto flatType = cast<RankedTensorType>(indices.getType());
    // Valid ONNX indices are [-rows, rows). Saturate outside that interval
    // to INVALID sentinels, not valid rows, before narrowing. This prevents
    // large i64 values from wrapping into valid i32 indices. Both sentinels
    // fit i32; after negative-index normalization they become -1 or rows,
    // which the runtime rejects. No valid index changes here.
    Value upper = scalar(rewriter.getI64Type(), rows);
    indices = createTFLOperation(
        rewriter, loc, "tfl.minimum", TypeRange{flatType}, {indices, upper})
                  ->getResult(0);
    if (!nonNegative) {
      Value lower = scalar(rewriter.getI64Type(), -rows - 1);
      indices = createTFLOperation(
          rewriter, loc, "tfl.maximum", TypeRange{flatType}, {indices, lower})
                    ->getResult(0);
    }
    auto narrowedType = RankedTensorType::get({count}, rewriter.getI32Type());
    indices = createTFLOperation(
        rewriter, loc, "tfl.cast", TypeRange{narrowedType}, {indices})
                  ->getResult(0);
  }

  if (!nonNegative) {
    auto flatType = cast<RankedTensorType>(indices.getType());
    auto conditionType = RankedTensorType::get({count}, rewriter.getI1Type());
    Value zero = scalar(rewriter.getI32Type(), 0);
    Value extent = scalar(rewriter.getI32Type(), rows);
    Value negative = createTFLOperation(
        rewriter, loc, "tfl.less", TypeRange{conditionType}, {indices, zero})
                         ->getResult(0);
    // The unselected Add branch must not overflow for large positive indices.
    Value toWrap = createTFLOperation(
        rewriter, loc, "tfl.minimum", TypeRange{flatType}, {indices, zero})
                       ->getResult(0);
    Value wrapped =
        createTFLOperation(rewriter, loc, "tfl.add", TypeRange{flatType},
            {toWrap, extent}, {getFusedActivationNone(rewriter)})
            ->getResult(0);
    indices = createTFLOperation(rewriter, loc, "tfl.select_v2",
        TypeRange{flatType}, {negative, wrapped, indices})
                  ->getResult(0);
  }

  auto flatResultType =
      RankedTensorType::get({count, width}, rewriter.getF32Type());
  Value result = createTFLOperation(rewriter, loc, "tfl.embedding_lookup",
      TypeRange{flatResultType}, {indices, data})
                     ->getResult(0);
  SmallVector<int64_t> naturalShape(physicalIndicesType.getShape());
  naturalShape.push_back(width);
  result = reshape(result, naturalShape);
  SmallVector<int64_t> naturalLayout =
      getONNXToTFLPhysicalToLogicalLayout(indicesType);
  naturalLayout.push_back(indicesType.getRank());
  FailureOr<Value> physicalResult = convertTFLValueLayout(rewriter, loc, result,
      stripONNXToTFLLayoutEncoding(resultType), naturalLayout,
      getONNXToTFLPhysicalToLogicalLayout(resultType));
  if (failed(physicalResult))
    return failure();
  rewriter.replaceOp(op, *physicalResult);
  return success();
}

class GatherLowering final : public OpConversionPattern<ONNXGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceIndicesType =
        dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    Type dataElementType =
        sourceDataType ? sourceDataType.getElementType() : Type();
    bool supportedDataType =
        dataElementType &&
        (dataElementType.isF32() || dataElementType.isInteger(1) ||
            dataElementType.isSignlessInteger(32) ||
            dataElementType.isSignlessInteger(64));
    if (!sourceDataType || !sourceIndicesType || !sourceResultType ||
        !sourceDataType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || sourceDataType.getRank() < 1 ||
        !supportedDataType ||
        sourceResultType.getElementType() != dataElementType ||
        !sourceIndicesType.hasStaticShape() ||
        (!sourceIndicesType.getElementType().isSignlessInteger(32) &&
            !sourceIndicesType.getElementType().isSignlessInteger(64))) {
      op.emitError("ONNXToTFL Gather requires matching static FP32, bool, "
                   "i32, or i64 "
                   "data/result and static i32/i64 indices");
      return failure();
    }

    int64_t dataRank = sourceDataType.getRank();
    int64_t indicesRank = sourceIndicesType.getRank();
    int64_t resultRank = sourceResultType.getRank();
    if (dataRank < 1 || dataRank > 6 || indicesRank > 6 || resultRank > 6) {
      op.emitError("ONNXToTFL Gather supports data, indices, and result ranks "
                   "up to 6");
      return failure();
    }
    int64_t axis = normalizeAxis(op.getAxis(), dataRank);
    if (axis < 0 || axis >= dataRank)
      return op.emitError() << "invalid Gather axis " << op.getAxis(),
             failure();

    SmallVector<int64_t> expectedShape;
    expectedShape.append(sourceDataType.getShape().begin(),
        sourceDataType.getShape().begin() + axis);
    expectedShape.append(sourceIndicesType.getShape().begin(),
        sourceIndicesType.getShape().end());
    expectedShape.append(sourceDataType.getShape().begin() + axis + 1,
        sourceDataType.getShape().end());
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError(
          "Gather result shape does not match data, indices, and axis");
      return failure();
    }

    int64_t axisSize = sourceDataType.getShape()[axis];
    Value indices;
    FailureOr<SmallVector<int64_t>> indexValues =
        getConstantIntValues(op.getIndices());
    if (failed(indexValues) && canUseEmbeddingLookup(data, adaptor.getIndices(),
                                   sourceDataType, sourceIndicesType, axis))
      return lowerEmbeddingLookup(op, data, adaptor.getIndices(),
          sourceDataType, sourceIndicesType, sourceResultType, rewriter);
    if (dataRank > 5 && indicesRank == 0 && resultRank == dataRank - 1 &&
        axisSize == 1 && succeeded(indexValues) && indexValues->size() == 1 &&
        ((*indexValues)[0] == 0 || (*indexValues)[0] == -1)) {
      auto physicalResultType = cast<RankedTensorType>(
          convertONNXToTFLPhysicalType(sourceResultType));
      Value resultShape = createI32ShapeConstant(
          rewriter, op.getLoc(), physicalResultType.getShape());
      Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{physicalResultType},
          ValueRange{adaptor.getData(), resultShape})
                         ->getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }
    if (succeeded(indexValues)) {
      if (static_cast<int64_t>(indexValues->size()) !=
          sourceIndicesType.getNumElements()) {
        op.emitError("Gather constant size does not match its tensor type");
        return failure();
      }
      SmallVector<int32_t> normalizedIndices;
      normalizedIndices.reserve(indexValues->size());
      for (int64_t rawIndex : *indexValues) {
        int64_t index = rawIndex < 0 ? rawIndex + axisSize : rawIndex;
        if (index < 0 || index >= axisSize) {
          op.emitError() << "Gather index " << rawIndex
                         << " is out of bounds for axis " << axis;
          return failure();
        }
        normalizedIndices.push_back(static_cast<int32_t>(index));
      }
      auto normalizedType = RankedTensorType::get(
          sourceIndicesType.getShape(), rewriter.getI32Type());
      indices = arith::ConstantOp::create(rewriter, op.getLoc(), normalizedType,
          DenseIntElementsAttr::get(
              normalizedType, ArrayRef<int32_t>(normalizedIndices)));
    } else {
      // ONNX permits negative runtime indices, while the TFLite Gather kernel
      // expects non-negative coordinates. Normalize them before gathering.
      indices = adaptor.getIndices();
      Type integerType = sourceIndicesType.getElementType();
      auto scalarType = RankedTensorType::get({}, integerType);
      auto makeScalar = [&](int64_t value) -> Value {
        DenseIntElementsAttr attr;
        if (integerType.isSignlessInteger(32))
          attr = DenseIntElementsAttr::get(
              scalarType, ArrayRef<int32_t>{static_cast<int32_t>(value)});
        else
          attr =
              DenseIntElementsAttr::get(scalarType, ArrayRef<int64_t>{value});
        return arith::ConstantOp::create(
            rewriter, op.getLoc(), scalarType, attr);
      };
      Value zero = makeScalar(0);
      Value extent = makeScalar(axisSize);
      auto physicalIndicesType = cast<RankedTensorType>(
          convertONNXToTFLPhysicalType(sourceIndicesType));
      auto conditionType = RankedTensorType::get(
          physicalIndicesType.getShape(), rewriter.getI1Type());
      Value negative = createTFLOperation(rewriter, op.getLoc(), "tfl.less",
          TypeRange{conditionType}, ValueRange{indices, zero})
                           ->getResult(0);
      SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
      Value wrapped = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
          TypeRange{physicalIndicesType}, ValueRange{indices, extent},
          fusedNone)
                          ->getResult(0);
      indices = createTFLOperation(rewriter, op.getLoc(), "tfl.select_v2",
          TypeRange{physicalIndicesType},
          ValueRange{negative, wrapped, indices})
                    ->getResult(0);
    }

    bool hasPerValueLayout = sourceDataType.getEncoding() ||
                             sourceIndicesType.getEncoding() ||
                             sourceResultType.getEncoding();
    if (hasPerValueLayout) {
      SmallVector<int64_t> dataLayout =
          getONNXToTFLPhysicalToLogicalLayout(sourceDataType);
      SmallVector<int64_t> indicesLayout =
          getONNXToTFLPhysicalToLogicalLayout(sourceIndicesType);
      int64_t physicalAxis = mapONNXToTFLLogicalAxis(sourceDataType, axis);
      if (physicalAxis < 0)
        return op.emitError("Gather data has an invalid layout encoding"),
               failure();

      SmallVector<int64_t> naturalResultLayout;
      naturalResultLayout.reserve(resultRank);
      auto mapDataDimension = [&](int64_t logicalDimension) {
        return logicalDimension < axis ? logicalDimension
                                       : logicalDimension + indicesRank - 1;
      };
      for (int64_t physicalDataDimension : dataLayout) {
        if (physicalDataDimension == axis) {
          for (int64_t physicalIndexDimension : indicesLayout)
            naturalResultLayout.push_back(axis + physicalIndexDimension);
        } else {
          naturalResultLayout.push_back(
              mapDataDimension(physicalDataDimension));
        }
      }
      if (static_cast<int64_t>(naturalResultLayout.size()) != resultRank)
        return op.emitError("Gather layouts do not match the result rank"),
               failure();

      auto logicalResultType = stripONNXToTFLLayoutEncoding(sourceResultType);
      auto naturalEncoding =
          DenseI64ArrayAttr::get(rewriter.getContext(), naturalResultLayout);
      auto naturalEncodedType =
          RankedTensorType::get(logicalResultType.getShape(),
              logicalResultType.getElementType(), naturalEncoding);
      Type naturalPhysicalType =
          convertONNXToTFLPhysicalType(naturalEncodedType);
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr(
              "axis", rewriter.getI32IntegerAttr(physicalAxis)),
          rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
      Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather",
          TypeRange{naturalPhysicalType}, ValueRange{data, indices}, attributes)
                           ->getResult(0);
      FailureOr<Value> result = convertTFLValueLayout(rewriter, op.getLoc(),
          gathered, logicalResultType, naturalResultLayout,
          getONNXToTFLPhysicalToLogicalLayout(sourceResultType));
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }

    bool usesFP32ActivationLayout = dataElementType.isF32();
    if (usesFP32ActivationLayout && dataRank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceDataType}, ValueRange{data, permutation})
                 ->getResult(0);
    }
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
        rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
    Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather",
        TypeRange{sourceResultType}, ValueRange{data, indices}, attributes)
                         ->getResult(0);
    if (!usesFP32ActivationLayout || resultRank != 4) {
      rewriter.replaceOp(op, gathered);
      return success();
    }

    Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
    Value permutation =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    Operation *physicalResult =
        createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
            TypeRange{physicalResultType}, ValueRange{gathered, permutation});
    rewriter.replaceOp(op, physicalResult->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXGatherOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<GatherLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
