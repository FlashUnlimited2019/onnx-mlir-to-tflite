/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {
namespace {

// Recognize complete, constant row windows, not just a few sampled indices.
// For each block of P rows, indices[r,c] = A - (r % P)*S + c,
// where A = (P-1)*S. Flattening a block of [P,D] starting at A and
// reinterpreting it with row width D-S gives exactly these windows:
//   A + r*(D-S) + c = r*D + (A-r*S+c).
// The cropped interval ends at P*D-S, so no padding or cross-block read is
// needed. This avoids materializing an output-sized GatherNd coordinate table.
FailureOr<Value> lowerConstantRowWindows(ONNXGatherElementsOp op,
    ConversionPatternRewriter &rewriter, Value data, RankedTensorType dataType,
    RankedTensorType resultType, ArrayRef<int64_t> indices) {
  if (dataType.getRank() != 2 || resultType.getRank() != 2 ||
      dataType.getShape()[0] != resultType.getShape()[0])
    return failure();
  int64_t rows = dataType.getShape()[0], width = dataType.getShape()[1];
  int64_t columns = resultType.getShape()[1];
  if (rows <= 0 || width <= 0 || columns <= 0 ||
      indices.size() != static_cast<uint64_t>(rows * columns))
    return failure();
  auto normalize = [width](int64_t index) -> int64_t {
    if (index < -width || index >= width)
      return -1;
    return index < 0 ? index + width : index;
  };
  int64_t start = normalize(indices.front());
  if (start < 0 || columns > width - start)
    return failure();
  int64_t step = 0, period = 1;
  if (rows > 1) {
    int64_t next = normalize(indices[columns]);
    if (next < 0 || next > start)
      return failure();
    step = start - next;
    if (step != 0) {
      if (start % step != 0)
        return failure();
      period = start / step + 1;
      if (rows % period != 0 || step >= width || columns > width - step ||
          // Avoid copying a wide intermediate just to gather a narrow window.
          width - step > 2 * columns ||
          period > std::numeric_limits<int32_t>::max() / width)
        return failure();
    }
  }
  int64_t flat = 0;
  for (int64_t row = 0; row < rows; ++row) {
    int64_t expected = start - (row % period) * step;
    for (int64_t column = 0; column < columns; ++column)
      if (normalize(indices[flat++]) != expected + column)
        return failure();
  }

  Location loc = op.getLoc();
  auto reshape = [&](Value input, ArrayRef<int64_t> shape) -> Value {
    auto type = RankedTensorType::get(shape, dataType.getElementType());
    Value dims = createI32ShapeConstant(rewriter, loc, shape);
    return createTFLOperation(
        rewriter, loc, "tfl.reshape", TypeRange{type}, ValueRange{input, dims})
        ->getResult(0);
  };
  auto slice = [&](Value input, ArrayRef<int64_t> begin,
                   ArrayRef<int64_t> shape) -> Value {
    auto type = RankedTensorType::get(shape, dataType.getElementType());
    Value starts = createI32ShapeConstant(rewriter, loc, begin);
    Value sizes = createI32ShapeConstant(rewriter, loc, shape);
    return createTFLOperation(rewriter, loc, "tfl.slice", TypeRange{type},
        ValueRange{input, starts, sizes})
        ->getResult(0);
  };
  if (step == 0)
    return slice(data, {0, start}, {rows, columns});
  int64_t batches = rows / period, rowWidth = width - step;
  Value blocks = reshape(data, {batches, period * width});
  Value shifted = slice(blocks, {0, start}, {batches, period * rowWidth});
  Value windows = reshape(shifted, {rows, rowWidth});
  return slice(windows, {0, 0}, {rows, columns});
}

// A broadcast index that is constant across each complete trailing slice
// selects rows, not individual scalar coordinates. Keep per-prefix indices
// distinct with batch_dims=1; a vector shared by every prefix uses
// batch_dims=0. Partial slices and partially broadcast batch prefixes use the
// general path.
FailureOr<Value> lowerBroadcastRowGather(ONNXGatherElementsOp op,
    ConversionPatternRewriter &rewriter, Value data, Value indices,
    RankedTensorType dataType, RankedTensorType resultType, int64_t axis,
    bool logicalData) {
  Operation *expand = indices.getDefiningOp();
  if (!expand || expand->getName().getStringRef() != "tfl.broadcast_to" ||
      expand->getNumOperands() != 2 || expand->getNumResults() != 1)
    return failure();
  Value compact = expand->getOperand(0);
  auto compactType = dyn_cast<RankedTensorType>(compact.getType());
  auto expandedType = dyn_cast<RankedTensorType>(indices.getType());
  int64_t rank = dataType.getRank();
  if (!compactType || !compactType.hasStaticShape() ||
      compactType.getEncoding() || !expandedType ||
      expandedType.getShape() != resultType.getShape() ||
      compactType.getRank() > rank ||
      compactType.getElementType() != expandedType.getElementType() ||
      (!compactType.getElementType().isSignlessInteger(32) &&
          !compactType.getElementType().isSignlessInteger(64)))
    return failure();
  auto targetShape = getConstantIntValues(expand->getOperand(1));
  if (failed(targetShape) || !llvm::equal(*targetShape, resultType.getShape()))
    return failure();
  SmallVector<int64_t> aligned(rank, 1);
  std::copy(compactType.getShape().begin(), compactType.getShape().end(),
      aligned.end() - compactType.getRank());
  bool shared = true, batched = true;
  int64_t batches = 1, width = 1;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim = dataType.getDimSize(i), out = resultType.getDimSize(i);
    if (aligned[i] <= 0 || (aligned[i] != 1 && aligned[i] != out) ||
        (i != axis && dim != out) || (i > axis && aligned[i] != 1))
      return failure();
    if (i < axis) {
      shared &= aligned[i] == 1;
      batched &= aligned[i] == dim;
      if (dim > std::numeric_limits<int32_t>::max() / batches)
        return failure();
      batches *= dim;
    } else if (i > axis) {
      if (dim > std::numeric_limits<int32_t>::max() / width)
        return failure();
      width *= dim;
    }
  }
  if ((!shared && !batched) || aligned[axis] != resultType.getDimSize(axis) ||
      compactType.getNumElements() >= expandedType.getNumElements())
    return failure();

  Location loc = op.getLoc();
  auto reshape = [&](Value value, ArrayRef<int64_t> shape) -> Value {
    auto type = RankedTensorType::get(
        shape, cast<RankedTensorType>(value.getType()).getElementType());
    if (value.getType() == type)
      return value;
    return createTFLOperation(rewriter, loc, "tfl.reshape", TypeRange{type},
        {value, createI32ShapeConstant(rewriter, loc, shape)})
        ->getResult(0);
  };
  // The legacy FP32 rank-four operand has not yet been restored by the caller.
  if (!logicalData && dataType.getElementType().isF32() && rank == 4)
    data =
        createTFLOperation(rewriter, loc, "tfl.transpose", TypeRange{dataType},
            {data, createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2})})
            ->getResult(0);
  int64_t length = resultType.getDimSize(axis);
  SmallVector<int64_t> indexShape = shared
                                        ? SmallVector<int64_t>{length}
                                        : SmallVector<int64_t>{batches, length};
  compact = reshape(compact, indexShape);
  auto indexType = cast<RankedTensorType>(compact.getType());
  auto scalarType = RankedTensorType::get({}, indexType.getElementType());
  auto scalar = [&](int64_t value) -> Value {
    return arith::ConstantOp::create(rewriter, loc, scalarType,
        DenseIntElementsAttr::get(scalarType,
            APInt(indexType.getElementType().getIntOrFloatBitWidth(), value)));
  };
  Value zero = scalar(0), extent = scalar(dataType.getDimSize(axis));
  auto conditionType = RankedTensorType::get(indexShape, rewriter.getI1Type());
  Value negative = createTFLOperation(
      rewriter, loc, "tfl.less", TypeRange{conditionType}, {compact, zero})
                       ->getResult(0);
  Value toWrap = compact;
  if (indexType.getElementType().isSignlessInteger(32) &&
      dataType.getDimSize(axis) > std::numeric_limits<int32_t>::max() / 2) {
    // Even the unselected Add branch must not overflow for a valid positive
    // index into a very large axis. Negative indices still wrap unchanged.
    toWrap = createTFLOperation(
        rewriter, loc, "tfl.minimum", TypeRange{indexType}, {compact, zero})
                 ->getResult(0);
  }
  Value wrapped =
      createTFLOperation(rewriter, loc, "tfl.add", TypeRange{indexType},
          {toWrap, extent}, {getFusedActivationNone(rewriter)})
          ->getResult(0);
  compact = createTFLOperation(rewriter, loc, "tfl.select_v2",
      TypeRange{indexType}, {negative, wrapped, compact})
                ->getResult(0);
  RankedTensorType gatheredType = resultType;
  if (!shared) {
    data = reshape(data, {batches, dataType.getDimSize(axis), width});
    gatheredType = RankedTensorType::get(
        {batches, length, width}, dataType.getElementType());
  }
  Value result = createTFLOperation(rewriter, loc, "tfl.gather",
      TypeRange{gatheredType}, {data, compact},
      {rewriter.getNamedAttr(
           "axis", rewriter.getI32IntegerAttr(shared ? axis : 1)),
          rewriter.getNamedAttr(
              "batch_dims", rewriter.getI32IntegerAttr(shared ? 0 : 1))})
                     ->getResult(0);
  return reshape(result, resultType.getShape());
}

class GatherElementsLowering final
    : public OpConversionPattern<ONNXGatherElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGatherElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    Value indices = adaptor.getIndices();
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceIndicesType =
        dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceDataType || !sourceIndicesType || !sourceResultType ||
        !sourceDataType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() ||
        sourceDataType.getElementType() != sourceResultType.getElementType() ||
        (!sourceDataType.getElementType().isF32() &&
            !sourceDataType.getElementType().isSignlessInteger(32) &&
            !sourceDataType.getElementType().isSignlessInteger(64)))
      return op.emitError("GatherElements requires matching static FP32, i32 "
                          "or i64 data/result tensors"),
             failure();
    bool isF32 = sourceDataType.getElementType().isF32();

    int64_t rank = sourceDataType.getRank();
    if (rank < 1 || rank > 5 || sourceIndicesType.getRank() != rank ||
        sourceResultType.getRank() != rank ||
        !sourceIndicesType.hasStaticShape() ||
        !sourceIndicesType.getElementType().isIntOrIndex() ||
        sourceResultType.getShape() != sourceIndicesType.getShape()) {
      op.emitError("ONNXToTFL GatherElements requires static, equal-rank "
                   "rank-1 through rank-5 data/indices/result tensors");
      return failure();
    }

    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError() << "invalid GatherElements axis " << op.getAxis(),
             failure();

    bool hasPerValueLayout = sourceDataType.getEncoding() ||
                             sourceIndicesType.getEncoding() ||
                             sourceResultType.getEncoding();
    SmallVector<int64_t> identityLayout(rank);
    std::iota(identityLayout.begin(), identityLayout.end(), 0);
    auto logicalDataType = stripONNXToTFLLayoutEncoding(sourceDataType);
    auto logicalIndicesType = stripONNXToTFLLayoutEncoding(sourceIndicesType);
    auto logicalResultType = stripONNXToTFLLayoutEncoding(sourceResultType);
    if (hasPerValueLayout) {
      FailureOr<Value> logicalData = convertTFLValueLayout(rewriter,
          op.getLoc(), data, logicalDataType,
          getONNXToTFLPhysicalToLogicalLayout(sourceDataType), identityLayout);
      FailureOr<Value> logicalIndices = convertTFLValueLayout(rewriter,
          op.getLoc(), indices, logicalIndicesType,
          getONNXToTFLPhysicalToLogicalLayout(sourceIndicesType),
          identityLayout);
      if (failed(logicalData) || failed(logicalIndices))
        return failure();
      data = *logicalData;
      indices = *logicalIndices;
    }

    ArrayRef<int64_t> dataShape = sourceDataType.getShape();
    ArrayRef<int64_t> indicesShape = sourceIndicesType.getShape();
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dataShape[dimension] <= 0 || indicesShape[dimension] <= 0 ||
          dataShape[dimension] > std::numeric_limits<int32_t>::max() ||
          indicesShape[dimension] > std::numeric_limits<int32_t>::max() ||
          (dimension != axis &&
              indicesShape[dimension] > dataShape[dimension])) {
        op.emitError("unsupported GatherElements static dimensions");
        return failure();
      }
    }

    FailureOr<SmallVector<int64_t>> indexValues =
        getConstantIntValues(op.getIndices());
    int64_t outputElements = sourceIndicesType.getNumElements();
    if (outputElements > std::numeric_limits<int32_t>::max() / rank) {
      op.emitError("GatherElements index tensor is too large");
      return failure();
    }

    if (rank == 2 && axis == 1 && succeeded(indexValues)) {
      FailureOr<Value> windows = lowerConstantRowWindows(
          op, rewriter, data, logicalDataType, logicalResultType, *indexValues);
      if (succeeded(windows)) {
        if (hasPerValueLayout) {
          windows = convertTFLValueLayout(rewriter, op.getLoc(), *windows,
              logicalResultType, identityLayout,
              getONNXToTFLPhysicalToLogicalLayout(sourceResultType));
          if (failed(windows))
            return failure();
        }
        rewriter.replaceOp(op, *windows);
        return success();
      }
    }

    auto rows = lowerBroadcastRowGather(op, rewriter, data, indices,
        logicalDataType, logicalResultType, axis, hasPerValueLayout);
    if (succeeded(rows)) {
      auto result = convertTFLValueLayout(rewriter, op.getLoc(), *rows,
          logicalResultType, identityLayout,
          getONNXToTFLPhysicalToLogicalLayout(sourceResultType));
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }

    // A full coordinate selects one scalar. Flatten the tuple prefix rather
    // than creating rank-5/rank-6 indices for rank-4/rank-5 GatherElements;
    // reshaping the scalar results restores the original logical order.
    bool flattenCoordinates = rank >= 4;
    SmallVector<int64_t> coordinateShape(indicesShape);
    coordinateShape.push_back(rank);
    if (flattenCoordinates)
      coordinateShape = {outputElements, rank};
    auto coordinateType =
        RankedTensorType::get(coordinateShape, rewriter.getI32Type());
    Value coordinateValue;
    if (succeeded(indexValues)) {
      if (static_cast<int64_t>(indexValues->size()) != outputElements) {
        op.emitError("GatherElements constant size does not match its type");
        return failure();
      }

      // TFLite has GatherNd rather than ONNX GatherElements. Materialize one
      // complete logical coordinate per output element; all coordinates except
      // the selected axis come from the output position itself.
      SmallVector<int32_t> coordinates;
      coordinates.reserve(outputElements * rank);
      SmallVector<int64_t> position(rank);
      for (int64_t flatIndex = 0; flatIndex < outputElements; ++flatIndex) {
        int64_t remainder = flatIndex;
        for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
          position[dimension] = remainder % indicesShape[dimension];
          remainder /= indicesShape[dimension];
        }
        int64_t gathered = (*indexValues)[flatIndex];
        if (gathered < 0)
          gathered += dataShape[axis];
        if (gathered < 0 || gathered >= dataShape[axis])
          return op.emitError()
                     << "GatherElements index " << (*indexValues)[flatIndex]
                     << " is out of bounds for axis " << axis,
                 failure();
        for (int64_t dimension = 0; dimension < rank; ++dimension) {
          int64_t coordinate =
              dimension == axis ? gathered : position[dimension];
          coordinates.push_back(static_cast<int32_t>(coordinate));
        }
      }
      coordinateValue =
          arith::ConstantOp::create(rewriter, op.getLoc(), coordinateType,
              DenseIntElementsAttr::get(
                  coordinateType, ArrayRef<int32_t>(coordinates)));
    } else {
      if (!sourceIndicesType.getElementType().isSignlessInteger(32) &&
          !sourceIndicesType.getElementType().isSignlessInteger(64)) {
        op.emitError("runtime GatherElements indices must be i32 or i64");
        return failure();
      }

      // Build the final GatherNd coordinate one component at a time. The
      // selected-axis component remains runtime data; every other component
      // is a compile-time output coordinate.
      auto i32IndicesType =
          RankedTensorType::get(indicesShape, rewriter.getI32Type());
      Value runtimeIndices = indices;
      if (sourceIndicesType.getElementType().isSignlessInteger(64))
        runtimeIndices = createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
            TypeRange{i32IndicesType}, ValueRange{runtimeIndices})
                             ->getResult(0);

      // TFLite's broadcast comparison kernel dispatches through a fixed 4D
      // implementation. Flatten rank-5 runtime indices while normalizing
      // negatives. The flattened coordinate path needs no rank-5 restoration.
      Value comparisonIndices = runtimeIndices;
      auto comparisonType = i32IndicesType;
      if (rank > 4) {
        comparisonType =
            RankedTensorType::get({outputElements}, rewriter.getI32Type());
        Value flatShape =
            createI32ShapeConstant(rewriter, op.getLoc(), {outputElements});
        comparisonIndices = createTFLOperation(rewriter, op.getLoc(),
            "tfl.reshape", TypeRange{comparisonType},
            ValueRange{runtimeIndices, flatShape})
                                ->getResult(0);
      }

      Value zero = createI32ScalarTensorConstant(rewriter, op.getLoc(), 0);
      Value extent = createI32ScalarTensorConstant(
          rewriter, op.getLoc(), static_cast<int32_t>(dataShape[axis]));
      auto conditionType = RankedTensorType::get(
          comparisonType.getShape(), rewriter.getI1Type());
      Value negative = createTFLOperation(rewriter, op.getLoc(), "tfl.less",
          TypeRange{conditionType}, ValueRange{comparisonIndices, zero})
                           ->getResult(0);
      SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
      Value wrapped = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
          TypeRange{comparisonType}, ValueRange{comparisonIndices, extent},
          fusedNone)
                          ->getResult(0);
      comparisonIndices = createTFLOperation(rewriter, op.getLoc(),
          "tfl.select_v2", TypeRange{comparisonType},
          ValueRange{negative, wrapped, comparisonIndices})
                              ->getResult(0);
      runtimeIndices = comparisonIndices;

      SmallVector<int64_t> componentShape(indicesShape);
      componentShape.push_back(1);
      if (flattenCoordinates)
        componentShape = {outputElements, 1};
      auto componentType =
          RankedTensorType::get(componentShape, rewriter.getI32Type());
      Value componentShapeValue =
          createI32ShapeConstant(rewriter, op.getLoc(), componentShape);
      Value runtimeComponent = createTFLOperation(rewriter, op.getLoc(),
          "tfl.reshape", TypeRange{componentType},
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
        values.reserve(outputElements);
        for (int64_t flatIndex = 0; flatIndex < outputElements; ++flatIndex) {
          int64_t remainder = flatIndex;
          for (int64_t coordinate = rank - 1; coordinate >= 0; --coordinate) {
            position[coordinate] = remainder % indicesShape[coordinate];
            remainder /= indicesShape[coordinate];
          }
          values.push_back(static_cast<int32_t>(position[dimension]));
        }
        components.push_back(
            arith::ConstantOp::create(rewriter, op.getLoc(), componentType,
                DenseIntElementsAttr::get(
                    componentType, ArrayRef<int32_t>(values))));
      }
      SmallVector<NamedAttribute> concatenateAttributes{
          rewriter.getNamedAttr(
              "axis", rewriter.getI32IntegerAttr(coordinateShape.size() - 1)),
          getFusedActivationNone(rewriter)};
      coordinateValue =
          createTFLOperation(rewriter, op.getLoc(), "tfl.concatenation",
              TypeRange{coordinateType}, components, concatenateAttributes)
              ->getResult(0);
    }

    // The per-value path has already restored logical dimension order. The
    // legacy path retains its established rank-4 NCHW/NHWC boundary.
    if (!hasPerValueLayout && isF32 && rank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceDataType}, ValueRange{data, permutation})
                 ->getResult(0);
    }
    RankedTensorType calculationResultType =
        hasPerValueLayout ? logicalResultType : sourceResultType;
    RankedTensorType gatherResultType =
        flattenCoordinates ? RankedTensorType::get({outputElements},
                                 calculationResultType.getElementType())
                           : calculationResultType;
    Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather_nd",
        TypeRange{gatherResultType}, ValueRange{data, coordinateValue})
                         ->getResult(0);
    if (flattenCoordinates) {
      Value shape = createI32ShapeConstant(
          rewriter, op.getLoc(), calculationResultType.getShape());
      gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{calculationResultType}, ValueRange{gathered, shape})
                     ->getResult(0);
    }
    if (hasPerValueLayout) {
      FailureOr<Value> result = convertTFLValueLayout(rewriter, op.getLoc(),
          gathered, logicalResultType, identityLayout,
          getONNXToTFLPhysicalToLogicalLayout(sourceResultType));
      if (failed(result))
        return failure();
      rewriter.replaceOp(op, *result);
      return success();
    }
    if (rank != 4 || !isF32) {
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

void populateLoweringONNXGatherElementsOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<GatherElementsLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
