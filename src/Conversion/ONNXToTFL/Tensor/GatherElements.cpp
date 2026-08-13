/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class GatherElementsLowering final
    : public OpConversionPattern<ONNXGatherElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGatherElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceIndicesType =
        dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceDataType || !sourceIndicesType || !sourceResultType ||
        failed(validateStaticF32Tensor(
            op, data.getType(), "GatherElements data")) ||
        failed(validateStaticF32Tensor(
            op, sourceResultType, "GatherElements result")))
      return failure();

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

    SmallVector<int64_t> coordinateShape(indicesShape);
    coordinateShape.push_back(rank);
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
      Value runtimeIndices = adaptor.getIndices();
      if (sourceIndicesType.getElementType().isSignlessInteger(64))
        runtimeIndices = createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
            TypeRange{i32IndicesType}, ValueRange{runtimeIndices})
                             ->getResult(0);

      // TFLite's broadcast comparison kernel dispatches through a fixed 4D
      // implementation. Flatten rank-5 runtime indices while normalizing
      // negatives, then restore their static logical shape.
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
      auto conditionType =
          RankedTensorType::get(comparisonType.getShape(), rewriter.getI1Type());
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
      if (rank > 4) {
        Value logicalShape =
            createI32ShapeConstant(rewriter, op.getLoc(), indicesShape);
        runtimeIndices = createTFLOperation(rewriter, op.getLoc(),
            "tfl.reshape", TypeRange{i32IndicesType},
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
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(rank)),
          getFusedActivationNone(rewriter)};
      coordinateValue =
          createTFLOperation(rewriter, op.getLoc(), "tfl.concatenation",
              TypeRange{coordinateType}, components, concatenateAttributes)
              ->getResult(0);
    }

    // Gather in logical order. For rank 4 this means NCHW around GatherNd,
    // with the usual physical NHWC representation restored afterward.
    if (rank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceDataType}, ValueRange{data, permutation})
                 ->getResult(0);
    }
    Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather_nd",
        TypeRange{sourceResultType}, ValueRange{data, coordinateValue})
                         ->getResult(0);
    if (rank != 4) {
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
