/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class GatherNDLowering final : public OpConversionPattern<ONNXGatherNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGatherNDOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    auto sourceDataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceIndicesType =
        dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceDataType || !sourceIndicesType || !sourceResultType ||
        failed(validateStaticF32Tensor(op, data.getType(), "GatherND data")) ||
        failed(
            validateStaticF32Tensor(op, sourceResultType, "GatherND result")) ||
        !sourceIndicesType.hasStaticShape() ||
        (!sourceIndicesType.getElementType().isSignlessInteger(32) &&
            !sourceIndicesType.getElementType().isSignlessInteger(64))) {
      op.emitError("ONNXToTFL GatherND requires static FP32 data/result and "
                   "static i32/i64 indices");
      return failure();
    }

    int64_t dataRank = sourceDataType.getRank();
    int64_t indicesRank = sourceIndicesType.getRank();
    int64_t resultRank = sourceResultType.getRank();
    int64_t batchDims = op.getBatchDims();
    if (dataRank < 1 || dataRank > 5 || indicesRank < 1 || indicesRank > 5 ||
        resultRank < 1 || resultRank > 5 || batchDims < 0 ||
        batchDims > indicesRank - 1 || batchDims >= dataRank) {
      op.emitError("ONNXToTFL GatherND supports ranks 1 through 5 and valid "
                   "nonnegative batch_dims");
      return failure();
    }

    ArrayRef<int64_t> dataShape = sourceDataType.getShape();
    ArrayRef<int64_t> indicesShape = sourceIndicesType.getShape();
    for (int64_t dimension : dataShape) {
      if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
        op.emitError("unsupported GatherND data dimension");
        return failure();
      }
    }
    for (int64_t dimension : indicesShape) {
      if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
        op.emitError("unsupported GatherND indices dimension");
        return failure();
      }
    }
    for (int64_t dimension = 0; dimension < batchDims; ++dimension) {
      if (indicesShape[dimension] != dataShape[dimension]) {
        op.emitError("GatherND batch dimensions must match data dimensions");
        return failure();
      }
    }

    int64_t indexDepth = indicesShape.back();
    if (indexDepth <= 0 || indexDepth > dataRank - batchDims) {
      op.emitError("GatherND index tuple depth exceeds the data rank");
      return failure();
    }
    SmallVector<int64_t> expectedResultShape(
        indicesShape.begin(), indicesShape.end() - 1);
    expectedResultShape.append(
        dataShape.begin() + batchDims + indexDepth, dataShape.end());
    if (!llvm::equal(expectedResultShape, sourceResultType.getShape())) {
      op.emitError("GatherND result shape does not match data and indices");
      return failure();
    }

    int64_t tupleCount = sourceIndicesType.getNumElements() / indexDepth;
    int64_t coordinateDepth = batchDims + indexDepth;
    if (tupleCount > std::numeric_limits<int32_t>::max() / coordinateDepth) {
      op.emitError("GatherND coordinate tensor is too large");
      return failure();
    }

    // TFL GatherNd has no batch_dims attribute. Expand every ONNX tuple into
    // a full coordinate by prefixing its statically known batch position.
    int64_t prefixRank = indicesRank - 1;
    SmallVector<int64_t> position(prefixRank);
    SmallVector<int64_t> coordinateShape(
        indicesShape.begin(), indicesShape.end() - 1);
    coordinateShape.push_back(coordinateDepth);
    auto coordinateType =
        RankedTensorType::get(coordinateShape, rewriter.getI32Type());
    Value coordinateValue;
    FailureOr<SmallVector<int64_t>> indexValues =
        getConstantIntValues(op.getIndices());
    if (succeeded(indexValues) && static_cast<int64_t>(indexValues->size()) ==
                                      sourceIndicesType.getNumElements()) {
      SmallVector<int32_t> coordinates;
      coordinates.reserve(tupleCount * coordinateDepth);
      for (int64_t tuple = 0; tuple < tupleCount; ++tuple) {
        int64_t remainder = tuple;
        for (int64_t dimension = prefixRank - 1; dimension >= 0; --dimension) {
          position[dimension] = remainder % indicesShape[dimension];
          remainder /= indicesShape[dimension];
        }
        for (int64_t dimension = 0; dimension < batchDims; ++dimension)
          coordinates.push_back(static_cast<int32_t>(position[dimension]));
        for (int64_t component = 0; component < indexDepth; ++component) {
          int64_t axis = batchDims + component;
          int64_t index = (*indexValues)[tuple * indexDepth + component];
          if (index < 0)
            index += dataShape[axis];
          if (index < 0 || index >= dataShape[axis]) {
            op.emitError() << "GatherND index "
                           << (*indexValues)[tuple * indexDepth + component]
                           << " is out of bounds for axis " << axis;
            return failure();
          }
          coordinates.push_back(static_cast<int32_t>(index));
        }
      }
      coordinateValue =
          arith::ConstantOp::create(rewriter, op.getLoc(), coordinateType,
              DenseIntElementsAttr::get(
                  coordinateType, ArrayRef<int32_t>(coordinates)));
    } else {
      auto runtimeIndexType =
          RankedTensorType::get(indicesShape, rewriter.getI32Type());
      Value runtimeIndices = adaptor.getIndices();
      auto workingIndexType = runtimeIndexType;
      bool flattenRuntimeIndices = indicesRank > 4;
      if (flattenRuntimeIndices) {
        auto flatSourceType =
            RankedTensorType::get({sourceIndicesType.getNumElements()},
                sourceIndicesType.getElementType());
        Value flatShape = createI32ShapeConstant(
            rewriter, op.getLoc(), flatSourceType.getShape());
        runtimeIndices = createTFLOperation(rewriter, op.getLoc(),
            "tfl.reshape", TypeRange{flatSourceType},
            ValueRange{runtimeIndices, flatShape})
                             ->getResult(0);
        workingIndexType = RankedTensorType::get(
            {sourceIndicesType.getNumElements()}, rewriter.getI32Type());
      }
      if (sourceIndicesType.getElementType().isSignlessInteger(64))
        runtimeIndices = createTFLOperation(rewriter, op.getLoc(), "tfl.cast",
            TypeRange{workingIndexType}, ValueRange{runtimeIndices})
                             ->getResult(0);

      // ONNX permits negative coordinates while TFLite GatherNd does not.
      // Normalize each runtime tuple component against its corresponding data
      // dimension before adding the implicit batch prefix.
      Value zero = createI32ScalarTensorConstant(rewriter, op.getLoc(), 0);
      auto conditionType = RankedTensorType::get(
          workingIndexType.getShape(), rewriter.getI1Type());
      Value isNegative = createTFLOperation(rewriter, op.getLoc(), "tfl.less",
          TypeRange{conditionType}, ValueRange{runtimeIndices, zero})
                             ->getResult(0);
      SmallVector<int32_t> dimensionSizes;
      dimensionSizes.reserve(indexDepth);
      for (int64_t component = 0; component < indexDepth; ++component)
        dimensionSizes.push_back(
            static_cast<int32_t>(dataShape[batchDims + component]));
      auto dimensionType =
          RankedTensorType::get({indexDepth}, rewriter.getI32Type());
      Value dimensions =
          arith::ConstantOp::create(rewriter, op.getLoc(), dimensionType,
              DenseIntElementsAttr::get(
                  dimensionType, ArrayRef<int32_t>(dimensionSizes)));
      SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
      Value wrapped = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
          TypeRange{workingIndexType}, ValueRange{runtimeIndices, dimensions},
          fusedNone)
                          ->getResult(0);
      Value normalized = createTFLOperation(rewriter, op.getLoc(),
          "tfl.select_v2", TypeRange{workingIndexType},
          ValueRange{isNegative, wrapped, runtimeIndices})
                             ->getResult(0);
      if (flattenRuntimeIndices) {
        Value restoredShape =
            createI32ShapeConstant(rewriter, op.getLoc(), indicesShape);
        normalized = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{runtimeIndexType}, ValueRange{normalized, restoredShape})
                         ->getResult(0);
      }
      if (batchDims == 0) {
        coordinateValue = normalized;
      } else {
        SmallVector<int64_t> prefixShape(
            indicesShape.begin(), indicesShape.end() - 1);
        prefixShape.push_back(batchDims);
        auto prefixType =
            RankedTensorType::get(prefixShape, rewriter.getI32Type());
        SmallVector<int32_t> prefixes;
        prefixes.reserve(tupleCount * batchDims);
        for (int64_t tuple = 0; tuple < tupleCount; ++tuple) {
          int64_t remainder = tuple;
          for (int64_t dimension = prefixRank - 1; dimension >= 0;
              --dimension) {
            position[dimension] = remainder % indicesShape[dimension];
            remainder /= indicesShape[dimension];
          }
          for (int64_t dimension = 0; dimension < batchDims; ++dimension)
            prefixes.push_back(static_cast<int32_t>(position[dimension]));
        }
        Value prefix = arith::ConstantOp::create(rewriter, op.getLoc(),
            prefixType,
            DenseIntElementsAttr::get(prefixType, ArrayRef<int32_t>(prefixes)));
        SmallVector<NamedAttribute> attributes{
            rewriter.getNamedAttr(
                "axis", rewriter.getI32IntegerAttr(indicesRank - 1)),
            getFusedActivationNone(rewriter)};
        coordinateValue = createTFLOperation(rewriter, op.getLoc(),
            "tfl.concatenation", TypeRange{coordinateType},
            ValueRange{prefix, normalized}, attributes)
                              ->getResult(0);
      }
    }

    // Gather against logical ONNX dimensions. Rank-4 FP32 values are physical
    // NHWC in the TFL graph, so restore data to NCHW independently of whether
    // the result itself is rank 4.
    if (dataRank == 4) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      data = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceDataType}, ValueRange{data, permutation})
                 ->getResult(0);
    }
    Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather_nd",
        TypeRange{sourceResultType}, ValueRange{data, coordinateValue})
                         ->getResult(0);
    if (resultRank != 4) {
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

void populateLoweringONNXGatherNDOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<GatherNDLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
