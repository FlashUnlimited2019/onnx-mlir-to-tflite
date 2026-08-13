/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "llvm/ADT/DenseSet.h"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

class ScatterNDLowering final : public OpConversionPattern<ONNXScatterNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXScatterNDOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto updatesType = dyn_cast<RankedTensorType>(op.getUpdates().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!dataType || !indicesType || !updatesType || !resultType ||
        !dataType.hasStaticShape() || !indicesType.hasStaticShape() ||
        !updatesType.hasStaticShape() || !resultType.hasStaticShape() ||
        failed(validateStaticF32Tensor(
            op, adaptor.getData().getType(), "ScatterND data")) ||
        failed(validateStaticF32Tensor(
            op, adaptor.getUpdates().getType(), "ScatterND updates")) ||
        failed(validateStaticF32Tensor(op, resultType, "ScatterND result")))
      return failure();

    int64_t dataRank = dataType.getRank();
    int64_t indicesRank = indicesType.getRank();
    int64_t updatesRank = updatesType.getRank();
    if (dataRank < 1 || dataRank > 5 || indicesRank < 1 || indicesRank > 6 ||
        updatesRank < 1 || updatesRank > 5 ||
        resultType.getShape() != dataType.getShape() ||
        dataType.getElementType() != updatesType.getElementType() ||
        dataType.getElementType() != resultType.getElementType() ||
        (!indicesType.getElementType().isSignlessInteger(32) &&
            !indicesType.getElementType().isSignlessInteger(64))) {
      op.emitError("ONNXToTFL ScatterND requires static FP32 rank-1 through "
                   "rank-5 data/updates/result and i32/i64 indices");
      return failure();
    }
    StringRef reduction = op.getReduction();
    if (reduction != "none" && reduction != "add" && reduction != "mul" &&
        reduction != "max" && reduction != "min") {
      op.emitError("ONNXToTFL ScatterND supports none/add/mul/max/min "
                   "reductions for unique constant indices");
      return failure();
    }

    ArrayRef<int64_t> dataShape = dataType.getShape();
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    int64_t indexDepth = indicesShape.back();
    if (indexDepth < 1 || indexDepth > dataRank) {
      op.emitError("ScatterND index tuple depth exceeds the data rank");
      return failure();
    }
    SmallVector<int64_t> expectedUpdatesShape(
        indicesShape.begin(), indicesShape.end() - 1);
    expectedUpdatesShape.append(
        dataShape.begin() + indexDepth, dataShape.end());
    if (!llvm::equal(expectedUpdatesShape, updatesType.getShape())) {
      op.emitError("ScatterND updates shape does not match indices and data");
      return failure();
    }

    for (int64_t dimension : dataShape) {
      if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
        op.emitError("unsupported ScatterND data dimension");
        return failure();
      }
    }
    for (int64_t dimension : indicesShape) {
      if (dimension <= 0 || dimension > std::numeric_limits<int32_t>::max()) {
        op.emitError("unsupported ScatterND indices dimension");
        return failure();
      }
    }

    FailureOr<SmallVector<int64_t>> indexValues =
        getConstantIntValues(op.getIndices());
    if (failed(indexValues) || static_cast<int64_t>(indexValues->size()) !=
                                   indicesType.getNumElements()) {
      op.emitError("ONNXToTFL ScatterND requires constant indices");
      return failure();
    }

    int64_t tupleCount = indicesType.getNumElements() / indexDepth;
    SmallVector<int32_t> normalizedIndices;
    normalizedIndices.reserve(indicesType.getNumElements());
    llvm::DenseSet<int64_t> coordinateKeys;
    for (int64_t tuple = 0; tuple < tupleCount; ++tuple) {
      int64_t key = 0;
      for (int64_t component = 0; component < indexDepth; ++component) {
        int64_t index = (*indexValues)[tuple * indexDepth + component];
        if (index < 0)
          index += dataShape[component];
        if (index < 0 || index >= dataShape[component]) {
          op.emitError() << "ScatterND index "
                         << (*indexValues)[tuple * indexDepth + component]
                         << " is out of bounds for axis " << component;
          return failure();
        }
        if (key > (std::numeric_limits<int64_t>::max() - index) /
                      dataShape[component]) {
          op.emitError("ScatterND coordinate space is too large");
          return failure();
        }
        key = key * dataShape[component] + index;
        normalizedIndices.push_back(static_cast<int32_t>(index));
      }
      if (!coordinateKeys.insert(key).second) {
        op.emitError("ONNXToTFL ScatterND static reduction requires unique "
                     "constant index tuples");
        return failure();
      }
    }

    bool flattenCoordinates = indicesRank > 5;
    SmallVector<int64_t> coordinateShape(indicesShape);
    if (flattenCoordinates)
      coordinateShape = {tupleCount, indexDepth};
    auto coordinateType =
        RankedTensorType::get(coordinateShape, rewriter.getI32Type());
    Value coordinates =
        arith::ConstantOp::create(rewriter, op.getLoc(), coordinateType,
            DenseIntElementsAttr::get(
                coordinateType, ArrayRef<int32_t>(normalizedIndices)));

    auto restoreLogicalRank4 = [&](Value value,
                                   RankedTensorType sourceType) -> Value {
      if (sourceType.getRank() != 4)
        return value;
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      return createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{value, permutation})
          ->getResult(0);
    };

    Value data = restoreLogicalRank4(adaptor.getData(), dataType);
    Value updates = restoreLogicalRank4(adaptor.getUpdates(), updatesType);
    auto calculationUpdatesType = updatesType;
    if (flattenCoordinates) {
      SmallVector<int64_t> calculationUpdatesShape{tupleCount};
      calculationUpdatesShape.append(
          dataShape.begin() + indexDepth, dataShape.end());
      calculationUpdatesType = RankedTensorType::get(
          calculationUpdatesShape, updatesType.getElementType());
      Value calculationUpdatesShapeValue = createI32ShapeConstant(
          rewriter, op.getLoc(), calculationUpdatesShape);
      updates = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{calculationUpdatesType},
          ValueRange{updates, calculationUpdatesShapeValue})
                    ->getResult(0);
    }
    Value oldValues = createTFLOperation(rewriter, op.getLoc(), "tfl.gather_nd",
        TypeRange{calculationUpdatesType}, ValueRange{data, coordinates})
                          ->getResult(0);
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Value replacements = updates;
    if (reduction != "none") {
      StringRef operation = reduction == "add"   ? "tfl.add"
                            : reduction == "mul" ? "tfl.mul"
                            : reduction == "max" ? "tfl.maximum"
                                                 : "tfl.minimum";
      ArrayRef<NamedAttribute> attributes =
          (reduction == "add" || reduction == "mul")
              ? ArrayRef<NamedAttribute>(fusedNone)
              : ArrayRef<NamedAttribute>{};
      replacements = createTFLOperation(rewriter, op.getLoc(), operation,
          TypeRange{calculationUpdatesType}, ValueRange{oldValues, updates},
          attributes)
                         ->getResult(0);
    }
    Value deltas = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
        TypeRange{calculationUpdatesType},
        ValueRange{replacements, oldValues}, fusedNone)
                       ->getResult(0);
    Value shape = createI32ShapeConstant(rewriter, op.getLoc(), dataShape);
    Value sparseDeltas =
        createTFLOperation(rewriter, op.getLoc(), "tfl.scatter_nd",
            TypeRange{dataType}, ValueRange{coordinates, deltas, shape})
            ->getResult(0);
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
        TypeRange{resultType}, ValueRange{data, sparseDeltas}, fusedNone)
                       ->getResult(0);

    if (resultType.getRank() == 4) {
      Type physicalResultType = convertRank4NCHWToNHWCType(resultType);
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, permutation})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXScatterNDOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ScatterNDLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
