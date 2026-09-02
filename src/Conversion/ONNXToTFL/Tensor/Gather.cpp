/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

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
        (dataElementType.isF32() || dataElementType.isInteger(1));
    if (!sourceDataType || !sourceIndicesType || !sourceResultType ||
        !sourceDataType.hasStaticShape() ||
        !sourceResultType.hasStaticShape() || sourceDataType.getRank() < 1 ||
        !supportedDataType ||
        sourceResultType.getElementType() != dataElementType ||
        !sourceIndicesType.hasStaticShape() ||
        (!sourceIndicesType.getElementType().isSignlessInteger(32) &&
            !sourceIndicesType.getElementType().isSignlessInteger(64))) {
      op.emitError("ONNXToTFL Gather requires matching static FP32 or bool "
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
