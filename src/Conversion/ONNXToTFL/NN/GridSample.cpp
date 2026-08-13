/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class GridSampleLowering final : public OpConversionPattern<ONNXGridSampleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGridSampleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto gridType = dyn_cast<RankedTensorType>(op.getGrid().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !gridType || !resultType || inputType.getRank() != 4 ||
        gridType.getRank() != 4 || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "GridSample input")) ||
        failed(validateStaticF32Tensor(op, gridType, "GridSample grid")) ||
        failed(validateStaticF32Tensor(op, resultType, "GridSample result")))
      return op.emitError("ONNXToTFL GridSample requires static rank-4 FP32 "
                          "input, grid, and result tensors"),
             failure();

    StringRef mode = op.getMode();
    if (mode != "nearest" && mode != "linear" && mode != "bilinear" &&
        mode != "cubic")
      return op.emitError("ONNXToTFL GridSample supports nearest, bilinear, "
                          "and cubic mode"),
             failure();
    StringRef paddingMode = op.getPaddingMode();
    if (paddingMode != "zeros" && paddingMode != "border" &&
        paddingMode != "reflection")
      return op.emitError("ONNXToTFL GridSample supports zeros, border, and "
                          "reflection padding"),
             failure();
    if (op.getAlignCorners() != 0 && op.getAlignCorners() != 1)
      return op.emitError("GridSample align_corners must be 0 or 1"), failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> gridShape = gridType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    int64_t batch = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t inputHeight = inputShape[2];
    int64_t inputWidth = inputShape[3];
    int64_t outputHeight = gridShape[1];
    int64_t outputWidth = gridShape[2];
    if (batch <= 0 || channels <= 0 || inputHeight <= 0 || inputWidth <= 0 ||
        outputHeight <= 0 || outputWidth <= 0 || gridShape[0] != batch ||
        gridShape[3] != 2 || resultShape[0] != batch ||
        resultShape[1] != channels || resultShape[2] != outputHeight ||
        resultShape[3] != outputWidth)
      return op.emitError("GridSample static shapes are inconsistent"),
             failure();

    Location loc = op.getLoc();
    Type f32 = rewriter.getF32Type();
    Type i32 = rewriter.getI32Type();
    auto componentType =
        RankedTensorType::get({batch, outputHeight, outputWidth, 1}, f32);
    auto indexType =
        RankedTensorType::get({batch, outputHeight, outputWidth, 1}, i32);
    auto conditionType = RankedTensorType::get(
        {batch, outputHeight, outputWidth, 1}, rewriter.getI1Type());
    auto coordinateType =
        RankedTensorType::get({batch, outputHeight, outputWidth, 3}, i32);
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};

    auto binary = [&](StringRef name, Type type, Value lhs, Value rhs,
                      bool fused = true) -> Value {
      ArrayRef<NamedAttribute> attributes =
          fused ? fusedNone : ArrayRef<NamedAttribute>{};
      return createTFLOperation(rewriter, loc, name, TypeRange{type},
          ValueRange{lhs, rhs}, attributes)
          ->getResult(0);
    };
    auto f32Scalar = [&](float value) {
      return createF32ScalarTensorConstant(rewriter, loc, value);
    };
    auto i32Scalar = [&](int32_t value) {
      return createI32ScalarTensorConstant(rewriter, loc, value);
    };

    // Every rank-4 FP32 value enters this bridge in physical NHWC form. Grid
    // is logically NHW2 rather than NCHW, so undo that generic permutation
    // before interpreting its final coordinate dimension.
    Value gridPermutation = createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2});
    Value logicalGrid = createTFLOperation(rewriter, loc, "tfl.transpose",
        TypeRange{gridType}, ValueRange{adaptor.getGrid(), gridPermutation})
                            ->getResult(0);
    auto sliceComponent = [&](int32_t component) -> Value {
      Value begin = createI32ShapeConstant(rewriter, loc, {0, 0, 0, component});
      Value size = createI32ShapeConstant(
          rewriter, loc, {batch, outputHeight, outputWidth, 1});
      return createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{componentType}, ValueRange{logicalGrid, begin, size})
          ->getResult(0);
    };

    auto denormalize = [&](Value coordinate, int64_t extent) -> Value {
      Value shifted =
          binary("tfl.add", componentType, coordinate, f32Scalar(1.0f));
      float scale = op.getAlignCorners() != 0
                        ? static_cast<float>(extent - 1) * 0.5f
                        : static_cast<float>(extent) * 0.5f;
      Value scaled =
          binary("tfl.mul", componentType, shifted, f32Scalar(scale));
      if (op.getAlignCorners() != 0)
        return scaled;
      return binary("tfl.add", componentType, scaled, f32Scalar(-0.5f));
    };
    Value x = denormalize(sliceComponent(0), inputWidth);
    Value y = denormalize(sliceComponent(1), inputHeight);

    auto clampToPixels = [&](Value coordinate, int64_t extent) -> Value {
      Value clamped = binary("tfl.maximum", componentType, coordinate,
          f32Scalar(0.0f), /*fused=*/false);
      return binary("tfl.minimum", componentType, clamped,
          f32Scalar(static_cast<float>(extent - 1)), /*fused=*/false);
    };
    auto reflectCoordinate = [&](Value coordinate, int64_t extent) -> Value {
      if (extent == 1)
        return binary("tfl.mul", componentType, coordinate, f32Scalar(0.0f));
      float low = op.getAlignCorners() != 0 ? 0.0f : -0.5f;
      float high = op.getAlignCorners() != 0
                       ? static_cast<float>(extent - 1)
                       : static_cast<float>(extent) - 0.5f;
      float span = high - low;
      Value shifted =
          binary("tfl.sub", componentType, coordinate, f32Scalar(low));
      Value distance = createTFLOperation(rewriter, loc, "tfl.abs",
          TypeRange{componentType}, ValueRange{shifted})
                           ->getResult(0);
      Value quotient =
          binary("tfl.div", componentType, distance, f32Scalar(span));
      quotient = createTFLOperation(rewriter, loc, "tfl.floor",
          TypeRange{componentType}, ValueRange{quotient})
                     ->getResult(0);
      Value covered =
          binary("tfl.mul", componentType, quotient, f32Scalar(span));
      Value remainder = binary("tfl.sub", componentType, distance, covered);
      Value quotientI32 = createTFLOperation(
          rewriter, loc, "tfl.cast", TypeRange{indexType}, ValueRange{quotient})
                              ->getResult(0);
      Value parity = binary("tfl.floor_mod", indexType, quotientI32,
          i32Scalar(2), /*fused=*/false);
      Value even = createTFLOperation(rewriter, loc, "tfl.equal",
          TypeRange{conditionType}, ValueRange{parity, i32Scalar(0)})
                       ->getResult(0);
      Value fromLow =
          binary("tfl.add", componentType, remainder, f32Scalar(low));
      Value fromHigh =
          binary("tfl.sub", componentType, f32Scalar(high), remainder);
      return createTFLOperation(rewriter, loc, "tfl.select_v2",
          TypeRange{componentType}, ValueRange{even, fromLow, fromHigh})
          ->getResult(0);
    };
    if (mode != "cubic") {
      if (paddingMode == "border") {
        x = clampToPixels(x, inputWidth);
        y = clampToPixels(y, inputHeight);
      } else if (paddingMode == "reflection") {
        x = clampToPixels(reflectCoordinate(x, inputWidth), inputWidth);
        y = clampToPixels(reflectCoordinate(y, inputHeight), inputHeight);
      }
    }

    SmallVector<int32_t> batchValues;
    batchValues.reserve(batch);
    for (int32_t index = 0; index < batch; ++index)
      batchValues.push_back(index);
    auto batchVectorType = RankedTensorType::get({batch}, i32);
    Value batchVector =
        arith::ConstantOp::create(rewriter, loc, batchVectorType,
            DenseIntElementsAttr::get(batchVectorType, batchValues));
    auto batchSingletonType = RankedTensorType::get({batch, 1, 1, 1}, i32);
    Value batchSingletonShape =
        createI32ShapeConstant(rewriter, loc, {batch, 1, 1, 1});
    Value batchSingleton = createTFLOperation(rewriter, loc, "tfl.reshape",
        TypeRange{batchSingletonType},
        ValueRange{batchVector, batchSingletonShape})
                               ->getResult(0);
    Value batchBroadcastShape = createI32ShapeConstant(
        rewriter, loc, {batch, outputHeight, outputWidth, 1});
    Value batchCoordinates = createTFLOperation(rewriter, loc,
        "tfl.broadcast_to", TypeRange{indexType},
        ValueRange{batchSingleton, batchBroadcastShape})
                                 ->getResult(0);

    auto gather = [&](Value yIndex, Value xIndex,
                      bool discretePadding) -> Value {
      bool maskOutOfBounds = true;
      if (discretePadding && paddingMode == "border") {
        xIndex = binary("tfl.maximum", indexType, xIndex, i32Scalar(0),
            /*fused=*/false);
        xIndex = binary("tfl.minimum", indexType, xIndex,
            i32Scalar(inputWidth - 1), /*fused=*/false);
        yIndex = binary("tfl.maximum", indexType, yIndex, i32Scalar(0),
            /*fused=*/false);
        yIndex = binary("tfl.minimum", indexType, yIndex,
            i32Scalar(inputHeight - 1), /*fused=*/false);
        maskOutOfBounds = false;
      } else if (discretePadding && paddingMode == "reflection") {
        Value xFloat = createTFLOperation(rewriter, loc, "tfl.cast",
            TypeRange{componentType}, ValueRange{xIndex})
                           ->getResult(0);
        Value yFloat = createTFLOperation(rewriter, loc, "tfl.cast",
            TypeRange{componentType}, ValueRange{yIndex})
                           ->getResult(0);
        xIndex =
            createTFLOperation(rewriter, loc, "tfl.cast", TypeRange{indexType},
                ValueRange{clampToPixels(
                    reflectCoordinate(xFloat, inputWidth), inputWidth)})
                ->getResult(0);
        yIndex =
            createTFLOperation(rewriter, loc, "tfl.cast", TypeRange{indexType},
                ValueRange{clampToPixels(
                    reflectCoordinate(yFloat, inputHeight), inputHeight)})
                ->getResult(0);
        maskOutOfBounds = false;
      }
      Value zero = i32Scalar(0);
      Value xInLower = createTFLOperation(rewriter, loc, "tfl.greater_equal",
          TypeRange{conditionType}, ValueRange{xIndex, zero})
                           ->getResult(0);
      Value xInUpper = createTFLOperation(rewriter, loc, "tfl.less",
          TypeRange{conditionType}, ValueRange{xIndex, i32Scalar(inputWidth)})
                           ->getResult(0);
      Value yInLower = createTFLOperation(rewriter, loc, "tfl.greater_equal",
          TypeRange{conditionType}, ValueRange{yIndex, zero})
                           ->getResult(0);
      Value yInUpper = createTFLOperation(rewriter, loc, "tfl.less",
          TypeRange{conditionType}, ValueRange{yIndex, i32Scalar(inputHeight)})
                           ->getResult(0);
      Value xInBounds = createTFLOperation(rewriter, loc, "tfl.logical_and",
          TypeRange{conditionType}, ValueRange{xInLower, xInUpper})
                            ->getResult(0);
      Value yInBounds = createTFLOperation(rewriter, loc, "tfl.logical_and",
          TypeRange{conditionType}, ValueRange{yInLower, yInUpper})
                            ->getResult(0);
      Value inBounds = createTFLOperation(rewriter, loc, "tfl.logical_and",
          TypeRange{conditionType}, ValueRange{xInBounds, yInBounds})
                           ->getResult(0);

      Value safeX =
          binary("tfl.maximum", indexType, xIndex, zero, /*fused=*/false);
      safeX = binary("tfl.minimum", indexType, safeX, i32Scalar(inputWidth - 1),
          /*fused=*/false);
      Value safeY =
          binary("tfl.maximum", indexType, yIndex, zero, /*fused=*/false);
      safeY = binary("tfl.minimum", indexType, safeY,
          i32Scalar(inputHeight - 1), /*fused=*/false);
      SmallVector<NamedAttribute> concatAttributes{
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(3)),
          getFusedActivationNone(rewriter)};
      Value coordinates = createTFLOperation(rewriter, loc, "tfl.concatenation",
          TypeRange{coordinateType}, ValueRange{batchCoordinates, safeY, safeX},
          concatAttributes)
                              ->getResult(0);
      Value sampled = createTFLOperation(rewriter, loc, "tfl.gather_nd",
          TypeRange{physicalResultType},
          ValueRange{adaptor.getX(), coordinates})
                          ->getResult(0);
      if (!maskOutOfBounds)
        return sampled;
      Value mask = createTFLOperation(rewriter, loc, "tfl.cast",
          TypeRange{componentType}, ValueRange{inBounds})
                       ->getResult(0);
      return binary("tfl.mul", physicalResultType, sampled, mask);
    };

    auto toI32 = [&](Value value) -> Value {
      return createTFLOperation(
          rewriter, loc, "tfl.cast", TypeRange{indexType}, ValueRange{value})
          ->getResult(0);
    };
    if (mode == "cubic") {
      Value xFloor = createTFLOperation(
          rewriter, loc, "tfl.floor", TypeRange{componentType}, ValueRange{x})
                         ->getResult(0);
      Value yFloor = createTFLOperation(
          rewriter, loc, "tfl.floor", TypeRange{componentType}, ValueRange{y})
                         ->getResult(0);
      Value xFraction = binary("tfl.sub", componentType, x, xFloor);
      Value yFraction = binary("tfl.sub", componentType, y, yFloor);

      auto cubicOne = [&](Value value) -> Value {
        Value square = binary("tfl.mul", componentType, value, value);
        Value inner = binary("tfl.mul", componentType, value, f32Scalar(1.25f));
        inner = binary("tfl.sub", componentType, inner, f32Scalar(2.25f));
        return binary("tfl.add", componentType,
            binary("tfl.mul", componentType, inner, square), f32Scalar(1.0f));
      };
      auto cubicTwo = [&](Value value) -> Value {
        Value result =
            binary("tfl.mul", componentType, value, f32Scalar(-0.75f));
        result = binary("tfl.add", componentType, result, f32Scalar(3.75f));
        result = binary("tfl.mul", componentType, result, value);
        result = binary("tfl.add", componentType, result, f32Scalar(-6.0f));
        result = binary("tfl.mul", componentType, result, value);
        return binary("tfl.add", componentType, result, f32Scalar(3.0f));
      };
      auto cubicWeights = [&](Value fraction) {
        SmallVector<Value> weights;
        weights.push_back(cubicTwo(
            binary("tfl.add", componentType, fraction, f32Scalar(1.0f))));
        weights.push_back(cubicOne(fraction));
        Value oneMinus =
            binary("tfl.sub", componentType, f32Scalar(1.0f), fraction);
        weights.push_back(cubicOne(oneMinus));
        weights.push_back(cubicTwo(
            binary("tfl.sub", componentType, f32Scalar(2.0f), fraction)));
        return weights;
      };
      SmallVector<Value> xWeights = cubicWeights(xFraction);
      SmallVector<Value> yWeights = cubicWeights(yFraction);
      Value xBase = toI32(xFloor);
      Value yBase = toI32(yFloor);
      auto offsetIndex = [&](Value base, int32_t offset) -> Value {
        if (offset == 0)
          return base;
        return binary("tfl.add", indexType, base, i32Scalar(offset));
      };

      SmallVector<Value> rows;
      for (int32_t row = 0; row < 4; ++row) {
        Value yIndex = offsetIndex(yBase, row - 1);
        Value rowValue;
        for (int32_t column = 0; column < 4; ++column) {
          Value sample = gather(yIndex, offsetIndex(xBase, column - 1),
              /*discretePadding=*/true);
          Value weighted =
              binary("tfl.mul", physicalResultType, sample, xWeights[column]);
          rowValue = column == 0 ? weighted
                                 : binary("tfl.add", physicalResultType,
                                       rowValue, weighted);
        }
        rows.push_back(rowValue);
      }
      Value result;
      for (int32_t row = 0; row < 4; ++row) {
        Value weighted =
            binary("tfl.mul", physicalResultType, rows[row], yWeights[row]);
        result = row == 0
                     ? weighted
                     : binary("tfl.add", physicalResultType, result, weighted);
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    if (mode == "nearest") {
      Value roundedX = createTFLOperation(
          rewriter, loc, "tfl.round", TypeRange{componentType}, ValueRange{x})
                           ->getResult(0);
      Value roundedY = createTFLOperation(
          rewriter, loc, "tfl.round", TypeRange{componentType}, ValueRange{y})
                           ->getResult(0);
      rewriter.replaceOp(op, gather(toI32(roundedY), toI32(roundedX), false));
      return success();
    }

    Value xFloor = createTFLOperation(
        rewriter, loc, "tfl.floor", TypeRange{componentType}, ValueRange{x})
                       ->getResult(0);
    Value yFloor = createTFLOperation(
        rewriter, loc, "tfl.floor", TypeRange{componentType}, ValueRange{y})
                       ->getResult(0);
    Value x0 = toI32(xFloor);
    Value y0 = toI32(yFloor);
    Value x1 = binary("tfl.add", indexType, x0, i32Scalar(1));
    Value y1 = binary("tfl.add", indexType, y0, i32Scalar(1));
    Value xWeight = binary("tfl.sub", componentType, x, xFloor);
    Value yWeight = binary("tfl.sub", componentType, y, yFloor);

    Value topLeft = gather(y0, x0, false);
    Value topRight = gather(y0, x1, false);
    Value bottomLeft = gather(y1, x0, false);
    Value bottomRight = gather(y1, x1, false);
    Value topDelta = binary("tfl.sub", physicalResultType, topRight, topLeft);
    Value top = binary("tfl.add", physicalResultType, topLeft,
        binary("tfl.mul", physicalResultType, topDelta, xWeight));
    Value bottomDelta =
        binary("tfl.sub", physicalResultType, bottomRight, bottomLeft);
    Value bottom = binary("tfl.add", physicalResultType, bottomLeft,
        binary("tfl.mul", physicalResultType, bottomDelta, xWeight));
    Value verticalDelta = binary("tfl.sub", physicalResultType, bottom, top);
    Value result = binary("tfl.add", physicalResultType, top,
        binary("tfl.mul", physicalResultType, verticalDelta, yWeight));
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXGridSampleOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<GridSampleLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
