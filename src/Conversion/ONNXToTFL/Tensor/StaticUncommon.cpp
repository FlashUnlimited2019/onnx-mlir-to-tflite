/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <cmath>

using namespace mlir;

namespace onnx_mlir {
namespace {

Value restoreLogicalRank4(ConversionPatternRewriter &rewriter, Location loc,
    Value value, RankedTensorType sourceType) {
  if (sourceType.getRank() != 4)
    return value;
  Value permutation = createI32ShapeConstant(rewriter, loc, {0, 3, 1, 2});
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{sourceType}, ValueRange{value, permutation})
      ->getResult(0);
}

Value makePhysicalRank4(ConversionPatternRewriter &rewriter, Location loc,
    Value value, RankedTensorType sourceType) {
  if (sourceType.getRank() != 4)
    return value;
  Type physicalType = convertRank4NCHWToNHWCType(sourceType);
  Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{physicalType}, ValueRange{value, permutation})
      ->getResult(0);
}

SmallVector<int64_t> integerArray(
    Operation *op, StringRef name, int64_t count, int64_t defaultValue) {
  if (auto values = op->getAttrOfType<ArrayAttr>(name)) {
    SmallVector<int64_t> result;
    for (Attribute value : values)
      result.push_back(cast<IntegerAttr>(value).getValue().getSExtValue());
    return result;
  }
  return SmallVector<int64_t>(count, defaultValue);
}

class CenterCropPadLowering final
    : public OpConversionPattern<ONNXCenterCropPadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXCenterCropPadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInputData().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutputData().getType());
    if (!inputType || !resultType || inputType.getRank() < 1 ||
        inputType.getRank() > 5 ||
        inputType.getRank() != resultType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "CenterCropPad input")) ||
        failed(validateStaticF32Tensor(op, resultType, "CenterCropPad result")))
      return failure();
    FailureOr<SmallVector<int64_t>> requested =
        getConstantIntValues(op.getShape());
    if (failed(requested))
      return op.emitError("CenterCropPad requires a constant target shape"),
             failure();
    if (auto axesAttr = op.getAxes()) {
      if (axesAttr->size() != requested->size())
        return op.emitError("CenterCropPad axes/shape sizes differ"), failure();
    } else if (requested->size() != static_cast<size_t>(inputType.getRank())) {
      return op.emitError("CenterCropPad full shape rank mismatch"), failure();
    }

    int64_t rank = inputType.getRank();
    SmallVector<int64_t> cropBegin(rank, 0);
    SmallVector<int64_t> cropSize(rank);
    SmallVector<int32_t> paddings;
    paddings.reserve(rank * 2);
    bool needsCrop = false;
    bool needsPad = false;
    for (int64_t axis = 0; axis < rank; ++axis) {
      int64_t input = inputType.getShape()[axis];
      int64_t output = resultType.getShape()[axis];
      int64_t difference = input - output;
      cropBegin[axis] = difference > 0 ? difference / 2 : 0;
      cropSize[axis] = std::min(input, output);
      int64_t padding = std::max<int64_t>(-difference, 0);
      paddings.push_back(static_cast<int32_t>(padding / 2));
      paddings.push_back(static_cast<int32_t>(padding - padding / 2));
      needsCrop |= difference > 0;
      needsPad |= difference < 0;
    }

    Location loc = op.getLoc();
    Value result =
        restoreLogicalRank4(rewriter, loc, adaptor.getInputData(), inputType);
    auto croppedType = RankedTensorType::get(cropSize, rewriter.getF32Type());
    if (needsCrop) {
      Value begin = createI32ShapeConstant(rewriter, loc, cropBegin);
      Value size = createI32ShapeConstant(rewriter, loc, cropSize);
      result = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{croppedType}, ValueRange{result, begin, size})
                   ->getResult(0);
    }
    if (needsPad) {
      auto paddingType =
          RankedTensorType::get({rank, 2}, rewriter.getI32Type());
      Value paddingValue = arith::ConstantOp::create(rewriter, loc, paddingType,
          DenseIntElementsAttr::get(paddingType, paddings));
      Value zero = createF32ScalarTensorConstant(rewriter, loc, 0.0f);
      result = createTFLOperation(rewriter, loc, "tfl.padv2",
          TypeRange{resultType}, ValueRange{result, paddingValue, zero})
                   ->getResult(0);
    }
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

class EyeLikeLowering final : public OpConversionPattern<ONNXEyeLikeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXEyeLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!resultType || resultType.getRank() != 2 ||
        failed(validateStaticF32Tensor(op, resultType, "EyeLike result")))
      return op.emitError("ONNXToTFL EyeLike requires a static rank-2 FP32 "
                          "result"),
             failure();
    int64_t rows = resultType.getShape()[0];
    int64_t columns = resultType.getShape()[1];
    int64_t diagonal = op.getK();
    SmallVector<float> values(rows * columns, 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
      int64_t column = row + diagonal;
      if (column >= 0 && column < columns)
        values[row * columns + column] = 1.0f;
    }
    rewriter.replaceOp(
        op, arith::ConstantOp::create(rewriter, op.getLoc(), resultType,
                DenseFPElementsAttr::get(resultType, values)));
    return success();
  }
};

class Col2ImLowering final : public OpConversionPattern<ONNXCol2ImOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXCol2ImOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    FailureOr<SmallVector<int64_t>> imageShape =
        getConstantIntValues(op.getImageShape());
    FailureOr<SmallVector<int64_t>> blockShape =
        getConstantIntValues(op.getBlockShape());
    if (!inputType || !resultType || inputType.getRank() != 3 ||
        resultType.getRank() < 3 || resultType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "Col2Im input")) ||
        failed(validateStaticF32Tensor(op, resultType, "Col2Im result")) ||
        failed(imageShape) || failed(blockShape) ||
        imageShape->size() != blockShape->size() ||
        resultType.getRank() != static_cast<int64_t>(imageShape->size()) + 2)
      return op.emitError("ONNXToTFL Col2Im requires static rank-1/2/3 "
                          "spatial shapes"),
             failure();
    int64_t spatialRank = imageShape->size();
    SmallVector<int64_t> strides = integerArray(op, "strides", spatialRank, 1);
    SmallVector<int64_t> dilations =
        integerArray(op, "dilations", spatialRank, 1);
    SmallVector<int64_t> pads = integerArray(op, "pads", 2 * spatialRank, 0);
    if (strides.size() != static_cast<size_t>(spatialRank) ||
        dilations.size() != static_cast<size_t>(spatialRank) ||
        pads.size() != static_cast<size_t>(2 * spatialRank) ||
        llvm::any_of(dilations, [](int64_t x) { return x != 1; }) ||
        llvm::any_of(pads, [](int64_t x) { return x != 0; }))
      return op.emitError("Col2Im static lowering requires unit dilation and "
                          "zero padding"),
             failure();
    int64_t kernelElements = 1;
    int64_t positions = 1;
    SmallVector<int64_t> positionShape;
    for (int64_t i = 0; i < spatialRank; ++i) {
      if ((*blockShape)[i] <= 0 || strides[i] != (*blockShape)[i] ||
          (*imageShape)[i] % (*blockShape)[i] != 0)
        return op.emitError("Col2Im lowering currently requires nonoverlapping "
                            "blocks that exactly tile the image"),
               failure();
      kernelElements *= (*blockShape)[i];
      positionShape.push_back((*imageShape)[i] / (*blockShape)[i]);
      positions *= positionShape.back();
    }
    int64_t batch = inputType.getShape()[0];
    if (inputType.getShape()[1] % kernelElements != 0 ||
        inputType.getShape()[2] != positions)
      return op.emitError("Col2Im input shape is inconsistent"), failure();
    int64_t channels = inputType.getShape()[1] / kernelElements;
    if (resultType.getShape()[0] != batch ||
        resultType.getShape()[1] != channels ||
        !llvm::equal(resultType.getShape().drop_front(2), *imageShape))
      return op.emitError("Col2Im result shape is inconsistent"), failure();

    SmallVector<int32_t> permutationIndices;
    permutationIndices.reserve(resultType.getNumElements());
    ArrayRef<int64_t> resultShape = resultType.getShape();
    for (int64_t linear = 0; linear < resultType.getNumElements(); ++linear) {
      int64_t remaining = linear;
      SmallVector<int64_t> outputIndex(resultType.getRank());
      for (int64_t i = resultType.getRank() - 1; i >= 0; --i) {
        outputIndex[i] = remaining % resultShape[i];
        remaining /= resultShape[i];
      }
      int64_t kernelIndex = 0;
      int64_t positionIndex = 0;
      for (int64_t i = 0; i < spatialRank; ++i) {
        int64_t kernelCoordinate = outputIndex[i + 2] % (*blockShape)[i];
        int64_t positionCoordinate = outputIndex[i + 2] / (*blockShape)[i];
        kernelIndex = kernelIndex * (*blockShape)[i] + kernelCoordinate;
        positionIndex = positionIndex * positionShape[i] + positionCoordinate;
      }
      int64_t inputChannel = outputIndex[1] * kernelElements + kernelIndex;
      int64_t inputLinear =
          (outputIndex[0] * inputType.getShape()[1] + inputChannel) *
              positions +
          positionIndex;
      permutationIndices.push_back(static_cast<int32_t>(inputLinear));
    }
    Location loc = op.getLoc();
    Value inputShape =
        createI32ShapeConstant(rewriter, loc, {inputType.getNumElements()});
    auto flatInputType = RankedTensorType::get(
        {inputType.getNumElements()}, rewriter.getF32Type());
    Value flatInput = createTFLOperation(rewriter, loc, "tfl.reshape",
        TypeRange{flatInputType}, ValueRange{adaptor.getInput(), inputShape})
                          ->getResult(0);
    auto indicesType = RankedTensorType::get(
        {resultType.getNumElements()}, rewriter.getI32Type());
    Value indices = arith::ConstantOp::create(rewriter, loc, indicesType,
        DenseIntElementsAttr::get(indicesType, permutationIndices));
    auto flatResultType = RankedTensorType::get(
        {resultType.getNumElements()}, rewriter.getF32Type());
    SmallVector<NamedAttribute> gatherAttributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
    Value gathered = createTFLOperation(rewriter, loc, "tfl.gather",
        TypeRange{flatResultType}, ValueRange{flatInput, indices},
        gatherAttributes)
                         ->getResult(0);
    Value outputShape =
        createI32ShapeConstant(rewriter, loc, resultType.getShape());
    Value result = createTFLOperation(rewriter, loc, "tfl.reshape",
        TypeRange{resultType}, ValueRange{gathered, outputShape})
                       ->getResult(0);
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

class AffineGridLowering final : public OpConversionPattern<ONNXAffineGridOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXAffineGridOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto thetaType = dyn_cast<RankedTensorType>(op.getTheta().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getGrid().getType());
    FailureOr<SmallVector<int64_t>> size = getConstantIntValues(op.getSize());
    FailureOr<SmallVector<float>> theta = getConstantF32Values(op.getTheta());
    if (!thetaType || !resultType ||
        failed(validateStaticF32Tensor(op, thetaType, "AffineGrid theta")) ||
        failed(validateStaticF32Tensor(op, resultType, "AffineGrid result")) ||
        failed(size) || failed(theta) ||
        (size->size() != 4 && size->size() != 5))
      return op.emitError("ONNXToTFL AffineGrid requires constant theta/size "
                          "and a static 2D or 3D FP32 result"),
             failure();
    int64_t spatialRank = size->size() - 2;
    int64_t batch = (*size)[0];
    if (thetaType.getRank() != 3 || thetaType.getShape()[0] != batch ||
        thetaType.getShape()[1] != spatialRank ||
        thetaType.getShape()[2] != spatialRank + 1 ||
        theta->size() != static_cast<size_t>(thetaType.getNumElements()))
      return op.emitError("AffineGrid theta shape is inconsistent"), failure();
    SmallVector<int64_t> expectedShape{batch};
    expectedShape.append(size->begin() + 2, size->end());
    expectedShape.push_back(spatialRank);
    if (!llvm::equal(expectedShape, resultType.getShape()))
      return op.emitError("AffineGrid result shape is inconsistent"), failure();

    auto coordinate = [&](int64_t index, int64_t extent) -> float {
      if (extent == 1)
        return 0.0f;
      if (op.getAlignCorners() != 0)
        return -1.0f + 2.0f * static_cast<float>(index) /
                           static_cast<float>(extent - 1);
      return -1.0f + (2.0f * static_cast<float>(index) + 1.0f) /
                         static_cast<float>(extent);
    };
    SmallVector<float> grid;
    grid.reserve(resultType.getNumElements());
    if (spatialRank == 2) {
      int64_t height = (*size)[2], width = (*size)[3];
      for (int64_t n = 0; n < batch; ++n)
        for (int64_t y = 0; y < height; ++y)
          for (int64_t x = 0; x < width; ++x) {
            float px = coordinate(x, width), py = coordinate(y, height);
            int64_t base = n * 6;
            grid.push_back((*theta)[base] * px + (*theta)[base + 1] * py +
                           (*theta)[base + 2]);
            grid.push_back((*theta)[base + 3] * px + (*theta)[base + 4] * py +
                           (*theta)[base + 5]);
          }
    } else {
      int64_t depth = (*size)[2], height = (*size)[3], width = (*size)[4];
      for (int64_t n = 0; n < batch; ++n)
        for (int64_t z = 0; z < depth; ++z)
          for (int64_t y = 0; y < height; ++y)
            for (int64_t x = 0; x < width; ++x) {
              float px = coordinate(x, width), py = coordinate(y, height);
              float pz = coordinate(z, depth);
              int64_t base = n * 12;
              for (int64_t row = 0; row < 3; ++row) {
                int64_t offset = base + row * 4;
                grid.push_back(
                    (*theta)[offset] * px + (*theta)[offset + 1] * py +
                    (*theta)[offset + 2] * pz + (*theta)[offset + 3]);
              }
            }
    }
    Value result = arith::ConstantOp::create(rewriter, op.getLoc(), resultType,
        DenseFPElementsAttr::get(resultType, grid));
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, op.getLoc(), result, resultType));
    return success();
  }
};

} // namespace

void populateLoweringONNXStaticUncommonTensorOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CenterCropPadLowering, EyeLikeLowering, Col2ImLowering,
      AffineGridLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
