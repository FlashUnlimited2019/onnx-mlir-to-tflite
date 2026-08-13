/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <cmath>

using namespace mlir;

namespace onnx_mlir {
namespace {

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

Value staticWeightedGather(ConversionPatternRewriter &rewriter, Location loc,
    Value input, ArrayRef<int32_t> coordinates, int64_t coordinateRank,
    ArrayRef<float> coefficients, int64_t outputElements,
    int64_t termsPerOutput, RankedTensorType resultType,
    bool reduceMaximum = false,
    ArrayRef<float> bias = {}) {
  int64_t terms = coefficients.size();
  auto coordinateType =
      RankedTensorType::get({terms, coordinateRank}, rewriter.getI32Type());
  Value coordinateValue = arith::ConstantOp::create(rewriter, loc,
      coordinateType, DenseIntElementsAttr::get(coordinateType, coordinates));
  auto termsType = RankedTensorType::get({terms}, rewriter.getF32Type());
  Value gathered = createTFLOperation(rewriter, loc, "tfl.gather_nd",
      TypeRange{termsType}, ValueRange{input, coordinateValue})
                       ->getResult(0);
  Value coefficientValue = arith::ConstantOp::create(rewriter, loc, termsType,
      DenseFPElementsAttr::get(termsType, coefficients));
  SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
  Value weighted = createTFLOperation(rewriter, loc, "tfl.mul",
      TypeRange{termsType}, ValueRange{gathered, coefficientValue}, fusedNone)
                       ->getResult(0);
  auto matrixType = RankedTensorType::get(
      {outputElements, termsPerOutput}, rewriter.getF32Type());
  Value matrixShape = createI32ShapeConstant(
      rewriter, loc, {outputElements, termsPerOutput});
  Value matrix = createTFLOperation(rewriter, loc, "tfl.reshape",
      TypeRange{matrixType}, ValueRange{weighted, matrixShape})
                     ->getResult(0);
  auto vectorType =
      RankedTensorType::get({outputElements}, rewriter.getF32Type());
  Value reductionAxis = createI32ShapeConstant(rewriter, loc, {1});
  SmallVector<NamedAttribute> reductionAttributes{
      rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(false))};
  Value result = createTFLOperation(rewriter, loc,
      reduceMaximum ? "tfl.reduce_max" : "tfl.sum",
      TypeRange{vectorType}, ValueRange{matrix, reductionAxis},
      reductionAttributes)
                     ->getResult(0);
  if (!bias.empty()) {
    Value biasValue = arith::ConstantOp::create(rewriter, loc, vectorType,
        DenseFPElementsAttr::get(vectorType, bias));
    result = createTFLOperation(rewriter, loc, "tfl.add",
        TypeRange{vectorType}, ValueRange{result, biasValue}, fusedNone)
                 ->getResult(0);
  }
  Value resultShape =
      createI32ShapeConstant(rewriter, loc, resultType.getShape());
  return createTFLOperation(rewriter, loc, "tfl.reshape",
      TypeRange{resultType}, ValueRange{result, resultShape})
      ->getResult(0);
}

class DeformConvLowering final
    : public OpConversionPattern<ONNXDeformConvOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXDeformConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto weightType = dyn_cast<RankedTensorType>(op.getW().getType());
    auto offsetType = dyn_cast<RankedTensorType>(op.getOffset().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    FailureOr<SmallVector<float>> weights = getConstantF32Values(op.getW());
    FailureOr<SmallVector<float>> offsets =
        getConstantF32Values(op.getOffset());
    if (!inputType || !weightType || !offsetType || !resultType ||
        inputType.getRank() != 4 || weightType.getRank() != 4 ||
        offsetType.getRank() != 4 || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "DeformConv input")) ||
        failed(validateStaticF32Tensor(op, weightType, "DeformConv weight")) ||
        failed(validateStaticF32Tensor(op, offsetType, "DeformConv offset")) ||
        failed(validateStaticF32Tensor(op, resultType, "DeformConv result")) ||
        failed(weights) || failed(offsets) || op.getGroup() != 1 ||
        op.getOffsetGroup() != 1)
      return op.emitError("ONNXToTFL DeformConv requires static rank-4 FP32 "
                          "tensors, constant weights/offsets, and group="
                          "offset_group=1"),
             failure();

    ArrayRef<int64_t> xShape = inputType.getShape();
    ArrayRef<int64_t> wShape = weightType.getShape();
    ArrayRef<int64_t> oShape = offsetType.getShape();
    ArrayRef<int64_t> yShape = resultType.getShape();
    int64_t batch = xShape[0], inputChannels = xShape[1];
    int64_t inputHeight = xShape[2], inputWidth = xShape[3];
    int64_t outputChannels = wShape[0], kernelHeight = wShape[2];
    int64_t kernelWidth = wShape[3], outputHeight = yShape[2];
    int64_t outputWidth = yShape[3];
    SmallVector<int64_t> kernel =
        integerArray(op, "kernel_shape", 0, 0);
    if (kernel.empty())
      kernel = {kernelHeight, kernelWidth};
    SmallVector<int64_t> strides = integerArray(op, "strides", 2, 1);
    SmallVector<int64_t> dilations = integerArray(op, "dilations", 2, 1);
    SmallVector<int64_t> pads = integerArray(op, "pads", 4, 0);
    int64_t kernelElements = kernelHeight * kernelWidth;
    if (kernel != SmallVector<int64_t>({kernelHeight, kernelWidth}) ||
        strides.size() != 2 || dilations.size() != 2 || pads.size() != 4 ||
        xShape[0] != yShape[0] || inputChannels != wShape[1] ||
        outputChannels != yShape[1] || oShape[0] != batch ||
        oShape[1] != kernelElements * 2 || oShape[2] != outputHeight ||
        oShape[3] != outputWidth)
      return op.emitError("DeformConv static shapes/attributes are "
                          "inconsistent"),
             failure();

    SmallVector<float> bias(outputChannels, 0.0f);
    if (!isa<NoneType>(op.getB().getType())) {
      FailureOr<SmallVector<float>> values = getConstantF32Values(op.getB());
      if (failed(values) || values->size() != static_cast<size_t>(outputChannels))
        return op.emitError("DeformConv bias must be constant"), failure();
      bias = *values;
    }
    SmallVector<float> mask(batch * kernelElements * outputHeight * outputWidth,
        1.0f);
    if (!isa<NoneType>(op.getMask().getType())) {
      auto maskType = dyn_cast<RankedTensorType>(op.getMask().getType());
      FailureOr<SmallVector<float>> values =
          getConstantF32Values(op.getMask());
      if (!maskType || maskType.getShape() !=
                           ArrayRef<int64_t>({batch, kernelElements,
                               outputHeight, outputWidth}) ||
          failed(values))
        return op.emitError("DeformConv mask must be a shape-consistent "
                            "constant"),
               failure();
      mask = *values;
    }

    int64_t outputElements = resultType.getNumElements();
    int64_t termsPerOutput = inputChannels * kernelElements * 4;
    SmallVector<int32_t> coordinates;
    SmallVector<float> coefficients;
    SmallVector<float> outputBias;
    coordinates.reserve(outputElements * termsPerOutput * 4);
    coefficients.reserve(outputElements * termsPerOutput);
    outputBias.reserve(outputElements);
    auto offsetAt = [&](int64_t n, int64_t channel, int64_t y, int64_t x) {
      return (*offsets)[((n * oShape[1] + channel) * outputHeight + y) *
                            outputWidth +
                        x];
    };
    auto maskAt = [&](int64_t n, int64_t kernelIndex, int64_t y, int64_t x) {
      return mask[((n * kernelElements + kernelIndex) * outputHeight + y) *
                      outputWidth +
                  x];
    };
    for (int64_t n = 0; n < batch; ++n)
      for (int64_t oc = 0; oc < outputChannels; ++oc)
        for (int64_t oy = 0; oy < outputHeight; ++oy)
          for (int64_t ox = 0; ox < outputWidth; ++ox) {
            outputBias.push_back(bias[oc]);
            for (int64_t ic = 0; ic < inputChannels; ++ic)
              for (int64_t ky = 0; ky < kernelHeight; ++ky)
                for (int64_t kx = 0; kx < kernelWidth; ++kx) {
                  int64_t k = ky * kernelWidth + kx;
                  // ONNX packs offsets as [kH, kW, spatial-axis].
                  float py = oy * strides[0] - pads[0] +
                             ky * dilations[0] + offsetAt(n, k * 2, oy, ox);
                  float px = ox * strides[1] - pads[1] +
                             kx * dilations[1] +
                             offsetAt(n, k * 2 + 1, oy, ox);
                  int64_t y0 = static_cast<int64_t>(std::floor(py));
                  int64_t x0 = static_cast<int64_t>(std::floor(px));
                  float dy = py - static_cast<float>(y0);
                  float dx = px - static_cast<float>(x0);
                  int64_t weightIndex =
                      ((oc * inputChannels + ic) * kernelHeight + ky) *
                          kernelWidth +
                      kx;
                  float base = (*weights)[weightIndex] *
                               maskAt(n, k, oy, ox);
                  int64_t ys[4] = {y0, y0, y0 + 1, y0 + 1};
                  int64_t xs[4] = {x0, x0 + 1, x0, x0 + 1};
                  float interpolation[4] = {(1.0f - dy) * (1.0f - dx),
                      (1.0f - dy) * dx, dy * (1.0f - dx), dy * dx};
                  for (int64_t neighbor = 0; neighbor < 4; ++neighbor) {
                    bool valid = ys[neighbor] >= 0 &&
                                 ys[neighbor] < inputHeight &&
                                 xs[neighbor] >= 0 && xs[neighbor] < inputWidth;
                    coordinates.append(
                        {static_cast<int32_t>(valid ? n : 0),
                            static_cast<int32_t>(valid ? ic : 0),
                            static_cast<int32_t>(valid ? ys[neighbor] : 0),
                            static_cast<int32_t>(valid ? xs[neighbor] : 0)});
                    coefficients.push_back(
                        valid ? base * interpolation[neighbor] : 0.0f);
                  }
                }
          }
    Location loc = op.getLoc();
    Value input = restoreLogicalRank4(rewriter, loc, adaptor.getX(), inputType);
    Value result = staticWeightedGather(rewriter, loc, input, coordinates, 4,
        coefficients, outputElements, termsPerOutput, resultType,
        /*reduceMaximum=*/false, outputBias);
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

class GridSampleRank5Lowering final
    : public OpConversionPattern<ONNXGridSampleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGridSampleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto gridType = dyn_cast<RankedTensorType>(op.getGrid().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !gridType || !resultType || inputType.getRank() != 5 ||
        gridType.getRank() != 5 || resultType.getRank() != 5)
      return failure();
    // AffineGrid may already have been rewritten to an arith.constant when
    // dialect conversion revisits this consumer, so inspect the adapted value.
    FailureOr<SmallVector<float>> grid =
        getConstantF32Values(adaptor.getGrid());
    if (failed(validateStaticF32Tensor(op, inputType, "GridSample input")) ||
        failed(validateStaticF32Tensor(op, gridType, "GridSample grid")) ||
        failed(validateStaticF32Tensor(op, resultType, "GridSample result")) ||
        failed(grid) || op.getMode() != "nearest" ||
        op.getPaddingMode() != "border")
      return op.emitError("rank-5 GridSample requires static FP32 tensors, a "
                          "constant grid, nearest mode, and border padding"),
             failure();
    ArrayRef<int64_t> x = inputType.getShape();
    ArrayRef<int64_t> g = gridType.getShape();
    ArrayRef<int64_t> y = resultType.getShape();
    if (g[0] != x[0] || g[4] != 3 || y[0] != x[0] || y[1] != x[1] ||
        y[2] != g[1] || y[3] != g[2] || y[4] != g[3])
      return op.emitError("rank-5 GridSample shapes are inconsistent"),
             failure();

    auto denormalize = [&](float value, int64_t extent) {
      return op.getAlignCorners() != 0
                 ? (value + 1.0f) * static_cast<float>(extent - 1) * 0.5f
                 : ((value + 1.0f) * static_cast<float>(extent) - 1.0f) *
                       0.5f;
    };
    SmallVector<int32_t> coordinates;
    coordinates.reserve(resultType.getNumElements() * 5);
    for (int64_t n = 0; n < y[0]; ++n)
      for (int64_t c = 0; c < y[1]; ++c)
        for (int64_t oz = 0; oz < y[2]; ++oz)
          for (int64_t oy = 0; oy < y[3]; ++oy)
            for (int64_t ox = 0; ox < y[4]; ++ox) {
              int64_t gridBase = (((n * g[1] + oz) * g[2] + oy) * g[3] + ox) * 3;
              int64_t ix = static_cast<int64_t>(
                  std::nearbyint(denormalize((*grid)[gridBase], x[4])));
              int64_t iy = static_cast<int64_t>(
                  std::nearbyint(denormalize((*grid)[gridBase + 1], x[3])));
              int64_t iz = static_cast<int64_t>(
                  std::nearbyint(denormalize((*grid)[gridBase + 2], x[2])));
              ix = std::clamp<int64_t>(ix, 0, x[4] - 1);
              iy = std::clamp<int64_t>(iy, 0, x[3] - 1);
              iz = std::clamp<int64_t>(iz, 0, x[2] - 1);
              coordinates.append({static_cast<int32_t>(n),
                  static_cast<int32_t>(c), static_cast<int32_t>(iz),
                  static_cast<int32_t>(iy), static_cast<int32_t>(ix)});
            }
    int64_t elements = resultType.getNumElements();
    auto coordinateType =
        RankedTensorType::get({elements, 5}, rewriter.getI32Type());
    Value indices = arith::ConstantOp::create(rewriter, op.getLoc(),
        coordinateType,
        DenseIntElementsAttr::get(coordinateType, coordinates));
    auto flatType =
        RankedTensorType::get({elements}, rewriter.getF32Type());
    Value gathered = createTFLOperation(rewriter, op.getLoc(), "tfl.gather_nd",
        TypeRange{flatType}, ValueRange{adaptor.getX(), indices})
                         ->getResult(0);
    Value shape =
        createI32ShapeConstant(rewriter, op.getLoc(), resultType.getShape());
    rewriter.replaceOp(op,
        createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{resultType}, ValueRange{gathered, shape})
            ->getResult(0));
    return success();
  }
};

class RoiAlignLowering final : public OpConversionPattern<ONNXRoiAlignOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXRoiAlignOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto roiType = dyn_cast<RankedTensorType>(op.getRois().getType());
    auto batchType =
        dyn_cast<RankedTensorType>(op.getBatchIndices().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    FailureOr<SmallVector<float>> rois = getConstantF32Values(op.getRois());
    FailureOr<SmallVector<int64_t>> batchIndices =
        getConstantIntValues(op.getBatchIndices());
    if (!inputType || !roiType || !batchType || !resultType ||
        inputType.getRank() != 4 || roiType.getRank() != 2 ||
        batchType.getRank() != 1 || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, inputType, "RoiAlign input")) ||
        failed(validateStaticF32Tensor(op, roiType, "RoiAlign rois")) ||
        failed(validateStaticF32Tensor(op, resultType, "RoiAlign result")) ||
        failed(rois) || failed(batchIndices) ||
        (op.getMode() != "avg" && op.getMode() != "max") ||
        op.getSamplingRatio() <= 0 ||
        (op.getCoordinateTransformationMode() != "half_pixel" &&
            op.getCoordinateTransformationMode() != "output_half_pixel"))
      return op.emitError("ONNXToTFL RoiAlign requires static rank-4 FP32 "
                          "input/result, constant ROIs/batches, avg/max mode, "
                          "and a fixed "
                          "positive sampling ratio"),
             failure();
    ArrayRef<int64_t> x = inputType.getShape();
    ArrayRef<int64_t> y = resultType.getShape();
    int64_t numRois = roiType.getShape()[0];
    int64_t channels = x[1], height = x[2], width = x[3];
    int64_t pooledHeight = op.getOutputHeight();
    int64_t pooledWidth = op.getOutputWidth();
    int64_t ratio = op.getSamplingRatio();
    if (roiType.getShape()[1] != 4 || batchType.getShape()[0] != numRois ||
        rois->size() != static_cast<size_t>(numRois * 4) ||
        batchIndices->size() != static_cast<size_t>(numRois) ||
        y != ArrayRef<int64_t>({numRois, channels, pooledHeight, pooledWidth}))
      return op.emitError("RoiAlign shapes are inconsistent"), failure();

    int64_t outputElements = resultType.getNumElements();
    int64_t termsPerOutput = ratio * ratio * 4;
    SmallVector<int32_t> coordinates;
    SmallVector<float> coefficients;
    coordinates.reserve(outputElements * termsPerOutput * 4);
    coefficients.reserve(outputElements * termsPerOutput);
    float scale = op.getSpatialScale().convertToFloat();
    float offset = op.getCoordinateTransformationMode() == "half_pixel"
                       ? 0.5f
                       : 0.0f;
    auto addNeighbor = [&](int64_t batch, int64_t channel, int64_t py,
                           int64_t px, float coefficient) {
      coordinates.append({static_cast<int32_t>(batch),
          static_cast<int32_t>(channel), static_cast<int32_t>(py),
          static_cast<int32_t>(px)});
      coefficients.push_back(op.getMode() == "avg"
                                 ? coefficient /
                                       static_cast<float>(ratio * ratio)
                                 : coefficient);
    };
    for (int64_t roi = 0; roi < numRois; ++roi) {
      int64_t batch = (*batchIndices)[roi];
      if (batch < 0 || batch >= x[0])
        return op.emitError("RoiAlign batch index is out of range"), failure();
      float startX = (*rois)[roi * 4] * scale - offset;
      float startY = (*rois)[roi * 4 + 1] * scale - offset;
      float endX = (*rois)[roi * 4 + 2] * scale - offset;
      float endY = (*rois)[roi * 4 + 3] * scale - offset;
      float binHeight = (endY - startY) / static_cast<float>(pooledHeight);
      float binWidth = (endX - startX) / static_cast<float>(pooledWidth);
      for (int64_t channel = 0; channel < channels; ++channel)
        for (int64_t ph = 0; ph < pooledHeight; ++ph)
          for (int64_t pw = 0; pw < pooledWidth; ++pw)
            for (int64_t iy = 0; iy < ratio; ++iy)
              for (int64_t ix = 0; ix < ratio; ++ix) {
                float sampleY = startY + ph * binHeight +
                                (iy + 0.5f) * binHeight /
                                    static_cast<float>(ratio);
                float sampleX = startX + pw * binWidth +
                                (ix + 0.5f) * binWidth /
                                    static_cast<float>(ratio);
                if (sampleY < -1.0f || sampleY > height || sampleX < -1.0f ||
                    sampleX > width) {
                  for (int64_t neighbor = 0; neighbor < 4; ++neighbor)
                    addNeighbor(0, 0, 0, 0, 0.0f);
                  continue;
                }
                sampleY = std::max(sampleY, 0.0f);
                sampleX = std::max(sampleX, 0.0f);
                int64_t y0 = static_cast<int64_t>(sampleY);
                int64_t x0 = static_cast<int64_t>(sampleX);
                int64_t y1, x1;
                if (y0 >= height - 1) {
                  y0 = y1 = height - 1;
                  sampleY = static_cast<float>(y0);
                } else {
                  y1 = y0 + 1;
                }
                if (x0 >= width - 1) {
                  x0 = x1 = width - 1;
                  sampleX = static_cast<float>(x0);
                } else {
                  x1 = x0 + 1;
                }
                float ly = sampleY - y0, lx = sampleX - x0;
                addNeighbor(batch, channel, y0, x0, (1.0f - ly) * (1.0f - lx));
                addNeighbor(batch, channel, y0, x1, (1.0f - ly) * lx);
                addNeighbor(batch, channel, y1, x0, ly * (1.0f - lx));
                addNeighbor(batch, channel, y1, x1, ly * lx);
              }
    }
    Location loc = op.getLoc();
    Value input = restoreLogicalRank4(rewriter, loc, adaptor.getX(), inputType);
    Value result = staticWeightedGather(rewriter, loc, input, coordinates, 4,
        coefficients, outputElements, termsPerOutput, resultType,
        /*reduceMaximum=*/op.getMode() == "max");
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

} // namespace

void populateLoweringONNXStaticSamplingOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<DeformConvLowering, GridSampleRank5Lowering, RoiAlignLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
