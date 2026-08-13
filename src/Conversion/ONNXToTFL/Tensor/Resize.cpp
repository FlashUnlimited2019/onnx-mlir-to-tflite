/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

using namespace mlir;

namespace onnx_mlir {
namespace {

struct AxisInterpolation {
  SmallVector<float> weights;
  SmallVector<float> bias;
  bool isIdentity = false;
};

bool isAbsentResizeInput(Value value) {
  if (isa<NoneType>(value.getType()))
    return true;
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  return shapedType && shapedType.hasStaticShape() &&
         shapedType.getNumElements() == 0;
}

double cubicKernel(double value, double coefficient) {
  value = std::abs(value);
  double square = value * value;
  double cube = square * value;
  if (value <= 1.0)
    return (coefficient + 2.0) * cube - (coefficient + 3.0) * square + 1.0;
  if (value < 2.0)
    return coefficient * cube - 5.0 * coefficient * square +
           8.0 * coefficient * value - 4.0 * coefficient;
  return 0.0;
}

LogicalResult normalizeCoefficients(
    Operation *op, SmallVectorImpl<double> &coefficients) {
  double sum = std::accumulate(coefficients.begin(), coefficients.end(), 0.0);
  if (!std::isfinite(sum) || std::abs(sum) < 1.0e-15)
    return op->emitError("Resize interpolation coefficients have zero sum");
  for (double &coefficient : coefficients)
    coefficient /= sum;
  return success();
}

FailureOr<SmallVector<double>> computeInterpolationCoefficients(Operation *op,
    StringRef mode, StringRef nearestMode, double ratio, double scale,
    bool antialias, double cubicCoefficient) {
  SmallVector<double> coefficients;
  if (mode == "nearest") {
    if (antialias)
      return op->emitError("nearest Resize does not support antialias"),
             failure();
    coefficients.assign({0.0, 0.0});
    if (ratio == std::floor(ratio)) {
      coefficients[1] = 1.0;
    } else if (nearestMode == "round_prefer_floor") {
      coefficients[ratio <= 0.5 ? 0 : 1] = 1.0;
    } else if (nearestMode == "round_prefer_ceil") {
      coefficients[ratio < 0.5 ? 0 : 1] = 1.0;
    } else if (nearestMode == "floor") {
      coefficients[0] = 1.0;
    } else if (nearestMode == "ceil") {
      coefficients[1] = 1.0;
    } else {
      return op->emitError("unsupported nearest Resize rounding mode"),
             failure();
    }
    return coefficients;
  }

  if (mode == "linear") {
    if (!antialias)
      return SmallVector<double>{1.0 - ratio, ratio};
    double filterScale = std::min(scale, 1.0);
    int64_t start = static_cast<int64_t>(std::floor(-1.0 / filterScale)) + 1;
    int64_t footprint = 2 - 2 * start;
    coefficients.reserve(footprint);
    for (int64_t index = start; index < start + footprint; ++index) {
      double argument = (static_cast<double>(index) - ratio) * filterScale;
      coefficients.push_back(std::max(0.0, 1.0 - std::abs(argument)));
    }
    if (failed(normalizeCoefficients(op, coefficients)))
      return failure();
    return coefficients;
  }

  if (mode == "cubic") {
    if (!antialias) {
      return SmallVector<double>{cubicKernel(ratio + 1.0, cubicCoefficient),
          cubicKernel(ratio, cubicCoefficient),
          cubicKernel(1.0 - ratio, cubicCoefficient),
          cubicKernel(2.0 - ratio, cubicCoefficient)};
    }
    double filterScale = std::min(scale, 1.0);
    int64_t start = static_cast<int64_t>(std::floor(-2.0 / filterScale)) + 1;
    int64_t end = 2 - start;
    coefficients.reserve(end - start);
    for (int64_t index = start; index < end; ++index) {
      double argument = filterScale * (static_cast<double>(index) - ratio);
      coefficients.push_back(cubicKernel(argument, cubicCoefficient));
    }
    if (failed(normalizeCoefficients(op, coefficients)))
      return failure();
    return coefficients;
  }

  return op->emitError("Resize supports nearest, linear, or cubic mode"),
         failure();
}

FailureOr<AxisInterpolation> buildAxisInterpolation(Operation *op,
    int64_t inputExtent, int64_t outputExtent, double scale, StringRef mode,
    StringRef nearestMode, StringRef coordinateMode, bool antialias,
    bool excludeOutside, double cubicCoefficient, double roiStart,
    double roiEnd, double extrapolationValue) {
  if (inputExtent <= 0 || outputExtent <= 0 || !std::isfinite(scale) ||
      scale <= 0.0)
    return op->emitError("Resize requires positive static extents and scale"),
           failure();

  AxisInterpolation interpolation;
  interpolation.weights.assign(inputExtent * outputExtent, 0.0f);
  interpolation.bias.assign(outputExtent, 0.0f);
  double outputWidth = scale * static_cast<double>(inputExtent);

  for (int64_t outputIndex = 0; outputIndex < outputExtent; ++outputIndex) {
    double coordinate;
    bool extrapolated = false;
    if (coordinateMode == "align_corners") {
      coordinate = outputWidth == 1.0
                       ? 0.0
                       : static_cast<double>(outputIndex) *
                             static_cast<double>(inputExtent - 1) /
                             (outputWidth - 1.0);
    } else if (coordinateMode == "asymmetric") {
      coordinate = static_cast<double>(outputIndex) / scale;
    } else if (coordinateMode == "tf_crop_and_resize") {
      coordinate =
          outputWidth == 1.0
              ? (roiEnd - roiStart) * static_cast<double>(inputExtent - 1) / 2.0
              : static_cast<double>(outputIndex) * (roiEnd - roiStart) *
                    static_cast<double>(inputExtent - 1) / (outputWidth - 1.0);
      coordinate += roiStart * static_cast<double>(inputExtent - 1);
      extrapolated = coordinate < 0.0 || coordinate > inputExtent - 1;
    } else if (coordinateMode == "pytorch_half_pixel") {
      coordinate = outputWidth == 1.0
                       ? -0.5
                       : (static_cast<double>(outputIndex) + 0.5) / scale - 0.5;
    } else if (coordinateMode == "half_pixel") {
      coordinate = (static_cast<double>(outputIndex) + 0.5) / scale - 0.5;
    } else if (coordinateMode == "half_pixel_symmetric") {
      double adjustment = static_cast<double>(outputExtent) / outputWidth;
      double center = static_cast<double>(inputExtent) / 2.0;
      double offset = center * (1.0 - adjustment);
      coordinate =
          offset + (static_cast<double>(outputIndex) + 0.5) / scale - 0.5;
    } else {
      return op->emitError("unsupported Resize coordinate transformation"),
             failure();
    }

    if (extrapolated) {
      interpolation.bias[outputIndex] = static_cast<float>(extrapolationValue);
      continue;
    }

    double coordinateFloor = std::floor(coordinate);
    double ratio =
        coordinate == coordinateFloor ? 1.0 : coordinate - coordinateFloor;
    FailureOr<SmallVector<double>> coefficients =
        computeInterpolationCoefficients(
            op, mode, nearestMode, ratio, scale, antialias, cubicCoefficient);
    if (failed(coefficients))
      return failure();

    int64_t count = static_cast<int64_t>(coefficients->size());
    int64_t padding = (count + 1) / 2;
    double paddedCoordinate = coordinate + static_cast<double>(padding);
    int64_t paddedFloor = static_cast<int64_t>(std::floor(paddedCoordinate));
    double fraction = paddedCoordinate - static_cast<double>(paddedFloor);
    int64_t offset;
    if (count % 2 == 0) {
      offset = fraction == 0.0 ? -(count / 2) : -(count / 2) + 1;
    } else {
      int64_t base = -((count - 1) / 2);
      offset = fraction <= 0.5 ? base : base + 1;
    }
    int64_t firstIndex = paddedFloor + offset - padding;

    if (excludeOutside) {
      for (int64_t index = 0; index < count; ++index) {
        int64_t sourceIndex = firstIndex + index;
        if (sourceIndex < 0 || sourceIndex >= inputExtent)
          (*coefficients)[index] = 0.0;
      }
      if (failed(normalizeCoefficients(op, *coefficients)))
        return failure();
    }

    for (int64_t index = 0; index < count; ++index) {
      int64_t sourceIndex =
          std::clamp(firstIndex + index, int64_t{0}, inputExtent - 1);
      interpolation.weights[sourceIndex * outputExtent + outputIndex] +=
          static_cast<float>((*coefficients)[index]);
    }
  }

  interpolation.isIdentity = inputExtent == outputExtent;
  if (interpolation.isIdentity) {
    for (int64_t outputIndex = 0; outputIndex < outputExtent; ++outputIndex) {
      if (std::abs(interpolation.bias[outputIndex]) > 1.0e-7f) {
        interpolation.isIdentity = false;
        break;
      }
      for (int64_t inputIndex = 0; inputIndex < inputExtent; ++inputIndex) {
        float expected = inputIndex == outputIndex ? 1.0f : 0.0f;
        if (std::abs(
                interpolation.weights[inputIndex * outputExtent + outputIndex] -
                expected) > 1.0e-7f) {
          interpolation.isIdentity = false;
          break;
        }
      }
      if (!interpolation.isIdentity)
        break;
    }
  }
  return interpolation;
}

Value applyAxisInterpolation(Operation *op, Value input,
    SmallVectorImpl<int64_t> &currentShape, int64_t axis,
    const AxisInterpolation &interpolation, int64_t outputExtent,
    ConversionPatternRewriter &rewriter) {
  if (interpolation.isIdentity)
    return input;

  Location loc = op->getLoc();
  int64_t rank = static_cast<int64_t>(currentShape.size());
  SmallVector<int64_t> permutation;
  permutation.reserve(rank);
  for (int64_t dimension = 0; dimension < rank; ++dimension)
    if (dimension != axis)
      permutation.push_back(dimension);
  permutation.push_back(axis);

  SmallVector<int64_t> permutedShape;
  permutedShape.reserve(rank);
  for (int64_t dimension : permutation)
    permutedShape.push_back(currentShape[dimension]);
  Value permuted = input;
  if (axis != rank - 1) {
    auto permutedType =
        RankedTensorType::get(permutedShape, rewriter.getF32Type());
    Value permutationValue = createI32ShapeConstant(rewriter, loc, permutation);
    permuted = createTFLOperation(rewriter, loc, "tfl.transpose",
        TypeRange{permutedType}, ValueRange{input, permutationValue})
                   ->getResult(0);
  }

  int64_t inputExtent = currentShape[axis];
  int64_t outerSize = 1;
  for (int64_t dimension = 0; dimension < rank - 1; ++dimension)
    outerSize *= permutedShape[dimension];
  auto flattenedType =
      RankedTensorType::get({outerSize, inputExtent}, rewriter.getF32Type());
  Value flattenedShape =
      createI32ShapeConstant(rewriter, loc, {outerSize, inputExtent});
  Value flattened = createTFLOperation(rewriter, loc, "tfl.reshape",
      TypeRange{flattenedType}, ValueRange{permuted, flattenedShape})
                        ->getResult(0);

  auto weightsType =
      RankedTensorType::get({inputExtent, outputExtent}, rewriter.getF32Type());
  Value weights = arith::ConstantOp::create(rewriter, loc, weightsType,
      DenseFPElementsAttr::get(weightsType, interpolation.weights));
  auto multipliedType =
      RankedTensorType::get({outerSize, outputExtent}, rewriter.getF32Type());
  SmallVector<NamedAttribute> matmulAttributes{
      rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(false))};
  Value multiplied = createTFLOperation(rewriter, loc, "tfl.batch_matmul",
      TypeRange{multipliedType}, ValueRange{flattened, weights},
      matmulAttributes)
                         ->getResult(0);

  bool hasBias = llvm::any_of(
      interpolation.bias, [](float value) { return value != 0.0f; });
  if (hasBias) {
    auto biasType =
        RankedTensorType::get({outputExtent}, rewriter.getF32Type());
    Value bias = arith::ConstantOp::create(rewriter, loc, biasType,
        DenseFPElementsAttr::get(biasType, interpolation.bias));
    SmallVector<NamedAttribute> addAttributes{getFusedActivationNone(rewriter)};
    multiplied = createTFLOperation(rewriter, loc, "tfl.add",
        TypeRange{multipliedType}, ValueRange{multiplied, bias}, addAttributes)
                     ->getResult(0);
  }

  permutedShape.back() = outputExtent;
  auto restoredPermutedType =
      RankedTensorType::get(permutedShape, rewriter.getF32Type());
  Value restoredPermutedShape =
      createI32ShapeConstant(rewriter, loc, permutedShape);
  Value result = createTFLOperation(rewriter, loc, "tfl.reshape",
      TypeRange{restoredPermutedType},
      ValueRange{multiplied, restoredPermutedShape})
                     ->getResult(0);

  currentShape[axis] = outputExtent;
  if (axis == rank - 1)
    return result;

  SmallVector<int64_t> inversePermutation(rank);
  for (int64_t destination = 0; destination < rank; ++destination)
    inversePermutation[permutation[destination]] = destination;
  auto restoredType =
      RankedTensorType::get(currentShape, rewriter.getF32Type());
  Value inverseValue =
      createI32ShapeConstant(rewriter, loc, inversePermutation);
  return createTFLOperation(rewriter, loc, "tfl.transpose",
      TypeRange{restoredType}, ValueRange{result, inverseValue})
      ->getResult(0);
}

std::optional<Value> tryFastResize(ONNXResizeOp op,
    ONNXResizeOp::Adaptor adaptor, RankedTensorType sourceType,
    RankedTensorType sourceResultType, ConversionPatternRewriter &rewriter) {
  if (adaptor.getAntialias() != 0 || adaptor.getAxes().has_value() ||
      adaptor.getExcludeOutside() != 0 ||
      adaptor.getCoordinateTransformationMode() == "tf_crop_and_resize" ||
      !isAbsentResizeInput(op.getRoi()))
    return std::nullopt;

  ArrayRef<int64_t> inputShape = sourceType.getShape();
  ArrayRef<int64_t> outputShape = sourceResultType.getShape();
  if (!isAbsentResizeInput(op.getScales())) {
    FailureOr<SmallVector<float>> scales = getConstantF32Values(op.getScales());
    if (failed(scales) || scales->size() != inputShape.size())
      return std::nullopt;
    for (auto [axis, scale] : llvm::enumerate(*scales)) {
      double inferredScale = static_cast<double>(outputShape[axis]) /
                             static_cast<double>(inputShape[axis]);
      if (static_cast<double>(scale) != inferredScale)
        return std::nullopt;
    }
  }
  StringRef mode = adaptor.getMode();
  StringRef coordinateMode = adaptor.getCoordinateTransformationMode();
  StringRef nearestMode = adaptor.getNearestMode();
  if (sourceType.getRank() != 4) {
    bool halfPixel = coordinateMode == "half_pixel";
    bool asymmetric = coordinateMode == "asymmetric";
    bool roundPreferFloor = nearestMode == "round_prefer_floor";
    bool floorMode = nearestMode == "floor";
    if (mode != "nearest" ||
        !((halfPixel && roundPreferFloor) ||
            (asymmetric && (floorMode || roundPreferFloor))))
      return std::nullopt;

    Value result = adaptor.getX();
    SmallVector<int64_t> currentShape(inputShape);
    for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
      int64_t inputExtent = inputShape[axis];
      int64_t outputExtent = outputShape[axis];
      if (inputExtent == outputExtent)
        continue;
      double scale =
          static_cast<double>(outputExtent) / static_cast<double>(inputExtent);
      SmallVector<int32_t> indices;
      indices.reserve(outputExtent);
      for (int64_t outputIndex = 0; outputIndex < outputExtent; ++outputIndex) {
        double coordinate =
            halfPixel ? (static_cast<double>(outputIndex) + 0.5) / scale - 0.5
                      : static_cast<double>(outputIndex) / scale;
        int64_t index = roundPreferFloor
                            ? static_cast<int64_t>(std::ceil(coordinate - 0.5))
                            : static_cast<int64_t>(std::floor(coordinate));
        index = std::clamp(index, int64_t{0}, inputExtent - 1);
        indices.push_back(static_cast<int32_t>(index));
      }
      auto indicesType =
          RankedTensorType::get({outputExtent}, rewriter.getI32Type());
      Value indicesValue = arith::ConstantOp::create(rewriter, op.getLoc(),
          indicesType, DenseIntElementsAttr::get(indicesType, indices));
      currentShape[axis] = outputExtent;
      auto nextType =
          RankedTensorType::get(currentShape, rewriter.getF32Type());
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
          rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.gather",
          TypeRange{nextType}, ValueRange{result, indicesValue}, attributes)
                   ->getResult(0);
    }
    return result;
  }

  if (inputShape[0] != outputShape[0] || inputShape[1] != outputShape[1])
    return std::nullopt;

  if (mode == "nearest" &&
      (coordinateMode == "half_pixel" || coordinateMode == "asymmetric") &&
      nearestMode == "round_prefer_floor") {
    bool halfPixel = coordinateMode == "half_pixel";
    Value toLogical =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
        TypeRange{sourceType}, ValueRange{adaptor.getX(), toLogical})
                       ->getResult(0);
    SmallVector<int64_t> currentShape(inputShape);
    for (int64_t axis : {int64_t{2}, int64_t{3}}) {
      int64_t inputExtent = inputShape[axis];
      int64_t outputExtent = outputShape[axis];
      if (inputExtent == outputExtent)
        continue;
      double scale =
          static_cast<double>(outputExtent) / static_cast<double>(inputExtent);
      SmallVector<int32_t> indices;
      indices.reserve(outputExtent);
      for (int64_t outputIndex = 0; outputIndex < outputExtent; ++outputIndex) {
        double coordinate =
            halfPixel ? (static_cast<double>(outputIndex) + 0.5) / scale - 0.5
                      : static_cast<double>(outputIndex) / scale;
        int64_t index = static_cast<int64_t>(std::ceil(coordinate - 0.5));
        indices.push_back(static_cast<int32_t>(
            std::clamp(index, int64_t{0}, inputExtent - 1)));
      }
      auto indicesType =
          RankedTensorType::get({outputExtent}, rewriter.getI32Type());
      Value indicesValue = arith::ConstantOp::create(rewriter, op.getLoc(),
          indicesType, DenseIntElementsAttr::get(indicesType, indices));
      currentShape[axis] = outputExtent;
      auto nextType =
          RankedTensorType::get(currentShape, rewriter.getF32Type());
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
          rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.gather",
          TypeRange{nextType}, ValueRange{result, indicesValue}, attributes)
                   ->getResult(0);
    }
    Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
    Value toPhysical =
        createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
    return createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
        TypeRange{physicalResultType}, ValueRange{result, toPhysical})
        ->getResult(0);
  }

  StringRef tflOpName;
  bool alignCorners = false;
  bool halfPixelCenters = false;
  if (mode == "nearest" && coordinateMode == "asymmetric" &&
      nearestMode == "floor") {
    tflOpName = "tfl.resize_nearest_neighbor";
  } else if (mode == "linear" && coordinateMode == "half_pixel") {
    tflOpName = "tfl.resize_bilinear";
    halfPixelCenters = true;
  } else if (mode == "linear" && coordinateMode == "align_corners") {
    tflOpName = "tfl.resize_bilinear";
    alignCorners = true;
  } else {
    return std::nullopt;
  }
  Value size = createI32ShapeConstant(
      rewriter, op.getLoc(), {outputShape[2], outputShape[3]});
  auto resultType =
      cast<RankedTensorType>(convertRank4NCHWToNHWCType(sourceResultType));
  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr(
          "align_corners", rewriter.getBoolAttr(alignCorners)),
      rewriter.getNamedAttr(
          "half_pixel_centers", rewriter.getBoolAttr(halfPixelCenters))};
  return createTFLOperation(rewriter, op.getLoc(), tflOpName,
      TypeRange{resultType}, ValueRange{adaptor.getX(), size}, attributes)
      ->getResult(0);
}

class ResizeLowering final : public OpConversionPattern<ONNXResizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXResizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getX();
    if (failed(validateStaticF32Tensor(op, input.getType(), "input")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();
    auto sourceType = cast<RankedTensorType>(op.getX().getType());
    auto sourceResultType = cast<RankedTensorType>(op.getY().getType());
    int64_t rank = sourceType.getRank();
    if (rank != sourceResultType.getRank() || rank < 1 || rank > 5)
      return op.emitError("ONNXToTFL Resize supports matching static rank-1 "
                          "through rank-5 tensors"),
             failure();
    for (int64_t dimension : sourceType.getShape())
      if (dimension <= 0)
        return op.emitError("Resize requires positive input dimensions"),
               failure();
    for (int64_t dimension : sourceResultType.getShape())
      if (dimension <= 0)
        return op.emitError("Resize requires positive output dimensions"),
               failure();

    if (std::optional<Value> fast = tryFastResize(
            op, adaptor, sourceType, sourceResultType, rewriter)) {
      rewriter.replaceOp(op, *fast);
      return success();
    }

    SmallVector<int64_t> axes;
    if (ArrayAttr axesAttribute = op.getAxesAttr()) {
      axes.reserve(axesAttribute.size());
      for (Attribute attribute : axesAttribute) {
        int64_t axis =
            normalizeAxis(cast<IntegerAttr>(attribute).getInt(), rank);
        if (axis < 0 || axis >= rank || llvm::is_contained(axes, axis))
          return op.emitError("Resize axes must be unique and in range"),
                 failure();
        axes.push_back(axis);
      }
    } else {
      axes.resize(rank);
      std::iota(axes.begin(), axes.end(), 0);
    }
    if (axes.empty())
      return op.emitError("Resize axes cannot be empty"), failure();

    ArrayRef<int64_t> inputShape = sourceType.getShape();
    ArrayRef<int64_t> outputShape = sourceResultType.getShape();
    for (int64_t dimension = 0; dimension < rank; ++dimension)
      if (!llvm::is_contained(axes, dimension) &&
          inputShape[dimension] != outputShape[dimension])
        return op.emitError("Resize changed a dimension not listed in axes"),
               failure();

    bool hasScales = !isAbsentResizeInput(op.getScales());
    bool hasSizes = !isAbsentResizeInput(op.getSizes());
    if (hasScales == hasSizes)
      return op.emitError("Resize requires exactly one of scales or sizes"),
             failure();
    SmallVector<double> scales(rank, 1.0);
    size_t parameterCount = axes.size();
    if (hasScales) {
      FailureOr<SmallVector<float>> values =
          getConstantF32Values(op.getScales());
      if (failed(values) || values->size() != parameterCount)
        return op.emitError("Resize requires compile-time scales matching "
                            "the axes count"),
               failure();
      for (auto [index, axis] : llvm::enumerate(axes))
        scales[axis] = (*values)[index];
    } else {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getSizes());
      if (failed(values) || values->size() != parameterCount)
        return op.emitError("Resize requires compile-time sizes matching the "
                            "axes count"),
               failure();
      StringRef aspectPolicy = adaptor.getKeepAspectRatioPolicy();
      if (aspectPolicy == "stretch") {
        for (int64_t axis : axes)
          scales[axis] = static_cast<double>(outputShape[axis]) /
                         static_cast<double>(inputShape[axis]);
      } else if (aspectPolicy == "not_larger" ||
                 aspectPolicy == "not_smaller") {
        SmallVector<double> requestedScales;
        requestedScales.reserve(parameterCount);
        for (auto [index, axis] : llvm::enumerate(axes))
          requestedScales.push_back(static_cast<double>((*values)[index]) /
                                    static_cast<double>(inputShape[axis]));
        double commonScale = aspectPolicy == "not_larger"
                                 ? *std::min_element(requestedScales.begin(),
                                       requestedScales.end())
                                 : *std::max_element(requestedScales.begin(),
                                       requestedScales.end());
        for (int64_t axis : axes)
          scales[axis] = commonScale;
      } else {
        return op.emitError("unsupported Resize aspect-ratio policy"),
               failure();
      }
    }

    StringRef coordinateMode = adaptor.getCoordinateTransformationMode();
    SmallVector<double> roiStarts(rank, 0.0);
    SmallVector<double> roiEnds(rank, 1.0);
    if (coordinateMode == "tf_crop_and_resize") {
      if (isAbsentResizeInput(op.getRoi()))
        return op.emitError("tf_crop_and_resize requires a constant ROI"),
               failure();
      FailureOr<SmallVector<float>> roi = getConstantF32Values(op.getRoi());
      if (failed(roi) || roi->size() != 2 * parameterCount)
        return op.emitError("Resize ROI must contain starts and ends for each "
                            "axis"),
               failure();
      for (auto [index, axis] : llvm::enumerate(axes)) {
        roiStarts[axis] = (*roi)[index];
        roiEnds[axis] = (*roi)[index + parameterCount];
      }
    }

    Value logicalInput = input;
    if (rank == 4) {
      Value toLogical =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      logicalInput = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{input, toLogical})
                         ->getResult(0);
    }

    SmallVector<int64_t> currentShape(inputShape);
    Value result = logicalInput;
    bool antialias = adaptor.getAntialias() != 0;
    bool excludeOutside = adaptor.getExcludeOutside() != 0;
    double cubicCoefficient = adaptor.getCubicCoeffA().convertToDouble();
    double extrapolationValue =
        adaptor.getExtrapolationValue().convertToDouble();
    for (int64_t axis : axes) {
      FailureOr<AxisInterpolation> interpolation =
          buildAxisInterpolation(op, currentShape[axis], outputShape[axis],
              scales[axis], adaptor.getMode(), adaptor.getNearestMode(),
              coordinateMode, antialias, excludeOutside, cubicCoefficient,
              roiStarts[axis], roiEnds[axis], extrapolationValue);
      if (failed(interpolation))
        return failure();
      result = applyAxisInterpolation(op, result, currentShape, axis,
          *interpolation, outputShape[axis], rewriter);
    }
    if (currentShape != SmallVector<int64_t>(outputShape))
      return op.emitError("Resize static interpolation produced wrong shape"),
             failure();

    if (rank == 4) {
      Value toPhysical =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 2, 3, 1});
      Type physicalResultType = convertRank4NCHWToNHWCType(sourceResultType);
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, toPhysical})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXResizeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ResizeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
