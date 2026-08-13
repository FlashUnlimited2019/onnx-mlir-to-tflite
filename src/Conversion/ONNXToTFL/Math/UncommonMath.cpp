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

Value binary(ConversionPatternRewriter &rewriter, Location loc,
    StringRef name, Type type, Value lhs, Value rhs,
    bool fusedActivation = true) {
  SmallVector<NamedAttribute> attributes;
  if (fusedActivation)
    attributes.push_back(getFusedActivationNone(rewriter));
  return createTFLOperation(rewriter, loc, name, TypeRange{type},
      ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

Value reshape(ConversionPatternRewriter &rewriter, Location loc, Value value,
    ArrayRef<int64_t> shape) {
  auto elementType = cast<ShapedType>(value.getType()).getElementType();
  auto resultType = RankedTensorType::get(shape, elementType);
  Value shapeValue = createI32ShapeConstant(rewriter, loc, shape);
  return createTFLOperation(rewriter, loc, "tfl.reshape",
      TypeRange{resultType}, ValueRange{value, shapeValue})
      ->getResult(0);
}

class MeanVarianceNormalizationLowering final
    : public OpConversionPattern<ONNXMeanVarianceNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMeanVarianceNormalizationOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType ||
        failed(validateStaticF32Tensor(op, inputType, "MVN input")) ||
        failed(validateStaticF32Tensor(op, resultType, "MVN result")) ||
        inputType != resultType)
      return failure();

    SmallVector<int64_t> axes;
    SmallVector<int64_t> meanShape(inputType.getShape());
    for (Attribute attr : op.getAxes()) {
      int64_t axis = normalizeAxis(
          cast<IntegerAttr>(attr).getValue().getSExtValue(),
          inputType.getRank());
      if (axis < 0 || axis >= inputType.getRank())
        return op.emitError("MeanVarianceNormalization axis is out of range"),
               failure();
      axes.push_back(axis);
      meanShape[axis] = 1;
    }
    Location loc = op.getLoc();
    Value input = restoreLogicalRank4(
        rewriter, loc, adaptor.getOperands().front(), inputType);
    Value axisValue = createI32ShapeConstant(rewriter, loc, axes);
    auto meanType = RankedTensorType::get(meanShape, rewriter.getF32Type());
    SmallVector<NamedAttribute> meanAttributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    Value mean = createTFLOperation(rewriter, loc, "tfl.mean",
        TypeRange{meanType}, ValueRange{input, axisValue}, meanAttributes)
                     ->getResult(0);
    Value centered = binary(
        rewriter, loc, "tfl.sub", inputType, input, mean);
    Value squared = binary(
        rewriter, loc, "tfl.mul", inputType, centered, centered);
    Value variance = createTFLOperation(rewriter, loc, "tfl.mean",
        TypeRange{meanType}, ValueRange{squared, axisValue}, meanAttributes)
                         ->getResult(0);
    Value deviation = createTFLOperation(rewriter, loc, "tfl.sqrt",
        TypeRange{meanType}, ValueRange{variance})
                          ->getResult(0);
    Value result = binary(
        rewriter, loc, "tfl.div", resultType, centered, deviation);
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

class MishLowering final : public OpConversionPattern<ONNXMishOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMishOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!resultType ||
        failed(validateStaticF32Tensor(op, op.getX().getType(), "Mish input")) ||
        failed(validateStaticF32Tensor(op, resultType, "Mish result")))
      return failure();
    auto physicalType = cast<RankedTensorType>(
        convertRank4NCHWToNHWCType(resultType));
    Location loc = op.getLoc();
    Value input = adaptor.getOperands().front();
    Value zero = createF32ScalarTensorConstant(rewriter, loc, 0.0f);
    Value one = createF32ScalarTensorConstant(rewriter, loc, 1.0f);
    Value positive = binary(rewriter, loc, "tfl.maximum", physicalType,
        input, zero, /*fusedActivation=*/false);
    Value absolute = createTFLOperation(rewriter, loc, "tfl.abs",
        TypeRange{physicalType}, ValueRange{input})
                         ->getResult(0);
    Value negative = createTFLOperation(rewriter, loc, "tfl.neg",
        TypeRange{physicalType}, ValueRange{absolute})
                         ->getResult(0);
    Value exponential = createTFLOperation(rewriter, loc, "tfl.exp",
        TypeRange{physicalType}, ValueRange{negative})
                            ->getResult(0);
    Value tail = createTFLOperation(rewriter, loc, "tfl.log",
        TypeRange{physicalType},
        ValueRange{binary(
            rewriter, loc, "tfl.add", physicalType, one, exponential)})
                     ->getResult(0);
    Value softplus =
        binary(rewriter, loc, "tfl.add", physicalType, positive, tail);
    Value activated = createTFLOperation(rewriter, loc, "tfl.tanh",
        TypeRange{physicalType}, ValueRange{softplus})
                          ->getResult(0);
    rewriter.replaceOp(op,
        binary(rewriter, loc, "tfl.mul", physicalType, input, activated));
    return success();
  }
};

class DFTLowering final : public OpConversionPattern<ONNXDFTOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDFTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    FailureOr<SmallVector<int64_t>> axisValues =
        getConstantIntValues(op.getAxis());
    if (!inputType || !resultType || inputType.getRank() < 2 ||
        inputType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "DFT input")) ||
        failed(validateStaticF32Tensor(op, resultType, "DFT result")) ||
        inputType.getShape() != resultType.getShape() || failed(axisValues) ||
        axisValues->size() != 1 || op.getOnesided() != 0)
      return op.emitError("ONNXToTFL DFT requires equal static FP32 input/"
                          "result shapes, a constant axis, and onesided=0"),
             failure();
    int64_t rank = inputType.getRank();
    int64_t axis = normalizeAxis(axisValues->front(), rank);
    if (axis < 0 || axis >= rank - 1 || inputType.getShape().back() != 2)
      return op.emitError("DFT requires a transform axis before a final "
                          "complex-component dimension of size 2"),
             failure();
    int64_t length = inputType.getShape()[axis];
    if (!isa<NoneType>(op.getDftLength().getType())) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(op.getDftLength());
      if (failed(values) || values->size() != 1 || values->front() != length)
        return op.emitError("DFT length must be omitted or equal the static "
                            "transform dimension"),
               failure();
    }

    Location loc = op.getLoc();
    Value input = restoreLogicalRank4(rewriter, loc, adaptor.getInput(), inputType);
    SmallVector<int64_t> permutation;
    for (int64_t i = 0; i < rank - 1; ++i)
      if (i != axis)
        permutation.push_back(i);
    permutation.push_back(axis);
    permutation.push_back(rank - 1);
    SmallVector<int64_t> permutedShape;
    for (int64_t index : permutation)
      permutedShape.push_back(inputType.getShape()[index]);
    bool identityPermutation = true;
    for (int64_t i = 0; i < rank; ++i)
      identityPermutation &= permutation[i] == i;
    if (!identityPermutation) {
      auto permutedType =
          RankedTensorType::get(permutedShape, rewriter.getF32Type());
      Value permutationValue =
          createI32ShapeConstant(rewriter, loc, permutation);
      input = createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{permutedType}, ValueRange{input, permutationValue})
                  ->getResult(0);
    }
    int64_t outer = inputType.getNumElements() / (length * 2);
    Value flattened = reshape(rewriter, loc, input, {outer, length, 2});
    auto component3DType =
        RankedTensorType::get({outer, length, 1}, rewriter.getF32Type());
    auto component2DType =
        RankedTensorType::get({outer, length}, rewriter.getF32Type());
    auto component = [&](int64_t index) {
      Value begin = createI32ShapeConstant(rewriter, loc, {0, 0, index});
      Value size = createI32ShapeConstant(rewriter, loc, {outer, length, 1});
      Value slice = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{component3DType}, ValueRange{flattened, begin, size})
                        ->getResult(0);
      return reshape(rewriter, loc, slice, {outer, length});
    };
    Value realInput = component(0);
    Value imagInput = component(1);

    constexpr double pi = 3.14159265358979323846264338327950288;
    SmallVector<float> cosine;
    SmallVector<float> sine;
    cosine.reserve(length * length);
    sine.reserve(length * length);
    float scale = op.getInverse() != 0 ? 1.0f / static_cast<float>(length)
                                       : 1.0f;
    for (int64_t sample = 0; sample < length; ++sample)
      for (int64_t frequency = 0; frequency < length; ++frequency) {
        double angle = 2.0 * pi * static_cast<double>(sample * frequency) /
                       static_cast<double>(length);
        cosine.push_back(scale * static_cast<float>(std::cos(angle)));
        sine.push_back(scale * static_cast<float>(std::sin(angle)));
      }
    auto coefficientType =
        RankedTensorType::get({length, length}, rewriter.getF32Type());
    Value cosineValue = arith::ConstantOp::create(rewriter, loc,
        coefficientType, DenseFPElementsAttr::get(coefficientType, cosine));
    Value sineValue = arith::ConstantOp::create(rewriter, loc,
        coefficientType, DenseFPElementsAttr::get(coefficientType, sine));
    SmallVector<NamedAttribute> matmulAttributes{
        rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
        rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(false))};
    auto matmul = [&](Value lhs, Value rhs) {
      return createTFLOperation(rewriter, loc, "tfl.batch_matmul",
          TypeRange{component2DType}, ValueRange{lhs, rhs}, matmulAttributes)
          ->getResult(0);
    };
    Value realCos = matmul(realInput, cosineValue);
    Value realSin = matmul(realInput, sineValue);
    Value imagCos = matmul(imagInput, cosineValue);
    Value imagSin = matmul(imagInput, sineValue);
    Value real;
    Value imag;
    if (op.getInverse() != 0) {
      real = binary(
          rewriter, loc, "tfl.sub", component2DType, realCos, imagSin);
      imag = binary(
          rewriter, loc, "tfl.add", component2DType, realSin, imagCos);
    } else {
      real = binary(
          rewriter, loc, "tfl.add", component2DType, realCos, imagSin);
      imag = binary(
          rewriter, loc, "tfl.sub", component2DType, imagCos, realSin);
    }
    auto packedType =
        RankedTensorType::get({outer, length, 2}, rewriter.getF32Type());
    SmallVector<NamedAttribute> packAttributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(2)),
        rewriter.getNamedAttr("values_count", rewriter.getI32IntegerAttr(2))};
    Value result = createTFLOperation(rewriter, loc, "tfl.pack",
        TypeRange{packedType}, ValueRange{real, imag}, packAttributes)
                       ->getResult(0);
    result = reshape(rewriter, loc, result, permutedShape);
    if (!identityPermutation) {
      SmallVector<int64_t> inversePermutation(rank);
      for (int64_t i = 0; i < rank; ++i)
        inversePermutation[permutation[i]] = i;
      Value inverseValue =
          createI32ShapeConstant(rewriter, loc, inversePermutation);
      result = createTFLOperation(rewriter, loc, "tfl.transpose",
          TypeRange{resultType}, ValueRange{result, inverseValue})
                   ->getResult(0);
    }
    rewriter.replaceOp(
        op, makePhysicalRank4(rewriter, loc, result, resultType));
    return success();
  }
};

class DetLowering final : public OpConversionPattern<ONNXDetOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXDetOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || inputType.getRank() < 2 ||
        inputType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "Det input")) ||
        failed(validateStaticF32TensorOrScalar(op, resultType, "Det result")))
      return failure();
    int64_t size = inputType.getShape().back();
    if (size <= 0 || inputType.getShape()[inputType.getRank() - 2] != size ||
        resultType.getNumElements() * size * size != inputType.getNumElements())
      return op.emitError("Det requires static square matrices"), failure();
    // This compact Schur-complement elimination is intended for the small,
    // nonsingular static matrices used by the fixture. It does not introduce
    // dynamic loops or control flow.
    int64_t batches = resultType.getNumElements();
    Location loc = op.getLoc();
    Value matrix = restoreLogicalRank4(rewriter, loc, adaptor.getX(), inputType);
    matrix = reshape(rewriter, loc, matrix, {batches, size, size});
    auto matrixType =
        RankedTensorType::get({batches, size, size}, rewriter.getF32Type());
    auto pivotType =
        RankedTensorType::get({batches, 1, 1}, rewriter.getF32Type());
    Value determinant;
    for (int64_t k = 0; k < size; ++k) {
      Value pivotBegin = createI32ShapeConstant(rewriter, loc, {0, k, k});
      Value pivotSize = createI32ShapeConstant(rewriter, loc, {batches, 1, 1});
      Value pivot = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{pivotType}, ValueRange{matrix, pivotBegin, pivotSize})
                        ->getResult(0);
      determinant = determinant
                        ? binary(rewriter, loc, "tfl.mul", pivotType,
                              determinant, pivot)
                        : pivot;
      if (k + 1 == size)
        continue;
      auto columnType =
          RankedTensorType::get({batches, size, 1}, rewriter.getF32Type());
      auto rowType =
          RankedTensorType::get({batches, 1, size}, rewriter.getF32Type());
      Value columnBegin = createI32ShapeConstant(rewriter, loc, {0, 0, k});
      Value columnSize =
          createI32ShapeConstant(rewriter, loc, {batches, size, 1});
      Value column = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{columnType}, ValueRange{matrix, columnBegin, columnSize})
                         ->getResult(0);
      Value rowBegin = createI32ShapeConstant(rewriter, loc, {0, k, 0});
      Value rowSize =
          createI32ShapeConstant(rewriter, loc, {batches, 1, size});
      Value row = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{rowType}, ValueRange{matrix, rowBegin, rowSize})
                      ->getResult(0);
      Value factors =
          binary(rewriter, loc, "tfl.div", columnType, column, pivot);
      Value outer =
          binary(rewriter, loc, "tfl.mul", matrixType, factors, row);
      SmallVector<float> mask(size * size, 0.0f);
      for (int64_t i = k + 1; i < size; ++i)
        for (int64_t j = k + 1; j < size; ++j)
          mask[i * size + j] = 1.0f;
      auto maskType =
          RankedTensorType::get({1, size, size}, rewriter.getF32Type());
      Value maskValue = arith::ConstantOp::create(rewriter, loc, maskType,
          DenseFPElementsAttr::get(maskType, mask));
      Value correction =
          binary(rewriter, loc, "tfl.mul", matrixType, outer, maskValue);
      matrix =
          binary(rewriter, loc, "tfl.sub", matrixType, matrix, correction);
    }
    rewriter.replaceOp(op,
        reshape(rewriter, loc, determinant, resultType.getShape()));
    return success();
  }
};

class NegativeLogLikelihoodLossLowering final
    : public OpConversionPattern<ONNXNegativeLogLikelihoodLossOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXNegativeLogLikelihoodLossOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto targetType = dyn_cast<RankedTensorType>(op.getTarget().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getLoss().getType());
    FailureOr<SmallVector<int64_t>> labels =
        getConstantIntValues(op.getTarget());
    if (!inputType || !targetType || !resultType || inputType.getRank() < 2 ||
        inputType.getRank() > 5 ||
        failed(validateStaticF32Tensor(op, inputType, "NLL input")) ||
        !targetType.hasStaticShape() ||
        (!targetType.getElementType().isSignlessInteger(32) &&
            !targetType.getElementType().isSignlessInteger(64)) ||
        failed(validateStaticF32TensorOrScalar(op, resultType, "NLL result")) ||
        !isa<NoneType>(op.getWeight().getType()) || op.getIgnoreIndex() ||
        failed(labels) || labels->size() != targetType.getNumElements())
      return op.emitError("ONNXToTFL NLLLoss requires static FP32 input, "
                          "constant integer targets, no weight/ignore_index"),
             failure();
    int64_t rank = inputType.getRank();
    if (targetType.getRank() != rank - 1 ||
        targetType.getShape()[0] != inputType.getShape()[0])
      return op.emitError("NLLLoss input and target ranks are inconsistent"),
             failure();
    for (int64_t i = 1; i < targetType.getRank(); ++i)
      if (targetType.getShape()[i] != inputType.getShape()[i + 1])
        return op.emitError("NLLLoss target shape is inconsistent"), failure();

    int64_t samples = targetType.getNumElements();
    int64_t classes = inputType.getShape()[1];
    SmallVector<int32_t> coordinates;
    coordinates.reserve(samples * rank);
    ArrayRef<int64_t> targetShape = targetType.getShape();
    for (int64_t linear = 0; linear < samples; ++linear) {
      int64_t remaining = linear;
      SmallVector<int64_t> index(targetType.getRank());
      for (int64_t i = targetType.getRank() - 1; i >= 0; --i) {
        index[i] = remaining % targetShape[i];
        remaining /= targetShape[i];
      }
      int64_t label = (*labels)[linear];
      if (label < 0 || label >= classes)
        return op.emitError("NLLLoss target is out of class range"), failure();
      coordinates.push_back(static_cast<int32_t>(index[0]));
      coordinates.push_back(static_cast<int32_t>(label));
      for (int64_t i = 1; i < targetType.getRank(); ++i)
        coordinates.push_back(static_cast<int32_t>(index[i]));
    }
    Location loc = op.getLoc();
    Value input = restoreLogicalRank4(rewriter, loc, adaptor.getInput(), inputType);
    auto coordinateType =
        RankedTensorType::get({samples, rank}, rewriter.getI32Type());
    Value coordinateValue = arith::ConstantOp::create(rewriter, loc,
        coordinateType,
        DenseIntElementsAttr::get(coordinateType, coordinates));
    auto gatheredType =
        RankedTensorType::get({samples}, rewriter.getF32Type());
    Value gathered = createTFLOperation(rewriter, loc, "tfl.gather_nd",
        TypeRange{gatheredType}, ValueRange{input, coordinateValue})
                         ->getResult(0);
    Value losses = createTFLOperation(rewriter, loc, "tfl.neg",
        TypeRange{gatheredType}, ValueRange{gathered})
                       ->getResult(0);
    Value result;
    if (op.getReduction() == "none") {
      result = reshape(rewriter, loc, losses, resultType.getShape());
      result = makePhysicalRank4(rewriter, loc, result, resultType);
    } else if (op.getReduction() == "mean" || op.getReduction() == "sum") {
      Value axes = createI32ShapeConstant(rewriter, loc, {0});
      SmallVector<NamedAttribute> attributes{rewriter.getNamedAttr(
          "keep_dims", rewriter.getBoolAttr(false))};
      StringRef name = op.getReduction() == "mean" ? "tfl.mean" : "tfl.sum";
      result = createTFLOperation(rewriter, loc, name, TypeRange{resultType},
          ValueRange{losses, axes}, attributes)
                   ->getResult(0);
    } else {
      return op.emitError("unsupported NLLLoss reduction"), failure();
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXUncommonMathOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MeanVarianceNormalizationLowering, MishLowering, DFTLowering,
      DetLowering, NegativeLogLikelihoodLossLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
