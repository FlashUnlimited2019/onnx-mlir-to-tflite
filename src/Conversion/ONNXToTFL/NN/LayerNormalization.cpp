/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

class LayerNormalizationLowering final
    : public OpConversionPattern<ONNXLayerNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLayerNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto scaleType = dyn_cast<RankedTensorType>(op.getScale().getType());
    auto biasType = dyn_cast<RankedTensorType>(op.getB().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !scaleType || !biasType || !resultType ||
        failed(validateStaticF32Tensor(
            op, inputType, "LayerNormalization input")) ||
        failed(validateStaticF32Tensor(
            op, scaleType, "LayerNormalization scale")) ||
        failed(
            validateStaticF32Tensor(op, biasType, "LayerNormalization bias")) ||
        failed(validateStaticF32Tensor(
            op, resultType, "LayerNormalization result"))) {
      op.emitError("ONNXToTFL LayerNormalization requires static FP32 input, "
                   "scale, bias, and result tensors");
      return failure();
    }

    // Lower last-axis LayerNormalization directly. Rank-4 activations are
    // physically NHWC, so logical axis 3 (W) and its scale/bias parameters
    // must be moved to physical axis 2.
    int64_t normalizedAxis = normalizeAxis(op.getAxis(), inputType.getRank());
    if (inputType.getRank() >= 2 && inputType.getRank() <= 5 &&
        resultType == inputType && normalizedAxis == inputType.getRank() - 1 &&
        scaleType.getRank() == 1 && biasType.getRank() == 1 &&
        scaleType.getShape()[0] == inputType.getShape().back() &&
        biasType.getShape() == scaleType.getShape() && op.getStashType() == 1 &&
        isa<NoneType>(op.getMean().getType()) &&
        isa<NoneType>(op.getInvStdDev().getType())) {
      auto calculationType =
          cast<RankedTensorType>(convertRank4NCHWToNHWCType(inputType));
      int64_t calculationAxis = inputType.getRank() == 4
                                    ? mapNCHWAxisToNHWC(normalizedAxis)
                                    : normalizedAxis;
      SmallVector<int64_t> statisticShape(calculationType.getShape());
      statisticShape[calculationAxis] = 1;
      auto statisticType =
          RankedTensorType::get(statisticShape, rewriter.getF32Type());
      Value axes =
          createI32ShapeConstant(rewriter, op.getLoc(), {calculationAxis});
      SmallVector<NamedAttribute> keepDims{
          rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
      SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
      Value input = adaptor.getX();
      Value mean = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
          TypeRange{statisticType}, ValueRange{input, axes}, keepDims)
                       ->getResult(0);
      Value centered = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
          TypeRange{calculationType}, ValueRange{input, mean}, fusedNone)
                           ->getResult(0);
      Value squared = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
          TypeRange{calculationType}, ValueRange{centered, centered}, fusedNone)
                          ->getResult(0);
      Value variance = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
          TypeRange{statisticType}, ValueRange{squared, axes}, keepDims)
                           ->getResult(0);
      Value epsilon = createF32ScalarTensorConstant(
          rewriter, op.getLoc(), op.getEpsilon().convertToFloat());
      Value varianceEpsilon = createTFLOperation(rewriter, op.getLoc(),
          "tfl.add", TypeRange{statisticType}, ValueRange{variance, epsilon},
          fusedNone)
                                  ->getResult(0);
      Value inverseStdDev = createTFLOperation(rewriter, op.getLoc(),
          "tfl.rsqrt", TypeRange{statisticType}, ValueRange{varianceEpsilon})
                                ->getResult(0);
      Value normalized = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
          TypeRange{calculationType}, ValueRange{centered, inverseStdDev},
          fusedNone)
                             ->getResult(0);

      Value scale = adaptor.getScale();
      Value bias = adaptor.getB();
      if (inputType.getRank() == 4) {
        SmallVector<int64_t> parameterShape(4, 1);
        parameterShape[calculationAxis] = scaleType.getShape()[0];
        auto parameterType =
            RankedTensorType::get(parameterShape, rewriter.getF32Type());
        Value parameterShapeValue =
            createI32ShapeConstant(rewriter, op.getLoc(), parameterShape);
        scale = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{parameterType}, ValueRange{scale, parameterShapeValue})
                    ->getResult(0);
        bias = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{parameterType}, ValueRange{bias, parameterShapeValue})
                   ->getResult(0);
      }
      Value scaled = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
          TypeRange{calculationType}, ValueRange{normalized, scale}, fusedNone)
                         ->getResult(0);
      Value shifted = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
          TypeRange{calculationType}, ValueRange{scaled, bias}, fusedNone)
                          ->getResult(0);
      SmallVector<NamedAttribute> noValueAttributes{
          rewriter.getNamedAttr("value", rewriter.getUnitAttr())};
      Operation *noValue =
          createTFLOperation(rewriter, op.getLoc(), "tfl.no_value",
              TypeRange{rewriter.getNoneType()}, {}, noValueAttributes);
      rewriter.replaceOp(op,
          ValueRange{shifted, noValue->getResult(0), noValue->getResult(0)});
      return success();
    }

    int64_t rank = inputType.getRank();
    if (rank < 3 || rank > 5 || scaleType.getRank() != rank - 1 ||
        biasType.getRank() != rank - 1 || resultType != inputType) {
      op.emitError("ONNXToTFL LayerNormalization supports rank-2 through "
                   "rank-5 last-axis normalization and rank-3 through rank-5 "
                   "InstanceNormalization forms");
      return failure();
    }
    int64_t channels = inputType.getShape()[1];
    SmallVector<int64_t> expectedParameterShape(rank - 1, 1);
    expectedParameterShape[0] = channels;
    if (normalizedAxis != 2 || op.getStashType() != 1 ||
        !llvm::equal(scaleType.getShape(), expectedParameterShape) ||
        !llvm::equal(biasType.getShape(), expectedParameterShape) ||
        !isa<NoneType>(op.getMean().getType()) ||
        !isa<NoneType>(op.getInvStdDev().getType())) {
      op.emitError("unsupported LayerNormalization configuration: expected "
                   "imported InstanceNormalization with axis=2, stash_type=1, "
                   "[C,1,...] scale/bias, and unused statistics outputs");
      return failure();
    }

    auto calculationType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    SmallVector<int64_t> reductionAxes;
    if (rank == 4) {
      reductionAxes = {1, 2};
    } else {
      for (int64_t axis = 2; axis < rank; ++axis)
        reductionAxes.push_back(axis);
    }
    SmallVector<int64_t> statisticShape(calculationType.getShape());
    for (int64_t axis : reductionAxes)
      statisticShape[axis] = 1;
    auto statisticType =
        RankedTensorType::get(statisticShape, rewriter.getF32Type());
    Value axes =
        createI32ShapeConstant(rewriter, op.getLoc(), reductionAxes);
    SmallVector<NamedAttribute> keepDims{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};
    Value input = adaptor.getX();
    Operation *mean = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
        TypeRange{statisticType}, ValueRange{input, axes}, keepDims);
    Operation *centered = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
        TypeRange{calculationType}, ValueRange{input, mean->getResult(0)},
        fusedNone);
    Operation *squared = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType},
        ValueRange{centered->getResult(0), centered->getResult(0)}, fusedNone);
    Operation *variance = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
        TypeRange{statisticType}, ValueRange{squared->getResult(0), axes},
        keepDims);
    Value epsilon = createF32ScalarTensorConstant(
        rewriter, op.getLoc(), op.getEpsilon().convertToFloat());
    Operation *varianceEpsilon = createTFLOperation(rewriter, op.getLoc(),
        "tfl.add", TypeRange{statisticType},
        ValueRange{variance->getResult(0), epsilon}, fusedNone);
    Operation *inverseStdDev = createTFLOperation(rewriter, op.getLoc(),
        "tfl.rsqrt", TypeRange{statisticType},
        ValueRange{varianceEpsilon->getResult(0)});
    Operation *normalized = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType},
        ValueRange{centered->getResult(0), inverseStdDev->getResult(0)},
        fusedNone);

    SmallVector<int64_t> parameterBroadcastShape;
    if (rank == 4) {
      parameterBroadcastShape = {1, 1, channels};
    } else {
      parameterBroadcastShape.assign(rank, 1);
      parameterBroadcastShape[1] = channels;
    }
    auto parameterType = RankedTensorType::get(
        parameterBroadcastShape, rewriter.getF32Type());
    Value parameterShape = createI32ShapeConstant(
        rewriter, op.getLoc(), parameterBroadcastShape);
    Operation *scale = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{parameterType},
        ValueRange{adaptor.getScale(), parameterShape});
    Operation *scaled = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType},
        ValueRange{normalized->getResult(0), scale->getResult(0)}, fusedNone);
    Operation *bias = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
        TypeRange{parameterType}, ValueRange{adaptor.getB(), parameterShape});
    Operation *shifted = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
        TypeRange{calculationType},
        ValueRange{scaled->getResult(0), bias->getResult(0)}, fusedNone);

    SmallVector<NamedAttribute> noValueAttributes{
        rewriter.getNamedAttr("value", rewriter.getUnitAttr())};
    Operation *noValue =
        createTFLOperation(rewriter, op.getLoc(), "tfl.no_value",
            TypeRange{rewriter.getNoneType()}, {}, noValueAttributes);
    rewriter.replaceOp(op, ValueRange{shifted->getResult(0),
                               noValue->getResult(0), noValue->getResult(0)});
    return success();
  }
};

class RMSLayerNormalizationLowering final
    : public OpConversionPattern<ONNXRMSLayerNormalizationOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXRMSLayerNormalizationOp op,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto scaleType = dyn_cast<RankedTensorType>(op.getScale().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    bool hasBias = !isa<NoneType>(op.getB().getType());
    auto biasType = dyn_cast<RankedTensorType>(op.getB().getType());
    if (!inputType || !scaleType || !resultType ||
        failed(validateStaticF32Tensor(
            op, inputType, "RMSLayerNormalization input")) ||
        failed(validateStaticF32Tensor(
            op, scaleType, "RMSLayerNormalization scale")) ||
        failed(validateStaticF32Tensor(
            op, resultType, "RMSLayerNormalization result")) ||
        (hasBias && (!biasType || failed(validateStaticF32Tensor(op, biasType,
                                      "RMSLayerNormalization bias"))))) {
      op.emitError("ONNXToTFL RMSLayerNormalization requires static FP32 "
                   "input, scale, optional bias, and result tensors");
      return failure();
    }

    int64_t rank = inputType.getRank();
    int64_t normalizedAxis = normalizeAxis(op.getAxis(), rank);
    bool biasMatchesScale = hasBias && biasType.getRank() == 1 &&
                            biasType.getShape() == scaleType.getShape();
    bool biasIsFullRankBroadcast = hasBias && biasType.getRank() == rank;
    if (biasIsFullRankBroadcast) {
      for (auto [biasDimension, inputDimension] :
          llvm::zip_equal(biasType.getShape(), inputType.getShape())) {
        if (biasDimension != 1 && biasDimension != inputDimension) {
          biasIsFullRankBroadcast = false;
          break;
        }
      }
    }
    if (rank < 2 || rank > 5 || resultType != inputType ||
        normalizedAxis != rank - 1 || scaleType.getRank() != 1 ||
        scaleType.getShape()[0] != inputType.getShape().back() ||
        (hasBias && !biasMatchesScale && !biasIsFullRankBroadcast) ||
        op.getStashType() != 1 || !isa<NoneType>(op.getInvStdDev().getType())) {
      op.emitError("ONNXToTFL RMSLayerNormalization supports rank-2 through "
                   "rank-5 static FP32 last-axis normalization with a "
                   "rank-1 scale, optional rank-1 or full-rank broadcast "
                   "bias, stash_type=1, and an unused InvStdDev output");
      return failure();
    }

    auto calculationType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(inputType));
    int64_t calculationAxis =
        rank == 4 ? mapNCHWAxisToNHWC(normalizedAxis) : normalizedAxis;
    SmallVector<int64_t> statisticShape(calculationType.getShape());
    statisticShape[calculationAxis] = 1;
    auto statisticType =
        RankedTensorType::get(statisticShape, rewriter.getF32Type());
    Value axes =
        createI32ShapeConstant(rewriter, op.getLoc(), {calculationAxis});
    SmallVector<NamedAttribute> keepDims{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    SmallVector<NamedAttribute> fusedNone{getFusedActivationNone(rewriter)};

    Value input = adaptor.getX();
    Value squared = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType}, ValueRange{input, input}, fusedNone)
                        ->getResult(0);
    Value meanSquare = createTFLOperation(rewriter, op.getLoc(), "tfl.mean",
        TypeRange{statisticType}, ValueRange{squared, axes}, keepDims)
                           ->getResult(0);
    Value epsilon = createF32ScalarTensorConstant(
        rewriter, op.getLoc(), op.getEpsilon().convertToFloat());
    Value varianceEpsilon = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
        TypeRange{statisticType}, ValueRange{meanSquare, epsilon}, fusedNone)
                                ->getResult(0);
    Value inverseRootMeanSquare = createTFLOperation(rewriter, op.getLoc(),
        "tfl.rsqrt", TypeRange{statisticType}, ValueRange{varianceEpsilon})
                                      ->getResult(0);
    Value normalized = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType}, ValueRange{input, inverseRootMeanSquare},
        fusedNone)
                           ->getResult(0);

    Value scale = adaptor.getScale();
    Value bias = hasBias ? adaptor.getB() : Value();
    if (rank == 4) {
      SmallVector<int64_t> parameterShape(4, 1);
      parameterShape[calculationAxis] = scaleType.getShape()[0];
      auto parameterType =
          RankedTensorType::get(parameterShape, rewriter.getF32Type());
      Value parameterShapeValue =
          createI32ShapeConstant(rewriter, op.getLoc(), parameterShape);
      scale = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{parameterType}, ValueRange{scale, parameterShapeValue})
                  ->getResult(0);
      if (biasMatchesScale)
        bias = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
            TypeRange{parameterType}, ValueRange{bias, parameterShapeValue})
                   ->getResult(0);
    }
    Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.mul",
        TypeRange{calculationType}, ValueRange{normalized, scale}, fusedNone)
                       ->getResult(0);
    if (hasBias)
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.add",
          TypeRange{calculationType}, ValueRange{result, bias}, fusedNone)
                   ->getResult(0);

    SmallVector<NamedAttribute> noValueAttributes{
        rewriter.getNamedAttr("value", rewriter.getUnitAttr())};
    Value noValue = createTFLOperation(rewriter, op.getLoc(), "tfl.no_value",
        TypeRange{rewriter.getNoneType()}, {}, noValueAttributes)
                        ->getResult(0);
    rewriter.replaceOp(op, ValueRange{result, noValue});
    return success();
  }
};

} // namespace

void populateLoweringONNXLayerNormalizationOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<LayerNormalizationLowering, RMSLayerNormalizationLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
