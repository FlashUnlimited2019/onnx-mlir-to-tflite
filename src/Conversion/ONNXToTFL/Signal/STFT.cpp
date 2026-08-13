/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <cmath>

using namespace mlir;

namespace onnx_mlir {
namespace {

class STFTLowering final : public OpConversionPattern<ONNXSTFTOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXSTFTOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto signalType = dyn_cast<RankedTensorType>(op.getSignal().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    bool supportedSignalRank =
        signalType &&
        (signalType.getRank() == 2 ||
            (signalType.getRank() == 3 && signalType.getShape().back() == 1));
    if (!supportedSignalRank || !resultType || resultType.getRank() != 4 ||
        failed(validateStaticF32Tensor(op, signalType, "STFT signal")) ||
        failed(validateStaticF32Tensor(op, resultType, "STFT result")))
      return op.emitError("ONNXToTFL STFT requires a static rank-2 real FP32 "
                          "signal (or rank-3 with a final size-1 component) "
                          "and rank-4 FP32 result"),
             failure();

    FailureOr<SmallVector<int64_t>> frameStepValues =
        getConstantIntValues(op.getFrameStep());
    FailureOr<SmallVector<int64_t>> frameLengthValues =
        getConstantIntValues(op.getFrameLength());
    if (failed(frameStepValues) || frameStepValues->size() != 1 ||
        failed(frameLengthValues) || frameLengthValues->size() != 1)
      return op.emitError(
                 "ONNXToTFL STFT requires constant frame_step/frame_length"),
             failure();
    int64_t frameStep = frameStepValues->front();
    int64_t frameLength = frameLengthValues->front();
    int64_t batch = signalType.getShape()[0];
    int64_t signalLength = signalType.getShape()[1];
    if (frameStep <= 0 || frameLength <= 0 || frameLength > signalLength)
      return op.emitError("ONNXToTFL STFT frame configuration is invalid"),
             failure();
    int64_t frameCount = (signalLength - frameLength) / frameStep + 1;
    int64_t oneSidedBins = frameLength / 2 + 1;
    if (op.getOnesided() != 0 && op.getOnesided() != 1)
      return op.emitError("ONNXToTFL STFT onesided must be 0 or 1"), failure();
    int64_t frequencyBins = op.getOnesided() != 0 ? oneSidedBins : frameLength;
    SmallVector<int64_t> expectedShape{batch, frameCount, frequencyBins, 2};
    if (!llvm::equal(expectedShape, resultType.getShape()))
      return op.emitError("ONNXToTFL STFT result shape is inconsistent"),
             failure();

    Location loc = op.getLoc();
    Value signal = adaptor.getSignal();
    if (signalType.getRank() == 3) {
      auto realSignalType =
          RankedTensorType::get({batch, signalLength}, rewriter.getF32Type());
      Value realSignalShape =
          createI32ShapeConstant(rewriter, loc, {batch, signalLength});
      signal = createTFLOperation(rewriter, loc, "tfl.reshape",
          TypeRange{realSignalType}, ValueRange{signal, realSignalShape})
                   ->getResult(0);
    }
    Value window = adaptor.getWindow();
    if (isa<NoneType>(window.getType())) {
      auto windowType =
          RankedTensorType::get({frameLength}, rewriter.getF32Type());
      window = arith::ConstantOp::create(
          rewriter, loc, windowType, DenseElementsAttr::get(windowType, 1.0f));
    } else {
      auto windowType = dyn_cast<RankedTensorType>(op.getWindow().getType());
      if (!windowType || !windowType.hasStaticShape() ||
          !windowType.getElementType().isF32() || windowType.getRank() != 1 ||
          windowType.getShape()[0] != frameLength)
        return op.emitError(
                   "ONNXToTFL STFT window must be static rank-1 FP32 and "
                   "match frame_length"),
               failure();
    }

    auto createReshape = [&](Value input, ArrayRef<int64_t> shape) -> Value {
      auto type = RankedTensorType::get(
          shape, cast<ShapedType>(input.getType()).getElementType());
      Value shapeValue = createI32ShapeConstant(rewriter, loc, shape);
      return createTFLOperation(rewriter, loc, "tfl.reshape", TypeRange{type},
          ValueRange{input, shapeValue})
          ->getResult(0);
    };
    auto concatenate = [&](ValueRange inputs, ArrayRef<int64_t> shape,
                           int32_t axis) -> Value {
      auto type = RankedTensorType::get(shape, rewriter.getF32Type());
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(axis)),
          getFusedActivationNone(rewriter)};
      return createTFLOperation(rewriter, loc, "tfl.concatenation",
          TypeRange{type}, inputs, attributes)
          ->getResult(0);
    };

    auto frameType =
        RankedTensorType::get({batch, frameLength}, rewriter.getF32Type());
    SmallVector<Value> frames;
    frames.reserve(frameCount);
    for (int64_t frame = 0; frame < frameCount; ++frame) {
      Value begin =
          createI32ShapeConstant(rewriter, loc, {0, frame * frameStep});
      Value size = createI32ShapeConstant(rewriter, loc, {batch, frameLength});
      Value slice = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{frameType}, ValueRange{signal, begin, size})
                        ->getResult(0);
      SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
      Value windowed = createTFLOperation(rewriter, loc, "tfl.mul",
          TypeRange{frameType}, ValueRange{slice, window}, attributes)
                           ->getResult(0);
      frames.push_back(createReshape(windowed, {batch, 1, frameLength}));
    }
    Value framed = frames.front();
    if (frames.size() != 1)
      framed =
          concatenate(frames, {batch, frameCount, frameLength}, /*axis=*/1);
    Value real;
    Value imag;
    bool powerOfTwo = (frameLength & (frameLength - 1)) == 0;
    if (powerOfTwo) {
      Value flattened =
          createReshape(framed, {batch * frameCount, 1, frameLength});
      auto complexType = ComplexType::get(rewriter.getF32Type());
      auto fftType = RankedTensorType::get(
          {batch * frameCount, 1, oneSidedBins}, complexType);
      Value fftLength = createI32ShapeConstant(rewriter, loc, {1, frameLength});
      Value fft = createTFLOperation(rewriter, loc, "tfl.rfft2d",
          TypeRange{fftType}, ValueRange{flattened, fftLength})
                      ->getResult(0);
      auto fftComponentType = RankedTensorType::get(
          {batch * frameCount, 1, oneSidedBins}, rewriter.getF32Type());
      real = createTFLOperation(rewriter, loc, "tfl.real",
          TypeRange{fftComponentType}, ValueRange{fft})
                 ->getResult(0);
      imag = createTFLOperation(rewriter, loc, "tfl.imag",
          TypeRange{fftComponentType}, ValueRange{fft})
                 ->getResult(0);
      real = createReshape(real, {batch, frameCount, oneSidedBins});
      imag = createReshape(imag, {batch, frameCount, oneSidedBins});

      if (op.getOnesided() == 0) {
        int64_t tailSize = frameLength - oneSidedBins;
        auto tailType = RankedTensorType::get(
            {batch, frameCount, tailSize}, rewriter.getF32Type());
        Value begin = createI32ShapeConstant(rewriter, loc, {0, 0, 1});
        Value size = createI32ShapeConstant(
            rewriter, loc, {batch, frameCount, tailSize});
        Value realTail = createTFLOperation(rewriter, loc, "tfl.slice",
            TypeRange{tailType}, ValueRange{real, begin, size})
                             ->getResult(0);
        Value imagTail = createTFLOperation(rewriter, loc, "tfl.slice",
            TypeRange{tailType}, ValueRange{imag, begin, size})
                             ->getResult(0);
        Value reverseAxis = createI32ShapeConstant(rewriter, loc, {2});
        realTail = createTFLOperation(rewriter, loc, "tfl.reverse_v2",
            TypeRange{tailType}, ValueRange{realTail, reverseAxis})
                       ->getResult(0);
        imagTail = createTFLOperation(rewriter, loc, "tfl.reverse_v2",
            TypeRange{tailType}, ValueRange{imagTail, reverseAxis})
                       ->getResult(0);
        imagTail = createTFLOperation(
            rewriter, loc, "tfl.neg", TypeRange{tailType}, ValueRange{imagTail})
                       ->getResult(0);
        real = concatenate(
            ValueRange{real, realTail}, {batch, frameCount, frameLength}, 2);
        imag = concatenate(
            ValueRange{imag, imagTail}, {batch, frameCount, frameLength}, 2);
      }
    } else {
      // TFLite RFFT2D accepts only power-of-two lengths. Use a static real
      // DFT matrix for other lengths so the exported graph remains builtin.
      constexpr double pi = 3.14159265358979323846264338327950288;
      SmallVector<float> realCoefficients;
      SmallVector<float> imagCoefficients;
      realCoefficients.reserve(frameLength * frequencyBins);
      imagCoefficients.reserve(frameLength * frequencyBins);
      for (int64_t sample = 0; sample < frameLength; ++sample) {
        for (int64_t frequency = 0; frequency < frequencyBins; ++frequency) {
          double angle = -2.0 * pi * static_cast<double>(sample) *
                         static_cast<double>(frequency) /
                         static_cast<double>(frameLength);
          realCoefficients.push_back(static_cast<float>(std::cos(angle)));
          imagCoefficients.push_back(static_cast<float>(std::sin(angle)));
        }
      }
      auto coefficientType = RankedTensorType::get(
          {frameLength, frequencyBins}, rewriter.getF32Type());
      Value realWeights =
          arith::ConstantOp::create(rewriter, loc, coefficientType,
              DenseFPElementsAttr::get(coefficientType, realCoefficients));
      Value imagWeights =
          arith::ConstantOp::create(rewriter, loc, coefficientType,
              DenseFPElementsAttr::get(coefficientType, imagCoefficients));
      Value flattened =
          createReshape(framed, {batch * frameCount, frameLength});
      auto flatResultType = RankedTensorType::get(
          {batch * frameCount, frequencyBins}, rewriter.getF32Type());
      SmallVector<NamedAttribute> matmulAttributes{
          rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(false)),
          rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(false))};
      real = createTFLOperation(rewriter, loc, "tfl.batch_matmul",
          TypeRange{flatResultType}, ValueRange{flattened, realWeights},
          matmulAttributes)
                 ->getResult(0);
      imag = createTFLOperation(rewriter, loc, "tfl.batch_matmul",
          TypeRange{flatResultType}, ValueRange{flattened, imagWeights},
          matmulAttributes)
                 ->getResult(0);
      real = createReshape(real, {batch, frameCount, frequencyBins});
      imag = createReshape(imag, {batch, frameCount, frequencyBins});
    }

    SmallVector<NamedAttribute> packAttributes{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(3)),
        rewriter.getNamedAttr("values_count", rewriter.getI32IntegerAttr(2))};
    Value logicalResult = createTFLOperation(rewriter, loc, "tfl.pack",
        TypeRange{resultType}, ValueRange{real, imag}, packAttributes)
                              ->getResult(0);
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    Value permutation = createI32ShapeConstant(rewriter, loc, {0, 2, 3, 1});
    Value physicalResult = createTFLOperation(rewriter, loc, "tfl.transpose",
        TypeRange{physicalResultType}, ValueRange{logicalResult, permutation})
                               ->getResult(0);
    rewriter.replaceOp(op, physicalResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXSTFTOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<STFTLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
