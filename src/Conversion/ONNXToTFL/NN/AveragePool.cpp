/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>
#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

SmallVector<int64_t> getAveragePoolArrayOr(
    Operation *op, StringRef name, ArrayRef<int64_t> defaultValues) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaultValues);
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute element : attr)
    values.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  return values;
}

int64_t getPhysicalAxis(int64_t rank, int64_t logicalAxis) {
  return rank == 4 ? mapNCHWAxisToNHWC(logicalAxis) : logicalAxis;
}

class AveragePoolLowering final
    : public OpConversionPattern<ONNXAveragePoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXAveragePoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || inputType.getRank() < 3 ||
        inputType.getRank() > 5 ||
        resultType.getRank() != inputType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "AveragePool input")) ||
        failed(validateStaticF32Tensor(op, resultType, "AveragePool result"))) {
      op.emitError("ONNXToTFL AveragePool supports static rank-3 through "
                   "rank-5 f32 tensors only");
      return failure();
    }

    int64_t rank = inputType.getRank();
    int64_t spatialRank = rank - 2;
    SmallVector<int64_t> defaultStrides(spatialRank, 1);
    SmallVector<int64_t> defaultPads(2 * spatialRank, 0);
    SmallVector<int64_t> defaultDilations(spatialRank, 1);
    SmallVector<int64_t> kernel = getAveragePoolArrayOr(op, "kernel_shape", {});
    SmallVector<int64_t> strides =
        getAveragePoolArrayOr(op, "strides", defaultStrides);
    SmallVector<int64_t> pads = getAveragePoolArrayOr(op, "pads", defaultPads);
    SmallVector<int64_t> dilations =
        getAveragePoolArrayOr(op, "dilations", defaultDilations);
    if (kernel.size() != static_cast<size_t>(spatialRank) ||
        strides.size() != static_cast<size_t>(spatialRank) ||
        pads.size() != static_cast<size_t>(2 * spatialRank) ||
        dilations.size() != static_cast<size_t>(spatialRank) ||
        llvm::any_of(kernel, [](int64_t value) { return value <= 0; }) ||
        llvm::any_of(strides, [](int64_t value) { return value <= 0; }) ||
        llvm::any_of(pads, [](int64_t value) { return value < 0; }) ||
        llvm::any_of(dilations, [](int64_t value) { return value != 1; })) {
      op.emitError("unsupported AveragePool attributes: expected positive "
                   "kernel/strides, nonnegative pads, and dilation=1");
      return failure();
    }
    StringRef autoPad = op.getAutoPad();
    if ((autoPad != "NOTSET" && autoPad != "VALID") ||
        (autoPad == "VALID" &&
            llvm::any_of(pads, [](int64_t value) { return value != 0; }))) {
      op.emitError() << "unsupported AveragePool auto_pad=" << autoPad;
      return failure();
    }
    if ((op.getCeilMode() != 0 && op.getCeilMode() != 1) ||
        (op.getCountIncludePad() != 0 && op.getCountIncludePad() != 1)) {
      op.emitError(
          "AveragePool ceil_mode and count_include_pad must be 0 or 1");
      return failure();
    }

    auto physicalInputType =
        dyn_cast<RankedTensorType>(adaptor.getX().getType());
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    if (!physicalInputType || physicalInputType.getRank() != rank ||
        inputType.getShape()[0] != resultType.getShape()[0] ||
        inputType.getShape()[1] != resultType.getShape()[1]) {
      op.emitError("AveragePool input/result batch or channel shape mismatch");
      return failure();
    }

    // Keep the compact native path for ordinary unpadded 2D pooling. ONNX
    // include-pad pooling can also use the builtin exactly: materialize its
    // zero padding as real tensor elements, then run VALID AveragePool so the
    // full kernel size is used as the divisor at every output position.
    bool noExplicitPads =
        llvm::all_of(pads, [](int64_t value) { return value == 0; });
    auto nativeOutputMatches = [&](int64_t paddedHeight, int64_t paddedWidth) {
      if (kernel[0] > paddedHeight || kernel[1] > paddedWidth)
        return false;
      int64_t expectedHeight = (paddedHeight - kernel[0]) / strides[0] + 1;
      int64_t expectedWidth = (paddedWidth - kernel[1]) / strides[1] + 1;
      return expectedHeight == resultType.getShape()[2] &&
             expectedWidth == resultType.getShape()[3];
    };
    bool nativeUnpadded2D =
        rank == 4 && op.getCountIncludePad() == 0 && noExplicitPads &&
        nativeOutputMatches(inputType.getShape()[2], inputType.getShape()[3]);
    bool nativeIncludePad2D = false;
    if (rank == 4 && op.getCountIncludePad() == 1) {
      int64_t paddedHeight = inputType.getShape()[2] + pads[0] + pads[2];
      int64_t paddedWidth = inputType.getShape()[3] + pads[1] + pads[3];
      nativeIncludePad2D = nativeOutputMatches(paddedHeight, paddedWidth);
    }
    bool native2D = nativeUnpadded2D || nativeIncludePad2D;
    if (native2D) {
      Value poolInput = adaptor.getX();
      if (nativeIncludePad2D && !noExplicitPads) {
        ArrayRef<int64_t> shape = physicalInputType.getShape();
        auto paddedType =
            RankedTensorType::get({shape[0], shape[1] + pads[0] + pads[2],
                                      shape[2] + pads[1] + pads[3], shape[3]},
                rewriter.getF32Type());
        auto paddingType = RankedTensorType::get({4, 2}, rewriter.getI32Type());
        SmallVector<int32_t> paddingValues{0, 0, static_cast<int32_t>(pads[0]),
            static_cast<int32_t>(pads[2]), static_cast<int32_t>(pads[1]),
            static_cast<int32_t>(pads[3]), 0, 0};
        Value paddingValue =
            arith::ConstantOp::create(rewriter, op.getLoc(), paddingType,
                DenseIntElementsAttr::get(
                    paddingType, ArrayRef<int32_t>(paddingValues)));
        poolInput = createTFLOperation(rewriter, op.getLoc(), "tfl.pad",
            TypeRange{paddedType}, ValueRange{poolInput, paddingValue})
                        ->getResult(0);
      }
      SmallVector<NamedAttribute> attributes{
          rewriter.getNamedAttr(
              "filter_height", rewriter.getI32IntegerAttr(kernel[0])),
          rewriter.getNamedAttr(
              "filter_width", rewriter.getI32IntegerAttr(kernel[1])),
          rewriter.getNamedAttr("padding", rewriter.getStringAttr("VALID")),
          rewriter.getNamedAttr(
              "stride_h", rewriter.getI32IntegerAttr(strides[0])),
          rewriter.getNamedAttr(
              "stride_w", rewriter.getI32IntegerAttr(strides[1])),
          getFusedActivationNone(rewriter)};
      Operation *pool =
          createTFLOperation(rewriter, op.getLoc(), "tfl.average_pool_2d",
              TypeRange{physicalResultType}, ValueRange{poolInput}, attributes);
      rewriter.replaceOp(op, pool->getResults());
      return success();
    }

    auto fitsI32 = [](int64_t value) {
      return value >= 0 && value <= std::numeric_limits<int32_t>::max();
    };
    if (!llvm::all_of(inputType.getShape(), fitsI32) ||
        !llvm::all_of(resultType.getShape(), fitsI32) ||
        !llvm::all_of(kernel, fitsI32) || !llvm::all_of(strides, fitsI32) ||
        !llvm::all_of(pads, fitsI32)) {
      op.emitError("AveragePool static-window parameters exceed the TFLite "
                   "int32 range");
      return failure();
    }

    SmallVector<int64_t> outputSpatialShape;
    int64_t windowCount = 1;
    int64_t kernelElementCount = 1;
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t outputDim = resultType.getShape()[2 + i];
      if (outputDim <= 0 || windowCount > 4096 / outputDim ||
          kernelElementCount >
              std::numeric_limits<int32_t>::max() / kernel[i]) {
        op.emitError("AveragePool static-window lowering is limited to 4096 "
                     "output windows and an int32 kernel element count");
        return failure();
      }
      outputSpatialShape.push_back(outputDim);
      windowCount *= outputDim;
      kernelElementCount *= kernel[i];
    }

    Location loc = op.getLoc();
    SmallVector<int64_t> reductionAxes;
    SmallVector<int64_t> cellShape(physicalResultType.getShape());
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t physicalAxis = getPhysicalAxis(rank, 2 + i);
      reductionAxes.push_back(physicalAxis);
      cellShape[physicalAxis] = 1;
    }
    Value reductionAxesValue =
        createI32ShapeConstant(rewriter, loc, reductionAxes);
    SmallVector<NamedAttribute> meanAttributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    SmallVector<Value> pieces;
    pieces.reserve(windowCount);

    for (int64_t flatIndex = 0; flatIndex < windowCount; ++flatIndex) {
      SmallVector<int64_t> outputIndices(spatialRank);
      int64_t remainder = flatIndex;
      for (int64_t i = spatialRank - 1; i >= 0; --i) {
        outputIndices[i] = remainder % outputSpatialShape[i];
        remainder /= outputSpatialShape[i];
      }

      SmallVector<int64_t> begin(rank, 0);
      SmallVector<int64_t> size(physicalInputType.getShape());
      int64_t validElementCount = 1;
      for (int64_t i = 0; i < spatialRank; ++i) {
        int64_t inputDim = inputType.getShape()[2 + i];
        int64_t start = outputIndices[i] * strides[i] - pads[i];
        int64_t stop = start + kernel[i];
        int64_t validStart = std::clamp(start, int64_t{0}, inputDim);
        int64_t validStop = std::clamp(stop, int64_t{0}, inputDim);
        int64_t validSize = std::max<int64_t>(validStop - validStart, 0);
        if (validSize == 0) {
          op.emitError("AveragePool output window contains no input elements");
          return failure();
        }
        int64_t physicalAxis = getPhysicalAxis(rank, 2 + i);
        begin[physicalAxis] = validStart;
        size[physicalAxis] = validSize;
        validElementCount *= validSize;
      }

      auto sliceType = RankedTensorType::get(size, rewriter.getF32Type());
      Value beginValue = createI32ShapeConstant(rewriter, loc, begin);
      Value sizeValue = createI32ShapeConstant(rewriter, loc, size);
      Value slice =
          createTFLOperation(rewriter, loc, "tfl.slice", TypeRange{sliceType},
              ValueRange{adaptor.getX(), beginValue, sizeValue})
              ->getResult(0);
      auto cellType = RankedTensorType::get(cellShape, rewriter.getF32Type());
      Value mean =
          createTFLOperation(rewriter, loc, "tfl.mean", TypeRange{cellType},
              ValueRange{slice, reductionAxesValue}, meanAttributes)
              ->getResult(0);
      if (op.getCountIncludePad() != 0 &&
          validElementCount != kernelElementCount) {
        float scale = static_cast<float>(validElementCount) /
                      static_cast<float>(kernelElementCount);
        Value scaleValue = createF32ScalarTensorConstant(rewriter, loc, scale);
        SmallVector<NamedAttribute> mulAttributes{
            getFusedActivationNone(rewriter)};
        mean = createTFLOperation(rewriter, loc, "tfl.mul", TypeRange{cellType},
            ValueRange{mean, scaleValue}, mulAttributes)
                   ->getResult(0);
      }
      pieces.push_back(mean);
    }

    SmallVector<int64_t> pieceShape(cellShape);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      int64_t extent = outputSpatialShape[i];
      int64_t physicalAxis = getPhysicalAxis(rank, 2 + i);
      SmallVector<int64_t> nextShape(pieceShape);
      nextShape[physicalAxis] = extent;
      auto nextType = RankedTensorType::get(nextShape, rewriter.getF32Type());
      SmallVector<Value> nextPieces;
      nextPieces.reserve(pieces.size() / extent);
      for (int64_t offset = 0, e = pieces.size(); offset < e;
           offset += extent) {
        if (extent == 1) {
          nextPieces.push_back(pieces[offset]);
          continue;
        }
        SmallVector<NamedAttribute> concatAttributes{
            rewriter.getNamedAttr(
                "axis", rewriter.getI32IntegerAttr(physicalAxis)),
            getFusedActivationNone(rewriter)};
        Value concatenated = createTFLOperation(rewriter, loc,
            "tfl.concatenation", TypeRange{nextType},
            ValueRange(ArrayRef<Value>(pieces).slice(offset, extent)),
            concatAttributes)
                                 ->getResult(0);
        nextPieces.push_back(concatenated);
      }
      pieces = std::move(nextPieces);
      pieceShape = std::move(nextShape);
    }
    if (pieces.size() != 1 ||
        !llvm::equal(pieceShape, physicalResultType.getShape())) {
      op.emitError("failed to assemble static AveragePool output shape");
      return failure();
    }
    rewriter.replaceOp(op, pieces.front());
    return success();
  }
};

} // namespace

void populateLoweringONNXAveragePoolOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<AveragePoolLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
