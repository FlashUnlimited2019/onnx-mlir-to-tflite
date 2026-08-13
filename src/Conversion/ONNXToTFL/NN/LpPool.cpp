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

SmallVector<int64_t> getLpPoolArrayOr(
    Operation *op, StringRef name, ArrayRef<int64_t> defaults) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return SmallVector<int64_t>(defaults);
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute element : attr)
    values.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  return values;
}

int64_t getLpPoolPhysicalAxis(int64_t rank, int64_t logicalAxis) {
  return rank == 4 ? mapNCHWAxisToNHWC(logicalAxis) : logicalAxis;
}

Value square(ConversionPatternRewriter &rewriter, Location loc, Value input,
    Type resultType) {
  SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
  return createTFLOperation(rewriter, loc, "tfl.mul", TypeRange{resultType},
      ValueRange{input, input}, attributes)
      ->getResult(0);
}

class GlobalLpPoolLowering final
    : public OpConversionPattern<ONNXGlobalLpPoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGlobalLpPoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || inputType.getRank() < 3 ||
        inputType.getRank() > 5 ||
        resultType.getRank() != inputType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "GlobalLpPool input")) ||
        failed(
            validateStaticF32Tensor(op, resultType, "GlobalLpPool result")) ||
        op.getP() != 2)
      return op.emitError("ONNXToTFL GlobalLpPool supports p=2 on static "
                          "rank-3 through rank-5 f32 tensors only"),
             failure();
    if (inputType.getShape()[0] != resultType.getShape()[0] ||
        inputType.getShape()[1] != resultType.getShape()[1] ||
        llvm::any_of(resultType.getShape().drop_front(2),
            [](int64_t dim) { return dim != 1; }))
      return op.emitError("GlobalLpPool result shape must preserve N/C and "
                          "reduce every spatial dimension to one"),
             failure();

    Location loc = op.getLoc();
    auto physicalInputType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(inputType));
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    Value squared = square(rewriter, loc, adaptor.getX(), physicalInputType);
    SmallVector<int64_t> axes;
    for (int64_t logicalAxis = 2; logicalAxis < inputType.getRank();
         ++logicalAxis)
      axes.push_back(getLpPoolPhysicalAxis(inputType.getRank(), logicalAxis));
    Value axesValue = createI32ShapeConstant(rewriter, loc, axes);
    SmallVector<NamedAttribute> sumAttributes{
        rewriter.getNamedAttr("keep_dims", rewriter.getBoolAttr(true))};
    Value sum = createTFLOperation(rewriter, loc, "tfl.sum",
        TypeRange{physicalResultType}, ValueRange{squared, axesValue},
        sumAttributes)
                    ->getResult(0);
    Value result = createTFLOperation(rewriter, loc, "tfl.sqrt",
        TypeRange{physicalResultType}, ValueRange{sum})
                       ->getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class LpPoolLowering final : public OpConversionPattern<ONNXLpPoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXLpPoolOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getX().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!inputType || !resultType || inputType.getRank() < 3 ||
        inputType.getRank() > 5 ||
        resultType.getRank() != inputType.getRank() ||
        failed(validateStaticF32Tensor(op, inputType, "LpPool input")) ||
        failed(validateStaticF32Tensor(op, resultType, "LpPool result")) ||
        op.getP() != 2)
      return op.emitError("ONNXToTFL LpPool supports p=2 on static rank-3 "
                          "through rank-5 f32 tensors only"),
             failure();

    int64_t rank = inputType.getRank();
    int64_t spatialRank = rank - 2;
    SmallVector<int64_t> kernel = getLpPoolArrayOr(op, "kernel_shape", {});
    SmallVector<int64_t> strides =
        getLpPoolArrayOr(op, "strides", SmallVector<int64_t>(spatialRank, 1));
    SmallVector<int64_t> pads =
        getLpPoolArrayOr(op, "pads", SmallVector<int64_t>(2 * spatialRank, 0));
    SmallVector<int64_t> dilations =
        getLpPoolArrayOr(op, "dilations", SmallVector<int64_t>(spatialRank, 1));
    if (kernel.size() != static_cast<size_t>(spatialRank) ||
        strides.size() != static_cast<size_t>(spatialRank) ||
        pads.size() != static_cast<size_t>(2 * spatialRank) ||
        dilations.size() != static_cast<size_t>(spatialRank) ||
        llvm::any_of(kernel, [](int64_t value) { return value <= 0; }) ||
        llvm::any_of(strides, [](int64_t value) { return value <= 0; }) ||
        llvm::any_of(pads, [](int64_t value) { return value < 0; }) ||
        llvm::any_of(dilations, [](int64_t value) { return value != 1; }))
      return op.emitError("unsupported LpPool attributes: expected positive "
                          "kernel/strides, nonnegative pads, dilation=1"),
             failure();
    if (op.getAutoPad() != "NOTSET" && op.getAutoPad() != "VALID")
      return op.emitError("ONNXToTFL LpPool supports auto_pad NOTSET/VALID "
                          "only"),
             failure();
    if (op.getCeilMode() != 0 && op.getCeilMode() != 1)
      return op.emitError("LpPool ceil_mode must be 0 or 1"), failure();
    if (inputType.getShape()[0] != resultType.getShape()[0] ||
        inputType.getShape()[1] != resultType.getShape()[1])
      return op.emitError("LpPool input/result batch or channel mismatch"),
             failure();

    auto fitsI32 = [](int64_t value) {
      return value >= 0 && value <= std::numeric_limits<int32_t>::max();
    };
    if (!llvm::all_of(inputType.getShape(), fitsI32) ||
        !llvm::all_of(resultType.getShape(), fitsI32) ||
        !llvm::all_of(kernel, fitsI32) || !llvm::all_of(strides, fitsI32) ||
        !llvm::all_of(pads, fitsI32))
      return op.emitError("LpPool static-window parameters exceed int32"),
             failure();

    SmallVector<int64_t> outputSpatialShape;
    int64_t windowCount = 1;
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t outputDim = resultType.getShape()[2 + i];
      if (outputDim <= 0 || windowCount > 4096 / outputDim)
        return op.emitError("LpPool static-window lowering is limited to "
                            "4096 output windows"),
               failure();
      outputSpatialShape.push_back(outputDim);
      windowCount *= outputDim;
    }

    Location loc = op.getLoc();
    auto physicalInputType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(inputType));
    auto physicalResultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(resultType));
    Value squared = square(rewriter, loc, adaptor.getX(), physicalInputType);
    SmallVector<int64_t> axes;
    SmallVector<int64_t> cellShape(physicalResultType.getShape());
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t physicalAxis = getLpPoolPhysicalAxis(rank, 2 + i);
      axes.push_back(physicalAxis);
      cellShape[physicalAxis] = 1;
    }
    Value axesValue = createI32ShapeConstant(rewriter, loc, axes);
    SmallVector<NamedAttribute> sumAttributes{
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
      for (int64_t i = 0; i < spatialRank; ++i) {
        int64_t inputDim = inputType.getShape()[2 + i];
        int64_t start = outputIndices[i] * strides[i] - pads[i];
        int64_t stop = start + kernel[i];
        int64_t validStart = std::clamp(start, int64_t{0}, inputDim);
        int64_t validStop = std::clamp(stop, int64_t{0}, inputDim);
        int64_t validSize = std::max<int64_t>(validStop - validStart, 0);
        if (validSize == 0)
          return op.emitError("LpPool output window contains no input "
                              "elements"),
                 failure();
        int64_t physicalAxis = getLpPoolPhysicalAxis(rank, 2 + i);
        begin[physicalAxis] = validStart;
        size[physicalAxis] = validSize;
      }

      auto sliceType = RankedTensorType::get(size, rewriter.getF32Type());
      Value beginValue = createI32ShapeConstant(rewriter, loc, begin);
      Value sizeValue = createI32ShapeConstant(rewriter, loc, size);
      Value slice = createTFLOperation(rewriter, loc, "tfl.slice",
          TypeRange{sliceType}, ValueRange{squared, beginValue, sizeValue})
                        ->getResult(0);
      auto cellType = RankedTensorType::get(cellShape, rewriter.getF32Type());
      Value sum = createTFLOperation(rewriter, loc, "tfl.sum",
          TypeRange{cellType}, ValueRange{slice, axesValue}, sumAttributes)
                      ->getResult(0);
      pieces.push_back(createTFLOperation(
          rewriter, loc, "tfl.sqrt", TypeRange{cellType}, ValueRange{sum})
                           ->getResult(0));
    }

    SmallVector<int64_t> pieceShape(cellShape);
    for (int64_t i = spatialRank - 1; i >= 0; --i) {
      int64_t extent = outputSpatialShape[i];
      int64_t physicalAxis = getLpPoolPhysicalAxis(rank, 2 + i);
      SmallVector<int64_t> nextShape(pieceShape);
      nextShape[physicalAxis] = extent;
      auto nextType = RankedTensorType::get(nextShape, rewriter.getF32Type());
      SmallVector<Value> nextPieces;
      nextPieces.reserve(pieces.size() / extent);
      for (int64_t offset = 0, end = pieces.size(); offset < end;
           offset += extent) {
        if (extent == 1) {
          nextPieces.push_back(pieces[offset]);
          continue;
        }
        SmallVector<NamedAttribute> attributes{
            rewriter.getNamedAttr(
                "axis", rewriter.getI32IntegerAttr(physicalAxis)),
            getFusedActivationNone(rewriter)};
        nextPieces.push_back(createTFLOperation(rewriter, loc,
            "tfl.concatenation", TypeRange{nextType},
            ValueRange(ArrayRef<Value>(pieces).slice(offset, extent)),
            attributes)
                                 ->getResult(0));
      }
      pieces = std::move(nextPieces);
      pieceShape = std::move(nextShape);
    }
    if (pieces.size() != 1 ||
        !llvm::equal(pieceShape, physicalResultType.getShape()))
      return op.emitError("failed to assemble static LpPool result"), failure();
    rewriter.replaceOp(op, pieces.front());
    return success();
  }
};

} // namespace

void populateLoweringONNXLpPoolOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<LpPoolLowering, GlobalLpPoolLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
