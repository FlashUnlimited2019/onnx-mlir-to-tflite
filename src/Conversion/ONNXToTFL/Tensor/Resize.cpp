/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <algorithm>
#include <cmath>

using namespace mlir;

namespace onnx_mlir {
namespace {

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
    auto sourceType = cast<RankedTensorType>(op->getOperand(0).getType());
    auto sourceResultType = cast<RankedTensorType>(op->getResult(0).getType());
    if (sourceType.getRank() != sourceResultType.getRank() ||
        sourceType.getRank() < 1 || sourceType.getRank() > 5) {
      op.emitError("ONNXToTFL Resize supports matching static rank-1 through "
                   "rank-5 tensors");
      return failure();
    }
    if (adaptor.getAntialias() != 0 || adaptor.getAxes().has_value()) {
      op.emitError("unsupported Resize configuration: antialias or axes");
      return failure();
    }
    ArrayRef<int64_t> inputShape = sourceType.getShape();
    ArrayRef<int64_t> outputShape = sourceResultType.getShape();
    if (sourceType.getRank() != 4) {
      bool halfPixel =
          adaptor.getCoordinateTransformationMode() == "half_pixel";
      bool asymmetric =
          adaptor.getCoordinateTransformationMode() == "asymmetric";
      bool roundPreferFloor = adaptor.getNearestMode() == "round_prefer_floor";
      bool floorMode = adaptor.getNearestMode() == "floor";
      if (adaptor.getMode() != "nearest" ||
          !((halfPixel && roundPreferFloor) || (asymmetric && floorMode)))
        return op.emitError("non-2D Resize supports nearest/half_pixel/"
                            "round_prefer_floor or nearest/asymmetric/floor"),
               failure();

      Value result = input;
      SmallVector<int64_t> currentShape(inputShape);
      for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
        int64_t inputExtent = inputShape[axis];
        int64_t outputExtent = outputShape[axis];
        if (inputExtent <= 0 || outputExtent <= 0)
          return op.emitError("non-2D Resize requires positive static "
                              "dimensions"),
                 failure();
        if (inputExtent == outputExtent)
          continue;

        double scale = static_cast<double>(outputExtent) /
                       static_cast<double>(inputExtent);
        SmallVector<int32_t> indices;
        indices.reserve(outputExtent);
        for (int64_t outputIndex = 0; outputIndex < outputExtent;
             ++outputIndex) {
          double coordinate =
              halfPixel ? (static_cast<double>(outputIndex) + 0.5) / scale - 0.5
                        : static_cast<double>(outputIndex) / scale;
          int64_t index =
              roundPreferFloor
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
      rewriter.replaceOp(op, result);
      return success();
    }
    if (inputShape[0] != outputShape[0] || inputShape[1] != outputShape[1]) {
      op.emitError("ONNXToTFL Resize only supports resizing H and W");
      return failure();
    }

    // TFLite's half_pixel_centers nearest kernel floors the transformed
    // coordinate. ONNX round_prefer_floor instead rounds to nearest (choosing
    // the lower index only at ties), which differs notably for downsampling.
    // Materialize the exact static ONNX H/W sample indices in logical NCHW.
    if (adaptor.getMode() == "nearest" &&
        adaptor.getCoordinateTransformationMode() == "half_pixel" &&
        adaptor.getNearestMode() == "round_prefer_floor") {
      Value toLogical =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      Value result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceType}, ValueRange{input, toLogical})
                         ->getResult(0);
      SmallVector<int64_t> currentShape(inputShape);
      for (int64_t axis : {int64_t{2}, int64_t{3}}) {
        int64_t inputExtent = inputShape[axis];
        int64_t outputExtent = outputShape[axis];
        if (inputExtent == outputExtent)
          continue;
        double scale = static_cast<double>(outputExtent) /
                       static_cast<double>(inputExtent);
        SmallVector<int32_t> indices;
        indices.reserve(outputExtent);
        for (int64_t outputIndex = 0; outputIndex < outputExtent;
             ++outputIndex) {
          double coordinate =
              (static_cast<double>(outputIndex) + 0.5) / scale - 0.5;
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
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{physicalResultType}, ValueRange{result, toPhysical})
                   ->getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value size = createI32ShapeConstant(
        rewriter, op.getLoc(), {outputShape[2], outputShape[3]});
    auto resultType =
        cast<RankedTensorType>(convertRank4NCHWToNHWCType(sourceResultType));
    StringRef tflOpName;
    bool alignCorners = false;
    bool halfPixelCenters = false;
    if (adaptor.getMode() == "nearest" &&
        adaptor.getCoordinateTransformationMode() == "asymmetric" &&
        adaptor.getNearestMode() == "floor") {
      tflOpName = "tfl.resize_nearest_neighbor";
    } else if (adaptor.getMode() == "nearest" &&
               adaptor.getCoordinateTransformationMode() == "half_pixel" &&
               adaptor.getNearestMode() == "round_prefer_floor") {
      tflOpName = "tfl.resize_nearest_neighbor";
      halfPixelCenters = true;
    } else if (adaptor.getMode() == "linear" &&
               adaptor.getCoordinateTransformationMode() == "half_pixel") {
      tflOpName = "tfl.resize_bilinear";
      halfPixelCenters = true;
    } else if (adaptor.getMode() == "linear" &&
               adaptor.getCoordinateTransformationMode() == "align_corners") {
      tflOpName = "tfl.resize_bilinear";
      alignCorners = true;
    } else {
      op.emitError("unsupported Resize configuration: expected "
                   "nearest/asymmetric/floor, nearest/half_pixel/"
                   "round_prefer_floor, linear/half_pixel, or "
                   "linear/align_corners");
      return failure();
    }
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr(
            "align_corners", rewriter.getBoolAttr(alignCorners)),
        rewriter.getNamedAttr(
            "half_pixel_centers", rewriter.getBoolAttr(halfPixelCenters))};
    Operation *resize = createTFLOperation(rewriter, op.getLoc(), tflOpName,
        TypeRange{resultType}, ValueRange{input, size}, attributes);
    rewriter.replaceOp(op, resize->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXResizeOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ResizeLowering>(typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
