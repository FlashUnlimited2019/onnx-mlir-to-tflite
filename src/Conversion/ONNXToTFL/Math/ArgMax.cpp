/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

Value createI64ScalarTensorConstant(
    ConversionPatternRewriter &rewriter, Location loc, int64_t value) {
  auto type = RankedTensorType::get({}, rewriter.getI64Type());
  auto attr = DenseIntElementsAttr::get(type, ArrayRef<int64_t>{value});
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

template <typename ONNXOp>
class ArgMinMaxLowering final : public OpConversionPattern<ONNXOp> {
public:
  ArgMinMaxLowering(TypeConverter &typeConverter, MLIRContext *context,
      StringRef onnxName, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), onnxName(onnxName),
        tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getData();
    auto sourceInputType = dyn_cast<RankedTensorType>(op.getData().getType());
    auto sourceResultType = dyn_cast<RankedTensorType>(op.getType());
    bool supportedInputElementType =
        sourceInputType &&
        (sourceInputType.getElementType().isF32() ||
            sourceInputType.getElementType().isSignlessInteger(32));
    if (!sourceInputType || !sourceResultType ||
        !sourceInputType.hasStaticShape() || !supportedInputElementType ||
        !sourceResultType.hasStaticShape() ||
        !sourceResultType.getElementType().isSignlessInteger(64)) {
      op.emitError() << "ONNXToTFL " << onnxName
                     << " requires a static FP32 or i32 input and a static "
                        "i64 result tensor";
      return failure();
    }

    int64_t rank = sourceInputType.getRank();
    if (rank < 2 || rank > 5) {
      op.emitError() << "ONNXToTFL " << onnxName
                     << " supports input ranks 2 through 5";
      return failure();
    }
    int64_t axis = normalizeAxis(op.getAxis(), rank);
    if (axis < 0 || axis >= rank)
      return op.emitError()
                 << "invalid " << onnxName << " axis " << op.getAxis(),
             failure();
    bool keepDims = op.getKeepdims() != 0;
    if ((op.getKeepdims() != 0 && op.getKeepdims() != 1) ||
        (op.getSelectLastIndex() != 0 && op.getSelectLastIndex() != 1)) {
      op.emitError() << onnxName
                     << " keepdims and select_last_index must be 0 or 1";
      return failure();
    }

    SmallVector<int64_t> expectedShape(sourceInputType.getShape());
    if (keepDims)
      expectedShape[axis] = 1;
    else
      expectedShape.erase(expectedShape.begin() + axis);
    if (!llvm::equal(expectedShape, sourceResultType.getShape())) {
      op.emitError() << onnxName
                     << " inferred result shape does not match its input, "
                        "axis, and keepdims attributes";
      return failure();
    }

    // ArgMin/ArgMax axes use logical ONNX order. Rank-4 FP32 values are
    // physically NHWC in this bridge, so restore NCHW before reduction. The
    // integer result remains in logical order and the following Cast lowering
    // performs the layout transition when it produces rank-4 FP32.
    if (rank == 4 && sourceInputType.getElementType().isF32()) {
      Value permutation =
          createI32ShapeConstant(rewriter, op.getLoc(), {0, 3, 1, 2});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.transpose",
          TypeRange{sourceInputType}, ValueRange{input, permutation})
                  ->getResult(0);
    }

    if (op.getSelectLastIndex() != 0) {
      Value reverseAxis = createI32ShapeConstant(rewriter, op.getLoc(), {axis});
      input = createTFLOperation(rewriter, op.getLoc(), "tfl.reverse_v2",
          TypeRange{sourceInputType}, ValueRange{input, reverseAxis})
                  ->getResult(0);
    }

    SmallVector<int64_t> nativeResultShape(sourceInputType.getShape());
    nativeResultShape.erase(nativeResultShape.begin() + axis);
    auto nativeResultType = RankedTensorType::get(
        nativeResultShape, sourceResultType.getElementType());
    Value axisValue = createI32ScalarTensorConstant(
        rewriter, op.getLoc(), static_cast<int32_t>(axis));
    Value result = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{nativeResultType}, ValueRange{input, axisValue})
                       ->getResult(0);
    if (op.getSelectLastIndex() != 0) {
      Value finalIndex = createI64ScalarTensorConstant(
          rewriter, op.getLoc(), sourceInputType.getShape()[axis] - 1);
      SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.sub",
          TypeRange{nativeResultType}, ValueRange{finalIndex, result},
          attributes)
                   ->getResult(0);
    }
    // TFLite ArgMin/ArgMax always removes the reduced axis. Reinsert the
    // singleton dimension explicitly for ONNX keepdims=1 instead of relying on
    // static result metadata that the runtime kernel will overwrite during
    // Prepare.
    if (keepDims) {
      Value shape = createI32ShapeConstant(
          rewriter, op.getLoc(), sourceResultType.getShape());
      result = createTFLOperation(rewriter, op.getLoc(), "tfl.reshape",
          TypeRange{sourceResultType}, ValueRange{result, shape})
                   ->getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  std::string onnxName;
  std::string tflName;
};

} // namespace

void populateLoweringONNXArgMaxOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ArgMinMaxLowering<ONNXArgMaxOp>>(
      typeConverter, context, "ArgMax", "tfl.arg_max");
  patterns.add<ArgMinMaxLowering<ONNXArgMinOp>>(
      typeConverter, context, "ArgMin", "tfl.arg_min");
}

} // namespace onnx_mlir
