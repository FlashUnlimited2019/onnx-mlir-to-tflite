/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<Value> createBatchMatMul(Operation *sourceOp, Value lhs, Value rhs,
    Type resultType, bool adjX, bool adjY,
    ConversionPatternRewriter &rewriter) {
  if (failed(validateStaticF32Tensor(sourceOp, lhs.getType(), "lhs")) ||
      failed(validateStaticF32Tensor(sourceOp, rhs.getType(), "rhs")) ||
      failed(validateStaticF32Tensor(sourceOp, resultType, "result")))
    return failure();

  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  if (lhsType.getRank() < 2 || lhsType.getRank() > 4 || rhsType.getRank() < 2 ||
      rhsType.getRank() > 4) {
    sourceOp->emitError(
        "ONNXToTFL MVP MatMul/Gemm supports operand ranks 2 through 4");
    return failure();
  }

  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(adjX)),
      rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(adjY))};
  Operation *newOp =
      createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.batch_matmul",
          TypeRange{resultType}, ValueRange{lhs, rhs}, attributes);
  return newOp->getResult(0);
}

class MatMulLowering final : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (llvm::any_of(op->getOperandTypes(), [](Type type) {
          auto ranked = dyn_cast<RankedTensorType>(type);
          return ranked && ranked.getRank() == 4;
        })) {
      op.emitError("rank-4 MatMul is not supported with NCHW-to-NHWC layout "
                   "conversion");
      return failure();
    }
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    FailureOr<Value> result = createBatchMatMul(op, adaptor.getOperands()[0],
        adaptor.getOperands()[1], resultType, false, false, rewriter);
    if (failed(result))
      return failure();
    rewriter.replaceOp(op, *result);
    return success();
  }
};

float getFloatAttributeOr(Operation *op, StringRef name, float defaultValue) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return static_cast<float>(attr.getValueAsDouble());
  return defaultValue;
}

bool getBoolIntegerAttributeOr(
    Operation *op, StringRef name, bool defaultValue) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getValue().getSExtValue() != 0;
  return defaultValue;
}

Value createBinaryTFL(StringRef name, Location loc, Value lhs, Value rhs,
    Type resultType, ConversionPatternRewriter &rewriter) {
  SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
  return createTFLOperation(rewriter, loc, name, TypeRange{resultType},
      ValueRange{lhs, rhs}, attributes)
      ->getResult(0);
}

class GemmLowering final : public OpConversionPattern<ONNXGemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();
    if (operands.size() != 3)
      return op.emitError("unsupported Gemm configuration: expected A, B, C"),
             failure();

    bool transA = getBoolIntegerAttributeOr(op, "transA", false);
    bool transB = getBoolIntegerAttributeOr(op, "transB", false);
    float alpha = getFloatAttributeOr(op, "alpha", 1.0f);
    float beta = getFloatAttributeOr(op, "beta", 1.0f);
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());

    FailureOr<Value> matmul = createBatchMatMul(
        op, operands[0], operands[1], resultType, transA, transB, rewriter);
    if (failed(matmul))
      return failure();
    Value result = *matmul;

    if (alpha != 1.0f) {
      Value alphaValue =
          createF32ScalarTensorConstant(rewriter, op.getLoc(), alpha);
      result = createBinaryTFL(
          "tfl.mul", op.getLoc(), result, alphaValue, resultType, rewriter);
    }

    Value bias = operands[2];
    if (!isa<NoneType>(bias.getType())) {
      if (failed(validateStaticF32Tensor(op, bias.getType(), "Gemm bias C")))
        return failure();
      if (beta != 1.0f) {
        Value betaValue =
            createF32ScalarTensorConstant(rewriter, op.getLoc(), beta);
        bias = createBinaryTFL(
            "tfl.mul", op.getLoc(), bias, betaValue, bias.getType(), rewriter);
      }
      result = createBinaryTFL(
          "tfl.add", op.getLoc(), result, bias, resultType, rewriter);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXMatMulGemmOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MatMulLowering, GemmLowering>(
      typeConverter, patterns.getContext());
}

} // namespace onnx_mlir
