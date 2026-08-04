/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<Value> adaptRank3ChannelBroadcast(Operation *sourceOp,
    unsigned operandIndex, Value convertedOperand, Type convertedResultType,
    ConversionPatternRewriter &rewriter) {
  auto sourceResultType =
      dyn_cast<RankedTensorType>(sourceOp->getResult(0).getType());
  auto sourceOperandType =
      dyn_cast<RankedTensorType>(sourceOp->getOperand(operandIndex).getType());
  auto resultType = dyn_cast<RankedTensorType>(convertedResultType);
  if (!sourceResultType || sourceResultType.getRank() != 4 ||
      !sourceOperandType || sourceOperandType.getRank() != 3)
    return convertedOperand;

  ArrayRef<int64_t> shape = sourceOperandType.getShape();
  if (shape[1] != 1 || shape[2] != 1 || !resultType ||
      resultType.getShape()[3] != shape[0]) {
    sourceOp->emitError()
        << "unsupported rank-3 to rank-4 broadcast under NHWC layout: operand "
        << sourceOperandType << ", result " << sourceResultType;
    return failure();
  }

  auto broadcastType = RankedTensorType::get(
      {1, 1, shape[0]}, sourceOperandType.getElementType());
  Value shapeValue = createI32ShapeConstant(
      rewriter, sourceOp->getLoc(), broadcastType.getShape());
  return createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.reshape",
      TypeRange{broadcastType}, ValueRange{convertedOperand, shapeValue})
      ->getResult(0);
}

template <typename ONNXOp>
class BinaryElementwiseLowering final : public OpConversionPattern<ONNXOp> {
public:
  BinaryElementwiseLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32Tensor(
            op, adaptor.getOperands()[0].getType(), "left operand")) ||
        failed(validateStaticF32Tensor(
            op, adaptor.getOperands()[1].getType(), "right operand")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    SmallVector<Value> operands(adaptor.getOperands());
    for (unsigned i = 0; i < operands.size(); ++i) {
      FailureOr<Value> adapted =
          adaptRank3ChannelBroadcast(op, i, operands[i], resultType, rewriter);
      if (failed(adapted))
        return failure();
      operands[i] = *adapted;
    }
    SmallVector<NamedAttribute> attributes{getFusedActivationNone(rewriter)};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, operands, attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  std::string tflName;
};

template <typename ONNXOp>
class UnaryElementwiseLowering final : public OpConversionPattern<ONNXOp> {
public:
  UnaryElementwiseLowering(
      TypeConverter &typeConverter, MLIRContext *context, StringRef tflName)
      : OpConversionPattern<ONNXOp>(typeConverter, context), tflName(tflName) {}

  using OpAdaptor = typename ONNXOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32Tensor(
            op, adaptor.getOperands()[0].getType(), "operand")) ||
        failed(
            validateStaticF32Tensor(op, op->getResult(0).getType(), "result")))
      return failure();

    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), tflName,
        TypeRange{resultType}, adaptor.getOperands());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  std::string tflName;
};

class IdentityLowering final : public OpConversionPattern<ONNXIdentityOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXIdentityOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (failed(validateStaticF32Tensor(
            op, adaptor.getOperands()[0].getType(), "operand")))
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};

class ReturnLowering final : public OpConversionPattern<ONNXReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class NoValueLowering final : public OpConversionPattern<ONNXNoneOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXNoneOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute> attributes{
        rewriter.getNamedAttr("value", rewriter.getUnitAttr())};
    Operation *newOp = createTFLOperation(rewriter, op.getLoc(), "tfl.no_value",
        TypeRange{this->getTypeConverter()->convertType(op.getType())}, {},
        attributes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *context = patterns.getContext();
  patterns.add<IdentityLowering, ReturnLowering, NoValueLowering>(
      typeConverter, context);
  patterns.add<BinaryElementwiseLowering<ONNXAddOp>>(
      typeConverter, context, "tfl.add");
  patterns.add<BinaryElementwiseLowering<ONNXSubOp>>(
      typeConverter, context, "tfl.sub");
  patterns.add<BinaryElementwiseLowering<ONNXMulOp>>(
      typeConverter, context, "tfl.mul");
  patterns.add<BinaryElementwiseLowering<ONNXDivOp>>(
      typeConverter, context, "tfl.div");
  patterns.add<UnaryElementwiseLowering<ONNXReluOp>>(
      typeConverter, context, "tfl.relu");
  patterns.add<UnaryElementwiseLowering<ONNXSigmoidOp>>(
      typeConverter, context, "tfl.logistic");
  patterns.add<UnaryElementwiseLowering<ONNXTanhOp>>(
      typeConverter, context, "tfl.tanh");
}

} // namespace onnx_mlir
