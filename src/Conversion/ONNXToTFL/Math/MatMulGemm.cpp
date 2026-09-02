/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include <limits>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<Value> createBatchMatMul(Operation *sourceOp, Value lhs, Value rhs,
    Type sourceResultType, bool adjX, bool adjY,
    const TypeConverter *typeConverter, ConversionPatternRewriter &rewriter) {
  if (failed(validateStaticF32Tensor(sourceOp, lhs.getType(), "lhs")) ||
      failed(validateStaticF32Tensor(sourceOp, rhs.getType(), "rhs")) ||
      failed(validateStaticF32Tensor(sourceOp, sourceResultType, "result")))
    return failure();

  auto lhsType = cast<RankedTensorType>(sourceOp->getOperand(0).getType());
  auto rhsType = cast<RankedTensorType>(sourceOp->getOperand(1).getType());
  auto resultType = cast<RankedTensorType>(sourceResultType);
  auto logicalLhsType = stripONNXToTFLLayoutEncoding(lhsType);
  auto logicalRhsType = stripONNXToTFLLayoutEncoding(rhsType);
  auto logicalResultType = stripONNXToTFLLayoutEncoding(resultType);
  if (lhsType.getRank() < 2 || rhsType.getRank() < 2) {
    sourceOp->emitError("ONNXToTFL MatMul/Gemm requires operand rank >= 2");
    return failure();
  }

  auto restoreLogicalLayout = [&](Value value,
                                  RankedTensorType sourceType) -> Value {
    SmallVector<int64_t> sourceLayout =
        getONNXToTFLPhysicalToLogicalLayout(sourceType);
    SmallVector<int64_t> identity(sourceType.getRank());
    std::iota(identity.begin(), identity.end(), 0);
    FailureOr<Value> logical =
        convertTFLValueLayout(rewriter, sourceOp->getLoc(), value,
            stripONNXToTFLLayoutEncoding(sourceType), sourceLayout, identity);
    return succeeded(logical) ? *logical : Value();
  };
  lhs = restoreLogicalLayout(lhs, lhsType);
  rhs = restoreLogicalLayout(rhs, rhsType);
  if (!lhs || !rhs)
    return failure();

  // TFLite's rank-3 BatchMatMul path is substantially less sensitive to
  // backend-specific handling of higher-rank batch dimensions. Collapse only
  // when every leading batch dimension is statically identical, so no
  // broadcasting semantics are changed. All other cases retain the general
  // higher-rank lowering below.
  // Keep rank-4 BatchMatMul native: flattening it adds three Reshape ops and
  // can be slower on backends that already optimize the rank-4 form. Target
  // rank-5 and higher inputs, where avoiding high-rank BatchMatMul is
  // worthwhile and keeps the emitted FlatBuffer within backend rank limits.
  bool collapseBatchDimensions = lhsType.getRank() > 4 &&
                                 lhsType.getRank() == rhsType.getRank() &&
                                 lhsType.getRank() == resultType.getRank();
  int64_t flattenedBatch = 1;
  if (collapseBatchDimensions) {
    for (int64_t dimension = 0; dimension < lhsType.getRank() - 2;
        ++dimension) {
      int64_t extent = lhsType.getShape()[dimension];
      if (extent <= 0 || extent != rhsType.getShape()[dimension] ||
          extent != resultType.getShape()[dimension] ||
          flattenedBatch > std::numeric_limits<int64_t>::max() / extent) {
        collapseBatchDimensions = false;
        break;
      }
      flattenedBatch *= extent;
    }
  }
  if ((lhsType.getRank() > 5 || rhsType.getRank() > 5) &&
      !collapseBatchDimensions) {
    sourceOp->emitError("ONNXToTFL MatMul/Gemm with operand rank > 5 "
                        "requires identical static batch dimensions");
    return failure();
  }

  auto collapseToRank3 = [&](Value value, RankedTensorType type) -> Value {
    SmallVector<int64_t> shape{flattenedBatch,
        type.getShape()[type.getRank() - 2], type.getShape().back()};
    auto collapsedType = RankedTensorType::get(shape, type.getElementType());
    Value shapeValue =
        createI32ShapeConstant(rewriter, sourceOp->getLoc(), shape);
    return createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.reshape",
        TypeRange{collapsedType}, ValueRange{value, shapeValue})
        ->getResult(0);
  };
  if (collapseBatchDimensions) {
    lhs = collapseToRank3(lhs, logicalLhsType);
    rhs = collapseToRank3(rhs, logicalRhsType);
  }

  SmallVector<NamedAttribute> attributes{
      rewriter.getNamedAttr("adj_x", rewriter.getBoolAttr(adjX)),
      rewriter.getNamedAttr("adj_y", rewriter.getBoolAttr(adjY))};
  Type batchMatMulResultType = logicalResultType;
  if (collapseBatchDimensions) {
    batchMatMulResultType = RankedTensorType::get(
        {flattenedBatch, resultType.getShape()[resultType.getRank() - 2],
            resultType.getShape().back()},
        resultType.getElementType());
  }
  Operation *newOp =
      createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.batch_matmul",
          TypeRange{batchMatMulResultType}, ValueRange{lhs, rhs}, attributes);
  Value result = newOp->getResult(0);
  if (collapseBatchDimensions) {
    Value resultShape = createI32ShapeConstant(
        rewriter, sourceOp->getLoc(), resultType.getShape());
    result = createTFLOperation(rewriter, sourceOp->getLoc(), "tfl.reshape",
        TypeRange{logicalResultType}, ValueRange{result, resultShape})
                 ->getResult(0);
  }
  SmallVector<int64_t> identity(resultType.getRank());
  std::iota(identity.begin(), identity.end(), 0);
  SmallVector<int64_t> targetLayout =
      getONNXToTFLPhysicalToLogicalLayout(resultType);
  FailureOr<Value> physical = convertTFLValueLayout(rewriter,
      sourceOp->getLoc(), result, logicalResultType, identity, targetLayout);
  if (failed(physical) ||
      (*physical).getType() != typeConverter->convertType(sourceResultType))
    return failure();
  return *physical;
}

class MatMulLowering final : public OpConversionPattern<ONNXMatMulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXMatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> result = createBatchMatMul(op, adaptor.getOperands()[0],
        adaptor.getOperands()[1], op->getResult(0).getType(), false, false,
        this->getTypeConverter(), rewriter);
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

    FailureOr<Value> matmul = createBatchMatMul(op, operands[0], operands[1],
        op->getResult(0).getType(), transA, transB, this->getTypeConverter(),
        rewriter);
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

class FoldMatrixTransposeIntoBatchMatMul final : public RewritePattern {
public:
  explicit FoldMatrixTransposeIntoBatchMatMul(MLIRContext *context)
      : RewritePattern("tfl.batch_matmul", 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto originalAdjX = op->getAttrOfType<BoolAttr>("adj_x");
    auto originalAdjY = op->getAttrOfType<BoolAttr>("adj_y");
    if (!originalAdjX || !originalAdjY)
      return failure();

    for (int64_t operandIndex = 0; operandIndex < 2; ++operandIndex) {
      StringRef adjointAttribute = operandIndex == 0 ? "adj_x" : "adj_y";
      BoolAttr adjoint = operandIndex == 0 ? originalAdjX : originalAdjY;

      Value oldOperand = op->getOperand(operandIndex);
      Operation *reshape = oldOperand.getDefiningOp();
      Operation *lastTranspose = reshape;
      if (reshape && reshape->getName().getStringRef() == "tfl.reshape") {
        if (reshape->getNumOperands() != 2 || reshape->getNumResults() != 1)
          continue;
        lastTranspose = reshape->getOperand(0).getDefiningOp();
      } else {
        reshape = nullptr;
      }
      if (!lastTranspose ||
          lastTranspose->getName().getStringRef() != "tfl.transpose" ||
          lastTranspose->getNumOperands() != 2 ||
          lastTranspose->getNumResults() != 1)
        continue;

      auto transposedType =
          dyn_cast<RankedTensorType>(lastTranspose->getResult(0).getType());
      if (!transposedType || !transposedType.hasStaticShape() ||
          transposedType.getRank() < 2)
        continue;
      int64_t sourceRank = transposedType.getRank();
      SmallVector<int64_t> composedPermutation;
      SmallVector<Operation *> transposeChain;
      Operation *transpose = lastTranspose;
      Value replacement;
      RankedTensorType inputType;
      bool validChain = true;
      while (transpose) {
        auto currentInputType =
            dyn_cast<RankedTensorType>(transpose->getOperand(0).getType());
        auto currentOutputType =
            dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
        FailureOr<SmallVector<int64_t>> permutation =
            getConstantIntValues(transpose->getOperand(1));
        if (!currentInputType || !currentOutputType ||
            !currentInputType.hasStaticShape() ||
            !currentOutputType.hasStaticShape() ||
            currentInputType.getRank() != sourceRank ||
            currentOutputType.getRank() != sourceRank ||
            currentInputType.getElementType() !=
                transposedType.getElementType() ||
            currentOutputType.getElementType() !=
                transposedType.getElementType() ||
            failed(permutation) ||
            static_cast<int64_t>(permutation->size()) != sourceRank) {
          validChain = false;
          break;
        }
        SmallVector<bool> seen(sourceRank, false);
        for (int64_t axis : *permutation) {
          if (axis < 0 || axis >= sourceRank || seen[axis]) {
            validChain = false;
            break;
          }
          seen[axis] = true;
        }
        if (!validChain)
          break;
        if (composedPermutation.empty()) {
          composedPermutation = *permutation;
        } else {
          for (int64_t &axis : composedPermutation)
            axis = (*permutation)[axis];
        }
        transposeChain.push_back(transpose);
        replacement = transpose->getOperand(0);
        inputType = currentInputType;
        Operation *previous = replacement.getDefiningOp();
        transpose =
            previous && previous->getName().getStringRef() == "tfl.transpose"
                ? previous
                : nullptr;
      }
      if (!validChain || !replacement || !inputType)
        continue;
      bool swapsMatrixDimensions = true;
      for (int64_t axis = 0; axis < sourceRank - 2; ++axis)
        swapsMatrixDimensions &= composedPermutation[axis] == axis;
      swapsMatrixDimensions &=
          composedPermutation[sourceRank - 2] == sourceRank - 1 &&
          composedPermutation[sourceRank - 1] == sourceRank - 2;
      if (!swapsMatrixDimensions)
        continue;

      RankedTensorType replacementType = inputType;
      SmallVector<int64_t> replacementShape;
      if (reshape) {
        auto flattenedType = dyn_cast<RankedTensorType>(oldOperand.getType());
        if (!flattenedType || !flattenedType.hasStaticShape() ||
            flattenedType.getRank() != 3 ||
            flattenedType.getElementType() != inputType.getElementType())
          continue;
        int64_t flattenedBatch = 1;
        bool validBatch = true;
        for (int64_t axis = 0; axis < sourceRank - 2; ++axis) {
          int64_t extent = inputType.getShape()[axis];
          if (extent <= 0 ||
              flattenedBatch > std::numeric_limits<int64_t>::max() / extent) {
            validBatch = false;
            break;
          }
          flattenedBatch *= extent;
        }
        if (!validBatch || flattenedType.getShape()[0] != flattenedBatch ||
            flattenedType.getShape()[1] !=
                transposedType.getShape()[sourceRank - 2] ||
            flattenedType.getShape()[2] !=
                transposedType.getShape()[sourceRank - 1] ||
            flattenedBatch > std::numeric_limits<int32_t>::max() ||
            inputType.getShape()[sourceRank - 2] >
                std::numeric_limits<int32_t>::max() ||
            inputType.getShape()[sourceRank - 1] >
                std::numeric_limits<int32_t>::max())
          continue;

        replacementShape = {flattenedBatch,
            inputType.getShape()[sourceRank - 2],
            inputType.getShape()[sourceRank - 1]};
        replacementType =
            RankedTensorType::get(replacementShape, inputType.getElementType());
      }

      auto lhsType = dyn_cast<RankedTensorType>(
          operandIndex == 0 ? replacementType : op->getOperand(0).getType());
      auto rhsType = dyn_cast<RankedTensorType>(
          operandIndex == 1 ? replacementType : op->getOperand(1).getType());
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 3 ||
          rhsType.getRank() != 3 || resultType.getRank() != 3)
        continue;
      bool adjX =
          operandIndex == 0 ? !adjoint.getValue() : originalAdjX.getValue();
      bool adjY =
          operandIndex == 1 ? !adjoint.getValue() : originalAdjY.getValue();
      int64_t rows = lhsType.getShape()[adjX ? 2 : 1];
      int64_t lhsDepth = lhsType.getShape()[adjX ? 1 : 2];
      int64_t rhsDepth = rhsType.getShape()[adjY ? 2 : 1];
      int64_t columns = rhsType.getShape()[adjY ? 1 : 2];
      if (lhsDepth != rhsDepth || resultType.getShape()[1] != rows ||
          resultType.getShape()[2] != columns)
        continue;

      if (!replacementShape.empty()) {
        auto shapeType = RankedTensorType::get({3}, rewriter.getI32Type());
        SmallVector<int32_t> narrowedShape;
        for (int64_t extent : replacementShape)
          narrowedShape.push_back(static_cast<int32_t>(extent));
        Value shape = arith::ConstantOp::create(rewriter, op->getLoc(),
            shapeType, DenseIntElementsAttr::get(shapeType, narrowedShape));
        OperationState reshapeState(op->getLoc(), "tfl.reshape");
        reshapeState.addOperands({replacement, shape});
        reshapeState.addTypes(replacementType);
        replacement = rewriter.create(reshapeState)->getResult(0);
      }
      rewriter.modifyOpInPlace(op, [&]() {
        op->setOperand(operandIndex, replacement);
        op->setAttr(
            adjointAttribute, rewriter.getBoolAttr(!adjoint.getValue()));
      });
      if (reshape && reshape->use_empty())
        rewriter.eraseOp(reshape);
      for (Operation *candidate : transposeChain)
        if (candidate->use_empty())
          rewriter.eraseOp(candidate);
      return success();
    }
    return failure();
  }
};

} // namespace

void populateLoweringONNXMatMulGemmOpToTFLPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<MatMulLowering, GemmLowering>(
      typeConverter, patterns.getContext());
}

void populateTFLMatMulPostLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldMatrixTransposeIntoBatchMatMul>(patterns.getContext());
}

} // namespace onnx_mlir
