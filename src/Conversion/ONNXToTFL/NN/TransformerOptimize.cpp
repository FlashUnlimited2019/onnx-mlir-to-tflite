/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "src/Dialect/ONNX/DialectBuilder.hpp"

#include <limits>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {
namespace {

FailureOr<DenseElementsAttr> getDenseF32Constant(Value value) {
  Attribute attribute;
  if (auto constant = value.getDefiningOp<ONNXConstantOp>())
    attribute = constant.getValueAttr();
  else if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    attribute = constant.getValue();
  auto elements = dyn_cast_or_null<DenseElementsAttr>(attribute);
  if (!elements || !elements.getElementType().isF32())
    return failure();
  return elements;
}

SmallVector<int64_t> getTransposePermutation(ONNXTransposeOp op) {
  SmallVector<int64_t> permutation;
  if (auto array = op->getAttrOfType<ArrayAttr>("perm")) {
    for (Attribute element : array)
      permutation.push_back(cast<IntegerAttr>(element).getInt());
  } else if (auto dense = op->getAttrOfType<DenseIntElementsAttr>("perm")) {
    for (APInt element : dense.getValues<APInt>())
      permutation.push_back(element.getSExtValue());
  }
  return permutation;
}

Operation *getSingleUser(Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

struct ProjectionBranch {
  ONNXAddOp biasAdd;
  ONNXMulOp scale;
  ONNXReshapeOp reshape;
  ONNXTransposeOp transpose;
  Value bias;
  Value scaleValue;
  SmallVector<float> biasValues;
  int64_t heads;
  int64_t headWidth;
};

FailureOr<ProjectionBranch> matchSplitProjectionBranch(Value splitResult) {
  auto splitType = dyn_cast<RankedTensorType>(splitResult.getType());
  auto add = dyn_cast_or_null<ONNXAddOp>(getSingleUser(splitResult));
  if (!splitType || !splitType.hasStaticShape() || splitType.getRank() != 3 ||
      !splitType.getElementType().isF32() || !add ||
      !add.getResult().hasOneUse())
    return failure();

  Value bias;
  if (add.getA() == splitResult)
    bias = add.getB();
  else if (add.getB() == splitResult)
    bias = add.getA();
  else
    return failure();
  FailureOr<SmallVector<float>> biasValues = getConstantF32Values(bias);
  if (failed(biasValues) ||
      static_cast<int64_t>(biasValues->size()) != splitType.getShape().back())
    return failure();

  Value current = add.getResult();
  ONNXMulOp scale;
  Value scaleValue;
  if (auto candidate = dyn_cast_or_null<ONNXMulOp>(getSingleUser(current))) {
    Value other;
    if (candidate.getA() == current)
      other = candidate.getB();
    else if (candidate.getB() == current)
      other = candidate.getA();
    FailureOr<SmallVector<float>> scaleValues = getConstantF32Values(other);
    if (other && succeeded(scaleValues) && scaleValues->size() == 1 &&
        candidate.getResult().hasOneUse()) {
      scale = candidate;
      scaleValue = other;
      current = scale.getResult();
    }
  }

  auto reshape = dyn_cast_or_null<ONNXReshapeOp>(getSingleUser(current));
  if (!reshape || !reshape.getResult().hasOneUse())
    return failure();
  auto transpose =
      dyn_cast_or_null<ONNXTransposeOp>(getSingleUser(reshape.getResult()));
  auto reshapeType = dyn_cast<RankedTensorType>(reshape.getResult().getType());
  auto transposeType =
      transpose ? dyn_cast<RankedTensorType>(transpose.getResult().getType())
                : RankedTensorType();
  if (!transpose || !reshapeType || !transposeType ||
      !reshapeType.hasStaticShape() || !transposeType.hasStaticShape() ||
      reshapeType.getRank() != 4 || transposeType.getRank() != 4 ||
      getTransposePermutation(transpose) !=
          SmallVector<int64_t>({0, 2, 1, 3}) ||
      reshapeType.getShape()[0] != splitType.getShape()[0] ||
      reshapeType.getShape()[1] != splitType.getShape()[1] ||
      reshapeType.getShape()[2] <= 0 || reshapeType.getShape()[3] <= 0 ||
      reshapeType.getShape()[2] > std::numeric_limits<int32_t>::max() ||
      reshapeType.getShape()[2] * reshapeType.getShape()[3] !=
          splitType.getShape()[2] ||
      transposeType.getShape() !=
          ArrayRef<int64_t>(
              {reshapeType.getShape()[0], reshapeType.getShape()[2],
                  reshapeType.getShape()[1], reshapeType.getShape()[3]}))
    return failure();

  return ProjectionBranch{add, scale, reshape, transpose, bias, scaleValue,
      std::move(*biasValues), reshapeType.getShape()[2],
      reshapeType.getShape()[3]};
}

class PackSplitProjectionPattern final : public OpRewritePattern<ONNXSplitOp> {
public:
  PackSplitProjectionPattern(MLIRContext *context)
      : OpRewritePattern<ONNXSplitOp>(context, PatternBenefit(10)) {}

  LogicalResult matchAndRewrite(
      ONNXSplitOp op, PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 3 ||
        !inputType.getElementType().isF32() || op->getNumResults() != 3)
      return failure();
    int64_t axis = normalizeAxis(op.getAxis(), inputType.getRank());
    if (axis != inputType.getRank() - 1)
      return failure();

    SmallVector<ProjectionBranch> branches;
    branches.reserve(3);
    for (Value result : op.getOutputs()) {
      FailureOr<ProjectionBranch> branch = matchSplitProjectionBranch(result);
      if (failed(branch))
        return failure();
      branches.push_back(std::move(*branch));
    }

    int64_t batch = inputType.getShape()[0];
    int64_t sequence = inputType.getShape()[1];
    int64_t headWidth = branches.front().headWidth;
    int64_t totalHeads = 0;
    int64_t totalFeatures = 0;
    SmallVector<float> combinedBias;
    SmallVector<int64_t> splitHeads;
    SmallVector<Type> splitResultTypes;
    for (ProjectionBranch &branch : branches) {
      if (branch.headWidth != headWidth)
        return failure();
      totalHeads += branch.heads;
      totalFeatures += branch.heads * headWidth;
      combinedBias.append(branch.biasValues);
      splitHeads.push_back(branch.heads);
      splitResultTypes.push_back(branch.transpose->getResult(0).getType());
    }
    if (totalFeatures != inputType.getShape().back())
      return failure();

    Location loc = op.getLoc();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    auto biasType =
        RankedTensorType::get({totalFeatures}, rewriter.getF32Type());
    Value bias = create.onnx.constant(
        DenseElementsAttr::get<float>(biasType, combinedBias));
    Value biased =
        ONNXAddOp::create(rewriter, loc, inputType, op.getInput(), bias);

    auto packedType = RankedTensorType::get(
        {batch, sequence, totalHeads, headWidth}, rewriter.getF32Type());
    Value packedShape =
        create.onnx.constantInt64({batch, sequence, totalHeads, headWidth});
    Value packed =
        ONNXReshapeOp::create(rewriter, loc, packedType, biased, packedShape);
    auto transposedType = RankedTensorType::get(
        {batch, totalHeads, sequence, headWidth}, rewriter.getF32Type());
    Value transposed = ONNXTransposeOp::create(rewriter, loc, transposedType,
        packed, rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    Value splitSizes = create.onnx.constantInt64(splitHeads);
    ONNXSplitOp packedSplit = ONNXSplitOp::create(rewriter, loc,
        splitResultTypes, transposed, splitSizes, /*axis=*/1, nullptr);

    SmallVector<Operation *> obsolete;
    for (auto [index, branch] : llvm::enumerate(branches)) {
      Value replacement = packedSplit.getOutputs()[index];
      if (branch.scale) {
        replacement = ONNXMulOp::create(rewriter, loc,
            branch.transpose.getResult().getType(), replacement,
            branch.scaleValue);
      }
      rewriter.replaceAllUsesWith(branch.transpose.getResult(), replacement);
      obsolete.push_back(branch.transpose);
      obsolete.push_back(branch.reshape);
      if (branch.scale)
        obsolete.push_back(branch.scale);
      obsolete.push_back(branch.biasAdd);
    }
    for (Operation *operation : obsolete)
      rewriter.eraseOp(operation);
    rewriter.eraseOp(op);
    return success();
  }
};

struct SeparateProjectionBranch {
  ONNXMatMulOp matmul;
  ONNXReshapeOp reshape;
  ONNXTransposeOp transpose;
  DenseElementsAttr weight;
  int64_t heads;
  int64_t headWidth;
};

FailureOr<SeparateProjectionBranch> matchSeparateProjection(ONNXMatMulOp op) {
  auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getB().getType());
  FailureOr<DenseElementsAttr> weight = getDenseF32Constant(op.getB());
  if (!outputType || !weightType || failed(weight) ||
      !outputType.hasStaticShape() || !weightType.hasStaticShape() ||
      outputType.getRank() != 3 || weightType.getRank() != 2 ||
      !outputType.getElementType().isF32() || !op.getResult().hasOneUse() ||
      weightType.getShape()[1] != outputType.getShape().back())
    return failure();
  auto reshape = dyn_cast_or_null<ONNXReshapeOp>(getSingleUser(op.getResult()));
  if (!reshape || !reshape.getResult().hasOneUse())
    return failure();
  auto transpose =
      dyn_cast_or_null<ONNXTransposeOp>(getSingleUser(reshape.getResult()));
  auto reshapeType = dyn_cast<RankedTensorType>(reshape.getResult().getType());
  auto transposeType =
      transpose ? dyn_cast<RankedTensorType>(transpose.getResult().getType())
                : RankedTensorType();
  if (!transpose || !reshapeType || !transposeType ||
      !reshapeType.hasStaticShape() || !transposeType.hasStaticShape() ||
      reshapeType.getRank() != 4 ||
      getTransposePermutation(transpose) !=
          SmallVector<int64_t>({0, 2, 1, 3}) ||
      reshapeType.getShape()[0] != outputType.getShape()[0] ||
      reshapeType.getShape()[1] != outputType.getShape()[1] ||
      reshapeType.getShape()[2] <= 0 || reshapeType.getShape()[3] <= 0 ||
      reshapeType.getShape()[2] * reshapeType.getShape()[3] !=
          outputType.getShape()[2] ||
      transposeType.getShape() !=
          ArrayRef<int64_t>(
              {reshapeType.getShape()[0], reshapeType.getShape()[2],
                  reshapeType.getShape()[1], reshapeType.getShape()[3]}))
    return failure();
  return SeparateProjectionBranch{op, reshape, transpose, *weight,
      reshapeType.getShape()[2], reshapeType.getShape()[3]};
}

class PackSeparateProjectionPattern final
    : public OpRewritePattern<ONNXMatMulOp> {
public:
  PackSeparateProjectionPattern(MLIRContext *context)
      : OpRewritePattern<ONNXMatMulOp>(context, PatternBenefit(9)) {}

  LogicalResult matchAndRewrite(
      ONNXMatMulOp op, PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getA().getType());
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 3 ||
        !inputType.getElementType().isF32())
      return failure();
    SmallVector<SeparateProjectionBranch> branches;
    for (Operation *user : op.getA().getUsers()) {
      auto candidate = dyn_cast<ONNXMatMulOp>(user);
      if (!candidate || candidate.getA() != op.getA() ||
          candidate->getBlock() != op->getBlock())
        continue;
      FailureOr<SeparateProjectionBranch> branch =
          matchSeparateProjection(candidate);
      if (succeeded(branch))
        branches.push_back(std::move(*branch));
    }
    if (branches.size() != 3)
      return failure();
    llvm::sort(branches, [](const SeparateProjectionBranch &lhs,
                             const SeparateProjectionBranch &rhs) {
      return lhs.matmul->isBeforeInBlock(rhs.matmul);
    });
    if (branches.front().matmul != op)
      return failure();

    auto firstOutputType =
        cast<RankedTensorType>(branches.front().matmul.getResult().getType());
    int64_t batch = firstOutputType.getShape()[0];
    int64_t sequence = firstOutputType.getShape()[1];
    int64_t inputWidth =
        cast<RankedTensorType>(branches.front().matmul.getB().getType())
            .getShape()[0];
    int64_t headWidth = branches.front().headWidth;
    int64_t totalHeads = 0;
    int64_t totalFeatures = 0;
    SmallVector<int64_t> branchFeatures;
    for (SeparateProjectionBranch &branch : branches) {
      auto outputType =
          cast<RankedTensorType>(branch.matmul->getResult(0).getType());
      auto weightType =
          cast<RankedTensorType>(branch.matmul->getOperand(1).getType());
      if (branch.headWidth != headWidth || outputType.getShape()[0] != batch ||
          outputType.getShape()[1] != sequence ||
          weightType.getShape()[0] != inputWidth)
        return failure();
      int64_t features = branch.heads * headWidth;
      totalHeads += branch.heads;
      totalFeatures += features;
      branchFeatures.push_back(features);
    }
    if (inputType.getShape()[0] != batch ||
        inputType.getShape()[1] != sequence ||
        inputType.getShape()[2] != inputWidth)
      return failure();

    SmallVector<SmallVector<float>> weights;
    for (SeparateProjectionBranch &branch : branches) {
      SmallVector<float> values;
      values.reserve(branch.weight.getNumElements());
      for (APFloat value : branch.weight.getValues<APFloat>())
        values.push_back(value.convertToFloat());
      weights.push_back(std::move(values));
    }
    SmallVector<float> combinedWeights;
    combinedWeights.reserve(inputWidth * totalFeatures);
    for (int64_t row = 0; row < inputWidth; ++row) {
      for (auto [index, values] : llvm::enumerate(weights)) {
        int64_t features = branchFeatures[index];
        combinedWeights.append(values.begin() + row * features,
            values.begin() + (row + 1) * features);
      }
    }

    Location loc = op.getLoc();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
    auto weightType = RankedTensorType::get(
        {inputWidth, totalFeatures}, rewriter.getF32Type());
    Value weight = create.onnx.constant(
        DenseElementsAttr::get<float>(weightType, combinedWeights));
    auto combinedType = RankedTensorType::get(
        {batch, sequence, totalFeatures}, rewriter.getF32Type());
    Value combined =
        ONNXMatMulOp::create(rewriter, loc, combinedType, op.getA(), weight);
    auto packedType = RankedTensorType::get(
        {batch, sequence, totalHeads, headWidth}, rewriter.getF32Type());
    Value packedShape =
        create.onnx.constantInt64({batch, sequence, totalHeads, headWidth});
    Value packed =
        ONNXReshapeOp::create(rewriter, loc, packedType, combined, packedShape);
    auto transposedType = RankedTensorType::get(
        {batch, totalHeads, sequence, headWidth}, rewriter.getF32Type());
    Value transposed = ONNXTransposeOp::create(rewriter, loc, transposedType,
        packed, rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    SmallVector<int64_t> splitHeads;
    SmallVector<Type> splitResultTypes;
    for (SeparateProjectionBranch &branch : branches) {
      splitHeads.push_back(branch.heads);
      splitResultTypes.push_back(branch.transpose->getResult(0).getType());
    }
    Value splitSizes = create.onnx.constantInt64(splitHeads);
    ONNXSplitOp split = ONNXSplitOp::create(rewriter, loc, splitResultTypes,
        transposed, splitSizes, /*axis=*/1, nullptr);

    for (auto [index, branch] : llvm::enumerate(branches))
      rewriter.replaceAllUsesWith(
          branch.transpose.getResult(), split.getOutputs()[index]);
    for (const SeparateProjectionBranch &branch : branches) {
      rewriter.eraseOp(branch.transpose);
      rewriter.eraseOp(branch.reshape);
      rewriter.eraseOp(branch.matmul);
    }
    return success();
  }
};

bool isPermutation(ArrayRef<int64_t> permutation, int64_t rank) {
  if (static_cast<int64_t>(permutation.size()) != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= rank || seen[axis])
      return false;
    seen[axis] = true;
  }
  return true;
}

FailureOr<DenseElementsAttr> transposeBroadcastConstant(DenseElementsAttr input,
    ArrayRef<int64_t> permutation, RankedTensorType sourceType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() ||
      !sourceType.hasStaticShape() ||
      inputType.getRank() != sourceType.getRank() ||
      !inputType.getElementType().isF32() ||
      inputType.getNumElements() != sourceType.getNumElements())
    return failure();
  int64_t rank = sourceType.getRank();
  SmallVector<int64_t> inputStrides(rank, 1), sourceStrides(rank, 1);
  for (int64_t axis = rank - 2; axis >= 0; --axis) {
    inputStrides[axis] =
        inputStrides[axis + 1] * inputType.getShape()[axis + 1];
    sourceStrides[axis] =
        sourceStrides[axis + 1] * sourceType.getShape()[axis + 1];
  }
  SmallVector<float> inputValues;
  inputValues.reserve(input.getNumElements());
  for (APFloat value : input.getValues<APFloat>())
    inputValues.push_back(value.convertToFloat());
  SmallVector<float> sourceValues(sourceType.getNumElements());
  for (int64_t sourceIndex = 0; sourceIndex < sourceType.getNumElements();
       ++sourceIndex) {
    int64_t remaining = sourceIndex;
    SmallVector<int64_t> sourceCoordinate(rank);
    for (int64_t axis = 0; axis < rank; ++axis) {
      sourceCoordinate[axis] = remaining / sourceStrides[axis];
      remaining %= sourceStrides[axis];
    }
    int64_t inputIndex = 0;
    for (int64_t outputAxis = 0; outputAxis < rank; ++outputAxis)
      inputIndex +=
          sourceCoordinate[permutation[outputAxis]] * inputStrides[outputAxis];
    sourceValues[sourceIndex] = inputValues[inputIndex];
  }
  return DenseElementsAttr::get<float>(sourceType, sourceValues);
}

bool areInversePermutations(ArrayRef<int64_t> first, ArrayRef<int64_t> second) {
  if (first.size() != second.size())
    return false;
  for (int64_t axis = 0, rank = first.size(); axis < rank; ++axis)
    if (first[second[axis]] != axis)
      return false;
  return true;
}

class FoldTransposeSoftmaxTransposePattern final : public RewritePattern {
public:
  explicit FoldTransposeSoftmaxTransposePattern(MLIRContext *context)
      : RewritePattern("tfl.transpose", 5, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    for (Operation *user : op->getResult(0).getUsers())
      if (user->getName().getStringRef() == "tfl.transpose")
        return failure();

    SmallVector<Operation *> trailingTransposes;
    Operation *current = op;
    while (current && current->getName().getStringRef() == "tfl.transpose") {
      if (current->getNumOperands() != 2 || current->getNumResults() != 1 ||
          (current != op && !current->getResult(0).hasOneUse()))
        return failure();
      trailingTransposes.push_back(current);
      current = current->getOperand(0).getDefiningOp();
    }
    Operation *softmax = current;
    if (!softmax || softmax->getName().getStringRef() != "tfl.softmax" ||
        softmax->getNumOperands() != 1 || softmax->getNumResults() != 1 ||
        !softmax->getResult(0).hasOneUse())
      return failure();

    SmallVector<Operation *> forwardTransposes;
    Value source = softmax->getOperand(0);
    while (Operation *transpose = source.getDefiningOp()) {
      if (transpose->getName().getStringRef() != "tfl.transpose")
        break;
      if (transpose->getNumOperands() != 2 || transpose->getNumResults() != 1 ||
          !transpose->getResult(0).hasOneUse())
        return failure();
      forwardTransposes.push_back(transpose);
      source = transpose->getOperand(0);
    }
    if (forwardTransposes.empty())
      return failure();

    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    auto permutedType =
        dyn_cast<RankedTensorType>(softmax->getOperand(0).getType());
    auto softmaxType =
        dyn_cast<RankedTensorType>(softmax->getResult(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!sourceType || !permutedType || !softmaxType || !resultType ||
        !sourceType.hasStaticShape() || !permutedType.hasStaticShape() ||
        !softmaxType.hasStaticShape() || !resultType.hasStaticShape() ||
        sourceType != resultType || permutedType != softmaxType ||
        !sourceType.getElementType().isF32() || sourceType.getRank() < 1)
      return failure();

    auto compose = [&](ArrayRef<Operation *> outermostToInnermost)
        -> FailureOr<SmallVector<int64_t>> {
      SmallVector<int64_t> composed;
      for (Operation *transpose : llvm::reverse(outermostToInnermost)) {
        auto inputType =
            dyn_cast<RankedTensorType>(transpose->getOperand(0).getType());
        auto outputType =
            dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
        FailureOr<SmallVector<int64_t>> permutation =
            getConstantIntValues(transpose->getOperand(1));
        if (!inputType || !outputType || !inputType.hasStaticShape() ||
            !outputType.hasStaticShape() ||
            inputType.getRank() != sourceType.getRank() ||
            outputType.getRank() != sourceType.getRank() ||
            inputType.getElementType() != sourceType.getElementType() ||
            outputType.getElementType() != sourceType.getElementType() ||
            failed(permutation) ||
            !isPermutation(*permutation, sourceType.getRank()))
          return failure();
        for (int64_t axis = 0; axis < sourceType.getRank(); ++axis)
          if (outputType.getShape()[axis] !=
              inputType.getShape()[(*permutation)[axis]])
            return failure();
        if (composed.empty()) {
          composed = std::move(*permutation);
          continue;
        }
        SmallVector<int64_t> next(sourceType.getRank());
        for (int64_t axis = 0; axis < sourceType.getRank(); ++axis)
          next[axis] = composed[(*permutation)[axis]];
        composed = std::move(next);
      }
      return composed;
    };
    FailureOr<SmallVector<int64_t>> forwardPermutation =
        compose(forwardTransposes);
    FailureOr<SmallVector<int64_t>> trailingPermutation =
        compose(trailingTransposes);
    if (failed(forwardPermutation) || failed(trailingPermutation) ||
        !areInversePermutations(*forwardPermutation, *trailingPermutation) ||
        forwardPermutation->back() != sourceType.getRank() - 1)
      return failure();

    OperationState state(op->getLoc(), "tfl.softmax");
    state.addOperands(source);
    state.addTypes(resultType);
    state.addAttributes(softmax->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    for (Operation *transpose : llvm::drop_begin(trailingTransposes))
      rewriter.eraseOp(transpose);
    rewriter.eraseOp(softmax);
    for (Operation *transpose : forwardTransposes)
      rewriter.eraseOp(transpose);
    return success();
  }
};

class FoldTransposeGeluTransposePattern final : public RewritePattern {
public:
  explicit FoldTransposeGeluTransposePattern(MLIRContext *context)
      : RewritePattern("tfl.transpose", 4, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    Operation *gelu = op->getOperand(0).getDefiningOp();
    if (!gelu || gelu->getName().getStringRef() != "tfl.gelu" ||
        gelu->getNumOperands() != 1 || gelu->getNumResults() != 1 ||
        !gelu->getResult(0).hasOneUse())
      return failure();
    Operation *firstTranspose = gelu->getOperand(0).getDefiningOp();
    if (!firstTranspose ||
        firstTranspose->getName().getStringRef() != "tfl.transpose" ||
        firstTranspose->getNumOperands() != 2 ||
        firstTranspose->getNumResults() != 1 ||
        !firstTranspose->getResult(0).hasOneUse())
      return failure();

    auto sourceType =
        dyn_cast<RankedTensorType>(firstTranspose->getOperand(0).getType());
    auto permutedType =
        dyn_cast<RankedTensorType>(firstTranspose->getResult(0).getType());
    auto geluType = dyn_cast<RankedTensorType>(gelu->getResult(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    FailureOr<SmallVector<int64_t>> firstPermutation =
        getConstantIntValues(firstTranspose->getOperand(1));
    FailureOr<SmallVector<int64_t>> secondPermutation =
        getConstantIntValues(op->getOperand(1));
    if (!sourceType || !permutedType || !geluType || !resultType ||
        !sourceType.hasStaticShape() || !permutedType.hasStaticShape() ||
        !geluType.hasStaticShape() || !resultType.hasStaticShape() ||
        sourceType != resultType || permutedType != geluType ||
        !sourceType.getElementType().isF32() || failed(firstPermutation) ||
        failed(secondPermutation) ||
        !isPermutation(*firstPermutation, sourceType.getRank()) ||
        !isPermutation(*secondPermutation, sourceType.getRank()) ||
        !areInversePermutations(*firstPermutation, *secondPermutation))
      return failure();
    for (int64_t axis = 0; axis < sourceType.getRank(); ++axis) {
      if (permutedType.getShape()[axis] !=
              sourceType.getShape()[(*firstPermutation)[axis]] ||
          resultType.getShape()[axis] !=
              geluType.getShape()[(*secondPermutation)[axis]])
        return failure();
    }

    OperationState state(op->getLoc(), "tfl.gelu");
    state.addOperands(firstTranspose->getOperand(0));
    state.addTypes(resultType);
    state.addAttributes(gelu->getAttrs());
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    rewriter.eraseOp(gelu);
    rewriter.eraseOp(firstTranspose);
    return success();
  }
};

class CollapseHeadLayoutTransposeChainPattern final : public RewritePattern {
public:
  explicit CollapseHeadLayoutTransposeChainPattern(MLIRContext *context)
      : RewritePattern("tfl.transpose", 6, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    SmallVector<Operation *> rankThreeTransposes;
    Operation *current = op;
    while (current && current->getName().getStringRef() == "tfl.transpose") {
      if (current->getNumOperands() != 2 || current->getNumResults() != 1 ||
          !current->getResult(0).hasOneUse())
        return failure();
      rankThreeTransposes.push_back(current);
      current = current->getOperand(0).getDefiningOp();
    }
    Operation *firstReshape = current;
    Operation *finalReshape = getSingleUser(op->getResult(0));
    if (!finalReshape ||
        finalReshape->getName().getStringRef() != "tfl.reshape" ||
        finalReshape->getNumOperands() != 2 ||
        finalReshape->getNumResults() != 1 || !firstReshape ||
        firstReshape->getName().getStringRef() != "tfl.reshape" ||
        firstReshape->getNumOperands() != 2 ||
        firstReshape->getNumResults() != 1 ||
        !firstReshape->getResult(0).hasOneUse())
      return failure();

    SmallVector<Operation *> rankFourTransposes;
    Value source = firstReshape->getOperand(0);
    while (Operation *transpose = source.getDefiningOp()) {
      if (transpose->getName().getStringRef() != "tfl.transpose")
        break;
      if (transpose->getNumOperands() != 2 || transpose->getNumResults() != 1 ||
          !transpose->getResult(0).hasOneUse())
        return failure();
      rankFourTransposes.push_back(transpose);
      source = transpose->getOperand(0);
    }
    if (rankFourTransposes.empty())
      return failure();

    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    auto headOrderedType = dyn_cast<RankedTensorType>(
        rankFourTransposes.front()->getResult(0).getType());
    auto firstReshapeType =
        dyn_cast<RankedTensorType>(firstReshape->getResult(0).getType());
    auto matrixTransposedType =
        dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto resultType =
        dyn_cast<RankedTensorType>(finalReshape->getResult(0).getType());
    FailureOr<SmallVector<int64_t>> firstShape =
        getConstantIntValues(firstReshape->getOperand(1));
    FailureOr<SmallVector<int64_t>> finalShape =
        getConstantIntValues(finalReshape->getOperand(1));
    if (!sourceType || !headOrderedType || !firstReshapeType ||
        !matrixTransposedType || !resultType || !sourceType.hasStaticShape() ||
        !headOrderedType.hasStaticShape() ||
        !firstReshapeType.hasStaticShape() ||
        !matrixTransposedType.hasStaticShape() ||
        !resultType.hasStaticShape() || !sourceType.getElementType().isF32() ||
        sourceType.getRank() != 4 || headOrderedType.getRank() != 4 ||
        firstReshapeType.getRank() != 3 ||
        matrixTransposedType.getRank() != 3 || resultType.getRank() != 4 ||
        sourceType.getElementType() != headOrderedType.getElementType() ||
        sourceType.getElementType() != firstReshapeType.getElementType() ||
        sourceType.getElementType() != matrixTransposedType.getElementType() ||
        sourceType.getElementType() != resultType.getElementType() ||
        failed(firstShape) || failed(finalShape))
      return failure();

    auto compose = [&](ArrayRef<Operation *> outermostToInnermost,
                       int64_t rank) -> FailureOr<SmallVector<int64_t>> {
      SmallVector<int64_t> composed;
      for (Operation *transpose : llvm::reverse(outermostToInnermost)) {
        auto inputType =
            dyn_cast<RankedTensorType>(transpose->getOperand(0).getType());
        auto outputType =
            dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
        FailureOr<SmallVector<int64_t>> permutation =
            getConstantIntValues(transpose->getOperand(1));
        if (!inputType || !outputType || !inputType.hasStaticShape() ||
            !outputType.hasStaticShape() || inputType.getRank() != rank ||
            outputType.getRank() != rank ||
            inputType.getElementType() != sourceType.getElementType() ||
            outputType.getElementType() != sourceType.getElementType() ||
            failed(permutation) || !isPermutation(*permutation, rank))
          return failure();
        for (int64_t axis = 0; axis < rank; ++axis)
          if (outputType.getShape()[axis] !=
              inputType.getShape()[(*permutation)[axis]])
            return failure();
        if (composed.empty()) {
          composed = std::move(*permutation);
          continue;
        }
        SmallVector<int64_t> next(rank);
        for (int64_t axis = 0; axis < rank; ++axis)
          next[axis] = composed[(*permutation)[axis]];
        composed = std::move(next);
      }
      return composed;
    };
    FailureOr<SmallVector<int64_t>> rankFourPermutation =
        compose(rankFourTransposes, 4);
    FailureOr<SmallVector<int64_t>> rankThreePermutation =
        compose(rankThreeTransposes, 3);
    if (failed(rankFourPermutation) || failed(rankThreePermutation) ||
        *rankFourPermutation != SmallVector<int64_t>({0, 2, 1, 3}) ||
        *rankThreePermutation != SmallVector<int64_t>({0, 2, 1}))
      return failure();

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    int64_t batch = sourceShape[0];
    int64_t sequence = sourceShape[1];
    int64_t heads = sourceShape[2];
    int64_t width = sourceShape[3];
    if (batch <= 0 || sequence <= 0 || heads <= 0 || width <= 0 ||
        batch > std::numeric_limits<int64_t>::max() / heads)
      return failure();
    int64_t flattenedBatch = batch * heads;
    SmallVector<int64_t> expectedFirstTranspose{batch, heads, sequence, width};
    SmallVector<int64_t> expectedFirstReshape{flattenedBatch, sequence, width};
    SmallVector<int64_t> expectedSecondTranspose{
        flattenedBatch, width, sequence};
    SmallVector<int64_t> expectedResult{batch, heads, width, sequence};
    if (headOrderedType.getShape() !=
            ArrayRef<int64_t>(expectedFirstTranspose) ||
        firstReshapeType.getShape() !=
            ArrayRef<int64_t>(expectedFirstReshape) ||
        matrixTransposedType.getShape() !=
            ArrayRef<int64_t>(expectedSecondTranspose) ||
        resultType.getShape() != ArrayRef<int64_t>(expectedResult) ||
        *firstShape != expectedFirstReshape || *finalShape != expectedResult)
      return failure();

    auto permutationType = RankedTensorType::get({4}, rewriter.getI32Type());
    Value directPermutation =
        arith::ConstantOp::create(rewriter, op->getLoc(), permutationType,
            DenseIntElementsAttr::get(
                permutationType, SmallVector<int32_t>{0, 2, 3, 1}));
    OperationState state(op->getLoc(), "tfl.transpose");
    state.addOperands({source, directPermutation});
    state.addTypes(resultType);
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(finalReshape, replacement->getResults());
    for (Operation *transpose : rankThreeTransposes)
      rewriter.eraseOp(transpose);
    rewriter.eraseOp(firstReshape);
    for (Operation *transpose : rankFourTransposes)
      rewriter.eraseOp(transpose);
    return success();
  }
};

class MoveBroadcastAddBeforeTransposePattern final : public RewritePattern {
public:
  explicit MoveBroadcastAddBeforeTransposePattern(MLIRContext *context)
      : RewritePattern("tfl.transpose", 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    Operation *add = op->getOperand(0).getDefiningOp();
    if (!add || add->getName().getStringRef() != "tfl.add" ||
        add->getNumOperands() != 2 || add->getNumResults() != 1 ||
        !add->getResult(0).hasOneUse())
      return failure();

    Operation *firstTranspose = nullptr;
    Value constant;
    if (Operation *candidate = add->getOperand(0).getDefiningOp();
        candidate && candidate->getName().getStringRef() == "tfl.transpose") {
      firstTranspose = candidate;
      constant = add->getOperand(1);
    } else if (Operation *candidate = add->getOperand(1).getDefiningOp();
               candidate &&
               candidate->getName().getStringRef() == "tfl.transpose") {
      firstTranspose = candidate;
      constant = add->getOperand(0);
    }
    if (!firstTranspose || firstTranspose->getNumOperands() != 2 ||
        firstTranspose->getNumResults() != 1 ||
        !firstTranspose->getResult(0).hasOneUse())
      return failure();

    FailureOr<DenseElementsAttr> constantAttr = getDenseF32Constant(constant);
    FailureOr<SmallVector<int64_t>> firstPermutation =
        getConstantIntValues(firstTranspose->getOperand(1));
    FailureOr<SmallVector<int64_t>> secondPermutation =
        getConstantIntValues(op->getOperand(1));
    auto sourceType =
        dyn_cast<RankedTensorType>(firstTranspose->getOperand(0).getType());
    auto transposedType =
        dyn_cast<RankedTensorType>(firstTranspose->getResult(0).getType());
    auto constantType = dyn_cast<RankedTensorType>(constant.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (failed(constantAttr) || failed(firstPermutation) ||
        failed(secondPermutation) || !sourceType || !transposedType ||
        !constantType || !resultType || !sourceType.hasStaticShape() ||
        !transposedType.hasStaticShape() || !constantType.hasStaticShape() ||
        !resultType.hasStaticShape() || !sourceType.getElementType().isF32() ||
        sourceType.getRank() < 2 || sourceType.getRank() > 5 ||
        transposedType.getRank() != sourceType.getRank() ||
        constantType.getRank() != sourceType.getRank() ||
        resultType.getRank() != sourceType.getRank() ||
        !isPermutation(*firstPermutation, sourceType.getRank()) ||
        !isPermutation(*secondPermutation, sourceType.getRank()))
      return failure();

    int64_t rank = sourceType.getRank();
    SmallVector<int64_t> inverseFirst(rank);
    for (int64_t outputAxis = 0; outputAxis < rank; ++outputAxis)
      inverseFirst[(*firstPermutation)[outputAxis]] = outputAxis;
    SmallVector<int64_t> sourceConstantShape(rank);
    for (int64_t sourceAxis = 0; sourceAxis < rank; ++sourceAxis) {
      int64_t outputAxis = inverseFirst[sourceAxis];
      int64_t extent = constantType.getShape()[outputAxis];
      if (extent != 1 && extent != transposedType.getShape()[outputAxis])
        return failure();
      sourceConstantShape[sourceAxis] = extent;
      if (extent != 1 && extent != sourceType.getShape()[sourceAxis])
        return failure();
    }
    auto sourceConstantType =
        RankedTensorType::get(sourceConstantShape, sourceType.getElementType());
    FailureOr<DenseElementsAttr> sourceConstant = transposeBroadcastConstant(
        *constantAttr, *firstPermutation, sourceConstantType);
    if (failed(sourceConstant))
      return failure();

    SmallVector<int64_t> composed(rank);
    for (int64_t axis = 0; axis < rank; ++axis)
      composed[axis] = (*firstPermutation)[(*secondPermutation)[axis]];
    for (int64_t axis = 0; axis < rank; ++axis)
      if (resultType.getShape()[axis] != sourceType.getShape()[composed[axis]])
        return failure();

    Value newConstant = arith::ConstantOp::create(
        rewriter, op->getLoc(), sourceConstantType, *sourceConstant);
    OperationState addState(op->getLoc(), "tfl.add");
    addState.addOperands({firstTranspose->getOperand(0), newConstant});
    addState.addTypes(sourceType);
    addState.addAttributes(add->getAttrs());
    Value replacement = rewriter.create(addState)->getResult(0);
    bool identity = true;
    for (int64_t axis = 0; axis < rank; ++axis)
      identity &= composed[axis] == axis;
    if (!identity) {
      auto permutationType =
          RankedTensorType::get({rank}, rewriter.getI32Type());
      SmallVector<int32_t> narrowed;
      for (int64_t axis : composed)
        narrowed.push_back(static_cast<int32_t>(axis));
      Value permutation =
          arith::ConstantOp::create(rewriter, op->getLoc(), permutationType,
              DenseIntElementsAttr::get(permutationType, narrowed));
      OperationState transposeState(op->getLoc(), "tfl.transpose");
      transposeState.addOperands({replacement, permutation});
      transposeState.addTypes(resultType);
      replacement = rewriter.create(transposeState)->getResult(0);
    } else if (replacement.getType() != resultType) {
      return failure();
    }

    rewriter.replaceOp(op, replacement);
    if (add->use_empty())
      rewriter.eraseOp(add);
    if (firstTranspose->use_empty())
      rewriter.eraseOp(firstTranspose);
    return success();
  }
};

class FoldBroadcastBatchIntoBatchMatMulPattern final : public RewritePattern {
public:
  explicit FoldBroadcastBatchIntoBatchMatMulPattern(MLIRContext *context)
      : RewritePattern("tfl.batch_matmul", 2, context) {}

  static bool preservesLinearOrder(
      ArrayRef<Operation *> viewOperations, int64_t elementCount) {
    constexpr int64_t maxCheckedElements = 4 * 1024 * 1024;
    if (elementCount <= 0 || elementCount > maxCheckedElements)
      return false;
    bool hasTranspose = false;
    for (Operation *operation : viewOperations) {
      auto inputType =
          dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto outputType =
          dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!inputType || !outputType || !inputType.hasStaticShape() ||
          !outputType.hasStaticShape() ||
          inputType.getNumElements() != elementCount ||
          outputType.getNumElements() != elementCount)
        return false;
      hasTranspose |= operation->getName().getStringRef() == "tfl.transpose";
    }
    if (!hasTranspose)
      return true;

    for (int64_t finalIndex = 0; finalIndex < elementCount; ++finalIndex) {
      int64_t currentIndex = finalIndex;
      for (Operation *operation : viewOperations) {
        auto inputType =
            dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
        auto outputType =
            dyn_cast<RankedTensorType>(operation->getResult(0).getType());
        if (!inputType || !outputType || !inputType.hasStaticShape() ||
            !outputType.hasStaticShape() ||
            inputType.getNumElements() != elementCount ||
            outputType.getNumElements() != elementCount)
          return false;
        if (operation->getName().getStringRef() == "tfl.reshape")
          continue;

        FailureOr<SmallVector<int64_t>> permutation =
            getConstantIntValues(operation->getOperand(1));
        int64_t rank = outputType.getRank();
        if (failed(permutation) || !isPermutation(*permutation, rank) ||
            inputType.getRank() != rank)
          return false;
        SmallVector<int64_t> outputStrides(rank, 1), inputStrides(rank, 1);
        for (int64_t axis = rank - 2; axis >= 0; --axis) {
          outputStrides[axis] =
              outputStrides[axis + 1] * outputType.getShape()[axis + 1];
          inputStrides[axis] =
              inputStrides[axis + 1] * inputType.getShape()[axis + 1];
        }
        int64_t remaining = currentIndex;
        int64_t inputIndex = 0;
        for (int64_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
          int64_t coordinate = remaining / outputStrides[outputAxis];
          remaining %= outputStrides[outputAxis];
          inputIndex += coordinate * inputStrides[(*permutation)[outputAxis]];
        }
        currentIndex = inputIndex;
      }
      if (currentIndex != finalIndex)
        return false;
    }
    return true;
  }

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    for (int64_t operandIndex = 0; operandIndex < 2; ++operandIndex) {
      Value oldOperand = op->getOperand(operandIndex);
      Value current = oldOperand;
      SmallVector<Operation *> viewOperations;
      Operation *scale = nullptr;
      Value scaleValue;
      while (Operation *definition = current.getDefiningOp()) {
        StringRef name = definition->getName().getStringRef();
        if ((name == "tfl.reshape" || name == "tfl.transpose") &&
            definition->getNumOperands() == 2 &&
            definition->getNumResults() == 1 &&
            definition->getResult(0).hasOneUse()) {
          viewOperations.push_back(definition);
          current = definition->getOperand(0);
          continue;
        }
        if (name == "tfl.mul" && !scale && definition->getNumOperands() == 2 &&
            definition->getNumResults() == 1 &&
            definition->getResult(0).hasOneUse()) {
          Value data;
          for (int64_t candidate = 0; candidate < 2; ++candidate) {
            FailureOr<DenseElementsAttr> constant =
                getDenseF32Constant(definition->getOperand(1 - candidate));
            if (succeeded(constant) && constant->getNumElements() == 1) {
              data = definition->getOperand(candidate);
              scaleValue = definition->getOperand(1 - candidate);
              break;
            }
          }
          if (data) {
            scale = definition;
            current = data;
            continue;
          }
        }
        break;
      }
      if (viewOperations.empty())
        continue;
      Operation *broadcast = current.getDefiningOp();
      if (!broadcast ||
          broadcast->getName().getStringRef() != "tfl.broadcast_to" ||
          broadcast->getNumOperands() != 2 || broadcast->getNumResults() != 1 ||
          !broadcast->getResult(0).hasOneUse())
        continue;

      Value source = broadcast->getOperand(0);
      auto sourceType = dyn_cast<RankedTensorType>(source.getType());
      auto broadcastType =
          dyn_cast<RankedTensorType>(broadcast->getResult(0).getType());
      auto flattenedType = dyn_cast<RankedTensorType>(oldOperand.getType());
      auto otherType = dyn_cast<RankedTensorType>(
          op->getOperand(1 - operandIndex).getType());
      auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!sourceType || !broadcastType || !flattenedType || !otherType ||
          !resultType || !sourceType.hasStaticShape() ||
          !broadcastType.hasStaticShape() || !flattenedType.hasStaticShape() ||
          !otherType.hasStaticShape() || !resultType.hasStaticShape() ||
          sourceType.getRank() < 3 ||
          sourceType.getRank() != broadcastType.getRank() ||
          flattenedType.getRank() != 3 || otherType.getRank() != 3 ||
          resultType.getRank() != 3 ||
          sourceType.getElementType() != flattenedType.getElementType())
        continue;

      int64_t rank = sourceType.getRank();
      int64_t broadcastBatch = 1;
      bool valid = true;
      for (int64_t axis = 0; axis < rank - 2; ++axis) {
        int64_t sourceExtent = sourceType.getShape()[axis];
        int64_t broadcastExtent = broadcastType.getShape()[axis];
        if (sourceExtent != 1 || broadcastExtent <= 0 ||
            broadcastBatch >
                std::numeric_limits<int64_t>::max() / broadcastExtent) {
          valid = false;
          break;
        }
        broadcastBatch *= broadcastExtent;
      }
      int64_t rows = sourceType.getShape()[rank - 2];
      int64_t columns = sourceType.getShape()[rank - 1];
      valid &= rows > 0 && columns > 0 &&
               rows <= std::numeric_limits<int32_t>::max() &&
               columns <= std::numeric_limits<int32_t>::max() &&
               broadcastType.getShape()[rank - 2] == rows &&
               broadcastType.getShape()[rank - 1] == columns &&
               flattenedType.getShape()[0] == broadcastBatch &&
               flattenedType.getShape()[1] == rows &&
               flattenedType.getShape()[2] == columns &&
               (otherType.getShape()[0] == broadcastBatch ||
                   otherType.getShape()[0] == 1) &&
               resultType.getShape()[0] == broadcastBatch;
      if (!valid ||
          !preservesLinearOrder(viewOperations, broadcastType.getNumElements()))
        continue;

      Operation *sourceReshape = source.getDefiningOp();
      Value replacementSource = source;
      if (sourceReshape &&
          sourceReshape->getName().getStringRef() == "tfl.reshape" &&
          sourceReshape->getNumOperands() == 2 &&
          sourceReshape->getNumResults() == 1 &&
          sourceReshape->getResult(0).hasOneUse()) {
        auto precedingType =
            dyn_cast<RankedTensorType>(sourceReshape->getOperand(0).getType());
        if (precedingType && precedingType.hasStaticShape() &&
            precedingType.getElementType() == sourceType.getElementType() &&
            precedingType.getNumElements() == sourceType.getNumElements())
          replacementSource = sourceReshape->getOperand(0);
        else
          sourceReshape = nullptr;
      } else {
        sourceReshape = nullptr;
      }

      auto replacementType = RankedTensorType::get(
          {1, rows, columns}, sourceType.getElementType());
      if (scale) {
        OperationState scaleState(op->getLoc(), "tfl.mul");
        scaleState.addOperands({replacementSource, scaleValue});
        scaleState.addTypes(replacementSource.getType());
        scaleState.addAttributes(scale->getAttrs());
        replacementSource = rewriter.create(scaleState)->getResult(0);
      }
      auto shapeType = RankedTensorType::get({3}, rewriter.getI32Type());
      Value shape = arith::ConstantOp::create(rewriter, op->getLoc(), shapeType,
          DenseIntElementsAttr::get(
              shapeType, SmallVector<int32_t>{1, static_cast<int32_t>(rows),
                             static_cast<int32_t>(columns)}));
      OperationState reshapeState(op->getLoc(), "tfl.reshape");
      reshapeState.addOperands({replacementSource, shape});
      reshapeState.addTypes(replacementType);
      Value replacement = rewriter.create(reshapeState)->getResult(0);
      rewriter.modifyOpInPlace(
          op, [&]() { op->setOperand(operandIndex, replacement); });

      for (Operation *view : viewOperations)
        rewriter.eraseOp(view);
      if (scale)
        rewriter.eraseOp(scale);
      rewriter.eraseOp(broadcast);
      if (sourceReshape && sourceReshape->use_empty())
        rewriter.eraseOp(sourceReshape);
      return success();
    }
    return failure();
  }
};

} // namespace

void populateONNXToTFLTransformerPreprocessingPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PackSplitProjectionPattern, PackSeparateProjectionPattern>(
      patterns.getContext());
}

void populateTFLTransformerPostLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<CollapseHeadLayoutTransposeChainPattern,
      FoldBroadcastBatchIntoBatchMatMulPattern,
      FoldTransposeGeluTransposePattern, FoldTransposeSoftmaxTransposePattern,
      MoveBroadcastAddBeforeTransposePattern>(patterns.getContext());
}

} // namespace onnx_mlir
