/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <limits>

using namespace mlir;

namespace onnx_mlir {
namespace {

// All rewrites operate on physical types after lowering. Restrict this first
// stage to ordinary static tensors; in particular, do not reinterpret encoded
// layouts or per-axis quantization. A failed guard leaves the graph unchanged.
RankedTensorType tensorType(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      type.getRank() > 8)
    return {};
  Type element = type.getElementType();
  if (!element.isF32() && !element.isSignlessInteger(1) &&
      !element.isSignlessInteger(32) && !element.isSignlessInteger(64))
    return {};
  for (int64_t dim : type.getShape())
    if (dim <= 0 || dim > std::numeric_limits<int32_t>::max())
      return {};
  return type;
}

bool isOp(Operation *op, StringRef name, unsigned operands) {
  return op && op->getName().getStringRef() == name &&
         op->getNumOperands() == operands && op->getNumResults() == 1 &&
         op->getNumRegions() == 0 && op->getNumSuccessors() == 0;
}

FailureOr<SmallVector<int64_t>> indexVector(Value value, int64_t length) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || type.getDimSize(0) != length ||
      type.getEncoding() || length < 0 || length > 8 ||
      (!type.getElementType().isSignlessInteger(32) &&
          !type.getElementType().isSignlessInteger(64)))
    return failure();
  return getConstantIntValues(value);
}

FailureOr<SmallVector<int64_t>> permutation(Operation *op) {
  if (!isOp(op, "tfl.transpose", 2))
    return failure();
  auto input = tensorType(op->getOperand(0));
  auto output = tensorType(op->getResult(0));
  auto permType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!input || !output || input.getRank() != output.getRank() ||
      input.getElementType() != output.getElementType() || !permType ||
      !permType.getElementType().isSignlessInteger(32))
    return failure();
  auto perm = indexVector(op->getOperand(1), input.getRank());
  if (failed(perm))
    return failure();
  SmallVector<bool> seen(input.getRank(), false);
  for (auto [axis, value] : llvm::enumerate(*perm)) {
    if (value < 0 || value >= input.getRank() || seen[value] ||
        output.getDimSize(axis) != input.getDimSize(value))
      return failure();
    seen[value] = true;
  }
  return perm;
}

bool inverse(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  for (auto [i, axis] : llvm::enumerate(b))
    if (a[axis] != static_cast<int64_t>(i))
      return false;
  return true;
}

Value constant(
    PatternRewriter &rewriter, Location loc, ArrayRef<int64_t> values) {
  SmallVector<int32_t> narrowed(values.begin(), values.end());
  auto type = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter.getI32Type());
  return arith::ConstantOp::create(
      rewriter, loc, type, DenseIntElementsAttr::get(type, narrowed));
}

Value createLayoutOp(PatternRewriter &rewriter, Location loc, StringRef name,
    Type type, ValueRange inputs, ArrayRef<NamedAttribute> attrs = {}) {
  OperationState state(loc, name);
  state.addOperands(inputs);
  state.addTypes(type);
  state.addAttributes(attrs);
  return rewriter.create(state)->getResult(0);
}

// Compatibility operations have no generated side-effect traits. Only erase
// the explicitly matched, validated data-movement nodes, and only when dead.
void eraseIfDead(PatternRewriter &rewriter, Operation *op) {
  if (op->use_empty())
    rewriter.eraseOp(op);
}

struct Window {
  SmallVector<int64_t> begin, size;
};

FailureOr<Window> sliceWindow(Operation *op) {
  if (!isOp(op, "tfl.slice", 3))
    return failure();
  auto input = tensorType(op->getOperand(0));
  auto output = tensorType(op->getResult(0));
  if (!input || !output || input.getRank() != output.getRank() ||
      input.getElementType() != output.getElementType())
    return failure();
  auto begin = indexVector(op->getOperand(1), input.getRank());
  auto size = indexVector(op->getOperand(2), input.getRank());
  if (failed(begin) || failed(size))
    return failure();
  for (int64_t i = 0; i < input.getRank(); ++i) {
    int64_t dim = input.getDimSize(i), start = (*begin)[i];
    if (start < 0 || start >= dim)
      return failure();
    if ((*size)[i] == -1)
      (*size)[i] = dim - start;
    if ((*size)[i] <= 0 || (*size)[i] > dim - start ||
        (*size)[i] != output.getDimSize(i))
      return failure();
  }
  return Window{*begin, *size};
}

// Fold a one-spatial-axis channel shuffle into a constant, ungrouped filter.
// [N,1,L,C*S] -> [N,C*S,1,L] -> [N,C,S,L] -> [N,C,L,S]
// -> [N,C,1,L*S] becomes a filter permutation followed by reshape + transpose.
// A following NHWC transpose is then canceled by ComposeTransposes.
class FoldConvChannelShuffle final : public RewritePattern {
public:
  explicit FoldConvChannelShuffle(MLIRContext *ctx)
      : RewritePattern("tfl.reshape", 6, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isOp(op, "tfl.reshape", 2))
      return failure();
    Operation *swap = op->getOperand(0).getDefiningOp();
    auto q = permutation(swap);
    if (failed(q) || *q != SmallVector<int64_t>({0, 1, 3, 2}))
      return failure();
    Operation *split = swap->getOperand(0).getDefiningOp();
    if (!isOp(split, "tfl.reshape", 2))
      return failure();
    Operation *toChannelsFirst = split->getOperand(0).getDefiningOp();
    auto p = permutation(toChannelsFirst);
    if (failed(p) || *p != SmallVector<int64_t>({0, 3, 1, 2}))
      return failure();
    Operation *conv = toChannelsFirst->getOperand(0).getDefiningOp();
    if (!isOp(conv, "tfl.conv_2d", 3))
      return failure();
    for (Operation *node : {conv, toChannelsFirst, split, swap})
      if (!node->getResult(0).hasOneUse())
        return failure();
    auto in = tensorType(conv->getOperand(0));
    auto weights = tensorType(conv->getOperand(1));
    auto bias = tensorType(conv->getOperand(2));
    auto a = tensorType(conv->getResult(0));
    auto b = tensorType(split->getResult(0));
    auto out = tensorType(op->getResult(0));
    if (!in || !weights || !bias || !a || !b || !out || in.getRank() != 4 ||
        weights.getRank() != 4 || bias.getRank() != 1 || a.getRank() != 4 ||
        b.getRank() != 4 || out.getRank() != 4 ||
        !in.getElementType().isF32() || !weights.getElementType().isF32() ||
        !bias.getElementType().isF32() || !a.getElementType().isF32() ||
        !b.getElementType().isF32() || !out.getElementType().isF32() ||
        a.getDimSize(1) != 1 || in.getDimSize(0) != a.getDimSize(0) ||
        weights.getDimSize(3) != in.getDimSize(3) ||
        weights.getDimSize(0) != a.getDimSize(3) ||
        bias.getDimSize(0) != a.getDimSize(3))
      return failure(); // Grouped/depthwise channel partitions cannot move.
    int64_t n = a.getDimSize(0), l = a.getDimSize(2);
    int64_t c = b.getDimSize(1), s = b.getDimSize(2);
    if (s <= 1 || c <= 1 || b.getDimSize(0) != n || b.getDimSize(3) != l ||
        c * s != a.getDimSize(3) ||
        out.getShape() != ArrayRef<int64_t>({n, c, 1, l * s}))
      return failure();
    auto splitShape = indexVector(split->getOperand(1), 4);
    auto outputShape = indexVector(op->getOperand(1), 4);
    if (failed(splitShape) || failed(outputShape) ||
        !llvm::equal(*splitShape, b.getShape()) ||
        !llvm::equal(*outputShape, out.getShape()))
      return failure();
    // Bound compile-time copying, and require ordinary dense FP32 constants.
    constexpr int64_t maxFilterElements = 16 * 1024 * 1024;
    int64_t filterElements = 1;
    for (int64_t dim : weights.getShape()) {
      if (dim > maxFilterElements / filterElements)
        return failure();
      filterElements *= dim;
    }
    auto w = getConstantF32Values(conv->getOperand(1));
    auto v = getConstantF32Values(conv->getOperand(2));
    if (failed(w) || failed(v) ||
        static_cast<int64_t>(w->size()) != weights.getNumElements() ||
        static_cast<int64_t>(v->size()) != bias.getNumElements())
      return failure();
    SmallVector<float> reorderedWeights(w->size()), reorderedBias(v->size());
    int64_t row = weights.getNumElements() / (c * s);
    for (int64_t ci = 0; ci < c; ++ci)
      for (int64_t si = 0; si < s; ++si) {
        int64_t oldChannel = ci * s + si, newChannel = si * c + ci;
        std::copy_n(w->begin() + oldChannel * row, row,
            reorderedWeights.begin() + newChannel * row);
        reorderedBias[newChannel] = (*v)[oldChannel];
      }
    Value filter = arith::ConstantOp::create(rewriter, op->getLoc(), weights,
        DenseElementsAttr::get(weights, ArrayRef<float>(reorderedWeights)));
    Value offset = arith::ConstantOp::create(rewriter, op->getLoc(), bias,
        DenseElementsAttr::get(bias, ArrayRef<float>(reorderedBias)));
    Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.conv_2d", a,
        ValueRange{conv->getOperand(0), filter, offset}, conv->getAttrs());
    auto physical =
        RankedTensorType::get({n, 1, l * s, c}, out.getElementType());
    Value shape = constant(rewriter, op->getLoc(), physical.getShape());
    result = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", physical,
        ValueRange{result, shape});
    Value perm = constant(rewriter, op->getLoc(), {0, 3, 1, 2});
    result = createLayoutOp(
        rewriter, op->getLoc(), "tfl.transpose", out, ValueRange{result, perm});
    rewriter.replaceOp(op, result);
    for (Operation *node : {swap, split, toChannelsFirst, conv})
      rewriter.eraseOp(node);
    return success();
  }
};

class ComposeTransposes final : public RewritePattern {
public:
  explicit ComposeTransposes(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 5, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q))
      return failure();
    Operation *first = op->getOperand(0).getDefiningOp();
    auto p = permutation(first);
    if (failed(p))
      return failure();
    if (inverse(*p, *q)) {
      rewriter.replaceOp(op, first->getOperand(0));
    } else {
      if (!first->getResult(0).hasOneUse())
        return failure();
      SmallVector<int64_t> combined;
      for (int64_t axis : *q)
        combined.push_back((*p)[axis]);
      Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.transpose",
          op->getResult(0).getType(),
          ValueRange{first->getOperand(0),
              constant(rewriter, op->getLoc(), combined)});
      rewriter.replaceOp(op, result);
    }
    eraseIfDead(rewriter, first);
    return success();
  }
};

bool isPointwise(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  bool binary = name == "tfl.add" || name == "tfl.sub" || name == "tfl.mul" ||
                name == "tfl.div" || name == "tfl.maximum" ||
                name == "tfl.minimum";
  bool unary = name == "tfl.relu" || name == "tfl.relu6" ||
               name == "tfl.tanh" || name == "tfl.logistic" ||
               name == "tfl.leaky_relu";
  return (binary || unary) && isOp(op, name, binary ? 2 : 1);
}

// Select moves elements without arithmetic: preserve NaN payloads and signed
// zeros by moving the boolean mask as well, never replacing selection by Mul.
// Only eliminate copies when every resulting operand permutation is a view
// (possibly after canceling an existing transpose). Arbitrary masks that need
// a new physical transpose and shared Select results retain the old graph.
class CommuteSelectTranspose final : public RewritePattern {
public:
  explicit CommuteSelectTranspose(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 4, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q) || q->size() > 4)
      return failure();
    Operation *select = op->getOperand(0).getDefiningOp();
    if (!isOp(select, "tfl.select_v2", 3) || !select->getResult(0).hasOneUse())
      return failure();
    auto from = tensorType(select->getResult(0));
    auto to = tensorType(op->getResult(0));
    SmallVector<std::pair<Value, RankedTensorType>> plans;
    SmallVector<Operation *> obsolete;
    SmallVector<int64_t> before(from.getRank(), 1), after(to.getRank(), 1);
    auto remember = [&](Operation *node) {
      if (!llvm::is_contained(obsolete, node))
        obsolete.push_back(node);
    };
    for (auto [number, operand] : llvm::enumerate(select->getOperands())) {
      auto type = tensorType(operand);
      Type element = number == 0 ? rewriter.getI1Type() : from.getElementType();
      if (!type || type.getElementType() != element ||
          type.getRank() > from.getRank())
        return failure();
      int64_t offset = from.getRank() - type.getRank();
      for (int64_t i = 0; i < type.getRank(); ++i) {
        int64_t dim = type.getDimSize(i), axis = offset + i;
        if (dim != 1 && dim != from.getDimSize(axis))
          return failure();
        before[axis] = std::max(before[axis], dim);
      }

      Operation *broadcast = operand.getDefiningOp();
      if (isOp(broadcast, "tfl.broadcast_to", 2)) {
        auto base = tensorType(broadcast->getOperand(0));
        auto shape = indexVector(broadcast->getOperand(1), type.getRank());
        if (!base || base.getElementType() != element ||
            base.getRank() > type.getRank() || failed(shape) ||
            !llvm::equal(*shape, type.getShape()))
          return failure();
        int64_t shift = type.getRank() - base.getRank();
        for (int64_t i = 0; i < base.getRank(); ++i)
          if (base.getDimSize(i) != 1 &&
              base.getDimSize(i) != type.getDimSize(shift + i))
            return failure();
        remember(broadcast);
        operand = broadcast->getOperand(0);
        type = base;
      }
      SmallVector<int64_t> aligned(from.getRank(), 1),
          sourceAxes(from.getRank(), -1);
      offset = from.getRank() - type.getRank();
      for (int64_t i = 0; i < type.getRank(); ++i) {
        aligned[offset + i] = type.getDimSize(i);
        sourceAxes[offset + i] = i;
      }
      Operation *first = operand.getDefiningOp();
      auto p = permutation(first);
      if (succeeded(p)) {
        for (int64_t i = 0; i < type.getRank(); ++i)
          sourceAxes[offset + i] = (*p)[i];
        remember(first);
        operand = first->getOperand(0);
        type = tensorType(operand);
      }
      SmallVector<int64_t> shape;
      int64_t last = -1;
      for (int64_t axis : *q) {
        shape.push_back(aligned[axis]);
        if (aligned[axis] != 1) {
          if (sourceAxes[axis] <= last)
            return failure(); // A real data permutation is still required.
          last = sourceAxes[axis];
        }
      }
      for (int64_t i = 0; i < to.getRank(); ++i)
        after[i] = std::max(after[i], shape[i]);
      auto target =
          type.getRank() == 0 ? type : RankedTensorType::get(shape, element);
      plans.emplace_back(operand, target);
    }
    // Removing a broadcast must not remove dimensions needed to infer the
    // Select output; e.g. three scalars cannot replace an expanded tensor.
    int64_t resultRank = 0;
    for (const auto &plan : plans)
      resultRank = std::max(resultRank, plan.second.getRank());
    if (before != from.getShape() || after != to.getShape() ||
        resultRank != to.getRank())
      return failure();
    SmallVector<Value> inputs;
    for (auto [value, type] : plans) {
      if (value.getType() != type)
        value = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", type,
            {value, constant(rewriter, op->getLoc(), type.getShape())});
      inputs.push_back(value);
    }
    Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.select_v2", to,
        inputs, select->getAttrs());
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(select);
    for (Operation *node : obsolete)
      eraseIfDead(rewriter, node);
    return success();
  }
};

class SingletonGatherToReshape final : public RewritePattern {
public:
  explicit SingletonGatherToReshape(MLIRContext *ctx)
      : RewritePattern("tfl.gather", 4, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isOp(op, "tfl.gather", 2))
      return failure();
    auto input = tensorType(op->getOperand(0));
    auto indices = tensorType(op->getOperand(1));
    auto output = tensorType(op->getResult(0));
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    auto batchAttr = op->getAttrOfType<IntegerAttr>("batch_dims");
    if (!input || !indices || !output || !axisAttr || !batchAttr ||
        batchAttr.getInt() != 0 || indices.getRank() != 0 ||
        (!indices.getElementType().isSignlessInteger(32) &&
            !indices.getElementType().isSignlessInteger(64)) ||
        input.getElementType() != output.getElementType() ||
        output.getRank() + 1 != input.getRank())
      return failure();
    int64_t axis = normalizeAxis(axisAttr.getInt(), input.getRank());
    if (axis < 0 || axis >= input.getRank() || input.getDimSize(axis) != 1)
      return failure();
    auto values = getConstantIntValues(op->getOperand(1));
    if (failed(values) || values->size() != 1 || (*values)[0] != 0)
      return failure();
    SmallVector<int64_t> shape(input.getShape());
    shape.erase(shape.begin() + axis);
    if (shape != output.getShape())
      return failure();
    Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", output,
        {op->getOperand(0), constant(rewriter, op->getLoc(), shape)});
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Move a permutation through pointwise arithmetic only when it eliminates
// layout copies without increasing the number of view operations. Broadcast
// dimensions are right-aligned before permutation; equal extents alone do not
// imply that two axes can be interchanged. No arithmetic is reassociated.
class CommutePointwiseTranspose final : public RewritePattern {
public:
  explicit CommutePointwiseTranspose(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 3, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q) || q->size() > 4)
      return failure();
    Operation *middle = op->getOperand(0).getDefiningOp();
    if (!isPointwise(middle) || !middle->getResult(0).hasOneUse())
      return failure();
    auto output = tensorType(middle->getResult(0));
    if (!output || !output.getElementType().isF32())
      return failure();

    struct OperandPlan {
      Value value;
      RankedTensorType type;
      bool transpose;
    };
    SmallVector<OperandPlan> plans;
    SmallVector<Operation *> removable;
    SmallVector<int64_t> broadcast(output.getRank(), 1);
    int64_t newViews = 0, newCopies = 0;
    for (Value operand : middle->getOperands()) {
      auto type = tensorType(operand);
      if (!type || type.getElementType() != output.getElementType() ||
          type.getRank() > output.getRank())
        return failure();
      SmallVector<int64_t> aligned(output.getRank(), 1);
      int64_t offset = output.getRank() - type.getRank();
      for (int64_t i = 0; i < type.getRank(); ++i) {
        int64_t dim = type.getDimSize(i), axis = offset + i;
        if (dim != 1 && dim != output.getDimSize(axis))
          return failure();
        broadcast[axis] = std::max(broadcast[axis], dim);
        aligned[axis] = dim;
      }
      auto *first = operand.getDefiningOp();
      auto p = permutation(first);
      if (succeeded(p) && inverse(*p, *q)) {
        plans.push_back(
            {first->getOperand(0), tensorType(first->getOperand(0)), false});
        if (llvm::all_of(operand.getUsers(),
                [&](Operation *user) { return user == middle; }) &&
            !llvm::is_contained(removable, first))
          removable.push_back(first);
        continue;
      }
      if (type.getRank() == 0) {
        plans.push_back({operand, type, false});
        continue;
      }
      SmallVector<int64_t> shape;
      int64_t lastRealAxis = -1;
      bool copy = false;
      for (int64_t axis : *q) {
        shape.push_back(aligned[axis]);
        if (aligned[axis] != 1) {
          copy |= axis < lastRealAxis;
          lastRealAxis = axis;
        }
      }
      // Avoid introducing an extra rank-promotion view before a real copy.
      if (copy && type.getRank() != output.getRank())
        return failure();
      auto target = RankedTensorType::get(shape, type.getElementType());
      plans.push_back({operand, target, copy});
      newCopies += copy;
      newViews += copy || target != type;
    }
    if (ArrayRef<int64_t>(broadcast) != output.getShape() ||
        newCopies >= 1 + static_cast<int64_t>(removable.size()) ||
        newViews > 1 + static_cast<int64_t>(removable.size()))
      return failure();

    SmallVector<Value> inputs;
    for (const auto &plan : plans) {
      Value value = plan.value;
      if (plan.transpose || value.getType() != plan.type) {
        Value parameter = constant(rewriter, op->getLoc(),
            plan.transpose ? ArrayRef<int64_t>(*q) : plan.type.getShape());
        value = createLayoutOp(rewriter, op->getLoc(),
            plan.transpose ? "tfl.transpose" : "tfl.reshape", plan.type,
            ValueRange{value, parameter});
      }
      inputs.push_back(value);
    }
    Value result =
        createLayoutOp(rewriter, op->getLoc(), middle->getName().getStringRef(),
            op->getResult(0).getType(), inputs, middle->getAttrs());
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(middle);
    for (Operation *first : removable)
      eraseIfDead(rewriter, first);
    return success();
  }
};

class CommutePadTranspose final : public RewritePattern {
public:
  explicit CommutePadTranspose(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 4, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q) || q->size() > 4)
      return failure();
    Operation *pad = op->getOperand(0).getDefiningOp();
    bool explicitValue = isOp(pad, "tfl.padv2", 3);
    if ((!explicitValue && !isOp(pad, "tfl.pad", 2)) ||
        !pad->getResult(0).hasOneUse())
      return failure();
    Operation *first = pad->getOperand(0).getDefiningOp();
    auto p = permutation(first);
    if (failed(p) || !inverse(*p, *q))
      return failure();
    auto input = tensorType(pad->getOperand(0));
    auto output = tensorType(pad->getResult(0));
    auto padsType = tensorType(pad->getOperand(1));
    if (!input || !output || !input.getElementType().isF32() ||
        output.getElementType() != input.getElementType() ||
        input.getRank() != output.getRank() || !padsType ||
        padsType.getRank() != 2 || padsType.getDimSize(0) != input.getRank() ||
        padsType.getDimSize(1) != 2 ||
        !padsType.getElementType().isSignlessInteger(32))
      return failure();
    if (explicitValue) {
      auto scalar = tensorType(pad->getOperand(2));
      if (!scalar || scalar.getRank() != 0 ||
          scalar.getElementType() != input.getElementType())
        return failure();
    }
    auto pads = getConstantIntValues(pad->getOperand(1));
    if (failed(pads) || pads->size() != 2 * q->size())
      return failure();
    for (int64_t i = 0; i < input.getRank(); ++i) {
      int64_t before = (*pads)[2 * i], after = (*pads)[2 * i + 1];
      if (before < 0 || after < 0 ||
          before + after != output.getDimSize(i) - input.getDimSize(i))
        return failure();
    }
    SmallVector<int32_t> reordered;
    for (int64_t axis : *q) {
      reordered.push_back((*pads)[2 * axis]);
      reordered.push_back((*pads)[2 * axis + 1]);
    }
    Value padding = arith::ConstantOp::create(rewriter, op->getLoc(), padsType,
        DenseIntElementsAttr::get(padsType, reordered));
    SmallVector<Value> inputs{first->getOperand(0), padding};
    if (explicitValue)
      inputs.push_back(pad->getOperand(2));
    Value result =
        createLayoutOp(rewriter, op->getLoc(), pad->getName().getStringRef(),
            op->getResult(0).getType(), inputs, pad->getAttrs());
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(pad);
    eraseIfDead(rewriter, first);
    return success();
  }
};

class CommuteWindowOrTile final : public RewritePattern {
public:
  explicit CommuteWindowOrTile(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 4, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q))
      return failure();
    Operation *middle = op->getOperand(0).getDefiningOp();
    bool tile = isOp(middle, "tfl.tile", 2);
    if ((!tile && !isOp(middle, "tfl.slice", 3)) ||
        !middle->getResult(0).hasOneUse())
      return failure();
    Operation *first = middle->getOperand(0).getDefiningOp();
    auto p = permutation(first);
    if (failed(p) || !inverse(*p, *q))
      return failure();
    auto input = tensorType(middle->getOperand(0));
    auto output = tensorType(middle->getResult(0));
    if (!input || !output || input.getRank() != output.getRank() ||
        input.getElementType() != output.getElementType())
      return failure();
    SmallVector<SmallVector<int64_t>> parameters;
    if (tile) {
      auto repeats = indexVector(middle->getOperand(1), input.getRank());
      if (failed(repeats))
        return failure();
      for (int64_t i = 0; i < input.getRank(); ++i) {
        int64_t repeat = (*repeats)[i], dim = input.getDimSize(i);
        if (repeat <= 0 || repeat > std::numeric_limits<int32_t>::max() / dim ||
            dim * repeat != output.getDimSize(i))
          return failure();
      }
      parameters.push_back(*repeats);
    } else {
      auto window = sliceWindow(middle);
      if (failed(window))
        return failure();
      parameters = {window->begin, window->size};
    }
    SmallVector<Value> inputs{first->getOperand(0)};
    for (const auto &parameter : parameters) {
      SmallVector<int64_t> reordered;
      for (int64_t axis : *q)
        reordered.push_back(parameter[axis]);
      inputs.push_back(constant(rewriter, op->getLoc(), reordered));
    }
    Value result =
        createLayoutOp(rewriter, op->getLoc(), middle->getName().getStringRef(),
            op->getResult(0).getType(), inputs, middle->getAttrs());
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(middle);
    eraseIfDead(rewriter, first);
    return success();
  }
};

class ComposeSlices final : public RewritePattern {
public:
  explicit ComposeSlices(MLIRContext *ctx)
      : RewritePattern("tfl.slice", 3, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto outer = sliceWindow(op);
    if (failed(outer))
      return failure();
    Operation *first = op->getOperand(0).getDefiningOp();
    auto inner = sliceWindow(first);
    if (failed(inner))
      return failure();
    // Normalized sizes are essential: a -1 in the outer slice denotes the
    // tail of the inner window, not the tail of the original tensor.
    SmallVector<int64_t> begin;
    for (auto [a, b] : llvm::zip(inner->begin, outer->begin))
      begin.push_back(a + b); // Bounds above imply a+b < input dimension.
    Value start = constant(rewriter, op->getLoc(), begin);
    Value size = constant(rewriter, op->getLoc(), outer->size);
    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperands({first->getOperand(0), start, size});
    });
    eraseIfDead(rewriter, first);
    return success();
  }
};

SmallVector<int64_t> nonSingletonAxes(RankedTensorType type) {
  SmallVector<int64_t> axes;
  for (int64_t i = 0; i < type.getRank(); ++i)
    if (type.getDimSize(i) != 1)
      axes.push_back(i);
  return axes;
}

SmallVector<int64_t> nonSingletonShape(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result;
  for (int64_t dim : shape)
    if (dim != 1)
      result.push_back(dim);
  return result;
}

bool isSingletonReshape(Operation *op) {
  if (!isOp(op, "tfl.reshape", 2))
    return false;
  auto input = tensorType(op->getOperand(0));
  auto output = tensorType(op->getResult(0));
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  auto shape = indexVector(op->getOperand(1), output.getRank());
  return succeeded(shape) && llvm::equal(*shape, output.getShape()) &&
         nonSingletonShape(input.getShape()) ==
             nonSingletonShape(output.getShape());
}

// Unit-axis views commute with pointwise operations if each broadcast axis is
// mapped by position among the non-unit axes. Require a strict reduction in
// views and do not duplicate shared computation or change reduction axes.
class CommutePointwiseReshape final : public RewritePattern {
public:
  explicit CommutePointwiseReshape(MLIRContext *ctx)
      : RewritePattern("tfl.reshape", 3, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isSingletonReshape(op))
      return failure();
    Operation *middle = op->getOperand(0).getDefiningOp();
    if (!isPointwise(middle) || !middle->getResult(0).hasOneUse())
      return failure();
    auto from = tensorType(middle->getResult(0));
    auto to = tensorType(op->getResult(0));
    if (!from.getElementType().isF32() || from.getRank() > 4 ||
        to.getRank() > 4)
      return failure();
    auto oldAxes = nonSingletonAxes(from), newAxes = nonSingletonAxes(to);
    SmallVector<std::pair<Value, RankedTensorType>> plans;
    SmallVector<Operation *> removable;
    SmallVector<int64_t> broadcast(from.getRank(), 1);
    int64_t newViews = 0;
    for (Value operand : middle->getOperands()) {
      auto type = tensorType(operand);
      if (!type || type.getElementType() != from.getElementType() ||
          type.getRank() > from.getRank())
        return failure();
      SmallVector<int64_t> aligned(from.getRank(), 1);
      int64_t offset = from.getRank() - type.getRank();
      for (int64_t i = 0; i < type.getRank(); ++i) {
        int64_t dim = type.getDimSize(i), axis = offset + i;
        if (dim != 1 && dim != from.getDimSize(axis))
          return failure();
        broadcast[axis] = std::max(broadcast[axis], dim);
        aligned[axis] = dim;
      }
      if (type.getRank() == 0) {
        plans.emplace_back(operand, type);
        continue;
      }
      SmallVector<int64_t> shape(to.getRank(), 1);
      for (auto [oldAxis, newAxis] : llvm::zip(oldAxes, newAxes))
        shape[newAxis] = aligned[oldAxis];
      auto target = RankedTensorType::get(shape, type.getElementType());
      Operation *first = operand.getDefiningOp();
      if (isSingletonReshape(first) &&
          first->getOperand(0).getType() == target) {
        plans.emplace_back(first->getOperand(0), target);
        if (llvm::all_of(operand.getUsers(),
                [&](Operation *user) { return user == middle; }) &&
            !llvm::is_contained(removable, first))
          removable.push_back(first);
      } else {
        plans.emplace_back(operand, target);
        newViews += type != target;
      }
    }
    // Scalar operands alone still infer a scalar result, not a rank-one
    // singleton tensor. Keep the explicit boundary view in that case.
    int64_t resultRank = 0;
    for (const auto &plan : plans)
      resultRank = std::max(resultRank, plan.second.getRank());
    if (resultRank != to.getRank() ||
        ArrayRef<int64_t>(broadcast) != from.getShape() ||
        newViews >= 1 + static_cast<int64_t>(removable.size()))
      return failure();
    SmallVector<Value> inputs;
    for (auto [value, type] : plans) {
      if (value.getType() != type)
        value = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", type,
            ValueRange{
                value, constant(rewriter, op->getLoc(), type.getShape())});
      inputs.push_back(value);
    }
    Value result = createLayoutOp(rewriter, op->getLoc(),
        middle->getName().getStringRef(), to, inputs, middle->getAttrs());
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(middle);
    for (Operation *first : removable)
      eraseIfDead(rewriter, first);
    return success();
  }
};

// Move removal of singleton axes before a transpose. Keep the identities of
// the remaining axes (not merely their extents), and never duplicate a shared
// transpose. The result has the requested shape without a high-rank restore.
class ReduceSingletonTranspose final : public RewritePattern {
public:
  explicit ReduceSingletonTranspose(MLIRContext *ctx)
      : RewritePattern("tfl.reshape", 3, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isSingletonReshape(op))
      return failure();
    Operation *first = op->getOperand(0).getDefiningOp();
    auto perm = permutation(first);
    if (failed(perm) || !first->getResult(0).hasOneUse())
      return failure();
    auto input = tensorType(first->getOperand(0));
    auto middle = tensorType(first->getResult(0));
    auto output = tensorType(op->getResult(0));
    if (output.getRank() > 4 || output.getRank() >= input.getRank())
      return failure();
    SmallVector<int64_t> retained;
    int64_t cursor = 0;
    for (int64_t dim : output.getShape()) {
      while (cursor < middle.getRank() && middle.getDimSize(cursor) != dim &&
             middle.getDimSize(cursor) == 1)
        ++cursor;
      if (cursor == middle.getRank() || middle.getDimSize(cursor) != dim)
        return failure();
      retained.push_back((*perm)[cursor++]);
    }
    for (; cursor < middle.getRank(); ++cursor)
      if (middle.getDimSize(cursor) != 1)
        return failure();

    SmallVector<int64_t> shape, mapping(input.getRank(), -1), reducedPerm;
    for (int64_t axis = 0; axis < input.getRank(); ++axis)
      if (llvm::is_contained(retained, axis)) {
        mapping[axis] = shape.size();
        shape.push_back(input.getDimSize(axis));
      }
    bool identity = true;
    for (auto [i, axis] : llvm::enumerate(retained)) {
      reducedPerm.push_back(mapping[axis]);
      identity &= mapping[axis] == static_cast<int64_t>(i);
    }
    Value target = constant(rewriter, op->getLoc(), shape);
    auto type = RankedTensorType::get(shape, input.getElementType());
    Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", type,
        ValueRange{first->getOperand(0), target});
    if (!identity) {
      Value reduced = constant(rewriter, op->getLoc(), reducedPerm);
      result = createLayoutOp(rewriter, op->getLoc(), "tfl.transpose", output,
          ValueRange{result, reduced});
    }
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(first);
    return success();
  }
};

// Hoist singleton-only branch reshapes after Concat. Equal non-unit prefixes
// and suffixes prove identical row-major copy boundaries, including a concat
// axis of length one. Splits/merges of real axes and shared views do not match.
// A branch transpose that only moves unit axes is another zero-copy view.
class HoistConcatSingletonReshapes final : public RewritePattern {
public:
  explicit HoistConcatSingletonReshapes(MLIRContext *ctx)
      : RewritePattern("tfl.concatenation", 3, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 1 ||
        op->getNumRegions() || op->getNumSuccessors())
      return failure();
    auto output = tensorType(op->getResult(0));
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    auto activation =
        op->getAttrOfType<StringAttr>("fused_activation_function");
    if (!output || !axisAttr || !axisAttr.getType().isSignlessInteger(32) ||
        !activation || activation.getValue() != "NONE")
      return failure();
    int64_t axis = axisAttr.getInt();
    if (axis < 0 || axis >= output.getRank())
      return failure();
    SmallVector<Operation *> reshapes;
    SmallVector<Value> sources;
    int64_t total = 0;
    for (Value operand : op->getOperands()) {
      if (!operand.hasOneUse())
        return failure();
      Operation *reshape = operand.getDefiningOp();
      if (auto perm = permutation(reshape); succeeded(perm)) {
        auto type = tensorType(reshape->getOperand(0));
        SmallVector<int64_t> order;
        for (int64_t axis : *perm)
          if (type.getDimSize(axis) != 1)
            order.push_back(axis);
        if (order != nonSingletonAxes(type) ||
            !reshape->getOperand(0).hasOneUse())
          return failure();
        reshapes.push_back(reshape);
        reshape = reshape->getOperand(0).getDefiningOp();
      }
      if (!isSingletonReshape(reshape))
        return failure();
      auto type = tensorType(operand);
      if (type.getRank() != output.getRank() ||
          type.getElementType() != output.getElementType())
        return failure();
      for (int64_t i = 0; i < type.getRank(); ++i)
        if (i != axis && type.getDimSize(i) != output.getDimSize(i))
          return failure();
      if (type.getDimSize(axis) > output.getDimSize(axis) - total)
        return failure();
      total += type.getDimSize(axis);
      sources.push_back(reshape->getOperand(0));
      reshapes.push_back(reshape);
    }
    if (total != output.getDimSize(axis))
      return failure();
    auto first = tensorType(sources.front());
    if (first.getRank() > output.getRank() || first.getRank() > 4)
      return failure();
    int64_t newAxis = -1;
    for (int64_t candidate = 0; candidate < first.getRank(); ++candidate) {
      bool valid = true;
      for (auto [source, operand] : llvm::zip(sources, op->getOperands())) {
        auto a = tensorType(source), b = tensorType(operand);
        if (a.getRank() != first.getRank() ||
            a.getDimSize(candidate) != b.getDimSize(axis) ||
            nonSingletonShape(a.getShape().take_front(candidate)) !=
                nonSingletonShape(b.getShape().take_front(axis)) ||
            nonSingletonShape(a.getShape().drop_front(candidate + 1)) !=
                nonSingletonShape(b.getShape().drop_front(axis + 1))) {
          valid = false;
          break;
        }
        for (int64_t i = 0; i < first.getRank(); ++i)
          valid &= i == candidate || a.getDimSize(i) == first.getDimSize(i);
      }
      if (valid) {
        newAxis = candidate;
        break;
      }
    }
    if (newAxis < 0)
      return failure();
    SmallVector<int64_t> shape(first.getShape());
    shape[newAxis] = total;
    auto type = RankedTensorType::get(shape, output.getElementType());
    NamedAttrList attrs(op->getAttrs());
    attrs.set("axis", rewriter.getI32IntegerAttr(newAxis));
    Value result = createLayoutOp(rewriter, op->getLoc(), "tfl.concatenation",
        type, sources, attrs.getAttrs());
    if (type != output) {
      Value target = constant(rewriter, op->getLoc(), output.getShape());
      result = createLayoutOp(rewriter, op->getLoc(), "tfl.reshape", output,
          ValueRange{result, target});
    }
    rewriter.replaceOp(op, result);
    for (Operation *reshape : reshapes)
      rewriter.eraseOp(reshape);
    return success();
  }
};

class CommuteSingletonReshape final : public RewritePattern {
public:
  explicit CommuteSingletonReshape(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 2, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q))
      return failure();
    Operation *middle = op->getOperand(0).getDefiningOp();
    if (!isOp(middle, "tfl.reshape", 2) || !middle->getResult(0).hasOneUse())
      return failure();
    Operation *first = middle->getOperand(0).getDefiningOp();
    auto p = permutation(first);
    if (failed(p))
      return failure();
    auto a = tensorType(first->getOperand(0));
    auto b = tensorType(middle->getOperand(0));
    auto c = tensorType(middle->getResult(0));
    auto d = tensorType(op->getResult(0));
    if (!a || !b || !c || !d || b.getElementType() != c.getElementType())
      return failure();
    auto shape = indexVector(middle->getOperand(1), c.getRank());
    if (failed(shape) || !llvm::equal(*shape, c.getShape()))
      return failure();
    auto bAxes = nonSingletonAxes(b), cAxes = nonSingletonAxes(c);
    if (bAxes.size() != cAxes.size())
      return failure();
    SmallVector<int64_t> sourceAxis(c.getRank(), -1);
    for (auto [bi, ci] : llvm::zip(bAxes, cAxes)) {
      if (b.getDimSize(bi) != c.getDimSize(ci))
        return failure();
      sourceAxis[ci] = (*p)[bi];
    }
    // Track axis identities, not just extents: equal-sized axes may carry
    // different data. Never generalize this rule to dimension splits/merges.
    SmallVector<int64_t> order;
    for (int64_t axis : *q)
      if (sourceAxis[axis] >= 0)
        order.push_back(sourceAxis[axis]);
    bool identity = order == nonSingletonAxes(a);
    if (!identity && !first->getResult(0).hasOneUse())
      return failure(); // Do not add a second layout copy for a shared
                        // producer.
    Value result = first->getOperand(0);
    if (!identity) {
      for (int64_t i = 0; i < a.getRank(); ++i)
        if (a.getDimSize(i) == 1)
          order.push_back(i);
      SmallVector<int64_t> transposedShape;
      for (int64_t axis : order)
        transposedShape.push_back(a.getDimSize(axis));
      auto type = RankedTensorType::get(transposedShape, a.getElementType());
      Value perm = constant(rewriter, op->getLoc(), order);
      result = createLayoutOp(rewriter, op->getLoc(), "tfl.transpose", type,
          ValueRange{result, perm});
    }
    if (result.getType() != d) {
      Value target = constant(rewriter, op->getLoc(), d.getShape());
      result = createLayoutOp(
          rewriter, op->getLoc(), "tfl.reshape", d, ValueRange{result, target});
    }
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(middle);
    eraseIfDead(rewriter, first);
    return success();
  }
};

} // namespace

LogicalResult optimizeTFLLocalLayouts(ModuleOp module) {
  RewritePatternSet patterns(module.getContext());
  patterns.add<FoldConvChannelShuffle, ComposeTransposes, CommuteWindowOrTile,
      ComposeSlices, CommuteSingletonReshape, ReduceSingletonTranspose,
      HoistConcatSingletonReshapes, CommutePointwiseTranspose,
      CommutePadTranspose, CommutePointwiseReshape, CommuteSelectTranspose,
      SingletonGatherToReshape>(module.getContext());
  SmallVector<Operation *> roots;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tfl.transpose" ||
        op->getName().getStringRef() == "tfl.slice" ||
        op->getName().getStringRef() == "tfl.reshape" ||
        op->getName().getStringRef() == "tfl.gather" ||
        op->getName().getStringRef() == "tfl.concatenation")
      roots.push_back(op);
  });
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
      .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled)
      .enableFolding(false)
      .enableConstantCSE(false);
  return applyOpPatternsGreedily(
      roots, FrozenRewritePatternSet(std::move(patterns)), config);
}

} // namespace onnx_mlir
