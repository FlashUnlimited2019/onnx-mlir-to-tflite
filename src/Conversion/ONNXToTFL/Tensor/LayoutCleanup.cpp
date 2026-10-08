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

class CancelInverseTransposes final : public RewritePattern {
public:
  explicit CancelInverseTransposes(MLIRContext *ctx)
      : RewritePattern("tfl.transpose", 5, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto q = permutation(op);
    if (failed(q))
      return failure();
    Operation *first = op->getOperand(0).getDefiningOp();
    auto p = permutation(first);
    if (failed(p) || !inverse(*p, *q) ||
        first->getOperand(0).getType() != op->getResult(0).getType())
      return failure();
    rewriter.replaceOp(op, first->getOperand(0));
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
  patterns.add<CancelInverseTransposes, CommuteWindowOrTile, ComposeSlices,
      CommuteSingletonReshape>(module.getContext());
  SmallVector<Operation *> roots;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tfl.transpose" ||
        op->getName().getStringRef() == "tfl.slice")
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
