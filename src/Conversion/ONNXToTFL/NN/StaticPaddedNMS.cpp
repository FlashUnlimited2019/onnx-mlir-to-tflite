/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "llvm/ADT/SmallPtrSet.h"

#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

using namespace mlir;

namespace onnx_mlir {
namespace {

bool integer(Value value, int64_t expected, int64_t rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto values = getConstantIntValues(value);
  return type && type.getRank() == rank && succeeded(values) &&
         values->size() == 1 && values->front() == expected;
}

// Prove the whole consumer, not merely a static annotation on NMS. For
// selected rows s, g = s[:,0] * C + s[:,1], the destination is
// g*K + sum((g[:,None] == g[None,:]) & (j <= i), axis=1) - 1.
// Thus each batch/class has K slots, with unused slots left unchanged.
class PaddingMatcher {
public:
  PaddingMatcher(ONNXNonMaxSuppressionOp nms, int64_t classes, int64_t limit)
      : nms(nms), classes(classes), limit(limit) {
    nodes.insert(nms);
  }

  llvm::SmallPtrSet<Operation *, 32> nodes;

  bool match(Value indices) {
    Value offsets = unsqueeze(indices, 1);
    if (!offsets)
      return false;
    return binary<ONNXAddOp>(
        offsets,
        [&](Value v) {
          return scaled(v, limit, [&](Value x) { return group(x); });
        },
        [&](Value v) { return ordinal(v); });
  }

private:
  template <typename Op>
  Op get(Value value) {
    auto op = value.getDefiningOp<Op>();
    if (op)
      nodes.insert(op);
    return op;
  }

  template <typename Op>
  bool binary(Value value, const std::function<bool(Value)> &a,
      const std::function<bool(Value)> &b) {
    auto op = get<Op>(value);
    return op && ((a(op->getOperand(0)) && b(op->getOperand(1))) ||
                     (a(op->getOperand(1)) && b(op->getOperand(0))));
  }

  Value unsqueeze(Value value, int64_t axis) {
    auto op = get<ONNXUnsqueezeOp>(value);
    if (!op || !integer(op.getAxes(), axis, 1))
      return {};
    return op.getData();
  }

  bool scaled(
      Value value, int64_t scale, const std::function<bool(Value)> &base) {
    if (scale == 1 && base(value))
      return true;
    return binary<ONNXMulOp>(
        value, base, [&](Value v) { return integer(v, scale, 0); });
  }

  bool column(Value value, int64_t index) {
    auto op = get<ONNXGatherOp>(value);
    return op && normalizeAxis(op.getAxis(), 2) == 1 &&
           op.getData() == nms.getResult() &&
           integer(op.getIndices(), index, 0);
  }

  bool group(Value value) {
    return binary<ONNXAddOp>(
        value,
        [&](Value v) {
          return scaled(v, classes, [&](Value x) { return column(x, 0); });
        },
        [&](Value v) { return column(v, 1); });
  }

  bool length(Value value) {
    if (auto squeeze = get<ONNXSqueezeOp>(value)) {
      if (!integer(squeeze.getAxes(), 0, 1))
        return false;
      auto dim = get<ONNXDimOp>(squeeze.getData());
      return dim && dim.getData() == nms.getResult() && dim.getAxis() == 0;
    }
    auto gather = get<ONNXGatherOp>(value);
    if (!gather || gather.getAxis() != 0 || !integer(gather.getIndices(), 0, 0))
      return false;
    auto shape = get<ONNXShapeOp>(gather.getData());
    return shape && shape.getData() == nms.getResult() &&
           shape.getStart() == 0 && (!shape.getEnd() || *shape.getEnd() >= 1);
  }

  bool sequence(Value value) {
    auto range = get<ONNXRangeOp>(value);
    return range && integer(range.getStart(), 0, 0) &&
           integer(range.getDelta(), 1, 0) && length(range.getLimit());
  }

  bool sameGroup(Value value) {
    return binary<ONNXEqualOp>(
        value,
        [&](Value v) {
          Value x = unsqueeze(v, 1);
          return x && group(x);
        },
        [&](Value v) {
          Value x = unsqueeze(v, 0);
          return x && group(x);
        });
  }

  bool preceding(Value value) {
    auto le = get<ONNXLessOrEqualOp>(value);
    if (!le)
      return false;
    Value columns = unsqueeze(le.getA(), 0);
    Value rows = unsqueeze(le.getB(), 1);
    return columns && rows && sequence(columns) && sequence(rows);
  }

  bool count(Value value) {
    Value data;
    if (auto reduce = get<ONNXReduceSumV11Op>(value)) {
      auto axes = reduce.getAxes();
      if (!axes || axes->size() != 1 ||
          cast<IntegerAttr>((*axes)[0]).getInt() != 1 || reduce.getKeepdims())
        return false;
      data = reduce.getData();
    } else if (auto reduce = get<ONNXReduceSumOp>(value)) {
      if (!integer(reduce.getAxes(), 1, 1) || reduce.getKeepdims())
        return false;
      data = reduce.getData();
    } else {
      return false;
    }
    auto castOp = get<ONNXCastOp>(data);
    if (!castOp || !castOp.getTo().isSignlessInteger(64))
      return false;
    return binary<ONNXAndOp>(
        castOp.getInput(), [&](Value v) { return sameGroup(v); },
        [&](Value v) { return preceding(v); });
  }

  bool ordinal(Value value) {
    if (auto sub = get<ONNXSubOp>(value))
      return count(sub.getA()) && integer(sub.getB(), 1, 0);
    return binary<ONNXAddOp>(
        value, [&](Value v) { return count(v); },
        [&](Value v) { return integer(v, -1, 0); });
  }

  ONNXNonMaxSuppressionOp nms;
  int64_t classes, limit;
};

// All generated values have fixed shapes and rank <= 3. Batch/class groups
// are processed together; only the bounded selection dimension is unrolled.
// Do not substitute TFL NMS_V4: its IoU boundary and equal-score ordering
// differ, and degenerate boxes can be selected repeatedly by its kernel.
class StaticNMSBuilder {
public:
  StaticNMSBuilder(PatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  RankedTensorType type(ArrayRef<int64_t> shape, Type element = {}) {
    return RankedTensorType::get(
        shape, element ? element : rewriter.getF32Type());
  }

  NamedAttribute attr(StringRef name, int64_t value) {
    return rewriter.getNamedAttr(name, rewriter.getI64IntegerAttr(value));
  }

  template <typename Op>
  Value op(
      Type result, ValueRange inputs, ArrayRef<NamedAttribute> attrs = {}) {
    return Op::create(rewriter, loc, TypeRange{result}, inputs, attrs)
        ->getResult(0);
  }

  Value ints(ArrayRef<int64_t> shape, ArrayRef<int64_t> values) {
    auto t = type(shape, rewriter.getI64Type());
    return constant(DenseIntElementsAttr::get(t, values));
  }

  Value fp(float value) {
    return constant(DenseFPElementsAttr::get(type({}), ArrayRef<float>{value}));
  }

  Value reshape(Value input, ArrayRef<int64_t> shape) {
    auto t = type(shape, cast<ShapedType>(input.getType()).getElementType());
    return op<ONNXReshapeOp>(
        t, {input, ints({static_cast<int64_t>(shape.size())}, shape)});
  }

  Value gather(
      Value data, Value indices, ArrayRef<int64_t> shape, int64_t axis) {
    return op<ONNXGatherOp>(
        type(shape, cast<ShapedType>(data.getType()).getElementType()),
        {data, indices}, {attr("axis", axis)});
  }

  Value build(ONNXNonMaxSuppressionOp nms, Value padding, int64_t batches,
      int64_t classes, int64_t boxes, int64_t limit, float iou, float score) {
    const int64_t groups = batches * classes;
    // Classes share boxes. Keep geometry batch-sized when caching pairwise
    // overlaps; never materialize a separate copy for each class.
    const bool cacheOverlaps =
        limit >= 3 && boxes <= 64 && batches * boxes <= 65536 / boxes;
    const int64_t geometryGroups = cacheOverlaps ? batches : groups;
    auto geometry = type({geometryGroups, boxes});
    auto matrix = type({groups, boxes});
    auto column = type({groups, 1});
    auto flags = type({groups, boxes}, rewriter.getI1Type());
    auto columnFlags = type({groups, 1}, rewriter.getI1Type());
    auto indicesType = type({groups, 1}, rewriter.getI64Type());
    SmallVector<int64_t> batchIndices, rowIndices, labels;
    for (int64_t g = 0; g < groups; ++g) {
      batchIndices.push_back(g / classes);
      rowIndices.push_back(g);
      for (int64_t k = 0; k < limit; ++k) {
        labels.push_back(g / classes);
        labels.push_back(g % classes);
      }
    }
    Value batchIndex = ints({groups}, batchIndices);
    Value rowIndex = ints({groups, 1}, rowIndices);
    SmallVector<int64_t> boxIds(boxes);
    std::iota(boxIds.begin(), boxIds.end(), 0);
    Value ids = ints({1, boxes}, boxIds);
    Value zero = fp(0), half = fp(0.5f), threshold = fp(score);
    Value minusInfinity = fp(-std::numeric_limits<float>::infinity());
    SmallVector<Value> coordinates;
    for (int64_t i = 0; i < 4; ++i) {
      Value component =
          gather(nms.getBoxes(), ints({}, {i}), {batches, boxes}, 2);
      if (!cacheOverlaps && classes != 1)
        component = gather(component, batchIndex, {groups, boxes}, 0);
      coordinates.push_back(component);
    }
    Value xmin, ymin, xmax, ymax;
    if (nms.getCenterPointBox() == 1) {
      Value w = op<ONNXMulOp>(geometry, {coordinates[2], half});
      Value h = op<ONNXMulOp>(geometry, {coordinates[3], half});
      xmin = op<ONNXSubOp>(geometry, {coordinates[0], w});
      xmax = op<ONNXAddOp>(geometry, {coordinates[0], w});
      ymin = op<ONNXSubOp>(geometry, {coordinates[1], h});
      ymax = op<ONNXAddOp>(geometry, {coordinates[1], h});
    } else {
      xmin = op<ONNXMinOp>(geometry, {coordinates[1], coordinates[3]});
      xmax = op<ONNXMaxOp>(geometry, {coordinates[1], coordinates[3]});
      ymin = op<ONNXMinOp>(geometry, {coordinates[0], coordinates[2]});
      ymax = op<ONNXMaxOp>(geometry, {coordinates[0], coordinates[2]});
    }
    Value area =
        op<ONNXMulOp>(geometry, {op<ONNXSubOp>(geometry, {xmax, xmin}),
                                    op<ONNXSubOp>(geometry, {ymax, ymin})});
    Value scores = reshape(nms.getScores(), {groups, boxes});
    Value eligible = op<ONNXGreaterOp>(flags, {scores, threshold});
    auto overlaps = [&](ArrayRef<Value> candidate, ArrayRef<Value> selected,
                        RankedTensorType shape) {
      auto boolType = type(shape.getShape(), rewriter.getI1Type());
      Value dx = op<ONNXSubOp>(
          shape, {op<ONNXMinOp>(shape, {candidate[2], selected[2]}),
                     op<ONNXMaxOp>(shape, {candidate[0], selected[0]})});
      Value dy = op<ONNXSubOp>(
          shape, {op<ONNXMinOp>(shape, {candidate[3], selected[3]}),
                     op<ONNXMaxOp>(shape, {candidate[1], selected[1]})});
      Value intersection = op<ONNXMulOp>(shape,
          {op<ONNXMaxOp>(shape, {dx, zero}), op<ONNXMaxOp>(shape, {dy, zero})});
      Value unionArea = op<ONNXSubOp>(shape,
          {op<ONNXAddOp>(shape, {candidate[4], selected[4]}), intersection});
      Value overlap = op<ONNXGreaterOp>(
          boolType, {op<ONNXDivOp>(shape, {intersection, unionArea}), fp(iou)});
      // Explicit guards preserve degenerate/negative-size box behavior.
      for (Value positive : {dx, dy, candidate[4], selected[4], unionArea}) {
        auto t = cast<RankedTensorType>(positive.getType());
        Value guard = op<ONNXGreaterOp>(
            type(t.getShape(), rewriter.getI1Type()), {positive, zero});
        overlap = op<ONNXAndOp>(boolType, {overlap, guard});
      }
      return overlap;
    };
    SmallVector<Value> boxData{xmin, ymin, xmax, ymax, area};
    Value pairwise, rowOffsets;
    // Reuse pairwise suppression only for small sets with several selections;
    // bound quadratic storage, and retain linear workspace for large sets.
    if (cacheOverlaps) {
      SmallVector<Value> candidate, selected;
      for (Value value : boxData) {
        candidate.push_back(reshape(value, {batches, 1, boxes}));
        selected.push_back(reshape(value, {batches, boxes, 1}));
      }
      auto pairwiseFlags = type({batches, boxes, boxes}, rewriter.getI1Type());
      Value same = op<ONNXEqualOp>(type({boxes, boxes}, rewriter.getI1Type()),
          {ids, reshape(ids, {boxes, 1})});
      // Self-removal is unconditional, including degenerate boxes and IoU=1.
      // Incorporating the diagonal once avoids rebuilding it at each step.
      pairwise = reshape(
          op<ONNXOrOp>(pairwiseFlags,
              {overlaps(candidate, selected, type({batches, boxes, boxes})),
                  same}),
          {batches * boxes, boxes});
      SmallVector<int64_t> offsets;
      for (int64_t g = 0; g < groups; ++g)
        offsets.push_back((g / classes) * boxes);
      rowOffsets = ints({groups, 1}, offsets);
    }
    SmallVector<Value> selectedColumns, validColumns;
    // Once removed, a score stays at -infinity. Updating scores directly is
    // equivalent to retaining a cumulative eligibility mask, but avoids two
    // boolean operators at every selection step. NaNs are filtered here.
    Value masked = op<ONNXWhereOp>(matrix, {eligible, scores, minusInfinity});
    for (int64_t k = 0; k < limit; ++k) {
      Value best = op<ONNXArgMaxOp>(indicesType, {masked},
          {attr("axis", 1), attr("keepdims", 1), attr("select_last_index", 0)});
      Value maxScore = op<ONNXReduceMaxV13Op>(column, {masked},
          {rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({1})),
              attr("keepdims", 1)});
      Value valid = op<ONNXGreaterOp>(columnFlags, {maxScore, threshold});
      selectedColumns.push_back(best);
      validColumns.push_back(valid);
      if (k + 1 == limit)
        break;
      Value overlap;
      if (pairwise) {
        Value index = op<ONNXAddOp>(indicesType, {best, rowOffsets});
        overlap = reshape(
            gather(pairwise, index, {groups, 1, boxes}, 0), {groups, boxes});
      } else {
        Value coords =
            op<ONNXConcatOp>(type({groups, 2}, rewriter.getI64Type()),
                {rowIndex, best}, {attr("axis", 1)});
        SmallVector<Value> selected;
        for (Value value : boxData)
          selected.push_back(
              reshape(op<ONNXGatherNDOp>(type({groups}), {value, coords},
                          {attr("batch_dims", 0)}),
                  {groups, 1}));
        overlap = overlaps(boxData, selected, matrix);
      }
      Value removed = overlap;
      if (!pairwise) {
        Value same = op<ONNXEqualOp>(flags, {ids, best});
        removed = op<ONNXOrOp>(flags, {same, overlap});
      }
      masked = op<ONNXWhereOp>(matrix, {removed, minusInfinity, masked});
    }
    Value indices =
        op<ONNXConcatOp>(type({groups, limit}, rewriter.getI64Type()),
            selectedColumns, {attr("axis", 1)});
    Value valid = op<ONNXConcatOp>(type({groups, limit}, rewriter.getI1Type()),
        validColumns, {attr("axis", 1)});
    Value triples =
        op<ONNXConcatOp>(type({groups, limit, 3}, rewriter.getI64Type()),
            {ints({groups, limit, 2}, labels),
                reshape(indices, {groups, limit, 1})},
            {attr("axis", 2)});
    return op<ONNXWhereOp>(
        padding.getType(), {reshape(valid, {groups * limit, 1}),
                               reshape(triples, {groups * limit, 3}), padding});
  }

private:
  Value constant(DenseElementsAttr value) {
    return op<ONNXConstantOp>(
        value.getType(), {}, {rewriter.getNamedAttr("value", value)});
  }
  PatternRewriter &rewriter;
  Location loc;
};

bool rewrite(ONNXScatterNDOp scatter, PatternRewriter &rewriter) {
  auto nms = scatter.getUpdates().getDefiningOp<ONNXNonMaxSuppressionOp>();
  if (!nms || scatter.getReduction() != "none")
    return false;
  auto boxesType = dyn_cast<RankedTensorType>(nms.getBoxes().getType());
  auto scoresType = dyn_cast<RankedTensorType>(nms.getScores().getType());
  auto outputType = dyn_cast<RankedTensorType>(scatter.getType());
  auto maximum = getConstantIntValues(nms.getMaxOutputBoxesPerClass());
  auto iou = getConstantF32Values(nms.getIouThreshold());
  auto threshold = getConstantF32Values(nms.getScoreThreshold());
  if (!boxesType || !scoresType || !outputType || !boxesType.hasStaticShape() ||
      !scoresType.hasStaticShape() || !outputType.hasStaticShape() ||
      boxesType.getRank() != 3 || scoresType.getRank() != 3 ||
      outputType.getRank() != 2 || !boxesType.getElementType().isF32() ||
      !scoresType.getElementType().isF32() ||
      !outputType.getElementType().isSignlessInteger(64) ||
      scatter.getData().getType() != outputType || failed(maximum) ||
      maximum->size() != 1 || failed(iou) || iou->size() != 1 ||
      failed(threshold) || threshold->size() != 1 ||
      !std::isfinite(iou->front()) || iou->front() < 0 || iou->front() > 1 ||
      !std::isfinite(threshold->front()) ||
      (nms.getCenterPointBox() != 0 && nms.getCenterPointBox() != 1))
    return false;
  int64_t batches = boxesType.getDimSize(0), boxes = boxesType.getDimSize(1);
  int64_t classes = scoresType.getDimSize(1), limit = maximum->front();
  // Bound graph expansion and all coordinate products before allocating.
  constexpr int64_t maxElements = 1024 * 1024;
  if (batches <= 0 || boxes <= 0 || classes <= 0 || limit <= 0 || limit > 256 ||
      batches > maxElements / classes ||
      batches * classes > maxElements / boxes ||
      batches * classes > maxElements / limit || boxesType.getDimSize(2) != 4 ||
      scoresType.getDimSize(0) != batches ||
      scoresType.getDimSize(2) != boxes ||
      outputType.getDimSize(0) != batches * classes * limit ||
      outputType.getDimSize(1) != 3)
    return false;
  PaddingMatcher matcher(nms, classes, limit);
  if (!matcher.match(scatter.getIndices()))
    return false;
  // Do not change the meaning of an independently observable variable-length
  // result, count, or index calculation, including a second padded consumer.
  for (Operation *node : matcher.nodes)
    for (Operation *user : node->getUsers())
      if (user != scatter && !matcher.nodes.contains(user))
        return false;
  rewriter.setInsertionPoint(scatter);
  StaticNMSBuilder builder(rewriter, scatter.getLoc());
  Value result = builder.build(nms, scatter.getData(), batches, classes, boxes,
      limit, iou->front(), threshold->front());
  rewriter.replaceOp(scatter, result);
  // All matched nodes are pure and now have no escaping uses. Erase leaves
  // first without relying on a later folder to interpret stale annotations.
  while (!matcher.nodes.empty()) {
    SmallVector<Operation *> dead;
    for (Operation *node : matcher.nodes)
      if (node->use_empty())
        dead.push_back(node);
    assert(!dead.empty() && "matched NMS consumer must be acyclic and closed");
    for (Operation *node : dead) {
      matcher.nodes.erase(node);
      rewriter.eraseOp(node);
    }
  }
  return true;
}

} // namespace

void rewriteONNXToTFLStaticPaddedNMS(ModuleOp module) {
  SmallVector<ONNXScatterNDOp> candidates;
  module.walk([&](ONNXScatterNDOp op) { candidates.push_back(op); });
  PatternRewriter rewriter(module.getContext());
  for (auto op : candidates)
    rewrite(op, rewriter);
}

} // namespace onnx_mlir
