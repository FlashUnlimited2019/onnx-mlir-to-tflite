/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"

#include <limits>
#include <memory>
#include <optional>

using namespace mlir;

namespace onnx_mlir {
namespace {

// These are optimization limits, not model restrictions. Exceeding a limit
// leaves the original operation intact. No floating-point arithmetic is folded.
constexpr int64_t maxElements = 1024 * 1024;
constexpr int64_t maxRank = 32;
struct FoldBudget {
  int64_t elements = 8 * maxElements;
  int64_t work = 64 * maxElements;
};

bool isIndexInteger(Type type) {
  return type.isSignlessInteger(1) || type.isSignlessInteger(32) ||
         type.isSignlessInteger(64);
}

struct TensorInfo {
  RankedTensorType type;
  int64_t elements;
  ArrayRef<int64_t> shape() const { return type.getShape(); }
};

std::optional<TensorInfo> getInfo(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      type.getRank() > maxRank)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    elements *= dim;
  }
  return TensorInfo{type, elements};
}

DenseIntElementsAttr getIntegerConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto attr = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                       : DenseIntElementsAttr{};
  if (!attr || attr.getType() != value.getType() ||
      !isIndexInteger(attr.getElementType()))
    return {};
  return attr;
}

FailureOr<SmallVector<int64_t>> getIndexVector(Value value) {
  auto attr = getIntegerConstant(value);
  if (!attr || attr.getType().getRank() != 1 ||
      attr.getNumElements() > maxRank || attr.getElementType().isInteger(1))
    return failure();
  SmallVector<int64_t> values;
  for (APInt v : attr.getValues<APInt>())
    values.push_back(v.getSExtValue());
  return values;
}

bool hasNoActivation(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("fused_activation_function");
  return attr && attr.getValue() == "NONE";
}

bool matchesBroadcast(ValueRange inputs, RankedTensorType result) {
  SmallVector<int64_t> shape(result.getRank(), 1);
  for (Value value : inputs) {
    auto info = getInfo(value);
    if (!info || info->type.getRank() > result.getRank())
      return false;
    int64_t offset = result.getRank() - info->type.getRank();
    for (int64_t i = 0; i < info->type.getRank(); ++i) {
      int64_t dim = info->shape()[i];
      int64_t &expected = shape[offset + i];
      if (expected != 1 && dim != 1 && expected != dim)
        return false;
      if (dim != 1)
        expected = dim;
    }
  }
  return llvm::equal(shape, result.getShape());
}

// Validate the entire concatenation, including operands not selected by a
// subsequent slice. The rewrite must not depend on a coincidental shape match.
FailureOr<int64_t> getConcatAxis(Operation *op) {
  if (!op || op->getName().getStringRef() != "tfl.concatenation" ||
      op->getNumResults() != 1 || op->getNumOperands() == 0 ||
      op->getNumOperands() > 1024 || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0 || !hasNoActivation(op))
    return failure();
  auto result = getInfo(op->getResult(0));
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!result || !axisAttr || axisAttr.getValue().getBitWidth() > 64 ||
      result->type.getRank() == 0)
    return failure();
  int64_t rank = result->type.getRank(), axis = axisAttr.getInt();
  if (axis < -rank || axis >= rank)
    return failure();
  if (axis < 0)
    axis += rank;
  int64_t total = 0;
  for (Value input : op->getOperands()) {
    auto info = getInfo(input);
    if (!info || info->type.getRank() != rank ||
        info->type.getElementType() != result->type.getElementType())
      return failure();
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis && info->shape()[i] != result->shape()[i])
        return failure();
    if (info->shape()[axis] > result->shape()[axis] - total)
      return failure();
    total += info->shape()[axis];
  }
  return total == result->shape()[axis] ? FailureOr<int64_t>(axis)
                                        : FailureOr<int64_t>(failure());
}

// Prove a nonnegative signed-integer range, not merely a nonnegative source.
// In particular, addition and narrowing casts must not wrap into negatives.
// Unknown operations, excessive work, and incomplete shape proofs bail out.
std::optional<int64_t> nonnegativeUpperBound(
    Value value, unsigned depth, int64_t &work) {
  auto info = getInfo(value);
  if (!info || depth == 0 || work-- <= 0 ||
      (!info->type.getElementType().isSignlessInteger(32) &&
          !info->type.getElementType().isSignlessInteger(64)))
    return std::nullopt;
  int64_t maximum = info->type.getElementType().isInteger(32)
                        ? std::numeric_limits<int32_t>::max()
                        : std::numeric_limits<int64_t>::max();
  if (auto constant = getIntegerConstant(value)) {
    int64_t count = constant.isSplat() ? 1 : info->elements;
    if (count > work)
      return std::nullopt;
    work -= count;
    int64_t bound = 0;
    for (APInt element : constant.getValues<APInt>()) {
      if (element.isNegative())
        return std::nullopt;
      bound = std::max(bound, element.getSExtValue());
      if (constant.isSplat())
        break;
    }
    return bound;
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->getNumResults() != 1 || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0)
    return std::nullopt;
  StringRef name = op->getName().getStringRef();
  auto read = [&](Value input) {
    return nonnegativeUpperBound(input, depth - 1, work);
  };
  if ((name == "tfl.arg_max" || name == "tfl.arg_min") &&
      op->getNumOperands() == 2) {
    auto input = getInfo(op->getOperand(0));
    auto axis = getIntegerConstant(op->getOperand(1));
    if (!input || !axis || axis.getNumElements() != 1 ||
        axis.getElementType().isInteger(1) ||
        (!input->type.getElementType().isF32() &&
            !input->type.getElementType().isSignlessInteger(32)))
      return std::nullopt;
    int64_t dim = (*axis.getValues<APInt>().begin()).getSExtValue();
    int64_t rank = input->type.getRank();
    if (dim < -rank || dim >= rank)
      return std::nullopt;
    if (dim < 0)
      dim += rank;
    SmallVector<int64_t> expected(input->shape());
    int64_t bound = expected[dim] - 1;
    expected.erase(expected.begin() + dim);
    if (bound > maximum || !llvm::equal(expected, info->shape()))
      return std::nullopt;
    return bound;
  }
  if (name == "tfl.reshape" && op->getNumOperands() == 2) {
    auto input = getInfo(op->getOperand(0));
    auto shape = getIndexVector(op->getOperand(1));
    if (!input || input->elements != info->elements ||
        input->type.getElementType() != info->type.getElementType() ||
        failed(shape) || !llvm::equal(*shape, info->shape()))
      return std::nullopt;
    return read(op->getOperand(0));
  }
  if (name == "tfl.cast" && op->getNumOperands() == 1) {
    auto input = getInfo(op->getOperand(0));
    if (!input || input->shape() != info->shape())
      return std::nullopt;
    auto bound = read(op->getOperand(0));
    return bound && *bound <= maximum ? bound : std::nullopt;
  }
  if (name == "tfl.add" && op->getNumOperands() == 2 && hasNoActivation(op) &&
      matchesBroadcast(op->getOperands(), info->type)) {
    for (Value input : op->getOperands())
      if (cast<RankedTensorType>(input.getType()).getElementType() !=
          info->type.getElementType())
        return std::nullopt;
    auto left = read(op->getOperand(0)), right = read(op->getOperand(1));
    if (!left || !right || *left > maximum - *right)
      return std::nullopt;
    return *left + *right;
  }
  if (name == "tfl.concatenation" && succeeded(getConcatAxis(op))) {
    int64_t bound = 0;
    for (Value input : op->getOperands()) {
      auto part = read(input);
      if (!part)
        return std::nullopt;
      bound = std::max(bound, *part);
    }
    return bound;
  }
  return std::nullopt;
}

class FoldNonnegativeIndexCheck final : public RewritePattern {
public:
  explicit FoldNonnegativeIndexCheck(MLIRContext *context)
      : RewritePattern("tfl.less", 4, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1 ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return failure();
    auto result = getInfo(op->getResult(0));
    auto input = getInfo(op->getOperand(0));
    auto zero = getIntegerConstant(op->getOperand(1));
    if (!result || !input || !zero || !zero.isSplat() ||
        !zero.getSplatValue<APInt>().isZero() ||
        zero.getElementType() != input->type.getElementType() ||
        !result->type.getElementType().isInteger(1) ||
        !matchesBroadcast(op->getOperands(), result->type))
      return failure();
    int64_t work = 4096;
    if (!nonnegativeUpperBound(op->getOperand(0), 16, work))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, result->type,
        DenseIntElementsAttr::get(result->type, ArrayRef<bool>{false}));
    return success();
  }
};

struct SliceWindow {
  SmallVector<int64_t> begin, size, step;
};

FailureOr<SliceWindow> getSliceWindow(Operation *op) {
  StringRef name = op->getName().getStringRef();
  bool strided = name == "tfl.strided_slice";
  if ((!strided && name != "tfl.slice") || op->getNumResults() != 1 ||
      op->getNumOperands() != (strided ? 4u : 3u) || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0)
    return failure();
  auto input = getInfo(op->getOperand(0));
  auto result = getInfo(op->getResult(0));
  auto begin = getIndexVector(op->getOperand(1));
  auto endOrSize = getIndexVector(op->getOperand(2));
  if (!input || !result || failed(begin) || failed(endOrSize) ||
      input->type.getRank() != result->type.getRank() ||
      input->type.getElementType() != result->type.getElementType() ||
      begin->size() != input->shape().size() ||
      endOrSize->size() != begin->size())
    return failure();
  SliceWindow window{*begin, {}, SmallVector<int64_t>(begin->size(), 1)};
  if (strided) {
    for (StringRef key : {"begin_mask", "end_mask", "ellipsis_mask",
             "new_axis_mask", "shrink_axis_mask"}) {
      auto attr = op->getAttrOfType<IntegerAttr>(key);
      if (!attr || !attr.getValue().isZero())
        return failure();
    }
    auto offset = op->getAttrOfType<BoolAttr>("offset");
    auto strides = getIndexVector(op->getOperand(3));
    if ((op->hasAttr("offset") && !offset) || (offset && offset.getValue()) ||
        failed(strides) || strides->size() != begin->size())
      return failure();
    window.step = *strides;
  }
  for (int64_t i = 0; i < input->type.getRank(); ++i) {
    int64_t start = (*begin)[i], dim = input->shape()[i];
    int64_t value = (*endOrSize)[i], step = window.step[i];
    if (start < 0 || start >= dim || step <= 0)
      return failure();
    int64_t size;
    if (strided) {
      if (value <= start || value > dim)
        return failure();
      size = 1 + (value - start - 1) / step;
    } else {
      size = value == -1 ? dim - start : value;
      if (size <= 0 || size > dim - start)
        return failure();
    }
    if (size != result->shape()[i])
      return failure();
    window.size.push_back(size);
  }
  return window;
}

SmallVector<int64_t> coordinates(int64_t index, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> coord(shape.size());
  for (int64_t i = shape.size(); i-- > 0;) {
    coord[i] = index % shape[i];
    index /= shape[i];
  }
  return coord;
}

int64_t linearIndex(ArrayRef<int64_t> coord, ArrayRef<int64_t> shape) {
  int64_t index = 0;
  for (auto [c, d] : llvm::zip(coord, shape))
    index = index * d + c;
  return index;
}

enum class Kind {
  Unknown,
  Constant,
  Reshape,
  Tile,
  Slice,
  Transpose,
  Reverse,
  Broadcast,
  Concat,
  Add,
  Less,
  Cast,
  Select
};

struct Plan {
  Kind kind = Kind::Unknown;
  std::optional<TensorInfo> info;
  DenseIntElementsAttr constant;
  SmallVector<Value> inputs;
  SmallVector<ArrayRef<int64_t>> shapes;
  SmallVector<int64_t> parameters, steps;
  int64_t axis = 0;
};

// Read individual integer elements through verified static operations. Unknown
// inputs remain unknown, but a slice of a constant Concat operand need not read
// another, runtime-dependent operand. Plans cache shape/attribute validation;
// neither unknown masks nor large source tensors are materialized.
class IntegerEvaluator {
public:
  explicit IntegerEvaluator(FoldBudget &budget) : budget(budget) {}

  FailureOr<APInt> read(Value value, int64_t index, unsigned depth = 0) {
    if (depth > 32 || !consume(1))
      return failure();
    Plan &p = getPlan(value);
    if (!p.info || index < 0 || index >= p.info->elements ||
        p.kind == Kind::Unknown)
      return failure();
    if (p.kind == Kind::Constant)
      return *(p.constant.getValues<APInt>().begin() + index);
    if (p.kind == Kind::Reshape)
      return read(p.inputs[0], index, depth + 1);
    SmallVector<int64_t> coord = coordinates(index, p.info->shape());
    auto broadcastRead = [&](unsigned operand) {
      ArrayRef<int64_t> shape = p.shapes[operand];
      SmallVector<int64_t> inputCoord(shape.size());
      int64_t offset = coord.size() - shape.size();
      for (int64_t i = 0, rank = shape.size(); i < rank; ++i)
        inputCoord[i] = shape[i] == 1 ? 0 : coord[offset + i];
      return read(p.inputs[operand], linearIndex(inputCoord, shape), depth + 1);
    };
    if (p.kind == Kind::Broadcast)
      return broadcastRead(0);
    if (p.kind == Kind::Select) {
      auto condition = broadcastRead(0);
      if (failed(condition))
        return failure();
      return broadcastRead(condition->isZero() ? 2 : 1);
    }
    if (p.kind == Kind::Add || p.kind == Kind::Less) {
      auto lhs = broadcastRead(0), rhs = broadcastRead(1);
      if (failed(lhs) || failed(rhs))
        return failure();
      if (p.kind == Kind::Less)
        return APInt(1, lhs->slt(*rhs));
      bool overflow = false;
      APInt sum = lhs->sadd_ov(*rhs, overflow);
      return overflow ? FailureOr<APInt>(failure()) : FailureOr<APInt>(sum);
    }
    if (p.kind == Kind::Cast) {
      auto input = read(p.inputs[0], index, depth + 1);
      if (failed(input))
        return failure();
      unsigned width = p.info->type.getElementType().getIntOrFloatBitWidth();
      if (width == 1)
        return APInt(1, !input->isZero());
      if (input->getBitWidth() == 1)
        return input->zext(width);
      // Do not bake implementation-dependent out-of-range narrowing into a
      // constant. Preserve the existing runtime cast for those values.
      if (!input->isSignedIntN(width))
        return failure();
      return input->sextOrTrunc(width);
    }
    if (p.kind == Kind::Concat) {
      for (unsigned i = 0; i < p.inputs.size(); ++i) {
        int64_t extent = p.shapes[i][p.axis];
        if (coord[p.axis] < extent)
          return read(p.inputs[i], linearIndex(coord, p.shapes[i]), depth + 1);
        coord[p.axis] -= extent;
      }
      return failure();
    }
    SmallVector<int64_t> inputCoord = coord;
    if (p.kind == Kind::Tile) {
      for (unsigned i = 0; i < coord.size(); ++i)
        inputCoord[i] %= p.shapes[0][i];
    } else if (p.kind == Kind::Slice) {
      for (unsigned i = 0; i < coord.size(); ++i)
        inputCoord[i] = p.parameters[i] + coord[i] * p.steps[i];
    } else if (p.kind == Kind::Transpose) {
      for (unsigned i = 0; i < coord.size(); ++i)
        inputCoord[p.parameters[i]] = coord[i];
    } else if (p.kind == Kind::Reverse) {
      for (int64_t axis : p.parameters)
        inputCoord[axis] = p.shapes[0][axis] - 1 - coord[axis];
    } else {
      return failure();
    }
    return read(p.inputs[0], linearIndex(inputCoord, p.shapes[0]), depth + 1);
  }

private:
  FoldBudget &budget;
  int64_t remaining = 4 * maxElements;
  llvm::DenseMap<Value, std::unique_ptr<Plan>> plans;

  bool consume(int64_t amount) {
    if (amount > remaining || amount > budget.work)
      return false;
    remaining -= amount;
    budget.work -= amount;
    return true;
  }

  Plan &getPlan(Value value) {
    auto &slot = plans[value];
    if (slot)
      return *slot;
    slot = std::make_unique<Plan>();
    Plan &p = *slot;
    p.info = getInfo(value);
    if (!p.info || !isIndexInteger(p.info->type.getElementType()))
      return p;
    p.constant = getIntegerConstant(value);
    if (p.constant) {
      p.kind = Kind::Constant;
      return p;
    }
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumOperands() > 1024 ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
        !consume(op->getNumOperands() + p.info->type.getRank() + 1))
      return p;
    StringRef name = op->getName().getStringRef();
    p.inputs.assign(op->operand_begin(), op->operand_end());
    for (Value input : p.inputs) {
      auto info = getInfo(input);
      if (!info)
        return p;
      p.shapes.push_back(info->shape());
    }
    Type element = p.info->type.getElementType();
    auto sameElement = [&](unsigned i) {
      return cast<RankedTensorType>(p.inputs[i].getType()).getElementType() ==
             element;
    };
    if (name == "tfl.concatenation") {
      auto axis = getConcatAxis(op);
      if (succeeded(axis)) {
        p.axis = *axis;
        p.kind = Kind::Concat;
      }
    } else if ((name == "tfl.add" || name == "tfl.less") &&
               p.inputs.size() == 2) {
      Type lhs = cast<RankedTensorType>(p.inputs[0].getType()).getElementType();
      Type rhs = cast<RankedTensorType>(p.inputs[1].getType()).getElementType();
      if (lhs != rhs ||
          (!lhs.isSignlessInteger(32) && !lhs.isSignlessInteger(64)) ||
          !matchesBroadcast(p.inputs, p.info->type))
        return p;
      if (name == "tfl.less" && element.isInteger(1))
        p.kind = Kind::Less;
      if (name == "tfl.add" && lhs == element && hasNoActivation(op))
        p.kind = Kind::Add;
    } else if (name == "tfl.cast" && p.inputs.size() == 1) {
      Type inputElement =
          cast<RankedTensorType>(p.inputs[0].getType()).getElementType();
      if (isIndexInteger(inputElement) && p.shapes[0] == p.info->shape())
        p.kind = Kind::Cast;
    } else if (name == "tfl.select_v2" && p.inputs.size() == 3) {
      if (cast<RankedTensorType>(p.inputs[0].getType())
              .getElementType()
              .isInteger(1) &&
          sameElement(1) && sameElement(2) &&
          matchesBroadcast(p.inputs, p.info->type))
        p.kind = Kind::Select;
    } else if (name == "tfl.reshape" && p.inputs.size() == 2 &&
               sameElement(0)) {
      auto shape = getIndexVector(p.inputs[1]);
      if (failed(shape) || shape->size() != p.info->shape().size() ||
          getInfo(p.inputs[0])->elements != p.info->elements)
        return p;
      bool inferred = false;
      for (auto [dim, actual] : llvm::zip(*shape, p.info->shape())) {
        if (dim == -1 && !inferred)
          inferred = true;
        else if (dim != actual)
          return p;
      }
      p.kind = Kind::Reshape;
    } else if (name == "tfl.broadcast_to" && p.inputs.size() == 2 &&
               sameElement(0)) {
      auto shape = getIndexVector(p.inputs[1]);
      if (succeeded(shape) && llvm::equal(*shape, p.info->shape()) &&
          p.shapes[0].size() <= shape->size()) {
        int64_t offset = shape->size() - p.shapes[0].size();
        for (unsigned i = 0; i < p.shapes[0].size(); ++i)
          if (p.shapes[0][i] != 1 && p.shapes[0][i] != (*shape)[offset + i])
            return p;
        p.kind = Kind::Broadcast;
      }
    } else if (name == "tfl.tile" && p.inputs.size() == 2 && sameElement(0) &&
               p.shapes[0].size() == p.info->shape().size()) {
      auto repeats = getIndexVector(p.inputs[1]);
      if (failed(repeats) || repeats->size() != p.shapes[0].size())
        return p;
      for (unsigned i = 0; i < repeats->size(); ++i)
        if ((*repeats)[i] <= 0 || p.info->shape()[i] % p.shapes[0][i] != 0 ||
            p.info->shape()[i] / p.shapes[0][i] != (*repeats)[i])
          return p;
      p.kind = Kind::Tile;
    } else if (name == "tfl.slice" || name == "tfl.strided_slice") {
      auto window = getSliceWindow(op);
      if (succeeded(window)) {
        p.parameters = window->begin;
        p.steps = window->step;
        p.kind = Kind::Slice;
      }
    } else if ((name == "tfl.transpose" || name == "tfl.reverse_v2") &&
               p.inputs.size() == 2 && sameElement(0) &&
               p.shapes[0].size() == p.info->shape().size()) {
      auto axes = getIndexVector(p.inputs[1]);
      int64_t rank = p.shapes[0].size();
      if (failed(axes) || (name == "tfl.transpose" &&
                              static_cast<int64_t>(axes->size()) != rank))
        return p;
      SmallVector<bool> seen(rank, false);
      for (int64_t &axis : *axes) {
        if (name == "tfl.reverse_v2" && axis >= -rank && axis < 0)
          axis += rank;
        if (axis < 0 || axis >= rank || seen[axis])
          return p;
        seen[axis] = true;
      }
      if (name == "tfl.transpose") {
        for (int64_t i = 0; i < rank; ++i)
          if (p.info->shape()[i] != p.shapes[0][(*axes)[i]])
            return p;
        p.kind = Kind::Transpose;
      } else {
        if (p.info->shape() != p.shapes[0])
          return p;
        p.kind = Kind::Reverse;
      }
      p.parameters = *axes;
    }
    return p;
  }
};

bool isIndexRoot(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases({"tfl.tile", "tfl.reshape", "tfl.add", "tfl.cast", "tfl.less",
                 "tfl.select_v2", "tfl.concatenation", "tfl.slice",
                 "tfl.strided_slice", "tfl.broadcast_to", "tfl.transpose",
                 "tfl.reverse_v2", "tfl.gather_nd"},
          true)
      .Default(false);
}

// Constant, possibly irregular coordinates can be linearized at compile time.
// Keep dynamic coordinates on the original path: a flat offset alone would
// not preserve per-axis bounds checking. The runtime gather reads a rank-2
// matrix using rank-1 indices and preserves duplicates and their ordering.
class FlattenConstantGatherND final : public RewritePattern {
public:
  explicit FlattenConstantGatherND(MLIRContext *context)
      : RewritePattern("tfl.gather_nd", 3, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1 ||
        op->getNumRegions() || op->getNumSuccessors())
      return failure();
    auto data = getInfo(op->getOperand(0));
    auto indices = getInfo(op->getOperand(1));
    auto result = getInfo(op->getResult(0));
    auto constant = getIntegerConstant(op->getOperand(1));
    if (!data || !indices || !result || !constant ||
        !data->type.getElementType().isF32() ||
        result->type.getElementType() != data->type.getElementType() ||
        indices->type.getElementType().isInteger(1) ||
        data->type.getRank() > 5 || result->type.getRank() > 5 ||
        indices->type.getRank() < 1 || indices->elements > maxElements)
      return failure();
    int64_t depth = indices->shape().back();
    if (depth < 2 || depth > data->type.getRank())
      return failure();
    SmallVector<int64_t> expected(indices->shape().drop_back());
    llvm::append_range(expected, data->shape().drop_front(depth));
    if (!llvm::equal(expected, result->shape()))
      return failure();
    int64_t rows = 1, tuples = indices->elements / depth;
    for (int64_t dim : data->shape().take_front(depth))
      rows *= dim; // getInfo already checked the full element count.
    int64_t columns = data->elements / rows;
    if (rows > std::numeric_limits<int32_t>::max() ||
        columns > std::numeric_limits<int32_t>::max() ||
        tuples > std::numeric_limits<int32_t>::max())
      return failure();
    SmallVector<int32_t> offsets;
    offsets.reserve(tuples);
    auto values = constant.getValues<APInt>();
    auto it = values.begin();
    for (int64_t t = 0; t < tuples; ++t) {
      int64_t offset = 0;
      for (int64_t d = 0; d < depth; ++d, ++it) {
        int64_t index = (*it).getSExtValue();
        // Unlike ONNX, TFL coordinates must already be nonnegative.
        if (index < 0 || index >= data->shape()[d])
          return failure();
        offset = offset * data->shape()[d] + index;
      }
      offsets.push_back(static_cast<int32_t>(offset));
    }
    auto i32 = rewriter.getI32Type();
    auto create = [&](StringRef name, Type type, ValueRange operands,
                      ArrayRef<NamedAttribute> attrs = {}) -> Value {
      OperationState state(op->getLoc(), name);
      state.addTypes(type);
      state.addOperands(operands);
      state.addAttributes(attrs);
      return rewriter.create(state)->getResult(0);
    };
    auto reshape = [&](Value value, RankedTensorType type) -> Value {
      if (value.getType() == type)
        return value;
      SmallVector<int32_t> shape;
      for (int64_t dim : type.getShape())
        shape.push_back(static_cast<int32_t>(dim));
      auto shapeType = RankedTensorType::get({type.getRank()}, i32);
      Value shapeValue =
          arith::ConstantOp::create(rewriter, op->getLoc(), shapeType,
              DenseIntElementsAttr::get(shapeType, ArrayRef<int32_t>(shape)));
      return create("tfl.reshape", type, {value, shapeValue});
    };
    for (int64_t dim : result->shape())
      if (dim > std::numeric_limits<int32_t>::max())
        return failure();
    auto matrixType =
        RankedTensorType::get({rows, columns}, rewriter.getF32Type());
    auto gatheredType =
        RankedTensorType::get({tuples, columns}, rewriter.getF32Type());
    auto offsetType = RankedTensorType::get({tuples}, i32);
    Value offsetsValue =
        arith::ConstantOp::create(rewriter, op->getLoc(), offsetType,
            DenseIntElementsAttr::get(offsetType, ArrayRef<int32_t>(offsets)));
    Value matrix = reshape(op->getOperand(0), matrixType);
    SmallVector<NamedAttribute> attrs{
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(0)),
        rewriter.getNamedAttr("batch_dims", rewriter.getI32IntegerAttr(0))};
    Value gathered =
        create("tfl.gather", gatheredType, {matrix, offsetsValue}, attrs);
    rewriter.replaceOp(op, reshape(gathered, result->type));
    return success();
  }
};

class FoldIntegerIndex final : public RewritePattern {
public:
  FoldIntegerIndex(MLIRContext *context, std::shared_ptr<FoldBudget> budget)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), budget(budget) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isIndexRoot(op) || op->getNumResults() != 1 || op->use_empty())
      return failure();
    Value result = op->getResult(0);
    auto info = getInfo(result);
    if (!info || !isIndexInteger(info->type.getElementType()) ||
        info->elements > maxElements || info->elements > budget->elements)
      return failure();
    IntegerEvaluator evaluator(*budget);
    SmallVector<APInt> values;
    for (int64_t i = 0; i < info->elements; ++i) {
      auto value = evaluator.read(result, i);
      if (failed(value))
        return failure();
      values.push_back(*value);
    }
    budget->elements -= info->elements;
    auto attr = DenseElementsAttr::get(info->type, values);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, info->type, attr);
    return success();
  }

private:
  std::shared_ptr<FoldBudget> budget;
};

class FoldUniformSelect final : public RewritePattern {
public:
  FoldUniformSelect(MLIRContext *context, std::shared_ptr<FoldBudget> budget)
      : RewritePattern("tfl.select_v2", 3, context), budget(budget) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1 ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return failure();
    auto result = getInfo(op->getResult(0));
    auto condition = getInfo(op->getOperand(0));
    if (!result || !condition ||
        !condition->type.getElementType().isInteger(1) ||
        condition->elements > maxElements ||
        (!result->type.getElementType().isF32() &&
            !isIndexInteger(result->type.getElementType())) ||
        !matchesBroadcast(op->getOperands(), result->type))
      return failure();
    for (Value branch : op->getOperands().drop_front())
      if (cast<RankedTensorType>(branch.getType()).getElementType() !=
          result->type.getElementType())
        return failure();
    IntegerEvaluator evaluator(*budget);
    auto first = evaluator.read(op->getOperand(0), 0);
    if (failed(first))
      return failure();
    Value selected = op->getOperand(first->isZero() ? 2 : 1);
    // Do not drop a required broadcast or insert a new runtime expansion.
    if (selected.getType() != result->type)
      return failure();
    for (int64_t i = 1; i < condition->elements; ++i) {
      auto value = evaluator.read(op->getOperand(0), i);
      if (failed(value) || *value != *first)
        return failure();
    }
    rewriter.replaceOp(op, selected);
    return success();
  }

private:
  std::shared_ptr<FoldBudget> budget;
};

class SimplifyConcatSlice final : public RewritePattern {
public:
  explicit SimplifyConcatSlice(MLIRContext *context)
      : RewritePattern("tfl.slice", 4, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    auto window = getSliceWindow(op);
    if (failed(window))
      return failure();
    Operation *concat = op->getOperand(0).getDefiningOp();
    auto axis = getConcatAxis(concat);
    if (failed(axis))
      return failure();
    int64_t offset = 0;
    for (Value part : concat->getOperands()) {
      auto type = cast<RankedTensorType>(part.getType());
      int64_t extent = type.getShape()[*axis];
      int64_t start = window->begin[*axis] - offset;
      if (start >= 0 && start < extent &&
          window->size[*axis] <= extent - start) {
        window->begin[*axis] = start;
        if (llvm::all_of(window->begin, [](int64_t v) { return v == 0; }) &&
            op->getResult(0).getType() == type) {
          rewriter.replaceOp(op, part);
          return success();
        }
        auto beginType = cast<RankedTensorType>(op->getOperand(1).getType());
        unsigned width = beginType.getElementType().getIntOrFloatBitWidth();
        SmallVector<APInt> beginValues;
        for (int64_t v : window->begin) {
          if (!APInt(64, v).isSignedIntN(width))
            return failure();
          beginValues.emplace_back(width, v);
        }
        // Only the begin vector changes: a retained -1 size still denotes
        // the same tail because containment proves this is the final operand.
        Value begin = arith::ConstantOp::create(rewriter, op->getLoc(),
            beginType, DenseElementsAttr::get(beginType, beginValues));
        rewriter.modifyOpInPlace(op, [&]() {
          op->setOperand(0, part);
          op->setOperand(1, begin);
        });
        return success();
      }
      offset += extent;
    }
    return failure();
  }
};

// The producer's compatibility dialect has no generated side-effect traits.
// Explicitly clean up only this whitelist of stateless operations; in
// particular, never remove an operation whose result is a function output.
class EraseDeadIndexOperation final : public RewritePattern {
public:
  explicit EraseDeadIndexOperation(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 5, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, PatternRewriter &rewriter) const override {
    if (!isIndexRoot(op) || !op->use_empty() || op->getNumResults() != 1 ||
        op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

LogicalResult optimizeTFLStaticIndexOperations(ModuleOp module) {
  auto budget = std::make_shared<FoldBudget>();
  RewritePatternSet patterns(module.getContext());
  patterns.add<FoldIntegerIndex, FoldUniformSelect>(
      module.getContext(), budget);
  patterns.add<FoldNonnegativeIndexCheck, SimplifyConcatSlice,
      FlattenConstantGatherND, EraseDeadIndexOperation>(module.getContext());
  SmallVector<Operation *> roots;
  module.walk([&](Operation *op) {
    if (isIndexRoot(op))
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
