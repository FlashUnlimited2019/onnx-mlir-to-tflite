/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#include "src/Conversion/ONNXToTFL/LayoutAnalysis.hpp"

#include "src/Conversion/ONNXToTFL/ONNXToTFLLegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

using namespace mlir;

namespace onnx_mlir {
namespace {

bool isStaticRankedTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() > 0;
}

bool canShareLayout(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || lhsType.getRank() != rhsType.getRank())
    return false;
  for (auto [lhsExtent, rhsExtent] :
      llvm::zip(lhsType.getShape(), rhsType.getShape())) {
    if (lhsExtent != rhsExtent && lhsExtent != 1 && rhsExtent != 1)
      return false;
  }
  return true;
}

SmallVector<int64_t> getPermutation(Operation *op, int64_t rank) {
  SmallVector<int64_t> permutation;
  if (auto array = op->getAttrOfType<ArrayAttr>("perm")) {
    for (Attribute element : array)
      permutation.push_back(
          cast<IntegerAttr>(element).getValue().getSExtValue());
  } else if (auto dense = op->getAttrOfType<DenseIntElementsAttr>("perm")) {
    for (APInt element : dense.getValues<APInt>())
      permutation.push_back(element.getSExtValue());
  }
  if (permutation.empty()) {
    permutation.resize(rank);
    std::iota(permutation.rbegin(), permutation.rend(), int64_t{0});
  }
  if (static_cast<int64_t>(permutation.size()) != rank)
    return {};
  SmallVector<bool> seen(rank, false);
  for (int64_t dimension : permutation) {
    if (dimension < 0 || dimension >= rank || seen[dimension])
      return {};
    seen[dimension] = true;
  }
  return permutation;
}

bool isLayoutPreservingElementwise(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases({"onnx.Identity", "onnx.Add", "onnx.Sub", "onnx.Mul"}, true)
      .Cases({"onnx.Div", "onnx.Pow", "onnx.Min", "onnx.Max"}, true)
      .Cases({"onnx.Mean", "onnx.Where", "onnx.And", "onnx.Or"}, true)
      .Cases({"onnx.Xor", "onnx.Equal", "onnx.Greater"}, true)
      .Cases(
          {"onnx.GreaterOrEqual", "onnx.Less", "onnx.LessOrEqual", "onnx.Not"},
          true)
      .Cases({"onnx.Relu", "onnx.Sigmoid", "onnx.Tanh", "onnx.Exp"}, true)
      .Cases({"onnx.Sqrt", "onnx.Abs", "onnx.Neg", "onnx.Log"}, true)
      .Cases({"onnx.Cos", "onnx.Sin", "onnx.Floor", "onnx.Ceil"}, true)
      .Cases({"onnx.Round", "onnx.Sign", "onnx.Erf", "onnx.IsInf"}, true)
      .Cases({"onnx.IsNaN", "onnx.Reciprocal", "onnx.Elu"}, true)
      .Cases({"onnx.LeakyRelu", "onnx.HardSigmoid", "onnx.HardSwish"}, true)
      .Default(false);
}

bool isChannelsLastAnchor(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "onnx.Conv" || name == "onnx.AveragePool" ||
         name == "onnx.MaxPool" || name == "onnx.MaxPoolSingleOut" ||
         name == "onnx.GlobalAveragePool";
}

bool isMigratedElementwise(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases({"onnx.Identity", "onnx.Add", "onnx.Sub", "onnx.Mul"}, true)
      .Cases({"onnx.Div", "onnx.Pow", "onnx.Min", "onnx.Max"}, true)
      .Cases({"onnx.Relu", "onnx.Sigmoid", "onnx.Tanh", "onnx.Exp"}, true)
      .Cases({"onnx.Log", "onnx.Sqrt", "onnx.Cos", "onnx.Sin"}, true)
      .Cases({"onnx.Abs", "onnx.Neg", "onnx.Round", "onnx.Sign"}, true)
      .Cases({"onnx.Floor", "onnx.Gelu", "onnx.Elu", "onnx.Reciprocal"}, true)
      .Cases({"onnx.HardSwish", "onnx.HardSigmoid", "onnx.LeakyRelu"}, true)
      .Default(false);
}

bool isMigratedLayoutOperation(Operation *op) {
  if (isa<ONNXConvOp, ONNXAveragePoolOp, ONNXMaxPoolSingleOutOp>(op)) {
    auto inputType =
        op->getNumOperands() == 0
            ? RankedTensorType()
            : dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    return inputType && inputType.getRank() == 4 &&
           inputType.getElementType().isF32();
  }
  return isMigratedElementwise(op) ||
         isa<ONNXMatMulOp, ONNXGemmOp, ONNXReshapeOp, ONNXReturnOp, ONNXSliceOp,
             ONNXSqueezeOp, ONNXUnsqueezeOp, ONNXConcatOp, ONNXTransposeOp,
             ONNXSoftmaxOp, ONNXGatherOp, ONNXGatherElementsOp, ONNXGatherNDOp,
             ONNXScatterElementsOp, ONNXScatterNDOp, ONNXReduceMeanOp,
             ONNXReduceMeanV13Op, ONNXReduceMaxV13Op, ONNXReduceSumOp,
             ONNXReduceSumV11Op, func::ReturnOp>(op);
}

bool isSameLayoutMigratedOperation(Operation *op) {
  if (isa<ONNXConvOp, ONNXAveragePoolOp, ONNXMaxPoolSingleOutOp>(op))
    return isMigratedLayoutOperation(op);
  return isMigratedElementwise(op) ||
         isa<ONNXSliceOp, ONNXConcatOp, ONNXSoftmaxOp, ONNXGatherElementsOp,
             ONNXScatterElementsOp, ONNXScatterNDOp>(op);
}

FailureOr<SmallVector<int64_t>> getReductionAxes(Operation *op, int64_t rank) {
  SmallVector<int64_t> axes;
  bool absent = false;
  if (auto reduce = dyn_cast<ONNXReduceMeanOp>(op)) {
    absent = isa<NoneType>(reduce.getAxes().getType());
    if (!absent) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(reduce.getAxes());
      if (failed(values))
        return failure();
      axes = std::move(*values);
    }
  } else if (auto reduce = dyn_cast<ONNXReduceSumOp>(op)) {
    absent = isa<NoneType>(reduce.getAxes().getType());
    if (!absent) {
      FailureOr<SmallVector<int64_t>> values =
          getConstantIntValues(reduce.getAxes());
      if (failed(values))
        return failure();
      axes = std::move(*values);
    }
  } else if (auto attr = op->getAttrOfType<ArrayAttr>("axes")) {
    for (Attribute element : attr)
      axes.push_back(cast<IntegerAttr>(element).getValue().getSExtValue());
  } else {
    absent = true;
  }

  bool noopWithEmptyAxes = false;
  if (auto attr = op->getAttrOfType<IntegerAttr>("noop_with_empty_axes"))
    noopWithEmptyAxes = attr.getValue().getSExtValue() != 0;
  if ((absent || axes.empty()) && noopWithEmptyAxes)
    return SmallVector<int64_t>{};
  if (absent || axes.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }
  SmallVector<bool> seen(rank, false);
  for (int64_t &axis : axes) {
    axis = normalizeAxis(axis, rank);
    if (axis < 0 || axis >= rank || seen[axis])
      return failure();
    seen[axis] = true;
  }
  return axes;
}

} // namespace

ONNXToTFLLayoutAnalysis::ONNXToTFLLayoutAnalysis(ModuleOp module)
    : module(module) {
  collectValues();
  analyze();
}

const ONNXToTFLLayoutAnalysis::Layout &ONNXToTFLLayoutAnalysis::getLayout(
    Value value) const {
  static const Layout unknown;
  auto found = layouts.find(value);
  return found == layouts.end() ? unknown : found->second;
}

bool ONNXToTFLLayoutAnalysis::hasKnownLayout(Value value) const {
  return getLayout(value).state == State::Known;
}

bool ONNXToTFLLayoutAnalysis::hasIdentityLayout(Value value) const {
  const Layout &layout = getLayout(value);
  if (layout.state != State::Known)
    return false;
  for (auto [position, dimension] : llvm::enumerate(layout.physicalToLogical))
    if (static_cast<int64_t>(position) != dimension)
      return false;
  return true;
}

void ONNXToTFLLayoutAnalysis::collectValues() {
  auto collect = [&](Value value) {
    if (!isStaticRankedTensor(value) || layouts.count(value))
      return;
    layouts.try_emplace(value, Layout{});
    values.push_back(value);
  };
  module.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      collect(operand);
    for (Value result : op->getResults())
      collect(result);
  });
}

ONNXToTFLLayoutAnalysis::Layout ONNXToTFLLayoutAnalysis::getIdentity(
    int64_t rank) {
  Layout layout;
  layout.state = State::Known;
  layout.physicalToLogical.resize(rank);
  std::iota(
      layout.physicalToLogical.begin(), layout.physicalToLogical.end(), 0);
  return layout;
}

ONNXToTFLLayoutAnalysis::Layout ONNXToTFLLayoutAnalysis::getChannelsLast(
    int64_t rank) {
  Layout layout = getIdentity(rank);
  if (rank < 3)
    return layout;
  layout.physicalToLogical.clear();
  layout.physicalToLogical.push_back(0);
  for (int64_t dimension = 2; dimension < rank; ++dimension)
    layout.physicalToLogical.push_back(dimension);
  layout.physicalToLogical.push_back(1);
  return layout;
}

bool ONNXToTFLLayoutAnalysis::merge(Value value, const Layout &proposed) {
  auto found = layouts.find(value);
  if (found == layouts.end() || proposed.state != State::Known)
    return false;
  Layout &current = found->second;
  if (current.state == State::Conflict)
    return false;
  if (current.state == State::Unknown) {
    current = proposed;
    return true;
  }
  if (current.physicalToLogical == proposed.physicalToLogical)
    return false;
  current.state = State::Conflict;
  current.physicalToLogical.clear();
  return true;
}

bool ONNXToTFLLayoutAnalysis::propagateEqual(Value lhs, Value rhs) {
  if (!canShareLayout(lhs, rhs))
    return false;
  return propagateSameRank(lhs, rhs);
}

bool ONNXToTFLLayoutAnalysis::propagateSameRank(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != rhsType.getRank())
    return false;
  bool changed = merge(rhs, getLayout(lhs));
  changed |= merge(lhs, getLayout(rhs));
  return changed;
}

bool ONNXToTFLLayoutAnalysis::propagateReshape(Operation *op) {
  if (op->getNumOperands() == 0 || op->getNumResults() != 1)
    return false;
  Value input = op->getOperand(0);
  Value output = op->getResult(0);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType || !inputType.hasStaticShape() ||
      !outputType.hasStaticShape())
    return false;

  bool changed = false;
  const Layout inputLayout = getLayout(input);
  if (inputLayout.state == State::Conflict) {
    changed |= merge(output, inputLayout);
  } else if (inputLayout.state == State::Known) {
    FailureOr<SmallVector<int64_t>> proposed =
        computeONNXToTFLSingletonReshapeLayout(inputType.getShape(),
            outputType.getShape(), inputLayout.physicalToLogical);
    if (succeeded(proposed))
      changed |= merge(output, Layout{State::Known, *proposed});
  }

  const Layout outputLayout = getLayout(output);
  if (outputLayout.state == State::Conflict) {
    changed |= merge(input, outputLayout);
  } else if (outputLayout.state == State::Known) {
    FailureOr<SmallVector<int64_t>> proposed =
        computeONNXToTFLSingletonReshapeLayout(outputType.getShape(),
            inputType.getShape(), outputLayout.physicalToLogical);
    if (succeeded(proposed))
      changed |= merge(input, Layout{State::Known, *proposed});
  }
  return changed;
}

bool ONNXToTFLLayoutAnalysis::propagateTranspose(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  Value input = op->getOperand(0);
  Value output = op->getResult(0);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!inputType || !outputType || inputType.getRank() != outputType.getRank())
    return false;
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> permutation = getPermutation(op, rank);
  if (permutation.empty())
    return false;

  bool changed = false;
  const Layout inputLayout = getLayout(input);
  if (inputLayout.state == State::Conflict) {
    changed |= merge(output, inputLayout);
  } else if (inputLayout.state == State::Known) {
    SmallVector<int64_t> inverse(rank);
    for (int64_t dimension = 0; dimension < rank; ++dimension)
      inverse[permutation[dimension]] = dimension;
    Layout proposed{State::Known, {}};
    proposed.physicalToLogical.reserve(rank);
    for (int64_t physicalDimension : inputLayout.physicalToLogical)
      proposed.physicalToLogical.push_back(inverse[physicalDimension]);
    changed |= merge(output, proposed);
  }

  const Layout outputLayout = getLayout(output);
  if (outputLayout.state == State::Conflict) {
    changed |= merge(input, outputLayout);
  } else if (outputLayout.state == State::Known) {
    Layout proposed{State::Known, {}};
    proposed.physicalToLogical.reserve(rank);
    for (int64_t physicalDimension : outputLayout.physicalToLogical)
      proposed.physicalToLogical.push_back(permutation[physicalDimension]);
    changed |= merge(input, proposed);
  }
  return changed;
}

void ONNXToTFLLayoutAnalysis::analyze() {
  module.walk([&](Operation *op) {
    if (isChannelsLastAnchor(op)) {
      if (op->getNumOperands() > 0) {
        auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
        if (type)
          merge(op->getOperand(0), getChannelsLast(type.getRank()));
      }
      if (op->getNumResults() > 0) {
        auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (type)
          merge(op->getResult(0), getChannelsLast(type.getRank()));
      }
    }
    if (isa<ONNXMatMulOp, ONNXGemmOp>(op)) {
      for (Value operand : op->getOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
          merge(operand, getIdentity(type.getRank()));
      }
      for (Value result : op->getResults()) {
        if (auto type = dyn_cast<RankedTensorType>(result.getType()))
          merge(result, getIdentity(type.getRank()));
      }
    }
    if (auto reshape = dyn_cast<ONNXReshapeOp>(op)) {
      auto inputType = dyn_cast<RankedTensorType>(reshape.getData().getType());
      auto outputType =
          dyn_cast<RankedTensorType>(reshape.getResult().getType());
      if (!inputType || !outputType)
        return;
      SmallVector<int64_t> identity =
          getIdentity(inputType.getRank()).physicalToLogical;
      if (failed(computeONNXToTFLSingletonReshapeLayout(
              inputType.getShape(), outputType.getShape(), identity))) {
        merge(reshape.getData(), getIdentity(inputType.getRank()));
        merge(reshape.getResult(), getIdentity(outputType.getRank()));
      }
    }
  });

  auto propagateToFixedPoint = [&]() {
    bool changed;
    do {
      changed = false;
      module.walk([&](Operation *op) {
        if (isa<ONNXTransposeOp>(op)) {
          changed |= propagateTranspose(op);
          return;
        }
        if (isa<ONNXReshapeOp, ONNXSqueezeOp, ONNXUnsqueezeOp>(op)) {
          changed |= propagateReshape(op);
          return;
        }
        if (auto gather = dyn_cast<ONNXGatherOp>(op)) {
          Value data = gather.getData();
          Value indices = gather.getIndices();
          Value result = gather.getResult();
          auto dataType = dyn_cast<RankedTensorType>(data.getType());
          auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
          auto resultType = dyn_cast<RankedTensorType>(result.getType());
          if (!dataType || !indicesType || !resultType)
            return;
          const Layout &dataLayout = getLayout(data);
          if (dataLayout.state != State::Known)
            return;
          int64_t dataRank = dataType.getRank();
          int64_t indicesRank = indicesType.getRank();
          int64_t axis = normalizeAxis(gather.getAxis(), dataRank);
          if (axis < 0 || axis >= dataRank)
            return;
          Layout indicesLayout = getLayout(indices);
          if (indicesLayout.state == State::Unknown)
            indicesLayout = getIdentity(indicesRank);
          if (indicesLayout.state != State::Known)
            return;
          auto mapDataDimension = [&](int64_t logicalDimension) {
            return logicalDimension < axis ? logicalDimension
                                           : logicalDimension + indicesRank - 1;
          };
          Layout proposed{State::Known, {}};
          for (int64_t physicalDataDimension : dataLayout.physicalToLogical) {
            if (physicalDataDimension == axis) {
              for (int64_t physicalIndexDimension :
                  indicesLayout.physicalToLogical)
                proposed.physicalToLogical.push_back(
                    axis + physicalIndexDimension);
            } else {
              proposed.physicalToLogical.push_back(
                  mapDataDimension(physicalDataDimension));
            }
          }
          if (static_cast<int64_t>(proposed.physicalToLogical.size()) ==
              resultType.getRank())
            changed |= merge(result, proposed);
          return;
        }
        if (auto gather = dyn_cast<ONNXGatherElementsOp>(op)) {
          changed |= propagateSameRank(gather.getData(), gather.getResult());
          return;
        }
        if (auto scatter = dyn_cast<ONNXScatterElementsOp>(op)) {
          changed |= propagateSameRank(scatter.getData(), scatter.getResult());
          changed |=
              propagateSameRank(scatter.getUpdates(), scatter.getResult());
          return;
        }
        if (auto scatter = dyn_cast<ONNXScatterNDOp>(op)) {
          changed |= propagateSameRank(scatter.getData(), scatter.getResult());
          return;
        }
        if (isa<ONNXSliceOp, ONNXConcatOp, ONNXSoftmaxOp>(op) &&
            op->getNumResults() == 1) {
          Value result = op->getResult(0);
          for (Value operand : op->getOperands())
            changed |= propagateSameRank(operand, result);
          return;
        }
        if (isa<ONNXReduceMeanOp, ONNXReduceMeanV13Op, ONNXReduceMaxV13Op,
                ONNXReduceSumOp, ONNXReduceSumV11Op>(op) &&
            op->getNumOperands() > 0 && op->getNumResults() == 1) {
          Value input = op->getOperand(0);
          Value result = op->getResult(0);
          auto inputType = dyn_cast<RankedTensorType>(input.getType());
          auto resultType = dyn_cast<RankedTensorType>(result.getType());
          if (!inputType || !resultType)
            return;
          FailureOr<SmallVector<int64_t>> axes =
              getReductionAxes(op, inputType.getRank());
          if (failed(axes))
            return;
          if (axes->empty()) {
            changed |= propagateEqual(input, result);
            return;
          }
          bool keepDims = true;
          if (auto attr = op->getAttrOfType<IntegerAttr>("keepdims"))
            keepDims = attr.getValue().getSExtValue() != 0;
          const Layout inputLayout = getLayout(input);
          if (inputLayout.state == State::Conflict) {
            changed |= merge(result, inputLayout);
          } else if (inputLayout.state == State::Known) {
            FailureOr<SmallVector<int64_t>> resultLayout =
                computeONNXToTFLReducedLayout(
                    inputLayout.physicalToLogical, *axes, keepDims);
            if (succeeded(resultLayout) &&
                static_cast<int64_t>(resultLayout->size()) ==
                    resultType.getRank())
              changed |= merge(result, Layout{State::Known, *resultLayout});
          }
          if (keepDims)
            changed |= propagateEqual(input, result);
          return;
        }
        if (!isLayoutPreservingElementwise(op) || op->getNumResults() != 1)
          return;
        Value result = op->getResult(0);
        for (Value operand : op->getOperands())
          changed |= propagateEqual(operand, result);
      });
    } while (changed);
  };
  propagateToFixedPoint();

  // Choose identity as the basis for each otherwise unconstrained component,
  // then derive the remaining layouts from its actual operations. Seeding one
  // value at a time is important: assigning identity independently to both
  // sides of a Transpose would turn a layout choice into real data movement.
  for (Value value : values) {
    if (getLayout(value).state != State::Unknown)
      continue;
    auto type = cast<RankedTensorType>(value.getType());
    merge(value, getIdentity(type.getRank()));
    propagateToFixedPoint();
  }
}

int64_t ONNXToTFLLayoutAnalysis::applySupportedTypeEncodings() {
  llvm::DenseSet<Value> eligible;
  for (Value value : values) {
    if (!hasKnownLayout(value))
      continue;
    if (auto result = dyn_cast<OpResult>(value)) {
      if (!isMigratedLayoutOperation(result.getOwner()))
        continue;
    }
    if (llvm::all_of(value.getUsers(), isMigratedLayoutOperation))
      eligible.insert(value);
  }

  // Same-layout lowering does not insert conversions between equal-rank
  // operands and results. Retain only complete, consistently laid-out groups;
  // MatMul, Reshape, and Transpose handle their boundaries explicitly.
  bool changed;
  do {
    changed = false;
    module.walk([&](Operation *op) {
      if (!isSameLayoutMigratedOperation(op) || op->getNumResults() != 1)
        return;
      Value result = op->getResult(0);
      auto resultType = dyn_cast<RankedTensorType>(result.getType());
      if (!resultType)
        return;
      SmallVector<Value> group;
      if (auto gather = dyn_cast<ONNXGatherElementsOp>(op)) {
        group = {gather.getData(), result};
      } else if (auto scatter = dyn_cast<ONNXScatterElementsOp>(op)) {
        group = {scatter.getData(), scatter.getUpdates(), result};
      } else if (auto scatter = dyn_cast<ONNXScatterNDOp>(op)) {
        group = {scatter.getData(), result};
      } else if (isa<ONNXConvOp, ONNXAveragePoolOp, ONNXMaxPoolSingleOutOp>(
                     op)) {
        group = {op->getOperand(0), result};
      } else {
        group.push_back(result);
        for (Value operand : op->getOperands()) {
          auto operandType = dyn_cast<RankedTensorType>(operand.getType());
          if (operandType && operandType.getRank() == resultType.getRank())
            group.push_back(operand);
        }
      }
      bool complete = llvm::all_of(group, [&](Value value) {
        return eligible.contains(value) && hasKnownLayout(value) &&
               getLayout(value).physicalToLogical ==
                   getLayout(result).physicalToLogical;
      });
      if (complete)
        return;
      for (Value value : group)
        changed |= eligible.erase(value);
    });
  } while (changed);

  for (Value value : eligible) {
    auto type = cast<RankedTensorType>(value.getType());
    auto encoding = DenseI64ArrayAttr::get(
        module.getContext(), getLayout(value).physicalToLogical);
    value.setType(RankedTensorType::get(
        type.getShape(), type.getElementType(), encoding));
  }

  module.walk([&](func::FuncOp function) {
    if (function->hasAttr("tf.entry_function")) {
      // Internal permutations are not part of the non-rank-4 graph ABI.
      // Bridge them with logical identity Transposes: layout-aware lowering
      // materializes only the required physical permutation, while internal
      // consumers keep their optimized encodings. Compare permutations, not
      // shapes, because exchanging equal-sized axes still reorders values.
      auto needsBoundaryConversion = [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        if (!tensor || tensor.getRank() == 4 || !tensor.getEncoding())
          return false;
        SmallVector<int64_t> layout =
            getONNXToTFLPhysicalToLogicalLayout(tensor);
        for (auto [axis, dimension] : llvm::enumerate(layout))
          if (static_cast<int64_t>(axis) != dimension)
            return true;
        return false;
      };
      OpBuilder builder(function.getContext());
      auto bridge = [&](Value value, RankedTensorType resultType) {
        SmallVector<int64_t> identity(resultType.getRank());
        std::iota(identity.begin(), identity.end(), 0);
        return ONNXTransposeOp::create(builder, function.getLoc(), resultType,
            value, builder.getI64ArrayAttr(identity));
      };
      Operation *terminator = function.getBody().front().getTerminator();
      for (BlockArgument argument : function.getArguments()) {
        if (!needsBoundaryConversion(argument.getType()))
          continue;
        auto internalType = cast<RankedTensorType>(argument.getType());
        argument.setType(stripONNXToTFLLayoutEncoding(internalType));
        builder.setInsertionPointToStart(&function.getBody().front());
        auto converted = bridge(argument, internalType);
        // Direct input-to-output aliases already have the public layout.
        // Preserve the SSA alias as well as its signature tensor mapping.
        argument.replaceUsesWithIf(converted.getResult(), [&](OpOperand &use) {
          return use.getOwner() != converted.getOperation() &&
                 use.getOwner() != terminator;
        });
      }
      builder.setInsertionPoint(terminator);
      for (OpOperand &operand : terminator->getOpOperands()) {
        if (!needsBoundaryConversion(operand.get().getType()))
          continue;
        auto internalType = cast<RankedTensorType>(operand.get().getType());
        operand.set(
            bridge(operand.get(), stripONNXToTFLLayoutEncoding(internalType))
                .getResult());
      }
    }
    SmallVector<Type> inputs;
    inputs.reserve(function.getNumArguments());
    for (BlockArgument argument : function.getArguments())
      inputs.push_back(argument.getType());
    SmallVector<Type> results(function.getResultTypes());
    Operation *terminator = function.getBody().front().getTerminator();
    if (terminator->getNumOperands() == results.size())
      llvm::copy(terminator->getOperandTypes(), results.begin());
    function.setType(FunctionType::get(module.getContext(), inputs, results));
  });
  return eligible.size();
}

std::string ONNXToTFLLayoutAnalysis::getDescription(const Layout &layout) {
  if (layout.state == State::Unknown)
    return "unknown";
  if (layout.state == State::Conflict)
    return "conflict";
  bool identity = true;
  for (auto [position, dimension] : llvm::enumerate(layout.physicalToLogical))
    identity &= static_cast<int64_t>(position) == dimension;
  if (identity)
    return "identity";
  if (layout.physicalToLogical.size() >= 3 &&
      layout.physicalToLogical.front() == 0 &&
      layout.physicalToLogical.back() == 1) {
    bool channelsLast = true;
    for (int64_t position = 1;
         position < static_cast<int64_t>(layout.physicalToLogical.size()) - 1;
         ++position)
      channelsLast &= layout.physicalToLogical[position] == position + 1;
    if (channelsLast)
      return "channels-last";
  }
  std::string description = "permuted[";
  llvm::raw_string_ostream stream(description);
  for (auto [position, dimension] : llvm::enumerate(layout.physicalToLogical)) {
    if (position != 0)
      stream << ',';
    stream << dimension;
  }
  stream << ']';
  return description;
}

void ONNXToTFLLayoutAnalysis::printDiagnostics(
    llvm::raw_ostream &stream, bool printValues) const {
  int64_t unknown = 0;
  int64_t identity = 0;
  int64_t channelsLast = 0;
  int64_t permuted = 0;
  int64_t conflict = 0;
  for (Value value : values) {
    std::string description = getDescription(getLayout(value));
    if (description == "unknown")
      ++unknown;
    else if (description == "identity")
      ++identity;
    else if (description == "channels-last")
      ++channelsLast;
    else if (description == "conflict")
      ++conflict;
    else
      ++permuted;
  }
  stream << "[onnx-to-tfl-layout] values=" << values.size()
         << " identity=" << identity << " channels-last=" << channelsLast
         << " permuted=" << permuted << " unknown=" << unknown
         << " conflict=" << conflict << '\n';
  if (!printValues)
    return;
  AsmState state(module);
  for (Value value : values) {
    stream << "[onnx-to-tfl-layout] ";
    value.printAsOperand(stream, state);
    stream << " rank=" << cast<RankedTensorType>(value.getType()).getRank()
           << " layout=" << getDescription(getLayout(value)) << '\n';
  }
}

} // namespace onnx_mlir
