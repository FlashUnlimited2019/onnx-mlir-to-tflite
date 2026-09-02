/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

#ifndef ONNX_MLIR_ONNX_TO_TFL_LAYOUT_ANALYSIS_H
#define ONNX_MLIR_ONNX_TO_TFL_LAYOUT_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace onnx_mlir {

class ONNXToTFLLayoutAnalysis {
public:
  enum class State { Unknown, Known, Conflict };

  struct Layout {
    State state = State::Unknown;
    llvm::SmallVector<int64_t> physicalToLogical;
  };

  explicit ONNXToTFLLayoutAnalysis(mlir::ModuleOp module);

  const Layout &getLayout(mlir::Value value) const;
  bool hasKnownLayout(mlir::Value value) const;
  bool hasIdentityLayout(mlir::Value value) const;
  void printDiagnostics(llvm::raw_ostream &stream, bool printValues) const;
  int64_t applySupportedTypeEncodings();

private:
  mlir::ModuleOp module;
  llvm::DenseMap<mlir::Value, Layout> layouts;
  llvm::SmallVector<mlir::Value> values;

  void analyze();
  void collectValues();
  bool merge(mlir::Value value, const Layout &proposed);
  bool propagateEqual(mlir::Value lhs, mlir::Value rhs);
  bool propagateSameRank(mlir::Value lhs, mlir::Value rhs);
  bool propagateReshape(mlir::Operation *op);
  bool propagateTranspose(mlir::Operation *op);
  static Layout getIdentity(int64_t rank);
  static Layout getChannelsLast(int64_t rank);
  static std::string getDescription(const Layout &layout);
};

} // namespace onnx_mlir

#endif
