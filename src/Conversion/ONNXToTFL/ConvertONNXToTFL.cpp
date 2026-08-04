/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- ConvertONNXToTFL.cpp - ONNX to TensorFlow Lite lowering -------===//

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

bool isSupportedONNXOperation(Operation *op) {
  return isa<ONNXEntryPointOp, ONNXReturnOp, ONNXNoneOp, ONNXConstantOp,
      ONNXIdentityOp, ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp, ONNXReluOp,
      ONNXSigmoidOp, ONNXTanhOp, ONNXMatMulOp, ONNXGemmOp, ONNXReshapeOp,
      ONNXTransposeOp, ONNXConcatOp, ONNXSoftmaxOp, ONNXConvOp,
      ONNXMaxPoolSingleOutOp, ONNXReduceMeanV13Op>(op);
}

StringRef getONNXValueName(func::FuncOp function, unsigned index, bool result) {
  StringAttr name =
      result ? function.getResultAttrOfType<StringAttr>(index, "onnx.name")
             : function.getArgAttrOfType<StringAttr>(index, "onnx.name");
  return name ? name.getValue() : StringRef();
}

std::string joinValueNames(func::FuncOp function, bool result) {
  unsigned count =
      result ? function.getNumResults() : function.getNumArguments();
  std::string joined;
  llvm::raw_string_ostream stream(joined);
  for (unsigned i = 0; i < count; ++i) {
    if (i != 0)
      stream << ',';
    StringRef name = getONNXValueName(function, i, result);
    if (name.empty())
      stream << (result ? "output" : "input") << i;
    else
      stream << name;
  }
  return joined;
}

LogicalResult prepareEntryPoint(ModuleOp module, TypeConverter &typeConverter) {
  SmallVector<ONNXEntryPointOp> entryPoints;
  module.walk([&](ONNXEntryPointOp op) { entryPoints.push_back(op); });
  if (entryPoints.size() != 1) {
    module.emitError()
        << "ONNXToTFL MVP requires exactly one ONNX entry point, "
        << "but found " << entryPoints.size();
    return failure();
  }

  ONNXEntryPointOp entry = entryPoints.front();
  auto symbol = entry->getAttrOfType<SymbolRefAttr>("func");
  if (!symbol) {
    entry.emitError("ONNX entry point is missing its func symbol");
    return failure();
  }
  func::FuncOp function =
      module.lookupSymbol<func::FuncOp>(symbol.getLeafReference());
  if (!function) {
    entry.emitError() << "cannot resolve ONNX entry function " << symbol;
    return failure();
  }
  if (function.getName() != "main" && module.lookupSymbol("main")) {
    entry.emitError("cannot rename ONNX entry function to main: symbol exists");
    return failure();
  }

  for (Type type : function.getArgumentTypes()) {
    if (failed(validateStaticF32Tensor(function, type, "function input")))
      return failure();
  }
  for (Type type : function.getResultTypes()) {
    if (failed(validateStaticF32Tensor(function, type, "function output")))
      return failure();
  }

  Builder builder(module.getContext());
  SmallVector<Type> convertedInputs;
  SmallVector<Type> convertedResults;
  convertedInputs.reserve(function.getNumArguments());
  convertedResults.reserve(function.getNumResults());
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
    Type converted = typeConverter.convertType(type);
    convertedInputs.push_back(converted);
    function.getArgument(index).setType(converted);
  }
  for (Type type : function.getResultTypes())
    convertedResults.push_back(typeConverter.convertType(type));
  function.setType(builder.getFunctionType(convertedInputs, convertedResults));

  SmallVector<NamedAttribute> entryAttributes{
      builder.getNamedAttr(
          "inputs", builder.getStringAttr(joinValueNames(function, false))),
      builder.getNamedAttr(
          "outputs", builder.getStringAttr(joinValueNames(function, true))),
      builder.getNamedAttr("control_outputs", builder.getStringAttr(""))};
  function->setAttr(
      "tf.entry_function", builder.getDictionaryAttr(entryAttributes));
  function.setPublic();
  function.setName("main");
  entry.erase();
  return success();
}

class ConvertONNXToTFLPass final
    : public PassWrapper<ConvertONNXToTFLPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToTFLPass)

  StringRef getArgument() const final { return "convert-onnx-to-tfl"; }
  StringRef getDescription() const final {
    return "Lower the ONNXToTFL MVP operator set to TFL dialect MLIR";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFLCompatibilityDialect, arith::ArithDialect,
        func::FuncDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    llvm::DenseSet<Type> targetRank4Types;
    module.walk([&](Operation *op) {
      for (Type type :
          llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        Type converted = convertRank4NCHWToNHWCType(type);
        if (converted != type)
          targetRank4Types.insert(converted);
      }
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([&](RankedTensorType type) -> Type {
      if (targetRank4Types.contains(type))
        return type;
      return convertRank4NCHWToNHWCType(type);
    });

    if (failed(prepareEntryPoint(module, typeConverter))) {
      signalPassFailure();
      return;
    }

    bool hasUnsupportedOperation = false;
    module.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getDialectNamespace() != "onnx")
        return WalkResult::advance();
      if (isSupportedONNXOperation(op))
        return WalkResult::advance();
      op->emitError() << "unsupported ONNX operation: " << op->getName();
      hasUnsupportedOperation = true;
      return WalkResult::interrupt();
    });
    if (hasUnsupportedOperation) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateLoweringONNXElementwiseOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXMatMulGemmOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSoftmaxOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConcatOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConstantOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReshapeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXTransposeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConvOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXMaxPoolOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReduceMeanOpToTFLPatterns(patterns, typeConverter);

    ConversionTarget target(*context);
    target.addLegalDialect<BuiltinDialect, TFLCompatibilityDialect,
        arith::ArithDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addIllegalDialect<ONNXDialect>();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    if (failed(verify(module))) {
      module.emitError("TFL compatibility module verification failed");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToTFLPass() {
  return std::make_unique<ConvertONNXToTFLPass>();
}

} // namespace onnx_mlir
