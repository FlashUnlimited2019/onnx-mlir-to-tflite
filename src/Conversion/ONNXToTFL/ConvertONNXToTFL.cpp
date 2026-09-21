/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

//===----- ConvertONNXToTFL.cpp - ONNX to TensorFlow Lite lowering -------===//

#include "src/Conversion/ONNXToTFL/LayoutAnalysis.hpp"
#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/JSON.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

bool isSupportedONNXOperation(Operation *op) {
  return isa<ONNXEntryPointOp, ONNXReturnOp, ONNXNoneOp, ONNXConstantOp,
             ONNXIdentityOp, ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp,
             ONNXPowOp, ONNXArgMaxOp, ONNXArgMinOp, ONNXCastOp, ONNXEqualOp,
             ONNXGreaterOp, ONNXGreaterOrEqualOp, ONNXModOp, ONNXReluOp,
             ONNXSigmoidOp, ONNXTanhOp, ONNXExpOp, ONNXSqrtOp, ONNXCosOp,
             ONNXSinOp, ONNXAbsOp, ONNXNegOp, ONNXRoundOp, ONNXSignOp,
             ONNXFloorOp, ONNXCeilOp, ONNXNotOp, ONNXHardmaxOp,
             ONNXReciprocalOp, ONNXBitShiftOp, ONNXEluOp, ONNXLessOp,
             ONNXLessOrEqualOp, ONNXGeluOp, ONNXLogOp, ONNXPReluOp, ONNXSeluOp,
             ONNXSoftplusOp, ONNXSoftsignOp, ONNXMeanOp, ONNXAndOp, ONNXOrOp,
             ONNXXorOp, ONNXCumSumOp, ONNXOneHotOp, ONNXLpNormalizationOp,
             ONNXCeluOp, ONNXThresholdedReluOp, ONNXAsinOp, ONNXAcosOp,
             ONNXAtanOp, ONNXAtanhOp, ONNXSinhOp, ONNXCoshOp, ONNXAsinhOp,
             ONNXAcoshOp, ONNXErfOp, ONNXIsInfOp, ONNXIsNaNOp, ONNXShrinkOp,
             ONNXIm2ColOp, ONNXMeanVarianceNormalizationOp, ONNXMishOp,
             ONNXDFTOp, ONNXDetOp, ONNXNegativeLogLikelihoodLossOp,
             ONNXMatMulOp, ONNXGemmOp, ONNXTopKOp, ONNXSTFTOp, ONNXReshapeOp,
             ONNXTransposeOp, ONNXConcatOp, ONNXExpandOp, ONNXSoftmaxOp,
             ONNXConvOp, ONNXMaxPoolSingleOutOp, ONNXReduceMeanOp,
             ONNXReduceMeanV13Op, ONNXReduceMaxOp, ONNXReduceMaxV13Op,
             ONNXReduceMinV13Op, ONNXReduceSumOp, ONNXReduceSumV11Op,
             ONNXReduceL1Op, ONNXReduceLogSumExpOp, ONNXReduceProdOp,
             ONNXReduceSumSquareOp, ONNXSliceOp, ONNXSplitOp, ONNXResizeOp,
             ONNXPadOp, ONNXFlattenOp, ONNXGatherOp, ONNXGatherElementsOp,
             ONNXScatterElementsOp, ONNXScatterNDOp, ONNXSqueezeOp,
             ONNXUnsqueezeOp, ONNXTileOp, ONNXWhereOp, ONNXGatherNDOp,
             ONNXClipOp, ONNXUpsampleAndPadOp, ONNXDepthToSpaceOp,
             ONNXSpaceToDepthOp, ONNXMinOp, ONNXMaxOp, ONNXLRNOp, ONNXLSTMOp,
             ONNXRNNOp, ONNXGRUOp, ONNXReverseSequenceOp, ONNXGridSampleOp,
             ONNXMaxPoolOp, ONNXMaxUnpoolOp, ONNXCompressOp, ONNXTriluOp,
             ONNXCenterCropPadOp, ONNXEyeLikeOp, ONNXCol2ImOp, ONNXAffineGridOp,
             ONNXDeformConvOp, ONNXRoiAlignOp, ONNXLpPoolOp, ONNXGlobalLpPoolOp,
             ONNXBatchNormalizationInferenceModeOp, ONNXAttentionOp>(op) ||
         isa<ONNXAveragePoolOp, ONNXHardSigmoidOp, ONNXHardSwishOp,
             ONNXLayerNormalizationOp, ONNXRMSLayerNormalizationOp,
             ONNXLeakyReluOp>(op) ||
         ([](Operation *candidate) {
           auto custom = dyn_cast<ONNXCustomOp>(candidate);
           if (!custom)
             return false;
           auto domain = candidate->getAttrOfType<StringAttr>("domain_name");
           if (!domain || domain.getValue() != "com.microsoft")
             return false;
           StringRef name = custom.getFunctionName();
           return name == "Attention" || name == "MultiHeadAttention" ||
                  name == "GroupQueryAttention";
         })(op);
}

std::string getEntryValueName(
    func::FuncOp function, unsigned index, bool result) {
  StringAttr name =
      result ? function.getResultAttrOfType<StringAttr>(index, "onnx.name")
             : function.getArgAttrOfType<StringAttr>(index, "onnx.name");
  if (name && !name.getValue().empty())
    return name.getValue().str();
  return (result ? "output" : "input") + std::to_string(index);
}

std::string joinValueNames(func::FuncOp function, bool result) {
  unsigned count =
      result ? function.getNumResults() : function.getNumArguments();
  std::string joined;
  llvm::raw_string_ostream stream(joined);
  for (unsigned i = 0; i < count; ++i) {
    if (i != 0)
      stream << ',';
    stream << getEntryValueName(function, i, result);
  }
  return joined;
}

LogicalResult validateEntryTensor(Operation *op, Type type, StringRef role) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return op->emitError()
           << "dynamic or unranked tensor shape is not supported by "
              "ONNXToTFL MVP ("
           << role << ")";
  }
  if (!tensorType.hasStaticShape()) {
    return op->emitError()
           << "dynamic tensor shape is not supported by ONNXToTFL MVP "
              "("
           << role << ": " << type << ")";
  }
  if (tensorType.getRank() < 1) {
    return op->emitError() << "ONNXToTFL MVP requires activation rank >= 1 ("
                           << role << ")";
  }
  Type elementType = tensorType.getElementType();
  if (!elementType.isF32() && !elementType.isInteger(1) &&
      !elementType.isSignlessInteger(32) &&
      !elementType.isSignlessInteger(64)) {
    return op->emitError()
           << "ONNXToTFL MVP only supports bool, f32, i32, or i64 entry "
              "tensors ("
           << role << ": " << type << ")";
  }
  return success();
}

LogicalResult prepareEntryPoint(ModuleOp module) {
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
    if (failed(validateEntryTensor(function, type, "function input")))
      return failure();
  }
  for (Type type : function.getResultTypes()) {
    if (failed(validateEntryTensor(function, type, "function output")))
      return failure();
  }

  Builder builder(module.getContext());
  SmallVector<NamedAttribute> entryAttributes{
      builder.getNamedAttr(
          "inputs", builder.getStringAttr(joinValueNames(function, false))),
      builder.getNamedAttr(
          "outputs", builder.getStringAttr(joinValueNames(function, true))),
      builder.getNamedAttr("control_outputs", builder.getStringAttr(""))};
  function->setAttr(
      "tf.entry_function", builder.getDictionaryAttr(entryAttributes));
  // These attributes let the TensorFlow exporter create a SignatureDef and
  // map its public names to the final tensor indices, including aliased
  // outputs. They are metadata only; no SavedModel or TensorFlow graph is used.
  module->setAttr("tf_saved_model.semantics", builder.getUnitAttr());
  function->setAttr("tf_saved_model.exported_names",
      builder.getStrArrayAttr({"serving_default"}));
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    function.setArgAttr(i, "tf_saved_model.index_path",
        builder.getStrArrayAttr({getEntryValueName(function, i, false)}));
  for (unsigned i = 0; i < function.getNumResults(); ++i)
    function.setResultAttr(i, "tf_saved_model.index_path",
        builder.getStrArrayAttr({getEntryValueName(function, i, true)}));
  // The TensorFlow exporter sorts signature maps by name. Preserve public
  // positional names separately so the driver can restore their order even
  // when multiple output names refer to the same tensor after optimization.
  llvm::json::Array inputNames, outputNames;
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    inputNames.push_back(getEntryValueName(function, i, false));
  for (unsigned i = 0; i < function.getNumResults(); ++i)
    outputNames.push_back(getEntryValueName(function, i, true));
  std::string signatureOrder;
  llvm::raw_string_ostream(signatureOrder) << llvm::json::Value(
      llvm::json::Object{{"version", 1}, {"signature_key", "serving_default"},
          {"inputs", std::move(inputNames)},
          {"outputs", std::move(outputNames)}});
  NamedAttrList metadata;
  if (auto existing = module->getAttrOfType<DictionaryAttr>("tfl.metadata"))
    metadata.append(existing.getValue());
  metadata.set(
      "onnx_mlir.signature_order", builder.getStringAttr(signatureOrder));
  module->setAttr("tfl.metadata", metadata.getDictionary(module.getContext()));
  function.setPublic();
  function.setName("main");
  entry.erase();
  return success();
}

LogicalResult finalizeEntryPointTypes(ModuleOp module) {
  func::FuncOp function = module.lookupSymbol<func::FuncOp>("main");
  if (!function)
    return module.emitError("ONNXToTFL entry function main is missing");

  FunctionType oldType = function.getFunctionType();
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.reserve(oldType.getNumInputs());
  results.reserve(oldType.getNumResults());
  for (Type type : oldType.getInputs())
    inputs.push_back(convertONNXToTFLPhysicalType(type));
  for (Type type : oldType.getResults())
    results.push_back(convertONNXToTFLPhysicalType(type));

  // The conversion driver represents a converted entry argument with an
  // unrealized cast because the surrounding function is deliberately kept
  // legal during operation conversion. Retype the ABI argument to NHWC and
  // remove that type-only boundary cast; no runtime Transpose is introduced.
  for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
    Type convertedType = inputs[index];
    if (argument.getType() == convertedType)
      continue;
    argument.setType(convertedType);
    SmallVector<UnrealizedConversionCastOp> casts;
    for (Operation *user : argument.getUsers())
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user))
        casts.push_back(cast);
    for (UnrealizedConversionCastOp cast : casts) {
      if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1 ||
          cast.getInputs().front() != argument ||
          cast.getOutputs().front().getType() != convertedType)
        continue;
      cast.getOutputs().front().replaceAllUsesWith(argument);
      cast.erase();
    }
  }
  function.setType(FunctionType::get(module.getContext(), inputs, results));

  SmallVector<UnrealizedConversionCastOp> remaining;
  module.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getInputs().size() != 1 || cast.getOutputs().size() != 1) {
      remaining.push_back(cast);
      return;
    }
    Value input = cast.getInputs().front();
    Value output = cast.getOutputs().front();
    Type inputType = input.getType();
    Type outputType = output.getType();
    if (inputType != outputType &&
        inputType != convertONNXToTFLPhysicalType(outputType) &&
        outputType != convertONNXToTFLPhysicalType(inputType)) {
      remaining.push_back(cast);
      return;
    }
    output.replaceAllUsesWith(input);
    cast.erase();
  });
  if (!remaining.empty()) {
    remaining.front().emitError(
        "unresolved ONNXToTFL type materialization after ABI conversion");
    return failure();
  }
  return success();
}

class ConvertONNXToTFLPass final
    : public PassWrapper<ConvertONNXToTFLPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToTFLPass)

  ConvertONNXToTFLPass() = default;
  ConvertONNXToTFLPass(const ConvertONNXToTFLPass &pass)
      : PassWrapper<ConvertONNXToTFLPass, OperationPass<ModuleOp>>() {
    layoutDiagnostics = pass.layoutDiagnostics.getValue();
    layoutMode = pass.layoutMode.getValue();
  }

  StringRef getArgument() const final { return "convert-onnx-to-tfl"; }
  StringRef getDescription() const final {
    return "Lower the ONNXToTFL MVP operator set to TFL dialect MLIR";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TFLCompatibilityDialect, arith::ArithDialect,
        func::FuncDialect>();
  }

  Option<std::string> layoutDiagnostics{*this, "layout-diagnostics",
      llvm::cl::desc(
          "Emit read-only per-value layout analysis: none, summary, values"),
      llvm::cl::init("none")};
  Option<std::string> layoutMode{*this, "layout-mode",
      llvm::cl::desc("Select per-value layout lowering: legacy, experimental"),
      llvm::cl::init("legacy")};

  void runOnOperation() final {
    ModuleOp module = getOperation();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return convertONNXToTFLPhysicalType(type);
    });

    if (failed(prepareEntryPoint(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet preprocessingPatterns(&getContext());
    populateONNXToTFLPreprocessingPatterns(preprocessingPatterns);
    populateONNXToTFLStaticUpdatePreprocessingPatterns(preprocessingPatterns);
    populateONNXToTFLTransformerPreprocessingPatterns(preprocessingPatterns);
    if (failed(
            applyPatternsGreedily(module, std::move(preprocessingPatterns)))) {
      module.emitError("ONNXToTFL preprocessing failed");
      signalPassFailure();
      return;
    }

    if (layoutDiagnostics != "none" && layoutDiagnostics != "summary" &&
        layoutDiagnostics != "values") {
      module.emitError() << "invalid layout-diagnostics mode: "
                         << layoutDiagnostics
                         << " (expected none, summary, or values)";
      signalPassFailure();
      return;
    }
    if (layoutMode != "legacy" && layoutMode != "experimental") {
      module.emitError() << "invalid layout-mode: " << layoutMode
                         << " (expected legacy or experimental)";
      signalPassFailure();
      return;
    }
    ONNXToTFLLayoutAnalysis layoutAnalysis(module);
    if (layoutDiagnostics != "none")
      layoutAnalysis.printDiagnostics(
          llvm::errs(), layoutDiagnostics == "values");
    if (layoutMode == "experimental") {
      int64_t encodedValues = layoutAnalysis.applySupportedTypeEncodings();
      if (layoutDiagnostics != "none")
        llvm::errs() << "[onnx-to-tfl-layout] encoded-values=" << encodedValues
                     << '\n';
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
    populateLoweringONNXElementwiseOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXAdditionalElementwiseOpToTFLPatterns(
        patterns, typeConverter);
    populateLoweringONNXAdditionalMathOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXUncommonMathOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXArgMaxOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXComparisonOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXLegacyMathOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXMatMulGemmOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSoftmaxOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXTopKOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConcatOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConstantOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXCastOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXExpandOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXFlattenOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXGatherOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXGatherElementsOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXScatterElementsOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXScatterNDOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXGatherNDOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReshapeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXTransposeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXUnsqueezeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSliceOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSplitOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSequenceOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSqueezeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXTileOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXResizeOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXDepthToSpaceOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSpaceToDepthOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXConvOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXGridSampleOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXStaticSamplingOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXMaxPoolOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXIndexedMaxPoolOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXAveragePoolOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXAttentionOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXBatchNormalizationOpToTFLPatterns(
        patterns, typeConverter);
    populateLoweringONNXLayerNormalizationOpToTFLPatterns(
        patterns, typeConverter);
    populateLoweringONNXLRNOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXLpPoolOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXLSTMOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXStaticRecurrentOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXSTFTOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReduceMeanOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReduceMaxOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXReduceSumOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXPadOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXUpsampleAndPadOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXWhereOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXCompressTriluOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXIm2ColOpToTFLPatterns(patterns, typeConverter);
    populateLoweringONNXStaticUncommonTensorOpToTFLPatterns(
        patterns, typeConverter);

    ConversionTarget target(*context);
    target.addLegalDialect<BuiltinDialect, TFLCompatibilityDialect,
        arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<ONNXDialect>();
    ConversionConfig config;
    config.buildMaterializations = false;
    if (failed(applyPartialConversion(module, target,
            FrozenRewritePatternSet(std::move(patterns)), config))) {
      signalPassFailure();
      return;
    }
    if (failed(optimizeTFLStaticIndexOperations(module))) {
      module.emitError("ONNXToTFL static index optimization failed");
      signalPassFailure();
      return;
    }
    RewritePatternSet postLoweringPatterns(context);
    populateTFLMatMulPostLoweringPatterns(postLoweringPatterns);
    populateTFLTransformerPostLoweringPatterns(postLoweringPatterns);
    SmallVector<Operation *> postLoweringRoots;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "tfl.batch_matmul" || name == "tfl.transpose")
        postLoweringRoots.push_back(op);
    });
    GreedyRewriteConfig postLoweringConfig;
    postLoweringConfig.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps)
        .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled)
        .enableFolding(false)
        .enableConstantCSE(false);
    if (failed(applyOpPatternsGreedily(postLoweringRoots,
            FrozenRewritePatternSet(std::move(postLoweringPatterns)),
            postLoweringConfig))) {
      module.emitError("ONNXToTFL post-lowering optimization failed");
      signalPassFailure();
      return;
    }
    if (failed(finalizeEntryPointTypes(module))) {
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
