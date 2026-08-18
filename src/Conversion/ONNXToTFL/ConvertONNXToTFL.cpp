/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright 2026 FlashUnlimited2019.

//===----- ConvertONNXToTFL.cpp - ONNX to TensorFlow Lite lowering -------===//

#include "src/Conversion/ONNXToTFL/ONNXToTFLCommon.hpp"

#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
             ONNXGeluOp, ONNXLogOp, ONNXPReluOp, ONNXSeluOp, ONNXSoftplusOp,
             ONNXSoftsignOp, ONNXMeanOp, ONNXAndOp, ONNXOrOp, ONNXXorOp,
             ONNXCumSumOp, ONNXOneHotOp, ONNXLpNormalizationOp, ONNXCeluOp,
             ONNXThresholdedReluOp, ONNXAsinOp, ONNXAcosOp, ONNXAtanOp,
             ONNXAtanhOp, ONNXSinhOp, ONNXCoshOp, ONNXAsinhOp, ONNXAcoshOp,
             ONNXErfOp, ONNXIsInfOp, ONNXIsNaNOp, ONNXShrinkOp, ONNXIm2ColOp,
             ONNXMeanVarianceNormalizationOp, ONNXMishOp, ONNXDFTOp, ONNXDetOp,
             ONNXNegativeLogLikelihoodLossOp, ONNXMatMulOp, ONNXGemmOp,
             ONNXTopKOp, ONNXSTFTOp, ONNXReshapeOp, ONNXTransposeOp,
             ONNXConcatOp, ONNXExpandOp, ONNXSoftmaxOp, ONNXConvOp,
             ONNXMaxPoolSingleOutOp, ONNXReduceMeanOp, ONNXReduceMeanV13Op,
             ONNXReduceMaxOp, ONNXReduceMaxV13Op, ONNXReduceMinV13Op,
             ONNXReduceSumOp, ONNXReduceSumV11Op, ONNXReduceL1Op,
             ONNXReduceLogSumExpOp, ONNXReduceProdOp, ONNXReduceSumSquareOp,
             ONNXSliceOp, ONNXSplitOp, ONNXResizeOp, ONNXPadOp, ONNXFlattenOp,
             ONNXGatherOp, ONNXGatherElementsOp, ONNXScatterElementsOp,
             ONNXScatterNDOp, ONNXSqueezeOp, ONNXUnsqueezeOp, ONNXTileOp,
             ONNXWhereOp, ONNXGatherNDOp, ONNXClipOp, ONNXUpsampleAndPadOp,
             ONNXDepthToSpaceOp, ONNXSpaceToDepthOp, ONNXMinOp, ONNXMaxOp,
             ONNXLRNOp, ONNXLSTMOp, ONNXRNNOp, ONNXGRUOp, ONNXReverseSequenceOp,
             ONNXGridSampleOp, ONNXMaxPoolOp, ONNXMaxUnpoolOp, ONNXCompressOp,
             ONNXTriluOp, ONNXCenterCropPadOp, ONNXEyeLikeOp, ONNXCol2ImOp,
             ONNXAffineGridOp, ONNXDeformConvOp, ONNXRoiAlignOp, ONNXLpPoolOp,
             ONNXGlobalLpPoolOp, ONNXBatchNormalizationInferenceModeOp,
             ONNXAttentionOp>(op) ||
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
    inputs.push_back(convertRank4NCHWToNHWCType(type));
  for (Type type : oldType.getResults())
    results.push_back(convertRank4NCHWToNHWCType(type));

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
        inputType != convertRank4NCHWToNHWCType(outputType) &&
        outputType != convertRank4NCHWToNHWCType(inputType)) {
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
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return convertRank4NCHWToNHWCType(type);
    });

    if (failed(prepareEntryPoint(module))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet preprocessingPatterns(&getContext());
    populateONNXToTFLPreprocessingPatterns(preprocessingPatterns);
    if (failed(
            applyPatternsGreedily(module, std::move(preprocessingPatterns)))) {
      module.emitError("ONNXToTFL preprocessing failed");
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
