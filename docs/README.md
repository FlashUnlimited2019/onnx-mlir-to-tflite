<!-- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-MLIR to TFLite Documentation

This directory documents the ONNX-to-TFLite compiler, its supported model
configurations, and its conversion workflow.

## Getting started

- [Quick Start](../README.md#quick-start): download and use a prebuilt compiler.
- [Model conversion](convert-model.md): prepare a model, choose a layout mode,
  convert it, and validate the result.
- [Build from source](build.md): build the compiler and its pinned dependencies.

## Technical reference

| Document | Contents |
| --- | --- |
| [Operator support](operator-support.md) | Supported operations, types, ranks, attributes, and test coverage |
| [Compiler optimizations](compiler-optimizations.md) | Implemented transformations and their applicability conditions |
| [Architecture](architecture.md) | Compilation stages, toolchain boundaries, and model export |
| [Layout policy](layout.md) | Public input/output contracts and internal layout handling |
| [Known limitations](known-limitations.md) | Unsupported configurations and implementation constraints |

## Upstream references

Inherited ONNX-MLIR documentation is maintained separately in
[docs-onnx-mlir/](../docs-onnx-mlir/README.md). Its build instructions and operator
support tables describe the upstream compiler, not this TFLite pipeline.

Project illustrations and branding are stored under `assets/diagrams/` and
`assets/logo/`, respectively. For contribution guidelines, see
[CONTRIBUTING.md](../CONTRIBUTING.md).
