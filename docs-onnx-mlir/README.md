<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upstream ONNX-MLIR Reference Documentation

This directory preserves documentation, examples, documentation checks, and
website resources inherited from ONNX-MLIR at upstream base commit
`7cea64f8aa2fbc56f859df768211a95dc2c72ad6`. Local paths and documentation build
rules have been adjusted for this repository's directory structure.

For ONNX-to-TFLite conversion, start with the
[project documentation](../docs/README.md). The build guides, operator support
tables, runtime interfaces, and accelerator documentation here describe the
upstream compiler; they do not define the capabilities of the TFLite pipeline.

## Reference entry points

- [Document index](DocumentList.md)
- [ONNX dialect](Dialects/onnx.md)
- [Operator definition workflow](ImportONNXDefs.md)
- [Lowering infrastructure](LoweringCode.md)
- [Testing](Testing.md)
- [Documentation maintenance](Documentation.md)

The CMake targets `check-doc`, `check-doc-example`, and `check-docs` retain their
names. Generated dialect documentation and upstream operator support tables
are written into this directory, not into `docs/`.
Generated-definition checks require a Python ONNX package matching the upstream
revision; see [TestingHighLevel.md](TestingHighLevel.md). Source checks exclude
dependency and build trees, and generated files are checked separately in the
active build directory.

The upstream website resources are retained for reference. Their presence
does not configure or publish a documentation site for this project.

Original licensing and attribution notices are preserved. See
[LICENSE](../LICENSE).
