<!--- SPDX-License-Identifier: Apache-2.0 -->

# Upstream Documentation Maintenance

This repository keeps inherited ONNX-MLIR material under `docs-onnx-mlir/`.
Documentation for the ONNX-to-TFLite pipeline belongs in [docs/](../docs/README.md)
instead; see [CONTRIBUTING.md](../CONTRIBUTING.md) for project contribution guidelines.

## Maintaining the upstream reference material

Keep reference pages and their resources in this directory, and update the
[document index](DocumentList.md) when adding a page. Preserve original licensing
and attribution notices.

The inherited Jekyll website configuration is retained in `_config.yml`,
`_layouts/`, and `_data/navigation.yml`. To build that reference site, use
`docs-onnx-mlir/` as the Jekyll source directory. Navigation URLs are relative
to the generated site, not to the repository root.

These files do not configure or publish a website for this project.
