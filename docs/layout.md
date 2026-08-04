# Layout policy

The layout contract is rank-sensitive:

- Exactly rank-4 activations use NHWC inside the TFLite model.
- Rank-1, rank-2, rank-3, and rank-5-or-higher activations retain the ONNX axis
  order. In particular, rank-3 and rank-5 tensors are never implicitly treated
  as spatial NHWC tensors.
- The MLP MVP contains no layout-sensitive operation, so its ranks and axes are
  unchanged.

For the convolution phase, an ONNX rank-4 graph boundary in NCHW is transposed
once to NHWC. The internal convolution/pooling region remains NHWC, and an
output declared by ONNX as NCHW is transposed back at the graph boundary. The
implementation must propagate this layout state per SSA value so adjacent ops
do not independently insert canceling transposes.

ONNX Conv weights use `[M, C/group, kH, kW]` (OIHW). TFL `conv_2d` filters use
`[M, kH, kW, C]` (OHWI), so constant weights are transposed `[0, 2, 3, 1]` at
compile time. Bias remains a rank-1 `[M]` tensor and is not layout-transformed.

ONNX strides and dilations `[H, W]` map to TFL `stride_h`, `stride_w`,
`dilation_h_factor`, and `dilation_w_factor`. `SAME_UPPER` can map to TFL
`SAME` when the inferred static shape agrees. `VALID` maps directly. Explicit
static symmetric padding can use `VALID` after a TFL pad operation; asymmetric
padding requires the same explicit treatment. `SAME_LOWER` is outside the MVP.

The planned convolution MVP restrictions are 2D only, `group == 1`, dilation
`[1, 1]`, and static padding. Depthwise and grouped convolution are not silently
mapped to ordinary Conv2D. A future depthwise implementation must require
`group == input_channels`, validate the channel multiplier, and materialize the
TFL depthwise filter layout `[1, kH, kW, input_channels * multiplier]`.

Transpose cleanup uses layout state and canonicalization: boundary/layout
transposes are tagged by their data-flow role, propagated through layout-neutral
elementwise operations, and inverse adjacent pairs are removed. Arbitrary ONNX
`Transpose` remains a semantic operation and is not removed merely because its
rank is four.

Conv/Pool lowering is not enabled in the first-stage implementation; requests
currently fail with an explicit unsupported-operation diagnostic. This document
defines the required second-stage behavior rather than claiming it is complete.
The driver passes `--disable-conv-to-matmul` to the ONNX importer so `onnx.Conv`
remains available to that future layout-aware lowering instead of being erased
early into the target-independent `onnx.Im2Col`/Gemm decomposition.
