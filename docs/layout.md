<!-- SPDX-License-Identifier: Apache-2.0 -->

# Layout policy

The layout contract is rank-sensitive:

- Exactly rank-4 activations use NHWC inside the TFLite model.
- Rank-1, rank-2, rank-3, and rank-5-or-higher graph boundaries retain the ONNX
  axis order. Direct Conv3D fallback explicitly converts its local rank-5
  region from NCDHW to NDHWC and back; no rank-5 ABI conversion is implied.
- Rank-3 channel parameters such as BatchNorm's `[C,1,1]` constants are not
  activations. When broadcast with a rank-4 activation they are reshaped to
  `[1,1,C]`; their values are not reordered.

The TFLite graph signature itself uses NHWC for rank-4 inputs and outputs. No
runtime boundary Transpose is inserted: callers pass NHWC to TFLite while ONNX
Runtime receives the corresponding NCHW array. `test/e2e/run_similarity.py`
performs this boundary adaptation. The internal convolution/pooling region
stays NHWC. Non-rank-4 graph inputs and outputs keep their ONNX order.

ONNX Conv2D weights use `[M, C/group, kH, kW]` (OIHW). TFL `conv_2d` filters
use `[M, kH, kW, C/group]` (OHWI), so constant weights are transposed
`[0, 2, 3, 1]` at compile time. Bias remains a rank-1 `[M]` tensor and is not
layout-transformed.
For true depthwise Conv, the intermediate `[M,kH,kW,1]` filter is converted to
TFL's `[1,kH,kW,M]` depthwise layout; TensorFlow folds this constant transpose
during TFL optimization.

This compile-time transpose is performed even when the numeric shape is
unchanged, for example OIHW `[4,3,3,3]` and OHWI `[4,3,3,3]`.

ONNX strides and dilations `[H, W]` map to TFL `stride_h`, `stride_w`,
`dilation_h_factor`, and `dilation_w_factor`. `SAME_UPPER` can map to TFL
`SAME`; `VALID` maps directly. Every nonzero `auto_pad=NOTSET` Conv padding is
materialized as NHWC `tfl.pad` followed by `tfl.conv_2d` with `VALID`.
This is necessary because TFL `SAME` and ONNX explicit symmetric padding can
choose different top/left alignment when stride is greater than one. MaxPool
uses `tfl.padv2` with the lowest finite f32 value, then `VALID`, so padded cells
cannot win the maximum. Standalone rank-4 Pad maps ONNX's `[N,C,H,W]`
begin/end vector to an NHWC `[4,2]` matrix; other ranks keep their logical axis
order. Constant Pad uses `tfl.padv2`, reflect uses `tfl.mirror_pad`, and edge
padding is represented exactly with boundary Slice, Tile, and Concatenation.
`SAME_LOWER` remains outside the Conv1D/2D path. Direct Conv3D materializes its
static lower-biased padding and runs with `VALID`.

Conv1D starts in logical NCL order. It is transposed to NLC, reshaped to
`[N,L,1,C]`, evaluated by Conv2D, and transposed back to NCL. ONNX Conv1D
filters `[M,C/group,k]` become `[M,k,1,C/group]` (or the corresponding TFL
depthwise layout).

A Conv3D with any spatial axis satisfying `kernel=1`, `stride=1`, and zero
explicit padding on that axis is exactly reducible without a Conv3D runtime
operator. The lowering permutes NCDHW so that preserved axis is adjacent to
batch, folds it into batch, runs Conv2D over the other two axes, then restores
NCDHW. This includes the requested `(1,h,w)` and `(d,1,1)` kernels and also the
safe `(d,1,w)` and `(d,h,1)` families. The `(d,1,1)` case is semantically
Conv1D; because TFLite has no builtin Conv1D, it is represented as Conv2D with
a singleton kernel dimension. Rank-5 graph boundaries themselves remain
NCDHW.

Regular, grouped, and true depthwise convolution are supported by these rank
lowerings. Depthwise validates the channel multiplier and materializes the TFL
filter layout `[1,kH,kW,input_channels*multiplier]`. A non-depthwise grouped
Conv is decomposed into channel/filter/bias splits, one Conv2D per group, and a
channel concatenation.

If no exact Conv3D reduction applies, the pass explicitly transposes NCDHW to
NDHWC, converts ONNX filters from `[M,C/group,kD,kH,kW]` to TFL
`[kD,kH,kW,C/group,M]`, emits `tfl.conv_3d`, and transposes the result back to
NCDHW. TFL Conv3D has no group attribute, so grouped and depthwise cases split
input/filter/bias by group, emit one ordinary Conv3D per group, and concatenate
the NDHWC outputs. Thus future reduction rules can be inserted before this
fallback without changing graph ABI or correctness.

The conversion remaps rank-4 types once while lowering the graph rather than
inserting repeated runtime transposes. Rank-4 Concat and Slice axes, including
the full vectors for positive-step StridedSlice and the axes used to implement
negative steps with ReverseV2, are remapped. Transpose
permutations are translated from logical NCHW to physical
NHWC, and Resize consumes/produces NHWC directly. Reshape crossing rank 4 and
Softmax on a non-final physical axis use explicit internal transposes where
preserving ONNX row-major or axis semantics requires them. Rank-4 MatMul is
temporarily transposed from physical NHWC back to logical ONNX order around
BatchMatMul, then returned to NHWC if its result is rank 4. Rank-4 ReduceSum
and ReduceMax use the same logical-order restoration. Rank-4 Split maps its
logical axis to the corresponding physical axis. Rank-4 Flatten similarly transposes
physical NHWC back to logical NCHW before its rank-2 reshape, as required by
ONNX row-major order.
For binary elementwise operations with a rank-4 result, lower-rank ONNX
operands align with the logical C/H/W suffix rather than the physical H/W/C
suffix. Rank-1 `[W]` and rank-2 `[H,W]` operands are therefore reshaped to
`[1,W,1]` and `[H,W,1]`; a general rank-3 `[C,H,W]` operand is transposed to
`[H,W,C]`. This preserves numpy broadcasting without transposing the main
rank-4 activation away from NHWC.
Importer-generated UpsampleAndPad inserts zeros directly along physical NHWC
height and width, so decomposed ConvTranspose does not add graph-boundary
transposes.
The driver passes `--disable-conv-to-matmul` so `onnx.Conv` remains available
to this layout-aware lowering.
