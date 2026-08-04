# Layout policy

The layout contract is rank-sensitive:

- Exactly rank-4 activations use NHWC inside the TFLite model.
- Rank-1, rank-2, rank-3, and rank-5-or-higher activations retain the ONNX axis
  order. In particular, rank-3 and rank-5 tensors are never implicitly treated
  as spatial NHWC tensors.
- Rank-3 channel parameters such as BatchNorm's `[C,1,1]` constants are not
  activations. When broadcast with a rank-4 activation they are reshaped to
  `[1,1,C]`; their values are not reordered.

The TFLite graph signature itself uses NHWC for rank-4 inputs and outputs. No
runtime boundary Transpose is inserted: callers pass NHWC to TFLite while ONNX
Runtime receives the corresponding NCHW array. `test/e2e/run_similarity.py`
performs this boundary adaptation. The internal convolution/pooling region
stays NHWC. Non-rank-4 graph inputs and outputs keep their ONNX order.

ONNX Conv weights use `[M, C/group, kH, kW]` (OIHW). TFL `conv_2d` filters use
`[M, kH, kW, C]` (OHWI), so constant weights are transposed `[0, 2, 3, 1]` at
compile time. Bias remains a rank-1 `[M]` tensor and is not layout-transformed.

This compile-time transpose is performed even when the numeric shape is
unchanged, for example OIHW `[4,3,3,3]` and OHWI `[4,3,3,3]`.

ONNX strides and dilations `[H, W]` map to TFL `stride_h`, `stride_w`,
`dilation_h_factor`, and `dilation_w_factor`. `SAME_UPPER` can map to TFL
`SAME`; `VALID` maps directly. Every nonzero `auto_pad=NOTSET` Conv padding is
materialized as NHWC `tfl.pad` followed by `tfl.conv_2d` with `VALID`.
This is necessary because TFL `SAME` and ONNX explicit symmetric padding can
choose different top/left alignment when stride is greater than one. MaxPool
uses `tfl.padv2` with the lowest finite f32 value, then `VALID`, so padded cells
cannot win the maximum. `SAME_LOWER` is outside the MVP.

The implemented convolution MVP restrictions are 2D only, `group == 1`, dilation
`[1, 1]`, and static padding. Depthwise and grouped convolution are not silently
mapped to ordinary Conv2D. A future depthwise implementation must require
`group == input_channels`, validate the channel multiplier, and materialize the
TFL depthwise filter layout `[1, kH, kW, input_channels * multiplier]`.

The conversion uses a rank-4 TypeConverter mapping rather than inserting
repeated runtime transposes. Rank-4 Concat axes are remapped. Rank-4 MatMul,
Softmax, arbitrary Transpose, and general rank-4 Reshape are rejected until
their layout semantics are implemented; they are never silently lowered with
the wrong axis interpretation. The driver passes `--disable-conv-to-matmul` so
`onnx.Conv` remains available to this layout-aware lowering.
