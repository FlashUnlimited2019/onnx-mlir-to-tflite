<!-- SPDX-License-Identifier: Apache-2.0 -->

# Known limitations

- Only static ranked tensors are accepted. Dynamic dimensions and unranked
  tensors fail conversion.
- Runtime activations must be FP32. Shape constants may be i32 or i64.
- Quantization, calibration, sparse tensors, custom ops, and Select TF/Flex ops
  are unsupported.
- Ordinary FlatBuffer metadata has a 2 GB size ceiling. For a large ONNX model,
  whether its constants are inline or use an adjacent external-data file, the
  driver automatically enables TensorFlow's buffer-offset export at 1.5 GiB.
  All constant payloads are then
  appended after the metadata and addressed with 64-bit offsets in one
  self-contained `.tflite`; they are not limited to FullyConnected weights.
  Consumers must support TFLite buffer offsets. The external-data filename is
  currently auto-detected for the common `<model>.onnx.data`/`<model>.data`
  layouts; use `--use-buffer-offset` for other layouts.
- ONNX control flow, sequences, maps, optionals, RNN/GRU, and functions beyond
  the single graph entry point are unsupported. LSTM has a constrained static
  lowering: layout 0 FP32 forward/reverse/bidirectional graphs with default
  activations, constant W/R and optional constant B/sequence_lens, optional
  runtime initial states, and no peepholes or input-forget coupling. The pass
  fully unrolls the sequence and emits no While or TensorList operations, so
  graph size grows linearly with sequence length and direction count.
- MatMul operands are restricted to ranks 2-5, matching the pinned TFL
  BatchMatMul verifier. Rank-4 inputs/results use internal layout-restoring
  transposes so ONNX matrix axes are preserved. ONNX rank-1 MatMul promotion
  is not implemented.
- Softmax with standard per-axis semantics accepts any static axis, maps it
  through the rank-4 NHWC layout, and uses temporary transposes when the target
  axis is not the final physical dimension. For ONNX Softmax before opset 13,
  the older flattened semantics are only claimed when the logical axis is last.
- ArgMin and ArgMax support static FP32 or i32 input ranks 2-5 and i64 results. Positive and
  negative axes, both keepdims values, and first/last minimum/maximum selection are
  mapped. Last-index mode reverses the selected logical axis and maps the
  native first index back, while rank-4 FP32 input is restored to NCHW. Cast
  supports static bool/i32/i64 to FP32, FP32 to i32/i64, and i32/i64
  interchange with
  unchanged shape; other ONNX Cast type combinations are not implemented.
- Equal, Greater, and GreaterOrEqual support static FP32 operand ranks up to 5 and
  numpy-style broadcasting. Rank-4 operands are restored to logical NCHW around the
  comparison so lower-rank trailing dimensions retain ONNX meaning. Because
  the TFLite broadcast comparison kernel is limited to a 4D execution path,
  rank-5 operands are explicitly broadcast to the result shape, flattened,
  compared, and reshaped back. This may materialize a larger temporary tensor.
  Remaining comparison operators and non-FP32 comparisons are not implemented.
- Sign supports static FP32 tensors through rank 5. Mod supports numpy-style
  broadcasting through rank 5 for static FP32 `fmod=1` and int64 `fmod=0/1`.
  Other element types and dynamic shapes are not implemented. Rank-5 Mod
  materializes broadcasts and flattens around the TFL FloorMod/sign-correction
  sequence, which may increase temporary memory. Other integer widths and Cast
  directions beyond the explicitly listed combinations are unsupported.
- Binary FP32 elementwise operations accept 0D tensor constants and
  intermediates using TFLite scalar broadcasting. Sub preserves operand order;
  the importer may canonicalize constant-right subtraction to addition with a
  negated constant. For rank-4 results, rank-1/2/3 operands are adapted from
  logical NCHW trailing-axis alignment to physical NHWC before the builtin
  elementwise operation.
- SpaceToDepth accepts static rank-4 FP32 inputs when a positive block size
  divides both spatial dimensions and maps directly to the identically ordered
  TFLite builtin. DepthToSpace accepts static rank-4 FP32 DCR and CRD modes. DCR maps directly
  to the TFLite builtin; CRD reorders flattened channels from `[C,R²]` to
  `[R²,C]` using rank-3 reshape/transpose operations before the same builtin.
  A static convex-upsampling form is recognized only when its
  static rank-6/rank-7 Reshape/Transpose/Softmax/Gather/Mul/ReduceSum topology,
  axes, shapes, and constant gather indices exactly match. It is rewritten to
  rank-at-most-4 Softmax/Gather/BatchMatMul/DepthToSpace operations. This is
  not general support for arbitrary rank-6/rank-7 graphs. Static i64 Add/Mul,
  Less/Greater comparisons, and Unsqueeze are supported for generated sampling
  coordinate/index chains; other general integer elementwise combinations are
  not claimed.
- Einsum has no direct TFL lowering. The pinned onnx-mlir importer must
  decompose a static FP32 equation into supported MatMul, Transpose, ReduceSum,
  Where, Mul, Reshape, and Unsqueeze operations. The 22 equations in the
  rank-varied fixture are validated; arbitrary equations, ellipsis forms, and
  dynamic shapes are not claimed. Where requires static same-type FP32 or i64
  values and a static boolean condition/result broadcast of rank at most 5.
- Rank-4 Transpose and Concat are supported by layout-aware permutation/axis
  mapping. Tile accepts static FP32 rank-1 through rank-5 inputs and constant,
  positive repeats with one value per axis; rank-4 repeats are remapped from
  logical NCHW to physical NHWC. Dynamic or nonpositive repeats are rejected.
  Reshape supports static same-type FP32 or i64 data/results and
  rejects `allowzero=1`; only FP32 rank-4 crossings need explicit layout
  transposes. Flatten requires a static input, valid
  static axis, and rank-2 result; rank-4 Flatten restores logical NCHW order
  before reshaping.
- Gelu supports static FP32 tensors in both exact and tanh-approximation modes
  through the TFLite Gelu builtin. Abs maps static FP32 tensors directly to
  the TFLite builtin. Expand requires a constant broadcast shape, static
  same-type FP32 or i64 input/result ranks up to 5, and materializes
  `tfl.broadcast_to`; FP32 rank-4 crossings are performed in logical NCHW
  order.
- The four-dimensional TFLite graph boundary and internal activation layout is
  NHWC. Callers must transpose ONNX NCHW rank-4 inputs; all other ranks keep
  ONNX order.
- Conv is limited to static FP32 shapes. Conv1D is represented as Conv2D with
  one singleton spatial dimension. Positive Conv2D dilation is preserved in
  the native TFLite Conv2D attributes; `SAME_LOWER` remains unsupported.
  Non-depthwise grouped Conv1D/2D is decomposed into per-group Conv2D plus
  channel concatenation; true depthwise forms use `tfl.depthwise_conv_2d`.
  Conv3D first attempts an exact Conv2D reduction when a spatial kernel
  dimension is 1, that axis has stride 1, and its explicit begin/end padding
  is zero. A group-1 Conv3D is also reduced when the depth kernel covers the
  complete unpadded input depth and produces one depth output: channel and
  depth are merged into the Conv2D input-channel dimension. Otherwise it is
  emitted as builtin TFLite Conv3D using local
  NCDHW/NDHWC transposes. Grouped and depthwise Conv3D are decomposed into
  multiple ordinary Conv3D operations because the TFL Conv3D verifier requires
  full input-channel filters. Explicit padding and `SAME_LOWER` are
  materialized as rank-5 Pad plus VALID Conv3D. Dynamic shapes remain
  unsupported, and target runtimes must provide the TFLite `CONV_3D` builtin.
  ConvTranspose is supported through the importer's UpsampleAndPad plus Conv
  decomposition for static rank-3/4 FP32. A static rank-5 form is also
  accepted when one spatial axis is an unchanged singleton with stride 1 and
  zero padding; that axis is removed around zero insertion and restored for
  the following reducible Conv3D. Other rank-5 UpsampleAndPad forms remain
  unsupported. Positive spatial strides and matching nonnegative pads are
  required; downstream Conv restrictions still apply. Representative
  ConvTranspose2D, ConvTranspose1D, and degenerate ConvTranspose3D paths are
  validated end to end.
- MaxPool accepts static FP32 rank-3 through rank-5 tensors. Ordinary rank-4
  single-result forms use the native 2D builtin. Rank-3/rank-5 and rank-3/rank-4
  indexed forms use exact statically enumerated windows and currently require
  zero padding, unit dilation, floor mode, `storage_order=0`, and NOTSET
  auto-padding. Indexed pooling preserves ONNX flattened i64 first-maximum
  indices; matching static rank-3/rank-4 MaxUnpool lowers through ScatterND.
  AveragePool accepts static rank-3 through rank-5 FP32,
  positive kernels/strides, nonnegative explicit pads, dilation 1, and
  NOTSET/VALID auto padding. Ordinary rank-4 pooling uses the native builtin;
  compatible rank-4 include-pad cases use explicit zero padding followed by
  native VALID AveragePool. Other oversized kernels, explicit padding, ceil
  mode, include-pad semantics, and ranks 3/5 use exact statically enumerated
  windows, currently capped at 4096 output windows per operation. A window
  with no valid input elements is rejected. Standalone ONNX Pad
  supports static rank-1 through rank-5 FP32 tensors in constant, reflect, and
  edge modes with constant nonnegative pads and no `axes` input. Negative pads
  (cropping), runtime pads, partial-axis padding, and wrap mode are unsupported.
- LRN supports static rank-4 FP32 tensors and maps directly to the TFLite
  LocalResponseNormalization builtin. Its channel window size must be positive
  and odd. Legacy inference Dropout is eliminated; an unused FP32 mask output
  from malformed old models is treated as an omitted optional output.
- Slice requires compile-time starts and ends; axes and steps must be omitted
  or compile-time constants. Positive and negative steps are supported;
  negative axes are reversed first and then sliced with a positive stride.
  Zero steps are rejected. Inputs above rank 5 are accepted only when enough
  unchanged singleton axes can be removed before the TFLite Slice and restored
  afterward. Split
  requires static FP32 shapes and compile-time split sizes or output count.
  Resize supports static rank-4 spatial scaling with nearest/asymmetric/floor,
  nearest/half_pixel/round_prefer_floor, linear/half_pixel, or
  linear/align_corners semantics.
- ReduceMean supports static FP32 tensors with omitted, attribute, or constant
  axes, negative-axis normalization, both keepdims values, and
  noop-with-empty-axes. Rank-4 inputs are restored to logical NCHW for general
  reductions and rank-4 results are converted back to physical NHWC; the
  common spatial `[2,3]` case retains a compact direct path. ReduceSum supports
  static constant/omitted
  axes and restores logical dimension order around rank-4 reductions.
  ReduceMaxV13 supports static FP32 or i64 same-type input/results and static
  attribute axes; rank-4 restoration applies only to FP32.
  ReduceL2 is supported through the importer's Mul/ReduceSum/Sqrt
  decomposition for static FP32 ranks 1-5. It intentionally retains Mul and
  does not emit the TFLite Square builtin.
- GlobalAveragePool relies on importer decomposition to keepdims ReduceMean.
  Static FP32 rank-3 through rank-5 tensors are validated; axes 0 and 1 retain
  batch/channel while every spatial axis from 2 onward is reduced.
- BatchNormalization has no direct lowering pattern; the onnx-mlir importer
  must fold it into Conv constants or decompose it into channel Mul/Add. This
  is validated for Conv-adjacent opset-7/12 graphs and standalone static
  rank-3/4/5 opset-18 tensors with constant inference parameters.
- Gather supports static FP32 data/results through rank 6 and static i32/i64 constant or
  runtime indices. Runtime negative indices are normalized before TFLite
  Gather; other out-of-range runtime values remain an execution-time error.
- GatherElements requires static FP32 data/result and i32/i64 indices at ranks
  1-5. Constant indices become full GatherNd coordinates at compile time.
  Runtime indices are narrowed to i32, negative values are normalized, and
  static non-selected-axis coordinate components are concatenated with the
  runtime selected-axis component before GatherNd. Large index tensors can
  therefore increase FlatBuffer constant size; invalid runtime bounds remain
  an execution-time error. Rank-3/4/5 feature and weight cases, including
  runtime-index paths, are covered.
- ScatterElements is currently restricted to an identity-index form: static
  equal-shape FP32 tensors, constant indices that select every
  update's own coordinate, and `reduction=none`. It lowers exactly to the
  updates SSA value. General scatter coordinates, partial updates, runtime
  indices, and other reduction modes are not implemented.
- ScatterND accepts static FP32 data/updates/results through rank 5 and
  constant, unique i32/i64 index tuples with `reduction` equal to `none`,
  `add`, `mul`, `max`, or `min`. It implements the selected result as data plus
  a scattered `(replacement-old)` delta; negative indices are normalized and
  rank-4 values are handled in logical NCHW order. Runtime or duplicate
  indices are not implemented. Large constant coordinate tensors increase
  FlatBuffer size.
- The opset-20 uncommon-operator paths are intentionally static and
  configuration-constrained. AffineGrid requires constant theta/size;
  rank-5 GridSample requires a constant grid with nearest/border mode;
  DeformConv requires constant weights, offsets, bias, and mask with
  `group=offset_group=1`; RoiAlign requires constant ROIs/batch indices and a
  positive fixed sampling ratio. Col2Im currently accepts only zero-padded,
  unit-dilation, nonoverlapping blocks that exactly tile 1D/2D/3D output.
  DFT uses compile-time coefficient matrices, and Det uses unpivoted static
  elimination intended for small nonsingular matrices. These paths emit no
  dynamic control flow, but their generated constant/index tensors can grow
  quickly with larger shapes.
- GatherND likewise requires static FP32 data/result ranks 1-5 and static-shape
  i32/i64 indices. Constant and runtime negative indices are normalized;
  valid nonnegative `batch_dims` are represented by prefixing full i32 batch
  coordinates. Rank-5 runtime index normalization is flattened to rank 1 so it
  does not depend on TFLite's four-dimensional comparison broadcast helper.
  Scalar results, zero-sized dimensions, and data/result ranks
  above 5 are not implemented; large coordinate tensors can increase
  FlatBuffer size.
- InstanceNormalization is supported through an importer-produced
  LayerNormalization form: static rank-4 FP32, spatial
  axes, `[C,1,1]` scale/bias, `stash_type=1`, and unused statistics outputs.
  Static rank-2 through rank-5 last-axis LayerNormalization with rank-1
  scale/bias is also supported. Other axes/shapes and requested
  Mean/InvStdDev outputs are unsupported.
- RMSLayerNormalization supports static rank-2 through rank-5 FP32 last-axis
  normalization with rank-1 scale, optional rank-1 or full-rank broadcast
  bias, `stash_type=1`, and an unused InvStdDev output. The full-rank bias
  covers importer-recomposed residual Add patterns. The importer recomposition
  preserves scalar epsilon from either legacy `value_float` or dense `value`
  constants.
- The bridge is coupled to both pinned MLIR revisions and to TensorFlow's TFL
  optimizer/exporter ODS/schema. Updating either repository requires rerunning
  parser, optimization, FlatBuffer round-trip, interpreter-load, and numerical
  tests.
- The two-process design materializes constants in textual MLIR. In one
  representative large convolutional-network test, each textual
  MLIR/round-trip file is about 167 MiB; enabling
  `--keep-intermediate-files` now retains both pre- and post-optimization TFL
  IR, so preserved intermediates use additional disk. The FlatBuffer is about
  84 MiB because weights dominate its size.
  In large-model buffer-offset mode, temporary MLIR files are placed beside
  the output rather than in a potentially small `/tmp` tmpfs; several times
  the final model size may still be needed while conversion is running.
- Optimization is float-only and structural. This prototype does not run
  quantization, sparsification, or unsafe approximate algebraic rewrites.
