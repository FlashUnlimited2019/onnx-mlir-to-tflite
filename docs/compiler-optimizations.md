<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compiler Optimizations

The ONNX-to-TFLite pipeline applies several semantics-preserving static graph
transformations in addition to ordinary operator legalization. Their primary
goals are to reduce unnecessary layout conversions, keep matrix operations at
backend-friendly ranks, and replace expensive general tensor updates with
simpler builtin operations when equivalence can be proven at compile time.

These transformations are separate from the optional TensorFlow/LiteRT TFL
optimizer. They are implemented on the ONNX-MLIR side and therefore remain
active when `--no-optimize-tfl` is used.

## Optimization stages

The relevant stages execute in this order:

```text
ONNX Dialect MLIR
    |
    |  ONNX preprocessing and structural rewrites
    v
Rewritten ONNX Dialect MLIR
    |
    |  per-value layout analysis
    |  layout encodings applied only in experimental mode
    v
Layout-annotated ONNX Dialect MLIR
    |
    |  full ONNX-to-TFL dialect conversion
    v
TFL Dialect MLIR
    |
    |  targeted TFL structural rewrites
    v
Locally optimized TFL Dialect MLIR
    |
    |  optional TensorFlow/LiteRT litert-opt pipeline
    v
Final TFL Dialect MLIR
```

The first four transformations occur inside
`onnx-mlir-opt --convert-onnx-to-tfl`. The final `litert-opt` stage is enabled
by default in the driver and can be disabled independently.

## ONNX preprocessing

Preprocessing runs before dialect conversion, while ONNX operation semantics
and logical axes are still explicit.

### Group normalization rank reduction

Static FP32 `GroupNormalization` with per-channel affine parameters reshapes
each instance/group into a row of a rank-two tensor. Mean and variance are
computed within each row; the normalized values are then reshaped to rank
three for per-channel scale/bias before restoring the output shape. This
avoids introducing rank-six intermediates for rank-five inputs. The rewrite
runs during importer canonicalization and is also registered for direct
ONNX-to-TFL conversion. It accepts ranks 2-5, positive groups dividing the
channel count, `stash_type=1`, and matching constant or runtime parameters.

### Static tensor-update recognition

General scatter lowering can require coordinate construction, gathers, sparse
updates, and several layout conversions. Statically provable update forms
are replaced with `Slice` and `Concat`:

- `Transpose -> ScatterND -> inverse Transpose` is recognized when constant
  indices describe one contiguous interval along a single logical axis.
- `ScatterElements` is recognized as a slice update when constant indices
  select the same contiguous interval for every other fully covered axis.
- `ScatterND` is recognized when its constant coordinate tuples enumerate a
  single-axis slab in row-major order, including complete trailing slices.
  Full coverage becomes the updates tensor, with a reshape if needed.

The rewrite requires static shapes, FP32 data, `reduction="none"`, compatible
update shapes, valid constant indices, and exact permutation relationships
where transposes are involved. Negative coordinates are normalized before
checking every tuple; holes, duplicates, reordered tuples, and partial coverage
on multiple axes do not match. Irregular indices and reduction forms retain
the general scatter lowering. A Slice contained in a single Concat operand
bypasses the other operands without changing their other consumers.

The rewrite uses source overwrite semantics, not the floating-point identity
`x + (u - x) = u`, which is unsafe under rounding and exceptional values.
Direct copies preserve NaNs, infinities, and signed zeros.

### Static coordinate selection

Unbatched FP32 `GatherND` with the same proven slab traversal becomes a Slice
and, if necessary, a Reshape. At the TFL stage, valid constant multidimensional
coordinates can also be linearized into rank-one indices for a rank-two
`Gather`, with reshapes restoring the original output shape. Duplicate and
irregular gather coordinates retain their ordering. Every coordinate is
checked independently before linearization; dynamic indices, oversized
inspection work, and unsupported shapes retain the existing lowering.

Rank-two FP32/i32/i64 `GatherElements` can also eliminate a constant index table when
every row selects a contiguous window. Fixed windows become a Slice. Repeating
windows whose starting offset decreases by a constant step become two Reshapes
and two Slices: the flattened row stride is reduced by that step. The rewrite
checks every normalized index, complete row-block coverage, bounds, and i32
shape limits, and limits the copied intermediate to twice the output size.
It does not change floating-point arithmetic or add runtime index
generation. Irregular, runtime, higher-rank, and partially covered cases retain
the existing gather paths. This can trade a few inexpensive layout operations
for much smaller constants; runtime performance still requires measurement.

For runtime `GatherElements`, an explicit index broadcast that repeats the
same index across each complete trailing slice becomes a row `Gather`.
Indices shared across all batch prefixes use `batch_dims=0`; fully specified
per-prefix indices use a rank-three view and `batch_dims=1`. Normalization of
negative indices happens on the compact indices, retaining their integer
width. Partial slices, partially broadcast batch prefixes, and channel-varying
indices retain the general lowering. Shared index broadcasts remain available
to other consumers.

Constant-index rules only consume constants available at their respective stages.
Coordinates that become constant later in TensorFlow/LiteRT optimization are
not automatically revisited by the ONNX-MLIR-side rewrites.

### Statically padded suppression

A bounded `NonMaxSuppression` consumer can be replaced only when its complete
indexing and scatter graph proves a fixed number of slots per batch and class.
The rewrite preserves untouched padding, score thresholds, strict IoU
comparison, and deterministic index tie-breaking. Independently observable
variable-length results prevent this rewrite.

Coordinates are extracted directly, small pairwise overlap tables are shared
across classes, and masked scores are updated between unrolled selections.
Self-removal is explicit even for degenerate boxes. Quadratic workspace is
bounded; larger inputs retain linear-workspace selection. The result uses
static builtin operations of rank at most three, without runtime loops or a
substitution of native NMS kernels with different boundary behavior.

### Transformer projection packing

Compatible projection branches are packed to share work and reduce repeated
view operations:

- A combined projection followed by `Split` and three compatible
  `Add`/`Reshape`/`Transpose` branches is rearranged to perform the compatible
  bias, reshape, and transpose work once before splitting. A branch-local
  scalar scale remains branch-local.
- Three compatible `MatMul` projections with constant weights are combined by
  concatenating the weights at compile time, performing one `MatMul`, and
  splitting the packed head representation afterward.

Both forms require a static rank-3 FP32 input, exactly three compatible
branches, consistent head width and batch/sequence dimensions, and the
expected single-use projection chains.

### Shared projections across a batch partition

Scalar gathers that cover a static batch axis exactly once can share one
constant-weight MatMul and bias addition before the gathers. Identical,
single-use LeakyReLU branches can share their activation as well. This avoids
repeated projection invocations without evaluating additional rows. The rule
requires static FP32 rank-3/4 tensors, a batch axis preceding the matrix axes,
identical weights and bias, and an exhaustive, unique partition of at most 64
indices. Incomplete selections, duplicate indices, different parameters, or
shared intermediate results retain their existing computation.

### Split bias placement

Constant biases following a last-axis split can be concatenated at compile
time and applied once before the split. This rule is independent of subsequent
head reshapes or permutations and supports unequal split widths, scalar biases,
and channel vectors with leading singleton dimensions. Each split result must
have only its bias addition as a user; biases that vary along other dimensions
are left unchanged. Dense bias materialization is bounded to avoid excessive
compile-time memory use.

The resulting `MatMul -> Add -> Split` form exposes projection bias fusion
and compatible preceding affine transforms to the optional TFL optimizer.

### Tanh-GELU recognition

Expanded tanh-GELU expressions are recognized before a generic optimizer can
absorb their final scale into downstream weights. The matcher handles both
`(0.5 * x) * factor` and `0.5 * (x * factor)`, commuted operands, and cubic
terms expressed with `Pow` or repeated multiplication. Constants, intermediate
shapes, broadcasting, and internal single-use requirements are checked.

The replacement preserves the existing tanh approximation. It does not replace
exact erf-GELU with an approximation or accept nearby coefficients. Fused and
expanded floating-point kernels need not be bitwise identical, so numerical
regression uses explicit error tolerances.

### Static sequence canonicalization

A statically shaped `SplitToSequence -> SequenceAt` chain with constant split
sizes and position can be replaced with an ordinary tensor `Split`. Scalar or
vector split specifications, negative constant positions, and `keepdims`
semantics are validated before rewriting. This removes sequence values from
graphs whose selected element is known at compile time.

### High-rank static simplification

Existing preprocessing also handles selected shape-proven high-rank forms:

- a static prefix slice of a concatenation retains only contributing inputs;
- singleton `Unsqueeze`/`Expand`/`Reshape` chains are rebuilt at a lower rank;
- a recognized static convex-upsample composition is rebuilt from lower-rank
  gather, matrix, reduction, and depth-to-space operations.

These rewrites are shape-based and do not depend on exported node names.

## Per-value layout optimization

The default `legacy` mode preserves the established rank-based policy: rank-4
FP32 activations are physically channels-last, while other ranks normally
retain their ONNX order. The `experimental` mode enables conservative
per-value layout propagation so non-convolutional regions do not have to
inherit an image layout merely because a tensor has rank four. ONNX
preprocessing, matrix-rank optimization, and TFL structural rewrites run in
both modes; only the per-value layout encodings require `experimental`.

### Analysis model

The analysis records a physical-to-logical dimension permutation for each
static ranked value. It uses:

- channels-last anchors for convolution and pooling regions;
- identity-layout anchors for `MatMul` and `Gemm`;
- exact propagation through `Transpose` and singleton-only reshape changes;
- same-layout propagation through compatible elementwise operations;
- axis-aware propagation through slice, concatenate, gather, scatter,
  softmax, and reduction operations.

Unconstrained connected components receive an identity basis and are then
propagated to a fixed point. Conflicting requirements are recorded rather than
silently choosing one layout.

### Conservative activation

Layout information is attached to tensor types only when the value's defining
operation and every consumer have migrated layout-aware lowering. Equal-layout
operation groups must be complete and consistent; otherwise their encodings
are removed and the established lowering path is used.

Current layout-aware consumers include:

| Category | Operations |
| --- | --- |
| Elementwise | Common unary and binary FP32 operations, including activation functions |
| Matrix | `MatMul`, `Gemm` |
| Convolutional | Rank-4 `Conv`, `AveragePool`, and single-output `MaxPool` |
| Shape and view | `Reshape`, `Squeeze`, `Unsqueeze`, `Transpose` |
| Axis-sensitive tensor | `Concat`, `Slice`, `Softmax`, `Gather`, `GatherElements`, `GatherND`, `ScatterElements`, `ScatterND` |
| Reduction | `ReduceMean`, `ReduceMax`, `ReduceSum` |

Lowering maps logical ONNX axes to physical TFL axes. Singleton-only reshape,
squeeze, and unsqueeze operations remain zero-copy whenever the inferred input
and output layouts agree. A logical transpose whose effect is already carried
by the type encoding is removed. Operations that require a fixed physical
layout receive explicit boundary conversions only where needed.

For rank-four FP32 arithmetic and Min/Max, lower-rank operands broadcast along
the logical ONNX suffix axes even when either operand has a permuted physical
layout. Lowering orders the existing operand axes and inserts only the needed
singleton dimensions. Reordering singleton axes requires only a reshape;
non-singleton axis changes use a transpose. This handles permutations that
move the logical batch axis, without expanding data or introducing rank-five
intermediates, and retains the established legacy broadcast lowering.

All experimental graph inputs and outputs retain their logical ONNX axis
order, including rank four and values lowered through the legacy fallback.
Boundary conversions are materialized when a non-identity permutation reaches
an entry argument or returned value; internal optimized layouts are retained.
Rank-4 FP32 public types carry explicit identity encodings so the implicit
NHWC fallback cannot change the interface. Checks compare axis permutations,
not just shapes, so equal-sized axes cannot hide data reordering. Direct
input/output aliases and signature port ordering are preserved. Legacy mode
continues to expose rank-4 FP32 boundaries as NHWC.

### Diagnostics and selection

Use the stable layout mode by default:

```bash
./build/bin/onnx-to-tflite model.onnx -o model.tflite \
  --layout-mode legacy
```

Enable per-value propagation and print an aggregate analysis summary with:

```bash
./build/bin/onnx-to-tflite model.onnx -o model.tflite \
  --layout-mode experimental \
  --layout-diagnostics summary
```

`--layout-diagnostics values` also prints the inferred layout for each value
and is intended for compiler debugging. Diagnostics are read-only and can be
used with either layout mode.

## ONNX-to-TFL lowering optimizations

These transformations are applied during dialect conversion, after ONNX
preprocessing and layout analysis have completed.

### Constant embedding table selection

An axis-zero `Gather` from an immutable FP32 matrix `[V,D]` can use the
builtin `EmbeddingLookup` when `D >= 128`, its indices are runtime i32/i64
tensors, and its static shapes fit the runtime's integer limits. Negative
axes are normalized before matching. The width threshold is a conservative
operator-selection policy, not a claim that all wide tables are learned
embeddings. Narrow tables, runtime data, other axes and data types, constant
indices, and empty or oversized shapes retain their existing lowering.

Indices are flattened to rank one and converted to i32. Before narrowing i64,
values outside `[-V,V)` are mapped to out-of-range sentinels so truncation
cannot turn an invalid lookup into a valid row. Negative-index normalization
also avoids overflow in unselected arithmetic branches. The result is reshaped
and layout-converted as needed, preserving both layout modes' public contracts.
If an integer maximum with a nonnegative constant proves indices nonnegative,
the redundant negative-index normalization is omitted; upper-bound protection
before i64 narrowing is retained.
The original weight buffer is reused; matching does not inspect or copy its
elements. Existing constant folding and static indexing rewrites run first.

This selects a builtin with hybrid quantized-table support; it does not
quantize weights. Generated tables remain FP32. Subsequent weight quantization
requires compatible tooling and separate numerical validation. Additional
index casts or reshapes can increase operator count, so latency should be
measured on the intended runtime rather than inferred from graph size.

### Convolution rank reduction

Convolution lowering selects the lowest exact implementation rank that can be
proved from static shapes and attributes. This lets several common volumetric
forms use the more widely supported TFL Conv2D operator.

| Source form | TFL implementation | Required conditions |
| --- | --- | --- |
| Conv1D | Conv2D with one singleton spatial dimension | Static rank-3 FP32 tensors and supported padding, stride, dilation, and grouping |
| Conv3D with a removable spatial axis | Conv2D after folding that axis into the batch dimension | The selected kernel extent and stride are one, the axis has no effective padding, and its input and output extents are equal |
| Full-depth Conv3D | Conv2D after merging input depth and channels | Group count is one, depth dilation is one, the depth kernel covers the complete input depth without depth padding, and the result has one depth position |
| Nonoverlapping tiled-depth Conv3D | Batched Conv2D after folding output-depth windows into batch and kernel depth into channels | Group count is one, depth dilation is one, depth stride equals the depth kernel, there is no depth padding, and the windows tile the input depth exactly |
| Other Conv3D forms | Native Conv3D | Correctness-preserving fallback when no rank-reduction proof applies |

The removable-axis rule applies to any of the three spatial axes. This includes
`(1, h, w)` kernels and more general forms with a singleton middle or trailing
axis. A `(d, 1, 1)` kernel becomes a Conv2D with only one non-singleton spatial
dimension, which is effectively a Conv1D computation.

The tiled-depth path can bypass a compatible rank-4 depth-major input chain
and pack it directly with rank-4 transpose and reshape operations, avoiding a
rank-5 or rank-6 activation before Conv2D. A following spatial-flatten and
token-transpose chain is also folded into one metadata reshape when its shapes
and permutations match exactly. Constant filter layout changes are folded
into the stored weights by the optional TFL optimizer.

Depthwise and grouped Conv1D or removable-axis Conv3D forms use the
corresponding Conv2D lowering. Non-depthwise grouped Conv2D is emitted as one
standard builtin operation: its group count is inferred from input and filter
channel dimensions. This avoids per-group splits, kernel invocations, and
concatenation, and requires grouped-Conv2D support in the downstream backend.
Non-reducible grouped Conv3D still uses split/convolve/concatenate around
native Conv3D operations. Failed
shape, grouping, padding, or output-extent checks simply disable rank
reduction.

FP32 depthwise convolutions with more than 4,832 output channels use a single
builtin Conv2D instead of DepthwiseConv2D. This avoids a zero-step loop in
affected native FP32 depthwise runtime kernels whose fixed accumulator holds
4,832 values; it is a runtime compatibility policy, not a TFLite format limit.
The group count is inferred from the input and OHWI filter shapes (or remains
one for a single input channel). The policy applies to Conv1D, Conv2D, and
removable-axis Conv3D lowering in both layout modes, including depth multipliers
greater than one. For constant filters, the default TFL optimization stage
folds filter reordering. Selecting this path introduces no per-group splitting,
concatenation, or additional activation rank.
Smaller depthwise convolutions retain their existing lowering. As with other
grouped convolutions, the target backend must support implicit Conv2D groups;
performance depends on its kernels and should be measured on that backend.

### Matrix and batch-rank optimization

Matrix lowering applies two conservative transformations:

- Rank-5 and higher `MatMul` inputs are collapsed to rank-3 `BatchMatMul` when every
  leading batch dimension is positive, static, and identical across both
  operands and the result. The result is reshaped back afterward. Rank-5
  broadcasted batch dimensions retain the general form; higher-rank broadcasts
  are rejected when they cannot be reduced safely. Rank-4 inputs remain native
  because flattening them can be slower on capable backends.
- A constant transpose chain that only swaps the final two matrix dimensions
  can be absorbed into the `adj_x` or `adj_y` attribute of `BatchMatMul`.
  Compatible batch-flattening reshapes are rebuilt directly at rank three.

Both transformations validate matrix dimensions, result shapes, permutation
composition, integer bounds, and broadcasting behavior before changing the
graph.

### Static GRU projection packing

Static FP32 GRUs can precompute input projections across all time steps by
flattening time and batch into matrix rows. Update/reset gate projections and
their sigmoid are combined, while the candidate gate remains separate to
preserve both `linear_before_reset` equations. Constant row slices and an equal
`Split` expose the results without introducing variable-split boundaries for
each gate. Execution remains statically unrolled with ordinary TFL builtins;
no recurrent, control-flow, or custom operator is introduced.

The rewrite preserves direction, initial states, bias placement, clipping,
and output ordering. Sequences shorter than three steps and projections above
a 4 MiB aggregate FP32 storage budget retain the previous lowering. The budget
limits the additional full-sequence projections, not total runtime memory.
Numerical tests cover both layouts, both reset forms, all directions, and
optional TFL optimization and runtime delegation. Floating-point matrix kernels
may produce small rounding differences; speedups remain backend-dependent.

### Equal-split selection

Static splits with equal, positive output widths use the TFL `Split` builtin.
Unequal or empty partitions retain `SplitV`. Axis mapping follows the selected
physical layout. This allows backends supporting equal splits to retain larger
accelerated subgraphs without requiring variable-split support; performance
still depends on the runtime and target hardware.

Single-part LSTM input or state splits are bypassed when their sole output
has exactly the input type and covers the entire split axis.

## TFL structural rewrites

After full ONNX legalization, targeted rewrites simplify structures that are
clearer in TFL form:

| Pattern | Transformation | Required proof |
| --- | --- | --- |
| Softmax layout round trip | Eliminate inverse transpose chains around `Softmax`, including one intervening scalar `Mul` | Composed permutations are inverse, the normalized final axis is unchanged, and any scale has one FP32 element without fused activation |
| GELU layout round trip | `Transpose -> GELU -> inverse Transpose` becomes `GELU` | Static FP32 types, exact inverse permutations, and matching shapes |
| Attention head-layout chain | Multiple rank-4/rank-3 transpose and reshape views become one direct transpose | Exact head, sequence, width, flattened-batch shapes and composed permutations |
| Broadcast add motion | A constant broadcast `Add` is moved before a transpose | The constant can be transposed at compile time and broadcast shapes remain valid |
| Batch broadcast folding | `BroadcastTo` plus view operations feeding `BatchMatMul` becomes a singleton-batch operand | Leading source dimensions are singleton and the complete view chain preserves linear element order |
| Singleton transpose rank reduction | `Transpose -> Reshape` becomes a lower-rank `Reshape -> Transpose`, or just a `Reshape` | The reshape only removes unit axes, the transpose has one use, and remaining axis identities are preserved; the replacement transpose is rank four or lower |
| Concat view hoisting | Branch-local singleton reshapes, optionally followed by transposes that only move unit axes, become one reshape after a lower- or equal-rank `Concat` | At least two single-use branches, matching copy boundaries and source axes, static positive shapes, and no fused activation; no real dimension is split, merged, or reordered |
| Adjacent transpose composition | Compose two permutations into one transpose, or bypass inverse permutations | Valid permutations and matching physical types; a nonidentity composition requires a single-use intermediate |
| Pointwise layout motion | Move a transpose through FP32 pointwise arithmetic or activation and cancel matching input transposes | Static ranks at most four, right-aligned broadcasting remapped by axis identity, preserved operand order and activation attributes, no duplicated computation, fewer layout copies, and no increase in view count |
| Pointwise singleton views | Move a singleton-only reshape through pointwise computation and bypass matching operand reshapes | Broadcast axes and inferred output rank remain valid, no real dimensions are split or merged, ranks at most four, and a strict reduction in view count |
| Padding layout round trip | `Transpose -> Pad/PadV2 -> inverse Transpose` becomes reordered padding | Static FP32 ranks at most four, nonnegative constant paddings, exact output shape, preserved scalar fill value, and a single-use padding result |
| Masked layout motion | Move a transpose through `SelectV2`, bypass compatible index-free broadcasts, and cancel operand transposes | Static ranks at most four; every operand becomes a view without reordering non-unit axes; exact broadcast/output shape; a single-use selection result; shared operand producers remain intact |
| Singleton-axis selection | Scalar-index `Gather` becomes a reshape | Constant index zero, selected extent one, `batch_dims=0`, and exact physical output shape/type; unchecked runtime indices and vector indices do not match |

The broadcast-folding proof checks the exact linear index mapping for bounded
static tensors. Optional scalar scaling is retained on the unbroadcast source.

Masked selection stays a selection, not multiplication by a numeric mask, so
the rewrite preserves NaNs, infinities, and signed zeros without changing
arithmetic. A mask requiring a new physical transpose does not match.

### Bounded integer division

A bounded analysis of constants, integer widening casts, views, and Min/Max
can prove `0 <= a <= A < INT32_MAX` and `b >= 1`. In that case, int64 division
uses the exact identity `a / b = a / min(b, A + 1)`: operands fit int32,
integer division computes the quotient, and an int64 cast restores the result
type. No range is assumed for graph inputs. Unproven bounds, negative
dividends, or an unrepresentable cap retain the existing general int64
division path; division-by-zero and quotient-overflow domain restrictions
remain unchanged.

### Convolution channel-to-spatial shuffle

A static channel-to-width shuffle after an ungrouped FP32 Conv2D can move its
channel permutation into constant OHWI weights and bias. The matched sequence
`[N,1,L,C*S] -> [N,C*S,1,L] -> [N,C,S,L] -> [N,C,L,S] -> [N,C,1,L*S]`
then needs only a reshape and the required output-layout transpose; an
immediately following inverse transpose is canceled. Exact shapes and
permutations are checked, all replaced intermediate values must have one use,
and constant-filter copying is bounded. Runtime filters, grouped/depthwise
convolutions, shared intermediate results, and different shuffle orders are
left unchanged. The transformation preserves the convolution's attributes and
does not introduce higher-rank activations or custom operations.

### Static indices, slices, and masks

A separate post-lowering phase works on physical TFL tensor shapes in both
layout modes:

- **Constant index folding:** evaluate proven constant `i1`, `i32`, and `i64`
  indexing chains through views, tiling, concatenation, slices, broadcasting,
  integer addition, comparisons, casts, and selection. Floating-point
  arithmetic is not folded; signed addition overflow and out-of-range integer
  narrowing retain the original operations.
- **Proven nonnegative indices:** remove negative-index checks only when
  `ArgMax`/`ArgMin`, constants, views, concatenation, and safe integer operations
  prove a nonnegative range. Addition overflow, unsafe narrowing, unknown
  runtime values, and analysis-budget exhaustion retain the checks.
- **Concat/Slice simplification:** when a static slice lies entirely within one
  concatenated operand, slice that operand directly, or reuse it if the slice
  selects it in full. Cross-boundary slices and fused activations are retained;
  other consumers of the concatenation are preserved.
- **Static mask elimination:** trace the selected elements of a boolean mask,
  including constant regions following a runtime-dependent concatenation. A
  provably uniform `SelectV2` condition selects its branch directly only when
  that branch has exactly the output type and shape. Runtime-dependent masks
  and branches requiring broadcasting are retained.

Only whitelisted stateless operations are removed when unused. Evaluation has
explicit rank, recursion, element-count, and work limits; exceeding a limit
skips the optimization. Shape, axis, element-type, and broadcasting checks do
not depend on model names or observed runtime inputs.

## Safety and fallback behavior

Layout and indexing rewrites preserve element mappings. Arithmetic recognition
preserves the specified operator formula and approximation mode; it is not a
promise of bitwise equality between different floating-point kernels. No rule
depends on model names or profile-guided assumptions. Depending on the pattern,
matching requires:

- static ranked tensors and exact element types;
- constant permutations, shapes, axes, or indices;
- explicit shape equality and element-count checks;
- compatible broadcasting and reduction semantics;
- single-use intermediate values before destructive folding;
- overflow checks when dimensions are multiplied or narrowed.

If a proof fails, the pattern does not rewrite the graph. The operation then
uses its established lowering or causes normal full-conversion failure if the
configuration is unsupported.

Post-lowering matrix and transformer rewriting uses existing and newly created
`BatchMatMul` and `Transpose` roots. Subsequent local layout cleanup targets
`Transpose`, `Slice`, `Reshape`, `Gather`, and `Concat` roots. Static index rewriting uses
its own operation whitelist and evaluation budgets. Generic folding, constant
CSE, and region simplification are disabled in these local phases so they do
not become unbounded whole-module optimizers.

Regression coverage includes positive and negative structural matches,
batch-rank reduction with and without broadcasting, layout diagnostics,
layout-aware axis handling, static update recognition and fallback, complete
MLIR legalization, FlatBuffer round trips, and end-to-end numerical checks.
`test/e2e/run_transformer_rewrites.py` additionally checks split/bias and GELU
forms across ranks 1–5, both layouts, and both settings of the optional TFL
optimizer. Run it with `--help` for converter and work-directory options.
`test/e2e/run_static_index_optimizations.py` checks index folding, shared
Concat/Slice uses, and static-mask selection against NumPy across ranks 1–5,
both layouts, and both TFL optimizer settings, including overflow fallbacks and
bitwise preservation of floating-point exceptional values and signed zero.
`test/e2e/run_layout_cleanup.py` checks local transpose, slice, tile, and
singleton-view rewrites with an independent element-copy oracle, including
equal-sized axes, shared branches, both layouts, optional TFL optimization,
and runtime delegation.
`test/e2e/run_pointwise_layouts.py` checks pointwise broadcasting, scalar ranks,
unit-axis views, shared branches, repeated operands, fused activations, and
padding against NumPy in both layouts, with and without downstream optimization
and runtime delegation. Arithmetic checks use explicit tolerances; copy-only
fixtures also exercise exceptional floating-point values.
`test/e2e/run_index_layout_cleanup.py` covers compact broadcast row gathers,
masked layout motion, singleton gathers, and proven integer-division bounds
in both layouts, with TFL optimization and runtime delegation on/off. It
checks partial-coverage fallbacks, shared uses, invalid indices, full-width
integer boundaries, and bitwise preservation of copied floating-point data.
Integer boundary cases use an exact integer oracle alongside ONNX Runtime.
`test/e2e/run_static_padded_nms.py` checks padded suppression against ONNX Runtime
across box formats, batches, classes, selection limits, threshold boundaries,
ties, empty selections, degenerate boxes, both workspace paths, and rejected
near-matches.

See [Convert an ONNX model](convert-model.md) for the complete driver option
reference and [Architecture](architecture.md) for the compiler and toolchain
boundaries.
