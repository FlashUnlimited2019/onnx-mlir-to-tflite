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

### Static tensor-update recognition

General scatter lowering can require coordinate construction, gathers, sparse
updates, and several layout conversions. Two statically provable update forms
are replaced with `Slice` and `Concat`:

- `Transpose -> ScatterND -> inverse Transpose` is recognized when constant
  indices describe one contiguous interval along a single logical axis.
- `ScatterElements` is recognized as a slice update when constant indices
  select the same contiguous interval for every other fully covered axis.

The rewrite requires static shapes, FP32 data, `reduction="none"`, compatible
update shapes, valid constant indices, and exact permutation relationships.
Irregular indices and reduction forms retain the general scatter lowering.

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

Non-rank-4 graph inputs and outputs retain the logical ONNX axis order in
both modes. Experimental layouts remain internal: boundary conversions are
materialized when a non-identity permutation reaches an entry argument or
returned value. This check uses axis permutations, not just tensor shapes,
so equal-sized axes cannot hide an unintended data reordering. Existing
rank-4 boundary layout policies are unchanged.

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
corresponding Conv2D lowering. Grouped forms are decomposed into split,
convolve, and concatenate operations when required. Non-reducible grouped
Conv3D uses the same decomposition around native Conv3D operations. Failed
shape, grouping, padding, or output-extent checks simply disable rank
reduction.

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

### Equal-split selection

Static splits with equal, positive output widths use the TFL `Split` builtin.
Unequal or empty partitions retain `SplitV`. Axis mapping follows the selected
physical layout. This allows backends supporting equal splits to retain larger
accelerated subgraphs without requiring variable-split support; performance
still depends on the runtime and target hardware.

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

The broadcast-folding proof checks the exact linear index mapping for bounded
static tensors. Optional scalar scaling is retained on the unbroadcast source.

### Static indices, slices, and masks

A separate post-lowering phase works on physical TFL tensor shapes in both
layout modes:

- **Constant index folding:** evaluate proven constant `i1`, `i32`, and `i64`
  indexing chains through views, tiling, concatenation, slices, broadcasting,
  integer addition, comparisons, casts, and selection. Floating-point
  arithmetic is not folded; signed addition overflow and out-of-range integer
  narrowing retain the original operations.
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

Post-lowering matrix and layout rewriting is restricted to existing and newly
created `BatchMatMul` and `Transpose` roots. Static index rewriting uses its own
operation whitelist and evaluation budgets. Generic folding, constant CSE, and
region simplification are disabled in these local phases so they do not become
unbounded whole-module optimizers.

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

See [Convert an ONNX model](convert-model.md) for the complete driver option
reference and [Architecture](architecture.md) for the compiler and toolchain
boundaries.
