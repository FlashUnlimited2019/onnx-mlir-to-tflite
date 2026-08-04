# ONNX to TFL operator support

The generated micro-models use opset 18. The supplied ResNet-34 uses opset 7;
only the configurations actually exercised by those models are claimed.

| ONNX op | Status | Opset | Target TFL op | Attribute restrictions | MLIR test | E2E test |
|---|---|---:|---|---|---|---|
| Constant | MVP | 7, 18 | `arith.constant` | dense f32/i32/i64 only; rank-4 f32 data transposed NCHW/OIHW→NHWC/OHWI | reshape, layout | MLP, reshape, sweep, ResNet |
| Identity | MVP | 18 | SSA forwarding | static f32 tensor | elementwise | sweep |
| Add | MVP | 7, 18 | `tfl.add` | TFL broadcast; rank-3 `[C,1,1]` parameters become `[1,1,C]` beside rank-4 activations | add, conv/pool/reduce | add, MLP, ResNet |
| Sub | MVP | 18 | `tfl.sub` | numpy-style TFL broadcast | elementwise | sweep |
| Mul | MVP | 7, 18 | `tfl.mul` | same channel-broadcast rule as Add | elementwise, conv/pool/reduce | sweep, ResNet |
| Div | MVP | 18 | `tfl.div` | f32 only | elementwise | sweep |
| Relu | MVP | 7, 18 | `tfl.relu`, then fused activation where legal | none | relu, layout | add, MLP, ResNet |
| Sigmoid | MVP | 18 | `tfl.logistic` | none | elementwise | sweep |
| Tanh | MVP | 18 | `tfl.tanh` | none | elementwise | sweep |
| MatMul | MVP | 18 | `tfl.batch_matmul`; constant rank-2 RHS optimized to `tfl.fully_connected` | operand ranks 2-3; rank 4 rejected under the layout conversion | matmul | MLP |
| Gemm | MVP | 7, 18 | batch matmul + mul/add; optimized to fused fully connected when legal | static f32; `transA/B`, alpha, beta mapped | gemm | sweep, ResNet |
| Reshape | MVP | 7, 18 | `tfl.reshape` | static result; `allowzero=0`; rank-4 crossing limited to `[N,C,1,1]`→non-rank-4 | reshape, diagnostics | reshape, ResNet |
| Transpose | MVP | 18 | `tfl.transpose` | valid static permutation; rank 4 rejected | other_ops, diagnostics | reshape |
| Concat | MVP | 18 | `tfl.concatenation` | valid static axis; rank-4 axis remapped | other_ops | sweep |
| Softmax | MVP | 18 | `tfl.softmax` | last axis only; beta 1; rank 4 rejected | other_ops, diagnostics | MLP |
| Conv | MVP | 7, 18 | `tfl.pad` + `tfl.conv_2d` | static 2D, f32, group=1, dilation=1; explicit nonnegative pads; absent bias becomes zero bias | conv/pool/reduce, diagnostics | simple Conv, ResNet |
| MaxPool | MVP | 7 | `tfl.padv2` + `tfl.max_pool_2d` | static 2D, dilation=1, ceil_mode=0, storage_order=0 | conv/pool/reduce | ResNet |
| BatchNormalization | Importer decomposition | 7 | folded Conv weights/bias or `tfl.mul` + `tfl.add` | inference-mode constants as in supplied ResNet | channel broadcast | ResNet |
| GlobalAveragePool | Importer decomposition | 7 | `tfl.mean` | static rank-4 spatial axes `[2,3]`, keepdims=1 | conv/pool/reduce | ResNet |
| ReduceMeanV13 | MVP internal form | 7 | `tfl.mean` | rank-4 requires axes `[2,3]`, keepdims=1 | conv/pool/reduce | ResNet |
| AveragePool | Not implemented | - | planned `tfl.average_pool_2d` | see layout policy | - | - |
| Flatten | Not implemented | - | planned `tfl.reshape` | static shape | - | - |
| Pad | Not implemented | - | planned `tfl.pad`/`padv2` | static pads | - | - |
