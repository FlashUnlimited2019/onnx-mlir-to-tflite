# ONNX to TFL operator support

The table only calls opset 18 supported because that is the version generated
and exercised by this prototype. Other opsets may be accepted by onnx-mlir's
importer but are not claimed without tests.

| ONNX op | Status | Opset | Target TFL op | Attribute restrictions | MLIR test | E2E test |
|---|---|---:|---|---|---|---|
| Constant | MVP | 18 | `arith.constant` | dense f32/i32/i64 only | reshape | MLP, reshape, sweep |
| Identity | MVP | 18 | SSA forwarding | static f32 tensor | elementwise | sweep |
| Add | MVP | 18 | `tfl.add` | numpy-style TFL broadcast | add | add, MLP |
| Sub | MVP | 18 | `tfl.sub` | numpy-style TFL broadcast | elementwise | sweep |
| Mul | MVP | 18 | `tfl.mul` | numpy-style TFL broadcast | elementwise | sweep |
| Div | MVP | 18 | `tfl.div` | f32 only | elementwise | sweep |
| Relu | MVP | 18 | `tfl.relu` | none | relu | add, MLP |
| Sigmoid | MVP | 18 | `tfl.logistic` | none | elementwise | sweep |
| Tanh | MVP | 18 | `tfl.tanh` | none | elementwise | sweep |
| MatMul | MVP | 18 | `tfl.batch_matmul` | operand ranks 2-4 | matmul | MLP |
| Gemm | MVP | 18 | batch matmul + mul/add | static f32; `transA/B`, alpha, beta mapped | gemm | sweep |
| Reshape | MVP | 18 | `tfl.reshape` | static result; `allowzero=0` | reshape, diagnostics | reshape |
| Transpose | MVP | 18 | `tfl.transpose` | valid static permutation | other_ops, diagnostics | reshape |
| Concat | MVP | 18 | `tfl.concatenation` | valid static axis | other_ops | sweep |
| Softmax | MVP | 18 | `tfl.softmax` | last axis only; beta 1 | other_ops, diagnostics | MLP |
| Conv | Not implemented | - | planned `tfl.conv_2d` | see layout policy | diagnostic | - |
| MaxPool | Not implemented | - | planned `tfl.max_pool_2d` | see layout policy | - | - |
| AveragePool | Not implemented | - | planned `tfl.average_pool_2d` | see layout policy | - | - |
| GlobalAveragePool | Not implemented | - | planned reduction/pool | static spatial axes | - | - |
| Flatten | Not implemented | - | planned `tfl.reshape` | static shape | - | - |
| Pad | Not implemented | - | planned `tfl.pad`/`padv2` | static pads | - | - |
