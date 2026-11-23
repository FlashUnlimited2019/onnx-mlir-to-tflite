#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check grouped kernels, channel shuffles, and partitioned projections.

Uses the shared convolution regression driver: both layouts, TFL optimization
on/off, native/delegated execution, and three independent random input sets.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import run_parallel_grouped_conv as regression


def make_model():
    inputs, outputs, nodes, constants = [], [], [], []
    rng = np.random.default_rng(704)

    def value(name, shape, runtime=True):
        info = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        if runtime:
            inputs.append(info)
        return info

    def const(name, data):
        constants.append(numpy_helper.from_array(np.asarray(data), name))
        return name

    def floats(name, shape, runtime=False):
        if runtime:
            value(name, shape)
            return name
        return const(name, rng.normal(0, 0.15, shape).astype(np.float32))

    def reshape(x, name, shape):
        target = const(name + "_shape", np.array(shape, dtype=np.int64))
        nodes.append(helper.make_node("Reshape", [x, target], [name]))
        return name

    # Grouped 1D/2D, each removable 3D axis, true 3D fallback, depthwise,
    # runtime filters, asymmetric padding, strides, and dilation.
    configs = [
        ([11], [3], [1], [2], [2, 1], 2, 4, 6),
        ([9], [3], [2], [1], [1, 1], 3, 6, 9),
        ([7], [7], [1], [1], [0, 0], 256, 512, 512),
        ([7, 9], [3, 2], [2, 1], [1, 2], [1, 0, 1, 1], 2, 4, 6),
        ([3, 7, 9], [1, 3, 3], [1, 1, 2], [1, 2, 1], [0, 2, 1, 0, 1, 1], 2, 4, 6),
        ([7, 3, 9], [3, 1, 3], [2, 1, 1], [1, 1, 2], [1, 0, 2, 1, 0, 2], 3, 6, 9),
        ([7, 9, 3], [3, 3, 1], [1, 2, 1], [1, 1, 1], [1, 1, 0, 1, 1, 0], 2, 4, 6),
        ([4, 5, 6], [2, 2, 2], [1, 1, 1], [1, 1, 1], [0] * 6, 2, 4, 6),
        ([9], [3], [1], [1], [1, 1], 4, 4, 8),
    ]
    for i, (
        spatial,
        kernel,
        strides,
        dilations,
        pads,
        group,
        channels,
        filters,
    ) in enumerate(configs):
        for runtime in (False, True):
            p = f"group_{i}_{int(runtime)}"
            x = p + "_x"
            value(x, [2, channels, *spatial])
            w = floats(p + "_w", [filters, channels // group, *kernel], runtime)
            b = floats(p + "_b", [filters], runtime)
            y = p + "_y"
            nodes.append(
                helper.make_node(
                    "Conv",
                    [x, w, b],
                    [y],
                    group=group,
                    kernel_shape=kernel,
                    strides=strides,
                    dilations=dilations,
                    pads=pads,
                )
            )
            extent = [
                (
                    d
                    + pads[j]
                    + pads[j + len(spatial)]
                    - dilations[j] * (kernel[j] - 1)
                    - 1
                )
                // strides[j]
                + 1
                for j, d in enumerate(spatial)
            ]
            outputs.append(value(y, [2, filters, *extent], False))

    # Exact shuffle positives plus guarded fallback for grouped, runtime,
    # shared convolutions and a different middle permutation.
    for i, (n, c, s, group, runtime, shared, swap) in enumerate(
        [
            (1, 3, 2, 1, False, False, True),
            (2, 2, 3, 1, False, False, True),
            (2, 4, 2, 2, False, False, True),
            (2, 3, 2, 1, True, False, True),
            (2, 3, 2, 1, False, True, True),
            (2, 3, 2, 1, False, False, False),
        ]
    ):
        p = f"shuffle_{i}"
        x = p + "_x"
        value(x, [n, 4, 3, 7])
        w = floats(p + "_w", [c * s, 4 // group, 3, 3], runtime)
        b = floats(p + "_b", [c * s], runtime)
        conv = p + "_conv"
        nodes.append(helper.make_node("Conv", [x, w, b], [conv], group=group))
        if shared:
            outputs.append(value(conv, [n, c * s, 1, 5], False))
        first = reshape(conv, p + "_split", [n, c, s, 5])
        middle = p + "_transposed"
        nodes.append(
            helper.make_node(
                "Transpose",
                [first],
                [middle],
                perm=[0, 1, 3, 2] if swap else [0, 2, 1, 3],
            )
        )
        last = reshape(middle, p + "_out", [n, c, 1, 5 * s])
        outputs.append(value(last, [n, c, 1, 5 * s], False))

    # A complete batch partition can share projection and activation. Other
    # users, missing/duplicate indices, differing parameters, and non-batch
    # selection must preserve their original semantics.
    for i, (shape, axis, indices, variant) in enumerate(
        [
            ([3, 2, 4], 0, [0, 1, 2], "same"),
            ([2, 3, 2, 4], 1, [-1, 0, 1], "same"),
            ([2, 3, 2, 4], -3, [0, 1, 2], "same"),
            ([3, 2, 4], 0, [0, 2], "same"),
            ([3, 2, 4], 0, [0, 1, 1], "same"),
            ([3, 2, 4], 0, [0, 1, 2], "weight"),
            ([3, 2, 4], 0, [0, 1, 2], "bias"),
            ([3, 2, 4], 0, [0, 1, 2], "activation"),
            ([3, 2, 4], 0, [0, 1, 2], "shared"),
            ([2, 3, 4], 1, [0, 1, 2], "same"),
        ]
    ):
        p = f"project_{i}"
        x = p + "_x"
        value(x, shape)
        w = floats(p + "_w", [4, 5])
        b = floats(p + "_b", [5])
        for j, index in enumerate(indices):
            q = p + f"_{j}"
            idx = const(q + "_index", np.array(index, dtype=np.int64))
            gather = q + "_gather"
            nodes.append(helper.make_node("Gather", [x, idx], [gather], axis=axis))
            gathered_shape = shape.copy()
            del gathered_shape[axis % len(shape)]
            if variant == "shared":
                outputs.append(value(gather, gathered_shape, False))
            weight = floats(q + "_w", [4, 5]) if variant == "weight" else w
            bias = floats(q + "_b", [5]) if variant == "bias" else b
            nodes.append(helper.make_node("MatMul", [gather, weight], [q + "_mm"]))
            nodes.append(helper.make_node("Add", [q + "_mm", bias], [q + "_add"]))
            nodes.append(
                helper.make_node(
                    "LeakyRelu",
                    [q + "_add"],
                    [q + "_out"],
                    alpha=0.03 * (j + 1) if variant == "activation" else 0.02,
                )
            )
            outputs.append(value(q + "_out", [*gathered_shape[:-1], 5], False))
    model = helper.make_model(
        helper.make_graph(nodes, "static_cleanup", inputs, outputs, constants),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    regression.make_model = make_model
    regression.main()
