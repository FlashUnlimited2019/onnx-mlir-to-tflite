#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify public axis order independently of internal layouts and equal extents."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from run_signatures import compare


def boundary_graph(input_anchor):
    nodes, inputs, outputs = [], [], []

    def value(name, shape, dtype):
        return helper.make_tensor_value_info(name, dtype, shape)

    if input_anchor:
        cases = [
            ([3, 2], [1, 0], TensorProto.FLOAT),
            ([4, 2, 3], [1, 2, 0], TensorProto.FLOAT),
            ([3, 3, 3], [2, 0, 1], TensorProto.FLOAT),
            ([2, 3, 4, 5], [2, 0, 1, 3], TensorProto.FLOAT),
            ([2, 2, 2, 2], [2, 0, 3, 1], TensorProto.FLOAT),
            ([2, 3, 2, 4, 2], [1, 2, 4, 0, 3], TensorProto.FLOAT),
            ([1, 2, 1, 3, 1, 4], [3, 0, 1, 2, 4, 5], TensorProto.FLOAT),
        ]
    else:
        cases = [
            ([2, 3], [1, 0], TensorProto.FLOAT),
            ([1, 2, 3], [0, 2, 1], TensorProto.FLOAT),
            ([2, 2, 2], [2, 0, 1], TensorProto.FLOAT),
            ([2, 3, 4, 5], [3, 1, 0, 2], TensorProto.FLOAT),
            ([2, 2, 2, 2], [2, 0, 3, 1], TensorProto.FLOAT),
            ([2, 3, 4, 5], [2, 0, 3, 1], TensorProto.BOOL),
            ([2, 3, 4, 5], [2, 0, 3, 1], TensorProto.INT32),
            ([2, 3, 4, 5], [2, 0, 3, 1], TensorProto.INT64),
            ([2, 3, 2, 2, 2], [4, 1, 3, 0, 2], TensorProto.FLOAT),
            ([1, 2, 1, 3, 4, 5], [5, 1, 3, 0, 4, 2], TensorProto.FLOAT),
            ([1, 2, 1, 3, 1, 4, 5], [5, 1, 3, 0, 6, 4, 2], TensorProto.FLOAT),
            ([2, 3, 4], [1, 2, 0], TensorProto.BOOL),
            ([2, 3, 4], [1, 2, 0], TensorProto.INT32),
            ([2, 3, 4], [1, 2, 0], TensorProto.INT64),
        ]
    for index, (shape, permutation, dtype) in enumerate(cases):
        source, moved, result, alias = (
            f"{role}{index}" for role in ("input", "moved", "result", "alias")
        )
        inputs.append(value(source, shape, dtype))
        moved_shape = [shape[axis] for axis in permutation]
        nodes.append(helper.make_node("Transpose", [source], [moved], perm=permutation))
        if input_anchor:
            weight = f"weight{index}"
            weight_shape = [moved_shape[-1], 5]
            if len(moved_shape) > 5:
                weight_shape = moved_shape[:-2] + weight_shape
            inputs.append(value(weight, weight_shape, dtype))
            nodes.append(helper.make_node("MatMul", [moved, weight], [result]))
            result_shape = moved_shape[:-1] + [5]
        else:
            nodes.append(
                helper.make_node(
                    "Relu" if dtype == TensorProto.FLOAT else "Identity",
                    [moved],
                    [result],
                )
            )
            result_shape = moved_shape
        nodes.append(helper.make_node("Identity", [result], [alias]))
        # A boundary value may be returned more than once and still have
        # internal consumers. Neither case may change its public axis order.
        outputs.extend(
            [
                value(source, shape, dtype),
                value(result, result_shape, dtype),
                value(alias, result_shape, dtype),
            ]
        )
    return helper.make_graph(
        nodes, "input_boundary" if input_anchor else "output_boundary", inputs, outputs
    )


def mixed_graph(equal_extents):
    shape = [2, 4, 4, 4] if equal_extents else [2, 4, 3, 5]
    rng = np.random.default_rng(41)

    def value(name, dims):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)

    constants = [
        numpy_helper.from_array(
            rng.normal(0, 0.2, (4, 4, 1, 1)).astype(np.float32), "weight"
        ),
        numpy_helper.from_array(np.linspace(0.7, 1.4, 4, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.linspace(-0.2, 0.3, 4, dtype=np.float32), "bias"),
        numpy_helper.from_array(
            np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4), "constant"
        ),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "weight"], ["conv"], kernel_shape=[1, 1]),
        helper.make_node(
            "GroupNormalization", ["conv", "scale", "bias"], ["norm"], num_groups=2
        ),
        helper.make_node(
            "Conv", ["norm", "weight"], ["conv_after_norm"], kernel_shape=[1, 1]
        ),
        helper.make_node("MatMul", ["conv", "matrix"], ["product"]),
        helper.make_node(
            "Conv", ["input", "runtime_filter"], ["runtime_conv"], kernel_shape=[1, 1]
        ),
        # This non-migrated consumer forces the entry's legacy fallback layout.
        helper.make_node("DepthToSpace", ["input"], ["expanded"], blocksize=2),
        helper.make_node("Transpose", ["conv"], ["channels_last"], perm=[0, 2, 3, 1]),
        helper.make_node("Identity", ["conv"], ["alias"]),
    ]
    return helper.make_graph(
        nodes,
        "mixed_equal" if equal_extents else "mixed_distinct",
        [
            value("input", shape),
            value("matrix", [1, 1, shape[-1], 7]),
            value("unused", shape),
            value("runtime_filter", [6, 4, 1, 1]),
        ],
        [
            value("input", shape),
            value("conv", shape),
            value("norm", shape),
            value("conv_after_norm", shape),
            value("product", shape[:-1] + [7]),
            value("expanded", [shape[0], 1, shape[2] * 2, shape[3] * 2]),
            value("channels_last", [shape[i] for i in [0, 2, 3, 1]]),
            value("alias", shape),
            value("constant", [1, 2, 3, 4]),
            value("runtime_conv", [shape[0], 6, shape[2], shape[3]]),
        ],
        constants,
    )


def filter_graphs():
    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    for group, channels, filters in [
        (1, 2, [4, 2, 3, 3]),
        (2, 4, [6, 2, 1, 3]),
        (4, 4, [8, 1, 3, 1]),
        (1, 3, [3, 3, 3, 3]),
    ]:
        shape = [1, channels, 5, 7]
        output_shape = [1, filters[0], 6 - filters[2], 8 - filters[3]]
        # Runtime filters may be public outputs as well as internal consumers.
        # The last case also has equal numeric filter extents in OIHW and OHWI.
        yield helper.make_graph(
            [
                helper.make_node("Mul", ["input", "scale"], ["scaled"]),
                helper.make_node("Conv", ["scaled", "filter"], ["output"], group=group),
            ],
            f"runtime_filter_g{group}_c{channels}",
            [value("input", shape), value("filter", filters)],
            [value("output", output_shape), value("filter", filters)],
            [numpy_helper.from_array(np.array(0.25, np.float32), "scale")],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    conversions = comparisons = 0
    with tempfile.TemporaryDirectory(
        prefix="layout-boundary-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        for graph in [
            boundary_graph(False),
            boundary_graph(True),
            mixed_graph(False),
            mixed_graph(True),
            *filter_graphs(),
        ]:
            source = root / f"{graph.name}.onnx"
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
            )
            onnx.checker.check_model(model)
            onnx.save(model, source)
            for layout in ("legacy", "experimental"):
                for optimize in (False, True):
                    target = root / f"{graph.name}.{layout}.{optimize}.tflite"
                    converted = subprocess.run(
                        [
                            sys.executable,
                            str(args.converter.resolve()),
                            str(source),
                            "-o",
                            str(target),
                            "--layout-mode",
                            layout,
                            "--optimize-tfl" if optimize else "--no-optimize-tfl",
                        ],
                        text=True,
                        capture_output=True,
                    )
                    if converted.returncode:
                        raise RuntimeError(converted.stdout + converted.stderr)
                    conversions += 1
                    for delegates in (False, True):
                        record = compare(source, target, False, layout, delegates)
                        comparisons += record["numeric_seeds"] * len(
                            record["output_names"]
                        )
                    print(
                        f"PASS {graph.name}: layout={layout}, optimize={optimize}, "
                        "delegates=off/on, seeds=3",
                        flush=True,
                    )
    print(f"PASS conversions={conversions}, output_comparisons={comparisons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
