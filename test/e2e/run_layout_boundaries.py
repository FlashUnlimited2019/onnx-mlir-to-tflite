#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify non-rank-4 entry layouts with permuted internal tensor layouts."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import onnx
from onnx import TensorProto, helper

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
            ([2, 3, 2, 4, 2], [1, 2, 4, 0, 3], TensorProto.FLOAT),
            ([1, 2, 1, 3, 1, 4], [3, 0, 1, 2, 4, 5], TensorProto.FLOAT),
        ]
    else:
        cases = [
            ([2, 3], [1, 0], TensorProto.FLOAT),
            ([1, 2, 3], [0, 2, 1], TensorProto.FLOAT),
            ([2, 2, 2], [2, 0, 1], TensorProto.FLOAT),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="layout-boundary-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        for input_anchor in (False, True):
            graph = boundary_graph(input_anchor)
            source = root / f"{graph.name}.onnx"
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
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
                    compare(source, target, False)
                    print(
                        f"PASS {graph.name}: layout={layout}, optimize={optimize}, seeds=3",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
