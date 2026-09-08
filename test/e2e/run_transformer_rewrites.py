#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Numerical regression for static split/bias and tanh-GELU rewrites.

Runs both layouts with and without the optional TFL optimizer. Generated
models and intermediates are confined to a temporary work directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


class Graph:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self.serial = 0

    def name(self):
        self.serial += 1
        return f"v{self.serial}"

    def constant(self, value, dtype=np.float32):
        name = self.name()
        self.initializers.append(
            numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        )
        return name

    def input(self, shape):
        name = self.name()
        self.inputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        )
        return name

    def op(self, kind, *inputs, count=1, **attributes):
        names = [self.name() for _ in range(count)]
        self.nodes.append(helper.make_node(kind, inputs, names, **attributes))
        return names[0] if count == 1 else names

    def output(self, name, shape):
        self.outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        )

    def save(self, path):
        graph = helper.make_graph(
            self.nodes,
            "static_rewrite_regression",
            self.inputs,
            self.outputs,
            self.initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        model.ir_version = 10
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(model)
        onnx.save(model, path)


def split_cases():
    graph = Graph()
    for rank in range(1, 6):
        for axis in sorted({0, rank - 1}):
            for sizes in ([2, 2, 2], [1, 2, 3]):
                # Distinct extents disambiguate rank-4 boundary layouts.
                shape = list(range(2, rank + 2))
                shape[axis] = sum(sizes)
                source = graph.input(shape)
                parts = graph.op(
                    "Split",
                    source,
                    graph.constant(sizes, np.int64),
                    count=3,
                    axis=axis - rank,
                )
                for index, (part, size) in enumerate(zip(parts, sizes)):
                    result_shape = list(shape)
                    result_shape[axis] = size
                    if axis == rank - 1:
                        bias = (np.arange(size, dtype=np.float32) + index) * 0.125
                        if index == 0:
                            bias = np.float32(0.25)
                        elif index == 1:
                            bias = bias.reshape([1] * (rank - 1) + [size])
                        part = graph.op("Add", graph.constant(bias), part)
                    graph.output(part, result_shape)
    # Bias varies along a non-split dimension: do not concatenate it as a
    # channel vector. A separately observed split output must remain intact.
    source = graph.input([2, 3, 4])
    a, b = graph.op("Split", source, graph.constant([2, 2], np.int64), count=2, axis=2)
    bias = graph.constant(np.arange(6, dtype=np.float32).reshape(2, 3, 1))
    graph.output(a, [2, 3, 2])
    graph.output(graph.op("Add", a, bias), [2, 3, 2])
    graph.output(graph.op("Add", b, bias), [2, 3, 2])
    return graph


def gelu_cases():
    graph = Graph()
    half, one = graph.constant(0.5), graph.constant(1.0)
    coeff, scale = graph.constant(0.044715), graph.constant(0.797884583)
    for rank in range(1, 6):
        for variant in range(3):
            shape = list(range(2, rank + 1)) + [7]
            source = graph.input(shape)
            if variant == 2:
                cube = graph.op("Pow", source, graph.constant(3.0))
            else:
                square = graph.op("Mul", source, source)
                cube = graph.op("Mul", square, source)
            inner = graph.op("Add", graph.op("Mul", coeff, cube), source)
            factor = graph.op(
                "Add", one, graph.op("Tanh", graph.op("Mul", scale, inner))
            )
            if variant == 1:
                result = graph.op("Mul", factor, graph.op("Mul", half, source))
            else:
                result = graph.op("Mul", half, graph.op("Mul", factor, source))
            graph.output(result, shape)
    # Near-matching coefficients are intentionally not the GELU formula.
    source = graph.input([3, 7])
    square = graph.op("Mul", source, source)
    cube = graph.op("Mul", source, square)
    inner = graph.op("Add", source, graph.op("Mul", cube, graph.constant(0.05)))
    factor = graph.op("Add", graph.op("Tanh", graph.op("Mul", inner, scale)), one)
    graph.output(graph.op("Mul", graph.op("Mul", source, factor), half), [3, 7])
    graph.output(graph.op("Gelu", source, approximate="none"), [3, 7])
    return graph


def compare(onnx_path, tflite_path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=2)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inputs) == len(session.get_inputs())
    assert len(outputs) == len(session.get_outputs())
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        feeds = {}
        for metadata, detail in zip(session.get_inputs(), inputs):
            value = rng.uniform(-10.0, 10.0, metadata.shape).astype(np.float32)
            feeds[metadata.name] = value
            physical = onnx_to_tfl_input(value, tuple(detail["shape"]))
            assert physical.shape == tuple(detail["shape"])
            interpreter.set_tensor(detail["index"], np.ascontiguousarray(physical))
        reference = session.run(None, feeds)
        interpreter.invoke()
        for expected, detail in zip(reference, outputs):
            actual = tfl_to_onnx_output(
                interpreter.get_tensor(detail["index"]), expected.shape
            )
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
    model = schema.Model.GetRootAsModel(tflite_path.read_bytes(), 0)
    graph = model.Subgraphs(0)
    counts = Counter(
        model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
        for i in range(graph.OperatorsLength())
    )
    if onnx_path.stem == "gelu":
        assert counts[schema.BuiltinOperator.GELU] == 16, counts
        assert counts[schema.BuiltinOperator.TANH] == 1, counts
    else:
        assert counts[schema.BuiltinOperator.SPLIT] > 0, counts
        assert counts[schema.BuiltinOperator.SPLIT_V] > 0, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    for name in ("onnx-mlir", "onnx-mlir-opt", "litert-opt", "flatbuffer-translate"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tool_args = []
    for name in ("onnx-mlir", "onnx-mlir-opt", "litert-opt", "flatbuffer-translate"):
        path = getattr(args, name.replace("-", "_"))
        if path:
            tool_args.extend([f"--{name}", str(path.resolve())])
    with tempfile.TemporaryDirectory(
        prefix="rewrite-regression-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        for name, graph in (("split", split_cases()), ("gelu", gelu_cases())):
            source = root / f"{name}.onnx"
            graph.save(source)
            for layout in ("legacy", "experimental"):
                for optimize in (False, True):
                    target = root / f"{name}.{layout}.{optimize}.tflite"
                    command = [
                        sys.executable,
                        str(args.converter.resolve()),
                        str(source),
                        "-o",
                        str(target),
                        "--layout-mode",
                        layout,
                        "--optimize-tfl" if optimize else "--no-optimize-tfl",
                        *tool_args,
                    ]
                    result = subprocess.run(command, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(result.stdout + result.stderr)
                    compare(source, target)
                    print(
                        f"PASS {name}: layout={layout}, optimize_tfl={optimize}, seeds=3",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
