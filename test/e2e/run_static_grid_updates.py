#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate constant-coordinate selection/update rewrites in both layouts.

Flat boundaries make internal rank/layout changes observable independently of
the public boundary convention. NumPy and ONNX Runtime are separate oracles.
"""

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


class Graph:
    def __init__(self):
        self.nodes, self.constants, self.inputs, self.outputs = [], [], [], []
        self.values, self.evaluations = {}, []
        self.serial = 0
        self.cases = 0

    def name(self):
        self.serial += 1
        return f"v{self.serial}"

    def constant(self, array):
        name = self.name()
        array = np.asarray(array, dtype=np.int64)
        self.constants.append(numpy_helper.from_array(array, name))
        self.values[name] = array
        return name

    def op(self, kind, inputs, compute, **attrs):
        name = self.name()
        self.nodes.append(helper.make_node(kind, inputs, [name], **attrs))
        self.evaluations.append((name, inputs, compute))
        return name

    def reshape(self, data, shape):
        return self.op(
            "Reshape", [data, self.constant(shape)], lambda x, s: x.reshape(s)
        )

    def input(self, shape):
        name = self.name()
        self.inputs.append(
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [int(np.prod(shape))]
            )
        )
        return self.reshape(name, shape)

    def output(self, data, shape):
        name = self.reshape(data, [int(np.prod(shape))])
        self.outputs.append(
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [int(np.prod(shape))]
            )
        )

    def case(self, shape, coordinates, reduction="none", gather=True):
        coordinates = np.asarray(coordinates, np.int64)
        depth = coordinates.shape[-1]
        value_shape = coordinates.shape[:-1] + tuple(shape[depth:])
        data, updates = self.input(shape), self.input(value_shape)
        indices = self.constant(coordinates)

        def select(x, i):
            return x[tuple(np.moveaxis(i, -1, 0))]

        def scatter(x, i, u):
            result = x.copy()
            at = tuple(np.moveaxis(i, -1, 0))
            if reduction == "none":
                result[at] = u
            else:
                fn = {
                    "add": np.add,
                    "mul": np.multiply,
                    "max": np.maximum,
                    "min": np.minimum,
                }[reduction]
                result[at] = fn(result[at], u)
            return result

        result = self.op(
            "ScatterND", [data, indices, updates], scatter, reduction=reduction
        )
        self.output(result, shape)
        # Keep both the updated tensor and its selected slab alive: rewrites
        # must preserve other consumers, not destructively alter shared data.
        if gather:
            self.output(self.op("GatherND", [data, indices], select), value_shape)
            self.output(self.op("GatherND", [result, indices], select), value_shape)
        self.cases += 1

    def model(self):
        graph = helper.make_graph(
            self.nodes, "static_grid_updates", self.inputs, self.outputs, self.constants
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        return model

    def feed_and_reference(self, seed, exceptional):
        rng = np.random.default_rng(seed)
        values = self.values.copy()
        feed = {}
        for i, value in enumerate(self.inputs):
            count = value.type.tensor_type.shape.dim[0].dim_value
            data = (rng.integers(-16, 17, count) / 8).astype(np.float32)
            if exceptional:
                special = np.array(
                    [np.nan, np.inf, -np.inf, -0.0, 1e20, 1.0], np.float32
                )
                data[: min(count, 6)] = np.roll(special, i)[: min(count, 6)]
            feed[value.name] = data
            values[value.name] = data
        for name, inputs, evaluate in self.evaluations:
            values[name] = evaluate(*(values[x] for x in inputs))
        return feed, {v.name: values[v.name] for v in self.outputs}


def regular_cases():
    graph = Graph()
    for rank in range(1, 6):
        for depth in range(1, rank + 1):
            for axis in range(depth):
                for start, extent in ((0, 1), (1, 2), (3, 1)):
                    shape = [2] * rank
                    shape[axis] = 4
                    ranges = [range(d) for d in shape[:depth]]
                    ranges[axis] = range(start, start + extent)
                    coords = np.array(list(itertools.product(*ranges)), np.int64)
                    if (rank + axis + start) % 2:
                        coords -= np.array(shape[:depth])
                    if start == 1:
                        coords = coords.reshape(
                            tuple(len(r) for r in ranges) + (depth,)
                        )
                    graph.case(shape, coords)
        shape = [2] * rank
        # Full coverage with trailing slices, including singleton index axes.
        depth = max(1, rank - 1)
        coords = np.array(list(itertools.product(*[range(2)] * depth)), np.int64)
        graph.case(shape, coords.reshape(1, -1, depth))
    # A rank-one index tensor selects a complete trailing slice.
    graph.case([4, 3], [2])
    return graph


def fallback_cases():
    graph = Graph()
    for coordinates in (
        [[0, 0], [0, 2]],  # Holes.
        [[1, 2], [1, 1], [1, 0]],  # Reversed traversal.
        [[0, 0], [0, 1], [1, 0], [1, 1]],  # Partial on two axes.
    ):
        graph.case([3, 3, 2], coordinates)
    for reduction in ("add", "mul", "min", "max"):
        graph.case([2, 3, 2], [[0, 0], [1, 0]], reduction=reduction)
    # Gather permits duplicates; ScatterND overwrite does not.
    data = graph.input([3, 4, 2])
    coords = graph.constant([[2, 3], [0, 0], [2, 3], [-1, -2]])
    result = graph.op(
        "GatherND", [data, coords], lambda x, i: x[tuple(np.moveaxis(i, -1, 0))]
    )
    graph.output(result, [4, 2])
    graph.cases += 1
    return graph


def assert_bits(actual, expected, label):
    np.testing.assert_array_equal(
        actual.view(np.uint32), expected.view(np.uint32), err_msg=label
    )


def counts(path):
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    enum = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    result = Counter()
    for i in range(model.Subgraphs(0).OperatorsLength()):
        op = model.Subgraphs(0).Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        result[enum[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]] += 1
    assert not any(x in result for x in ("CUSTOM", "WHILE", "IF"))
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="static-grid-updates-"))
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(work.resolve())
    summary = {"comparisons": 0, "models": []}
    for label, graph in (("regular", regular_cases()), ("fallback", fallback_cases())):
        model = graph.model()
        source = work / f"{label}.onnx"
        onnx.save(model, source)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(source), sess_options=options, providers=["CPUExecutionProvider"]
        )
        fixtures = []
        for seed in (11, 29, 47):
            for exceptional in ((False, True) if label == "regular" else (False,)):
                feed, reference = graph.feed_and_reference(seed, exceptional)
                outputs = session.run(None, feed)
                for item, value in zip(session.get_outputs(), outputs):
                    assert_bits(value, reference[item.name], f"ORT {label} {item.name}")
                fixtures.append((feed, reference))
        for layout, optimize in itertools.product(
            ("legacy", "experimental"), (False, True)
        ):
            target = work / f"{label}.{layout}.{int(optimize)}.tflite"
            command = [
                str(args.converter.resolve()),
                str(source.resolve()),
                "-o",
                str(target.resolve()),
                "--layout-mode",
                layout,
                "--verify-each",
                "--optimize-tfl" if optimize else "--no-optimize-tfl",
            ]
            with target.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            operators = counts(target)
            if label == "regular":
                assert not operators.get("SCATTER_ND") and not operators.get(
                    "GATHER_ND"
                ), operators
            else:
                assert operators.get("SCATTER_ND") and operators.get(
                    "GATHER"
                ), operators
            for delegate in (False, True):
                resolver = tf.lite.experimental.OpResolverType
                runtime = tf.lite.Interpreter(
                    model_path=str(target),
                    num_threads=1,
                    experimental_op_resolver_type=(
                        resolver.AUTO
                        if delegate
                        else resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                    ),
                )
                run = runtime.get_signature_runner("serving_default")
                for feed, reference in fixtures:
                    actual = run(**feed)
                    for name in reference:
                        assert_bits(
                            actual[name],
                            reference[name],
                            f"{target.name} {delegate} {name}",
                        )
                        summary["comparisons"] += 1
            entry = {"model": target.name, "cases": graph.cases, "operators": operators}
            summary["models"].append(entry)
            print(json.dumps(entry), flush=True)
    # Invalid coordinates must not become apparently valid linear offsets;
    # duplicate overwrite coordinates must not slip through slab recognition.
    for label, coordinates in (
        ("duplicate", [[0, 0], [0, 0]]),
        ("positive_oob", [[0, 3], [1, 0]]),
        ("negative_oob", [[-3, 0], [1, 0]]),
    ):
        graph = Graph()
        graph.case([2, 3], coordinates, gather=False)
        source = work / f"{label}.onnx"
        onnx.save(graph.model(), source)
        for layout in ("legacy", "experimental"):
            with (work / f"{label}.{layout}.log").open("w") as log:
                result = subprocess.run(
                    [
                        str(args.converter.resolve()),
                        str(source.resolve()),
                        "-o",
                        str((work / f"{label}.{layout}.tflite").resolve()),
                        "--layout-mode",
                        layout,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            assert result.returncode != 0, (label, layout)
    summary["rejected_invalid_models"] = 6
    (work / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"PASS: {summary['comparisons']} bit-exact output comparisons; 6 rejection checks",
        flush=True,
    )


if __name__ == "__main__":
    main()
