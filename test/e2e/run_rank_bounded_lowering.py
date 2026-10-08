#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check low-rank ConvTranspose and GatherElements lowering in both layouts.

Rank-four fixtures must not introduce rank-five intermediates or constants.
Existing rank-three and rank-five inputs are also checked for regressions.
Flat graph boundaries isolate the operator semantics from external layouts.
"""

import argparse
from collections import Counter
import json
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
    def __init__(self, name):
        self.name = name
        self.nodes, self.inputs, self.outputs, self.initializers = [], [], [], []
        self.input_specs = {}

    def constant(self, name, data):
        self.initializers.append(numpy_helper.from_array(np.asarray(data), name))
        return name

    def input(self, name, shape, dtype=np.float32, extent=None):
        size = int(np.prod(shape))
        source = name + "_flat"
        onnx_type = {
            np.float32: TensorProto.FLOAT,
            np.int32: TensorProto.INT32,
            np.int64: TensorProto.INT64,
        }[dtype]
        if dtype != np.float32:
            # Integer boundaries already preserve logical order. Keep them
            # ranked so this fixture does not require an integer Reshape.
            self.inputs.append(helper.make_tensor_value_info(name, onnx_type, shape))
            self.input_specs[name] = (tuple(shape), dtype, extent)
            return name
        self.inputs.append(helper.make_tensor_value_info(source, onnx_type, [size]))
        self.input_specs[source] = (size, dtype, extent)
        target = self.constant(name + "_shape", np.array(shape, dtype=np.int64))
        self.nodes.append(helper.make_node("Reshape", [source, target], [name]))
        return name

    def output(self, name, shape):
        target = self.constant(name + "_shape", np.array([-1], dtype=np.int64))
        self.nodes.append(helper.make_node("Reshape", [name, target], [name + "_flat"]))
        self.outputs.append(
            helper.make_tensor_value_info(
                name + "_flat", TensorProto.FLOAT, [int(np.prod(shape))]
            )
        )

    def save(self, path):
        model = helper.make_model(
            helper.make_graph(
                self.nodes, self.name, self.inputs, self.outputs, self.initializers
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=8,
        )
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, path)

    def feed(self, seed, exceptional=False):
        rng = np.random.default_rng(seed)
        values = {}
        for name, (size, dtype, extent) in self.input_specs.items():
            if extent is None:
                data = rng.normal(0, 0.5, size).astype(dtype)
                if exceptional:
                    data[: min(size, 4)] = np.array(
                        [np.nan, np.inf, -np.inf, -0.0], dtype=dtype
                    )[: min(size, 4)]
            else:
                data = rng.integers(-extent, extent, size=size, dtype=dtype)
            values[name] = data
        return values


def gather_graph(rank):
    graph = Graph(f"gather_rank{rank}")
    rng = np.random.default_rng(71)
    shape = [2, 3, 4, 5, 2][:rank]
    data = graph.input("data", shape)
    for axis in range(rank):
        for dtype in (np.int32, np.int64):
            for constant in (False, True):
                name = f"a{axis}_{np.dtype(dtype).name}_{constant}"
                indices_shape = [max(1, n - 1) for n in shape]
                # The selected axis may be larger than the corresponding data
                # axis. Non-selected dimensions deliberately remain smaller.
                indices_shape[axis] = shape[axis] + 1
                if constant:
                    indices = graph.constant(
                        name + "_idx",
                        rng.integers(
                            -shape[axis], shape[axis], size=indices_shape, dtype=dtype
                        ),
                    )
                else:
                    indices = graph.input(
                        name + "_idx", indices_shape, dtype, shape[axis]
                    )
                graph.nodes.append(
                    helper.make_node(
                        "GatherElements", [data, indices], [name], axis=axis - rank
                    )
                )
                graph.output(name, indices_shape)
    return graph


def transpose_conv_graph(rank):
    graph = Graph(f"transpose_conv_rank{rank}")
    rng = np.random.default_rng(72)
    if rank == 3:
        cases = [
            ([4], [3], [2], [1, 0], [1], [1], 1),
            ([3], [2], [1], [0, 1], [2], [0], 2),
        ]
    elif rank == 4:
        cases = [
            ([3, 4], [2, 2], [2, 2], [0, 0, 0, 0], [1, 1], [0, 0], 1),
            ([2, 3], [4, 4], [4, 4], [0, 0, 0, 0], [1, 1], [0, 0], 1),
            ([3, 4], [3, 2], [2, 3], [1, 0, 0, 1], [1, 1], [1, 2], 1),
            ([3, 2], [2, 3], [1, 2], [0, 1, 1, 0], [2, 1], [0, 1], 2),
            ([1, 3], [2, 2], [3, 1], [0, 0, 0, 1], [1, 1], [0, 0], 1),
            ([3, 2], [2, 3], [1, 1], [0, 1, 1, 0], [1, 1], [0, 0], 1),
        ]
    else:
        cases = [
            (
                [1, 2, 3],
                [1, 2, 2],
                [1, 2, 3],
                [0, 0, 1, 0, 1, 0],
                [1, 1, 1],
                [0, 0, 1],
                1,
            ),
            (
                [2, 1, 3],
                [2, 1, 2],
                [2, 1, 2],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
                1,
            ),
            (
                [2, 3, 1],
                [2, 2, 1],
                [2, 3, 1],
                [0, 1, 0, 1, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
                1,
            ),
            (
                [2, 2, 3],
                [2, 2, 2],
                [2, 1, 2],
                [0, 1, 0, 1, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
                1,
            ),
        ]
    for i, (
        spatial,
        kernel,
        strides,
        pads,
        dilations,
        output_padding,
        group,
    ) in enumerate(cases):
        name = f"case{i}"
        x = graph.input(name + "_x", [2, 2] + spatial)
        weights = graph.constant(
            name + "_w", rng.normal(0, 0.2, [2, 2] + kernel).astype(np.float32)
        )
        bias = graph.constant(
            name + "_b", rng.normal(0, 0.1, [2 * group]).astype(np.float32)
        )
        graph.nodes.append(
            helper.make_node(
                "ConvTranspose",
                [x, weights, bias],
                [name],
                strides=strides,
                pads=pads,
                dilations=dilations,
                output_padding=output_padding,
                group=group,
            )
        )
        dimensions = len(spatial)
        output = [2, 2 * group] + [
            (size - 1) * stride
            + (k - 1) * dilation
            + 1
            + extra
            - pads[axis]
            - pads[axis + dimensions]
            for axis, (size, stride, k, dilation, extra) in enumerate(
                zip(spatial, strides, kernel, dilations, output_padding)
            )
        ]
        graph.output(name, output)
    return graph


def validate(converter, root):
    names = {
        v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)
    }
    report = []
    for kind in ("gather", "transpose_conv"):
        for rank in (3, 4, 5):
            fixture = (
                gather_graph(rank) if kind == "gather" else transpose_conv_graph(rank)
            )
            source = root / f"{fixture.name}.onnx"
            fixture.save(source)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            reference = ort.InferenceSession(
                str(source), sess_options=options, providers=["CPUExecutionProvider"]
            )
            cases = []
            for seed in (101, 102, 103):
                feed = fixture.feed(seed, exceptional=kind == "gather" and seed == 103)
                cases.append(
                    (
                        feed,
                        dict(
                            zip(
                                [v.name for v in reference.get_outputs()],
                                reference.run(None, feed),
                            )
                        ),
                    )
                )
            for mode in ("legacy", "experimental"):
                for optimize in (True, False):
                    target = root / f"{fixture.name}.{mode}.{optimize}.tflite"
                    command = [
                        str(converter),
                        str(source),
                        "-o",
                        str(target),
                        "--layout-mode",
                        mode,
                        "--verify-each",
                    ]
                    if not optimize:
                        command.append("--no-optimize-tfl")
                    process = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    target.with_suffix(".log").write_text(process.stdout)
                    if process.returncode:
                        raise RuntimeError(process.stdout)
                    model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
                    maximum = max(
                        model.Subgraphs(g).Tensors(i).ShapeLength()
                        for g in range(model.SubgraphsLength())
                        for i in range(model.Subgraphs(g).TensorsLength())
                    )
                    assert maximum <= max(rank, 4), (
                        fixture.name,
                        mode,
                        optimize,
                        maximum,
                    )
                    counts = Counter()
                    for g in range(model.SubgraphsLength()):
                        graph = model.Subgraphs(g)
                        for i in range(graph.OperatorsLength()):
                            code = model.OperatorCodes(graph.Operators(i).OpcodeIndex())
                            counts[
                                names[
                                    max(
                                        code.BuiltinCode(), code.DeprecatedBuiltinCode()
                                    )
                                ]
                            ] += 1
                    assert not {"CUSTOM", "IF", "WHILE"} & set(counts)
                    for delegate in (True, False):
                        kwargs = (
                            {}
                            if delegate
                            else {
                                "experimental_op_resolver_type": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                            }
                        )
                        runtime = tf.lite.Interpreter(
                            model_path=str(target), num_threads=1, **kwargs
                        )
                        runtime.allocate_tensors()
                        run = runtime.get_signature_runner("serving_default")
                        for feed, expected in cases:
                            actual = run(**feed)
                            for name, want in expected.items():
                                got = actual[name]
                                assert (
                                    got.shape == want.shape and got.dtype == want.dtype
                                )
                                if kind == "gather":
                                    np.testing.assert_array_equal(
                                        got.view(np.uint32), want.view(np.uint32)
                                    )
                                else:
                                    np.testing.assert_allclose(
                                        got, want, rtol=2e-5, atol=2e-6
                                    )
                    entry = dict(
                        fixture=fixture.name,
                        cases=len(fixture.outputs),
                        layout=mode,
                        optimize=optimize,
                        max_rank=maximum,
                        operators=sum(counts.values()),
                    )
                    report.append(entry)
                    print("PASS", entry, flush=True)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        validate(args.converter.resolve(), args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="rank-bounded-") as directory:
            validate(args.converter.resolve(), Path(directory))


if __name__ == "__main__":
    main()
