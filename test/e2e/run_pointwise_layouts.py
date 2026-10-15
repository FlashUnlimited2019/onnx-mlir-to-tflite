#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check pointwise/view and padding rewrites against independent NumPy oracles.

Exercise both layouts, downstream optimization, and runtime delegates. Include
shared branches, repeated operands, unequal and equal axes, scalar/lower-rank
broadcasting, fused activation, and unit-axis insertion/removal. Flat public
tensors isolate these physical rewrites from the public layout contract.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_layout_cleanup import transpose
from run_static_index_optimizations import Graph, command


def pointwise(graph, name, inputs, fused=False):
    compute = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "relu": lambda x: np.maximum(x, 0),
        "relu6": lambda x: np.clip(x, 0, 6),
        "tanh": np.tanh,
        "logistic": lambda x: 1 / (1 + np.exp(-x)),
        "leaky_relu": lambda x: np.where(x >= 0, x, x * np.float32(0.125)),
    }[name]
    attrs = ""
    if name in ("add", "sub", "mul", "div"):
        attrs = f'fused_activation_function = "{"RELU6" if fused else "NONE"}"'
    if name == "leaky_relu":
        attrs = "alpha = 0.125 : f32"
    return graph.op(
        name,
        inputs,
        np.broadcast_shapes(*(v.shape for v in inputs)),
        np.float32,
        lambda *xs: np.clip(compute(*xs), 0, 6) if fused else compute(*xs),
        attrs,
    )


def make_graph():
    graph = Graph()
    rng = np.random.default_rng(517)
    graph.copy_inputs = set()
    binary = ("add", "sub", "mul", "div", "minimum", "maximum")
    unary = ("relu", "relu6", "tanh", "logistic", "leaky_relu")
    for shape in ((2, 3), (2, 3, 4), (2, 2, 2), (1, 3, 1, 4)):
        rank = len(shape)
        for name in binary + unary:
            for sharing in ("none", "first", "middle"):
                for rhs in (
                    ("scalar", "unit", "full", "lower", "pair", "same")
                    if name in binary
                    else ("unary",)
                ):
                    p = rng.permutation(rank)
                    first = transpose(graph, graph.input(shape, np.float32), p)
                    inputs = [first]
                    if rhs == "pair":
                        inputs.append(
                            transpose(graph, graph.input(shape, np.float32), p)
                        )
                    elif rhs == "same":
                        inputs.append(first)
                    elif name in binary:
                        other_shape = {
                            "scalar": (),
                            "unit": tuple(
                                d if i == rank - 1 else 1
                                for i, d in enumerate(first.shape)
                            ),
                            "full": first.shape,
                            "lower": first.shape[-1:],
                        }[rhs]
                        inputs.append(graph.input(other_shape, np.float32))
                    middle = pointwise(
                        graph, name, inputs, name in binary[:4] and rhs == "pair"
                    )
                    if sharing == "first":
                        graph.output(first)
                    if sharing == "middle":
                        graph.output(middle)
                    graph.output(transpose(graph, middle, np.argsort(p)))
                    graph.cases += 1

    for source, viewed in (
        ((1,), ()),
        ((), (1,)),
        ((1, 3, 1, 4), (1, 3, 4)),
        ((2, 1, 2), (1, 2, 2, 1)),
        ((1, 1), (1, 1, 1)),
        # Non-unit merges are negative cases and must not commute.
        ((2, 3, 4), (6, 4)),
    ):
        for name in binary + unary:
            for sharing in ("none", "first", "middle"):
                for rhs in (
                    ("scalar", "unit", "same") if name in binary else ("unary",)
                ):
                    first = graph.reshape(graph.input(source, np.float32), viewed)
                    inputs = [first]
                    if rhs == "same":
                        inputs.append(first)
                    elif rhs in ("scalar", "unit"):
                        shape = () if rhs == "scalar" or not viewed else (viewed[-1],)
                        inputs.append(graph.input(shape, np.float32))
                    middle = pointwise(graph, name, inputs)
                    if sharing == "first":
                        graph.output(first)
                    if sharing == "middle":
                        graph.output(middle)
                    graph.output(graph.reshape(middle, source))
                    graph.cases += 1

    for rank in range(1, 5):
        for explicit in (False, True):
            for sharing in ("none", "first", "middle"):
                graph.copy_inputs.add(len(graph.inputs))
                data = graph.input((2, 3, 2, 4)[:rank], np.float32)
                p = rng.permutation(rank)
                first = transpose(graph, data, p)
                pads = rng.integers(0, 3, (rank, 2), dtype=np.int32)
                fill = np.float32(-2.5 if explicit else 0)
                operands = [first, graph.constant(pads)]
                if explicit:
                    operands.append(graph.constant(fill, np.float32))
                middle = graph.op(
                    "padv2" if explicit else "pad",
                    operands,
                    np.array(first.shape) + pads.sum(axis=1),
                    np.float32,
                    lambda x, ps, v=fill: np.pad(x, ps, constant_values=v),
                )
                if sharing == "first":
                    graph.output(first)
                if sharing == "middle":
                    graph.output(middle)
                graph.output(transpose(graph, middle, np.argsort(p)))
                graph.cases += 1
    return graph


def sample(graph, seed):
    rng = np.random.default_rng(seed)
    values, feed = graph.constants.copy(), {}
    for i, value in enumerate(graph.inputs):
        # Nonzero finite operands also test division without accidental poles.
        data = rng.choice(
            np.array([-3, -1, -0.125, 0.125, 1, 3], np.float32), value.shape
        )
        if i in graph.copy_inputs and data.size >= 4:
            data[:4] = [np.nan, np.inf, -np.inf, -0.0]
        values[value.name] = data
        feed[f"input{i}"] = data
    for result, inputs, compute in graph.evaluations:
        values[result.name] = np.asarray(
            compute(*(values[v.name] for v in inputs)), dtype=result.dtype
        )
        assert values[result.name].shape == result.shape
    return feed, {f"output{i}": values[v.name] for i, v in enumerate(graph.outputs)}


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--onnx-mlir-opt", type=Path, default=repo / "build/bin/onnx-mlir-opt"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tf_tools = repo / "tensorflow/bazel-bin/tensorflow/compiler/mlir/lite"
    graph = make_graph()
    source = args.work_dir / "input.mlir"
    source.write_text(graph.module())
    samples = [sample(graph, seed) for seed in (61, 62, 63)]
    report = []
    for mode in ("legacy", "experimental"):
        lowered = args.work_dir / f"{mode}.mlir"
        command(
            [
                args.onnx_mlir_opt,
                source,
                "--allow-unregistered-dialect",
                f"--convert-onnx-to-tfl=layout-mode={mode}",
                "--verify-each",
                "-o",
                lowered,
            ]
        )
        for optimize in (False, True):
            parsed = args.work_dir / f"{mode}.{optimize}.mlir"
            flags = ["--tfl-optimize", "--canonicalize", "--cse"] if optimize else []
            command(
                [
                    tf_tools / "litert-opt",
                    lowered,
                    "--verify-each",
                    *flags,
                    "-o",
                    parsed,
                ]
            )
            target = parsed.with_suffix(".tflite")
            command(
                [
                    tf_tools / "flatbuffer_translate",
                    "-mlir-to-tflite-flatbuffer",
                    parsed,
                    "-o",
                    target,
                    "-emit-builtin-tflite-ops=true",
                    "-emit-select-tf-ops=false",
                    "-emit-custom-ops=false",
                ]
            )
            model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            sub = model.Subgraphs(0)
            assert (
                max(sub.Tensors(i).ShapeLength() for i in range(sub.TensorsLength()))
                <= 4
            )
            for delegates in (False, True):
                resolver = (
                    tf.lite.experimental.OpResolverType.AUTO
                    if delegates
                    else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                )
                runtime = tf.lite.Interpreter(
                    model_path=str(target),
                    num_threads=2,
                    experimental_op_resolver_type=resolver,
                )
                runner = runtime.get_signature_runner("serving_default")
                for feed, expected in samples:
                    actual = runner(**feed)
                    assert set(actual) == set(expected)
                    for name, ref in expected.items():
                        assert (
                            actual[name].shape == ref.shape
                            and actual[name].dtype == ref.dtype
                        ), (
                            name,
                            actual[name].shape,
                            ref.shape,
                            actual[name].dtype,
                            ref.dtype,
                        )
                        np.testing.assert_allclose(
                            actual[name],
                            ref,
                            rtol=2e-6,
                            atol=2e-7,
                            equal_nan=True,
                            err_msg=name,
                        )
                del runner, runtime
                row = dict(
                    layout=mode,
                    optimize=optimize,
                    delegates=delegates,
                    cases=graph.cases,
                    input_sets=len(samples),
                    status="PASS",
                )
                report.append(row)
                print(json.dumps(row), flush=True)
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
