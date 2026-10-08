#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Exercise local layout rewrites with an independent element-copy oracle.

Physical TFL fixtures keep flat graph boundaries and test both layout modes,
with and without downstream optimization and runtime delegates. Shared inputs
and intermediate outputs are observable so accidental branch deletion fails.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_static_index_optimizations import Graph, command


def transpose(graph, value, perm, dtype=np.int32):
    return graph.op(
        "transpose",
        [value, graph.constant(perm, dtype)],
        [value.shape[i] for i in perm],
        value.dtype,
        np.transpose,
    )


def make_graph():
    graph = Graph()
    rng = np.random.default_rng(173)
    for dtype in (np.float32, np.int32, np.int64, np.bool_):
        for rank in range(1, 5):
            shape = [2, 3, 2, 4][:rank]
            for sharing in ("none", "input", "middle"):
                data = graph.input(shape, dtype)
                p = rng.permutation(rank)
                q = np.argsort(p)
                for kind in ("inverse", "tile", "slice"):
                    first = transpose(graph, data, p)
                    if sharing == "input":
                        graph.output(first)
                    if kind == "tile":
                        repeats = rng.integers(1, 4, size=rank)
                        middle = graph.op(
                            "tile",
                            [first, graph.constant(repeats, np.int64)],
                            np.array(first.shape) * repeats,
                            dtype,
                            np.tile,
                        )
                    elif kind == "slice":
                        begin = [int(rng.integers(0, d)) for d in first.shape]
                        size = [
                            -1 if i % 2 else max(1, d - start - 1)
                            for i, (d, start) in enumerate(zip(first.shape, begin))
                        ]
                        middle = graph.slice(first, begin, size)
                    else:
                        middle = first
                    if sharing == "middle":
                        graph.output(middle)
                    graph.output(transpose(graph, middle, q))
                    graph.cases += 1
                # Compose three slices, including tails of a bounded window.
                a = graph.slice(data, [0] * rank, [d - 1 for d in shape])
                b = graph.slice(a, [0] * rank, [-1] * rank)
                c = graph.slice(b, [0] * rank, [1] * rank)
                if sharing != "none":
                    graph.output(a)
                    graph.output(b)
                graph.output(c)
                graph.cases += 1
        for shape in ((2, 3, 4), (2, 2), (1, 2, 1, 3), (1, 1)):
            for sharing in ("none", "input", "middle"):
                for _ in range(4):
                    data = graph.input(shape, dtype)
                    first = transpose(graph, data, rng.permutation(len(shape)))
                    nonunit = [d for d in first.shape if d != 1]
                    at = int(rng.integers(0, len(nonunit) + 1))
                    target = nonunit[:at] + [1] + nonunit[at:]
                    middle = graph.reshape(first, target)
                    if sharing == "input":
                        graph.output(first)
                    if sharing == "middle":
                        graph.output(middle)
                    graph.output(transpose(graph, middle, rng.permutation(len(target))))
                    graph.cases += 1
        # These non-singleton dimension merges must retain their real reorder.
        data = graph.input((2, 3, 4), dtype)
        a = transpose(graph, data, (0, 2, 1))
        b = graph.reshape(a, (4, 2, 3))
        graph.output(transpose(graph, b, (0, 2, 1)))
        graph.cases += 1
        # Inverses must be checked by axis identity, not equal shape alone.
        data = graph.input((2, 2, 2), dtype)
        a = transpose(graph, data, (1, 2, 0))
        b = graph.op("tile", [a, graph.constant([1, 1, 2])], (2, 2, 4), dtype, np.tile)
        graph.output(transpose(graph, b, (1, 2, 0)))
        graph.cases += 1
    return graph


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--onnx-mlir-opt", type=Path, default=repo / "build/bin/onnx-mlir-opt"
    )
    tf_tools = repo / "tensorflow/bazel-bin/tensorflow/compiler/mlir/lite"
    parser.add_argument("--litert-opt", type=Path, default=tf_tools / "litert-opt")
    parser.add_argument(
        "--flatbuffer-translate", type=Path, default=tf_tools / "flatbuffer_translate"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    graph = make_graph()
    source = args.work_dir / "input.mlir"
    source.write_text(graph.module())
    samples = [graph.values(seed) for seed in (71, 72, 73)]
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
        assert lowered.read_text().count('"tfl.transpose"') < graph.module().count(
            '"tfl.transpose"'
        )
        for optimize in (False, True):
            parsed = args.work_dir / f"{mode}.{optimize}.mlir"
            flags = ["--tfl-optimize", "--canonicalize", "--cse"] if optimize else []
            command([args.litert_opt, lowered, "--verify-each", *flags, "-o", parsed])
            target = parsed.with_suffix(".tflite")
            command(
                [
                    args.flatbuffer_translate,
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
            subgraph = model.Subgraphs(0)
            assert (
                max(
                    subgraph.Tensors(i).ShapeLength()
                    for i in range(subgraph.TensorsLength())
                )
                <= 4
            )
            allowed = {
                schema.BuiltinOperator.RESHAPE,
                schema.BuiltinOperator.TRANSPOSE,
                schema.BuiltinOperator.TILE,
                schema.BuiltinOperator.SLICE,
            }
            assert all(
                model.OperatorCodes(subgraph.Operators(i).OpcodeIndex()).BuiltinCode()
                in allowed
                for i in range(subgraph.OperatorsLength())
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
                        value = actual[name]
                        assert value.shape == ref.shape and value.dtype == ref.dtype
                        if ref.dtype == np.float32:
                            value, ref = value.view(np.uint32), ref.view(np.uint32)
                        np.testing.assert_array_equal(value, ref, err_msg=name)
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
