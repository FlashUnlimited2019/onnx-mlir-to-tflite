#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify int64 arg reductions under the signed-int32 value-range contract."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def make_model():
    shapes = [(2, 3), (2, 4, 3), (2, 1, 3, 5), (1, 2, 3, 1, 4), (2, 1, 3), (1, 77)]
    inputs, outputs, nodes, cases = [], [], [], []
    for i, shape in enumerate(shapes):
        name = f"input{i}"
        inputs.append(helper.make_tensor_value_info(name, TensorProto.INT64, shape))
        for axis in range(len(shape)):
            for op in ("ArgMax", "ArgMin"):
                for keep in (0, 1):
                    for last in (0, 1):
                        out = f"output{len(outputs)}"
                        out_shape = list(shape)
                        if keep:
                            out_shape[axis] = 1
                        else:
                            del out_shape[axis]
                        nodes.append(
                            helper.make_node(
                                op,
                                [name],
                                [out],
                                axis=axis if keep else axis - len(shape),
                                keepdims=keep,
                                select_last_index=last,
                            )
                        )
                        outputs.append(
                            helper.make_tensor_value_info(
                                out, TensorProto.INT64, out_shape
                            )
                        )
                        cases.append((out, name, axis, op, keep, last))
    model = helper.make_model(
        helper.make_graph(nodes, "integer_arg_reductions", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model, shapes, cases


def make_inputs(shapes):
    bounds = np.array(
        [
            -(2**31),
            2**31 - 1,
            -(2**31),
            2**31 - 1,
            2**24,
            2**24 + 1,
            -(2**24),
            -(2**24) - 1,
            0,
            0,
            1,
            -1,
        ],
        np.int64,
    )
    rng = np.random.default_rng(517)
    return [
        {f"input{i}": np.resize(bounds, shape) for i, shape in enumerate(shapes)},
        {
            f"input{i}": np.full(shape, -(2**31), np.int64)
            for i, shape in enumerate(shapes)
        },
        {
            f"input{i}": rng.integers(-(2**31), 2**31, shape, dtype=np.int64)
            for i, shape in enumerate(shapes)
        },
    ]


def oracle(feed, cases):
    results = {}
    for out, name, axis, op, keep, last in cases:
        data = feed[name]
        values = np.flip(data, axis=axis) if last else data
        result = (np.argmax if op == "ArgMax" else np.argmin)(values, axis=axis)
        if last:
            result = data.shape[axis] - 1 - result
        if keep:
            result = np.expand_dims(result, axis)
        results[out] = result.astype(np.int64)
    return results


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--converter", type=Path, default=repo / "build/bin/onnx-to-tflite"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model, shapes, cases = make_model()
    source = args.work_dir / "integer_arg_reductions.onnx"
    onnx.save(model, source)
    feeds = make_inputs(shapes)
    expected = [oracle(feed, cases) for feed in feeds]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    reference = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    for feed, want in zip(feeds, expected):
        actual = reference.run(None, feed)
        for value, meta in zip(actual, reference.get_outputs()):
            np.testing.assert_array_equal(value, want[meta.name])
    report = []
    for mode in ("legacy", "experimental"):
        for optimize in (False, True):
            target = args.work_dir / f"{mode}.{optimize}.tflite"
            command = [
                str(args.converter),
                str(source),
                "-o",
                str(target),
                "--layout-mode",
                mode,
                "--verify-each",
                "--optimize-tfl" if optimize else "--no-optimize-tfl",
            ]
            with target.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command,
                    env=dict(os.environ, TMPDIR=str(args.work_dir.resolve())),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            flatbuffer = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            allowed = {
                schema.BuiltinOperator.RESHAPE,
                schema.BuiltinOperator.CAST,
                schema.BuiltinOperator.ARG_MAX,
                schema.BuiltinOperator.ARG_MIN,
                schema.BuiltinOperator.REVERSE_V2,
                schema.BuiltinOperator.SUB,
            }
            graph = flatbuffer.Subgraphs(0)
            assert all(
                flatbuffer.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                in allowed
                for i in range(graph.OperatorsLength())
            )
            for delegate in (False, True):
                resolver = (
                    tf.lite.experimental.OpResolverType.AUTO
                    if delegate
                    else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                )
                runtime = tf.lite.Interpreter(
                    model_path=str(target),
                    num_threads=2,
                    experimental_op_resolver_type=resolver,
                )
                runner = runtime.get_signature_runner("serving_default")
                for feed, want in zip(feeds, expected):
                    actual = runner(**feed)
                    assert set(actual) == set(want)
                    for name, value in want.items():
                        assert (
                            actual[name].shape == value.shape
                            and actual[name].dtype == np.int64
                        )
                        np.testing.assert_array_equal(actual[name], value, err_msg=name)
                row = dict(
                    mode=mode,
                    optimize=optimize,
                    delegate=delegate,
                    cases=len(cases),
                    input_sets=len(feeds),
                    status="PASS",
                )
                report.append(row)
                print(json.dumps(row), flush=True)
                del runner, runtime
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
