#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify static per-channel GroupNormalization across layouts and ranks."""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def fixtures():
    nodes, initializers, outputs, info, cases = [], [], [], [], []
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
        for n, s in [("input", [72]), ("scale", [6]), ("bias", [6])]
    ]
    shapes = [[12, 6], [2, 6, 6], [2, 6, 2, 3], [2, 6, 1, 2, 3]]
    for shape, groups, runtime in itertools.product(
        shapes, (1, 2, 3, 6), (False, True)
    ):
        index = len(cases)
        name = f"case{index}"
        epsilon = None if groups == 1 else 0.0 if groups == 2 else 1e-3
        scale = np.linspace(-1.1, 1.3, 6, dtype=np.float32)
        bias = np.linspace(-0.2, 0.4, 6, dtype=np.float32)
        initializers += [
            numpy_helper.from_array(np.array(shape, np.int64), name + "_shape"),
            numpy_helper.from_array(np.array([72], np.int64), name + "_flat"),
        ]
        params = ["scale", "bias"]
        if not runtime:
            params = [name + "_scale", name + "_bias"]
            initializers += [
                numpy_helper.from_array(scale, params[0]),
                numpy_helper.from_array(bias, params[1]),
            ]
        attrs = {"num_groups": groups}
        if epsilon is not None:
            attrs["epsilon"] = epsilon
        nodes += [
            helper.make_node("Reshape", ["input", name + "_shape"], [name + "_x"]),
            helper.make_node(
                "GroupNormalization", [name + "_x", *params], [name + "_y"], **attrs
            ),
            helper.make_node("Reshape", [name + "_y", name + "_flat"], [name]),
        ]
        info += [
            helper.make_tensor_value_info(name + suffix, TensorProto.FLOAT, shape)
            for suffix in ("_x", "_y")
        ]
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [72]))
        cases.append(
            (
                name,
                shape,
                groups,
                runtime,
                epsilon if epsilon is not None else 1e-5,
                scale,
                bias,
            )
        )
    model = helper.make_model(
        helper.make_graph(
            nodes, "group_normalization", inputs, outputs, initializers, value_info=info
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    return model, cases


def reference(feed, cases):
    outputs = {}
    for name, shape, groups, runtime, epsilon, scale, bias in cases:
        rows = feed["input"].reshape(shape[0] * groups, -1)
        centered = rows - rows.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = centered / np.sqrt(
                (centered * centered).mean(axis=1, keepdims=True) + np.float32(epsilon)
            )
        if runtime:
            scale, bias = feed["scale"], feed["bias"]
        outputs[name] = (
            normalized.reshape(shape[0], 6, -1) * scale[None, :, None]
            + bias[None, :, None]
        ).reshape(-1)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(work))
    env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env["PATH"]
    model, cases = fixtures()
    source = work / "group_normalization.onnx"
    onnx.save(model, source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    tests = []
    for seed, kind in [
        (13, "random"),
        (37, "random"),
        (59, "random"),
        (13, "zero"),
        (13, "constant"),
        (13, "small"),
    ]:
        rng = np.random.default_rng(seed)
        feed = {
            name: rng.normal(0, 0.5, count).astype(np.float32)
            for name, count in (("input", 72), ("scale", 6), ("bias", 6))
        }
        if kind == "zero":
            feed["input"].fill(0)
        elif kind == "constant":
            feed["input"].fill(2)
        elif kind == "small":
            feed["input"] *= np.float32(1e-4)
        independent = reference(feed, cases)
        expected = dict(
            zip([v.name for v in session.get_outputs()], session.run(None, feed))
        )
        for name in expected:
            np.testing.assert_allclose(
                expected[name],
                independent[name],
                rtol=3e-5,
                atol=3e-6,
                err_msg=f"ORT {name} {kind}",
            )
        tests.append((feed, expected, independent))
    comparisons = 0
    for layout, optimize in itertools.product(
        ("legacy", "experimental"), (False, True)
    ):
        target = work / f"{layout}.{int(optimize)}.tflite"
        with target.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    str(args.converter.resolve()),
                    str(source),
                    "-o",
                    str(target),
                    "--layout-mode",
                    layout,
                    "--optimize-tfl" if optimize else "--no-optimize-tfl",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        flatbuffer = schema.Model.GetRootAsModel(target.read_bytes(), 0)
        assert flatbuffer.SubgraphsLength() == 1
        graph = flatbuffer.Subgraphs(0)
        assert (
            max(graph.Tensors(i).ShapeLength() for i in range(graph.TensorsLength()))
            <= 5
        )
        forbidden = {
            schema.BuiltinOperator.CUSTOM,
            schema.BuiltinOperator.WHILE,
            schema.BuiltinOperator.IF,
            schema.BuiltinOperator.SQUARE,
        }
        assert all(
            flatbuffer.OperatorCodes(i).BuiltinCode() not in forbidden
            for i in range(flatbuffer.OperatorCodesLength())
        )
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
            for feed, expected, independent in tests:
                actual = run(**feed)
                for name in expected:
                    np.testing.assert_allclose(
                        actual[name],
                        expected[name],
                        rtol=3e-5,
                        atol=3e-6,
                        err_msg=f"{target.name} {name}",
                    )
                    np.testing.assert_allclose(
                        actual[name], independent[name], rtol=3e-5, atol=3e-6
                    )
                    comparisons += 1
        print("PASS", target.name, flush=True)
    rejections = 0
    for attribute, value in [
        ("num_groups", 0),
        ("num_groups", -1),
        ("num_groups", 5),
        ("num_groups", 7),
        ("stash_type", 10),
        ("epsilon", -1.0),
        ("epsilon", float("inf")),
        ("epsilon", float("nan")),
    ]:
        invalid = copy.deepcopy(model)
        node = next(n for n in invalid.graph.node if n.op_type == "GroupNormalization")
        kept = [a for a in node.attribute if a.name != attribute]
        del node.attribute[:]
        node.attribute.extend(kept + [helper.make_attribute(attribute, value)])
        source = work / f"reject_{attribute}_{value}.onnx"
        onnx.save(invalid, source)
        for layout in ("legacy", "experimental"):
            target = source.with_suffix(f".{layout}.tflite")
            with target.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    [
                        str(args.converter.resolve()),
                        str(source),
                        "-o",
                        str(target),
                        "--layout-mode",
                        layout,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            assert result.returncode != 0 and not target.exists()
            rejections += 1
    result = dict(
        cases=len(cases), output_comparisons=comparisons, rejections=rejections
    )
    (work / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", result)


if __name__ == "__main__":
    main()
