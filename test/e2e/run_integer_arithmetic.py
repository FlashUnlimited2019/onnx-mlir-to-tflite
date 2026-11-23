#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Check static integer arithmetic, scalar bounds, and integer GatherElements."""

import argparse
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


@dataclass
class Fixture:
    name: str
    model: onnx.ModelProto
    feeds: list
    division: bool = False


def fixture(name, op, feeds, shape, dtype, constants=None, division=False, **attrs):
    constants = constants or {}
    first = feeds[0]
    arguments = list(first) + list(constants)
    if op == "Clip":
        arguments = [
            "data",
            "lo" if "lo" in arguments else "",
            "hi" if "hi" in arguments else "",
        ]
    inputs = [
        helper.make_tensor_value_info(
            key, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        for key, value in first.items()
    ]
    output = helper.make_tensor_value_info(
        "result", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), shape
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op, arguments, ["result"], **attrs)],
            name,
            inputs,
            [output],
            [numpy_helper.from_array(value, key) for key, value in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    return Fixture(name, model, feeds, division)


def exact_quotient(a, b):
    a, b = np.broadcast_arrays(a, b)
    values = []
    for x, y in zip(a.flat, b.flat):
        x, y = int(x), int(y)
        assert y and not (x == -(1 << 63) and y == -1)
        q = abs(x) // abs(y)
        values.append(-q if (x < 0) != (y < 0) else q)
    return np.asarray(values, dtype=np.int64).reshape(a.shape)


def fixtures():
    lo, hi = -(1 << 63), (1 << 63) - 1
    boundaries = [
        lo,
        lo + 1,
        lo + 1023,
        -(1 << 53) - 1,
        -(1 << 32) - 1,
        -17,
        -3,
        -1,
        0,
        1,
        2,
        3,
        17,
        (1 << 24) - 1,
        (1 << 24) + 1,
        (1 << 32) + 1,
        (1 << 53) - 1,
        (1 << 53) + 1,
        hi - 1023,
        hi - 1,
        hi,
    ]
    pairs = [
        (a, b)
        for a, b in itertools.product(boundaries, repeat=2)
        if b != 0 and (a, b) != (lo, -1)
    ]
    scalar_pairs = [
        (lo, 1),
        (lo, lo),
        (hi, 1),
        (hi, -1),
        (lo, 3),
        (lo, -3),
        (0, -7),
        (-17, 3),
        (17, -3),
        ((1 << 53) + 1, 3),
    ]
    yield fixture(
        "div_scalar",
        "Div",
        [
            dict(a=np.array(a, np.int64), b=np.array(b, np.int64))
            for a, b in scalar_pairs
        ],
        (),
        np.int64,
        division=True,
    )
    rng = np.random.default_rng(317)
    feeds = []
    for seed in range(8):
        a = rng.integers(lo, hi, size=8192, dtype=np.int64, endpoint=True)
        b = rng.integers(lo, hi, size=8192, dtype=np.int64, endpoint=True)
        # Exercise small divisors and quotients near exact-integer/FP32 limits.
        b[:4096] = rng.choice(
            np.array([1, -1, 2, -2, 3, -3, 7, -7, 31, -31, 1 << 24, 1 << 32], np.int64),
            4096,
        )
        a[: len(pairs)], b[: len(pairs)] = np.asarray(pairs, np.int64).T
        b[b == 0] = 1
        b[(a == lo) & (b == -1)] = 1
        feeds.append(dict(a=a, b=b))
    yield fixture("div_full_range", "Div", feeds, (8192,), np.int64, division=True)
    for name, ashape, bshape in [
        ("matrix", (3, 1), (1, 5)),
        ("rank3", (2, 3, 4), (4,)),
        ("rank4", (2, 1, 4, 1), (3, 1, 5)),
        ("rank5", (2, 1, 3, 1, 5), (1, 4, 1, 6, 1)),
        ("scalar_lhs", (), (2, 3, 4, 5)),
    ]:
        feeds = [
            dict(
                a=rng.integers(lo + 1, hi, size=ashape, dtype=np.int64),
                b=rng.choice(np.array([-7, -3, 1, 3, 7, hi], np.int64), size=bshape),
            )
            for _ in range(3)
        ]
        yield fixture(
            "div_" + name,
            "Div",
            feeds,
            np.broadcast_shapes(ashape, bshape),
            np.int64,
            division=True,
        )
    for divisor in (3, -3, lo):
        yield fixture(
            f"div_constant_{divisor}",
            "Div",
            [dict(a=np.asarray(boundaries, np.int64))],
            (len(boundaries),),
            np.int64,
            constants=dict(b=np.array(divisor, np.int64)),
            division=True,
        )
    fast = fixture(
        "div_bounded",
        "Div",
        [dict(a=np.array([-(1 << 31), -99, -1, 0, 1, 99, (1 << 31) - 1], np.int32))],
        (7,),
        np.int64,
        constants=dict(b=np.array(3, np.int64)),
        division=True,
    )
    fast.model.graph.node[0].input[0] = "wide"
    fast.model.graph.node.insert(
        0, helper.make_node("Cast", ["a"], ["wide"], to=onnx.TensorProto.INT64)
    )
    yield fast

    for dtype in (np.int32, np.int64, np.float32):
        limits = np.iinfo(dtype) if dtype != np.float32 else None
        values = (
            [
                limits.min,
                limits.min + 1,
                -101,
                -1,
                0,
                1,
                101,
                limits.max - 1,
                limits.max,
            ]
            if limits
            else [-100.5, -3.25, -1.0, 0.0, 0.5, 3.0, 100.25]
        )
        if dtype == np.int64:
            values += [-(1 << 53) - 1, (1 << 53) + 1]
        for shape in ((), (23,), (2, 3, 4, 5), (2, 3, 2, 4, 5)):
            for bounds in ("both", "min", "max", "none"):
                feeds = []
                ranges = [(-3, 7), (7, -3), (-101, 101)]
                if limits:
                    ranges.append((limits.min, limits.max))
                if dtype == np.int64:
                    ranges.append((-(1 << 53) - 1, (1 << 53) + 1))
                for index, (lower, upper) in enumerate(ranges):
                    feed = dict(data=np.resize(np.array(values, dtype), shape))
                    if not shape:
                        feed["data"] = np.array(
                            values[index] if index < 3 else values[-1], dtype
                        )
                    if bounds in ("both", "min"):
                        feed["lo"] = np.array(lower, dtype)
                    if bounds in ("both", "max"):
                        feed["hi"] = np.array(upper, dtype)
                    feeds.append(feed)
                yield fixture(
                    f"clip_{np.dtype(dtype).name}_r{len(shape)}_{bounds}",
                    "Clip",
                    feeds,
                    shape,
                    dtype,
                )

    for fmod in (0, 1):
        yield fixture(
            f"mod_scalar_{fmod}",
            "Mod",
            [
                dict(a=np.array(a, np.int64), b=np.array(b, np.int64))
                for a, b in [
                    (-17, 3),
                    (17, -3),
                    (-17, -3),
                    (0, 3),
                    (lo, 3),
                    (hi, -3),
                    ((1 << 53) + 1, 7),
                ]
            ],
            (),
            np.int64,
            fmod=fmod,
        )
    for shape in ((), (2, 3, 4, 5), (2, 3, 2, 4, 5)):
        # Keep the mathematical difference within the signed result range.
        feeds = [
            dict(
                a=rng.integers(-(1 << 61), 1 << 61, size=shape, dtype=np.int64),
                b=np.array(v, np.int64),
            )
            for v in (-7, 0, 11)
        ]
        yield fixture(f"sub_i64_r{len(shape)}", "Sub", feeds, shape, np.int64)

    for dtype in (np.int32, np.int64):
        values = np.array([np.iinfo(dtype).min, np.iinfo(dtype).max, 0, -1, 1], dtype)
        for shape, axis in [
            ((7,), 0),
            ((3, 5), -1),
            ((2, 3, 5), 1),
            ((2, 3, 4, 5), 2),
            ((2, 3, 4, 2, 5), -1),
        ]:
            out_shape = list(shape)
            out_shape[axis] = 3
            idx = rng.integers(
                -shape[axis], shape[axis], size=out_shape, dtype=np.int64
            )
            for runtime in (False, True):
                feeds = []
                for shift in range(3):
                    feed = dict(data=np.resize(np.roll(values, shift), shape))
                    if runtime:
                        feed["indices"] = (
                            idx
                            if shift == 0
                            else rng.integers(
                                -shape[axis],
                                shape[axis],
                                size=out_shape,
                                dtype=np.int64,
                            )
                        )
                    feeds.append(feed)
                yield fixture(
                    f"gather_{np.dtype(dtype).name}_r{len(shape)}_{runtime}",
                    "GatherElements",
                    feeds,
                    out_shape,
                    dtype,
                    constants={} if runtime else dict(indices=idx),
                    axis=axis,
                )
        windows = np.array([[2, 3, 4], [1, 2, 3], [0, 1, 2]], np.int32)
        yield fixture(
            f"gather_{np.dtype(dtype).name}_windows",
            "GatherElements",
            [dict(data=np.resize(values, (3, 5)))],
            (3, 3),
            dtype,
            constants=dict(indices=windows),
            axis=1,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    reports, comparisons, elements = [], 0, 0
    reference_precision_differences = 0
    for case in fixtures():
        if args.filter and args.filter not in case.name:
            continue
        onnx.checker.check_model(case.model, full_check=True)
        source = args.work_dir / f"{case.name}.onnx"
        onnx.save(case.model, source)
        reference = ort.InferenceSession(
            str(source), options, providers=["CPUExecutionProvider"]
        )
        expected = [reference.run(None, feed)[0] for feed in case.feeds]
        if case.division:
            constants = {
                v.name: numpy_helper.to_array(v) for v in case.model.graph.initializer
            }
            for feed, result in zip(case.feeds, expected):
                values = dict(constants, **feed)
                np.testing.assert_array_equal(
                    result, exact_quotient(values["a"], values["b"])
                )
        if case.model.graph.node[0].op_type == "Mod":
            fmod = next(
                a.i for a in case.model.graph.node[0].attribute if a.name == "fmod"
            )
            for index, (feed, result) in enumerate(zip(case.feeds, expected)):
                a, b = int(feed["a"]), int(feed["b"])
                value = abs(a) % abs(b) if fmod else a % b
                if fmod and a < 0:
                    value = -value
                exact = np.array(value, np.int64)
                # Some reference runtime versions use double fmod for int64,
                # losing integer precision above 2**53. Use integer arithmetic
                # as the oracle there; ordinary inputs must still match ORT.
                if not np.array_equal(result, exact):
                    assert fmod and abs(a) > (1 << 53)
                    reference_precision_differences += 1
                expected[index] = exact
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                target = args.work_dir / f"{case.name}.{layout}.{optimize}.tflite"
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
                        command,
                        check=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, TMPDIR=str(args.work_dir.resolve())),
                    )
                fb = schema.Model.GetRootAsModel(target.read_bytes(), 0)
                graph = fb.Subgraphs(0)
                codes = [
                    fb.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                    for i in range(graph.OperatorsLength())
                ]
                assert not set(codes) & {
                    schema.BuiltinOperator.CUSTOM,
                    schema.BuiltinOperator.IF,
                    schema.BuiltinOperator.WHILE,
                }
                for delegates in (False, True):
                    interpreter = tf.lite.Interpreter(
                        model_path=str(target),
                        num_threads=1,
                        experimental_op_resolver_type=(
                            tf.lite.experimental.OpResolverType.AUTO
                            if delegates
                            else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                        ),
                    )
                    runner = interpreter.get_signature_runner("serving_default")
                    details = runner.get_input_details()
                    for feed, result in zip(case.feeds, expected):
                        actual = runner(
                            **{
                                key: np.array(
                                    onnx_to_tfl_input(
                                        value, tuple(details[key]["shape"]), layout
                                    ),
                                    copy=True,
                                    order="C",
                                )
                                for key, value in feed.items()
                            }
                        )["result"]
                        value = tfl_to_onnx_output(actual, result.shape, layout)
                        assert (
                            value.dtype == result.dtype and value.shape == result.shape
                        )
                        np.testing.assert_array_equal(
                            value,
                            result,
                            err_msg=f"{case.name}, {layout}, opt={optimize}, delegates={delegates}",
                        )
                        comparisons += 1
                        elements += value.size
                    del runner, interpreter
                reports.append(
                    dict(
                        case=case.name,
                        layout=layout,
                        optimize=optimize,
                        operators=len(codes),
                        passed=True,
                    )
                )
                print("PASS", case.name, layout, optimize, flush=True)
    report = dict(
        conversions=len(reports),
        comparisons=comparisons,
        elements=elements,
        reference_precision_differences=reference_precision_differences,
        cases=reports,
    )
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS",
        len(reports),
        "conversions,",
        comparisons,
        "exact output comparisons,",
        elements,
        "elements",
        flush=True,
    )


if __name__ == "__main__":
    main()
