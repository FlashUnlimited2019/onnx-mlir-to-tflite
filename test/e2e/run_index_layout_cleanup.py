#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Exercise guarded row gathers, masked layouts, singleton gathers, and Div."""

import argparse
from dataclasses import dataclass
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
class Case:
    name: str
    model: onnx.ModelProto
    feeds: list
    no_gather_nd: bool = False
    integer_div: bool = False


def make_case(name, nodes, feeds, outputs, constants=None, **checks):
    constants = constants or {}
    inputs = [
        helper.make_tensor_value_info(
            k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape
        )
        for k, v in feeds[0].items()
    ]
    output_infos = [
        helper.make_tensor_value_info(
            k, helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), shape
        )
        for k, dtype, shape in outputs
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            name,
            inputs,
            output_infos,
            [numpy_helper.from_array(v, k) for k, v in constants.items()],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    return Case(name, model, feeds, **checks)


def values(dtype, shape, seed, special=False):
    rng = np.random.default_rng(seed)
    if dtype == np.float32:
        if special:
            bits = np.array(
                [
                    0x80000000,
                    0,
                    0x7F800000,
                    0xFF800000,
                    0x7FC00001,
                    0x7FC01234,
                    0x3F800000,
                ],
                np.uint32,
            )
            return np.resize(bits, shape).view(np.float32)
        return rng.normal(size=shape).astype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(dtype)
    bounds = np.iinfo(dtype)
    edge = np.array([bounds.min, bounds.max, 0, -1, 1, 37], dtype)
    return np.resize(np.roll(edge, seed % len(edge)), shape)


def fixtures():
    configs = [
        ("rows", (7, 5), 0, (4, 1), (4, 5), True),
        ("shared", (2, 7, 5), 1, (1, 4, 1), (2, 4, 5), True),
        ("batched", (2, 7, 5), -2, (2, 4, 1), (2, 4, 5), True),
        ("rank4_shared", (2, 3, 7, 5), 2, (1, 1, 4, 1), (2, 3, 4, 5), True),
        ("rank4_batched", (2, 3, 7, 5), -2, (2, 3, 4, 1), (2, 3, 4, 5), True),
        ("rank5_batched", (2, 3, 7, 2, 5), 2, (2, 3, 4, 1, 1), (2, 3, 4, 2, 5), True),
        ("scalar_index", (2, 7, 5), 1, (), (2, 1, 5), True),
        ("partial_prefix", (3, 7, 5), 1, (2, 4, 1), (2, 4, 5), False),
        ("partial_suffix", (2, 7, 5), 1, (2, 4, 1), (2, 4, 3), False),
        ("partial_batch_broadcast", (2, 3, 7, 5), 2, (2, 1, 4, 1), (2, 3, 4, 5), False),
        ("channel_dependent", (2, 7, 5), 1, (2, 4, 5), (2, 4, 5), False),
    ]
    for dtype in (np.float32, np.int32, np.int64):
        # ONNX Expand currently accepts i64, but not i32, indices. The
        # physical i32 broadcast form is covered by the MLIR regression.
        for idx_dtype in (np.int64,):
            for name, shape, axis, compact_shape, target, hit in configs:
                feeds = []
                for seed in range(3):
                    idx = np.random.default_rng(seed + 71).integers(
                        -shape[axis], shape[axis], size=compact_shape, dtype=idx_dtype
                    )
                    feeds.append(
                        dict(data=values(dtype, shape, seed, seed == 2), indices=idx)
                    )
                # Expose the expanded indices as well: optimized consumers must
                # not destroy a broadcast that is shared with a graph output.
                exposed = name == "batched"
                outputs = [("result", dtype, target)]
                if exposed:
                    outputs.append(("expanded", idx_dtype, target))
                yield make_case(
                    f"row_{name}_{np.dtype(dtype).name}_{np.dtype(idx_dtype).name}",
                    [
                        helper.make_node("Expand", ["indices", "shape"], ["expanded"]),
                        helper.make_node(
                            "GatherElements",
                            ["data", "expanded"],
                            ["result"],
                            axis=axis,
                        ),
                    ],
                    feeds,
                    outputs,
                    dict(shape=np.array(target, np.int64)),
                    no_gather_nd=hit,
                )

    select_configs = [
        ("unit_composition", (5, 1, 3), (1, 2, 0), (0, 2, 1), (1, 1, 5), True, ""),
        ("inverse", (2, 3, 5), (0, 2, 1), (0, 2, 1), (2, 5, 1), False, ""),
        ("scalar_mask", (2, 3, 5), (0, 2, 1), (0, 2, 1), (), False, ""),
        ("lower_rank_mask", (2, 3, 5), (0, 2, 1), (0, 2, 1), (3,), False, ""),
        (
            "shared_input",
            (5, 1, 3),
            (1, 2, 0),
            (0, 2, 1),
            (1, 1, 5),
            True,
            "transposed",
        ),
        ("shared_select", (5, 1, 3), (1, 2, 0), (0, 2, 1), (1, 1, 5), True, "selected"),
        ("mask_needs_copy", (2, 3, 5), (0, 2, 1), (0, 2, 1), (2, 5, 3), False, ""),
        (
            "equal_axes_not_inverse",
            (3, 3, 3),
            (1, 2, 0),
            (1, 2, 0),
            (3, 3, 3),
            False,
            "",
        ),
        ("rank4", (2, 3, 4, 5), (0, 2, 3, 1), (0, 3, 1, 2), (1, 4, 5, 1), False, ""),
        (
            "rank5_fallback",
            (2, 3, 4, 2, 5),
            (0, 2, 3, 4, 1),
            (0, 4, 1, 2, 3),
            (),
            False,
            "",
        ),
    ]
    for dtype in (np.float32, np.int64):
        for name, shape, p, q, mask_shape, expand, expose in select_configs:
            mid = tuple(shape[i] for i in p)
            out = tuple(mid[i] for i in q)
            feeds = [
                dict(
                    data=values(dtype, shape, seed, seed == 2),
                    mask=values(np.bool_, mask_shape, seed + 11),
                    other=np.array(-0.0 if dtype == np.float32 else -(1 << 63), dtype),
                )
                for seed in range(3)
            ]
            nodes = [helper.make_node("Transpose", ["data"], ["transposed"], perm=p)]
            constants = {}
            if expand:
                constants["shape"] = np.array(mid, np.int64)
                nodes.append(
                    helper.make_node("Expand", ["mask", "shape"], ["wide_mask"])
                )
            nodes += [
                helper.make_node(
                    "Where",
                    ["wide_mask" if expand else "mask", "other", "transposed"],
                    ["selected"],
                ),
                helper.make_node("Transpose", ["selected"], ["result"], perm=q),
            ]
            outputs = [("result", dtype, out)]
            if expose:
                outputs.append((expose, dtype, mid))
            yield make_case(
                f"select_{name}_{np.dtype(dtype).name}",
                nodes,
                feeds,
                outputs,
                constants,
            )

    for dtype in (np.float32, np.int32, np.int64, np.bool_):
        for shape, axis in [
            ((1,), 0),
            ((2, 1), -1),
            ((3, 1, 5), 1),
            ((2, 3, 1, 5), -2),
            ((2, 1, 3, 4, 5), 1),
        ]:
            for index in (0, -1):
                norm = axis % len(shape)
                out = shape[:norm] + shape[norm + 1 :]
                yield make_case(
                    f"unit_gather_{np.dtype(dtype).name}_r{len(shape)}_{index}",
                    [
                        helper.make_node(
                            "Gather", ["data", "indices"], ["result"], axis=axis
                        )
                    ],
                    [
                        dict(data=values(dtype, shape, seed, seed == 2))
                        for seed in range(3)
                    ],
                    [("result", dtype, out)],
                    dict(indices=np.array(index, np.int64)),
                )
    for name, shape, index, runtime in [
        ("vector_index", (3, 1, 5), np.array([0], np.int64), False),
        ("runtime_index", (3, 1, 5), np.array(0, np.int64), True),
        ("nonunit_axis", (3, 2, 5), np.array(0, np.int64), False),
    ]:
        feeds = []
        for seed in range(3):
            feed = dict(data=values(np.float32, shape, seed, seed == 2))
            if runtime:
                feed["indices"] = np.array(-1 if seed else 0, np.int64)
            feeds.append(feed)
        out = shape[:1] + index.shape + shape[2:]
        yield make_case(
            "gather_fallback_" + name,
            [helper.make_node("Gather", ["data", "indices"], ["result"], axis=1)],
            feeds,
            [("result", np.float32, out)],
            {} if runtime else dict(indices=index),
        )

    # Runtime bounds rather than unchecked assumptions about integer inputs.
    for name, ashape, bshape, lower, upper, positive, hit in [
        ("scalar", (), (), 0, 31, True, True),
        ("vector", (17,), (), 0, 2047, True, True),
        ("rank4", (2, 1, 4, 1), (3, 1, 5), 0, 65535, True, True),
        ("rank5", (2, 1, 3, 1, 5), (1, 4, 1, 2, 1), 0, 2047, True, True),
        ("zero", (17,), (), 0, 0, True, True),
        ("cap_limit", (17,), (), 0, (1 << 31) - 2, True, True),
        ("negative_lhs", (17,), (), -9, 2047, True, False),
        ("cap_overflow", (17,), (), 0, (1 << 31) - 1, True, False),
        ("missing_lower", (17,), (), None, 2047, True, False),
        ("unproven_divisor", (17,), (), 0, 2047, False, False),
    ]:
        nodes = []
        constants = dict(upper=np.array(upper, np.int64), one=np.array(1, np.int64))
        if lower is not None:
            constants["lower"] = np.array(lower, np.int64)
        nodes.append(
            helper.make_node(
                "Clip",
                ["a", "lower" if lower is not None else "", "upper"],
                ["bounded"],
            )
        )
        if positive:
            nodes.append(helper.make_node("Max", ["b", "one"], ["positive"]))
        nodes.append(
            helper.make_node(
                "Div", ["bounded", "positive" if positive else "b"], ["result"]
            )
        )
        feeds = []
        for divisor in (1, 3, 2048, (1 << 31) + 1, (1 << 63) - 1, -3):
            a = np.resize(
                np.array(
                    [-(1 << 63), (1 << 63) - 1, 0, 1, 31, 2047, (1 << 31) - 2], np.int64
                ),
                ashape,
            )
            if not ashape:
                a = np.array(29, np.int64)
            b = np.full(bshape, divisor, np.int64)
            feeds.append(dict(a=a, b=b))
        yield make_case(
            "bounded_div_" + name,
            nodes,
            feeds,
            [("result", np.int64, np.broadcast_shapes(ashape, bshape))],
            constants,
            integer_div=hit,
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
    reports = []
    comparisons = 0
    rejections = 0
    for case in fixtures():
        if args.filter and args.filter not in case.name:
            continue
        onnx.checker.check_model(case.model, full_check=True)
        source = args.work_dir / (case.name + ".onnx")
        onnx.save(case.model, source)
        reference = ort.InferenceSession(
            str(source), options, providers=["CPUExecutionProvider"]
        )
        expected = [reference.run(None, feed) for feed in case.feeds]
        if case.name.startswith("select_"):
            # The CPU reference Where kernel canonicalizes selected -0 to +0.
            # Check its numerical results, but use an element-copy oracle for
            # signed zeros and NaN payloads rather than weakening that check.
            p, q = [
                list(n.attribute[0].ints)
                for n in case.model.graph.node
                if n.op_type == "Transpose"
            ]
            for feed, refs in zip(case.feeds, expected):
                transposed = feed["data"].transpose(p)
                selected = np.where(feed["mask"], feed["other"], transposed)
                exact = dict(
                    result=selected.transpose(q),
                    selected=selected,
                    transposed=transposed,
                )
                for i, info in enumerate(case.model.graph.output):
                    np.testing.assert_array_equal(refs[i], exact[info.name])
                    refs[i] = exact[info.name]
        if case.name.startswith("bounded_div_"):
            constants = {
                v.name: numpy_helper.to_array(v) for v in case.model.graph.initializer
            }
            for feed, refs in zip(case.feeds, expected):
                a = np.minimum(feed["a"], constants["upper"])
                if "lower" in constants:
                    a = np.maximum(a, constants["lower"])
                b = (
                    np.maximum(feed["b"], 1)
                    if any(n.op_type == "Max" for n in case.model.graph.node)
                    else feed["b"]
                )
                # Python integers avoid both float conversion and INT64_MIN
                # absolute-value overflow when defining truncation toward zero.
                quotient = np.frompyfunc(
                    lambda x, y: (abs(int(x)) // abs(int(y)))
                    * (-1 if (x < 0) != (y < 0) else 1),
                    2,
                    1,
                )
                exact = np.asarray(quotient(a, b), dtype=np.int64)
                if not np.array_equal(refs[0], exact):
                    # The CPU reference's broadcast Div can mishandle divisors
                    # beyond i32. Retain and report the discrepancy explicitly.
                    assert np.any(np.abs(b.astype(object)) > np.iinfo(np.int32).max)
                    print(
                        "REFERENCE_BOUNDARY_DIFFERENCE",
                        case.name,
                        int(b.flat[0]),
                        flush=True,
                    )
                refs[0] = exact
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                target = args.work_dir / f"{case.name}.{layout}.{optimize}.tflite"
                cmd = [
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
                        cmd,
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
                if case.no_gather_nd:
                    assert schema.BuiltinOperator.GATHER_ND not in codes, (
                        case.name,
                        codes,
                    )
                if case.integer_div:
                    divisions = [
                        graph.Operators(i)
                        for i, c in enumerate(codes)
                        if c == schema.BuiltinOperator.DIV
                    ]
                    assert all(
                        graph.Tensors(o.Inputs(0)).Type() == schema.TensorType.INT32
                        for o in divisions
                    ), (case.name, codes)
                for delegates in (False, True):
                    interp = tf.lite.Interpreter(
                        model_path=str(target),
                        num_threads=1,
                        experimental_op_resolver_type=(
                            tf.lite.experimental.OpResolverType.AUTO
                            if delegates
                            else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                        ),
                    )
                    run = interp.get_signature_runner("serving_default")
                    details = run.get_input_details()
                    for feed, refs in zip(case.feeds, expected):
                        actual = run(
                            **{
                                k: np.array(
                                    onnx_to_tfl_input(
                                        v, tuple(details[k]["shape"]), layout
                                    ),
                                    copy=True,
                                    order="C",
                                )
                                for k, v in feed.items()
                            }
                        )
                        for info, ref in zip(case.model.graph.output, refs):
                            out = tfl_to_onnx_output(
                                actual[info.name], ref.shape, layout
                            )
                            assert out.dtype == ref.dtype and out.shape == ref.shape
                            np.testing.assert_array_equal(
                                out,
                                ref,
                                err_msg=f"{case.name} {layout} {optimize} {delegates}",
                            )
                            if out.dtype == np.float32:
                                np.testing.assert_array_equal(
                                    out.view(np.uint32), ref.view(np.uint32)
                                )
                            comparisons += 1
                    if case.name == "row_shared_float32_int64":
                        # In particular, do not truncate a large i64 index to
                        # an in-range i32 index or bypass the runtime check.
                        for invalid in (7, -8, 1 << 32, (1 << 63) - 1, -(1 << 63)):
                            feed = dict(case.feeds[0])
                            feed["indices"] = np.full_like(feed["indices"], invalid)
                            try:
                                run(**feed)
                            except RuntimeError:
                                rejections += 1
                            else:
                                raise AssertionError(
                                    f"accepted invalid row index {invalid}"
                                )
                    if case.name == "gather_fallback_runtime_index":
                        for invalid in (1, -2):
                            feed = dict(
                                case.feeds[0], indices=np.array(invalid, np.int64)
                            )
                            try:
                                run(**feed)
                            except RuntimeError:
                                rejections += 1
                            else:
                                raise AssertionError(
                                    "unchecked singleton gather became a view"
                                )
                    del run, interp
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
    (args.work_dir / "results.json").write_text(
        json.dumps(
            dict(
                conversions=len(reports),
                comparisons=comparisons,
                rejections=rejections,
                cases=reports,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS",
        len(reports),
        "conversions,",
        comparisons,
        "exact output comparisons,",
        rejections,
        "invalid-index rejections",
        flush=True,
    )


if __name__ == "__main__":
    main()
