#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Check constant sliding windows and conservative GatherElements fallbacks."""

import argparse
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

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


def fixtures():
    def windows(batches, period, width, columns, step=1):
        row = np.arange(batches * period)[:, None]
        return (period - 1 - row % period) * step + np.arange(columns)[None, :]

    for number, config in enumerate(
        [(1, 2, 5, 3, 1), (2, 4, 7, 4, 1), (3, 3, 9, 5, 2), (2, 3, 8, 1, 2)]
    ):
        b, p, d, c, s = config
        for negative in (False, True):
            indices = windows(b, p, d, c, s)
            if negative:
                # Mix positive and equivalent negative indices within each row.
                indices[:, ::2] -= d
            dtype = np.int32 if negative else np.int64
            yield f"sliding_{number}_{negative}", (b * p, d), indices.astype(
                dtype
            ), 1, False, d - s <= 2 * c, False
    for rows in (1, 5):
        indices = np.broadcast_to(np.arange(2, 5), (rows, 3)).copy()
        yield f"fixed_{rows}", (rows, 8), indices, -1, False, True, False
    # A full-width window and a one-element input are valid degeneracies.
    yield "identity", (3, 4), np.tile(np.arange(4), (3, 1)), 1, False, True, False
    yield "singleton", (1, 1), np.array([[0]]), -1, False, True, False
    normal = windows(2, 3, 7, 3)
    yield "shared_output", (6, 7), normal, 1, False, True, True
    changed = normal.copy()
    changed[4, 1] += 1
    yield "interior_mismatch", (6, 7), changed, 1, False, False, False
    changed = normal.copy()
    changed[-1, -1] += 1
    yield "last_mismatch", (6, 7), changed, 1, False, False, False
    yield "reversed_columns", (6, 7), normal[:, ::-1].copy(), 1, False, False, False
    yield "partial_period", (5, 7), normal[:5], 1, False, False, False
    yield "partial_rows", (7, 7), normal, 1, False, False, False
    yield "runtime_indices", (6, 7), normal, 1, True, False, False
    yield "other_axis", (7, 6), normal.T.copy(), 0, False, False, False
    higher = np.array([[[0, 3], [2, 1]], [[1, 0], [3, 2]]], np.int64)
    yield "rank3", (2, 2, 4), higher, 2, False, False, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    comparisons = 0
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for name, shape, indices, axis, runtime, optimized, expose in fixtures():
        dtype = TensorProto.INT32 if indices.dtype == np.int32 else TensorProto.INT64
        inputs = [helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)]
        index_info = helper.make_tensor_value_info("indices", dtype, indices.shape)
        if runtime:
            inputs.append(index_info)
        outputs = [
            helper.make_tensor_value_info("result", TensorProto.FLOAT, indices.shape)
        ]
        if expose:
            outputs.append(index_info)
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "GatherElements", ["data", "indices"], ["result"], axis=axis
                    )
                ],
                name,
                inputs,
                outputs,
                [] if runtime else [numpy_helper.from_array(indices, "indices")],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        source = args.work_dir / f"{name}.onnx"
        onnx.save(model, source)
        reference = ort.InferenceSession(
            str(source), options, providers=["CPUExecutionProvider"]
        )
        feeds = []
        for seed in (3, 17, 29):
            feed = {
                "data": np.random.default_rng(seed)
                .normal(size=shape)
                .astype(np.float32)
            }
            if runtime:
                feed["indices"] = indices
            feeds.append(feed)
        # These are element moves, so signed zeros and NaN payloads must survive.
        bits = np.array(
            [0x80000000, 0, 0x7F800000, 0xFF800000, 0x7FC00001, 0x3F800000],
            dtype=np.uint32,
        )
        special = {"data": np.resize(bits, shape).view(np.float32)}
        if runtime:
            special["indices"] = indices
        feeds.append(special)
        expected = [reference.run(None, feed) for feed in feeds]
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                target = args.work_dir / f"{name}.{layout}.{optimize}.tflite"
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
                assert schema.BuiltinOperator.CUSTOM not in codes
                if optimized:
                    assert schema.BuiltinOperator.GATHER_ND not in codes, (name, codes)
                    assert schema.BuiltinOperator.GATHER not in codes, (name, codes)
                elif not optimize:
                    # Existing cleanup can flatten small constant GatherNd
                    # tables to Gather; both remain valid fallback paths.
                    assert any(
                        code in codes
                        for code in (
                            schema.BuiltinOperator.GATHER_ND,
                            schema.BuiltinOperator.GATHER,
                        )
                    ), (name, codes)
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
                    for feed, results in zip(feeds, expected):
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
                        )
                        for info, result in zip(outputs, results):
                            value = tfl_to_onnx_output(
                                actual[info.name], result.shape, layout
                            )
                            np.testing.assert_array_equal(value, result)
                            assert value.dtype == result.dtype
                            if value.dtype == np.float32:
                                np.testing.assert_array_equal(
                                    value.view(np.uint32), result.view(np.uint32)
                                )
                            comparisons += 1
                    del runner, interpreter
                reports.append(
                    dict(
                        case=name,
                        layout=layout,
                        optimize=optimize,
                        sliding_window=optimized,
                        operators=len(codes),
                        passed=True,
                    )
                )
                print("PASS", name, layout, optimize, flush=True)
    report = dict(conversions=len(reports), comparisons=comparisons, cases=reports)
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS",
        len(reports),
        "conversions,",
        comparisons,
        "exact output comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
