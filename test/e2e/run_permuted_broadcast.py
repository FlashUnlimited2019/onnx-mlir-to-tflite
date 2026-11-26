#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Exercise logical suffix broadcasting across independently permuted layouts."""

import argparse
from itertools import permutations, product
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


def make_model(permutation, equal_extents):
    result_shape = (2, 3, 3, 2) if equal_extents else (2, 3, 4, 5)
    inverse = np.argsort(permutation)
    input_shape = tuple(result_shape[i] for i in inverse)
    inputs, outputs, constants, nodes = [], [], [], []

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    inputs.append(value("x", input_shape))
    constants.append(numpy_helper.from_array(np.array(input_shape, np.int64), "shape"))
    nodes.extend(
        [
            helper.make_node("Reshape", ["x", "shape"], ["anchor"]),
            helper.make_node("Transpose", ["anchor"], ["a"], perm=permutation),
        ]
    )
    ops = ("Add", "Sub", "Mul", "Div", "Pow", "Min", "Max")
    cases = 0
    for rank in range(4):
        for order in permutations(range(rank)):
            for mask in product((False, True), repeat=rank):
                logical_shape = tuple(
                    1 if singleton else result_shape[4 - rank + i]
                    for i, singleton in enumerate(mask)
                )
                raw_shape = tuple(logical_shape[i] for i in np.argsort(order))
                name = f"b{cases}"
                # Include constants as well as runtime operands, without
                # inspecting or assuming the values in compiler matching.
                if cases % 5 == 0:
                    data = np.linspace(0.25, 1.5, int(np.prod(raw_shape))).astype(
                        np.float32
                    )
                    constants.append(
                        numpy_helper.from_array(data.reshape(raw_shape), name)
                    )
                else:
                    inputs.append(value(name, raw_shape))
                moved = f"m{cases}"
                if rank:
                    nodes.append(
                        helper.make_node("Transpose", [name], [moved], perm=order)
                    )
                else:
                    moved = name
                for reverse in (False, True):
                    op = ops[(cases + int(reverse)) % len(ops)]
                    operands = [moved, "a"] if reverse else ["a", moved]
                    result = f"y{cases}_{int(reverse)}"
                    if op in ("Min", "Max"):
                        operands.append("scalar")
                    nodes.append(helper.make_node(op, operands, [result]))
                    outputs.append(value(result, result_shape))
                cases += 1
    constants.append(numpy_helper.from_array(np.array(0.8, np.float32), "scalar"))
    model = helper.make_model(
        helper.make_graph(nodes, "permuted_broadcast", inputs, outputs, constants),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    reports, comparisons = [], 0
    for equal_extents in (False, True):
        for permutation in permutations(range(4)):
            label = "".join(map(str, permutation)) + f"_{int(equal_extents)}"
            model = make_model(permutation, equal_extents)
            source = args.work_dir / f"{label}.onnx"
            onnx.save(model, source)
            session = ort.InferenceSession(
                str(source), opts, providers=["CPUExecutionProvider"]
            )
            feeds, refs = [], []
            for seed in (0, 17, 59):
                rng = np.random.default_rng(seed)
                feed = {
                    v.name: np.array(rng.uniform(0.25, 1.5, v.shape), dtype=np.float32)
                    for v in session.get_inputs()
                }
                feeds.append(feed)
                refs.append(
                    dict(
                        zip(
                            [v.name for v in session.get_outputs()],
                            session.run(None, feed),
                        )
                    )
                )
            del session
            for layout in ("legacy", "experimental"):
                for optimize in (False, True):
                    target = args.work_dir / f"{label}.{layout}.{int(optimize)}.tflite"
                    with target.with_suffix(".log").open("w") as log:
                        subprocess.run(
                            [
                                str(args.converter.resolve()),
                                str(source),
                                "-o",
                                str(target),
                                "--layout-mode",
                                layout,
                                "--verify-each",
                                "--optimize-tfl" if optimize else "--no-optimize-tfl",
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=120,
                            env=dict(os.environ, TMPDIR=str(args.work_dir.resolve())),
                        )
                    fb = schema.Model.GetRootAsModel(target.read_bytes(), 0)
                    graph = fb.Subgraphs(0)
                    assert (
                        max(
                            graph.Tensors(i).ShapeLength()
                            for i in range(graph.TensorsLength())
                        )
                        <= 4
                    )
                    allowed = {
                        getattr(schema.BuiltinOperator, name)
                        for name in (
                            "ADD",
                            "SUB",
                            "MUL",
                            "DIV",
                            "POW",
                            "MINIMUM",
                            "MAXIMUM",
                            "RESHAPE",
                            "TRANSPOSE",
                        )
                    }
                    assert all(
                        fb.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                        in allowed
                        for i in range(graph.OperatorsLength())
                    )
                    for native in (False, True):
                        runtime = tf.lite.Interpreter(
                            model_path=str(target),
                            num_threads=1,
                            experimental_op_resolver_type=(
                                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                                if native
                                else tf.lite.experimental.OpResolverType.AUTO
                            ),
                        )
                        runner = runtime.get_signature_runner("serving_default")
                        details = runner.get_input_details()
                        for feed, expected in zip(feeds, refs):
                            # Constant-folded scalar paths can remove inputs;
                            # align through signature names, never positions.
                            actual = runner(
                                **{
                                    name: np.array(
                                        onnx_to_tfl_input(
                                            feed[name], tuple(d["shape"]), layout
                                        ),
                                        copy=True,
                                        order="C",
                                    )
                                    for name, d in details.items()
                                }
                            )
                            assert actual.keys() == expected.keys()
                            for name, ref in expected.items():
                                got = tfl_to_onnx_output(
                                    actual[name], ref.shape, layout
                                )
                                np.testing.assert_allclose(
                                    got,
                                    ref,
                                    rtol=2e-6,
                                    atol=2e-6,
                                    err_msg=f"{label} {layout} {optimize} {native} {name}",
                                )
                                comparisons += 1
                        del runner, runtime
                    reports.append(
                        dict(
                            case=label,
                            layout=layout,
                            optimize=optimize,
                            operators=graph.OperatorsLength(),
                            outputs=len(refs[0]),
                        )
                    )
                    print("PASS", label, layout, optimize, flush=True)
    result = dict(conversions=len(reports), comparisons=comparisons, cases=reports)
    (args.work_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
