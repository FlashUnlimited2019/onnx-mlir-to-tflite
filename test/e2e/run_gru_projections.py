#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate static GRU projection packing against ONNX Runtime.

Exercise both reset equations, all directions, clipping, optional biases and
initial states, and the short-sequence fallback. Flat outputs make the oracle
independent of rank-four boundary conventions. Test both layout modes, with
and without the optional TFL optimizer and default runtime delegates.
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


def build_model(path):
    rng = np.random.default_rng(2409)
    nodes, inputs, outputs, initializers = [], [], [], []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(value, name))
        return name

    # T, B, I, H, initial state, bias, clip. Include odd gate widths,
    # singleton dimensions, saturation and sequences on both sides of the guard.
    variants = [
        (1, 2, 3, 4, "dynamic", True, None),
        (2, 1, 4, 3, "none", False, None),
        (3, 2, 5, 3, "dynamic", True, 0.5),
        (5, 3, 2, 7, "constant", True, None),
        (7, 2, 1, 1, "none", False, 2.0),
        (17, 1, 7, 5, "dynamic", False, None),
    ]
    for direction in ("forward", "reverse", "bidirectional"):
        directions = 2 if direction == "bidirectional" else 1
        for reset in (0, 1):
            for i, (t, b, width, h, state, bias, clip) in enumerate(variants):
                prefix = f"{direction}_{reset}_{i}"
                x = prefix + "_x"
                inputs.append(
                    helper.make_tensor_value_info(x, TensorProto.FLOAT, [t, b, width])
                )
                w = constant(
                    prefix + "_w",
                    rng.normal(0, 0.2, (directions, 3 * h, width)).astype(np.float32),
                )
                r = constant(
                    prefix + "_r",
                    rng.normal(0, 0.2, (directions, 3 * h, h)).astype(np.float32),
                )
                bias_name = ""
                if bias:
                    bias_name = constant(
                        prefix + "_bias",
                        rng.normal(0, 0.2, (directions, 6 * h)).astype(np.float32),
                    )
                state_name = ""
                if state != "none":
                    state_name = prefix + "_state"
                    if state == "constant":
                        constant(
                            state_name,
                            rng.normal(0, 0.4, (directions, b, h)).astype(np.float32),
                        )
                    else:
                        inputs.append(
                            helper.make_tensor_value_info(
                                state_name, TensorProto.FLOAT, [directions, b, h]
                            )
                        )
                attributes = dict(
                    direction=direction, hidden_size=h, linear_before_reset=reset
                )
                if clip is not None:
                    attributes["clip"] = clip
                y, yh = prefix + "_y", prefix + "_yh"
                nodes.append(
                    helper.make_node(
                        "GRU",
                        [x, w, r, bias_name, "", state_name],
                        [y, yh],
                        **attributes,
                    )
                )
                for name, size in (
                    (y, t * directions * b * h),
                    (yh, directions * b * h),
                ):
                    shape = constant(name + "_shape", np.array([size], dtype=np.int64))
                    nodes.append(
                        helper.make_node("Reshape", [name, shape], [name + "_flat"])
                    )
                    outputs.append(
                        helper.make_tensor_value_info(
                            name + "_flat", TensorProto.FLOAT, [size]
                        )
                    )
    model = helper.make_model(
        helper.make_graph(
            nodes, "static_gru_projections", inputs, outputs, initializers
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def validate(converter, root):
    source = root / "static_gru_projections.onnx"
    build_model(source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    cases = []
    for seed, scale in ((31, 0.0), (32, 0.5), (33, 5.0)):
        rng = np.random.default_rng(seed)
        feed = {
            value.name: rng.normal(0, scale, value.shape).astype(np.float32)
            for value in reference.get_inputs()
        }
        cases.append((feed, reference.run(None, feed)))
    names = {
        v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)
    }
    allowed = {
        "ADD",
        "SUB",
        "MUL",
        "LOGISTIC",
        "TANH",
        "FULLY_CONNECTED",
        "BATCH_MATMUL",
        "RESHAPE",
        "TRANSPOSE",
        "SPLIT",
        "SPLIT_V",
        "SLICE",
        "PACK",
        "MINIMUM",
        "MAXIMUM",
    }
    report = []
    for mode in ("legacy", "experimental"):
        for optimize in (True, False):
            target = root / f"{mode}_{optimize}.tflite"
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
            proc = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            (root / f"{mode}_{optimize}.log").write_text(proc.stdout)
            if proc.returncode:
                raise RuntimeError(proc.stdout)
            model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            assert model.SubgraphsLength() == 1
            graph = model.Subgraphs(0)
            counts = Counter()
            for i in range(graph.OperatorsLength()):
                code = model.OperatorCodes(graph.Operators(i).OpcodeIndex())
                counts[
                    names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
                ] += 1
            assert set(counts) <= allowed, counts
            assert counts["SPLIT"] > 0, "packed z/r gates must use equal splitting"
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
                signature = runtime.get_signature_runner("serving_default")
                max_error = 0.0
                for feed, expected in cases:
                    actual = signature(**feed)
                    for info, want in zip(reference.get_outputs(), expected):
                        got = actual[info.name]
                        assert got.shape == want.shape and got.dtype == want.dtype
                        assert np.isfinite(got).all()
                        np.testing.assert_allclose(
                            got,
                            want,
                            rtol=2e-5,
                            atol=2e-6,
                            err_msg=f"{mode}/{optimize}/{delegate}/{info.name}",
                        )
                        max_error = max(max_error, float(np.max(np.abs(got - want))))
                entry = dict(
                    layout=mode,
                    optimize_tfl=optimize,
                    delegate=delegate,
                    operators=sum(counts.values()),
                    operator_counts=dict(counts),
                    max_abs_error=max_error,
                )
                report.append(entry)
                print(json.dumps(entry), flush=True)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: 36 GRU configurations, 3 inputs, both layouts/optimizer/delegate settings"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        validate(args.converter.resolve(), args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="gru-projections-") as directory:
            validate(args.converter.resolve(), Path(directory))


if __name__ == "__main__":
    main()
