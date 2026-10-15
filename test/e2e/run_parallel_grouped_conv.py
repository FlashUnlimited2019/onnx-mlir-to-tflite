#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check parallel grouped convolutions and branched batch normalization."""

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


def make_model():
    inputs, outputs, nodes, constants = [], [], [], []
    rng = np.random.default_rng(918)
    configs = [(1, 4, 6), (2, 4, 6), (3, 6, 9), (4, 4, 4), (4, 4, 8)]
    for group, channels, filters in configs:
        for bias in (False, True):
            prefix = f"g{group}_c{channels}_f{filters}_b{int(bias)}"
            x = prefix + "_input"
            inputs.append(
                helper.make_tensor_value_info(x, TensorProto.FLOAT, [2, channels, 7, 9])
            )

            def constant(suffix, value):
                name = prefix + suffix
                constants.append(
                    numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)
                )
                return name

            results = []
            for branch in range(2):
                w = constant(
                    f"_w{branch}",
                    rng.normal(0, 0.2, (filters, channels // group, 3, 3)),
                )
                operands = [x, w]
                if bias:
                    operands.append(
                        constant(f"_b{branch}", rng.normal(0, 0.2, filters))
                    )
                y = prefix + f"_conv{branch}"
                nodes.append(
                    helper.make_node(
                        "Conv",
                        operands,
                        [y],
                        group=group,
                        pads=[1, 1, 1, 1],
                        kernel_shape=[3, 3],
                    )
                )
                results.append(y)
            # A normalized and an unnormalized branch can expose unsafe
            # merging even if the original model has only one convolution.
            bn = prefix + "_normalized"
            scale = constant("_scale", rng.uniform(0.5, 2, filters))
            offset = constant("_offset", rng.normal(0, 0.2, filters))
            mean = constant("_mean", rng.normal(0, 0.2, filters))
            variance = constant("_variance", rng.uniform(0.1, 2, filters))
            nodes.append(
                helper.make_node(
                    "BatchNormalization",
                    [results[0], scale, offset, mean, variance],
                    [bn],
                    epsilon=1.0e-5,
                )
            )
            results.append(bn)
            for result in results:
                outputs.append(
                    helper.make_tensor_value_info(
                        result, TensorProto.FLOAT, [2, filters, 7, 9]
                    )
                )
            # Exercise both the common-concat and independent-consumer paths.
            if group > 1:
                name = prefix + "_concat"
                nodes.append(
                    helper.make_node("Concat", [results[1], results[0]], [name], axis=1)
                )
                outputs.append(
                    helper.make_tensor_value_info(
                        name, TensorProto.FLOAT, [2, filters * 2, 7, 9]
                    )
                )
    model = helper.make_model(
        helper.make_graph(nodes, "parallel_convolutions", inputs, outputs, constants),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--converter", type=Path, default=repo / "build/bin/onnx-to-tflite"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / "parallel_convolutions.onnx"
    onnx.save(make_model(), source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    reference = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    feeds, expected = [], []
    for seed in (19, 31, 47):
        rng = np.random.default_rng(seed)
        feed = {
            meta.name: rng.normal(0, 1, meta.shape).astype(np.float32)
            for meta in reference.get_inputs()
        }
        feeds.append(feed)
        expected.append(
            dict(
                zip(
                    [meta.name for meta in reference.get_outputs()],
                    reference.run(None, feed),
                )
            )
        )
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
            model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            for i in range(model.OperatorCodesLength()):
                assert model.OperatorCodes(i).BuiltinCode() not in (
                    schema.BuiltinOperator.CUSTOM,
                    schema.BuiltinOperator.WHILE,
                    schema.BuiltinOperator.IF,
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
                input_shapes = {
                    name: tuple(meta["shape"])
                    for name, meta in runner.get_input_details().items()
                }
                for feed, want in zip(feeds, expected):
                    actual = runner(
                        **{
                            name: onnx_to_tfl_input(value, input_shapes[name], mode)
                            for name, value in feed.items()
                        }
                    )
                    for name, value in want.items():
                        converted = tfl_to_onnx_output(actual[name], value.shape, mode)
                        np.testing.assert_allclose(
                            converted, value, rtol=2.0e-5, atol=2.0e-5, err_msg=name
                        )
                row = dict(
                    mode=mode,
                    optimize=optimize,
                    delegate=delegate,
                    outputs=len(expected[0]),
                    input_sets=len(feeds),
                    status="PASS",
                )
                report.append(row)
                print(json.dumps(row), flush=True)
                del runner, runtime
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
