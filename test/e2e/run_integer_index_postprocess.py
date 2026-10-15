#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify integer Flatten, Gather, and bounded division without precision loss."""

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


def make_model():
    nodes, inputs, outputs, constants, feeds, expected = [], [], [], [], {}, {}

    def output(name, value, dtype):
        outputs.append(helper.make_tensor_value_info(name, dtype, value.shape))
        expected[name] = value

    for dtype, numpy_type in (
        (TensorProto.INT32, np.int32),
        (TensorProto.INT64, np.int64),
    ):
        for rank in range(1, 6):
            shape = (2, 3, 2, 3, 2)[:rank]
            name = f"data_{dtype}_{rank}"
            bounds = np.iinfo(numpy_type)
            data = np.resize(
                np.array(
                    [bounds.min, bounds.max, -1, 0, 1, 16777217], dtype=numpy_type
                ),
                shape,
            )
            inputs.append(helper.make_tensor_value_info(name, dtype, shape))
            feeds[name] = data
            for axis in range(rank + 1):
                result = f"{name}_flatten_{axis}"
                normalized_axis = axis - rank if axis < rank else axis
                nodes.append(
                    helper.make_node("Flatten", [name], [result], axis=normalized_axis)
                )
                output(
                    result,
                    data.reshape(
                        int(np.prod(shape[:axis])), int(np.prod(shape[axis:]))
                    ),
                    dtype,
                )
            for axis in range(rank):
                for constant in (False, True):
                    index_name = f"{name}_indices_{axis}_{int(constant)}"
                    indices = np.array([[0, -1], [1, -2]], dtype=numpy_type)
                    if constant:
                        constants.append(numpy_helper.from_array(indices, index_name))
                    else:
                        inputs.append(
                            helper.make_tensor_value_info(
                                index_name, dtype, indices.shape
                            )
                        )
                        feeds[index_name] = indices
                    result = index_name + "_gather"
                    nodes.append(
                        helper.make_node(
                            "Gather", [name, index_name], [result], axis=axis - rank
                        )
                    )
                    output(result, np.take(data, indices, axis=axis), dtype)
    # A widening Cast proves that narrowing the dividend is lossless.
    dividend = np.array(
        [[-2147483648, -16777217, -7], [2147483647, 16777217, 7]], dtype=np.int32
    )
    inputs.append(
        helper.make_tensor_value_info("dividend", TensorProto.INT32, dividend.shape)
    )
    feeds["dividend"] = dividend
    nodes.append(helper.make_node("Cast", ["dividend"], ["wide"], to=TensorProto.INT64))
    for divisor in (1, 2, 3, 80, 2147483647):
        name = f"divisor_{divisor}"
        constants.append(numpy_helper.from_array(np.array(divisor, np.int64), name))
        result = name + "_quotient"
        nodes.append(helper.make_node("Div", ["wide", name], [result]))
        values = dividend.astype(np.int64)
        output(result, np.sign(values) * (np.abs(values) // divisor), TensorProto.INT64)
    model = helper.make_model(
        helper.make_graph(
            nodes, "integer_index_postprocess", inputs, outputs, constants
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    return model, feeds, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build/bin/onnx-to-tflite",
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.work_dir / "integer_index_postprocess.onnx"
    model, feed, expected = make_model()
    onnx.save(model, source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    reference = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    for value, meta in zip(reference.run(None, feed), reference.get_outputs()):
        np.testing.assert_array_equal(value, expected[meta.name])
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
            fb = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            assert all(
                fb.OperatorCodes(i).BuiltinCode()
                not in (
                    schema.BuiltinOperator.CUSTOM,
                    schema.BuiltinOperator.WHILE,
                    schema.BuiltinOperator.IF,
                )
                for i in range(fb.OperatorCodesLength())
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
                actual = runner(**feed)
                assert actual.keys() == expected.keys()
                for name, value in expected.items():
                    assert actual[name].dtype == value.dtype
                    np.testing.assert_array_equal(actual[name], value, err_msg=name)
                row = dict(
                    mode=mode,
                    optimize=optimize,
                    delegate=delegate,
                    outputs=len(expected),
                    status="PASS",
                )
                report.append(row)
                print(json.dumps(row), flush=True)
                del runner, runtime
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
