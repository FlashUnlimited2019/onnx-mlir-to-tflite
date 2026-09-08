#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check named TFLite inference across layouts, optimizers and buffer storage."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


def models():
    def value(name, shape, dtype=TensorProto.FLOAT):
        return helper.make_tensor_value_info(name, dtype, shape)

    shape = [1, 2, 3, 4]
    yield helper.make_graph(
        [
            helper.make_node("Add", ["z/input:0", "a.bias"], ["sum"]),
            helper.make_node("Relu", ["sum"], ["relu"]),
            helper.make_node("Identity", ["sum"], ["alias"]),
        ],
        "named_multi_io",
        [
            value("z/input:0", shape),
            value("a.bias", [1, 2, 1, 1]),
            value("unused", [2]),
        ],
        [
            value("relu", shape),
            value("sum", shape),
            value("alias", shape),
            value("z/input:0", shape),
        ],
    )
    yield helper.make_graph(
        [],
        "typed_passthrough",
        [
            value("z_int64", [2, 3], TensorProto.INT64),
            value("a_bool", [2, 3], TensorProto.BOOL),
            value("int32", [2, 3], TensorProto.INT32),
        ],
        [
            value("a_bool", [2, 3], TensorProto.BOOL),
            value("z_int64", [2, 3], TensorProto.INT64),
            value("int32", [2, 3], TensorProto.INT32),
        ],
    )
    yield helper.make_graph(
        [],
        "constant_no_inputs",
        [],
        [value("constant", [2, 3])],
        [
            numpy_helper.from_array(
                np.arange(6, dtype=np.float32).reshape(2, 3), "constant"
            )
        ],
    )


def compare(source, target, use_buffer_offset):
    model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
    assert model.SignatureDefsLength() == 1
    signature = model.SignatureDefs(0)
    assert signature.SignatureKey() == b"serving_default"
    assert signature.SubgraphIndex() == 0
    graph = model.Subgraphs(0)
    inputs = {
        signature.Inputs(i).Name().decode(): signature.Inputs(i).TensorIndex()
        for i in range(signature.InputsLength())
    }
    outputs = {
        signature.Outputs(i).Name().decode(): signature.Outputs(i).TensorIndex()
        for i in range(signature.OutputsLength())
    }
    assert len(inputs) == signature.InputsLength() == graph.InputsLength()
    assert len(outputs) == signature.OutputsLength() == graph.OutputsLength()
    assert set(inputs.values()) == {
        graph.Inputs(i) for i in range(graph.InputsLength())
    }
    assert set(outputs.values()) == {
        graph.Outputs(i) for i in range(graph.OutputsLength())
    }
    if use_buffer_offset and source.stem == "constant_no_inputs":
        assert any(model.Buffers(i).Offset() > 1 for i in range(model.BuffersLength()))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(source), sess_options=options, providers=["CPUExecutionProvider"]
    )
    interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=1)
    interpreter.allocate_tensors()
    assert interpreter.get_signature_list() == {
        "serving_default": {"inputs": sorted(inputs), "outputs": sorted(outputs)}
    }
    assert set(inputs) == {item.name for item in session.get_inputs()}
    assert set(outputs) == {item.name for item in session.get_outputs()}
    details = {item["index"]: item for item in interpreter.get_input_details()}
    # Use a second interpreter to compare the named and index-based interfaces.
    named_interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=1)
    runner = named_interpreter.get_signature_runner("serving_default")
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        feeds, physical_feeds = {}, {}
        for metadata in session.get_inputs():
            detail = details[inputs[metadata.name]]
            if detail["dtype"] == np.bool_:
                data = rng.integers(0, 2, metadata.shape).astype(np.bool_)
            elif np.issubdtype(detail["dtype"], np.integer):
                data = rng.integers(-100, 100, metadata.shape, dtype=detail["dtype"])
            else:
                data = rng.normal(size=metadata.shape).astype(np.float32)
            feeds[metadata.name] = data
            physical = np.ascontiguousarray(
                onnx_to_tfl_input(data, tuple(detail["shape"]))
            )
            physical_feeds[metadata.name] = physical
            interpreter.set_tensor(detail["index"], physical)
        expected = session.run(list(outputs), feeds)
        interpreter.invoke()
        actual = runner(**physical_feeds)
        assert set(actual) == set(outputs)
        for (name, index), reference in zip(outputs.items(), expected):
            np.testing.assert_array_equal(actual[name], interpreter.get_tensor(index))
            converted = tfl_to_onnx_output(actual[name], reference.shape)
            assert (
                converted.dtype == reference.dtype
                and converted.shape == reference.shape
            )
            np.testing.assert_allclose(converted, reference, rtol=1e-6, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="signature-regression-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        for graph in models():
            source = root / f"{graph.name}.onnx"
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
            )
            onnx.checker.check_model(model)
            onnx.save(model, source)
            for layout in ("legacy", "experimental"):
                for optimize in (False, True):
                    for offset in (False, True):
                        target = (
                            root / f"{graph.name}.{layout}.{optimize}.{offset}.tflite"
                        )
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(args.converter.resolve()),
                                str(source),
                                "-o",
                                str(target),
                                "--layout-mode",
                                layout,
                                "--optimize-tfl" if optimize else "--no-optimize-tfl",
                                (
                                    "--use-buffer-offset"
                                    if offset
                                    else "--no-use-buffer-offset"
                                ),
                            ],
                            text=True,
                            capture_output=True,
                        )
                        if result.returncode:
                            raise RuntimeError(result.stdout + result.stderr)
                        compare(source, target, offset)
                        print(
                            f"PASS {graph.name}: layout={layout}, optimize={optimize}, "
                            f"buffer_offset={offset}, seeds=3",
                            flush=True,
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
