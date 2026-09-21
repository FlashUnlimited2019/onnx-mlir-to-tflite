#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify typed Tile and LessOrEqual semantics across layouts and optimizers."""

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


def value(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def tile_graph():
    nodes, inputs, outputs, initializers = [], [], [], []
    shapes = [
        ([16], [2]),
        ([2, 3], [3, 2]),
        ([2, 3, 4], [2, 1, 3]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([2, 2, 2, 2], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
        ([2, 1, 3, 2, 4], [1, 3, 1, 2, 1]),
        ([1, 2, 1, 3, 4, 1], [1, 2, 1, 1, 2, 1]),
        ([1, 2, 1, 3, 1, 4, 1], [1, 2, 1, 1, 1, 2, 1]),
        ([1, 2, 1, 3, 1, 4, 1, 1], [1, 2, 1, 1, 1, 2, 1, 1]),
    ]
    for dtype in (
        TensorProto.FLOAT,
        TensorProto.INT64,
        TensorProto.INT32,
        TensorProto.BOOL,
    ):
        for index, (shape, repeats) in enumerate(shapes):
            prefix = f"type{dtype}_case{index}"
            source, result, flat, repeat, flat_shape = (
                f"{prefix}_{suffix}"
                for suffix in ("input", "result", "flat", "repeats", "flat_shape")
            )
            result_shape = [dim * count for dim, count in zip(shape, repeats)]
            elements = int(np.prod(result_shape))
            inputs.append(value(source, dtype, shape))
            # ONNX Reshape lowering currently accepts i64 but not i32.
            # Widen losslessly for the flattened check and retain the direct
            # i32 output to verify both the data type and its runtime shape.
            flat_dtype = TensorProto.INT64 if dtype == TensorProto.INT32 else dtype
            outputs.extend(
                [
                    value(result, dtype, result_shape),
                    value(flat, flat_dtype, [elements]),
                ]
            )
            initializers.extend(
                [
                    numpy_helper.from_array(np.array(repeats, np.int64), repeat),
                    numpy_helper.from_array(np.array([elements], np.int64), flat_shape),
                ]
            )
            nodes.append(helper.make_node("Tile", [source, repeat], [result]))
            flat_input = result
            if flat_dtype != dtype:
                flat_input = f"{prefix}_wide"
                nodes.append(
                    helper.make_node("Cast", [result], [flat_input], to=flat_dtype)
                )
            # Flattening can hide a wrong Tile shape while retaining
            # its incorrect data order. Check both outputs.
            nodes.append(helper.make_node("Reshape", [flat_input, flat_shape], [flat]))
    return helper.make_graph(nodes, "typed_tile", inputs, outputs, initializers)


def comparison_graph():
    nodes, inputs, outputs, initializers = [], [], [], []
    shapes = [
        ([16], [16]),
        ([1, 1], [1, 256]),
        ([2, 3, 4], [1, 4]),
        ([2, 3, 4, 5], [3, 1, 5]),
        ([2, 1, 3, 4, 5], [1, 3, 1, 5]),
    ]
    for dtype in (TensorProto.FLOAT, TensorProto.INT64):
        for index, (lhs_shape, rhs_shape) in enumerate(shapes):
            prefix = f"type{dtype}_case{index}"
            lhs, rhs, result = (f"{prefix}_{suffix}" for suffix in ("lhs", "rhs", "y"))
            inputs.extend([value(lhs, dtype, lhs_shape), value(rhs, dtype, rhs_shape)])
            result_shape = list(np.broadcast_shapes(lhs_shape, rhs_shape))
            outputs.append(value(result, TensorProto.BOOL, result_shape))
            nodes.append(helper.make_node("LessOrEqual", [lhs, rhs], [result]))

    inputs.append(value("feature", TensorProto.FLOAT, [1, 2, 3, 4]))
    initializers.extend(
        [
            numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
            numpy_helper.from_array(np.array([1, 2, 3, 4], np.int64), "repeats"),
        ]
    )
    nodes.extend(
        [
            helper.make_node("LessOrEqual", ["feature", "zero"], ["mask"]),
            helper.make_node("Tile", ["mask", "repeats"], ["tiled_mask"]),
            helper.make_node(
                "Cast", ["tiled_mask"], ["float_mask"], to=TensorProto.FLOAT
            ),
        ]
    )
    outputs.extend(
        [
            value("tiled_mask", TensorProto.BOOL, [1, 4, 9, 16]),
            value("float_mask", TensorProto.FLOAT, [1, 4, 9, 16]),
        ]
    )
    return helper.make_graph(nodes, "typed_comparison", inputs, outputs, initializers)


def input_data(metadata, phase, rng):
    dtype = {
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }[metadata.type]
    if dtype == np.float32:
        limit = np.finfo(dtype).max
        edges = [-np.inf, -limit, -3, -1, -0.0, 0.0, 1, 3, limit, np.inf, np.nan]
    elif dtype == np.bool_:
        edges = [True, False, False, True, False]
    else:
        limits = np.iinfo(dtype)
        edges = [limits.min, limits.min + 1, -1, 0, 1, limits.max - 1, limits.max]
        if dtype == np.int64:
            edges.extend([-(2**53) - 1, 2**53, 2**53 + 1, 2**31 + 1])
    if phase < 2:
        shift = phase if metadata.name.endswith("_rhs") else 0
        return np.resize(
            np.roll(np.array(edges, dtype=dtype), shift), int(np.prod(metadata.shape))
        ).reshape(metadata.shape)
    if dtype == np.float32:
        return rng.normal(size=metadata.shape).astype(dtype)
    if dtype == np.bool_:
        return rng.integers(0, 2, size=metadata.shape).astype(dtype)
    return rng.integers(-1000, 1000, size=metadata.shape, dtype=dtype)


def uses_nhwc(logical_shape, dtype, physical_shape):
    logical_shape, physical_shape = tuple(logical_shape), tuple(physical_shape)
    if dtype == np.float32 and len(logical_shape) == 4:
        nhwc = tuple(logical_shape[axis] for axis in (0, 2, 3, 1))
        # Nontrivial Tile and comparison retain legacy boundaries. Prefer
        # NHWC for the equal-sized-axis fixture, whose Tile is not folded.
        if physical_shape == nhwc:
            return True
        # An all-one Tile can be eliminated before layout analysis, leaving
        # a Reshape region with identity layout in experimental mode.
    assert physical_shape == logical_shape, (logical_shape, physical_shape)
    return False


def compare(source, target):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(source), options, providers=["CPUExecutionProvider"]
    )
    interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=1)
    declared_shapes = {
        detail["index"]: tuple(detail["shape"])
        for detail in interpreter.get_output_details()
    }
    interpreter.allocate_tensors()
    # Runtime Prepare may recompute Tile shapes and expose an invalid export
    # that MLIR verification and FlatBuffer roundtrip alone did not detect.
    for detail in interpreter.get_output_details():
        assert tuple(detail["shape"]) == declared_shapes[detail["index"]], detail[
            "name"
        ]
    runner = interpreter.get_signature_runner("serving_default")
    input_details, output_details = (
        runner.get_input_details(),
        runner.get_output_details(),
    )
    assert set(input_details) == {metadata.name for metadata in session.get_inputs()}
    assert set(output_details) == {metadata.name for metadata in session.get_outputs()}
    for phase in range(3):
        rng = np.random.default_rng(phase)
        feeds, physical_feeds = {}, {}
        for metadata in session.get_inputs():
            data = input_data(metadata, phase, rng)
            detail = input_details[metadata.name]
            assert data.dtype == detail["dtype"]
            feeds[metadata.name] = data
            if uses_nhwc(data.shape, data.dtype, detail["shape"]):
                data = data.transpose(0, 2, 3, 1)
            physical_feeds[metadata.name] = np.ascontiguousarray(data)
        expected = session.run(None, feeds)
        actual = runner(**physical_feeds)
        assert set(actual) == set(output_details)
        for metadata, reference in zip(session.get_outputs(), expected):
            candidate = actual[metadata.name]
            assert candidate.shape == tuple(output_details[metadata.name]["shape"])
            assert candidate.dtype == reference.dtype
            if uses_nhwc(reference.shape, reference.dtype, candidate.shape):
                candidate = candidate.transpose(0, 3, 1, 2)
            # Tile is an exact copy and comparison results are boolean;
            # tolerance-based float comparison would hide integer corruption.
            np.testing.assert_array_equal(candidate, reference, err_msg=metadata.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="typed-ops-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        for graph in (tile_graph(), comparison_graph()):
            source = root / f"{graph.name}.onnx"
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7
            )
            onnx.checker.check_model(model)
            onnx.save(model, source)
            for layout in ("legacy", "experimental"):
                for optimize in (False, True):
                    target = root / f"{graph.name}.{layout}.{optimize}.tflite"
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(args.converter.resolve()),
                            str(source),
                            "-o",
                            str(target),
                            "--layout-mode",
                            layout,
                            "--verify-each",
                            "--optimize-tfl" if optimize else "--no-optimize-tfl",
                        ],
                        text=True,
                        capture_output=True,
                        timeout=120,
                    )
                    if result.returncode:
                        raise RuntimeError(result.stdout + result.stderr)
                    compare(source, target)
                    print(
                        f"PASS {graph.name}: layout={layout}, optimize={optimize}, "
                        f"outputs={len(graph.output)}, input_sets=3",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
