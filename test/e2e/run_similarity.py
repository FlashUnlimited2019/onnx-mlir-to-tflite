#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare ONNX Runtime and TFLite with rank-4 NCHW/NHWC adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tflite", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--stddev", type=float, default=0.5)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-relative-euclidean", type=float, default=1e-3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--onnx-reference",
        action="store_true",
        help=(
            "use ONNX's ReferenceEvaluator when ONNX Runtime does not "
            "implement an operator in the model"
        ),
    )
    parser.add_argument(
        "--canonicalize-unsorted-topk",
        action="store_true",
        help=(
            "compare sorted=0 TopK nodes using the valid deterministic "
            "sorted ordering emitted by TFLite TopKV2"
        ),
    )
    parser.add_argument(
        "--integer-input-range",
        action="append",
        default=[],
        metavar="NAME:LOW:HIGH",
        help=(
            "generate an integer input in the half-open interval [LOW, HIGH); "
            "repeat once for every integer graph input"
        ),
    )
    return parser.parse_args()


def parse_integer_input_ranges(specifications: list[str]) -> dict[str, tuple[int, int]]:
    ranges: dict[str, tuple[int, int]] = {}
    for specification in specifications:
        parts = specification.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "integer input range must have the form NAME:LOW:HIGH: "
                f"{specification}"
            )
        name, low_text, high_text = parts
        low, high = int(low_text), int(high_text)
        if not name or low >= high:
            raise ValueError(f"invalid integer input range: {specification}")
        ranges[name] = (low, high)
    return ranges


def onnx_to_tfl_input(value: np.ndarray) -> np.ndarray:
    """Rank-4 activations are NHWC in TFLite; all other ranks are unchanged."""
    if value.ndim == 4:
        return np.transpose(value, (0, 2, 3, 1))
    return value


def tfl_to_onnx_output(value: np.ndarray) -> np.ndarray:
    if value.ndim == 4:
        return np.transpose(value, (0, 3, 1, 2))
    return value


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    integer_ranges = parse_integer_input_ranges(args.integer_input_range)
    session_options = ort.SessionOptions()
    # This opset-7 model declares initializers as graph inputs. They remain
    # constants for this comparison; suppress the repetitive optimization
    # warnings so the numerical report stays readable.
    session_options.log_severity_level = 3
    onnx_model: str | bytes = str(args.onnx)
    if args.canonicalize_unsorted_topk:
        model = onnx.load(args.onnx)
        canonicalized = 0
        for node in model.graph.node:
            if node.op_type != "TopK":
                continue
            sorted_attribute = next(
                (
                    attribute
                    for attribute in node.attribute
                    if attribute.name == "sorted"
                ),
                None,
            )
            if sorted_attribute is not None and sorted_attribute.i == 0:
                sorted_attribute.i = 1
                canonicalized += 1
        if canonicalized == 0:
            raise ValueError(
                "--canonicalize-unsorted-topk found no TopK node with sorted=0"
            )
        print(
            f"canonicalized {canonicalized} sorted=0 TopK node(s) to the "
            "equivalent deterministic sorted ordering"
        )
        onnx_model = model.SerializeToString()
    if args.onnx_reference:
        from onnx.reference import ReferenceEvaluator

        model = (
            onnx.load_from_string(onnx_model)
            if isinstance(onnx_model, bytes)
            else onnx.load(args.onnx)
        )
        initializer_names = {
            initializer.name for initializer in model.graph.initializer
        }
        type_names = {
            onnx.TensorProto.FLOAT: "tensor(float)",
            onnx.TensorProto.INT32: "tensor(int32)",
            onnx.TensorProto.INT64: "tensor(int64)",
        }

        def metadata(value_info: onnx.ValueInfoProto) -> SimpleNamespace:
            tensor_type = value_info.type.tensor_type
            return SimpleNamespace(
                name=value_info.name,
                type=type_names.get(
                    tensor_type.elem_type, f"tensor({tensor_type.elem_type})"
                ),
                shape=[dimension.dim_value for dimension in tensor_type.shape.dim],
            )

        input_metadata = [
            metadata(value_info)
            for value_info in model.graph.input
            if value_info.name not in initializer_names
        ]
        output_metadata = [metadata(value_info) for value_info in model.graph.output]
        evaluator = ReferenceEvaluator(model)
        run_onnx = lambda feeds: evaluator.run(None, feeds)
        baseline_name = "ONNX ReferenceEvaluator"
    else:
        session = ort.InferenceSession(
            onnx_model,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        input_metadata = session.get_inputs()
        output_metadata = session.get_outputs()
        run_onnx = lambda feeds: session.run(None, feeds)
        baseline_name = "ONNX Runtime"
    ort_inputs: dict[str, np.ndarray] = {}
    for metadata in input_metadata:
        if metadata.type not in {"tensor(float)", "tensor(int32)", "tensor(int64)"}:
            raise ValueError(
                f"only FP32, i32, and i64 inputs are supported: "
                f"{metadata.name} {metadata.type}"
            )
        if any(not isinstance(dim, int) or dim <= 0 for dim in metadata.shape):
            raise ValueError(
                f"input must have a static positive shape: "
                f"{metadata.name} {metadata.shape}"
            )
        if metadata.type == "tensor(float)":
            ort_inputs[metadata.name] = rng.normal(
                args.mean, args.stddev, metadata.shape
            ).astype(np.float32)
            continue
        if metadata.name not in integer_ranges:
            raise ValueError(
                f"integer input {metadata.name} requires "
                "--integer-input-range NAME:LOW:HIGH"
            )
        low, high = integer_ranges[metadata.name]
        dtype = np.int32 if metadata.type == "tensor(int32)" else np.int64
        ort_inputs[metadata.name] = rng.integers(low, high, metadata.shape, dtype=dtype)

    ort_outputs = run_onnx(ort_inputs)
    interpreter = tf.lite.Interpreter(
        model_path=str(args.tflite), num_threads=args.num_threads
    )
    interpreter.allocate_tensors()
    tfl_inputs = interpreter.get_input_details()
    if len(tfl_inputs) != len(input_metadata):
        raise AssertionError(
            f"input count differs: ONNX={len(input_metadata)}, "
            f"TFLite={len(tfl_inputs)}"
        )

    print(
        f"random input: seed={args.seed}, normal(mean={args.mean}, "
        f"stddev={args.stddev}); baseline={baseline_name}"
    )
    for ort_metadata, tfl_metadata in zip(input_metadata, tfl_inputs):
        onnx_value = ort_inputs[ort_metadata.name]
        tfl_value = onnx_to_tfl_input(onnx_value)
        expected_shape = tuple(int(dim) for dim in tfl_metadata["shape"])
        if tfl_value.shape != expected_shape:
            raise AssertionError(
                f"input shape differs for {ort_metadata.name}: "
                f"ONNX={onnx_value.shape}, converted={tfl_value.shape}, "
                f"TFLite={expected_shape}"
            )
        if tfl_value.dtype != tfl_metadata["dtype"]:
            raise AssertionError(
                f"input dtype differs for {ort_metadata.name}: "
                f"ONNX={tfl_value.dtype}, TFLite={tfl_metadata['dtype']}"
            )
        interpreter.set_tensor(tfl_metadata["index"], tfl_value)
        print(
            f"input {ort_metadata.name}: ONNX {onnx_value.shape} -> "
            f"TFLite {tfl_value.shape}"
        )

    interpreter.invoke()
    tfl_outputs = [
        tfl_to_onnx_output(interpreter.get_tensor(metadata["index"]))
        for metadata in interpreter.get_output_details()
    ]
    if len(ort_outputs) != len(tfl_outputs):
        raise AssertionError(
            f"output count differs: ONNX={len(ort_outputs)}, "
            f"TFLite={len(tfl_outputs)}"
        )

    failed = False
    for metadata, expected, actual in zip(output_metadata, ort_outputs, tfl_outputs):
        if expected.shape != actual.shape:
            raise AssertionError(
                f"output {metadata.name} shape differs: ONNX={expected.shape}, "
                f"TFLite converted={actual.shape}"
            )
        if expected.dtype != actual.dtype:
            raise AssertionError(
                f"output {metadata.name} dtype differs: ONNX={expected.dtype}, "
                f"TFLite={actual.dtype}"
            )

        reference = expected.astype(np.float64).ravel()
        candidate = actual.astype(np.float64).ravel()
        difference = candidate - reference
        reference_norm = float(np.linalg.norm(reference))
        candidate_norm = float(np.linalg.norm(candidate))
        euclidean = float(np.linalg.norm(difference))
        relative_euclidean = euclidean / max(reference_norm, np.finfo(np.float64).tiny)
        denominator = reference_norm * candidate_norm
        cosine = (
            float(np.dot(reference, candidate) / denominator)
            if denominator != 0.0
            else float(reference_norm == candidate_norm)
        )
        max_abs = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        max_index = int(np.argmax(np.abs(difference)))

        print(f"output {metadata.name}:")
        print(f"  shape: {expected.shape}, dtype: {expected.dtype}")
        print(f"  cosine similarity: {cosine:.16g}")
        print(f"  euclidean distance: {euclidean:.16g}")
        print(f"  relative euclidean distance: {relative_euclidean:.16g}")
        print(f"  RMSE: {rmse:.16g}")
        print(
            f"  max absolute error: {max_abs:.12g} at flat index {max_index} "
            f"(ONNX={reference[max_index]:.12g}, "
            f"TFLite={candidate[max_index]:.12g})"
        )
        if cosine < args.min_cosine:
            print(f"  FAIL: cosine {cosine} is below {args.min_cosine}")
            failed = True
        if relative_euclidean > args.max_relative_euclidean:
            print(
                f"  FAIL: relative Euclidean {relative_euclidean} exceeds "
                f"{args.max_relative_euclidean}"
            )
            failed = True

    if failed:
        return 1
    print(
        f"PASS: cosine >= {args.min_cosine} and relative Euclidean <= "
        f"{args.max_relative_euclidean}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
