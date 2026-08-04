#!/usr/bin/env python3
"""Compare ONNX Runtime and TFLite with rank-4 NCHW/NHWC adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    return parser.parse_args()


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
    session_options = ort.SessionOptions()
    # This opset-7 model declares initializers as graph inputs. They remain
    # constants for this comparison; suppress the repetitive optimization
    # warnings so the numerical report stays readable.
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_inputs: dict[str, np.ndarray] = {}
    for metadata in session.get_inputs():
        if metadata.type != "tensor(float)":
            raise ValueError(
                f"only FP32 inputs are supported: {metadata.name} {metadata.type}"
            )
        if any(not isinstance(dim, int) or dim <= 0 for dim in metadata.shape):
            raise ValueError(
                f"input must have a static positive shape: "
                f"{metadata.name} {metadata.shape}"
            )
        ort_inputs[metadata.name] = rng.normal(
            args.mean, args.stddev, metadata.shape
        ).astype(np.float32)

    ort_outputs = session.run(None, ort_inputs)
    interpreter = tf.lite.Interpreter(
        model_path=str(args.tflite), num_threads=args.num_threads
    )
    interpreter.allocate_tensors()
    tfl_inputs = interpreter.get_input_details()
    if len(tfl_inputs) != len(session.get_inputs()):
        raise AssertionError(
            f"input count differs: ONNX={len(session.get_inputs())}, "
            f"TFLite={len(tfl_inputs)}"
        )

    print(
        f"random input: seed={args.seed}, normal(mean={args.mean}, "
        f"stddev={args.stddev})"
    )
    for ort_metadata, tfl_metadata in zip(session.get_inputs(), tfl_inputs):
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
    for metadata, expected, actual in zip(
        session.get_outputs(), ort_outputs, tfl_outputs
    ):
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
            print(
                f"  FAIL: cosine {cosine} is below {args.min_cosine}"
            )
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
