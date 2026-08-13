#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare ONNX Runtime and TFLite Interpreter outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tflite", required=True, type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def report_mismatch(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    abs_error = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    relative = abs_error / np.maximum(np.abs(expected.astype(np.float64)), 1e-30)
    index = np.unravel_index(int(np.argmax(abs_error)), abs_error.shape)
    rel_index = np.unravel_index(int(np.argmax(relative)), relative.shape)
    print(f"output {name} mismatch", file=sys.stderr)
    print(f"  max absolute error: {abs_error[index]} at {index}", file=sys.stderr)
    print(f"  max relative error: {relative[rel_index]} at {rel_index}", file=sys.stderr)
    print(f"  ONNX value at max abs: {expected[index]}", file=sys.stderr)
    print(f"  TFLite value at max abs: {actual[index]}", file=sys.stderr)
    print(f"  ONNX output:\n{expected}", file=sys.stderr)
    print(f"  TFLite output:\n{actual}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_inputs: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        if any(not isinstance(dim, int) or dim <= 0 for dim in item.shape):
            raise ValueError(f"test expects a static positive input shape: {item.name} {item.shape}")
        if item.type != "tensor(float)":
            raise ValueError(f"test only supports FP32 inputs: {item.name} {item.type}")
        ort_inputs[item.name] = rng.normal(0.0, 0.5, item.shape).astype(np.float32)
    ort_outputs = session.run(None, ort_inputs)

    interpreter = tf.lite.Interpreter(model_path=str(args.tflite))
    interpreter.allocate_tensors()
    tfl_inputs = interpreter.get_input_details()
    if len(tfl_inputs) != len(session.get_inputs()):
        raise AssertionError(
            f"input count differs: ONNX={len(session.get_inputs())}, TFLite={len(tfl_inputs)}"
        )
    for ort_input, tfl_input in zip(session.get_inputs(), tfl_inputs):
        value = ort_inputs[ort_input.name]
        if tuple(tfl_input["shape"]) != value.shape:
            raise AssertionError(
                f"input shape differs for {ort_input.name}: "
                f"ONNX={value.shape}, TFLite={tuple(tfl_input['shape'])}"
            )
        interpreter.set_tensor(tfl_input["index"], value)
    interpreter.invoke()
    tfl_outputs = [
        interpreter.get_tensor(detail["index"]) for detail in interpreter.get_output_details()
    ]

    if len(ort_outputs) != len(tfl_outputs):
        raise AssertionError(
            f"output count differs: ONNX={len(ort_outputs)}, TFLite={len(tfl_outputs)}"
        )
    failed = False
    for metadata, expected, actual in zip(session.get_outputs(), ort_outputs, tfl_outputs):
        if expected.shape != actual.shape:
            print(
                f"output {metadata.name} shape differs: ONNX={expected.shape}, "
                f"TFLite={actual.shape}",
                file=sys.stderr,
            )
            failed = True
            continue
        if expected.dtype != actual.dtype:
            print(
                f"output {metadata.name} dtype differs: ONNX={expected.dtype}, "
                f"TFLite={actual.dtype}",
                file=sys.stderr,
            )
            failed = True
            continue
        if not np.allclose(expected, actual, rtol=args.rtol, atol=args.atol):
            report_mismatch(metadata.name, expected, actual)
            failed = True
    if failed:
        return 1
    print(
        f"PASS: {args.onnx.name} ({len(ort_outputs)} output(s), "
        f"rtol={args.rtol}, atol={args.atol})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
