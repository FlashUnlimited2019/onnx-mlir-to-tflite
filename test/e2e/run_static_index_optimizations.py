#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check static index rewrites through both MLIR builds and the TFLite runtime.

Fixtures use the producer's TFL compatibility dialect, including integer forms
created internally by lowering. Flat entry tensors isolate the physical-layout
rewrites from ONNX boundary layout conversion. NumPy is the independent oracle.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import tensorflow as tf


@dataclass
class Value:
    name: str
    shape: tuple
    dtype: np.dtype

    @property
    def type(self):
        element = {
            np.dtype(np.float32): "f32",
            np.dtype(np.int32): "i32",
            np.dtype(np.int64): "i64",
            np.dtype(np.bool_): "i1",
        }[self.dtype]
        dimensions = "".join(f"{d}x" for d in self.shape)
        return f"tensor<{dimensions}{element}>"


class Graph:
    def __init__(self):
        self.lines, self.inputs, self.outputs, self.evaluations = [], [], [], []
        self.constants = {}
        self.counter = 0
        self.cases = 0

    def value(self, shape, dtype):
        result = Value(f"%v{self.counter}", tuple(shape), np.dtype(dtype))
        self.counter += 1
        return result

    def constant(self, array, dtype=np.int32):
        array = np.asarray(array, dtype=dtype)
        result = self.value(array.shape, array.dtype)

        def literal(a):
            if a.ndim:
                return "[" + ", ".join(literal(x) for x in a) + "]"
            if a.dtype == np.float32:
                return f"0x{int(a.view(np.uint32)):08X}"
            if a.dtype == np.bool_:
                return "true" if a.item() else "false"
            return str(a.item())

        self.lines.append(
            f"{result.name} = arith.constant dense<{literal(array)}> : {result.type}"
        )
        self.constants[result.name] = array
        return result

    def op(self, name, inputs, shape, dtype, compute, attrs=""):
        result = self.value(shape, dtype)
        operands = ", ".join(x.name for x in inputs)
        types = ", ".join(x.type for x in inputs)
        attr_text = f" {{{attrs}}}" if attrs else ""
        self.lines.append(
            f'{result.name} = "tfl.{name}"({operands}){attr_text} '
            f": ({types}) -> {result.type}"
        )
        self.evaluations.append((result, inputs, compute))
        return result

    def reshape(self, data, shape):
        target = self.constant(shape)
        return self.op(
            "reshape", [data, target], shape, data.dtype, lambda x, s: x.reshape(s)
        )

    def input(self, shape, dtype):
        elements = int(np.prod(shape))
        flat = Value(f"%arg{len(self.inputs)}", (elements,), np.dtype(dtype))
        self.inputs.append(flat)
        return flat if tuple(shape) == flat.shape else self.reshape(flat, shape)

    def output(self, value):
        flat = self.reshape(value, (int(np.prod(value.shape)),))
        self.outputs.append(flat)

    def concat(self, inputs, axis):
        shape = list(inputs[0].shape)
        shape[axis] = sum(v.shape[axis] for v in inputs)
        return self.op(
            "concatenation",
            inputs,
            shape,
            inputs[0].dtype,
            lambda *xs: np.concatenate(xs, axis=axis),
            f'axis = {axis} : i32, fused_activation_function = "NONE"',
        )

    def slice(self, value, begin, size):
        actual_size = [
            d - a if b == -1 else b for d, a, b in zip(value.shape, begin, size)
        ]

        def compute(x, a, b):
            return x[
                tuple(
                    slice(int(start), None if length == -1 else int(start + length))
                    for start, length in zip(a, b)
                )
            ]

        return self.op(
            "slice",
            [value, self.constant(begin), self.constant(size)],
            actual_size,
            value.dtype,
            compute,
        )

    def module(self):
        inputs = ", ".join(f"{v.name}: {v.type}" for v in self.inputs)
        result_types = ", ".join(v.type for v in self.outputs)
        body = "\n".join("    " + line for line in self.lines)
        results = ", ".join(v.name for v in self.outputs)
        return (
            "module {\n"
            f"  func.func @test({inputs}) -> ({result_types}) {{\n{body}\n"
            f"    return {results} : {result_types}\n  }}\n"
            '  "onnx.EntryPoint"() {func = @test} : () -> ()\n}\n'
        )

    def values(self, seed):
        rng = np.random.default_rng(seed)
        values = self.constants.copy()
        feed = {}
        for i, value in enumerate(self.inputs):
            if value.dtype == np.float32:
                array = rng.normal(size=value.shape).astype(np.float32)
                # Copy/selection rewrites must also preserve exceptional values
                # and the sign of zero; no floating arithmetic is exercised.
                if array.size >= 4:
                    array[:4] = [np.nan, np.inf, -np.inf, -0.0]
            elif value.dtype == np.bool_:
                array = rng.integers(0, 2, value.shape).astype(np.bool_)
            else:
                array = rng.integers(-100, 100, value.shape, dtype=value.dtype)
            values[value.name] = array
            feed[f"input{i}"] = array
        with np.errstate(over="ignore", invalid="ignore"):
            for result, inputs, compute in self.evaluations:
                array = compute(*(values[v.name] for v in inputs))
                assert array.shape == result.shape and array.dtype == result.dtype
                values[result.name] = array
        expected = {f"output{i}": values[v.name] for i, v in enumerate(self.outputs)}
        return feed, expected


def constant_cases(graph):
    for dtype in (np.int32, np.int64, np.bool_):
        for rank in range(1, 6):
            shape = (2,) * rank
            data = graph.constant(np.arange(2**rank).reshape(shape) % 3, dtype)
            repeats = [1] * (rank - 1) + [2]
            tiled_shape = shape[:-1] + (4,)
            tiled = graph.op(
                "tile", [data, graph.constant(repeats)], tiled_shape, dtype, np.tile
            )
            perm = list(reversed(range(rank)))
            transposed = graph.op(
                "transpose",
                [tiled, graph.constant(perm)],
                tuple(tiled_shape[i] for i in perm),
                dtype,
                np.transpose,
            )
            reverse = graph.op(
                "reverse_v2",
                [transposed, graph.constant([-1])],
                transposed.shape,
                dtype,
                lambda x, axes: np.flip(x, tuple(axes)),
            )
            graph.output(reverse)
            graph.cases += 1
    data = graph.constant([-2, 0, 2, np.iinfo(np.int64).max], np.int64)
    boolean = graph.op(
        "cast", [data], data.shape, np.bool_, lambda x: x.astype(np.bool_)
    )
    wide = graph.op(
        "cast", [boolean], data.shape, np.int64, lambda x: x.astype(np.int64)
    )
    graph.output(wide)
    for dtype in (np.int32, np.int64):
        data = graph.constant([-3, -1, 0, 2], dtype)
        rows = graph.reshape(data, (4, 1))
        add = graph.op(
            "add",
            [rows, graph.constant([0, 1, 2], dtype)],
            (4, 3),
            dtype,
            np.add,
            'fused_activation_function = "NONE"',
        )
        less = graph.op(
            "less", [add, graph.constant(0, dtype)], (4, 3), np.bool_, np.less
        )
        adjusted = graph.op(
            "add",
            [add, graph.constant(8, dtype)],
            (4, 3),
            dtype,
            np.add,
            'fused_activation_function = "NONE"',
        )
        selected = graph.op("select_v2", [less, adjusted, add], (4, 3), dtype, np.where)
        graph.output(selected)
    large = graph.constant([2147483648, -2147483649], np.int64)
    narrowed = graph.op("cast", [large], (2,), np.int32, lambda x: x.astype(np.int32))
    graph.output(narrowed)
    largest = graph.constant([2147483647, -2147483648], np.int32)
    overflow = graph.op(
        "add",
        [largest, graph.constant([1, -1])],
        (2,),
        np.int32,
        np.add,
        'fused_activation_function = "NONE"',
    )
    graph.output(overflow)
    graph.cases += 5


def window_cases(graph):
    for rank in range(1, 6):
        for axis in range(rank):
            for dtype in (np.float32, np.int32, np.int64, np.bool_):
                pieces = []
                for extent in (3, 5, 4):
                    shape = [2] * rank
                    shape[axis] = extent
                    pieces.append(graph.input(shape, dtype))
                merged = graph.concat(pieces, axis)
                # Keep another consumer/output to detect destructive rewrites
                # of a shared Concat instead of its individual Slice uses.
                graph.output(merged)
                for start, length in ((3, 5), (4, 2), (2, 3), (9, -1)):
                    begin, size = [0] * rank, [2] * rank
                    begin[axis], size[axis] = start, length
                    graph.output(graph.slice(merged, begin, size))
                    graph.cases += 1
                if rank > 1:
                    begin, size = [0] * rank, [2] * rank
                    begin[axis], size[axis] = 4, 2
                    other_axis = (axis + 1) % rank
                    begin[other_axis], size[other_axis] = 1, 1
                    graph.output(graph.slice(merged, begin, size))
                    graph.cases += 1


def mask_cases(graph):
    for dtype in (np.float32, np.int64):
        for batch in (1, 2):
            for step in (1, 2, 3, 4):
                for fill in (False, True):
                    history = graph.input((batch, 12), np.bool_)
                    suffix = graph.constant(np.full((batch, 12), fill), np.bool_)
                    merged = graph.concat([history, suffix], 1)
                    down = graph.op(
                        "strided_slice",
                        [
                            merged,
                            graph.constant([0, 0]),
                            graph.constant([batch, 24]),
                            graph.constant([1, step]),
                        ],
                        (batch, 24 // step),
                        np.bool_,
                        lambda x, a, b, c: x[
                            tuple(
                                slice(int(i), int(j), int(k))
                                for i, j, k in zip(a, b, c)
                            )
                        ],
                        "begin_mask = 0 : i32, end_mask = 0 : i32, ellipsis_mask = 0 : i32, "
                        "new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false",
                    )
                    tail = 12 // step
                    for mixed in (False, True):
                        part = graph.slice(down, [0, tail - int(mixed)], [batch, tail])
                        mask = graph.reshape(part, (batch, 1, 1, tail))
                        shape = (batch, 2, 3, tail)
                        mask = graph.op(
                            "broadcast_to",
                            [mask, graph.constant(shape)],
                            shape,
                            np.bool_,
                            lambda x, s: np.broadcast_to(x, s),
                        )
                        current = graph.input(shape, dtype)
                        zero = graph.constant(0, dtype)
                        branches = [current, zero] if fill else [zero, current]
                        selected = graph.op(
                            "select_v2", [mask, *branches], shape, dtype, np.where
                        )
                        graph.output(selected)
                        graph.cases += 1
    # A known condition does not justify dropping a necessary branch broadcast.
    data = graph.input((2,), np.float32)
    selected = graph.op(
        "select_v2",
        [graph.constant([False, False], np.bool_), data, graph.constant(7, np.float32)],
        (2,),
        np.float32,
        np.where,
    )
    graph.output(selected)
    graph.cases += 1


def command(args):
    result = subprocess.run(
        [str(a) for a in args], text=True, capture_output=True, timeout=180
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--onnx-mlir-opt", type=Path, default=repo / "build/bin/onnx-mlir-opt"
    )
    tf_tools = repo / "tensorflow/bazel-bin/tensorflow/compiler/mlir/lite"
    parser.add_argument("--litert-opt", type=Path, default=tf_tools / "litert-opt")
    parser.add_argument(
        "--flatbuffer-translate", type=Path, default=tf_tools / "flatbuffer_translate"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    graph = Graph()
    constant_cases(graph)
    window_cases(graph)
    mask_cases(graph)
    with tempfile.TemporaryDirectory(
        prefix="static-index-", dir=args.work_dir
    ) as directory:
        root = Path(directory)
        source = root / "input.mlir"
        source.write_text(graph.module())
        for mode in ("legacy", "experimental"):
            lowered = root / f"{mode}.mlir"
            command(
                [
                    args.onnx_mlir_opt,
                    source,
                    "--allow-unregistered-dialect",
                    f"--convert-onnx-to-tfl=layout-mode={mode}",
                    "--verify-each",
                    "-o",
                    lowered,
                ]
            )
            for optimize in (False, True):
                parsed = root / f"{mode}.{optimize}.mlir"
                flags = (
                    ["--tfl-optimize", "--canonicalize", "--cse"] if optimize else []
                )
                command(
                    [args.litert_opt, lowered, "--verify-each", *flags, "-o", parsed]
                )
                target = root / f"{mode}.{optimize}.tflite"
                command(
                    [
                        args.flatbuffer_translate,
                        "-mlir-to-tflite-flatbuffer",
                        parsed,
                        "-o",
                        target,
                        "-emit-builtin-tflite-ops=true",
                        "-emit-select-tf-ops=false",
                        "-emit-custom-ops=false",
                    ]
                )
                interpreter = tf.lite.Interpreter(model_path=str(target), num_threads=2)
                runner = interpreter.get_signature_runner("serving_default")
                for seed in (20260910, 20260911, 20260912):
                    feed, expected = graph.values(seed)
                    actual = runner(**feed)
                    assert set(actual) == set(expected)
                    for name, reference in expected.items():
                        candidate = actual[name]
                        assert candidate.dtype == reference.dtype
                        assert candidate.shape == reference.shape
                        if candidate.dtype == np.float32:
                            candidate = candidate.view(np.uint32)
                            reference = reference.view(np.uint32)
                        np.testing.assert_array_equal(
                            candidate, reference, err_msg=name
                        )
                print(
                    json.dumps(
                        {
                            "layout": mode,
                            "optimize": optimize,
                            "cases": graph.cases,
                            "outputs": len(graph.outputs),
                            "input_sets": 3,
                            "status": "PASS",
                        }
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
