#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare static contrib lowering with ONNX Runtime and independent NumPy."""

import argparse
import copy
import itertools
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


class Graph:
    def __init__(self):
        self.inputs, self.outputs, self.info, self.nodes, self.initializers = (
            [],
            [],
            [],
            [],
            [],
        )
        self.constants, self.evaluations = {}, []
        self.serial = 0
        self.cases = 0

    def name(self):
        self.serial += 1
        return f"v{self.serial}"

    def constant(self, value):
        name = self.name()
        value = np.asarray(value)
        self.initializers.append(numpy_helper.from_array(value, name))
        self.constants[name] = value
        return name

    def node(self, op, inputs, shapes, evaluate, domain="", **attrs):
        names = [self.name() for _ in shapes]
        self.nodes.append(helper.make_node(op, inputs, names, domain=domain, **attrs))
        self.info.extend(
            helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
            for n, s in zip(names, shapes)
        )
        self.evaluations.append((names, inputs, evaluate))
        return names

    def reshape(self, data, shape):
        return self.node(
            "Reshape",
            [data, self.constant(np.array(shape, np.int64))],
            [shape],
            lambda x, s: [x.reshape(s)],
        )[0]

    def input(self, shape):
        name = self.name()
        self.inputs.append(
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, [int(np.prod(shape))]
            )
        )
        return self.reshape(name, shape)

    def output(self, value, shape):
        count = int(np.prod(shape))
        name = self.reshape(value, [count])
        self.outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [count])
        )

    def model(self):
        graph = helper.make_graph(
            self.nodes,
            "static_contrib",
            self.inputs,
            self.outputs,
            self.initializers,
            value_info=self.info,
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        model.ir_version = 9
        return model

    def reference(self, seed):
        rng = np.random.default_rng(seed)
        feed = {
            v.name: rng.normal(
                0, 0.5, (v.type.tensor_type.shape.dim[0].dim_value,)
            ).astype(np.float32)
            for v in self.inputs
        }
        values = {**self.constants, **feed}
        for names, inputs, evaluate in self.evaluations:
            outputs = evaluate(*(values[n] if n else None for n in inputs))
            values.update(zip(names, outputs))
        return feed, {v.name: values[v.name] for v in self.outputs}


def rms(x, gamma, epsilon):
    mean_square = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    inverse = np.float32(1) / np.sqrt(mean_square + np.float32(epsilon))
    return x * inverse * gamma


def normalization_cases():
    graph = Graph()
    for rank, positive_axis in itertools.product(range(2, 6), (False, True)):
        shape = [2] * (rank - 1) + [6]
        data = graph.input(shape)
        gamma = graph.constant(np.linspace(0.7, 1.2, 6, dtype=np.float32))
        result = graph.node(
            "SimplifiedLayerNormalization",
            [data, gamma],
            [shape],
            lambda x, g: [rms(x, g, 1e-5)],
            axis=rank - 1 if positive_axis else -1,
            epsilon=1e-5,
            stash_type=1,
        )[0]
        graph.output(result, shape)
        graph.cases += 1
    for rank, bias in itertools.product((2, 3), (False, True)):
        shape = [2] * (rank - 1) + [6]
        inputs = [
            graph.input(shape),
            graph.input(shape),
            graph.constant(np.linspace(0.7, 1.2, 6, dtype=np.float32)),
        ]
        if bias:
            inputs.append(graph.constant(np.linspace(-0.2, 0.3, 6, dtype=np.float32)))

        def evaluate(x, skip, gamma, bias=None):
            summed = x + skip
            if bias is not None:
                summed = summed + bias
            return [rms(summed, gamma, 1e-5)]

        result = graph.node(
            "SkipSimplifiedLayerNormalization",
            inputs,
            [shape],
            evaluate,
            domain="com.microsoft",
            epsilon=1e-5,
        )[0]
        graph.output(result, shape)
        graph.cases += 1
    # Small magnitudes distinguish the two operators' different defaults.
    for skip, epsilon in itertools.product((False, True), (None, 0.0)):
        shape = [2, 6]
        x = graph.node(
            "Mul",
            [graph.input(shape), graph.constant(np.float32(1e-4))],
            [shape],
            lambda x, factor: [x * factor],
        )[0]
        gamma = graph.input([6])
        effective_epsilon = (1e-12 if skip else 1e-5) if epsilon is None else epsilon
        inputs = [x, x, gamma, ""] if skip else [x, gamma]

        def evaluate(*values, skip=skip, epsilon=effective_epsilon):
            data = values[0] + values[1] if skip else values[0]
            return [rms(data, values[2 if skip else 1], epsilon)]

        result = graph.node(
            (
                "SkipSimplifiedLayerNormalization"
                if skip
                else "SimplifiedLayerNormalization"
            ),
            inputs,
            [shape],
            evaluate,
            domain="com.microsoft" if skip else "",
            **({} if epsilon is None else {"epsilon": epsilon}),
        )[0]
        graph.output(result, shape)
        graph.cases += 1
    return graph


def rope_reference(x, positions, cos, sin, heads, dimension, interleaved):
    original = x.shape
    bsnh = (
        x.reshape(x.shape[0], x.shape[1], heads, -1)
        if x.ndim == 3
        else x.transpose(0, 2, 1, 3)
    )
    batch, sequence, _, width = bsnh.shape
    dimension = dimension or width
    if positions.ndim < 2:
        positions = np.broadcast_to(
            positions.reshape(1, 1) + np.arange(sequence), (batch, sequence)
        )
    c, s = cos[positions][:, :, None, :], sin[positions][:, :, None, :]
    result = bsnh.copy()
    first = np.arange(dimension // 2) * (2 if interleaved else 1)
    second = first + (1 if interleaved else dimension // 2)
    a, b = bsnh[..., first], bsnh[..., second]
    result[..., first] = a * c - b * s
    result[..., second] = a * s + b * c
    return result.reshape(original) if x.ndim == 3 else result.transpose(0, 2, 1, 3)


def rotary_cases():
    graph = Graph()
    for rank, interleaved, dimension, position_form in itertools.product(
        (3, 4), (0, 1), (0, 4), (0, 1, 2)
    ):
        batch, sequence, heads, width = 2, 3, 3, 8
        shape = (
            [batch, sequence, heads * width]
            if rank == 3
            else [batch, heads, sequence, width]
        )
        actual_dimension = dimension or width
        angles = np.arange(12, dtype=np.float32)[:, None] * np.linspace(
            0.01, 0.2, actual_dimension // 2, dtype=np.float32
        )
        positions = (
            np.array(2, np.int64)
            if position_form == 0
            else (
                np.array([2], np.int64)
                if position_form == 1
                else np.array([[5, 2, 5], [0, 8, 1]], np.int64)
            )
        )
        inputs = [
            graph.input(shape),
            graph.constant(positions),
            graph.constant(np.cos(angles)),
            graph.constant(np.sin(angles)),
        ]
        attrs = {"interleaved": interleaved}
        if dimension:
            attrs.update(num_heads=heads, rotary_embedding_dim=dimension)

        def evaluate(x, p, c, s, h=heads, d=dimension, inter=interleaved):
            return [rope_reference(x, p, c, s, h, d, inter)]

        result = graph.node(
            "RotaryEmbedding",
            inputs,
            [shape],
            evaluate,
            domain="com.microsoft",
            **attrs,
        )[0]
        graph.output(result, shape)
        graph.cases += 1
    # Runtime caches are supported; only positions must be compile-time known.
    shape = [1, 2, 8]
    inputs = [
        graph.input(shape),
        graph.constant(np.array([[1, 0]], np.int64)),
        graph.input([4, 2]),
        graph.input([4, 2]),
    ]
    result = graph.node(
        "RotaryEmbedding",
        inputs,
        [shape],
        lambda x, p, c, s: [rope_reference(x, p, c, s, 2, 4, 1)],
        domain="com.microsoft",
        num_heads=2,
        rotary_embedding_dim=4,
        interleaved=1,
    )[0]
    graph.output(result, shape)
    graph.cases += 1
    return graph


def attention_reference(q, k, v, heads, kv_heads, scale):
    batch, sequence, hidden = q.shape
    width = hidden // heads
    q = q.reshape(batch, sequence, heads, width).transpose(0, 2, 1, 3)
    pk = k.reshape(batch, sequence, kv_heads, width).transpose(0, 2, 1, 3)
    pv = v.reshape(batch, sequence, kv_heads, width).transpose(0, 2, 1, 3)
    k = np.repeat(pk, heads // kv_heads, axis=1)
    v = np.repeat(pv, heads // kv_heads, axis=1)
    logits = (q @ k.swapaxes(-1, -2)) * np.float32(scale or 1 / np.sqrt(width))
    logits = np.where(
        np.tril(np.ones((sequence, sequence), bool)), logits, np.float32(-1e20)
    )
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    y = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, sequence, hidden)
    return [y, pk, pv]


def attention_cases():
    graph = Graph()
    for packed, batch, heads, kv_heads in itertools.product(
        (False, True), (1, 2), (2, 4), (1, 2)
    ):
        sequence, width = 3, 8
        shape = [batch, sequence, heads * width]
        kv_shape = [batch, sequence, kv_heads * width]
        cache_shape = [batch, kv_heads, sequence, width]
        if packed:
            inputs = [
                graph.input([batch, sequence, (heads + 2 * kv_heads) * width]),
                "",
                "",
                "",
                "",
            ]
        else:
            inputs = [
                graph.input(shape),
                graph.input(kv_shape),
                graph.input(kv_shape),
                "",
                "",
            ]
        inputs += [
            graph.constant(np.full((batch,), sequence - 1, np.int32)),
            graph.constant(np.array(sequence, np.int32)),
        ]
        scale = 0.25

        def evaluate(
            q, k, v, pk, pv, lengths, total, h=heads, kv=kv_heads, packed=packed
        ):
            if packed:
                q, k, v = np.split(q, [h * width, (h + kv) * width], axis=-1)
            return attention_reference(q, k, v, h, kv, scale)

        shapes = [shape, cache_shape, cache_shape]
        results = graph.node(
            "GroupQueryAttention",
            inputs,
            shapes,
            evaluate,
            domain="com.microsoft",
            num_heads=heads,
            kv_num_heads=kv_heads,
            scale=scale,
        )
        for result, result_shape in zip(results, shapes):
            graph.output(result, result_shape)
        graph.cases += 1
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(work)
    env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env["PATH"]
    comparisons = 0
    fixtures = [
        ("norm", normalization_cases()),
        ("rope", rotary_cases()),
        ("gqa", attention_cases()),
    ]
    for name, graph in fixtures:
        source = work / f"{name}.onnx"
        onnx.save(graph.model(), source)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(source), opts, providers=["CPUExecutionProvider"]
        )
        references = []
        for seed in (13, 37, 59):
            feed, independent = graph.reference(seed)
            ort_outputs = dict(
                zip([o.name for o in session.get_outputs()], session.run(None, feed))
            )
            for key in independent:
                np.testing.assert_allclose(
                    ort_outputs[key], independent[key], rtol=2e-5, atol=3e-6
                )
            references.append((feed, independent, ort_outputs))
        for layout, optimize in itertools.product(
            ("legacy", "experimental"), (False, True)
        ):
            target = work / f"{name}.{layout}.{int(optimize)}.tflite"
            command = [
                str(args.converter.resolve()),
                str(source),
                "-o",
                str(target),
                "--layout-mode",
                layout,
                "--verify-each",
                "--optimize-tfl" if optimize else "--no-optimize-tfl",
            ]
            with target.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            model = schema.Model.GetRootAsModel(target.read_bytes(), 0)
            forbidden = {
                schema.BuiltinOperator.CUSTOM,
                schema.BuiltinOperator.WHILE,
                schema.BuiltinOperator.IF,
            }
            assert model.SubgraphsLength() == 1
            assert all(
                max(
                    model.OperatorCodes(i).BuiltinCode(),
                    model.OperatorCodes(i).DeprecatedBuiltinCode(),
                )
                not in forbidden
                for i in range(model.OperatorCodesLength())
            )
            for delegated in (False, True):
                resolver = tf.lite.experimental.OpResolverType
                runtime = tf.lite.Interpreter(
                    model_path=str(target),
                    num_threads=1,
                    experimental_op_resolver_type=(
                        resolver.AUTO
                        if delegated
                        else resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                    ),
                )
                run = runtime.get_signature_runner("serving_default")
                for feed, independent, ort_outputs in references:
                    actual = run(**feed)
                    for key in independent:
                        np.testing.assert_allclose(
                            actual[key],
                            independent[key],
                            rtol=2e-5,
                            atol=3e-6,
                            err_msg=f"{target.name} {key}",
                        )
                        np.testing.assert_allclose(
                            actual[key], ort_outputs[key], rtol=2e-5, atol=3e-6
                        )
                        comparisons += 1
            print("PASS", target.name, graph.cases, "cases", flush=True)
    rejected = 0
    invalid_models = []
    for family, attribute, bad_value in (
        ("norm", "axis", 0),
        ("norm", "stash_type", 10),
        ("norm", "epsilon", -1.0),
        ("rope", "interleaved", 2),
        ("rope", "scale", 2.0),
        ("rope", "is_packed_batching", 1),
        ("rope", "rotary_embedding_dim", 3),
        ("rope", "num_heads", 5),
        ("gqa", "do_rotary", 1),
        ("gqa", "qk_output", 1),
        ("gqa", "local_window_size", 2),
    ):
        model = copy.deepcopy(dict(fixtures)[family].model())
        node = next(
            n
            for n in model.graph.node
            if n.op_type
            in (
                "SimplifiedLayerNormalization",
                "RotaryEmbedding",
                "GroupQueryAttention",
            )
        )
        kept = [a for a in node.attribute if a.name != attribute]
        del node.attribute[:]
        node.attribute.extend(kept + [helper.make_attribute(attribute, bad_value)])
        invalid_models.append((f"{family}_{attribute}", model))
    for value in (-1, 11, np.iinfo(np.int64).max):
        model = copy.deepcopy(dict(fixtures)["rope"].model())
        node = next(n for n in model.graph.node if n.op_type == "RotaryEmbedding")
        position = next(i for i in model.graph.initializer if i.name == node.input[1])
        position.CopyFrom(
            numpy_helper.from_array(np.array(value, np.int64), position.name)
        )
        invalid_models.append((f"rope_position_{value}", model))
    model = copy.deepcopy(dict(fixtures)["rope"].model())
    node = next(n for n in model.graph.node if n.op_type == "RotaryEmbedding")
    position = next(i for i in model.graph.initializer if i.name == node.input[1])
    model.graph.input.append(
        helper.make_tensor_value_info(position.name, TensorProto.INT64, [1])
    )
    model.graph.initializer.remove(position)
    invalid_models.append(("rope_runtime_position", model))
    model = copy.deepcopy(dict(fixtures)["norm"].model())
    node = next(
        n for n in model.graph.node if n.op_type == "SimplifiedLayerNormalization"
    )
    node.output.append("auxiliary_statistic")
    model.graph.output.append(
        helper.make_tensor_value_info("auxiliary_statistic", TensorProto.FLOAT, [2, 1])
    )
    invalid_models.append(("norm_auxiliary_output", model))
    model = copy.deepcopy(dict(fixtures)["gqa"].model())
    node = next(n for n in model.graph.node if n.op_type == "GroupQueryAttention")
    info = next(v for v in model.graph.value_info if v.name == node.output[1])
    info.type.tensor_type.shape.dim[1].dim_value = 3
    invalid_models.append(("gqa_invalid_present_shape", model))
    for name, model in invalid_models:
        source = work / f"reject_{name}.onnx"
        onnx.save(model, source)
        for layout in ("legacy", "experimental"):
            target = source.with_suffix(f".{layout}.tflite")
            with target.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    [
                        str(args.converter.resolve()),
                        str(source),
                        "-o",
                        str(target),
                        "--layout-mode",
                        layout,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            assert result.returncode != 0 and not target.exists(), (name, layout)
            rejected += 1
    result = {
        "output_comparisons": comparisons,
        "rejection_checks": rejected,
        "cases": {name: graph.cases for name, graph in fixtures},
    }
    (work / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", result, flush=True)


if __name__ == "__main__":
    main()
