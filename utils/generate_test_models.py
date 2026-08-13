#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Generate deterministic, small ONNX models for ONNXToTFL tests."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SEED = 20260804
OPSET = 18
IR_VERSION = 11


def save(graph: onnx.GraphProto, path: Path) -> None:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="onnx-mlir-onnx-to-tflite-poc",
    )
    # Pin the protobuf IR independently of the installed onnx package default.
    # IR 11 supports this opset/model subset and is loadable by the test ORT.
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(path)


def add_relu(output_dir: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["sum"], name="add"),
        helper.make_node("Relu", ["sum"], ["z"], name="relu"),
    ]
    save(
        helper.make_graph(nodes, "add_relu", [x, y], [z]), output_dir / "add_relu.onnx"
    )


def mlp(output_dir: Path, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [2, 3])
    arrays = {
        "w1": rng.normal(0.0, 0.25, [4, 5]).astype(np.float32),
        "b1": rng.normal(0.0, 0.1, [5]).astype(np.float32),
        "w2": rng.normal(0.0, 0.25, [5, 3]).astype(np.float32),
        "b2": rng.normal(0.0, 0.1, [3]).astype(np.float32),
    }
    initializers = [
        numpy_helper.from_array(value, name) for name, value in arrays.items()
    ]
    nodes = [
        helper.make_node("MatMul", ["input", "w1"], ["hidden_mm"], name="matmul_1"),
        helper.make_node("Add", ["hidden_mm", "b1"], ["hidden_bias"], name="bias_1"),
        helper.make_node("Relu", ["hidden_bias"], ["hidden"], name="relu"),
        helper.make_node("MatMul", ["hidden", "w2"], ["logits_mm"], name="matmul_2"),
        helper.make_node("Add", ["logits_mm", "b2"], ["logits"], name="bias_2"),
        helper.make_node(
            "Softmax", ["logits"], ["probabilities"], name="softmax", axis=-1
        ),
    ]
    save(
        helper.make_graph(nodes, "mlp", [x], [y], initializers), output_dir / "mlp.onnx"
    )


def reshape_transpose(output_dir: Path) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [6, 4])
    shape = numpy_helper.from_array(np.array([4, 6], dtype=np.int64), "shape")
    nodes = [
        helper.make_node("Reshape", ["input", "shape"], ["reshaped"], name="reshape"),
        helper.make_node(
            "Transpose", ["reshaped"], ["output"], name="transpose", perm=[1, 0]
        ),
    ]
    save(
        helper.make_graph(nodes, "reshape_transpose", [x], [y], [shape]),
        output_dir / "reshape_transpose.onnx",
    )


def operator_sweep(output_dir: Path, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
    concat_out = helper.make_tensor_value_info(
        "concat_output", TensorProto.FLOAT, [2, 8]
    )
    gemm_transb_out = helper.make_tensor_value_info(
        "gemm_transb_output", TensorProto.FLOAT, [2, 3]
    )
    gemm_transa_out = helper.make_tensor_value_info(
        "gemm_transa_output", TensorProto.FLOAT, [4, 3]
    )
    arrays = {
        "bias4": np.array([0.2, -0.1, 0.3, -0.4], dtype=np.float32),
        "scale4": np.array([1.5, -0.5, 2.0, 0.25], dtype=np.float32),
        "divisor4": np.array([0.75, 1.25, -2.0, 0.5], dtype=np.float32),
        "gemm_weight_b": rng.normal(0.0, 0.2, [3, 4]).astype(np.float32),
        "gemm_bias_b": rng.normal(0.0, 0.1, [3]).astype(np.float32),
        "gemm_weight_a": rng.normal(0.0, 0.2, [2, 3]).astype(np.float32),
        "gemm_bias_a": rng.normal(0.0, 0.1, [3]).astype(np.float32),
    }
    initializers = [
        numpy_helper.from_array(value, name) for name, value in arrays.items()
    ]
    nodes = [
        helper.make_node("Identity", ["input"], ["identity"], name="identity"),
        helper.make_node("Sub", ["identity", "bias4"], ["sub"], name="sub"),
        helper.make_node("Mul", ["sub", "scale4"], ["mul"], name="mul"),
        helper.make_node("Div", ["mul", "divisor4"], ["div"], name="div"),
        helper.make_node("Sigmoid", ["div"], ["sigmoid"], name="sigmoid"),
        helper.make_node("Tanh", ["sigmoid"], ["tanh"], name="tanh"),
        helper.make_node(
            "Concat",
            ["tanh", "tanh"],
            ["concat_output"],
            name="concat",
            axis=1,
        ),
        helper.make_node(
            "Gemm",
            ["input", "gemm_weight_b", "gemm_bias_b"],
            ["gemm_transb_output"],
            name="gemm_transb",
            alpha=0.5,
            beta=2.0,
            transB=1,
        ),
        helper.make_node(
            "Gemm",
            ["input", "gemm_weight_a", "gemm_bias_a"],
            ["gemm_transa_output"],
            name="gemm_transa",
            transA=1,
        ),
    ]
    save(
        helper.make_graph(
            nodes,
            "operator_sweep",
            [x],
            [concat_out, gemm_transb_out, gemm_transa_out],
            initializers,
        ),
        output_dir / "operator_sweep.onnx",
    )


def simple_conv(output_dir: Path, rng: np.random.Generator) -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 6, 6])
    weight = numpy_helper.from_array(
        rng.normal(0.0, 0.2, [4, 3, 3, 3]).astype(np.float32), "weight"
    )
    bias = numpy_helper.from_array(np.zeros([4], dtype=np.float32), "bias")
    conv = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        name="conv",
        kernel_shape=[3, 3],
    )
    save(
        helper.make_graph([conv], "simple_conv", [x], [y], [weight, bias]),
        output_dir / "simple_conv.onnx",
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "test" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    add_relu(output_dir)
    mlp(output_dir, rng)
    reshape_transpose(output_dir)
    operator_sweep(output_dir, rng)
    simple_conv(output_dir, rng)


if __name__ == "__main__":
    main()
