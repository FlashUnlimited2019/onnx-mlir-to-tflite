#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Generate deterministic ONNX fixtures for Conv1D/Conv3D lowering tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnx.utils import Extractor

SEED = 20260805


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    return parser.parse_args()


def save_model(model: onnx.ModelProto, path: Path) -> None:
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(path)


def make_conv3d_model(
    path: Path,
    rng: np.random.Generator,
    input_shape: list[int],
    output_channels: int,
    kernel: list[int],
    pads: list[int],
    group: int = 1,
) -> None:
    channels_per_group = input_shape[1] // group
    weight_shape = [output_channels, channels_per_group, *kernel]
    weight = numpy_helper.from_array(
        rng.normal(0.0, 0.2, weight_shape).astype(np.float32), "weight"
    )
    bias = numpy_helper.from_array(
        rng.normal(0.0, 0.05, [output_channels]).astype(np.float32), "bias"
    )
    spatial_output = [
        input_shape[index + 2] + pads[index] + pads[index + 3] - kernel[index] + 1
        for index in range(3)
    ]
    output_shape = [input_shape[0], output_channels, *spatial_output]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weight", "bias"],
                ["output"],
                name=path.stem,
                kernel_shape=kernel,
                pads=pads,
                strides=[1, 1, 1],
                dilations=[1, 1, 1],
                group=group,
            )
        ],
        path.stem,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        [weight, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="onnx-mlir-conv-rank-tests",
    )
    model.ir_version = 8
    save_model(model, path)


def make_supported_projection(source: Path, path: Path) -> None:
    model = shape_inference.infer_shapes(onnx.load(source))
    branch_outputs = [
        "conv1d_regular_flat",
        "conv1d_grouped_flat",
        "conv1d_depthwise_separable_flat",
        "conv2d_regular_flat",
        "conv2d_grouped_flat",
        "conv2d_depthwise_separable_flat",
    ]
    extracted = Extractor(model).extract_model(["input_flat"], branch_outputs)
    dimensions = []
    for output in extracted.graph.output:
        shape = output.type.tensor_type.shape.dim
        dimensions.append([dimension.dim_value for dimension in shape])
    if any(len(shape) != 2 or shape[0] != 1 for shape in dimensions):
        raise ValueError(f"unexpected supported projection shapes: {dimensions}")
    total = sum(shape[1] for shape in dimensions)
    del extracted.graph.output[:]
    extracted.graph.node.append(
        helper.make_node(
            "Concat",
            branch_outputs,
            ["supported_output"],
            name="Concat_supported_conv_outputs",
            axis=1,
        )
    )
    extracted.graph.output.append(
        helper.make_tensor_value_info("supported_output", TensorProto.FLOAT, [1, total])
    )
    extracted.graph.name = "conv_many_3_4_5d_supported_projection"
    save_model(extracted, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    make_supported_projection(
        args.source_model, args.output_dir / "conv_many_supported.onnx"
    )
    make_conv3d_model(
        args.output_dir / "conv3d_k1hw.onnx",
        rng,
        [1, 2, 4, 7, 9],
        3,
        [1, 3, 5],
        [0, 1, 2, 0, 1, 2],
    )
    make_conv3d_model(
        args.output_dir / "conv3d_kd11.onnx",
        rng,
        [1, 2, 8, 4, 5],
        3,
        [3, 1, 1],
        [1, 0, 0, 1, 0, 0],
    )
    make_conv3d_model(
        args.output_dir / "conv3d_kd1w.onnx",
        rng,
        [1, 2, 8, 4, 6],
        4,
        [3, 1, 2],
        [1, 0, 0, 1, 0, 1],
    )
    make_conv3d_model(
        args.output_dir / "conv3d_group2_k1hw.onnx",
        rng,
        [1, 4, 3, 6, 7],
        6,
        [1, 3, 3],
        [0, 1, 1, 0, 1, 1],
        group=2,
    )
    make_conv3d_model(
        args.output_dir / "conv3d_depthwise_kd11.onnx",
        rng,
        [1, 3, 7, 4, 5],
        6,
        [3, 1, 1],
        [1, 0, 0, 1, 0, 0],
        group=3,
    )


if __name__ == "__main__":
    main()
