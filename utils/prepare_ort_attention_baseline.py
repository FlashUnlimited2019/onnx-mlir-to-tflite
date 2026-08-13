#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Make the constructed attention fixture loadable by the pinned ORT.

The conversion always consumes the original model. This utility only creates
the numerical-reference copy needed by ONNX Runtime 1.23, whose maximum IR and
released ai.onnx opset are older and whose GroupQueryAttention schema requires
present_key/present_value output names even when they are unused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def static_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value_info.type.tensor_type
    shape = [dimension.dim_value for dimension in tensor_type.shape.dim]
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{value_info.name} does not have a positive static shape")
    return shape


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ir-version", type=int, default=11)
    parser.add_argument("--onnx-opset", type=int, default=23)
    args = parser.parse_args()

    model = onnx.load(args.input)
    model.ir_version = args.ir_version
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = args.onnx_opset

    values = {
        value.name: value
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    added_outputs = 0
    for node in model.graph.node:
        if not (
            node.domain == "com.microsoft"
            and node.op_type == "GroupQueryAttention"
            and len(node.output) == 1
        ):
            continue
        if len(node.input) < 7 or (node.input[3] or node.input[4]):
            raise ValueError(
                f"{node.name}: automatic ORT baseline repair only supports no-cache GQA"
            )
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        query_heads = int(attributes["num_heads"])
        kv_heads = int(attributes["kv_num_heads"])
        query_shape = static_shape(values[node.input[0]])
        if node.input[1]:
            key_shape = static_shape(values[node.input[1]])
            batch, key_sequence, kv_width = key_shape
            if kv_width % kv_heads:
                raise ValueError(f"{node.name}: key width does not divide kv heads")
            head_size = kv_width // kv_heads
        else:
            batch, key_sequence, packed_width = query_shape
            divisor = query_heads + 2 * kv_heads
            if packed_width % divisor:
                raise ValueError(f"{node.name}: packed width does not match heads")
            head_size = packed_width // divisor

        present_shape = [batch, kv_heads, key_sequence, head_size]
        for suffix in ("present_key", "present_value"):
            name = f"{node.name}_{suffix}"
            node.output.append(name)
            value_info = helper.make_tensor_value_info(
                name, TensorProto.FLOAT, present_shape
            )
            model.graph.value_info.append(value_info)
            values[name] = value_info
            added_outputs += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"generated ORT-only baseline {args.output} "
        f"(IR {model.ir_version}, ai.onnx opset {args.onnx_opset}, "
        f"added {added_outputs} GQA present outputs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
