#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Print reproducible structural statistics for a TFLite FlatBuffer."""

from __future__ import annotations

import argparse
from collections import Counter
import mmap
from pathlib import Path
import sys

from tensorflow.lite.python import schema_py_generated as schema_fb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-operators", type=int)
    parser.add_argument("--max-constant-tensors", type=int)
    parser.add_argument("--max-tensor-rank", type=int)
    parser.add_argument("--forbid-op", action="append", default=[])
    return parser.parse_args()


def builtin_operator_names() -> dict[int, str]:
    return {
        value: name
        for name, value in vars(schema_fb.BuiltinOperator).items()
        if name.isupper() and isinstance(value, int)
    }


def main() -> int:
    args = parse_args()
    file_size = args.model.stat().st_size
    with args.model.open("rb") as model_file, mmap.mmap(
        model_file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        if len(data) < 8 or data[4:8] != b"TFL3":
            raise ValueError(f"not a TFLite FlatBuffer: {args.model}")

        model = schema_fb.Model.GetRootAsModel(data, 0)
        subgraph_count = model.SubgraphsLength()
        builtin_names = builtin_operator_names()
        op_counts: Counter[str] = Counter()
        tensor_count = 0
        constant_tensor_count = 0
        tensor_rank_counts: Counter[int] = Counter()
        used_constant_buffers: set[int] = set()

        for subgraph_index in range(subgraph_count):
            subgraph = model.Subgraphs(subgraph_index)
            tensor_count += subgraph.TensorsLength()
            for operator_index in range(subgraph.OperatorsLength()):
                operator = subgraph.Operators(operator_index)
                opcode = model.OperatorCodes(operator.OpcodeIndex())
                builtin_code = opcode.BuiltinCode()
                name = builtin_names.get(
                    builtin_code, f"BUILTIN_{builtin_code}"
                )
                if name == "CUSTOM":
                    custom_code = opcode.CustomCode()
                    if custom_code:
                        name = f"CUSTOM:{custom_code.decode('utf-8')}"
                op_counts[name] += 1
            for tensor_index in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(tensor_index)
                tensor_rank_counts[tensor.ShapeLength()] += 1
                buffer_index = tensor.Buffer()
                if not buffer_index:
                    continue
                buffer = model.Buffers(buffer_index)
                has_inline_data = buffer.DataLength() != 0
                has_offset_data = (
                    hasattr(buffer, "Offset")
                    and buffer.Offset() != 0
                    and buffer.Size() != 0
                )
                if has_inline_data or has_offset_data:
                    constant_tensor_count += 1
                    used_constant_buffers.add(buffer_index)

    print(f"file: {args.model}")
    print(f"bytes: {file_size}")
    print(f"subgraphs: {subgraph_count}")
    print(f"operators: {sum(op_counts.values())}")
    print(f"tensors: {tensor_count}")
    print(
        "tensor ranks: "
        + ", ".join(
            f"rank {rank}: {count}"
            for rank, count in sorted(tensor_rank_counts.items())
        )
    )
    print(f"constant tensors: {constant_tensor_count}")
    print(f"constant buffers: {len(used_constant_buffers)}")
    print("operator counts:")
    for name, count in sorted(op_counts.items()):
        print(f"  {name}: {count}")

    failed = False
    operator_count = sum(op_counts.values())
    if args.max_operators is not None and operator_count > args.max_operators:
        print(
            f"FAIL: operators {operator_count} exceed {args.max_operators}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.max_constant_tensors is not None
        and constant_tensor_count > args.max_constant_tensors
    ):
        print(
            f"FAIL: constant tensors {constant_tensor_count} exceed "
            f"{args.max_constant_tensors}",
            file=sys.stderr,
        )
        failed = True
    max_tensor_rank = max(tensor_rank_counts, default=0)
    if (
        args.max_tensor_rank is not None
        and max_tensor_rank > args.max_tensor_rank
    ):
        print(
            f"FAIL: maximum tensor rank {max_tensor_rank} exceeds "
            f"{args.max_tensor_rank}",
            file=sys.stderr,
        )
        failed = True
    for name in args.forbid_op:
        if name == "CUSTOM":
            forbidden_count = sum(
                count
                for operator_name, count in op_counts.items()
                if operator_name == "CUSTOM"
                or operator_name.startswith("CUSTOM:")
            )
        elif name == "Flex":
            forbidden_count = sum(
                count
                for operator_name, count in op_counts.items()
                if operator_name.startswith("CUSTOM:Flex")
            )
        else:
            forbidden_count = op_counts[name]
        if forbidden_count:
            print(
                f"FAIL: forbidden operator {name} appears "
                f"{forbidden_count} time(s)",
                file=sys.stderr,
            )
            failed = True
    if failed:
        return 1
    if (
        args.max_operators is not None
        or args.max_constant_tensors is not None
        or args.max_tensor_rank is not None
        or args.forbid_op
    ):
        print("PASS: structural constraints satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
