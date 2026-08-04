#!/usr/bin/env python3
"""Print reproducible structural statistics for a TFLite FlatBuffer."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from tensorflow.lite.python import schema_py_generated as schema_fb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-operators", type=int)
    parser.add_argument("--max-constant-tensors", type=int)
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
    data = args.model.read_bytes()
    if len(data) < 8 or data[4:8] != b"TFL3":
        raise ValueError(f"not a TFLite FlatBuffer: {args.model}")

    model = schema_fb.Model.GetRootAsModel(data, 0)
    builtin_names = builtin_operator_names()
    op_counts: Counter[str] = Counter()
    tensor_count = 0
    constant_tensor_count = 0
    used_constant_buffers: set[int] = set()

    for subgraph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_index)
        tensor_count += subgraph.TensorsLength()
        for operator_index in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(operator_index)
            opcode = model.OperatorCodes(operator.OpcodeIndex())
            builtin_code = opcode.BuiltinCode()
            name = builtin_names.get(builtin_code, f"BUILTIN_{builtin_code}")
            if name == "CUSTOM":
                custom_code = opcode.CustomCode()
                if custom_code:
                    name = f"CUSTOM:{custom_code.decode('utf-8')}"
            op_counts[name] += 1
        for tensor_index in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_index)
            buffer_index = tensor.Buffer()
            if buffer_index and model.Buffers(buffer_index).DataLength():
                constant_tensor_count += 1
                used_constant_buffers.add(buffer_index)

    print(f"file: {args.model}")
    print(f"bytes: {len(data)}")
    print(f"subgraphs: {model.SubgraphsLength()}")
    print(f"operators: {sum(op_counts.values())}")
    print(f"tensors: {tensor_count}")
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
    for name in args.forbid_op:
        if op_counts[name]:
            print(
                f"FAIL: forbidden operator {name} appears {op_counts[name]} time(s)",
                file=sys.stderr,
            )
            failed = True
    if failed:
        return 1
    if (
        args.max_operators is not None
        or args.max_constant_tensors is not None
        or args.forbid_op
    ):
        print("PASS: structural constraints satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
