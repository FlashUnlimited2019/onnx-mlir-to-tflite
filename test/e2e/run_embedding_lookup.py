#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Check guarded embedding selection, index bounds, layouts, and hybrid tables."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Optional

import flatbuffers
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from run_similarity import onnx_to_tfl_input, tfl_to_onnx_output


@dataclass
class Case:
    name: str
    shape: tuple = (7, 128)
    indices_shape: tuple = (2, 3)
    index_dtype: type = np.int64
    data_dtype: type = np.float32
    axis: int = 0
    runtime_table: bool = False
    constant_indices: bool = False
    shared: bool = False
    clipped: bool = False
    maximum_bound: Optional[int] = None
    bound_first: bool = False
    external: bool = False
    special: bool = False
    embedding: bool = True
    index_permutation: tuple = ()
    conversion_error: str = ""


def cases():
    for dtype in (np.int32, np.int64):
        for shape in (
            (),
            (4,),
            (2, 3),
            (2, 3, 4),
            (2, 2, 2),
            (2, 1, 3, 2),
            (1, 2, 1, 2, 2),
        ):
            yield Case(
                f"rank{len(shape)}_{np.dtype(dtype).name}_{''.join(map(str, shape))}",
                indices_shape=shape,
                index_dtype=dtype,
            )
    yield Case("negative_axis", axis=-2)
    yield Case("wide_odd", shape=(7, 129))
    yield Case("wide_table", shape=(7, 512))
    yield Case("single_row", shape=(1, 128))
    yield Case("clipped_padding", clipped=True)
    yield Case("clipped_padding_i32", clipped=True, index_dtype=np.int32)
    yield Case("maximum_positive", maximum_bound=1)
    yield Case("maximum_reversed", maximum_bound=0, bound_first=True)
    yield Case("maximum_negative", maximum_bound=-2)
    yield Case("maximum_negative_i32", maximum_bound=-2, index_dtype=np.int32)
    yield Case("external_table", external=True)
    yield Case("shared_table", shared=True)
    yield Case("special_bits", special=True)
    yield Case(
        "transposed_indices", indices_shape=(2, 3, 4), index_permutation=(2, 0, 1)
    )
    yield Case(
        "equal_extent_transpose", indices_shape=(2, 2, 2), index_permutation=(2, 0, 1)
    )
    for width in (1, 64, 127):
        yield Case(f"narrow_{width}", shape=(7, width), embedding=False)
    yield Case("runtime_table", runtime_table=True, embedding=False)
    yield Case("axis_one", shape=(7, 128), axis=1, embedding=False)
    yield Case("axis_negative_one", shape=(7, 128), axis=-1, embedding=False)
    yield Case("rank_three_table", shape=(7, 2, 128), embedding=False)
    for dtype in (np.int32, np.int64, np.bool_):
        yield Case(f"table_{np.dtype(dtype).name}", data_dtype=dtype, embedding=False)
    yield Case("constant_indices", constant_indices=True, embedding=False)
    # The importer currently treats an explicit zero input extent as dynamic.
    # Preserve that existing rejection; the direct-MLIR test covers fallback
    # selection with an actual static zero extent.
    yield Case(
        "empty_indices",
        indices_shape=(0,),
        embedding=False,
        conversion_error="dynamic tensor shape is not supported",
    )


def make_case(case):
    rng = np.random.default_rng(239)
    table = rng.normal(size=case.shape).astype(case.data_dtype)
    if case.special:
        bits = np.array(
            [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC12345, 0x3F800000],
            np.uint32,
        )
        table = np.resize(bits, case.shape).view(np.float32)
    axis = case.axis % len(case.shape)
    rows = case.shape[axis]
    indices = np.resize(
        np.array([-rows, -1, 0, rows - 1, 0, -1], case.index_dtype), case.indices_shape
    )
    dtype = helper.np_dtype_to_tensor_dtype(np.dtype(case.index_dtype))
    table_dtype = helper.np_dtype_to_tensor_dtype(np.dtype(case.data_dtype))
    lookup_shape = (
        tuple(case.indices_shape[i] for i in case.index_permutation)
        if case.index_permutation
        else case.indices_shape
    )
    result_shape = case.shape[:axis] + lookup_shape + case.shape[axis + 1 :]
    inputs, constants, nodes = [], [], []
    if case.runtime_table:
        inputs.append(helper.make_tensor_value_info("table", table_dtype, case.shape))
    else:
        constants.append(numpy_helper.from_array(table, "table"))
    if case.constant_indices:
        constants.append(numpy_helper.from_array(indices, "ids"))
    else:
        inputs.append(helper.make_tensor_value_info("ids", dtype, case.indices_shape))
    lookup = "ids"
    if case.index_permutation:
        nodes.append(
            helper.make_node(
                "Transpose", ["ids"], ["transposed"], perm=case.index_permutation
            )
        )
        lookup = "transposed"
    if case.maximum_bound is not None:
        constants.append(
            numpy_helper.from_array(
                np.array(case.maximum_bound, case.index_dtype), "bound"
            )
        )
        operands = [lookup, "bound"]
        if case.bound_first:
            operands.reverse()
        nodes.append(helper.make_node("Max", operands, ["bounded"]))
        lookup = "bounded"
    if case.clipped:
        constants.append(numpy_helper.from_array(np.array(0, case.index_dtype), "zero"))
        constants.append(numpy_helper.from_array(np.array([-1], np.int64), "mask_axes"))
        mask_ids, mask_zero = "ids", "zero"
        if case.index_dtype == np.int32:
            # The comparison lowering supports i64, so widen exactly for the
            # mask while retaining i32 indices on the lookup branch.
            nodes.append(
                helper.make_node("Cast", ["ids"], ["mask_ids"], to=TensorProto.INT64)
            )
            constants.append(
                numpy_helper.from_array(np.array(0, np.int64), "mask_zero")
            )
            mask_ids, mask_zero = "mask_ids", "mask_zero"
        nodes.extend(
            [
                helper.make_node("Clip", ["ids", "zero"], ["clipped"]),
                helper.make_node("GreaterOrEqual", [mask_ids, mask_zero], ["valid"]),
                helper.make_node("Cast", ["valid"], ["mask"], to=TensorProto.FLOAT),
                helper.make_node("Unsqueeze", ["mask", "mask_axes"], ["expanded_mask"]),
            ]
        )
        lookup = "clipped"
    nodes.append(
        helper.make_node(
            "Gather",
            ["table", lookup],
            ["raw" if case.clipped else "result"],
            axis=case.axis,
        )
    )
    if case.clipped:
        nodes.append(helper.make_node("Mul", ["raw", "expanded_mask"], ["result"]))
    outputs = [helper.make_tensor_value_info("result", table_dtype, result_shape)]
    if case.shared:
        inputs.append(
            helper.make_tensor_value_info("other_ids", dtype, case.indices_shape)
        )
        nodes.append(
            helper.make_node(
                "Gather", ["table", "other_ids"], ["other_result"], axis=case.axis
            )
        )
        outputs.extend(
            [
                helper.make_tensor_value_info(
                    "other_result", table_dtype, result_shape
                ),
                helper.make_tensor_value_info("table", table_dtype, case.shape),
            ]
        )
    model = helper.make_model(
        helper.make_graph(nodes, case.name, inputs, outputs, constants),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = []
    for seed in (0, 31, 59):
        feed = {}
        if case.runtime_table:
            feed["table"] = table.copy()
        if not case.constant_indices:
            feed["ids"] = (
                indices.copy()
                if seed == 0
                else np.random.default_rng(seed).integers(
                    -rows, rows, size=case.indices_shape, dtype=case.index_dtype
                )
            )
        if case.shared:
            feed["other_ids"] = np.array(
                np.remainder(feed["ids"] + 1, rows), dtype=case.index_dtype
            )
        feeds.append(feed)
    if case.clipped or case.maximum_bound is not None:
        # Lower clipping makes even the most negative source integer valid.
        feed = {k: v.copy() for k, v in feeds[0].items()}
        feed["ids"] = np.full(
            case.indices_shape, np.iinfo(case.index_dtype).min, case.index_dtype
        )
        feeds.append(feed)
    return model, feeds, table, rows


def interpreter(path, delegate):
    return tf.lite.Interpreter(
        model_path=str(path),
        num_threads=1,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.AUTO
            if delegate
            else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )


def run(runner, feed, layout):
    details = runner.get_input_details()
    return runner(
        **{
            k: np.array(
                onnx_to_tfl_input(v, tuple(details[k]["shape"]), layout),
                copy=True,
                order="C",
            )
            for k, v in feed.items()
        }
    )


def check_outputs(actual, expected, layout, exact=True):
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        value = tfl_to_onnx_output(actual[name], reference.shape, layout)
        assert value.dtype == reference.dtype
        if exact:
            np.testing.assert_array_equal(value, reference)
            if value.dtype == np.float32:
                np.testing.assert_array_equal(
                    value.view(np.uint32), reference.view(np.uint32)
                )
        else:
            np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-7)


def hybrid_model(target, per_row):
    # Test-only post-quantization of a small fixture. The converter still emits
    # FP32 weights; this checks that the selected builtin accepts hybrid tables.
    model = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(target.read_bytes(), 0)
    )
    graph = model.subgraphs[0]
    ops = [
        op
        for op in graph.operators
        if model.operatorCodes[op.opcodeIndex].builtinCode
        == schema.BuiltinOperator.EMBEDDING_LOOKUP
    ]
    assert len(ops) == 1
    op = ops[0]
    weight = graph.tensors[op.inputs[1]]
    table = np.frombuffer(model.buffers[weight.buffer].data, np.float32).reshape(
        weight.shape
    )
    scales = np.maximum(
        np.max(np.abs(table), axis=1) if per_row else np.array([np.max(np.abs(table))]),
        np.float32(1e-6),
    ).astype(np.float32) / np.float32(127)
    divisor = scales[:, None] if per_row else scales[0]
    quantized = np.clip(np.rint(table / divisor), -127, 127).astype(np.int8)
    weight.type = schema.TensorType.INT8
    weight.quantization = schema.QuantizationParametersT()
    weight.quantization.scale = scales
    weight.quantization.zeroPoint = np.zeros(len(scales), np.int64)
    weight.quantization.quantizedDimension = 0
    model.buffers[weight.buffer].data = np.frombuffer(quantized.tobytes(), np.uint8)
    model.operatorCodes[op.opcodeIndex].version = 4 if per_row else 1
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    path = target.with_name(target.stem + f".hybrid_{int(per_row)}.tflite")
    path.write_bytes(bytes(builder.Output()))
    return path, quantized.astype(np.float32) * divisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    selected = [c for c in cases() if not args.case or c.name in args.case]
    assert selected, "No matching fixtures"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    comparisons = rejections = hybrid_checks = conversion_rejections = 0
    reports = []
    for case in selected:
        model, feeds, table, rows = make_case(case)
        source = args.work_dir / f"{case.name}.onnx"
        if case.external:
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=source.name + ".data",
                size_threshold=0,
            )
        else:
            onnx.save(model, source)
        reference = ort.InferenceSession(
            str(source), options, providers=["CPUExecutionProvider"]
        )
        expected = [
            dict(
                zip(
                    [v.name for v in reference.get_outputs()], reference.run(None, feed)
                )
            )
            for feed in feeds
        ]
        invalid = []
        if case.embedding:
            info = np.iinfo(case.index_dtype)
            values = [rows, rows + 1, info.max]
            if not case.clipped and case.maximum_bound is None:
                values.extend([-rows - 1, info.min])
            if case.index_dtype == np.int64:
                values.extend([2**32, 2**32 + rows - 1, 2**40])
                if not case.clipped and case.maximum_bound is None:
                    values.extend([-(2**32), -(2**32) + 1])
            for value in values:
                feed = {k: v.copy() for k, v in feeds[0].items()}
                feed["ids"] = np.full(case.indices_shape, value, case.index_dtype)
                try:
                    reference.run(None, feed)
                except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
                    invalid.append(feed)
                else:
                    raise AssertionError(
                        (case.name, "reference accepted invalid index", value)
                    )
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                target = args.work_dir / f"{case.name}.{layout}.{int(optimize)}.tflite"
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
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, TMPDIR=str(args.work_dir.resolve())),
                    )
                if case.conversion_error:
                    assert completed.returncode != 0
                    assert (
                        case.conversion_error in target.with_suffix(".log").read_text()
                    )
                    assert not target.exists()
                    conversion_rejections += 1
                    print("EXPECTED REJECTION", case.name, layout, optimize, flush=True)
                    continue
                completed.check_returncode()
                fb = schema.Model.GetRootAsModel(target.read_bytes(), 0)
                graph = fb.Subgraphs(0)
                codes = [
                    fb.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                    for i in range(graph.OperatorsLength())
                ]
                assert not set(codes).intersection(
                    {
                        schema.BuiltinOperator.CUSTOM,
                        schema.BuiltinOperator.WHILE,
                        schema.BuiltinOperator.IF,
                    }
                )
                expected_count = (2 if case.shared else 1) if case.embedding else 0
                assert (
                    codes.count(schema.BuiltinOperator.EMBEDDING_LOOKUP)
                    == expected_count
                ), (case.name, layout, optimize, codes)
                if case.embedding:
                    assert schema.BuiltinOperator.GATHER not in codes
                    if case.clipped or (
                        case.maximum_bound is not None and case.maximum_bound >= 0
                    ):
                        assert schema.BuiltinOperator.LESS not in codes
                        assert schema.BuiltinOperator.SELECT_V2 not in codes
                    weight_tensors = []
                    for i, code in enumerate(codes):
                        if code != schema.BuiltinOperator.EMBEDDING_LOOKUP:
                            continue
                        op = graph.Operators(i)
                        lookup = graph.Tensors(op.Inputs(0))
                        weight = graph.Tensors(op.Inputs(1))
                        assert (
                            lookup.Type() == schema.TensorType.INT32
                            and lookup.ShapeLength() == 1
                        )
                        assert weight.ShapeAsNumpy().tolist() == list(case.shape)
                        assert weight.Type() == schema.TensorType.FLOAT32
                        np.testing.assert_array_equal(
                            np.frombuffer(
                                fb.Buffers(weight.Buffer()).DataAsNumpy(), np.uint32
                            ),
                            table.reshape(-1).view(np.uint32),
                        )
                        weight_tensors.append(op.Inputs(1))
                    if case.shared:
                        assert len(set(weight_tensors)) == 1
                for delegate in (False, True):
                    runtime = interpreter(target, delegate)
                    runner = runtime.get_signature_runner("serving_default")
                    for feed, results in zip(feeds, expected):
                        check_outputs(run(runner, feed, layout), results, layout)
                        comparisons += len(results)
                    for feed in invalid:
                        try:
                            run(runner, feed, layout)
                        except (RuntimeError, ValueError):
                            rejections += 1
                        else:
                            raise AssertionError(
                                (
                                    case.name,
                                    "runtime accepted invalid index",
                                    feed["ids"],
                                )
                            )
                    # An error must not poison subsequent valid invocation.
                    check_outputs(run(runner, feeds[0], layout), expected[0], layout)
                    comparisons += len(expected[0])
                    del runner, runtime
                if case.name == "wide_odd":
                    for per_row in (False, True):
                        quant_path, restored_table = hybrid_model(target, per_row)
                        for delegate in (False, True):
                            runtime = interpreter(quant_path, delegate)
                            runner = runtime.get_signature_runner("serving_default")
                            for feed in feeds:
                                expected_quant = {
                                    "result": np.take(
                                        restored_table, feed["ids"], axis=0
                                    )
                                }
                                check_outputs(
                                    run(runner, feed, layout),
                                    expected_quant,
                                    layout,
                                    exact=False,
                                )
                                hybrid_checks += 1
                            del runner, runtime
                reports.append(
                    dict(
                        case=case.name,
                        layout=layout,
                        optimize=optimize,
                        embeddings=expected_count,
                        operators=len(codes),
                        passed=True,
                    )
                )
                print("PASS", case.name, layout, optimize, flush=True)
    report = dict(
        conversions=len(reports),
        comparisons=comparisons,
        rejected_invocations=rejections,
        hybrid_comparisons=hybrid_checks,
        expected_conversion_rejections=conversion_rejections,
        cases=reports,
    )
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
