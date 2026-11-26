#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Verify wide FP32 depthwise compatibility without per-group graph expansion."""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess

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
    channels: int = 6144
    multiplier: int = 1
    spatial: tuple = (9,)
    kernel: tuple = (4,)
    pads: tuple = ()
    strides: tuple = ()
    dilations: tuple = ()
    auto_pad: str = "NOTSET"
    batch: int = 1
    bias: bool = True
    runtime_filter: bool = False
    runtime_bias: bool = False
    relu: bool = False
    group: int = 0
    outputs: int = 0
    shared_filter: bool = False


def cases():
    yield Case("wide_1d", pads=(3, 3))
    yield Case("boundary_2d", channels=4832, spatial=(5, 4), kernel=(3, 2))
    yield Case("above_boundary_2d", channels=4833, spatial=(5, 4), kernel=(3, 2))
    yield Case("multiplier_boundary", channels=2416, multiplier=2)
    yield Case("multiplier_above", channels=2417, multiplier=2)
    yield Case(
        "multiplier_three",
        channels=2048,
        multiplier=3,
        batch=2,
        spatial=(7, 5),
        kernel=(3, 2),
        strides=(2, 1),
        pads=(1, 0, 1, 1),
        relu=True,
    )
    yield Case("single_input_channel", channels=1, multiplier=4833)
    yield Case("same_stride", strides=(2,), auto_pad="SAME_UPPER")
    yield Case("dilated_1d", spatial=(11,), dilations=(2,), pads=(4, 2), bias=False)
    yield Case(
        "dilated_2d",
        spatial=(7, 6),
        kernel=(2, 3),
        strides=(2, 1),
        dilations=(1, 2),
        pads=(1, 2, 0, 1),
    )
    yield Case(
        "runtime_filter_2d",
        channels=4833,
        spatial=(5, 4),
        kernel=(2, 2),
        runtime_filter=True,
        runtime_bias=True,
    )
    yield Case("runtime_filter_1d", runtime_filter=True, bias=False)
    yield Case(
        "collapse_depth",
        spatial=(2, 4, 5),
        kernel=(1, 2, 2),
        strides=(1, 2, 1),
        dilations=(1, 1, 2),
    )
    yield Case("collapse_height", spatial=(4, 2, 5), kernel=(2, 1, 2))
    yield Case(
        "collapse_width",
        spatial=(4, 5, 2),
        kernel=(2, 2, 1),
        runtime_filter=True,
    )
    yield Case(
        "reduced_boundary",
        channels=4832,
        spatial=(2, 4, 5),
        kernel=(1, 2, 2),
    )
    yield Case(
        "reduced_multiplier",
        channels=2048,
        multiplier=3,
        spatial=(4, 2, 3),
        kernel=(2, 1, 1),
        relu=True,
    )
    yield Case("ordinary_groups", channels=4, group=2, outputs=4834)
    yield Case("ordinary_conv", channels=3, group=1, outputs=4833)
    yield Case(
        "small_depthwise",
        channels=16,
        multiplier=2,
        spatial=(5, 4),
        kernel=(3, 3),
        pads=(1, 1, 1, 1),
        relu=True,
    )
    for channels in (64, 1024, 4096, 8192):
        yield Case(f"channel_sweep_{channels}", channels=channels)
    for channels, multiplier in ((2048, 4), (768, 8)):
        yield Case(f"multiplier_{multiplier}", channels=channels, multiplier=multiplier)
    yield Case("valid_1d", auto_pad="VALID", bias=False)
    yield Case("pointwise_2d", spatial=(2, 3), kernel=(1, 1))
    yield Case(
        "same_2d", spatial=(7, 9), kernel=(2, 3), strides=(2, 2), auto_pad="SAME_UPPER"
    )
    yield Case(
        "single_input_2d", channels=1, multiplier=6144, spatial=(5, 4), kernel=(2, 2)
    )
    yield Case(
        "reduced_same",
        spatial=(2, 6, 7),
        kernel=(1, 2, 3),
        strides=(1, 2, 1),
        dilations=(1, 2, 1),
        auto_pad="SAME_UPPER",
    )
    yield Case(
        "shared_constant", spatial=(5, 4), kernel=(2, 3), shared_filter=True, relu=True
    )
    yield Case(
        "shared_runtime",
        spatial=(5, 4),
        kernel=(2, 3),
        shared_filter=True,
        runtime_filter=True,
        runtime_bias=True,
    )


def make_case(case):
    rank = len(case.spatial)
    group = case.group or case.channels
    outputs = case.outputs or case.channels * case.multiplier
    pads = case.pads or (0,) * (rank * 2)
    strides = case.strides or (1,) * rank
    dilations = case.dilations or (1,) * rank
    result_spatial = tuple(
        (
            math.ceil(size / stride)
            if case.auto_pad == "SAME_UPPER"
            else (size + pads[i] + pads[i + rank] - dilation * (kernel - 1) - 1)
            // stride
            + 1
        )
        for i, (size, kernel, stride, dilation) in enumerate(
            zip(case.spatial, case.kernel, strides, dilations)
        )
    )
    assert min(result_spatial) > 0
    input_shape = (case.batch, case.channels) + case.spatial
    weight_shape = (outputs, case.channels // group) + case.kernel
    output_shape = (case.batch, outputs) + result_spatial
    rng = np.random.default_rng(71)
    weight = rng.normal(0, 0.2, weight_shape).astype(np.float32)
    weight *= (
        ((np.arange(outputs) % 13 + 1) / 7)
        .astype(np.float32)
        .reshape((outputs,) + (1,) * (len(weight_shape) - 1))
    )
    bias = rng.normal(0, 0.1, (outputs,)).astype(np.float32)
    value = lambda name, shape: helper.make_tensor_value_info(
        name, TensorProto.FLOAT, shape
    )
    inputs = [value("x", input_shape)]
    constants = []
    if case.runtime_filter:
        inputs.append(value("w", weight_shape))
    else:
        constants.append(numpy_helper.from_array(weight, "w"))
    if case.bias:
        if case.runtime_bias:
            inputs.append(value("b", (outputs,)))
        else:
            constants.append(numpy_helper.from_array(bias, "b"))
    attrs = dict(
        group=group, kernel_shape=case.kernel, strides=strides, dilations=dilations
    )
    if case.auto_pad == "NOTSET":
        attrs["pads"] = pads
    else:
        attrs["auto_pad"] = case.auto_pad
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w"] + (["b"] if case.bias else []),
            ["conv" if case.relu else "y"],
            **attrs,
        )
    ]
    if case.relu:
        nodes.append(helper.make_node("Relu", ["conv"], ["y"]))
    graph_outputs = [value("y", output_shape)]
    if case.shared_filter:
        inputs.append(value("x_other", input_shape))
        nodes.append(
            helper.make_node(
                "Conv",
                ["x_other", "w"] + (["b"] if case.bias else []),
                ["conv_other" if case.relu else "y_other"],
                **attrs,
            )
        )
        if case.relu:
            nodes.append(helper.make_node("Relu", ["conv_other"], ["y_other"]))
        graph_outputs.append(value("y_other", output_shape))
    model = helper.make_model(
        helper.make_graph(nodes, case.name, inputs, graph_outputs, constants),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    onnx.checker.check_model(model, full_check=True)
    feeds = []
    for seed in (0, 31, 59):
        r = np.random.default_rng(seed)
        feed = {"x": r.normal(size=input_shape).astype(np.float32)}
        if case.shared_filter:
            feed["x_other"] = r.normal(size=input_shape).astype(np.float32)
        if case.runtime_filter:
            feed["w"] = weight + r.normal(0, 0.01, weight_shape).astype(np.float32)
        if case.runtime_bias:
            feed["b"] = bias + r.normal(0, 0.01, bias.shape).astype(np.float32)
        feeds.append(feed)
    depthwise = group == case.channels and outputs <= 4832
    return model, feeds, outputs, depthwise


def reference_model(model, case):
    # Some CPU reference kernels reject SAME padding with dilation. Resolve
    # that padding explicitly for the reference only; convert the original
    # auto_pad model so the compiler's padding calculation is still tested.
    if case.auto_pad != "SAME_UPPER" or not any(d > 1 for d in case.dilations):
        return model.SerializeToString()
    reference = onnx.ModelProto()
    reference.CopyFrom(model)
    strides = case.strides or (1,) * len(case.spatial)
    totals = [
        max(
            0,
            (math.ceil(size / stride) - 1) * stride
            + dilation * (kernel - 1)
            + 1
            - size,
        )
        for size, stride, kernel, dilation in zip(
            case.spatial, strides, case.kernel, case.dilations
        )
    ]
    pads = [p // 2 for p in totals] + [p - p // 2 for p in totals]
    for node in reference.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = [a for a in node.attribute if a.name != "auto_pad"]
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.attribute.append(helper.make_attribute("pads", pads))
    onnx.checker.check_model(reference, full_check=True)
    return reference.SerializeToString()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    selected = [c for c in cases() if not args.case or c.name in args.case]
    assert selected
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    reports = []
    comparisons = 0
    for case in selected:
        model, feeds, outputs, depthwise = make_case(case)
        source = args.work_dir / (case.name + ".onnx")
        onnx.save(model, source)
        reference = ort.InferenceSession(
            reference_model(model, case), options, providers=["CPUExecutionProvider"]
        )
        output_names = [item.name for item in reference.get_outputs()]
        expected = [
            dict(zip(output_names, reference.run(None, feed))) for feed in feeds
        ]
        del reference
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                target = args.work_dir / f"{case.name}.{layout}.{int(optimize)}.tflite"
                with target.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            str(args.converter.resolve()),
                            str(source),
                            "-o",
                            str(target),
                            "--layout-mode",
                            layout,
                            "--verify-each",
                            "--optimize-tfl" if optimize else "--no-optimize-tfl",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                        env=dict(os.environ, TMPDIR=str(args.work_dir.resolve())),
                    )
                fb = schema.Model.GetRootAsModel(target.read_bytes(), 0)
                graph = fb.Subgraphs(0)
                codes = [
                    fb.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                    for i in range(graph.OperatorsLength())
                ]
                conv_code = (
                    schema.BuiltinOperator.DEPTHWISE_CONV_2D
                    if depthwise
                    else schema.BuiltinOperator.CONV_2D
                )
                conv_count = 2 if case.shared_filter else 1
                assert codes.count(conv_code) == conv_count, (
                    case.name,
                    layout,
                    optimize,
                    codes,
                )
                assert (
                    sum(
                        codes.count(c)
                        for c in (
                            schema.BuiltinOperator.CONV_2D,
                            schema.BuiltinOperator.DEPTHWISE_CONV_2D,
                            schema.BuiltinOperator.CONV_3D,
                        )
                    )
                    == conv_count
                )
                assert not set(codes).intersection(
                    {
                        schema.BuiltinOperator.CUSTOM,
                        schema.BuiltinOperator.WHILE,
                        schema.BuiltinOperator.IF,
                        schema.BuiltinOperator.SPLIT,
                        schema.BuiltinOperator.SPLIT_V,
                        schema.BuiltinOperator.CONCATENATION,
                    }
                )
                max_rank = max(
                    graph.Tensors(i).ShapeLength() for i in range(graph.TensorsLength())
                )
                assert max_rank <= max(4, len(case.spatial) + 2)
                conv = graph.Operators(codes.index(conv_code))
                filter_tensor = graph.Tensors(conv.Inputs(1))
                assert filter_tensor.ShapeLength() == 4
                if not depthwise:
                    assert filter_tensor.Shape(0) == outputs
                    assert filter_tensor.Shape(3) == case.channels // (
                        case.group or case.channels
                    )
                if optimize and not case.runtime_filter:
                    assert fb.Buffers(filter_tensor.Buffer()).DataLength() > 0
                # With TFL optimization disabled, identical runtime-filter
                # layout transposes need not be CSE'd. Constant storage must
                # still be shared, and the default optimized path must share
                # either kind of filter.
                if case.shared_filter and (optimize or not case.runtime_filter):
                    weight_ids = {
                        graph.Operators(i).Inputs(1)
                        for i, code in enumerate(codes)
                        if code == conv_code
                    }
                    assert len(weight_ids) == 1, "Shared filter was duplicated"
                max_abs = 0.0
                for delegate in (False, True):
                    # A native-kernel regression must fail the test promptly,
                    # not leave an indefinitely spinning inference process.
                    signal.alarm(30)
                    runtime = tf.lite.Interpreter(
                        model_path=str(target),
                        num_threads=1,
                        experimental_op_resolver_type=(
                            tf.lite.experimental.OpResolverType.AUTO
                            if delegate
                            else tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                        ),
                    )
                    runner = runtime.get_signature_runner("serving_default")
                    details = runner.get_input_details()
                    for feed, expected_value in zip(feeds, expected):
                        actual = runner(
                            **{
                                name: np.ascontiguousarray(
                                    onnx_to_tfl_input(
                                        array, tuple(details[name]["shape"]), layout
                                    )
                                )
                                for name, array in feed.items()
                            }
                        )
                        assert actual.keys() == expected_value.keys()
                        for name, reference_value in expected_value.items():
                            value = tfl_to_onnx_output(
                                actual[name], reference_value.shape, layout
                            )
                            np.testing.assert_allclose(
                                value, reference_value, rtol=3e-5, atol=3e-5
                            )
                            max_abs = max(
                                max_abs, float(np.max(np.abs(value - reference_value)))
                            )
                            comparisons += 1
                    del runner, runtime
                    signal.alarm(0)
                reports.append(
                    dict(
                        case=case.name,
                        layout=layout,
                        optimize=optimize,
                        operators=len(codes),
                        max_rank=max_rank,
                        max_abs=max_abs,
                        passed=True,
                    )
                )
                print("PASS", case.name, layout, optimize, flush=True)
    report = dict(conversions=len(reports), comparisons=comparisons, cases=reports)
    (args.work_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
