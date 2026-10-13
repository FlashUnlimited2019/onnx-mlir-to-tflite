#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate bounded NMS consumers against ORT, including unsafe near-matches."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def make_model(
    batches=2,
    classes=3,
    boxes=7,
    limit=3,
    center=1,
    iou=0.5,
    threshold=0.0,
    opset=12,
    mutation=None,
):
    nodes, initializers = [], []

    def const(name, value, dtype=np.int64):
        initializers.append(nh.from_array(np.asarray(value, dtype), name))
        return name

    def node(kind, inputs, output, **attrs):
        nodes.append(h.make_node(kind, inputs, [output], **attrs))
        return output

    zero, one = const("zero", 0), const("one", 1)
    c = const("classes", classes)
    k = const("limit", limit)
    const("iou", iou, np.float32)
    const("threshold", threshold, np.float32)
    node(
        "NonMaxSuppression",
        ["boxes", "scores", k, "iou", "threshold"],
        "selected",
        center_point_box=center,
    )
    node("Shape", ["selected"], "shape")
    node("Gather", ["shape", zero], "length", axis=0)
    range_start = const("start", 1 if mutation == "range_start" else 0)
    range_limit = const("frozen_length", 2) if mutation == "frozen_length" else "length"
    node("Range", [range_start, range_limit, one], "positions")
    node("Gather", ["selected", zero], "batch", axis=1)
    node("Gather", ["selected", one], "class", axis=1)
    multiplier = const("group_multiplier", classes + 1) if mutation == "group" else c
    node("Mul", ["batch", multiplier], "group_base")
    node("Add", ["group_base", "class"], "group")

    def unsqueeze(source, output, axis):
        if opset >= 13:
            return node("Unsqueeze", [source, const(output + "_axis", [axis])], output)
        return node("Unsqueeze", [source], output, axes=[axis])

    unsqueeze("group", "group_i", 1)
    unsqueeze("group", "group_j", 0)
    node("Equal", ["group_i", "group_j"], "same_group")
    unsqueeze("positions", "i", 1)
    unsqueeze("positions", "j", 0)
    node("LessOrEqual", ["i", "j"] if mutation == "reverse" else ["j", "i"], "prefix")
    node("And", ["same_group", "prefix"], "mask")
    node("Cast", ["mask"], "counts", to=T.INT64)
    if opset >= 13:
        node("ReduceSum", ["counts", const("reduce_axes", [1])], "count", keepdims=0)
    else:
        node("ReduceSum", ["counts"], "count", axes=[1], keepdims=0)
    subtract = const("subtract", 2) if mutation == "ordinal" else one
    node("Sub", ["count", subtract], "ordinal")
    stride = const("stride", limit + 1) if mutation == "stride" else k
    node("Mul", ["group", stride], "group_offset")
    node("Add", ["group_offset", "ordinal"], "offset")
    unsqueeze("offset", "indices", 1)
    rows = batches * classes * limit
    node("ScatterND", ["padding", "indices", "selected"], "output")
    inputs = [
        h.make_tensor_value_info("boxes", T.FLOAT, [batches, boxes, 4]),
        h.make_tensor_value_info("scores", T.FLOAT, [batches, classes, boxes]),
        h.make_tensor_value_info("padding", T.INT64, [rows, 3]),
    ]
    outputs = [h.make_tensor_value_info("output", T.INT64, [rows, 3])]
    if mutation == "shared":
        outputs.append(h.make_tensor_value_info("selected", T.INT64, [None, 3]))
    elif mutation == "shared_count":
        unsqueeze("length", "shared_length", 0)
        outputs.append(h.make_tensor_value_info("shared_length", T.INT64, [1]))
    elif mutation == "runtime_threshold":
        initializers = [v for v in initializers if v.name != "threshold"]
        inputs.append(h.make_tensor_value_info("threshold", T.FLOAT, [1]))
    model = h.make_model(
        h.make_graph(nodes, "bounded_suppression", inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", opset)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model


def feeds(config):
    b, c, n, k, center, iou, threshold, _ = config
    rng = np.random.default_rng(91237)
    for sample in range(40):
        boxes = rng.normal(size=(b, n, 4)).astype(np.float32)
        if center:
            boxes[..., 2:] = np.abs(boxes[..., 2:])
        scores = rng.normal(size=(b, c, n)).astype(np.float32)
        if sample == 0:
            scores.fill(threshold - 1)  # empty output
        elif sample == 1:
            scores.fill(threshold)  # equality is excluded
        elif sample in (2, 3):
            scores.fill(threshold + 1)  # deterministic index tie-break
            if sample == 2:
                boxes[:] = [0, 0, 1, 1]  # identical boxes
        elif sample == 4:
            scores[:] = np.arange(n, dtype=np.float32) + threshold + 1
            boxes[:] = 0  # degenerate boxes must not suppress others
        elif sample == 5:
            scores[:] = threshold + 1
            if center:
                boxes[..., 2:] *= -1  # invalid sizes do not suppress others
            else:
                boxes = boxes[..., [2, 3, 0, 1]]  # reversed corners
        elif sample == 6:
            scores[..., ::2] = np.inf
            scores[..., 1::2] = np.nan
        elif sample == 7:
            scores[:] = threshold + 1
            # Adjacent unit boxes have exactly zero intersection.
            boxes[:] = [[i, 0, 1, 1] if center else [0, i, 1, i + 1] for i in range(n)]
        elif sample == 8:
            scores[:] = threshold + 1
            # Identical unit boxes exercise IoU==1.
            boxes[:] = [0, 0, 1, 1]
        elif sample == 9:
            scores[:] = threshold - 1
            scores[..., :2] = threshold + 1
            # Exactly representable IoU==0.5.
            boxes[:] = [1, 0.5, 2, 1] if center else [0, 0, 1, 2]
            if n > 1:
                boxes[:, 1] = [0.5, 0.5, 1, 1] if center else [0, 0, 1, 1]
        padding = rng.integers(-(2**60), 2**60, (b * c * k, 3), dtype=np.int64)
        yield {"boxes": boxes, "scores": scores, "padding": padding}


def structure(path):
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    graph = model.Subgraphs(0)
    names = {
        v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)
    }
    counts = Counter(
        names[model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()]
        for i in range(graph.OperatorsLength())
    )
    assert not set(counts) & {
        "CUSTOM",
        "IF",
        "WHILE",
        "NON_MAX_SUPPRESSION_V4",
        "NON_MAX_SUPPRESSION_V5",
    }
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        assert tensor.ShapeLength() <= 3
        assert (
            all(d >= 0 for d in tensor.ShapeAsNumpy()) if tensor.ShapeLength() else True
        )
        if tensor.ShapeSignatureLength():
            assert all(d >= 0 for d in tensor.ShapeSignatureAsNumpy())
    return dict(counts)


def convert(driver, source, dest, layout, optimize, should_pass=True):
    cmd = [
        sys.executable,
        str(driver),
        str(source),
        "-o",
        str(dest),
        "--layout-mode",
        layout,
    ]
    if not optimize:
        cmd.append("--no-optimize-tfl")
    run = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    dest.with_suffix(".log").write_text(run.stdout)
    if should_pass:
        if run.returncode:
            raise RuntimeError(run.stdout[-6000:])
    else:
        assert run.returncode and not dest.exists(), run.stdout
        assert (
            "unsupported ONNX operation" in run.stdout
            or "dynamic tensor shape" in run.stdout
        ), run.stdout


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--driver", type=Path, default=repo / "build/bin/onnx-to-tflite"
    )
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (b, c, n, k, center, iou, threshold, opset)
        for center in (0, 1)
        for b, c, n, k, iou, threshold, opset in [
            (1, 1, 6, 3, 0.5, 0.0, 12),
            (2, 3, 7, 2, 0.5, -0.5, 13),
            (2, 1, 5, 1, 0.0, 0.25, 12),
            (1, 3, 4, 6, 1.0, 0.5, 13),
            (1, 1, 1, 2, 0.5, 0.0, 12),
            (2, 2, 5, 3, 0.0, -1.0, 13),
            (1, 2, 65, 2, 0.5, 0.0, 12),
            (2, 4, 7, 4, 0.5, -0.5, 13),  # reuse boxes across classes/batches
            (1, 17, 64, 3, 0.5, 0.0, 12),  # bound cache by batches, not classes
            (2, 3, 65, 4, 0.5, 0.0, 13),  # large-box linear-workspace fallback
            (17, 2, 64, 3, 0.5, 0.0, 12),  # pairwise cache budget fallback
        ]
    ]
    results = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    for case, config in enumerate(configs):
        source = args.work_dir / f"case_{case}.onnx"
        onnx.save(make_model(*config), source)
        session = ort.InferenceSession(
            str(source), options, providers=["CPUExecutionProvider"]
        )
        samples = list(feeds(config))
        expected = [session.run(None, feed)[0] for feed in samples]
        for layout in ("legacy", "experimental"):
            for optimize in (False, True):
                dest = args.work_dir / f"case_{case}_{layout}_{int(optimize)}.tflite"
                convert(args.driver, source, dest, layout, optimize)
                counts = structure(dest)
                for delegate in (False, True):
                    kwargs = (
                        {}
                        if delegate
                        else {
                            "experimental_op_resolver_type": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                        }
                    )
                    runtime = tf.lite.Interpreter(
                        model_path=str(dest), num_threads=1, **kwargs
                    )
                    runtime.allocate_tensors()
                    for sample, (feed, reference) in enumerate(zip(samples, expected)):
                        for detail, metadata in zip(
                            runtime.get_input_details(), session.get_inputs()
                        ):
                            runtime.set_tensor(detail["index"], feed[metadata.name])
                        runtime.invoke()
                        actual = runtime.get_tensor(
                            runtime.get_output_details()[0]["index"]
                        )
                        np.testing.assert_array_equal(
                            actual,
                            reference,
                            err_msg=f"case={case} layout={layout} opt={optimize} delegate={delegate} sample={sample}",
                        )
                    results.append(
                        {
                            "case": case,
                            "layout": layout,
                            "optimize": optimize,
                            "delegate": delegate,
                            "samples": len(samples),
                            "operators": sum(counts.values()),
                        }
                    )
                print(
                    f"PASS case={case} layout={layout} optimize={optimize}: 40 inputs, both runtimes",
                    flush=True,
                )
    rejected = (
        "range_start",
        "frozen_length",
        "group",
        "reverse",
        "ordinal",
        "stride",
        "shared",
        "shared_count",
        "runtime_threshold",
    )
    for mutation in rejected:
        source = args.work_dir / f"reject_{mutation}.onnx"
        onnx.save(make_model(mutation=mutation), source)
        for layout in ("legacy", "experimental"):
            convert(
                args.driver,
                source,
                args.work_dir / f"reject_{mutation}_{layout}.tflite",
                layout,
                True,
                should_pass=False,
            )
        print(f"PASS rejection: {mutation}", flush=True)
    (args.work_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"PASS: {len(configs)} fixtures, {len(results) * 40} exact output comparisons, {len(rejected)*2} guarded rejections"
    )


if __name__ == "__main__":
    main()
