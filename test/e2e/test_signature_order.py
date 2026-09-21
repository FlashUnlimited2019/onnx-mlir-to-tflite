#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""Check signature-only edits, aliasing, malformed metadata and large offsets."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema


DRIVER = (
    Path(__file__).resolve().parents[2] / "src/Tools/onnx-to-tflite/onnx-to-tflite.py"
)
SPEC = importlib.util.spec_from_file_location("signature_driver", DRIVER)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def fixture():
    model = schema.ModelT()
    model.version = 3
    graph = schema.SubGraphT()
    graph.name = "main"
    graph.inputs = [2, 0, 1]
    graph.outputs = [2, 2, 0]
    graph.tensors = []
    for i in range(3):
        tensor = schema.TensorT()
        tensor.name = f"tensor{i}"
        tensor.shape = [1]
        graph.tensors.append(tensor)
    model.subgraphs = [graph]
    signature = schema.SignatureDefT()
    signature.signatureKey = "serving_default"
    order = {
        "version": 1,
        "signature_key": "serving_default",
        "inputs": ['z/"input":0', "a_input", "m_输入"],
        "outputs": ["z_alias", "a_alias", "m_output"],
    }

    def maps(names, indices):
        entries = []
        for name, index in sorted(zip(names, indices)):
            entry = schema.TensorMapT()
            entry.name, entry.tensorIndex = name, index
            entries.append(entry)
        return entries

    signature.inputs = maps(order["inputs"], graph.inputs)
    signature.outputs = maps(order["outputs"], graph.outputs)
    model.signatureDefs = [signature]
    empty, metadata_buffer, weight_buffer = (schema.BufferT() for _ in range(3))
    metadata_buffer.data = json.dumps(order, ensure_ascii=False).encode()
    weight_buffer.data = b"constant-payload-must-not-change"
    model.buffers = [empty, metadata_buffer, weight_buffer]
    metadata = schema.MetadataT()
    metadata.name = "onnx_mlir.signature_order"
    metadata.buffer = 1
    model.metadata = [metadata]
    return model, order


def pack(model):
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


class SignatureOrderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="signature-order-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.tflite"

    def verify_order(self, header, order):
        model = schema.Model.GetRootAsModel(header, 0)
        graph, signature = model.Subgraphs(0), model.SignatureDefs(0)
        for label, length, entry, main in (
            ("inputs", signature.InputsLength(), signature.Inputs, graph.Inputs),
            ("outputs", signature.OutputsLength(), signature.Outputs, graph.Outputs),
        ):
            self.assertEqual(
                [entry(i).Name().decode() for i in range(length)], order[label]
            )
            self.assertEqual(
                [entry(i).TensorIndex() for i in range(length)],
                [main(i) for i in range(length)],
            )

    def test_only_references_change_and_idempotent(self):
        model, order = fixture()
        before = pack(model)
        self.path.write_bytes(before)
        with self.path.open("rb") as stream:
            plan = driver._SignatureFlatBuffer(stream).plan()
        allowed = {position + i for position, _ in plan for i in range(4)}
        self.assertTrue(allowed)
        driver._normalize_signature_order(self.path)
        after = self.path.read_bytes()
        self.assertEqual(len(before), len(after))
        changed = {i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]}
        self.assertTrue(changed)
        self.assertLessEqual(changed, allowed)
        self.verify_order(after, order)
        driver._normalize_signature_order(self.path)
        self.assertEqual(self.path.read_bytes(), after)

    def test_no_python_package_dependencies(self):
        model, order = fixture()
        self.path.write_bytes(pack(model))
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import runpy, pathlib, sys; "
                "runpy.run_path(sys.argv[1])['_normalize_signature_order']"
                "(pathlib.Path(sys.argv[2]))",
                str(DRIVER),
                str(self.path),
            ],
            check=True,
        )
        self.verify_order(self.path.read_bytes(), order)

    def test_large_sparse_file_and_external_metadata(self):
        for external_metadata in (False, True):
            with self.subTest(external_metadata=external_metadata):
                model, order = fixture()
                weight = bytes(model.buffers[2].data)
                weight_offset = (1 << 32) + 4096
                model.buffers[2].data = None
                model.buffers[2].offset = weight_offset
                model.buffers[2].size = len(weight)
                payload = bytes(model.buffers[1].data)
                metadata_offset = weight_offset + 4096
                if external_metadata:
                    model.buffers[1].data = None
                    model.buffers[1].offset = metadata_offset
                    model.buffers[1].size = len(payload)
                before = pack(model)
                with self.path.open("wb") as stream:
                    stream.write(before)
                    stream.seek(weight_offset)
                    stream.write(weight)
                    if external_metadata:
                        stream.seek(metadata_offset)
                        stream.write(payload)
                old_size = self.path.stat().st_size
                driver._normalize_signature_order(self.path)
                self.assertEqual(self.path.stat().st_size, old_size)
                with self.path.open("rb") as stream:
                    after = stream.read(len(before))
                    stream.seek(weight_offset)
                    self.assertEqual(stream.read(len(weight)), weight)
                    if external_metadata:
                        stream.seek(metadata_offset)
                        self.assertEqual(stream.read(len(payload)), payload)
                self.verify_order(after, order)
                parsed = schema.Model.GetRootAsModel(after, 0)
                self.assertEqual(parsed.Buffers(2).Offset(), weight_offset)
                self.assertEqual(parsed.Buffers(2).Size(), len(weight))

    def test_reject_invalid_mappings_before_any_write(self):
        for case in (
            "missing_metadata",
            "duplicate_metadata",
            "invalid_buffer",
            "missing_signature",
            "multiple_signatures",
            "wrong_key",
            "wrong_subgraph",
            "wrong_main",
            "missing_input",
            "duplicate_input",
            "wrong_input_index",
            "wrong_output_index",
            "negative_main_index",
            "out_of_range_tensor",
            "invalid_json",
            "duplicate_order_name",
            "unknown_order_name",
            "wrong_order_count",
            "wrong_version",
            "wrong_schema",
        ):
            with self.subTest(case=case):
                model, order = fixture()
                signature, graph = model.signatureDefs[0], model.subgraphs[0]
                if case == "missing_metadata":
                    model.metadata = []
                elif case == "duplicate_metadata":
                    model.metadata *= 2
                elif case == "invalid_buffer":
                    model.metadata[0].buffer = 999
                elif case == "missing_signature":
                    model.signatureDefs = []
                elif case == "multiple_signatures":
                    model.signatureDefs *= 2
                elif case == "wrong_key":
                    signature.signatureKey = "other"
                elif case == "wrong_subgraph":
                    signature.subgraphIndex = 1
                elif case == "wrong_main":
                    graph.name = "other"
                elif case == "missing_input":
                    signature.inputs.pop()
                elif case == "duplicate_input":
                    signature.inputs[1].name = signature.inputs[0].name
                elif case == "wrong_input_index":
                    signature.inputs[0].tensorIndex = 1
                elif case == "wrong_output_index":
                    signature.outputs[0].tensorIndex = 1
                elif case == "negative_main_index":
                    graph.inputs[0] = -1
                elif case == "out_of_range_tensor":
                    graph.inputs[0] = signature.inputs[2].tensorIndex = 99
                elif case == "invalid_json":
                    model.buffers[1].data = b"{"
                elif case in (
                    "duplicate_order_name",
                    "unknown_order_name",
                    "wrong_order_count",
                    "wrong_version",
                ):
                    if case == "duplicate_order_name":
                        order["inputs"][1] = order["inputs"][0]
                    elif case == "unknown_order_name":
                        order["outputs"][0] = "unknown"
                    elif case == "wrong_order_count":
                        order["outputs"].pop()
                    else:
                        order["version"] = 2
                    model.buffers[1].data = json.dumps(order).encode()
                elif case == "wrong_schema":
                    model.version = 4
                before = pack(model)
                self.path.write_bytes(before)
                with self.assertRaises(RuntimeError):
                    driver._normalize_signature_order(self.path)
                self.assertEqual(self.path.read_bytes(), before)

    def test_truncated_and_out_of_bounds_buffers(self):
        model, _ = fixture()
        valid = pack(model)
        candidates = [valid[:i] for i in (0, 4, 8, 16, len(valid) // 2)]
        candidates.append(struct.pack("<I", 0xFFFFFFFC) + valid[4:])
        candidates.append(valid[:4] + b"NOPE" + valid[8:])
        for before in candidates:
            self.path.write_bytes(before)
            with self.assertRaises(RuntimeError):
                driver._normalize_signature_order(self.path)
            self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
