#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

"""ONNX -> ONNX dialect -> TFL dialect -> TFLite FlatBuffer driver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
from typing import List, Optional


def _find_tool(name: str, explicit: Optional[str], script: Path) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"configured {name} does not exist: {candidate}")

    from_path = shutil.which(name)
    if from_path:
        return Path(from_path).resolve()

    bin_dir = script.resolve().parent
    candidates = [bin_dir / name]
    tensorflow_tool = {
        "flatbuffer_translate": "flatbuffer_translate",
        "litert-opt": "litert-opt",
    }.get(name)
    if tensorflow_tool:
        # Support both the source-tree script and its CMake build-tree copy.
        # Looking through ancestors avoids depending on a particular CMake
        # configuration directory (Release/bin versus bin symlinks).
        for ancestor in (bin_dir, *bin_dir.parents):
            candidates.extend(
                [
                    ancestor
                    / "tensorflow"
                    / "bazel-bin"
                    / "tensorflow"
                    / "compiler"
                    / "mlir"
                    / "lite"
                    / tensorflow_tool,
                    ancestor / "build" / "tensorflow" / tensorflow_tool,
                ]
            )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"unable to find {name}; searched PATH and: {searched}")


def _run(command: List[str], stage: str) -> None:
    print("+ " + " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lower a static FP32 ONNX model through MLIR to TFLite"
    )
    parser.add_argument("input", type=Path, help="input .onnx model")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--onnx-mlir", help="path to the onnx-mlir binary")
    parser.add_argument("--onnx-mlir-opt", help="path to onnx-mlir-opt")
    parser.add_argument(
        "--flatbuffer-translate", help="path to TensorFlow flatbuffer_translate"
    )
    parser.add_argument("--litert-opt", help="path to TensorFlow litert-opt")
    optimize_group = parser.add_mutually_exclusive_group()
    optimize_group.add_argument(
        "--optimize-tfl",
        dest="optimize_tfl",
        action="store_true",
        help="run TensorFlow TFL canonicalization/fusion before export",
    )
    optimize_group.add_argument(
        "--no-optimize-tfl",
        dest="optimize_tfl",
        action="store_false",
        help="skip TensorFlow TFL canonicalization/fusion",
    )
    parser.set_defaults(optimize_tfl=True)
    parser.add_argument("--dump-onnx-mlir", action="store_true")
    parser.add_argument("--dump-tfl-mlir", action="store_true")
    parser.add_argument("--keep-intermediate-files", action="store_true")
    verify_group = parser.add_mutually_exclusive_group()
    verify_group.add_argument("--verify-each", dest="verify_each", action="store_true")
    verify_group.add_argument(
        "--no-verify-each", dest="verify_each", action="store_false"
    )
    parser.set_defaults(verify_each=True)
    parser.add_argument(
        "--layout-mode",
        choices=("legacy", "experimental"),
        default="legacy",
        help="select the ONNX-to-TFL per-value layout strategy",
    )
    parser.add_argument(
        "--layout-diagnostics",
        choices=("none", "summary", "values"),
        default="none",
        help="print per-value layout analysis diagnostics",
    )
    buffer_offset_group = parser.add_mutually_exclusive_group()
    buffer_offset_group.add_argument(
        "--use-buffer-offset",
        dest="use_buffer_offset",
        action="store_true",
        help=(
            "append constants outside the FlatBuffer metadata and reference "
            "them by file offset; enabled automatically for a model with a "
            "large adjacent .data file"
        ),
    )
    buffer_offset_group.add_argument(
        "--no-use-buffer-offset",
        dest="use_buffer_offset",
        action="store_false",
        help="disable TFLite buffer offsets",
    )
    parser.set_defaults(use_buffer_offset=None)
    return parser.parse_args()


def _should_use_buffer_offset(input_path: Path) -> bool:
    """Detect the common ONNX `<model>.onnx.data` large-model layout."""
    candidates = {
        input_path,
        input_path.with_name(input_path.name + ".data"),
        input_path.with_suffix(".data"),
    }
    total_size = sum(
        candidate.stat().st_size for candidate in candidates if candidate.is_file()
    )
    # TensorFlow begins automatically switching near this threshold. Detect it
    # before conversion too, so intermediate files avoid the size-limited /tmp
    # tmpfs and round-trip verification does not materialize every constant.
    return total_size >= 1536 * 1024 * 1024


class _SignatureFlatBuffer:
    """Bounded access to signature metadata using only the Python stdlib.

    Field numbers follow TensorFlow's schema.fbs. Do not unpack/repack the
    model: large files can contain weights addressed by absolute file offsets.
    TensorFlow's full schema verifier still runs after normalization.
    """

    def __init__(self, stream):
        self.stream = stream
        self.size = os.fstat(stream.fileno()).st_size

    def check(self, position, size):
        if position < 0 or size < 0 or position > self.size - size:
            raise RuntimeError("signature metadata references bytes outside the file")

    def read(self, position, size):
        self.check(position, size)
        self.stream.seek(position)
        data = self.stream.read(size)
        if len(data) != size:
            raise RuntimeError("truncated signature metadata")
        return data

    def scalar(self, position, fmt):
        return struct.unpack(fmt, self.read(position, struct.calcsize(fmt)))[0]

    def indirect(self, position):
        offset = self.scalar(position, "<I")
        if offset < 4:
            raise RuntimeError("invalid signature metadata offset")
        target = position + offset
        self.check(target, 4)
        return target

    def field(self, table, slot, width=4):
        vtable = table - self.scalar(table, "<i")
        vsize = self.scalar(vtable, "<H")
        object_size = self.scalar(vtable + 2, "<H")
        if vsize < 4 or vsize % 2 or object_size < 4:
            raise RuntimeError("invalid signature metadata table")
        self.check(vtable, vsize)
        self.check(table, object_size)
        entry = 4 + slot * 2
        offset = self.scalar(vtable + entry, "<H") if entry < vsize else 0
        if not offset:
            return None
        if offset < 4 or offset + width > object_size:
            raise RuntimeError("invalid signature metadata field")
        return table + offset

    def uint(self, table, slot, fmt="<I"):
        position = self.field(table, slot, struct.calcsize(fmt))
        return self.scalar(position, fmt) if position is not None else 0

    def vector(self, table, slot, width=4):
        position = self.field(table, slot)
        if position is None:
            return 0, 0
        vector = self.indirect(position)
        count = self.scalar(vector, "<I")
        self.check(vector + 4, count * width)
        return vector + 4, count

    def string(self, table, slot):
        position = self.field(table, slot)
        if position is None:
            raise RuntimeError("missing signature metadata string")
        start = self.indirect(position)
        size = self.scalar(start, "<I")
        if size > 16 * 1024 * 1024:
            raise RuntimeError("signature metadata string exceeds 16 MiB")
        if self.read(start + 4 + size, 1) != b"\0":
            raise RuntimeError("unterminated signature metadata string")
        return self.read(start + 4, size).decode("utf-8")

    def order(self, model):
        metadata, count = self.vector(model, 6)
        matches = []
        for i in range(count):
            entry = self.indirect(metadata + 4 * i)
            if self.string(entry, 0) == "onnx_mlir.signature_order":
                matches.append(self.uint(entry, 1))
        if len(matches) != 1:
            raise RuntimeError(
                "expected one onnx_mlir.signature_order metadata entry; "
                "use an onnx-mlir-opt built with this driver"
            )
        buffers, count = self.vector(model, 4)
        if matches[0] >= count:
            raise RuntimeError("invalid signature order buffer index")
        buffer = self.indirect(buffers + 4 * matches[0])
        start, size = self.vector(buffer, 0, 1)
        offset = self.uint(buffer, 1, "<Q")
        if offset > 1:
            if size:
                raise RuntimeError("signature order buffer has ambiguous storage")
            start, size = offset, self.uint(buffer, 2, "<Q")
        if not 0 < size <= 16 * 1024 * 1024:
            raise RuntimeError("invalid signature order metadata size")
        order = json.loads(self.read(start, size))
        if (
            not isinstance(order, dict)
            or type(order.get("version")) is not int
            or order["version"] != 1
            or order.get("signature_key") != "serving_default"
        ):
            raise RuntimeError("unsupported signature order metadata")
        return order

    def plan(self):
        if self.read(4, 4) != b"TFL3":
            raise RuntimeError("exported file does not have the TFL3 identifier")
        model = self.indirect(0)
        if self.uint(model, 0) != 3:
            raise RuntimeError("unsupported TFLite model schema version")
        order = self.order(model)
        subgraphs, subgraph_count = self.vector(model, 2)
        signatures, count = self.vector(model, 7)
        if subgraph_count < 1 or count != 1:
            raise RuntimeError(
                "expected one serving_default signature and a main graph"
            )
        graph = self.indirect(subgraphs)
        signature = self.indirect(signatures)
        if (
            self.string(signature, 2) != order["signature_key"]
            or self.uint(signature, 4) != 0
            or self.string(graph, 4) != "main"
        ):
            raise RuntimeError("serving_default does not reference the main graph")
        _, tensor_count = self.vector(graph, 0)
        patches = []
        for slot, label in enumerate(("inputs", "outputs")):
            names = order.get(label)
            if (
                not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
            ):
                raise RuntimeError(f"invalid ordered signature {label} names")
            main_vector, main_count = self.vector(graph, slot + 1)
            vector, count = self.vector(signature, slot)
            if len(names) != main_count or count != main_count:
                raise RuntimeError(f"signature/main {label} count mismatch")
            entries = {}
            for i in range(count):
                entry = self.indirect(vector + 4 * i)
                name = self.string(entry, 0)
                if name in entries:
                    raise RuntimeError(f"duplicate signature {label} name: {name}")
                entries[name] = (entry, self.uint(entry, 1))
            if set(entries) != set(names):
                raise RuntimeError(
                    f"signature {label} names differ from the ONNX entry"
                )
            for i, name in enumerate(names):
                entry, tensor_index = entries[name]
                if tensor_index >= tensor_count or tensor_index != self.scalar(
                    main_vector + 4 * i, "<i"
                ):
                    raise RuntimeError(
                        f"signature/main {label} mapping mismatch: {name}"
                    )
                position = vector + 4 * i
                offset = entry - position
                if not 4 <= offset <= 0xFFFFFFFF:
                    raise RuntimeError("signature entry cannot be reordered in place")
                replacement = struct.pack("<I", offset)
                if self.read(position, 4) != replacement:
                    patches.append((position, replacement))
        if len({position for position, _ in patches}) != len(patches):
            raise RuntimeError("overlapping signature vectors")
        return patches


def _normalize_signature_order(path: Path) -> None:
    """Reorder only TensorMap vector references in the unpublished output.

    Names, tensor indices, tables, weights, file length and buffer offsets stay
    unchanged. Validate both directions completely before applying any writes.
    """
    try:
        with path.open("r+b") as stream:
            reader = _SignatureFlatBuffer(stream)
            for position, replacement in reader.plan():
                stream.seek(position)
                stream.write(replacement)
            stream.flush()
            if reader.plan():
                raise RuntimeError("signature order verification failed")
    except (ValueError, struct.error) as error:
        raise RuntimeError(f"invalid signature metadata: {error}") from error


def main() -> int:
    args = _parse_args()
    script = Path(__file__)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        print(f"error: input model does not exist: {input_path}", file=sys.stderr)
        return 2
    if input_path.suffix.lower() != ".onnx":
        print(f"error: expected a .onnx input, got: {input_path}", file=sys.stderr)
        return 2

    use_buffer_offset = (
        _should_use_buffer_offset(input_path)
        if args.use_buffer_offset is None
        else args.use_buffer_offset
    )
    if use_buffer_offset:
        print(
            "large-model mode: constants will use TFLite buffer offsets",
            file=sys.stderr,
        )

    try:
        onnx_mlir = _find_tool("onnx-mlir", args.onnx_mlir, script)
        onnx_mlir_opt = _find_tool("onnx-mlir-opt", args.onnx_mlir_opt, script)
        flatbuffer_translate = _find_tool(
            "flatbuffer_translate",
            args.flatbuffer_translate or os.environ.get("FLATBUFFER_TRANSLATE"),
            script,
        )
        litert_opt = None
        if args.optimize_tfl:
            litert_opt = _find_tool(
                "litert-opt",
                args.litert_opt or os.environ.get("LITERT_OPT"),
                script,
            )
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_dir = output_path.with_suffix(output_path.suffix + ".intermediates")
    if args.keep_intermediate_files and kept_dir.exists():
        print(
            "error: intermediate directory already exists; refusing to overwrite: "
            f"{kept_dir}",
            file=sys.stderr,
        )
        return 2
    temporary = tempfile.TemporaryDirectory(
        prefix="onnx-to-tflite-",
        # Large textual MLIR intermediates can exceed a tmpfs even when the
        # final offset-buffer TFLite file fits comfortably on disk.
        dir=output_path.parent if use_buffer_offset else None,
    )
    work_dir = Path(temporary.name)
    try:
        onnx_prefix = work_dir / "model"
        onnx_mlir_path = work_dir / "model.onnx.mlir"
        unoptimized_tfl_mlir_path = work_dir / "model.unoptimized.tfl.mlir"
        tfl_mlir_path = work_dir / "model.tfl.mlir"
        flatbuffer_path = work_dir / "model.tflite"
        verified_mlir_path = work_dir / "verified-roundtrip.mlir"

        _run(
            [
                str(onnx_mlir),
                str(input_path),
                "--EmitONNXIR",
                # Keep Conv in ONNX dialect for the future layout-aware
                # ONNXToTFL lowering instead of decomposing it to Im2Col/Gemm.
                "--disable-conv-to-matmul",
                "-o",
                str(onnx_prefix),
            ],
            "ONNX import",
        )
        conversion_output = (
            unoptimized_tfl_mlir_path if args.optimize_tfl else tfl_mlir_path
        )
        opt_command = [
            str(onnx_mlir_opt),
            str(onnx_mlir_path),
            "--shape-inference",
            (
                "--convert-onnx-to-tfl="
                f"layout-mode={args.layout_mode} "
                f"layout-diagnostics={args.layout_diagnostics}"
            ),
            "--canonicalize",
            "-o",
            str(conversion_output),
        ]
        if args.verify_each:
            opt_command.append("--verify-each=true")
        _run(opt_command, "ONNX to TFL dialect conversion")
        if args.optimize_tfl:
            assert litert_opt is not None
            tfl_opt_command = [
                str(litert_opt),
                str(unoptimized_tfl_mlir_path),
                # Our ONNX lowering intentionally emits BatchMatMul for
                # MatMul/Gemm. TensorFlow's own pass turns safe rank-2
                # constant-RHS cases into FullyConnected before the general
                # optimizer fuses bias and activation operands.
                "--tfl-optimize-batch-matmul",
                "--tfl-optimize",
                "--canonicalize",
                "--cse",
                "--symbol-dce",
                "-o",
                str(tfl_mlir_path),
            ]
            if args.verify_each:
                tfl_opt_command.append("--verify-each=true")
            _run(tfl_opt_command, "TFL dialect optimization")
        export_command = [
            str(flatbuffer_translate),
            "-mlir-to-tflite-flatbuffer",
            str(tfl_mlir_path),
            "-o",
            str(flatbuffer_path),
            "-emit-builtin-tflite-ops=true",
            "-emit-select-tf-ops=false",
            "-emit-custom-ops=false",
        ]
        if use_buffer_offset:
            export_command.append("--use-buffer-offset")
        _run(export_command, "TFLite FlatBuffer export")
        if not flatbuffer_path.is_file() or flatbuffer_path.stat().st_size == 0:
            raise RuntimeError("FlatBuffer exporter produced an empty output")
        with flatbuffer_path.open("rb") as flatbuffer:
            if flatbuffer.read(8)[4:8] != b"TFL3":
                raise RuntimeError("exported file does not have the TFL3 identifier")

        _normalize_signature_order(flatbuffer_path)

        # Importing the file again exercises TensorFlow's schema verifier and
        # ensures the result is parseable as an actual TFLite model.
        verification_command = [
            str(flatbuffer_translate),
            "--tflite-flatbuffer-to-mlir",
            str(flatbuffer_path),
            "-o",
            str(verified_mlir_path),
        ]
        if use_buffer_offset:
            verification_command.append("--use-external-constant")
        _run(verification_command, "TFLite FlatBuffer verification")

        # Only publish a file after export and schema round-trip validation.
        # Stage it next to the destination so the final atomic rename never
        # crosses a filesystem boundary (the work directory is usually /tmp).
        staged_path: Optional[Path] = None
        try:
            with flatbuffer_path.open("rb") as source:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{output_path.name}.",
                    dir=output_path.parent,
                    delete=False,
                ) as staged:
                    shutil.copyfileobj(source, staged)
                    staged_path = Path(staged.name)
            os.replace(staged_path, output_path)
            staged_path = None
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

        if args.dump_onnx_mlir:
            shutil.copy2(onnx_mlir_path, output_path.with_suffix(".onnx.mlir"))
        if args.dump_tfl_mlir:
            shutil.copy2(tfl_mlir_path, output_path.with_suffix(".tfl.mlir"))
        if args.keep_intermediate_files:
            shutil.copytree(work_dir, kept_dir)
        print(f"generated {output_path} ({output_path.stat().st_size} bytes)")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
