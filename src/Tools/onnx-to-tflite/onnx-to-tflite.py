#!/usr/bin/env python3
"""ONNX -> ONNX dialect -> TFL dialect -> TFLite FlatBuffer driver."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _find_tool(name: str, explicit: str | None, script: Path) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"configured {name} does not exist: {candidate}")

    from_path = shutil.which(name)
    if from_path:
        return Path(from_path).resolve()

    bin_dir = script.resolve().parent
    workspace = bin_dir.parents[3] if len(bin_dir.parents) > 3 else None
    candidates = [bin_dir / name]
    if workspace:
        candidates.extend(
            [
                workspace / "tensorflow" / "bazel-bin" / "tensorflow" / "compiler"
                / "mlir" / "lite" / "flatbuffer_translate",
                workspace / "build" / "tensorflow" / "flatbuffer_translate",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"unable to find {name}; searched PATH and: {searched}")


def _run(command: list[str], stage: str) -> None:
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
        description="Lower a static FP32 ONNX model through MLIR to TFLite")
    parser.add_argument("input", type=Path, help="input .onnx model")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--onnx-mlir", help="path to the onnx-mlir binary")
    parser.add_argument("--onnx-mlir-opt", help="path to onnx-mlir-opt")
    parser.add_argument(
        "--flatbuffer-translate", help="path to TensorFlow flatbuffer_translate")
    parser.add_argument("--dump-onnx-mlir", action="store_true")
    parser.add_argument("--dump-tfl-mlir", action="store_true")
    parser.add_argument("--keep-intermediate-files", action="store_true")
    parser.add_argument(
        "--verify-each", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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

    try:
        onnx_mlir = _find_tool("onnx-mlir", args.onnx_mlir, script)
        onnx_mlir_opt = _find_tool(
            "onnx-mlir-opt", args.onnx_mlir_opt, script)
        flatbuffer_translate = _find_tool(
            "flatbuffer_translate",
            args.flatbuffer_translate or os.environ.get("FLATBUFFER_TRANSLATE"),
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
    temporary = tempfile.TemporaryDirectory(prefix="onnx-to-tflite-")
    work_dir = Path(temporary.name)
    try:
        onnx_prefix = work_dir / "model"
        onnx_mlir_path = work_dir / "model.onnx.mlir"
        tfl_mlir_path = work_dir / "model.tfl.mlir"
        flatbuffer_path = work_dir / "model.tflite"
        verified_mlir_path = work_dir / "verified-roundtrip.mlir"

        _run(
            [str(onnx_mlir), str(input_path), "--EmitONNXIR", "-o", str(onnx_prefix)],
            "ONNX import",
        )
        opt_command = [
            str(onnx_mlir_opt),
            str(onnx_mlir_path),
            "--shape-inference",
            "--convert-onnx-to-tfl",
            "--canonicalize",
            "-o",
            str(tfl_mlir_path),
        ]
        if args.verify_each:
            opt_command.append("--verify-each=true")
        _run(opt_command, "ONNX to TFL dialect conversion")
        _run(
            [
                str(flatbuffer_translate),
                "-mlir-to-tflite-flatbuffer",
                str(tfl_mlir_path),
                "-o",
                str(flatbuffer_path),
                "-emit-builtin-tflite-ops=true",
                "-emit-select-tf-ops=false",
                "-emit-custom-ops=false",
            ],
            "TFLite FlatBuffer export",
        )
        if not flatbuffer_path.is_file() or flatbuffer_path.stat().st_size == 0:
            raise RuntimeError("FlatBuffer exporter produced an empty output")
        with flatbuffer_path.open("rb") as flatbuffer:
            if flatbuffer.read(8)[4:8] != b"TFL3":
                raise RuntimeError("exported file does not have the TFL3 identifier")

        # Importing the file again exercises TensorFlow's schema verifier and
        # ensures the result is parseable as an actual TFLite model.
        _run(
            [
                str(flatbuffer_translate),
                "--tflite-flatbuffer-to-mlir",
                str(flatbuffer_path),
                "-o",
                str(verified_mlir_path),
            ],
            "TFLite FlatBuffer verification",
        )

        # Only publish a file after export and schema round-trip validation.
        os.replace(flatbuffer_path, output_path)

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
