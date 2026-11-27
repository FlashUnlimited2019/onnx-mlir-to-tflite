# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlashUnlimited2019.

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check


class ScanningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def create(self, name, text="Documentation.\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def scanned(self, excludes):
        with patch.object(check.logger, "info") as log:
            check.main(str(self.root), excludes)
        return {call.args[0] for call in log.call_args_list}

    def test_document_suffix_directories(self):
        page = self.create("directory.md/nested.dc/page.md")
        self.create("reference.txt")
        self.assertEqual(self.scanned([]), {f"Checking {page}..."})

    def test_relative_exclusion_prunes_without_mutating_argument(self):
        self.create("dependency/README.md")
        page = self.create("dependency-extra/README.md")
        excludes = ["dependency"]
        original_walk = check.os.walk
        visited = []

        def record_walk(*args, **kwargs):
            for result in original_walk(*args, **kwargs):
                visited.append(Path(result[0]))
                yield result

        with patch.object(check.os, "walk", record_walk):
            self.assertEqual(self.scanned(excludes), {f"Checking {page}..."})
        self.assertNotIn(self.root / "dependency", visited)
        self.assertEqual(excludes, ["dependency"])

    def test_absolute_exclusion(self):
        self.create("build/result.md")
        page = self.create("README.md")
        self.assertEqual(
            self.scanned([str(self.root / "build")]), {f"Checking {page}..."}
        )

    def test_excluding_root(self):
        self.create("README.md")
        self.assertEqual(self.scanned([str(self.root)]), set())

    def test_markdown_and_directive_files_still_checked(self):
        self.create("reference.txt", "matching\n")
        page = self.create(
            "page.md", "[same-as-file]: <> (reference.txt)\n```text\nmatching\n```\n"
        )
        directive = self.create(
            "rule.dc",
            'file-same-as-stdout({"file": "reference.txt", "cmd": '
            + repr([sys.executable, "-c", "print('matching')"])
            + "})\n",
        )
        self.assertEqual(
            self.scanned([]), {f"Checking {page}...", f"Checking {directive}..."}
        )
        page.write_text(page.read_text().replace("matching", "different"))
        with self.assertRaises(ValueError):
            self.scanned([])


if __name__ == "__main__":
    unittest.main()
