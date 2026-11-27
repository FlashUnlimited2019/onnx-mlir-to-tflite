# SPDX-License-Identifier: Apache-2.0

# ===-------------------- check.py - Documentation Checker ----------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import argparse
import os, sys

from utils import setup_logger, DocCheckerCtx

logger = setup_logger("doc-check")

# Make common utilities visible by adding them to system paths.
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(doc_check_base_dir)

from doc_parser import try_parse_and_handle_directive

parser = argparse.ArgumentParser()
parser.add_argument(
    "root_dir", help="directory in which to look for documentation to operate on"
)

parser.add_argument(
    "--exclude_dirs",
    nargs="+",
    help="directories to exclude, specified relative to root_dir or as absolute paths",
    default=[],
)


def main(root_dir, exclude_dirs):
    root_dir = os.path.abspath(root_dir)
    excluded = [
        os.path.normcase(os.path.abspath(os.path.join(root_dir, directory)))
        for directory in exclude_dirs
    ]

    def is_excluded(path):
        path = os.path.normcase(path)
        return any(
            path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)
            for directory in excluded
        )

    ctx = DocCheckerCtx(root_dir)
    # Prune dependency/build trees before descending into them. A directory
    # whose name ends in .md or .dc is not itself a documentation file.
    for directory, subdirs, filenames in os.walk(root_dir, followlinks=False):
        if is_excluded(directory):
            subdirs.clear()
            continue
        subdirs[:] = sorted(
            name for name in subdirs if not is_excluded(os.path.join(directory, name))
        )
        for name in sorted(filenames):
            doc_file = os.path.join(directory, name)
            if not name.endswith((".md", ".dc")) or not os.path.isfile(doc_file):
                continue
            logger.info("Checking {}...".format(doc_file))
            with ctx.open_doc(doc_file) as markdown_file:
                while not markdown_file.eof():
                    try_parse_and_handle_directive(ctx)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
