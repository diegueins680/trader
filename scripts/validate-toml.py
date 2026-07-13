#!/usr/bin/env python3
"""Parse deploy TOML files with Python's standard-library TOML parser."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = (
    ROOT / "fly.toml",
    ROOT / "fly.research.toml",
    ROOT / "haskell/web/fly.frontend.toml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="TOML files to validate (defaults to all checked-in Fly configs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or DEFAULT_CONFIGS
    failed = False
    for path in paths:
        if path.is_absolute() and path.is_relative_to(ROOT):
            display = path.relative_to(ROOT)
        else:
            display = path
        try:
            with path.open("rb") as handle:
                tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"ERROR: invalid TOML {display}: {exc}", file=sys.stderr)
            failed = True
        else:
            print(f"validated TOML: {display}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
