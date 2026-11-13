#!/usr/bin/env python3
"""Refresh SHA-256 pins of already-pinned files after a reviewed edit.

Rewrites only the hash strings of the named paths in the formal source lock and
in the recorded receipt's copy of it. It adds no pin and removes none. The
canonical verifier still recomputes everything and compares.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ("formal/research/toolchain.json", "formal/research/results.json")


def main(paths: list[str]) -> int:
    lock = json.loads((ROOT / TARGETS[0]).read_text())["sourceHashes"]
    for path in paths:
        if path not in lock:
            raise SystemExit(f"not a pinned path: {path}")
        new = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        old = lock[path]
        for target in TARGETS:
            text = (ROOT / target).read_text()
            if old == new:
                continue
            if text.count(old) != 1:
                raise SystemExit(f"{target}: expected exactly one occurrence of the old pin for {path}")
            (ROOT / target).write_text(text.replace(old, new))
        print(path, old[:12], "->", new[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
