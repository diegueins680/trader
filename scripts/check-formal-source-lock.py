#!/usr/bin/env python3
"""Fast source-pin and module-inventory preflight; not a proof certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LOCK = "formal/research/toolchain.json"
CAPABILITY = "formal/research/capability-source-contract.json"
MAX_SOURCE = 64 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def source_path(root, relative):
    require(type(relative) is str and 0 < len(relative) <= 512 and
            re.fullmatch(r"[A-Za-z0-9_./-]+", relative) and
            all(part not in ("", ".", "..") for part in relative.split("/")),
            "unsafe source path")
    path = root
    for part in relative.split("/"):
        path = path / part
        require(not path.is_symlink(), "symlinked source path")
    require(path.is_file(), "missing regular source: " + relative)
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate metadata key")
        result[key] = value
    return result


def metadata(root, relative):
    path = source_path(root, relative)
    require(path.stat().st_size <= 2 * 1024 * 1024, "metadata size limit")
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    require(len(raw) <= 2 * 1024 * 1024, "metadata size limit")
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(type(value) is dict and value.get("schemaVersion") == 1 and
            type(value.get("schemaVersion")) is int, "unsupported metadata schema")
    return value, raw


def digest(root, relative):
    path = source_path(root, relative)
    require(path.stat().st_size <= MAX_SOURCE, "source size limit")
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(65536):
            size += len(block)
            require(size <= MAX_SOURCE, "source size limit")
            hasher.update(block)
    return hasher.hexdigest()


def check_source_lock(root=ROOT):
    root = Path(root).resolve()
    lock, _ = metadata(root, LOCK)
    capability, capability_raw = metadata(root, CAPABILITY)
    pins = lock.get("sourceHashes")
    require(type(pins) is dict and 1 <= len(pins) <= 4096 and CAPABILITY in pins,
            "missing or invalid source pins")
    bindings = []
    for relative, expected in pins.items():
        # Validate paths even for captured metadata, then hash those same bytes.
        source_path(root, relative)
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
                "invalid source digest")
        if relative == CAPABILITY:
            value, raw = capability, capability_raw
        elif relative.endswith(("-source.json", "-source-contract.json")):
            value, raw = metadata(root, relative)
        else:
            value, raw = None, None
        actual = hashlib.sha256(raw).hexdigest() if raw is not None else digest(root, relative)
        require(actual == expected, "source hash mismatch: " + relative)
        if value is not None:
            for field in ("hashes", "supportHashes", "supportSourceHashes"):
                if field in value:
                    require(type(value[field]) is dict and len(value[field]) <= 4096,
                            "invalid reviewed source binding map")
                    bindings.extend(value[field].items())
            if "source" in value and "sha256" in value:
                bindings.append((value["source"], value["sha256"]))
    for relative, expected in bindings:
        require(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
                "invalid reviewed source digest")
        require(digest(root, relative) == expected, "reviewed source binding mismatch: " + relative)
    inventory = capability.get("sourceInventory")
    require(type(inventory) is list and 1 <= len(inventory) <= 4096 and
            all(type(p) is str for p in inventory), "invalid application inventory")
    for relative in inventory:
        source_path(root, relative)
        require(relative in pins, "application module has no source pin")
    actual_inventory = sorted(str(p.relative_to(root)) for p in (root / "haskell/app").rglob("*.hs"))
    require(inventory == actual_inventory, "application module inventory mismatch")
    return {"valid": True, "scope": "source-lock-and-inventory",
            "pinnedFiles": len(pins), "applicationModules": len(inventory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout root to inspect")
    args = parser.parse_args()
    try:
        result = check_source_lock(args.root)
    except (OSError, UnicodeError):
        print("Formal source lock invalid: unable to read required source metadata")
        return 1
    except ValueError as error:
        # JSON parsing errors may include raw input; keep them separate.
        message = "invalid JSON metadata" if isinstance(error, json.JSONDecodeError) else str(error)
        print("Formal source lock invalid: " + message)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
