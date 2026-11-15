"""Disabled, retryable publication of verified reports on a local POSIX filesystem."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat

VERSION = "report-bundle-v2"
TARGET = "report-bundle-v2.json"
NAMES = frozenset({"experiment-registry.csv", "all-seed-results.csv", "symbol-fold-base-results.csv",
                   "multi-seed-training.json", "ope-report.json", "evaluation-summary.json",
                   "experiment-manifest.json"})
MAX_REPORT = 4194304
MAX_COMBINED = 16777216
MAX_BUNDLE = 33554432


def encode_bundle_v2(reports: object) -> bytes | None:
    if type(reports) is not dict or set(reports) != NAMES:
        return None
    if any(type(value) is not bytes or len(value) > MAX_REPORT for value in reports.values()):
        return None
    if sum(len(value) for value in reports.values()) > MAX_COMBINED:
        return None
    try:
        value = {"schema": VERSION, "promotion": "research", "enabled": False,
                 "liveAuthorization": False,
                 "reports": {name: reports[name].decode("utf-8") for name in sorted(NAMES)}}
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
        return raw if len(raw) <= MAX_BUNDLE else None
    except (UnicodeError, ValueError, TypeError, MemoryError):
        return None


def _matches(directory: int, raw: bytes) -> bool:
    try:
        fd = os.open(TARGET, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("report target is not a regular file")
        offset = 0
        while offset < len(raw):
            block = os.read(fd, min(65536, len(raw) - offset))
            if not block or block != raw[offset:offset + len(block)]:
                raise OSError("existing report bundle conflicts with verified reports")
            offset += len(block)
        if os.read(fd, 1):
            raise OSError("existing report bundle has trailing data")
        os.fsync(fd)
        return True
    finally:
        os.close(fd)


def _write(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        count = os.write(fd, raw[offset:offset + 65536])
        if not 0 < count <= min(65536, len(raw) - offset):
            raise OSError("incomplete staging write")
        offset += count
    os.fsync(fd)


def publish_bundle_v2(directory: object, reports: object, *, enabled: object = False,
                      version: object = VERSION) -> str | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if type(directory) is not str or not 1 <= len(directory) <= 4096:
        return None
    raw = encode_bundle_v2(reports)
    if raw is None:
        return None
    parent = None
    owned = None
    try:
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if not _matches(parent, raw):
            temporary = ".report-bundle-v2." + secrets.token_hex(16)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            owned = temporary
            try:
                _write(fd, raw)
            finally:
                os.close(fd)
            try:
                os.link(temporary, TARGET, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            except FileExistsError:
                if not _matches(parent, raw):
                    raise OSError("concurrent report publication disappeared")
        os.fsync(parent)
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, MemoryError):
        return None
    finally:
        if parent is not None:
            if owned is not None:
                try:
                    os.unlink(owned, dir_fd=parent)
                except OSError:
                    pass
            os.close(parent)
