#!/usr/bin/env python3
"""Replay-verify a market-context receipt against its frozen external archive.

This module is deliberately offline and read-only. A successful result proves
archive and receipt integrity only; it grants no research, experiment, holdout,
model, promotion, deployment, order, or live authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import market_context_source as source
import verify_market_context_bundle as bundle


VERIFICATION_SCHEMA_ID = "binance_usdm_market_context_receipt_verification_v1"
VERIFICATION_SCHEMA_VERSION = 1
RECEIPT_KEYS = {
    "schemaId",
    "schemaVersion",
    "bundleVerifierPath",
    "bundleVerifierSha256",
    "collectionStatusSha256",
    "collectionCodeCommit",
    "collectedCollectorSha256",
    "collectedSourceVerifierSha256",
    "sourceLicenseManifestSha256",
    "sourceManifestSha256",
    "sourceVerificationReceiptSha256",
    "sourceVerificationReceipt",
    "panelSchemaId",
    "panelSchemaVersion",
    "panelSha256",
    "panelRows",
    "collectionStatusVerified",
    "sourceBundleVerified",
    "researchAdmission",
    "experimentUse",
    "holdoutUse",
    "modelUse",
    "promotionUse",
    "deploymentUse",
    "orderUse",
    "liveAuthorizationUse",
}
TRUE_FIELDS = ("collectionStatusVerified", "sourceBundleVerified")
FALSE_FIELDS = (
    "researchAdmission",
    "experimentUse",
    "holdoutUse",
    "modelUse",
    "promotionUse",
    "deploymentUse",
    "orderUse",
    "liveAuthorizationUse",
)
HASH_FIELDS = (
    "bundleVerifierSha256",
    "collectionStatusSha256",
    "collectedCollectorSha256",
    "collectedSourceVerifierSha256",
    "sourceLicenseManifestSha256",
    "sourceManifestSha256",
    "sourceVerificationReceiptSha256",
    "panelSha256",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same_json_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _regular_non_symlink(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return resolved


def _validate_receipt(receipt: dict[str, object]) -> None:
    bundle._exact_object(receipt, RECEIPT_KEYS, "bundle receipt")
    if (
        receipt["schemaId"] != bundle.BUNDLE_RECEIPT_SCHEMA_ID
        or type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != bundle.BUNDLE_RECEIPT_SCHEMA_VERSION
        or receipt["bundleVerifierPath"] != bundle.BUNDLE_VERIFIER_PATH
    ):
        raise ValueError("bundle receipt identity is incompatible")
    if any(receipt[field] is not True for field in TRUE_FIELDS):
        raise ValueError("bundle receipt does not confirm both verification layers")
    if any(receipt[field] is not False for field in FALSE_FIELDS):
        raise ValueError("bundle receipt contains forbidden downstream authority")
    for field in HASH_FIELDS:
        value = receipt[field]
        if not isinstance(value, str) or not bundle.SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"bundle receipt {field} is malformed")
    commit = receipt["collectionCodeCommit"]
    if not isinstance(commit, str) or not bundle.COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("bundle receipt collectionCodeCommit is malformed")
    bundle._require_git_commit(commit)
    if (
        receipt["bundleVerifierSha256"]
        != _sha256(bundle._git_blob(commit, bundle.BUNDLE_VERIFIER_PATH))
    ):
        raise ValueError("bundle receipt verifier is not bound to the collection commit")
    current_verifier_sha = _sha256(
        Path(bundle.__file__).resolve(strict=True).read_bytes()
    )
    if receipt["bundleVerifierSha256"] != current_verifier_sha:
        raise ValueError("current bundle verifier is not the receipt's exact version")
    source_receipt = receipt["sourceVerificationReceipt"]
    if not isinstance(source_receipt, dict):
        raise ValueError("embedded source-verification receipt is malformed")
    if receipt["sourceVerificationReceiptSha256"] != _sha256(
        source._json_bytes(source_receipt)
    ):
        raise ValueError("embedded source-verification receipt SHA-256 changed")
    if type(receipt["panelRows"]) is not int or receipt["panelRows"] <= 0:
        raise ValueError("bundle receipt panelRows must be a positive integer")


def _archive_inventory(
    archive: Path, manifest: dict[str, object]
) -> tuple[dict[str, str], int]:
    expected_files = {
        "collection-status.json",
        "source-manifest.json",
        *bundle._manifest_artifact_paths(manifest),
    }
    expected_directories: set[str] = set()
    for logical in expected_files:
        path = PurePosixPath(logical)
        for parent in path.parents:
            if str(parent) != ".":
                expected_directories.add(str(parent))
    actual_files: set[str] = set()
    actual_hashes: dict[str, str] = {}
    actual_directories: set[str] = set()
    archive_bytes = 0
    for path in archive.rglob("*"):
        logical = path.relative_to(archive).as_posix()
        if path.is_symlink():
            raise ValueError("frozen archive contains a symlink")
        if path.is_file():
            actual_files.add(logical)
            payload = path.read_bytes()
            archive_bytes += len(payload)
            actual_hashes[logical] = _sha256(payload)
        elif path.is_dir():
            actual_directories.add(logical)
        else:
            raise ValueError("frozen archive contains a non-file entry")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError("bundle receipt archive inventory is incomplete or contains extras")
    return actual_hashes, archive_bytes


def verify_receipt(receipt_path: Path, archive_dir: Path) -> dict[str, object]:
    receipt_file = _regular_non_symlink(receipt_path, "bundle receipt")
    if archive_dir.is_symlink():
        raise ValueError("frozen archive must be a non-symlink directory")
    archive = archive_dir.resolve(strict=True)
    if not archive.is_dir():
        raise ValueError("frozen archive must be a non-symlink directory")
    receipt, receipt_payload = bundle._strict_object(receipt_file, "bundle receipt")
    _validate_receipt(receipt)

    status_path = _regular_non_symlink(
        archive / "collection-status.json", "collection status"
    )
    manifest_path = _regular_non_symlink(
        archive / "source-manifest.json", "source manifest"
    )
    status_payload = status_path.read_bytes()
    if _sha256(status_payload) != receipt["collectionStatusSha256"]:
        raise ValueError("bundle receipt status SHA-256 disagrees with the archive")
    manifest, manifest_payload = bundle._strict_object(
        manifest_path, "source manifest"
    )
    if _sha256(manifest_payload) != receipt["sourceManifestSha256"]:
        raise ValueError("bundle receipt manifest SHA-256 disagrees with the archive")
    archive_hashes, archive_bytes = _archive_inventory(archive, manifest)

    panel, recomputed_receipt = bundle.verify_bundle(status_path)
    if not _same_json_value(receipt, recomputed_receipt):
        raise ValueError("bundle receipt disagrees with replayed verification")
    if _sha256(panel) != receipt["panelSha256"]:
        raise ValueError("bundle receipt panel SHA-256 disagrees with replayed verification")
    replay_hashes, replay_bytes = _archive_inventory(archive, manifest)
    if (replay_hashes, replay_bytes) != (archive_hashes, archive_bytes):
        raise ValueError("frozen archive changed during receipt verification")
    return {
        "schemaId": VERIFICATION_SCHEMA_ID,
        "schemaVersion": VERIFICATION_SCHEMA_VERSION,
        "receiptSha256": _sha256(receipt_payload),
        "collectionStatusSha256": receipt["collectionStatusSha256"],
        "sourceManifestSha256": receipt["sourceManifestSha256"],
        "panelSha256": receipt["panelSha256"],
        "panelRows": receipt["panelRows"],
        "archiveFilesVerified": len(replay_hashes),
        "archiveBytesVerified": replay_bytes,
        "collectionStatusVerified": True,
        "sourceBundleVerified": True,
        "researchAdmission": False,
        "experimentUse": False,
        "holdoutUse": False,
        "modelUse": False,
        "promotionUse": False,
        "deploymentUse": False,
        "orderUse": False,
        "liveAuthorizationUse": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = verify_receipt(args.receipt, args.archive)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"market-context receipt verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
