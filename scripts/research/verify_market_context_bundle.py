#!/usr/bin/env python3
"""Verify collector status and a frozen Binance market-context source bundle.

This module is deliberately offline. It binds the collector's successful
status to the source manifest, historical Git blobs, raw-response verifier,
and derived panel. Its receipt is integrity evidence only: it grants no
experiment, holdout, model, promotion, deployment, order, or live authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import market_context_source as source


STATUS_SCHEMA_ID = "binance_usdm_market_context_collection_status_v1"
STATUS_SCHEMA_VERSION = 1
BUNDLE_RECEIPT_SCHEMA_ID = "binance_usdm_market_context_bundle_receipt_v1"
BUNDLE_RECEIPT_SCHEMA_VERSION = 1
BUNDLE_VERIFIER_PATH = "scripts/research/verify_market_context_bundle.py"
COLLECTOR_PATH = "scripts/research/collect_market_context.py"
SOURCE_VERIFIER_PATH = "scripts/research/market_context_source.py"
SOURCE_LICENSE_MANIFEST_PATH = (
    "research-notes/market-prediction-2026-09-04/"
    "data-source-license-manifest.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
STATUS_KEYS = {
    "schemaId",
    "schemaVersion",
    "sourceId",
    "startedAtMs",
    "codeCommit",
    "provenanceTrackedClean",
    "collectorPath",
    "collectorSha256",
    "verifierPath",
    "verifierSha256",
    "runtime",
    "outcomeUse",
    "modelUse",
    "orderUse",
    "liveAuthorizationUse",
    "state",
    "completedAtMs",
    "completedArtifactPaths",
    "sourceManifestPublished",
    "sourceManifestSha256",
    "eligiblePopulationCount",
}
AUTHORITY_FIELDS = (
    "outcomeUse",
    "modelUse",
    "orderUse",
    "liveAuthorizationUse",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    value = source._decode_json(payload, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, payload


def _exact_object(
    value: object, expected_keys: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{label} keys are incompatible")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > source.TIMESTAMP_MAX:
        raise ValueError(f"{label} must be a non-negative Int64")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _git_blob(code_commit: str, path: str) -> bytes:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "show", f"{code_commit}:{path}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"recorded Git blob is unavailable for {path}") from error
    return result.stdout


def _require_git_commit(code_commit: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", code_commit],
            cwd=repository_root,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("recorded Git commit is unavailable") from error
    if result.stdout != b"commit\n":
        raise ValueError("recorded Git object is not a commit")


def _manifest_artifact_paths(manifest: dict[str, object]) -> list[str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source manifest artifacts are unavailable")
    peers = artifacts.get("peerKlines")
    if not isinstance(peers, list):
        raise ValueError("source manifest peerKlines are unavailable")
    ordered = [
        artifacts.get("serverTimeBefore"),
        artifacts.get("exchangeInfo"),
        artifacts.get("ticker24hr"),
        *peers,
        artifacts.get("serverTimeAfter"),
    ]
    paths: list[str] = []
    for record in ordered:
        if not isinstance(record, dict):
            raise ValueError("source manifest artifact record is unavailable")
        paths.append(_string(record.get("logicalPath"), "artifact logicalPath"))
    return paths


def _verify_license_blob(payload: bytes) -> str:
    value = source._decode_json(payload, "historical source-license manifest")
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
        raise ValueError("historical source-license manifest is incompatible")
    matches = [
        item
        for item in value["sources"]
        if isinstance(item, dict) and item.get("id") == source.SOURCE_ID
    ]
    if len(matches) != 1:
        raise ValueError("historical source-license record is missing or duplicated")
    record = matches[0]
    if record.get("status") != "existing_authorized_public_read_only_source":
        raise ValueError("historical source-license record is not public read-only")
    return _sha256(payload)


def _validate_status(
    status: dict[str, object],
    status_payload: bytes,
    manifest: dict[str, object],
    manifest_payload: bytes,
) -> dict[str, object]:
    _exact_object(status, STATUS_KEYS, "collection status")
    if (
        status["schemaId"] != STATUS_SCHEMA_ID
        or status["sourceId"] != source.SOURCE_ID
    ):
        raise ValueError("collection status identity is incompatible")
    if _integer(status["schemaVersion"], "collection schemaVersion") != (
        STATUS_SCHEMA_VERSION
    ):
        raise ValueError("collection status identity is incompatible")
    if status["state"] != "complete_unverified":
        raise ValueError("collection status is not complete_unverified")
    if status["sourceManifestPublished"] is not True:
        raise ValueError("collection status does not confirm manifest publication")
    if status["provenanceTrackedClean"] is not True:
        raise ValueError("collection status does not confirm clean provenance")
    if any(status[field] is not False for field in AUTHORITY_FIELDS):
        raise ValueError("collection status contains forbidden downstream authority")
    if status["collectorPath"] != COLLECTOR_PATH:
        raise ValueError("collection status collector path is incompatible")
    if status["verifierPath"] != SOURCE_VERIFIER_PATH:
        raise ValueError("collection status source-verifier path is incompatible")
    runtime = _exact_object(status["runtime"], {"python"}, "collection runtime")
    _string(runtime["python"], "collection Python runtime")
    code_commit = _string(status["codeCommit"], "collection codeCommit")
    if not COMMIT_PATTERN.fullmatch(code_commit):
        raise ValueError("collection codeCommit is malformed")
    _require_git_commit(code_commit)
    if manifest.get("codeCommit") != code_commit:
        raise ValueError("collection status and source manifest commits disagree")
    declared_manifest_sha = _string(
        status["sourceManifestSha256"], "collection sourceManifestSha256"
    )
    if not SHA256_PATTERN.fullmatch(declared_manifest_sha):
        raise ValueError("collection sourceManifestSha256 is malformed")
    if declared_manifest_sha != _sha256(manifest_payload):
        raise ValueError("collection source-manifest SHA-256 changed")
    collector_sha = _string(status["collectorSha256"], "collectorSha256")
    verifier_sha = _string(status["verifierSha256"], "verifierSha256")
    if not SHA256_PATTERN.fullmatch(collector_sha) or not SHA256_PATTERN.fullmatch(
        verifier_sha
    ):
        raise ValueError("collection provenance SHA-256 is malformed")
    if collector_sha != _sha256(_git_blob(code_commit, COLLECTOR_PATH)):
        raise ValueError("collector SHA-256 disagrees with the recorded Git commit")
    if verifier_sha != _sha256(_git_blob(code_commit, SOURCE_VERIFIER_PATH)):
        raise ValueError("source-verifier SHA-256 disagrees with the recorded Git commit")
    license_sha = _verify_license_blob(
        _git_blob(code_commit, SOURCE_LICENSE_MANIFEST_PATH)
    )
    started = _integer(status["startedAtMs"], "collection startedAtMs")
    completed = _integer(status["completedAtMs"], "collection completedAtMs")
    manifest_started = _integer(manifest.get("startedAtMs"), "manifest startedAtMs")
    manifest_completed = _integer(
        manifest.get("completedAtMs"), "manifest completedAtMs"
    )
    decision = _integer(manifest.get("decisionTime"), "manifest decisionTime")
    if not started <= manifest_started <= manifest_completed <= decision <= completed:
        raise ValueError("collection status and manifest clocks are incoherent")
    completed_paths = status["completedArtifactPaths"]
    if completed_paths != _manifest_artifact_paths(manifest):
        raise ValueError("collection status artifact inventory disagrees with manifest")
    population = manifest.get("population")
    eligible = population.get("eligibleSymbols") if isinstance(population, dict) else None
    population_count = _integer(
        status["eligiblePopulationCount"], "eligiblePopulationCount"
    )
    if not isinstance(eligible, list) or population_count != len(eligible):
        raise ValueError("collection status population count disagrees with manifest")
    return {
        "collectionStatusSha256": _sha256(status_payload),
        "collectionCodeCommit": code_commit,
        "collectedCollectorSha256": collector_sha,
        "collectedSourceVerifierSha256": verifier_sha,
        "sourceLicenseManifestSha256": license_sha,
    }


def _executing_verifier_sha(
    code_commit: str, collected_source_verifier_sha: object
) -> str:
    bundle_payload = Path(__file__).resolve(strict=True).read_bytes()
    bundle_sha = _sha256(bundle_payload)
    if bundle_sha != _sha256(_git_blob(code_commit, BUNDLE_VERIFIER_PATH)):
        raise ValueError(
            "executing bundle-verifier bytes disagree with the collection commit"
        )
    source_sha = _sha256(Path(source.__file__).resolve(strict=True).read_bytes())
    if source_sha != collected_source_verifier_sha:
        raise ValueError(
            "executing source-verifier bytes disagree with the collection commit"
        )
    return bundle_sha


def verify_bundle(status_path: Path) -> tuple[bytes, dict[str, object]]:
    if status_path.is_symlink():
        raise ValueError("collection status must be a regular non-symlink file")
    status_file = status_path.resolve(strict=True)
    if not status_file.is_file():
        raise ValueError("collection status must be a regular non-symlink file")
    if status_file.name != "collection-status.json":
        raise ValueError("collection status filename is incompatible")
    root = status_file.parent
    manifest_path = root / "source-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("source manifest is missing or unsafe")
    status, status_payload = _strict_object(status_file, "collection status")
    manifest, manifest_payload = _strict_object(manifest_path, "source manifest")
    status_evidence = _validate_status(
        status, status_payload, manifest, manifest_payload
    )
    bundle_verifier_sha = _executing_verifier_sha(
        status_evidence["collectionCodeCommit"],
        status_evidence["collectedSourceVerifierSha256"],
    )
    panel, source_receipt = source.verify_and_derive(manifest_path)
    manifest_sha = _sha256(manifest_payload)
    if (
        source_receipt.get("sourceManifestSha256") != manifest_sha
        or source_receipt.get("sourceManifestCodeCommit")
        != status_evidence["collectionCodeCommit"]
    ):
        raise ValueError("source manifest changed during bundle verification")
    source_receipt_payload = source._json_bytes(source_receipt)
    receipt = {
        "schemaId": BUNDLE_RECEIPT_SCHEMA_ID,
        "schemaVersion": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "bundleVerifierPath": BUNDLE_VERIFIER_PATH,
        "bundleVerifierSha256": bundle_verifier_sha,
        **status_evidence,
        "sourceManifestSha256": manifest_sha,
        "sourceVerificationReceiptSha256": _sha256(source_receipt_payload),
        "sourceVerificationReceipt": source_receipt,
        "panelSchemaId": source_receipt["panelSchemaId"],
        "panelSchemaVersion": source_receipt["panelSchemaVersion"],
        "panelSha256": source_receipt["panelSha256"],
        "panelRows": source_receipt["panelRows"],
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
    return panel, receipt


def _validate_outputs(status_path: Path, outputs: tuple[Path, Path]) -> None:
    status = status_path.resolve(strict=True)
    root = status.parent
    protected = {status, root / "source-manifest.json"}
    raw_root = root / "raw"
    resolved = [path.resolve() for path in outputs]
    if resolved[0] == resolved[1]:
        raise ValueError("panel and bundle receipt outputs must be different paths")
    for output in resolved:
        if output in protected or raw_root == output or raw_root in output.parents:
            raise ValueError("outputs cannot overwrite frozen bundle evidence")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verify", choices=["verify"])
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--panel-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)
    if (args.panel_output is None) != (args.receipt_output is None):
        parser.error("--panel-output and --receipt-output must be provided together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        panel, receipt = verify_bundle(args.status)
        if args.panel_output is not None:
            _validate_outputs(args.status, (args.panel_output, args.receipt_output))
            source._write_bytes_atomic(args.panel_output, panel)
            source._write_bytes_atomic(args.receipt_output, source._json_bytes(receipt))
        print(json.dumps(receipt, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print(f"market-context bundle verification failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
