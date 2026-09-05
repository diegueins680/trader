#!/usr/bin/env python3
"""Verify a metadata-only derivatives receipt against a frozen archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys

import collect_datafeed as collector


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_TYPE = "metadata_only_collection_artifact_receipt"
RECEIPT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
TOP_LEVEL_KEYS = {
    "schemaVersion",
    "receiptId",
    "receiptType",
    "collection",
    "source",
    "universe",
    "status",
    "artifacts",
    "verification",
    "outcomeBoundary",
}
COLLECTION_KEYS = {
    "statusSchemaVersion",
    "artifactSchema",
    "derivativesObservationSchema",
    "featureAvailabilitySchema",
    "interval",
    "startedAt",
    "finishedAt",
    "codeCommit",
    "state",
    "failedSymbols",
    "provenanceIssues",
    "provenanceTrackedClean",
}
SOURCE_KEYS = {
    "provider",
    "access",
    "licenseManifest",
    "retrievalCommand",
    "verificationCommand",
    "archivePolicy",
}
OUTCOME_BOUNDARY = {
    "admission": "acquisition_metadata_only",
    "returnsComputed": False,
    "ranksComputed": False,
    "weightsComputed": False,
    "pnlComputed": False,
    "riskMetricsComputed": False,
    "forecastMetricsComputed": False,
    "economicMetricsComputed": False,
    "holdoutsOpened": 0,
    "ordersPlaced": 0,
    "modelInputsChanged": False,
    "liveAuthorizationChanged": False,
}


def _read_receipt(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in receipt: {key}")
            value[key] = item
        return value

    payload = path.read_bytes()
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError("receipt must contain a JSON object")
    return value


def _require_exact_keys(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} keys are malformed")
    return value


def _same_json_value(left: object, right: object) -> bool:
    """Compare parsed JSON without treating booleans as integers."""
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


def _logical_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is malformed")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return path


def _resolve_archive_path(root: Path, logical: PurePosixPath, label: str) -> Path:
    candidate = (root / Path(*logical.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the archive root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} is not a regular file")
    return candidate


def _receipt_artifact(
    value: object,
    *,
    expected_logical_path: str,
    status_record: object,
    label: str,
) -> dict[str, object]:
    record = _require_exact_keys(value, {"logicalPath", "rows", "sha256"}, label)
    logical = _logical_path(record["logicalPath"], f"{label}.logicalPath")
    if str(logical) != expected_logical_path:
        raise ValueError(f"{label} has an unexpected logical path")
    status_artifact = _require_exact_keys(
        status_record, {"path", "rows", "sha256"}, f"{label} status artifact"
    )
    rows = record["rows"]
    sha256 = record["sha256"]
    if type(rows) is not int or rows <= 0:
        raise ValueError(f"{label} row count must be positive")
    if not isinstance(sha256, str) or not collector.SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(f"{label} sha256 is malformed")
    if rows != status_artifact["rows"] or sha256 != status_artifact["sha256"]:
        raise ValueError(f"{label} disagrees with the collector status")
    return record


def _archive_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file():
            logical = path.relative_to(root).as_posix()
            _resolve_archive_path(
                root, _logical_path(logical, "archive path"), "archive path"
            )
            files.add(logical)
    return files


def verify_receipt(
    receipt_path: str | os.PathLike[str],
    archive_dir: str | os.PathLike[str],
) -> dict[str, object]:
    """Verify a schema-1 receipt and every byte in its frozen archive."""
    receipt_file = Path(receipt_path).expanduser().resolve(strict=True)
    if not receipt_file.is_file():
        raise ValueError("receipt path is not a file")
    archive = Path(archive_dir).expanduser().resolve(strict=True)
    if not archive.is_dir():
        raise ValueError("archive path is not a directory")
    receipt = _require_exact_keys(
        _read_receipt(receipt_file), TOP_LEVEL_KEYS, "receipt"
    )
    receipt_id = receipt["receiptId"]
    if (
        type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != RECEIPT_SCHEMA_VERSION
        or receipt["receiptType"] != RECEIPT_TYPE
        or not isinstance(receipt_id, str)
        or not RECEIPT_ID_PATTERN.fullmatch(receipt_id)
    ):
        raise ValueError("receipt identity or schema is unsupported")
    if not _same_json_value(receipt["outcomeBoundary"], OUTCOME_BOUNDARY):
        raise ValueError("receipt outcome boundary is not acquisition-only")

    status_receipt = _require_exact_keys(
        receipt["status"], {"logicalPath", "sha256"}, "receipt status"
    )
    status_logical = _logical_path(
        status_receipt["logicalPath"], "receipt status.logicalPath"
    )
    if str(status_logical) != ".collector/last-run.json":
        raise ValueError("receipt status path is unsupported")
    status_sha256 = status_receipt["sha256"]
    if not isinstance(status_sha256, str) or not collector.SHA256_PATTERN.fullmatch(
        status_sha256
    ):
        raise ValueError("receipt status sha256 is malformed")
    status_path = _resolve_archive_path(archive, status_logical, "receipt status")
    status, status_payload = collector._read_json_object(status_path)
    if hashlib.sha256(status_payload).hexdigest() != status_sha256:
        raise ValueError("receipt status sha256 does not match the archive")

    verified = collector.verify_collection_artifacts(status_path, cache_dir=archive)
    if verified.get("statusSha256") != status_sha256:
        raise ValueError("artifact verification disagrees with the receipt status")

    collection = _require_exact_keys(
        receipt["collection"], COLLECTION_KEYS, "receipt collection"
    )
    expected_collection = {
        "statusSchemaVersion": status.get("schemaVersion"),
        "artifactSchema": status.get("artifactSchema"),
        "derivativesObservationSchema": status.get("derivativesObservationSchema"),
        "featureAvailabilitySchema": status.get("featureAvailabilitySchema"),
        "interval": status.get("interval"),
        "startedAt": status.get("startedAt"),
        "finishedAt": status.get("finishedAt"),
        "codeCommit": status.get("commit"),
        "state": status.get("state"),
        "failedSymbols": status.get("failedSymbols"),
        "provenanceIssues": status.get("provenanceIssues"),
        "provenanceTrackedClean": status.get("provenanceTrackedClean"),
    }
    if not _same_json_value(collection, expected_collection):
        raise ValueError("receipt collection disagrees with the collector status")

    source = _require_exact_keys(receipt["source"], SOURCE_KEYS, "receipt source")
    if (
        source["provider"] != "Binance USD-M Futures"
        or source["access"] != "public_read_only"
        or source["licenseManifest"] != status.get("dataSourceLicenseManifest")
        or source["licenseManifest"] != collector.SOURCE_LICENSE_MANIFEST
        or any(
            not isinstance(source[key], str) or not source[key]
            for key in ("retrievalCommand", "verificationCommand", "archivePolicy")
        )
    ):
        raise ValueError("receipt source boundary is unsupported")

    universe = _require_exact_keys(
        receipt["universe"], {"symbols", "stableOrder"}, "receipt universe"
    )
    symbols = status.get("symbols")
    if (
        universe["symbols"] != symbols
        or universe["stableOrder"] != "exact_fixed_collector_order"
        or not isinstance(symbols, list)
    ):
        raise ValueError("receipt universe disagrees with the collector status")

    receipt_artifacts = receipt["artifacts"]
    status_results = status.get("results")
    if (
        not isinstance(receipt_artifacts, dict)
        or set(receipt_artifacts) != set(symbols)
        or not isinstance(status_results, dict)
    ):
        raise ValueError("receipt artifact universe is incomplete")
    expected_files = {str(status_logical)}
    artifact_count = 0
    interval = status["interval"]
    verified_symbols = verified.get("symbols")
    if not isinstance(verified_symbols, dict):
        raise ValueError("collector verification result is malformed")
    for symbol in symbols:
        artifact_group = _require_exact_keys(
            receipt_artifacts[symbol], {"cache", "observations"}, f"{symbol} artifacts"
        )
        status_artifacts = _require_exact_keys(
            status_results[symbol].get("artifacts"),
            {"cache", "observations"},
            f"{symbol} status artifacts",
        )
        cache_logical = f"{symbol}_{interval}.csv"
        cache_record = _receipt_artifact(
            artifact_group["cache"],
            expected_logical_path=cache_logical,
            status_record=status_artifacts["cache"],
            label=f"{symbol} cache",
        )
        _resolve_archive_path(
            archive,
            _logical_path(cache_record["logicalPath"], f"{symbol} cache path"),
            f"{symbol} cache",
        )
        expected_files.add(cache_logical)
        artifact_count += 1

        observations = _require_exact_keys(
            artifact_group["observations"],
            set(collector.feed.DERIVATIVE_FIELDS),
            f"{symbol} observations",
        )
        status_observations = _require_exact_keys(
            status_artifacts["observations"],
            set(collector.feed.DERIVATIVE_FIELDS),
            f"{symbol} status observations",
        )
        verified_symbol = verified_symbols.get(symbol)
        if not isinstance(verified_symbol, dict):
            raise ValueError(f"{symbol} collector verification is missing")
        if (
            cache_record["rows"] != verified_symbol.get("rows")
            or cache_record["sha256"] != verified_symbol.get("cacheSha256")
        ):
            raise ValueError(f"{symbol} cache disagrees with artifact verification")
        verified_observations = verified_symbol.get("observations")
        for feature in collector.feed.DERIVATIVE_FIELDS:
            receipt_logical_path = f"{symbol}_{interval}_{feature}_v2.csv"
            archive_logical_path = f".observations/{receipt_logical_path}"
            record = _receipt_artifact(
                observations[feature],
                expected_logical_path=receipt_logical_path,
                status_record=status_observations[feature],
                label=f"{symbol} {feature} observation",
            )
            _resolve_archive_path(
                archive,
                _logical_path(archive_logical_path, f"{symbol} {feature} path"),
                f"{symbol} {feature} observation",
            )
            if (
                not isinstance(verified_observations, dict)
                or record["rows"] != verified_observations.get(feature)
            ):
                raise ValueError(
                    f"{symbol} {feature} disagrees with artifact verification"
                )
            expected_files.add(archive_logical_path)
            artifact_count += 1

    verification = _require_exact_keys(
        receipt["verification"],
        {"result", "archiveFileCount", "archiveApproximateMiB"},
        "receipt verification",
    )
    actual_files = _archive_files(archive)
    archive_bytes = sum(
        (archive / Path(*PurePosixPath(item).parts)).stat().st_size
        for item in actual_files
    )
    archive_mib = round(archive_bytes / (1024 * 1024), 1)
    recorded_archive_mib = verification["archiveApproximateMiB"]
    if (
        verification["result"] != "verified_in_place_and_relocated"
        or type(verification["archiveFileCount"]) is not int
        or verification["archiveFileCount"] != artifact_count + 1
        or verification["archiveFileCount"] != len(actual_files)
        or isinstance(recorded_archive_mib, bool)
        or not isinstance(recorded_archive_mib, (int, float))
        or not math.isfinite(recorded_archive_mib)
        or abs(recorded_archive_mib - archive_mib) > 0.11
        or actual_files != expected_files
    ):
        raise ValueError("receipt archive inventory disagrees with the frozen archive")

    return {
        "status": "verified",
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "receiptId": receipt_id,
        "statusSha256": status_sha256,
        "symbolsVerified": len(symbols),
        "artifactsVerified": artifact_count,
        "archiveFilesVerified": len(actual_files),
        "archiveBytesVerified": archive_bytes,
        "outcomeAdmission": "acquisition_metadata_only",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="committed receipt JSON")
    parser.add_argument("--archive", required=True, help="frozen external archive root")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = verify_receipt(args.receipt, args.archive)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"receipt verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
