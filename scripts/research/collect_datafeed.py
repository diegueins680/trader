#!/usr/bin/env python3
"""Run one locked, observable refresh of the research market-data cache."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import datafeed as feed


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
)
INTERVAL = "1h"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{3,24}USDT$")
STATE_DIR_NAME = ".collector"
FRESHNESS_GRACE_MS = 300_000
DEFAULT_MAX_RUN_SECONDS = 3000
MAX_RUN_SECONDS_LIMIT = 3500
STATUS_SCHEMA_VERSION = 3
ARTIFACT_SCHEMA_ID = "binance_derivatives_collection_artifacts_v3"
REQUEST_ORDER_POLICY = "utc_epoch_hour_rotation_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SOURCE_LICENSE_MANIFEST = (
    "research-notes/market-prediction-2026-09-04/"
    "data-source-license-manifest.json"
)


class CollectorStopped(BaseException):
    """Carry a terminal collector state out of an asynchronous signal."""

    def __init__(self, state: str, reason: str, exit_code: int):
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.exit_code = exit_code


class CollectorInterrupted(CollectorStopped):
    """Raised by SIGINT/SIGTERM."""


class CollectorTimedOut(CollectorStopped):
    """Raised when the one-shot collector exceeds its wall-clock budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_order(symbols: list[str], started_at: str) -> list[str]:
    """Rotate a fixed universe by UTC epoch hour without changing membership."""
    if not symbols:
        raise ValueError("request order requires at least one symbol")
    if not isinstance(started_at, str) or not started_at.endswith("Z"):
        raise ValueError("request order requires a canonical UTC start time")
    try:
        started = datetime.fromisoformat(started_at[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("request order requires a canonical UTC start time") from error
    if started.utcoffset() != timedelta(0):
        raise ValueError("request order requires a canonical UTC start time")
    epoch_seconds = int(started.timestamp())
    if epoch_seconds < 0:
        raise ValueError("request order start time predates the Unix epoch")
    offset = (epoch_seconds // 3600) % len(symbols)
    return symbols[offset:] + symbols[:offset]


def _validate_status_request_order(
    status: dict[str, object], symbols: list[str]
) -> list[str]:
    """Validate new rotated statuses while admitting legacy schema-3 statuses."""
    has_order = "requestOrder" in status
    has_policy = "requestOrderPolicy" in status
    if not has_order and not has_policy:
        return list(symbols)
    if not has_order or not has_policy:
        raise ValueError("collector request order metadata is incomplete")
    if status.get("requestOrderPolicy") != REQUEST_ORDER_POLICY:
        raise ValueError("collector request order policy is unsupported")
    request_order = status.get("requestOrder")
    if (
        not isinstance(request_order, list)
        or len(request_order) != len(symbols)
        or any(not isinstance(symbol, str) for symbol in request_order)
        or len(set(request_order)) != len(request_order)
        or set(request_order) != set(symbols)
    ):
        raise ValueError("collector request order is not a symbol permutation")
    started_at = status.get("startedAt")
    if not isinstance(started_at, str):
        raise ValueError("collector request order start time is missing")
    if request_order != _request_order(symbols, started_at):
        raise ValueError("collector request order disagrees with its UTC start time")
    return request_order


def _symbols_from_environment() -> list[str]:
    raw = os.environ.get("TRADER_RESEARCH_SYMBOLS", " ".join(DEFAULT_SYMBOLS))
    symbols = raw.replace(",", " ").split()
    if not symbols:
        raise ValueError("TRADER_RESEARCH_SYMBOLS must contain at least one symbol")
    invalid = [symbol for symbol in symbols if not SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid:
        raise ValueError(
            "research symbols must be uppercase USDT contracts: "
            + ", ".join(invalid)
        )
    if len(set(symbols)) != len(symbols):
        raise ValueError("TRADER_RESEARCH_SYMBOLS contains duplicates")
    return symbols


def _max_run_seconds_from_environment() -> int:
    raw = os.environ.get(
        "TRADER_RESEARCH_MAX_RUN_SECONDS", str(DEFAULT_MAX_RUN_SECONDS)
    )
    try:
        seconds = int(raw)
    except ValueError as error:
        raise ValueError("TRADER_RESEARCH_MAX_RUN_SECONDS must be an integer") from error
    if not 1 <= seconds <= MAX_RUN_SECONDS_LIMIT:
        raise ValueError(
            f"TRADER_RESEARCH_MAX_RUN_SECONDS must be between 1 and "
            f"{MAX_RUN_SECONDS_LIMIT}"
        )
    return seconds


def _repository_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _provenance_tracked_clean() -> bool:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "scripts/research/collect_datafeed.py",
                "scripts/research/datafeed.py",
                "scripts/research/historical_datafeed.py",
                SOURCE_LICENSE_MANIFEST,
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, rows: int) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "rows": rows,
    }


def _read_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    payload = path.read_bytes()
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(value, dict):
        raise ValueError("collector status must contain a JSON object")
    return value, payload


@contextmanager
def _artifact_read_lock(cache_dir: Path):
    """Share an existing collector lock without mutating a relocated archive."""
    lock_path = cache_dir / STATE_DIR_NAME / feed.COLLECTOR_LOCK_FILE
    if not lock_path.is_file():
        yield
        return
    with lock_path.open("rb") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("collector writer is active for the artifact cache") from error
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"artifact CSV is empty: {path}") from error
    if not header or any(not column for column in header):
        raise ValueError(f"artifact CSV header is malformed: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"artifact CSV header contains duplicates: {path}")
    return header


def _verify_artifact_record(
    record: object,
    recorded_path: Path,
    actual_path: Path,
) -> tuple[Path, int]:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "rows"}:
        raise ValueError("collector artifact record is malformed")
    path_value = record.get("path")
    sha256 = record.get("sha256")
    rows = record.get("rows")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("collector artifact path is not absolute")
    if Path(path_value) != recorded_path:
        raise ValueError("collector artifact path disagrees with its scope")
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("collector artifact sha256 is malformed")
    if type(rows) is not int or rows < 0:
        raise ValueError("collector artifact row count is malformed")
    resolved = actual_path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"collector artifact is not a file: {resolved}")
    if _file_sha256(resolved) != sha256:
        raise ValueError(f"collector artifact sha256 mismatch: {resolved}")
    return resolved, rows


def verify_collection_artifacts(
    status_path: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Verify one green collector status and every artifact it binds."""
    status_file = Path(status_path).expanduser().resolve(strict=True)
    status, status_payload = _read_json_object(status_file)
    if status.get("schemaVersion") != STATUS_SCHEMA_VERSION:
        raise ValueError("unsupported collector status schemaVersion")
    if status.get("state") != "pass":
        raise ValueError("collector status is not a complete pass")
    if status.get("failedSymbols") != []:
        raise ValueError("collector status has failed symbols")
    if status.get("provenanceIssues") != []:
        raise ValueError("collector status has provenance issues")
    if status.get("artifactSchema") != ARTIFACT_SCHEMA_ID:
        raise ValueError("collector artifact schema is unsupported")
    if (
        status.get("derivativesObservationSchema")
        != feed.DERIVATIVE_OBSERVATION_SCHEMA_ID
        or status.get("featureAvailabilitySchema")
        != feed.FEATURE_AVAILABILITY_SCHEMA_ID
        or status.get("dataSourceLicenseManifest") != SOURCE_LICENSE_MANIFEST
    ):
        raise ValueError("collector provenance schema is unsupported")
    commit = status.get("commit")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("collector code commit is missing or malformed")
    if status.get("provenanceTrackedClean") is not True:
        raise ValueError("collector provenance files did not match the code commit")
    runtime = status.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"python", "numpy", "pandas"}
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise ValueError("collector runtime provenance is malformed")
    interval = status.get("interval")
    if interval != INTERVAL or interval not in feed.INTERVAL_MS:
        raise ValueError("collector status interval is unsupported")
    symbols = status.get("symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(
            not isinstance(symbol, str) or not SYMBOL_PATTERN.fullmatch(symbol)
            for symbol in symbols
        )
        or len(symbols) != len(set(symbols))
    ):
        raise ValueError("collector status symbols are malformed")
    _validate_status_request_order(status, symbols)
    results = status.get("results")
    if not isinstance(results, dict) or set(results) != set(symbols):
        raise ValueError("collector status results disagree with its symbols")
    recorded_cache_value = status.get("cache")
    if not isinstance(recorded_cache_value, str):
        raise ValueError("collector status cache path is malformed")
    recorded_cache = Path(recorded_cache_value)
    if not recorded_cache.is_absolute():
        raise ValueError("collector status cache path is not absolute")
    actual_cache = (
        Path(cache_dir).expanduser().resolve(strict=True)
        if cache_dir is not None
        else recorded_cache
    )
    if not actual_cache.is_dir():
        raise ValueError("collector artifact cache directory is unavailable")

    verified: dict[str, dict[str, object]] = {}
    with _artifact_read_lock(actual_cache):
        for symbol in symbols:
            verified[symbol] = _verify_collection_symbol(
                symbol,
                results[symbol],
                interval,
                recorded_cache,
                actual_cache,
            )
    return {
        "status": "verified",
        "schemaVersion": STATUS_SCHEMA_VERSION,
        "artifactSchema": ARTIFACT_SCHEMA_ID,
        "statusSha256": hashlib.sha256(status_payload).hexdigest(),
        "symbols": verified,
    }


def _verify_collection_symbol(
    symbol: str,
    result: object,
    interval: str,
    recorded_cache: Path,
    actual_cache: Path,
) -> dict[str, object]:
    if not isinstance(result, dict):
        raise ValueError(f"{symbol} collector result is malformed")
    if result.get("status") != "ok" or result.get("issues") != []:
        raise ValueError(f"{symbol} collector result is not admissible")
    if result.get("artifactSchema") != ARTIFACT_SCHEMA_ID:
        raise ValueError(f"{symbol} artifact schema is unsupported")
    if (
        result.get("derivativesObservationSchema")
        != feed.DERIVATIVE_OBSERVATION_SCHEMA_ID
        or result.get("featureAvailabilitySchema")
        != feed.FEATURE_AVAILABILITY_SCHEMA_ID
    ):
        raise ValueError(f"{symbol} availability schema is unsupported")
    columns = result.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) for column in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError(f"{symbol} column manifest is malformed")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "cache",
        "observations",
    }:
        raise ValueError(f"{symbol} artifact set is incomplete")
    observations = artifacts.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(
        feed.DERIVATIVE_FIELDS
    ):
        raise ValueError(f"{symbol} observation artifact set is incomplete")

    cache_name = f"{symbol}_{interval}.csv"
    cache_path, cache_rows = _verify_artifact_record(
        artifacts.get("cache"),
        recorded_cache / cache_name,
        actual_cache / cache_name,
    )
    if _read_csv_header(cache_path) != columns:
        raise ValueError(f"{symbol} cache columns disagree with its manifest")
    frame = pd.read_csv(cache_path)
    if len(frame) != cache_rows or result.get("rows") != cache_rows:
        raise ValueError(f"{symbol} cache row count disagrees with its manifest")
    coverage = feed.validate_derivative_v2_panel(frame, interval)
    if coverage != result.get("derivativesV2Coverage"):
        raise ValueError(f"{symbol} v2 coverage disagrees with its manifest")
    fields = feed.DERIVATIVE_FIELDS
    if any(field not in columns for field in fields):
        raise ValueError(f"{symbol} cache is missing legacy derivative columns")
    finite = {
        field: int(
            np.isfinite(
                pd.to_numeric(frame[field], errors="coerce").to_numpy(
                    dtype=float
                )
            ).sum()
        )
        for field in fields
    }
    joint_finite = int(
        np.isfinite(
            frame.loc[:, fields]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        .all(axis=1)
        .sum()
    )
    if finite != result.get("finite") or joint_finite != result.get(
        "jointFinite"
    ):
        raise ValueError(f"{symbol} legacy coverage disagrees with its manifest")
    timestamps = pd.to_numeric(frame["openTime"], errors="raise")
    if (
        timestamps.empty
        or timestamps.duplicated().any()
        or not timestamps.is_monotonic_increasing
    ):
        raise ValueError(f"{symbol} cache timestamps are not sorted and unique")
    if (
        result.get("latestOpenTime") != int(timestamps.iloc[-1])
        or result.get("freshLatestOpenTime") != int(timestamps.iloc[-1])
    ):
        raise ValueError(f"{symbol} cache tail disagrees with its manifest")

    interval_ms = feed.INTERVAL_MS[interval]
    snapshot_end = int(timestamps.iloc[-1]) + interval_ms - 1

    refresh_series = result.get("refreshSeries")
    if not isinstance(refresh_series, dict) or set(refresh_series) != set(fields):
        raise ValueError(f"{symbol} refresh series is malformed")
    observation_rows: dict[str, int] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for feature in fields:
        observation_name = f"{symbol}_{interval}_{feature}_v2.csv"
        observation_path, row_count = _verify_artifact_record(
            observations.get(feature),
            recorded_cache / feed.DERIVATIVE_OBSERVATION_DIR / observation_name,
            actual_cache / feed.DERIVATIVE_OBSERVATION_DIR / observation_name,
        )
        if _read_csv_header(observation_path) != list(
            feed.DERIVATIVE_OBSERVATION_COLUMNS
        ):
            raise ValueError(f"{symbol} {feature} ledger columns changed")
        ledger = feed._validated_observation_frame(
            pd.read_csv(observation_path), symbol, interval, feature
        )
        series = refresh_series.get(feature)
        max_age_ms = (
            feed.FUNDING_FRESHNESS_MS
            if feature == "funding"
            else 2 * interval_ms
        )
        if (
            len(ledger) != row_count
            or not isinstance(series, dict)
            or series.get("status") != "ok"
            or series.get("v2Status") != "ok"
            or series.get("v2Observations") != row_count
        ):
            raise ValueError(f"{symbol} {feature} ledger evidence changed")
        _validate_refresh_series_evidence(
            symbol,
            feature,
            series,
            snapshot_end,
            interval_ms,
            max_age_ms,
        )
        observation_rows[feature] = row_count
        ledgers[feature] = ledger

    closes = timestamps.to_numpy(dtype=np.int64) + interval_ms - 1
    fresh_rows = result.get("freshRows")
    if type(fresh_rows) is not int or not 1 <= fresh_rows <= cache_rows:
        raise ValueError(f"{symbol} fresh row count is malformed")
    for feature in fields:
        max_age_ms = (
            feed.FUNDING_FRESHNESS_MS
            if feature == "funding"
            else 2 * interval_ms
        )
        aligned = feed.align_derivative_observations_v2(
            closes,
            ledgers[feature],
            max_age_ms=max_age_ms,
        )
        prefix = f"{feature}V2"
        source_columns = (
            ("value", "Value"),
            ("observed", "Observed"),
            ("fresh", "Fresh"),
            ("eventTime", "EventTime"),
            ("availabilityTime", "AvailabilityTime"),
        )
        versioned = frame[prefix + "Value"].notna().to_numpy()
        if not versioned[-fresh_rows:].all():
            raise ValueError(f"{symbol} {feature} fresh tail is unversioned")
        for source, suffix in source_columns:
            actual = pd.to_numeric(
                frame[prefix + suffix], errors="raise"
            ).to_numpy(dtype=float)
            expected = aligned[source].to_numpy(dtype=float)
            if not np.array_equal(
                actual[versioned], expected[versioned], equal_nan=True
            ):
                raise ValueError(
                    f"{symbol} {feature} {suffix} disagrees with its ledger"
                )
    _verify_artifact_record(
        artifacts.get("cache"),
        recorded_cache / cache_name,
        actual_cache / cache_name,
    )
    for feature in fields:
        observation_name = f"{symbol}_{interval}_{feature}_v2.csv"
        _verify_artifact_record(
            observations.get(feature),
            recorded_cache / feed.DERIVATIVE_OBSERVATION_DIR / observation_name,
            actual_cache / feed.DERIVATIVE_OBSERVATION_DIR / observation_name,
        )
    return {
        "cacheSha256": artifacts["cache"]["sha256"],
        "rows": cache_rows,
        "observations": observation_rows,
        "coverage": coverage,
    }


def _validate_refresh_series_evidence(
    symbol: str,
    feature: str,
    series: dict[str, object],
    snapshot_end: int,
    interval_ms: int,
    max_age_ms: int,
) -> None:
    integer_fields = (
        "observations",
        "finite",
        "latestTimestamp",
        "latestObservationTimestamp",
        "lagMs",
        "trailingUnavailable",
    )
    if any(type(series.get(field)) is not int for field in integer_fields):
        raise ValueError(f"{symbol} {feature} refresh evidence is malformed")
    observations = series["observations"]
    finite = series["finite"]
    latest = series["latestTimestamp"]
    latest_observation = series["latestObservationTimestamp"]
    lag_ms = series["lagMs"]
    trailing_unavailable = series["trailingUnavailable"]
    if (
        observations <= 0
        or finite <= 0
        or finite > observations
        or latest < 0
        or latest_observation < latest
        or latest_observation > snapshot_end
        or lag_ms != snapshot_end - latest
        or lag_ms < 0
        or lag_ms > max_age_ms
        or trailing_unavailable < 0
        or trailing_unavailable > observations - finite
        or latest_observation - latest
        != trailing_unavailable * interval_ms
    ):
        raise ValueError(f"{symbol} {feature} refresh evidence is malformed")


def _cache_result(
    symbol: str, refresh: dict[str, object] | None
) -> dict[str, object]:
    if not refresh or refresh.get("status") != "updated":
        state = refresh.get("status") if refresh else "missing"
        raise RuntimeError(f"{symbol} refresh did not update klines ({state})")
    fresh_rows = refresh.get("freshRows")
    fresh_latest = refresh.get("freshLatestOpenTime")
    if type(fresh_rows) is not int or fresh_rows <= 0 or type(fresh_latest) is not int:
        raise RuntimeError(f"{symbol} refresh evidence is malformed")
    now_ms = int(time.time() * 1000)
    interval_ms = feed.INTERVAL_MS[INTERVAL]
    latest_close = fresh_latest + interval_ms
    lag_ms = now_ms - latest_close
    if lag_ms < -FRESHNESS_GRACE_MS or lag_ms > interval_ms + FRESHNESS_GRACE_MS:
        raise RuntimeError(f"{symbol} newest fetched kline is stale (lagMs={lag_ms})")

    path = Path(feed._cache_path(symbol, INTERVAL))
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{symbol} cache is empty after refresh")
    if _read_csv_header(path) != list(frame.columns):
        raise ValueError(f"{symbol} cache header is ambiguous")
    v2_coverage = feed.validate_derivative_v2_panel(frame, INTERVAL)
    timestamps = pd.to_numeric(frame["openTime"], errors="raise").astype(np.int64)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{symbol} cache timestamps are not sorted and unique")
    if int(timestamps.iloc[-1]) != fresh_latest:
        raise ValueError(f"{symbol} cache tail does not match the fetched kline")
    fields = feed.DERIVATIVE_FIELDS
    finite = {
        field: int(
            np.isfinite(
                pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
            ).sum()
        )
        for field in fields
    }
    joint = np.isfinite(
        frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ).all(axis=1)
    series = refresh.get("series")
    if not isinstance(series, dict):
        raise RuntimeError(f"{symbol} refresh has no derivative-series evidence")
    issues = [
        field
        for field in fields
        if not isinstance(series.get(field), dict)
        or series[field].get("status") != "ok"
        or series[field].get("observationSchema")
        != feed.DERIVATIVE_OBSERVATION_SCHEMA_ID
        or series[field].get("v2Status") != "ok"
    ]
    fresh_tail_start = max(0, len(frame) - fresh_rows)
    for field in fields:
        prefix = f"{field}V2"
        tail = frame.iloc[fresh_tail_start:]
        if tail[[prefix + "Value", prefix + "Observed", prefix + "Fresh"]].isna().any().any():
            if field not in issues:
                issues.append(field)
    observation_artifacts: dict[str, dict[str, object] | None] = {}
    for field in fields:
        observation_path = Path(feed._observation_path(symbol, INTERVAL, field))
        if not observation_path.is_file():
            observation_artifacts[field] = None
            if field not in issues:
                issues.append(field)
            continue
        observations = feed._validated_observation_frame(
            pd.read_csv(observation_path), symbol, INTERVAL, field
        )
        observation_artifacts[field] = _artifact_record(
            observation_path, len(observations)
        )
    return {
        "status": "ok" if not issues else "degraded",
        "rows": len(frame),
        "latestOpenTime": int(timestamps.iloc[-1]),
        "freshRows": fresh_rows,
        "freshLatestOpenTime": fresh_latest,
        "freshnessLagMs": lag_ms,
        "refreshSeries": series,
        "issues": issues,
        "finite": finite,
        "jointFinite": int(joint.sum()),
        "derivativesObservationSchema": feed.DERIVATIVE_OBSERVATION_SCHEMA_ID,
        "featureAvailabilitySchema": feed.FEATURE_AVAILABILITY_SCHEMA_ID,
        "derivativesV2Coverage": v2_coverage,
        "artifactSchema": ARTIFACT_SCHEMA_ID,
        "columns": list(frame.columns),
        "artifacts": {
            "cache": _artifact_record(path, len(frame)),
            "observations": observation_artifacts,
        },
    }


def _run_locked(cache_dir: Path, symbols: list[str], status_path: Path) -> int:
    started_monotonic = time.monotonic()
    results: dict[str, dict[str, object]] = {}
    status: dict[str, object] = {
        "schemaVersion": STATUS_SCHEMA_VERSION,
        "artifactSchema": ARTIFACT_SCHEMA_ID,
        "derivativesObservationSchema": feed.DERIVATIVE_OBSERVATION_SCHEMA_ID,
        "featureAvailabilitySchema": feed.FEATURE_AVAILABILITY_SCHEMA_ID,
        "dataSourceLicenseManifest": SOURCE_LICENSE_MANIFEST,
        "state": "starting",
        "cache": str(cache_dir),
        "interval": INTERVAL,
        "symbols": symbols,
        "results": results,
    }
    failures = 0
    try:
        started_at = _utc_now()
        request_order = _request_order(symbols, started_at)
        commit = _repository_commit()
        provenance_tracked_clean = _provenance_tracked_clean()
        provenance_issues = []
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            provenance_issues.append("code_commit_unavailable")
        if not provenance_tracked_clean:
            provenance_issues.append("provenance_files_differ_from_commit")
        status.update(
            {
                "state": "running",
                "startedAt": started_at,
                "updatedAt": started_at,
                "commit": commit,
                "provenanceTrackedClean": provenance_tracked_clean,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
                "requestOrder": request_order,
                "requestOrderPolicy": REQUEST_ORDER_POLICY,
                "provenanceIssues": provenance_issues,
            }
        )
        _write_json_atomic(status_path, status)
        for symbol_index, symbol in enumerate(request_order):
            try:
                refreshes = feed.update_cache(
                    [symbol], INTERVAL, acquire_lock=False
                )
                results[symbol] = _cache_result(symbol, refreshes.get(symbol))
                if results[symbol]["status"] != "ok":
                    failures += 1
                    print(
                        f"{symbol}: derivative refresh degraded "
                        f"({', '.join(results[symbol]['issues'])})",
                        file=sys.stderr,
                    )
            except feed.BinanceRateLimitError as error:
                failures += 1
                results[symbol] = {
                    "status": "error",
                    "failureKind": "provider_rate_limit",
                    "error": str(error),
                    "httpStatus": error.http_status,
                    "bannedUntilMs": error.banned_until_ms,
                    "retryAfterSeconds": error.retry_after_seconds,
                }
                for skipped_symbol in request_order[symbol_index + 1 :]:
                    failures += 1
                    results[skipped_symbol] = {
                        "status": "skipped",
                        "failureKind": "provider_rate_limit_circuit_open",
                        "error": "not attempted after Binance public API rate limit",
                    }
                print(
                    f"{symbol}: refresh stopped ({results[symbol]['error']}); "
                    f"skipped {len(request_order) - symbol_index - 1} remaining symbols",
                    file=sys.stderr,
                )
                status["results"] = results
                status["updatedAt"] = _utc_now()
                _write_json_atomic(status_path, status)
                break
            except Exception as error:
                failures += 1
                results[symbol] = {
                    "status": "error",
                    "error": str(error).replace("\n", " ")[:240],
                }
                print(f"{symbol}: refresh failed ({results[symbol]['error']})", file=sys.stderr)
            status["results"] = results
            status["updatedAt"] = _utc_now()
            _write_json_atomic(status_path, status)

        finished_at = _utc_now()
        complete = failures == 0 and not provenance_issues
        status.update(
            {
                "state": "pass" if complete else "partial_failure",
                "finishedAt": finished_at,
                "updatedAt": finished_at,
                "durationSeconds": round(time.monotonic() - started_monotonic, 3),
                "failedSymbols": [
                    symbol
                    for symbol, result in results.items()
                    if result["status"] != "ok"
                ],
                "results": results,
            }
        )
        _write_json_atomic(status_path, status)
        print(
            f"research data collector {status['state']}: "
            f"{len(symbols) - failures}/{len(symbols)} symbols refreshed"
        )
        return 0 if complete else 1
    except CollectorStopped as stopped:
        stopped_at = _utc_now()
        status.update(
            {
                "state": stopped.state,
                "finishedAt": stopped_at,
                "updatedAt": stopped_at,
                "durationSeconds": round(time.monotonic() - started_monotonic, 3),
                "stopReason": stopped.reason,
                "failedSymbols": [
                    symbol
                    for symbol, result in results.items()
                    if result["status"] != "ok"
                ],
                "results": results,
            }
        )
        _write_terminal_status(status_path, status)
        return stopped.exit_code


def main() -> int:
    try:
        symbols = _symbols_from_environment()
        max_run_seconds = _max_run_seconds_from_environment()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    cache_dir = Path(
        os.environ.get("TRADER_RESEARCH_CACHE") or feed.CACHE_DIR
    ).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    feed.CACHE_DIR = str(cache_dir)
    state_dir = cache_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "last-run.json"
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    }
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGALRM, _deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max_run_seconds)
    try:
        try:
            with feed.cache_writer_lock(blocking=False):
                return _run_locked(cache_dir, symbols, status_path)
        except feed.CacheWriterBusy:
            print(f"research data collector already running for {cache_dir}")
            return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def _interrupt(signum: int, _frame: object) -> None:
    raise CollectorInterrupted("interrupted", signal.Signals(signum).name, 130)


def _deadline(_signum: int, _frame: object) -> None:
    raise CollectorTimedOut("timeout", "wall_clock_deadline", 124)


def _write_terminal_status(path: Path, status: dict[str, object]) -> None:
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }
    try:
        for handled_signal in handled_signals:
            signal.signal(handled_signal, signal.SIG_IGN)
        _write_json_atomic(path, status)
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def _cli(argv: list[str]) -> int:
    if not argv:
        return main()
    parser = argparse.ArgumentParser(
        description="Collect or verify the public derivatives research cache"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify-artifacts",
        help="verify a green collector status and its bound cache artifacts",
    )
    verify_parser.add_argument("--status", required=True)
    verify_parser.add_argument(
        "--cache-dir",
        help="relocated cache root containing bytes that match the recorded hashes",
    )
    args = parser.parse_args(argv)
    if args.command != "verify-artifacts":
        parser.error("unsupported command")
    try:
        verification = verify_collection_artifacts(
            args.status,
            cache_dir=args.cache_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"derivatives artifact verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(verification, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
