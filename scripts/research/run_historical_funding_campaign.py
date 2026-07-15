#!/usr/bin/env python3
"""Run the fixed historical residual-momentum funding campaign.

The committed registration fixes the universe, time grid, six trials, and all
validation gates. The final chronological holdout remains sealed unless every
development gate passes and ``--open-final-holdout`` is explicitly requested.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

import campaign_runner as C
import diagnostics
import funding_campaign as F
import historical_datafeed as feed


CAMPAIGN_ID = "residual_momentum_funding_only_v1"
REGISTRATION_VERSION = 1
REGISTRATION_PATH = (
    C.REPOSITORY_ROOT
    / "research-notes/registrations/residual-momentum-funding-only-v1.json"
)
IMPLEMENTATION_FILES = (
    "campaign_runner.py",
    "diagnostics.py",
    "funding_campaign.py",
    "harness.py",
    "historical_datafeed.py",
    "run_historical_funding_campaign.py",
)
REGISTERED_PANEL_COLUMNS = ("openTime", "closeTime", "close")
HOLDOUT_REGISTRY_DIR = C.HOLDOUT_REGISTRY_DIR
REGISTERED_SYMBOLS = (
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
REGISTERED_START_OPEN_TIME = 1_600_819_200_000
REGISTERED_END_OPEN_TIME = 1_777_564_800_000
REGISTERED_OUTCOME_END_EXCLUSIVE = 1_777_593_600_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered six-trial historical funding campaign"
    )
    parser.add_argument(
        "--snapshot-dir",
        default=".tmp/research/historical-funding-snapshot-v1",
        help="Immutable raw Binance snapshot directory",
    )
    parser.add_argument(
        "--output-dir",
        default=".tmp/research/historical-funding-campaign-v1",
        help="Registered evidence directory",
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Acquire the fixed public snapshot if it is not already sealed",
    )
    parser.add_argument("--open-final-holdout", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"JSON artifact is unreadable: {path}") from error


def _read_json_bytes(path: Path) -> tuple[bytes, object]:
    try:
        payload = path.read_bytes()
        return payload, json.loads(payload)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"JSON artifact is unreadable: {path}") from error


@contextmanager
def _snapshot_lock(snapshot_dir: Path, deadline: float) -> Iterator[None]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with (snapshot_dir / ".snapshot.lock").open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("snapshot lock deadline exceeded")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _validate_registration(
    registration: dict[str, object],
) -> dict[str, object]:
    try:
        universe = registration["universe"]
        registered_data = registration["registeredData"]
        strategy = registration["strategy"]
        validation = registration["validation"]
        promotion = registration["promotion"]
        holdout_policy = registration["holdoutPolicy"]
        trials = registration["trials"]
        if not all(
            isinstance(value, dict)
            for value in (
                universe,
                registered_data,
                strategy,
                validation,
                promotion,
                holdout_policy,
            )
        ):
            raise TypeError
        symbols = universe["symbols"]
        if not isinstance(symbols, list) or any(
            not isinstance(symbol, str) or not symbol for symbol in symbols
        ):
            raise TypeError
        if len(symbols) != len(set(symbols)):
            raise ValueError("registered symbols must be unique")
        expected_trials = [spec.to_dict() for spec in F.campaign_specs(feed.CONTRACT_INTERVAL_MS)]
        normalized_trials = [
            {
                "trial_id": row["id"],
                "horizon_hours": int(row["horizonHours"]),
                "horizon_bars": int(row["horizonBars"]),
                "variant": row["variant"],
            }
            for row in trials
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("campaign registration has an invalid schema") from error

    fixed_checks = {
        "campaign": registration.get("campaign") == CAMPAIGN_ID,
        "registrationVersion": registration.get("registrationVersion")
        == REGISTRATION_VERSION,
        "interval": universe.get("interval") == feed.CONTRACT_INTERVAL,
        "intervalMilliseconds": universe.get("intervalMilliseconds")
        == feed.CONTRACT_INTERVAL_MS,
        "trialLedger": normalized_trials == expected_trials,
        "trialCount": len(expected_trials) == 6,
        "symbols": tuple(symbols) == REGISTERED_SYMBOLS,
        "startOpenTime": int(registered_data["startOpenTime"])
        == REGISTERED_START_OPEN_TIME,
        "endOpenTime": int(registered_data["endOpenTime"])
        == REGISTERED_END_OPEN_TIME,
        "outcomeEndTimeExclusive": int(
            registered_data["outcomeEndTimeExclusive"]
        )
        == REGISTERED_OUTCOME_END_EXCLUSIVE,
        "contractEndpoint": registered_data["contractEndpoint"]
        == "/fapi/v1/continuousKlines",
        "fundingEndpoint": registered_data["fundingEndpoint"]
        == "/fapi/v1/fundingRate",
        "markEndpoint": registered_data["markPriceEndpoint"]
        == "/fapi/v1/markPriceKlines",
        "markInterval": registered_data["markPriceInterval"] == "1h",
        "rowSplit": int(registered_data["developmentRows"])
        + int(registered_data["holdoutBars"])
        == int(registered_data["rows"]),
        "developmentRows": int(registered_data["developmentRows"]) == 4910,
        "developmentCutoff": int(registered_data["developmentCutoffOpenTime"])
        == int(registered_data["startOpenTime"])
        + (int(registered_data["developmentRows"]) - 1)
        * feed.CONTRACT_INTERVAL_MS,
        "holdoutStart": int(registered_data["holdoutStartOpenTime"])
        == int(registered_data["developmentCutoffOpenTime"])
        + feed.CONTRACT_INTERVAL_MS,
        "fundingWindowEnd": int(registered_data["fundingWindowEndInclusive"])
        == int(registered_data["outcomeEndTimeExclusive"]) - 1,
        "returnRows": int(registered_data["holdoutReturnRows"])
        == int(registered_data["holdoutBars"]) - 1,
        "lifetimeTrials": int(validation["priorTrialCount"])
        + int(validation["newTrialCount"])
        == int(validation["lifetimeTrialCount"]),
        "newTrials": int(validation["newTrialCount"]) == len(expected_trials),
        "signalDelay": int(strategy["signalDelayBars"]) == 1,
        "rebalanceBars": int(strategy["rebalanceBars"]) == 1,
        "betaLookbackBars": int(strategy["betaLookbackBars"]) == 21,
        "fundingZLookbackBars": int(strategy["fundingZLookbackBars"]) == 21,
        "topNPerSide": int(strategy["topNPerSide"]) == 1,
        "grossExposure": float(strategy["grossExposure"]) == 1.0,
        "costBps": float(strategy["costBpsPerUnitTurnover"]) == 5.0,
        "fundingCrowdingZ": float(strategy["fundingCrowdingZ"]) == 2.0,
        "fundingCoverage": float(promotion["minimumResolvedFundingFraction"])
        == 1.0,
        "fundingGap": int(registered_data["maximumFundingGapMilliseconds"])
        == feed.CONTRACT_INTERVAL_MS + 60_000,
        "acquisitionDeadline": int(registered_data["acquisitionMaxSeconds"])
        == 3600,
        "bootstrapBlock": int(validation["bootstrapBlockBars"]) == 3,
        "outerInitialTrain": int(validation["outerInitialTrain"]) == 2444,
        "outerTestSize": int(validation["outerTestSize"]) == 349,
        "outerFoldCount": int(validation["outerFoldCount"]) == 7,
        "innerInitialTrain": int(validation["innerInitialTrain"]) == 1222,
        "innerTestSize": int(validation["innerTestSize"]) == 244,
        "pairedHypotheses": int(validation["pairedComparisonHypotheses"]) == 3,
        "pairedFamilyAlpha": float(
            validation["pairedComparisonFamilyWiseAlpha"]
        )
        == 0.05,
        "stressTests": validation["stressTests"]
        == ["cost1_5x", "cost2x", "additionalDelay1bar"],
        "requireOuterCi": promotion["requireOuterOosSharpeCiAboveZero"] is True,
        "requireCostCi": promotion["requireCost2xSharpeCiAboveZero"] is True,
        "requireDelayCi": promotion[
            "requireAdditionalDelaySharpeCiAboveZero"
        ]
        is True,
        "requirePairedCi": promotion[
            "requirePairedFundingImprovementSharpeCiAboveZero"
        ]
        is True,
        "holdoutReserved": holdout_policy["reservedByDefault"] is True,
        "holdoutGated": holdout_policy[
            "openOnlyAfterEveryDevelopmentGatePasses"
        ]
        is True,
        "holdoutRegistry": holdout_policy["overlapAwareOneShotRegistry"] is True,
    }
    failed = [name for name, passed in fixed_checks.items() if not passed]
    if failed:
        raise ValueError(
            "campaign registration violates fixed constraints: " + ", ".join(failed)
        )
    start = int(registered_data["startOpenTime"])
    end = int(registered_data["endOpenTime"])
    expected_rows = (end - start) // feed.CONTRACT_INTERVAL_MS + 1
    if expected_rows != int(registered_data["rows"]):
        raise ValueError("registered market-data row count does not match its time grid")
    if int(registered_data["outcomeEndTimeExclusive"]) != end + feed.CONTRACT_INTERVAL_MS:
        raise ValueError("registered outcome end does not match the final bar")
    return registration


def _registration(path: Path = REGISTRATION_PATH) -> dict[str, object]:
    return _validate_registration(_read_json_object(path))


def _registration_and_sha(
    path: Path = REGISTRATION_PATH,
) -> tuple[dict[str, object], str]:
    payload, value = _read_json_bytes(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return _validate_registration(value), hashlib.sha256(payload).hexdigest()


def _registration_sha(path: Path = REGISTRATION_PATH) -> str:
    return C._file_digest(path)


def _implementation_sha() -> str:
    return C._implementation_digest(Path(__file__).resolve().parent, IMPLEMENTATION_FILES)


def _artifact_name(symbol: str, kind: str) -> str:
    suffix = {
        "contract": "contract-8h.json",
        "mark": "mark-price-1h.json",
        "funding": "funding-events.json",
    }.get(kind)
    if suffix is None:
        raise ValueError(f"unknown snapshot artifact kind: {kind}")
    return f"{symbol}-{suffix}"


def _require_acquisition_time(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("historical snapshot acquisition deadline exceeded")


def _acquire_snapshot(
    snapshot_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    deadline: float,
) -> dict[str, object]:
    _require_acquisition_time(deadline)
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if manifest_path.exists():
        return _read_json_object(manifest_path)
    universe = registration["universe"]
    registered_data = registration["registeredData"]
    if not isinstance(universe, Mapping) or not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid acquisition settings")
    start = int(registered_data["startOpenTime"])
    last_open = int(registered_data["endOpenTime"])
    funding_end = int(registered_data["fundingWindowEndInclusive"])
    symbols = [str(symbol) for symbol in universe["symbols"]]
    progress_path = snapshot_dir / "snapshot-progress.json"
    header = {
        "campaign": CAMPAIGN_ID,
        "snapshotVersion": 1,
        "registrationSha256": registration_sha,
        "window": {
            "startOpenTime": start,
            "endOpenTime": last_open,
            "fundingWindowEndInclusive": funding_end,
        },
        "symbols": symbols,
    }
    if progress_path.exists():
        progress = _read_json_object(progress_path)
        for name, expected in header.items():
            if progress.get(name) != expected:
                raise ValueError(
                    f"snapshot progress mismatch for {name}; use a new snapshot directory"
                )
        artifacts_value = progress.get("artifacts")
        if not isinstance(artifacts_value, dict):
            raise ValueError("snapshot progress has no artifacts object")
        artifacts: dict[str, dict[str, object]] = {
            str(key): dict(value)
            for key, value in artifacts_value.items()
            if isinstance(value, Mapping)
        }
        if len(artifacts) != len(artifacts_value):
            raise ValueError("snapshot progress artifact metadata is invalid")
    else:
        progress = {
            **header,
            "startedAt": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "artifacts": {},
        }
        artifacts = {}
        C._write_json(progress_path, progress)

    expected_artifact_keys = {
        f"{symbol}:{kind}"
        for symbol in symbols
        for kind in ("contract", "mark", "funding")
    }
    if not set(artifacts).issubset(expected_artifact_keys):
        raise ValueError("snapshot progress contains an unregistered artifact")

    fetchers = {
        "contract": lambda symbol: feed.fetch_contract_klines(
            symbol, start, last_open, deadline=deadline
        ),
        "mark": lambda symbol: feed.fetch_mark_price_klines(
            symbol, start, funding_end, deadline=deadline
        ),
        "funding": lambda symbol: feed.fetch_funding_events(
            symbol, start, funding_end, deadline=deadline
        ),
    }
    for symbol in symbols:
        for kind, fetcher in fetchers.items():
            _require_acquisition_time(deadline)
            key = f"{symbol}:{kind}"
            path = snapshot_dir / _artifact_name(symbol, kind)
            existing = artifacts.get(key)
            if isinstance(existing, Mapping) and path.is_file():
                payload, value = _read_json_bytes(path)
                digest = hashlib.sha256(payload).hexdigest()
                if (
                    existing.get("file") == path.name
                    and existing.get("sha256") == digest
                    and isinstance(value, list)
                    and len(value) == int(existing.get("rows", -1))
                ):
                    continue
                raise ValueError(f"resumable snapshot artifact changed: {key}")
            if existing is not None:
                raise ValueError(f"resumable snapshot artifact is missing: {key}")
            rows = fetcher(symbol)
            _require_acquisition_time(deadline)
            if not rows:
                raise ValueError(f"{symbol} {kind} snapshot is empty")
            feed.write_artifact_atomic(rows, path)
            _require_acquisition_time(deadline)
            artifacts[key] = {
                "file": path.name,
                "rows": len(rows),
                "sha256": C._file_digest(path),
            }
            progress["artifacts"] = artifacts
            C._write_json(progress_path, progress)
            _require_acquisition_time(deadline)

    _require_acquisition_time(deadline)
    manifest: dict[str, object] = {
        **header,
        "acquiredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifacts": artifacts,
    }
    C._write_json(manifest_path, manifest)
    if time.monotonic() >= deadline:
        manifest_path.unlink(missing_ok=True)
        raise TimeoutError("historical snapshot acquisition deadline exceeded")
    progress_path.unlink(missing_ok=True)
    return manifest


def _load_snapshot(
    snapshot_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    acquire: bool,
) -> tuple[dict[str, object], str, dict[str, dict[str, list[object]]]]:
    registered_data = registration["registeredData"]
    if not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid acquisition deadline")
    deadline = time.monotonic() + int(registered_data["acquisitionMaxSeconds"])
    with _snapshot_lock(snapshot_dir, deadline):
        return _load_snapshot_locked(
            snapshot_dir, registration, registration_sha, acquire, deadline
        )


def _load_snapshot_locked(
    snapshot_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    acquire: bool,
    deadline: float,
) -> tuple[dict[str, object], str, dict[str, dict[str, list[object]]]]:
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if not manifest_path.exists():
        if not acquire:
            raise ValueError("historical snapshot is absent; pass --acquire")
        _acquire_snapshot(
            snapshot_dir, registration, registration_sha, deadline
        )
    manifest_payload, manifest_value = _read_json_bytes(manifest_path)
    _require_acquisition_time(deadline)
    if not isinstance(manifest_value, dict):
        raise ValueError("snapshot manifest must contain an object")
    manifest = manifest_value
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()

    universe = registration["universe"]
    registered_data = registration["registeredData"]
    if not isinstance(universe, Mapping) or not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid snapshot settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    expected_window = {
        "startOpenTime": int(registered_data["startOpenTime"]),
        "endOpenTime": int(registered_data["endOpenTime"]),
        "fundingWindowEndInclusive": int(registered_data["fundingWindowEndInclusive"]),
    }
    expected_header = {
        "campaign": CAMPAIGN_ID,
        "snapshotVersion": 1,
        "registrationSha256": registration_sha,
        "window": expected_window,
        "symbols": symbols,
    }
    for name, expected in expected_header.items():
        if manifest.get(name) != expected:
            raise ValueError(f"snapshot manifest mismatch for {name}")
    artifact_manifest = manifest.get("artifacts")
    if not isinstance(artifact_manifest, dict):
        raise ValueError("snapshot manifest has no artifacts object")

    loaded: dict[str, dict[str, list[object]]] = {}
    for symbol in symbols:
        _require_acquisition_time(deadline)
        loaded[symbol] = {}
        for kind in ("contract", "mark", "funding"):
            key = f"{symbol}:{kind}"
            metadata = artifact_manifest.get(key)
            if not isinstance(metadata, dict):
                raise ValueError(f"snapshot manifest is missing {key}")
            expected_name = _artifact_name(symbol, kind)
            if metadata.get("file") != expected_name:
                raise ValueError(f"snapshot filename mismatch for {key}")
            path = snapshot_dir / expected_name
            if not path.is_file():
                raise ValueError(f"snapshot artifact is missing for {key}")
            payload, value = _read_json_bytes(path)
            _require_acquisition_time(deadline)
            if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
                raise ValueError(f"snapshot hash mismatch for {key}")
            if not isinstance(value, list) or len(value) != int(metadata.get("rows", -1)):
                raise ValueError(f"snapshot row count mismatch for {key}")
            loaded[symbol][kind] = value
    if set(artifact_manifest) != {
        f"{symbol}:{kind}"
        for symbol in symbols
        for kind in ("contract", "mark", "funding")
    }:
        raise ValueError("snapshot manifest contains an unregistered artifact")
    _require_acquisition_time(deadline)
    return manifest, manifest_sha, loaded


def _contract_panel(
    snapshot: Mapping[str, Mapping[str, Sequence[object]]],
    registration: Mapping[str, object],
) -> dict[str, pd.DataFrame]:
    universe = registration["universe"]
    registered_data = registration["registeredData"]
    if not isinstance(universe, Mapping) or not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid panel settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    start = int(registered_data["startOpenTime"])
    end = int(registered_data["endOpenTime"])
    row_count = int(registered_data["rows"])
    expected_times = np.arange(
        start,
        end + feed.CONTRACT_INTERVAL_MS,
        feed.CONTRACT_INTERVAL_MS,
        dtype=np.int64,
    )
    panel = {}
    for symbol in symbols:
        rows = snapshot[symbol]["contract"]
        frame = pd.DataFrame(rows)
        required = {"openTime", "closeTime", "close"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{symbol} contract snapshot is missing {sorted(missing)}")
        frame = frame.loc[:, REGISTERED_PANEL_COLUMNS].copy()
        frame["openTime"] = pd.to_numeric(frame["openTime"], errors="raise").astype(
            np.int64
        )
        frame["closeTime"] = pd.to_numeric(
            frame["closeTime"], errors="raise"
        ).astype(np.int64)
        frame["close"] = pd.to_numeric(frame["close"], errors="raise").astype(float)
        frame = frame.sort_values("openTime").reset_index(drop=True)
        if len(frame) != row_count or not np.array_equal(
            frame["openTime"].to_numpy(dtype=np.int64), expected_times
        ):
            raise ValueError(f"{symbol} contract snapshot changed its registered grid")
        expected_close_times = expected_times + feed.CONTRACT_INTERVAL_MS - 1
        if not np.array_equal(
            frame["closeTime"].to_numpy(dtype=np.int64), expected_close_times
        ):
            raise ValueError(f"{symbol} contract closeTime grid is invalid")
        closes = frame["close"].to_numpy(dtype=float)
        if not np.isfinite(closes).all() or np.any(closes <= 0):
            raise ValueError(f"{symbol} contract closes must be finite and positive")
        panel[symbol] = frame
    C._validate_market_grid(panel, expected_times.tolist(), feed.CONTRACT_INTERVAL_MS)
    return panel


def _finite_float(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _resolved_settlements(
    snapshot: Mapping[str, Mapping[str, Sequence[object]]],
    registration: Mapping[str, object],
) -> tuple[list[F.FundingSettlement], pd.DataFrame, dict[str, object]]:
    universe = registration["universe"]
    registered_data = registration["registeredData"]
    if not isinstance(universe, Mapping) or not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid settlement settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    start = int(registered_data["startOpenTime"])
    end = int(registered_data["fundingWindowEndInclusive"])
    settlements: list[F.FundingSettlement] = []
    audit_rows = []
    fallback_count = 0
    maximum_observed_gap = 0
    event_counts: dict[str, int] = {}

    for symbol in symbols:
        mark_rows = snapshot[symbol]["mark"]
        mark_opens: dict[int, float] = {}
        for raw in mark_rows:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{symbol} mark-price row is not an object")
            open_time = int(raw.get("openTime", -1))
            mark_open = _finite_float(raw.get("open"), f"{symbol} mark open")
            if mark_open <= 0 or open_time in mark_opens:
                raise ValueError(f"{symbol} mark-price snapshot is invalid")
            mark_opens[open_time] = mark_open
        expected_mark_times = np.arange(
            start,
            end + 1,
            feed.MARK_PRICE_INTERVAL_MS,
            dtype=np.int64,
        )
        if list(mark_opens) != expected_mark_times.tolist():
            raise ValueError(f"{symbol} mark-price snapshot changed its hourly grid")

        funding_rows = snapshot[symbol]["funding"]
        if not funding_rows:
            raise ValueError(f"{symbol} funding-event snapshot is empty")
        previous_time: int | None = None
        seen: set[int] = set()
        event_times: list[int] = []
        for raw in funding_rows:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{symbol} funding event is not an object")
            event_symbol = str(raw.get("symbol", symbol))
            if event_symbol != symbol:
                raise ValueError(f"{symbol} funding artifact contains {event_symbol}")
            funding_time = int(raw.get("fundingTime", -1))
            if not start <= funding_time <= end:
                raise ValueError(f"{symbol} funding event is outside the registered window")
            if funding_time in seen or (
                previous_time is not None and funding_time <= previous_time
            ):
                raise ValueError(f"{symbol} funding events are duplicate or unordered")
            seen.add(funding_time)
            previous_time = funding_time
            event_times.append(funding_time)
            rate = _finite_float(raw.get("fundingRate"), f"{symbol} funding rate")
            raw_mark = raw.get("markPrice")
            if raw_mark is None or str(raw_mark).strip() == "":
                mark_open_time = (
                    funding_time // feed.MARK_PRICE_INTERVAL_MS
                ) * feed.MARK_PRICE_INTERVAL_MS
                if mark_open_time not in mark_opens:
                    raise ValueError(
                        f"{symbol} funding event has no causal mark-price fallback"
                    )
                resolved_mark = mark_opens[mark_open_time]
                mark_source = "containing_1h_mark_open"
                fallback_count += 1
            else:
                mark_open_time = None
                resolved_mark = _finite_float(raw_mark, f"{symbol} funding mark")
                mark_source = "funding_event_mark"
            if resolved_mark <= 0:
                raise ValueError(f"{symbol} resolved funding mark must be positive")
            settlements.append(
                F.FundingSettlement(
                    symbol=symbol,
                    funding_time=funding_time,
                    rate=rate,
                    resolved_mark_price=resolved_mark,
                )
            )
            audit_rows.append(
                {
                    "symbol": symbol,
                    "fundingTime": funding_time,
                    "fundingRate": rate,
                    "resolvedMarkPrice": resolved_mark,
                    "markSource": mark_source,
                    "markOpenTime": mark_open_time,
                }
            )
        gaps = np.diff(np.asarray(event_times, dtype=np.int64))
        symbol_maximum_gap = int(np.max(gaps)) if len(gaps) else 0
        maximum_allowed_gap = int(registered_data["maximumFundingGapMilliseconds"])
        first_close = start + feed.CONTRACT_INTERVAL_MS - 1
        if (
            event_times[0] > first_close
            or end - event_times[-1] > maximum_allowed_gap
            or symbol_maximum_gap > maximum_allowed_gap
        ):
            raise ValueError(f"{symbol} funding schedule exceeds the registered maximum gap")
        maximum_observed_gap = max(maximum_observed_gap, symbol_maximum_gap)
        event_counts[symbol] = len(event_times)

    audit = pd.DataFrame(audit_rows).sort_values(
        ["fundingTime", "symbol"], kind="stable"
    ).reset_index(drop=True)
    if len(audit) != len(settlements):
        raise AssertionError("settlement audit lost acquired funding events")
    coverage = {
        "acquiredEvents": len(audit),
        "resolvedEvents": len(settlements),
        "resolvedFraction": float(len(settlements) / len(audit)) if len(audit) else 0.0,
        "fallbackMarkEvents": fallback_count,
        "eventMarkEvents": len(audit) - fallback_count,
        "eventCountBySymbol": event_counts,
        "maximumObservedGapMilliseconds": maximum_observed_gap,
        "maximumAllowedGapMilliseconds": int(
            registered_data["maximumFundingGapMilliseconds"]
        ),
        "resolutionScope": "every event returned by bounded endpoint pagination",
    }
    return settlements, audit, coverage


def _close_frame(panel: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    first = next(iter(panel.values()))
    close_times = pd.Index(
        first["closeTime"].to_numpy(dtype=np.int64), name="closeTime"
    )
    columns = {}
    for symbol, frame in sorted(panel.items()):
        if not np.array_equal(
            frame["closeTime"].to_numpy(dtype=np.int64),
            close_times.to_numpy(dtype=np.int64),
        ):
            raise ValueError("contract closeTime axes are not identical")
        columns[symbol] = frame["close"].to_numpy(dtype=float)
    return pd.DataFrame(columns, index=close_times)


def _funding_rate_frame(
    close: pd.DataFrame, settlements: Sequence[F.FundingSettlement]
) -> pd.DataFrame:
    close_times = close.index.to_numpy(dtype=np.int64)
    by_symbol: dict[str, list[F.FundingSettlement]] = {
        str(symbol): [] for symbol in close.columns
    }
    for settlement in settlements:
        if settlement.symbol not in by_symbol:
            raise ValueError(f"unknown settlement symbol: {settlement.symbol}")
        by_symbol[settlement.symbol].append(settlement)

    columns = {}
    for symbol in close.columns:
        events = sorted(
            by_symbol[str(symbol)], key=lambda settlement: settlement.funding_time
        )
        values = np.empty(len(close_times), dtype=float)
        event_index = 0
        latest: float | None = None
        for row, close_time in enumerate(close_times):
            while (
                event_index < len(events)
                and events[event_index].funding_time <= close_time
            ):
                latest = events[event_index].rate
                event_index += 1
            if latest is None:
                raise ValueError(
                    f"{symbol} has no settled funding observation by closeTime={close_time}"
                )
            values[row] = latest
        columns[str(symbol)] = values
    result = pd.DataFrame(columns, index=close.index.copy())
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("aligned funding-rate frame contains non-finite values")
    return result


def _residual_momentum(
    close: pd.DataFrame,
    beta_lookback_bars: int,
    horizons: Sequence[int],
) -> dict[int, pd.DataFrame]:
    simple_returns = close.pct_change(fill_method=None)
    market_returns = simple_returns.mean(axis=1, skipna=False)
    market_mean = market_returns.rolling(
        beta_lookback_bars, min_periods=beta_lookback_bars
    ).mean()
    market_variance = market_returns.rolling(
        beta_lookback_bars, min_periods=beta_lookback_bars
    ).var(ddof=0)
    beta = {}
    for symbol in close.columns:
        asset = simple_returns[symbol]
        covariance = (
            (asset * market_returns)
            .rolling(beta_lookback_bars, min_periods=beta_lookback_bars)
            .mean()
            - asset.rolling(
                beta_lookback_bars, min_periods=beta_lookback_bars
            ).mean()
            * market_mean
        )
        beta[str(symbol)] = covariance / market_variance.where(
            market_variance > 1e-16
        )
    beta_frame = pd.DataFrame(beta, index=close.index)

    result = {}
    for horizon_hours in horizons:
        horizon_bars = horizon_hours * 3_600_000 // feed.CONTRACT_INTERVAL_MS
        asset_momentum = close.divide(close.shift(horizon_bars)) - 1.0
        market_momentum = (
            (1.0 + market_returns)
            .rolling(horizon_bars, min_periods=horizon_bars)
            .apply(np.prod, raw=True)
            - 1.0
        )
        result[horizon_hours] = asset_momentum.subtract(
            beta_frame.mul(market_momentum, axis=0)
        )
    return result


def _strategy_config(
    registration: Mapping[str, object],
    *,
    cost_multiplier: float = 1.0,
    additional_delay: int = 0,
) -> F.FundingCampaignConfig:
    strategy = registration["strategy"]
    if not isinstance(strategy, Mapping):
        raise ValueError("campaign registration has invalid strategy settings")
    return F.FundingCampaignConfig(
        interval_ms=feed.CONTRACT_INTERVAL_MS,
        funding_crowding_z=float(strategy["fundingCrowdingZ"]),
        top_n=int(strategy["topNPerSide"]),
        gross_exposure=float(strategy["grossExposure"]),
        cost_per_turnover=float(strategy["costBpsPerUnitTurnover"])
        / 10_000
        * cost_multiplier,
        signal_delay_bars=int(strategy["signalDelayBars"]) + additional_delay,
    )


def _trial_inputs(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], pd.DataFrame]:
    strategy = registration["strategy"]
    if not isinstance(strategy, Mapping):
        raise ValueError("campaign registration has invalid feature settings")
    close = _close_frame(panel)
    funding_rates = _funding_rate_frame(close, settlements)
    funding_z = F.causal_funding_zscore(
        funding_rates, int(strategy["fundingZLookbackBars"])
    )
    momentum = _residual_momentum(
        close,
        int(strategy["betaLookbackBars"]),
        F.HORIZON_HOURS,
    )
    return close, momentum, funding_z


def _trials_on_panel(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: F.FundingCampaignConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], tuple[F.FundingCampaignSpec, ...]]:
    close, momentum, funding_z = _trial_inputs(panel, settlements, registration)
    _, close_time_details, specs = F.run_trial_matrix(
        close, momentum, funding_z, settlements, config
    )
    details = {}
    for name, frame in close_time_details.items():
        adapted = frame.reset_index()
        adapted["openTime"] = (
            pd.to_numeric(adapted["closeTime"], errors="raise").astype(np.int64)
            - feed.CONTRACT_INTERVAL_MS
            + 1
        )
        adapted = adapted.drop(columns="closeTime").set_index("openTime")
        details[name] = adapted
    matrix = pd.DataFrame(
        {name: frame["net"] for name, frame in details.items()},
        index=next(iter(details.values())).index,
    )
    return matrix, details, specs


def _filter_settlements(
    settlements: Sequence[F.FundingSettlement], end_close_time: int
) -> list[F.FundingSettlement]:
    return [
        settlement
        for settlement in settlements
        if settlement.funding_time <= end_close_time
    ]


def _frame_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="NA",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _registered_input_manifest(
    output_dir: Path,
    panel: Mapping[str, pd.DataFrame],
    settlement_audit: pd.DataFrame,
    development_panel: Mapping[str, pd.DataFrame],
    development_settlement_audit: pd.DataFrame,
    snapshot_manifest_sha: str,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
) -> tuple[dict[str, object], str]:
    manifest_path = output_dir / "campaign-manifest.json"
    panel_path = output_dir / "registered-development-panel.csv"
    settlements_path = output_dir / "registered-development-settlements.csv"
    if _implementation_sha() != implementation_sha:
        raise ValueError("campaign implementation changed during this run")
    if _registration_sha() != registration_sha:
        raise ValueError("campaign registration changed during this run")
    panel_sha = C._panel_digest(panel, REGISTERED_PANEL_COLUMNS)

    long_development_panel = pd.concat(
        [
            frame.assign(symbol=symbol)
            for symbol, frame in sorted(development_panel.items())
        ],
        ignore_index=True,
    ).loc[:, ["symbol", *REGISTERED_PANEL_COLUMNS]]
    expected_header = {
        "campaign": CAMPAIGN_ID,
        "registrationVersion": REGISTRATION_VERSION,
        "registrationSha256": registration_sha,
        "implementationSha256": implementation_sha,
        "snapshotManifestSha256": snapshot_manifest_sha,
        "symbols": list(registration["universe"]["symbols"]),
        "trials": list(registration["trials"]),
        "registeredData": {
            **dict(registration["registeredData"]),
            "panelSha256": panel_sha,
            "settlementsSha256": _frame_digest(settlement_audit),
        },
    }
    if manifest_path.exists():
        manifest = _read_json_object(manifest_path)
        for name, expected in expected_header.items():
            if manifest.get(name) != expected:
                raise ValueError(
                    f"campaign manifest mismatch for {name}; use a new output directory"
                )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("campaign manifest has no artifacts object")
        if (
            artifacts.get("registeredDevelopmentPanel") != panel_path.name
            or artifacts.get("registeredDevelopmentSettlements")
            != settlements_path.name
        ):
            raise ValueError("campaign manifest registered input filenames changed")
        for path, key in (
            (panel_path, "registeredPanelSha256"),
            (settlements_path, "registeredSettlementsSha256"),
        ):
            if not path.is_file() or C._file_digest(path) != artifacts.get(key):
                raise ValueError(f"registered input artifact changed: {path.name}")
        return manifest, C._json_digest(manifest)

    if (output_dir / "summary.json").exists():
        raise ValueError(
            "output directory has campaign artifacts but no manifest; use a new directory"
        )
    C._write_csv_atomic(
        long_development_panel,
        panel_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    C._write_csv_atomic(
        development_settlement_audit,
        settlements_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="NA",
    )
    manifest = {
        **expected_header,
        "artifacts": {
            "registeredDevelopmentPanel": panel_path.name,
            "registeredPanelSha256": C._file_digest(panel_path),
            "registeredDevelopmentSettlements": settlements_path.name,
            "registeredSettlementsSha256": C._file_digest(settlements_path),
        },
    }
    C._write_json(manifest_path, manifest)
    return manifest, C._json_digest(manifest)


def _nested_sizes(registration: Mapping[str, object]) -> dict[str, int]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid validation settings")
    return {
        "initialTrain": int(validation["outerInitialTrain"]),
        "outerTest": int(validation["outerTestSize"]),
        "innerInitialTrain": int(validation["innerInitialTrain"]),
        "innerTest": int(validation["innerTestSize"]),
    }


def _lifetime_multiple_testing(
    diagnostic_matrix: pd.DataFrame,
    champion: str,
    registration: Mapping[str, object],
) -> dict[str, object]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid multiple-testing settings")
    lifetime_trials = int(validation["lifetimeTrialCount"])
    try:
        result = diagnostics.deflated_sharpe_ratio(
            diagnostic_matrix,
            selected_trial=champion,
            periods_per_year=365.0,
            independent_trials=1,
        ).to_dict()
        single_probability = float(result["probability"])
        adjusted_probability = max(
            0.0, 1.0 - min(1.0, lifetime_trials * (1.0 - single_probability))
        )
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": lifetime_trials,
            "singleTrialProbability": single_probability,
            "adjustedProbability": adjusted_probability,
            "underlyingProbabilisticSharpe": result,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": lifetime_trials,
            "adjustedProbability": 0.0,
            "error": str(error),
        }


def _paired_funding_comparison(
    matrix: pd.DataFrame,
    registration: Mapping[str, object],
    periods_per_year: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid paired-test settings")
    hypotheses = int(validation["pairedComparisonHypotheses"])
    if hypotheses != len(F.HORIZON_HOURS):
        raise ValueError("paired-comparison family size changed from registration")
    family_alpha = float(validation["pairedComparisonFamilyWiseAlpha"])
    comparison_alpha = family_alpha / hypotheses
    rows = []
    for horizon_hours in F.HORIZON_HOURS:
        base_trial = f"resmom_{horizon_hours}h_base"
        funding_trial = f"resmom_{horizon_hours}h_funding_only"
        difference = matrix[funding_trial] - matrix[base_trial]
        interval = C._bootstrap_ci(
            difference,
            periods_per_year,
            feed.CONTRACT_INTERVAL_MS,
            bootstrap_reps,
            bootstrap_seed,
            alpha=comparison_alpha,
        )
        rows.append(
            {
                "horizonHours": horizon_hours,
                "baseTrial": base_trial,
                "fundingTrial": funding_trial,
                "fundingMinusBaseMetrics": C._metrics(
                    difference, periods_per_year
                ),
                "fundingMinusBaseSimultaneousSharpeCi": C._ci_json(interval),
                "passed": math.isfinite(interval[0]) and interval[0] > 0,
            }
        )
    return {
        "method": "paired_funding_minus_base_block_bootstrap",
        "familyWiseAlpha": family_alpha,
        "comparisonAlpha": comparison_alpha,
        "comparisonConfidenceLevel": 1.0 - comparison_alpha,
        "successRule": "at least one matched horizon has a Bonferroni simultaneous Sharpe CI lower bound above zero",
        "passed": any(bool(row["passed"]) for row in rows),
        "horizons": rows,
        "registered": bool(
            registration["promotion"][
                "requirePairedFundingImprovementSharpeCiAboveZero"
            ]
        ),
    }


def _stress_campaign(
    label: str,
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    matrix_index: pd.Index,
    outer_folds: pd.DataFrame,
    periods_per_year: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], pd.DataFrame, tuple[float, float]]:
    if label == "cost1_5x":
        config = _strategy_config(registration, cost_multiplier=1.5)
    elif label == "cost2x":
        config = _strategy_config(registration, cost_multiplier=2.0)
    elif label == "additionalDelay1bar":
        config = _strategy_config(registration, additional_delay=1)
    else:
        raise ValueError(f"unknown registered stress: {label}")
    _, raw_details, _ = _trials_on_panel(
        panel, settlements, registration, config
    )
    matrix, details = C._reprice_details(
        raw_details, matrix_index, config.cost_per_turnover
    )
    frame, candidates = C._nested_input(matrix, details)
    oos = C._evaluate_outer_choices(
        frame, candidates, outer_folds, config.cost_per_turnover
    )
    interval = C._bootstrap_ci(
        oos["net"],
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        bootstrap_reps,
        bootstrap_seed,
    )
    return (
        {
            "nestedOuterOos": {
                "metrics": C._metrics(
                    oos["net"], periods_per_year, oos["active"]
                ),
                "sharpeBootstrap95": C._ci_json(interval),
            }
        },
        oos,
        interval,
    )


@contextmanager
def _output_lock(output_dir: Path, deadline: float) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".campaign.lock").open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("campaign output lock deadline exceeded")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    with _output_lock(output_dir, time.monotonic() + 3600.0):
        return _run_locked(args, output_dir)


def _run_locked(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, object]:
    registration, registration_sha = _registration_and_sha()
    implementation_sha = _implementation_sha()
    snapshot_dir = Path(args.snapshot_dir)
    C._assert_output_holdout_not_consumed(HOLDOUT_REGISTRY_DIR, output_dir)
    _snapshot_manifest, snapshot_manifest_sha, snapshot = _load_snapshot(
        snapshot_dir, registration, registration_sha, bool(args.acquire)
    )
    panel = _contract_panel(snapshot, registration)
    settlements, settlement_audit, settlement_coverage = _resolved_settlements(
        snapshot, registration
    )
    registered_data = registration["registeredData"]
    validation = registration["validation"]
    promotion = registration["promotion"]
    universe = registration["universe"]
    strategy = registration["strategy"]
    if not all(
        isinstance(value, Mapping)
        for value in (registered_data, validation, promotion, universe, strategy)
    ):
        raise ValueError("campaign registration has invalid runtime settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    development_cutoff = int(registered_data["developmentCutoffOpenTime"])
    holdout_start = int(registered_data["holdoutStartOpenTime"])
    holdout_end = int(registered_data["endOpenTime"])
    development_panel = C._truncate_panel(panel, development_cutoff)
    if any(
        len(frame) != int(registered_data["developmentRows"])
        for frame in development_panel.values()
    ):
        raise ValueError("development panel does not match the registered row count")
    development_end_close = (
        development_cutoff + feed.CONTRACT_INTERVAL_MS - 1
    )
    development_settlements = _filter_settlements(
        settlements, development_end_close
    )
    development_settlement_audit = settlement_audit[
        pd.to_numeric(settlement_audit["fundingTime"], errors="raise")
        <= development_end_close
    ].copy()
    manifest, campaign_manifest_sha = _registered_input_manifest(
        output_dir,
        panel,
        settlement_audit,
        development_panel,
        development_settlement_audit,
        snapshot_manifest_sha,
        registration,
        registration_sha,
        implementation_sha,
    )

    config = _strategy_config(registration)
    matrix_raw, details_raw, specs = _trials_on_panel(
        development_panel, development_settlements, registration, config
    )
    warmup = int(validation["featureWarmupRows"])
    if len(matrix_raw) <= warmup:
        raise ValueError("development window is too short after feature warmup")
    evaluation_index = matrix_raw.index[warmup:]
    if len(evaluation_index) != int(validation["developmentEvaluationRows"]):
        raise ValueError("development evaluation rows changed from registration")
    matrix, details = C._reprice_details(
        details_raw, evaluation_index, config.cost_per_turnover
    )
    periods_per_year = C._periods_per_year(feed.CONTRACT_INTERVAL_MS)
    sizes = _nested_sizes(registration)
    label_horizon = int(validation["labelHorizonBars"])
    nested_frame, candidates = C._nested_input(matrix, details)
    nested = C._run_nested_selector(
        nested_frame,
        candidates,
        sizes,
        label_horizon,
        config.cost_per_turnover,
    )
    outer_lengths = (
        nested.outer_folds["test_stop"] - nested.outer_folds["test_start"]
    ).to_numpy(dtype=int)
    if (
        len(outer_lengths) != int(validation["outerFoldCount"])
        or not np.all(outer_lengths == sizes["outerTest"])
    ):
        raise ValueError("outer folds changed from the registered complete-fold policy")
    champion, final_selection_scores, final_selection_folds = (
        C._rolling_select_candidate(
            nested_frame,
            candidates,
            sizes["innerInitialTrain"],
            sizes["innerTest"],
            label_horizon,
            config.cost_per_turnover,
        )
    )

    trial_metrics = {
        name: C._metrics(matrix[name], periods_per_year, details[name]["active"])
        for name in matrix.columns
    }
    nested_metrics = C._metrics(
        nested.oos["net"], periods_per_year, nested.oos["active"]
    )
    bootstrap_reps = int(validation["bootstrapReplications"])
    bootstrap_seed = int(validation["bootstrapSeed"])
    nested_ci = C._bootstrap_ci(
        nested.oos["net"],
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        bootstrap_reps,
        bootstrap_seed,
    )
    fold_metrics = C._fold_metrics(nested.oos, periods_per_year)
    worst_fold_return = min(
        float(metrics["totalReturn"]) for metrics in fold_metrics.values()
    )
    regime_report, _, labelled_nested_oos = C._regime_report(
        nested.oos,
        C._market_regime_labels(development_panel, feed.CONTRACT_INTERVAL_MS),
        periods_per_year,
        int(promotion["minimumRegimeObservations"]),
        float(promotion["maximumRegimeLoss"]),
    )
    selection_diagnostics, diagnostic_matrix, pbo_matrix = C._diagnostics(
        matrix,
        champion,
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        int(validation["pboSlices"]),
    )
    lifetime_multiple_testing = _lifetime_multiple_testing(
        diagnostic_matrix, champion, registration
    )
    paired_funding_comparison = _paired_funding_comparison(
        matrix,
        registration,
        periods_per_year,
        bootstrap_reps,
        bootstrap_seed,
    )

    stress_results: dict[str, object] = {}
    stress_paths: dict[str, pd.DataFrame] = {}
    stress_intervals: dict[str, tuple[float, float]] = {}
    for label in ("cost1_5x", "cost2x", "additionalDelay1bar"):
        result, path, interval = _stress_campaign(
            label,
            development_panel,
            development_settlements,
            registration,
            matrix.index,
            nested.outer_folds,
            periods_per_year,
            bootstrap_reps,
            bootstrap_seed,
        )
        stress_results[label] = result
        stress_paths[label] = path
        stress_intervals[label] = interval

    dsr_probability = selection_diagnostics.get("deflatedSharpe", {}).get(
        "probability", 0.0
    )
    pbo_probability = selection_diagnostics.get("pbo", {}).get(
        "probability", 1.0
    )
    active_fraction = float(nested_metrics["activeObservations"]) / float(
        nested_metrics["observations"]
    )
    gates = {
        "symbolCount": len(panel) >= int(promotion["minimumSymbols"]),
        "resolvedFunding": float(settlement_coverage["resolvedFraction"])
        >= float(promotion["minimumResolvedFundingFraction"]),
        "outerOosObservations": int(nested_metrics["observations"])
        >= int(promotion["minimumOuterOosObservations"]),
        "outerOosActiveFraction": active_fraction
        >= float(promotion["minimumActiveFraction"]),
        "outerOosSharpeCiAboveZero": math.isfinite(nested_ci[0])
        and nested_ci[0] > 0,
        "worstOuterFoldLoss": worst_fold_return
        >= -float(promotion["maximumWorstFoldLoss"]),
        "regimeLoss": bool(regime_report["lossCapPassed"]),
        "regimeCoverage": bool(regime_report["observationCoveragePassed"]),
        "currentCampaignDeflatedSharpe": float(dsr_probability)
        >= float(promotion["currentCampaignDeflatedSharpeProbabilityMinimum"]),
        "lifetimeBonferroniPsr": float(
            lifetime_multiple_testing.get("adjustedProbability", 0.0)
        )
        >= float(promotion["lifetimeBonferroniPsrProbabilityMinimum"]),
        "pbo": float(pbo_probability) <= float(promotion["maximumPbo"]),
        "pairedFundingImprovement": bool(paired_funding_comparison["passed"]),
        "cost2xOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["cost2x"][0]
        )
        and stress_intervals["cost2x"][0] > 0,
        "additionalDelayOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["additionalDelay1bar"][0]
        )
        and stress_intervals["additionalDelay1bar"][0] > 0,
    }
    ready_for_holdout = all(gates.values())

    holdout_window = C._holdout_window(
        symbols, feed.CONTRACT_INTERVAL, holdout_start, holdout_end
    )
    holdout_identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": manifest["registeredData"]["panelSha256"],
            "window": holdout_window,
        }
    )
    holdout_marker = HOLDOUT_REGISTRY_DIR / f"{holdout_identity}.json"
    output_holdout_record = output_dir / "final-holdout-opened.json"
    if args.open_final_holdout:
        C._assert_holdout_available(
            HOLDOUT_REGISTRY_DIR, holdout_window, output_holdout_record
        )
    final_holdout: dict[str, object] = {
        "status": "reserved",
        "identitySha256": holdout_identity,
        "openRequested": bool(args.open_final_holdout),
        "startOpenTime": holdout_start,
        "endOpenTime": holdout_end,
        "outcomeEndTimeExclusive": int(holdout_window["outcomeEndTimeExclusive"]),
        "rows": int(registered_data["holdoutReturnRows"]),
    }
    holdout_completion_record: dict[str, object] | None = None
    evaluated_holdout: pd.DataFrame | None = None
    if args.open_final_holdout and not ready_for_holdout:
        final_holdout["openBlockedBy"] = [
            name for name, passed in gates.items() if not passed
        ]
    elif args.open_final_holdout:
        holdout_returns_path = output_dir / "final-holdout-returns.csv"
        holdout_result_path = output_dir / "final-holdout-result.json"
        opening_record = {
            "registryVersion": C.HOLDOUT_REGISTRY_VERSION,
            "status": "opening",
            "campaign": CAMPAIGN_ID,
            "registrationSha256": registration_sha,
            "campaignManifestSha256": campaign_manifest_sha,
            "holdoutIdentitySha256": holdout_identity,
            "candidate": champion,
            "window": holdout_window,
            "artifacts": {
                "outputDirectory": str(output_dir.resolve()),
                "returns": str(holdout_returns_path.resolve()),
                "result": str(holdout_result_path.resolve()),
            },
        }
        C._reserve_holdout(
            HOLDOUT_REGISTRY_DIR,
            holdout_marker,
            holdout_window,
            output_holdout_record,
            opening_record,
        )
        _, full_details, _ = _trials_on_panel(
            panel, settlements, registration, config
        )
        full_matrix = pd.DataFrame(
            {name: detail["net"] for name, detail in full_details.items()}
        )
        full_frame, full_candidates = C._nested_input(full_matrix, full_details)
        holdout_frame = full_frame[
            (full_frame["openTime"] >= holdout_start)
            & (full_frame["openTime"] <= holdout_end)
        ]
        if len(holdout_frame) != int(registered_data["holdoutReturnRows"]):
            raise ValueError("final holdout return rows changed from registration")
        evaluated_holdout = C._evaluate_nested_candidate(
            full_candidates[champion], holdout_frame
        )
        evaluated_holdout.insert(
            0, "openTime", holdout_frame["openTime"].to_numpy()
        )
        evaluated_holdout = C._reprice_path(
            evaluated_holdout, config.cost_per_turnover
        )
        holdout_ci = C._bootstrap_ci(
            evaluated_holdout["net"],
            periods_per_year,
            feed.CONTRACT_INTERVAL_MS,
            bootstrap_reps,
            bootstrap_seed,
        )
        final_holdout = {
            "status": "pass"
            if math.isfinite(holdout_ci[0]) and holdout_ci[0] > 0
            else "fail",
            "openRequested": True,
            "identitySha256": holdout_identity,
            "startOpenTime": holdout_start,
            "endOpenTime": holdout_end,
            "outcomeEndTimeExclusive": int(
                holdout_window["outcomeEndTimeExclusive"]
            ),
            "rows": len(evaluated_holdout),
            "metrics": C._metrics(
                evaluated_holdout["net"],
                periods_per_year,
                evaluated_holdout["active"],
            ),
            "sharpeBootstrap95": C._ci_json(holdout_ci),
        }
        C._write_csv_atomic(evaluated_holdout, holdout_returns_path, index=False)
        returns_sha = C._file_digest(holdout_returns_path)
        final_holdout["evidence"] = {
            "returns": str(holdout_returns_path.resolve()),
            "returnsSha256": returns_sha,
        }
        holdout_result_record = {
            **opening_record,
            "status": "evaluated",
            "result": final_holdout,
            "artifacts": {
                **opening_record["artifacts"],
                "returnsSha256": returns_sha,
            },
        }
        C._write_json(holdout_result_path, holdout_result_record)
        holdout_completion_record = {
            **opening_record,
            "status": "completed",
            "result": final_holdout,
            "artifacts": {
                **holdout_result_record["artifacts"],
                "resultSha256": C._file_digest(holdout_result_path),
            },
        }

    summary = {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "status": C._campaign_status(ready_for_holdout, final_holdout),
        "symbols": symbols,
        "interval": feed.CONTRACT_INTERVAL,
        "configuration": {
            "strategy": dict(strategy),
            "nestedSizes": sizes,
            "labelHorizonBars": label_horizon,
            "innerFoldPolicy": validation["innerFoldPolicy"],
            "outerFoldPolicy": validation["outerFoldPolicy"],
        },
        "data": {
            "registeredRows": int(registered_data["rows"]),
            "developmentRows": int(registered_data["developmentRows"]),
            "featureWarmupRows": warmup,
            "trialReturnRows": len(matrix),
            "panelSha256": manifest["registeredData"]["panelSha256"],
            "snapshotManifestSha256": snapshot_manifest_sha,
            "settlements": settlement_coverage,
            "survivorshipLimitation": universe["survivorshipLimitation"],
        },
        "trials": [spec.to_dict() for spec in specs],
        "champion": champion,
        "finalSelection": {
            "rule": validation["selectionRule"],
            "scores": C._json_records(final_selection_scores),
            "folds": C._json_records(final_selection_folds),
        },
        "championDevelopmentMetrics": trial_metrics[champion],
        "nestedOuterOos": {
            "metrics": nested_metrics,
            "activeFraction": active_fraction,
            "sharpeBootstrap95": C._ci_json(nested_ci),
            "foldMetrics": fold_metrics,
            "regimes": regime_report,
        },
        "selectionDiagnostics": selection_diagnostics,
        "lifetimeMultipleTesting": lifetime_multiple_testing,
        "pairedFundingComparison": paired_funding_comparison,
        "stress": stress_results,
        "promotionGates": gates,
        "finalHoldout": final_holdout,
    }

    C._write_csv_atomic(matrix, output_dir / "trial-returns.csv", index_label="openTime")
    C._write_csv_atomic(
        diagnostic_matrix,
        output_dir / "diagnostic-trial-returns.csv",
        index_label="openTime",
    )
    C._write_csv_atomic(
        pbo_matrix,
        output_dir / "pbo-trial-returns.csv",
        index_label="openTime",
    )
    C._write_csv_atomic(labelled_nested_oos, output_dir / "nested-oos.csv", index=False)
    C._write_csv_atomic(nested.outer_folds, output_dir / "outer-folds.csv", index=False)
    C._write_csv_atomic(nested.inner_scores, output_dir / "inner-scores.csv", index=False)
    C._write_csv_atomic(
        final_selection_scores,
        output_dir / "final-selection-scores.csv",
        index=False,
    )
    C._write_csv_atomic(
        final_selection_folds,
        output_dir / "final-selection-folds.csv",
        index=False,
    )
    trial_paths = pd.concat(
        [
            frame.reindex(matrix.index).reset_index().assign(trial=name)
            for name, frame in details.items()
        ],
        ignore_index=True,
    )
    C._write_csv_atomic(trial_paths, output_dir / "trial-paths.csv", index=False)
    for label, path in stress_paths.items():
        C._write_csv_atomic(
            path, output_dir / f"stress-{label}-nested-oos.csv", index=False
        )
    C._write_json(
        output_dir / "trial-ledger.json",
        {
            "campaign": CAMPAIGN_ID,
            "trialCount": len(specs),
            "trials": [
                {
                    "specification": spec.to_dict(),
                    "metrics": trial_metrics[spec.trial_id],
                    "finalSelectionScore": C._finite_number(
                        final_selection_scores.loc[
                            final_selection_scores["candidate"] == spec.trial_id,
                            "score",
                        ].iloc[0]
                    ),
                }
                for spec in specs
            ],
        },
    )
    C._write_json(output_dir / "summary.json", summary)
    if holdout_completion_record is not None:
        C._write_json(output_holdout_record, holdout_completion_record)
        C._write_json(holdout_marker, holdout_completion_record)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"historical funding campaign failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
