#!/usr/bin/env python3
"""Verify frozen public Binance market-context responses and derive panel v2.

This module is deliberately offline: it never issues a request. A later public
collector may create the raw response bundle, but it is not its own admission
authority. This verifier re-hashes and re-decodes every registered response,
recomputes the complete eligible population and close-to-close peer returns,
and emits a deterministic panel plus a separate verification receipt.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


SOURCE_MANIFEST_SCHEMA_ID = "binance_usdm_market_context_source_manifest_v1"
SOURCE_MANIFEST_SCHEMA_VERSION = 1
PANEL_SCHEMA_ID = "binance_usdm_market_context_panel_v2"
PANEL_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_ID = "binance_usdm_market_context_verification_receipt_v1"
RECEIPT_SCHEMA_VERSION = 1
VERIFIER_PATH = "scripts/research/market_context_source.py"
SOURCE_ID = "binance-usdm-public-market-data"
SOURCE_LICENSE_RECORD = (
    "research-notes/market-prediction-2026-09-04/"
    "data-source-license-manifest.json#binance-usdm-public-market-data"
)
ELIGIBILITY_PREDICATE_ID = "usdm_trading_perpetual_quote_onboarded_v1"
INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
PANEL_COLUMNS = (
    "schemaId",
    "sourceId",
    "sourceManifestSha256",
    "quote",
    "intervalMs",
    "barOpenTime",
    "decisionTime",
    "universeEventTime",
    "universeAvailabilityTime",
    "populationCount",
    "memberOrdinal",
    "symbol",
    "quoteVolume",
    "eligible",
    "tickerOpenTime",
    "tickerCloseTime",
    "peerObserved",
    "peerEventTime",
    "peerAvailabilityTime",
    "peerSimpleReturn",
)
TIMESTAMP_MAX = 9_223_372_036_854_775_807
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Z0-9]+$")
PROVIDER_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]+$")
MANIFEST_KEYS = {
    "schemaId",
    "schemaVersion",
    "sourceId",
    "sourceLicenseRecord",
    "quote",
    "interval",
    "intervalMs",
    "barOpenTime",
    "barEndTime",
    "decisionTime",
    "maxClockSkewMs",
    "eligibilityPredicateId",
    "population",
    "startedAtMs",
    "completedAtMs",
    "codeCommit",
    "artifacts",
}
ARTIFACT_KEYS = {
    "logicalPath",
    "sha256",
    "bytes",
    "endpoint",
    "params",
    "requestStartedAtMs",
    "responseCompletedAtMs",
    "httpStatus",
    "usedWeight1m",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error


def _read_json_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    value = _decode_json(payload, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, payload


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} keys are incompatible")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _timestamp(value: object, label: str) -> int:
    result = _integer(value, label)
    if result > TIMESTAMP_MAX:
        raise ValueError(f"{label} exceeds Int64")
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{label} is not numeric")
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} is not numeric") from error
    if not decimal.is_finite() or decimal < 0 or (positive and decimal == 0):
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {relation}")
    converted = float(decimal)
    if not math.isfinite(converted) or converted < 0 or (positive and converted == 0):
        raise ValueError(f"{label} is not representable as a finite Double")
    return decimal


def _safe_raw_path(root: Path, logical_path: object) -> Path:
    logical = _string(logical_path, "artifact logicalPath")
    pure = PurePosixPath(logical)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.parts[0] != "raw"
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".json"
    ):
        raise ValueError("artifact logicalPath is unsafe")
    candidate = root.joinpath(*pure.parts)
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if resolved.parent != resolved_root / "raw" or candidate.is_symlink():
        raise ValueError("artifact path escapes the flat raw directory")
    if not resolved.is_file():
        raise ValueError("artifact path is not a regular file")
    return resolved


def _verify_artifact(
    root: Path,
    value: object,
    *,
    label: str,
    endpoint: str,
    params: dict[str, object],
) -> tuple[dict[str, object], object, bytes]:
    record = _exact_object(value, ARTIFACT_KEYS, label)
    if record["endpoint"] != endpoint or record["params"] != params:
        raise ValueError(f"{label} endpoint or parameters changed")
    if record["httpStatus"] != 200:
        raise ValueError(f"{label} was not an HTTP 200 response")
    _integer(record["usedWeight1m"], f"{label} usedWeight1m", minimum=1)
    started = _timestamp(record["requestStartedAtMs"], f"{label} requestStartedAtMs")
    completed = _timestamp(
        record["responseCompletedAtMs"], f"{label} responseCompletedAtMs"
    )
    if completed < started:
        raise ValueError(f"{label} request timestamps are reversed")
    path = _safe_raw_path(root, record["logicalPath"])
    payload = path.read_bytes()
    byte_count = _integer(record["bytes"], f"{label} bytes")
    if byte_count != len(payload):
        raise ValueError(f"{label} byte count changed")
    digest = record["sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{label} SHA-256 is malformed")
    if _sha256(payload) != digest:
        raise ValueError(f"{label} SHA-256 changed")
    return record, _decode_json(payload, label), payload


def _server_time(payload: object, label: str) -> int:
    if not isinstance(payload, dict) or "serverTime" not in payload:
        raise ValueError(f"{label} has no serverTime")
    return _timestamp(payload["serverTime"], f"{label} serverTime")


def _symbol_metadata(
    payload: object, quote: str, bar_end: int
) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("exchangeInfo has no symbols array")
    eligible: list[str] = []
    excluded: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["symbols"]):
        if not isinstance(raw, dict):
            raise ValueError(f"exchangeInfo symbol {index} is not an object")
        symbol = _string(raw.get("symbol"), f"exchangeInfo symbol {index}")
        if not PROVIDER_SYMBOL_PATTERN.fullmatch(symbol) or symbol in seen:
            raise ValueError("exchangeInfo symbols are malformed or duplicated")
        seen.add(symbol)
        contract_type = _string(raw.get("contractType"), f"{symbol} contractType")
        quote_asset = _string(raw.get("quoteAsset"), f"{symbol} quoteAsset")
        status = _string(raw.get("status"), f"{symbol} status")
        onboard_date = _timestamp(raw.get("onboardDate"), f"{symbol} onboardDate")
        reasons = []
        if contract_type != "PERPETUAL":
            reasons.append("contract_type")
        if quote_asset != quote:
            reasons.append("quote_asset")
        if status != "TRADING":
            reasons.append("status")
        if onboard_date > bar_end:
            reasons.append("onboard_after_bar")
        if reasons:
            excluded.append({"symbol": symbol, "reasons": reasons})
        else:
            if not symbol.endswith(quote) or len(symbol) <= len(quote):
                raise ValueError(f"eligible symbol {symbol} is outside quote scope")
            eligible.append(symbol)
    if not eligible:
        raise ValueError("eligible population is empty")
    return eligible, excluded


def _ticker_rows(payload: object) -> dict[str, dict[str, object]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("ticker24hr must contain a non-empty array")
    result: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"ticker24hr row {index} is not an object")
        symbol = _string(raw.get("symbol"), f"ticker24hr row {index} symbol")
        if not PROVIDER_SYMBOL_PATTERN.fullmatch(symbol) or symbol in result:
            raise ValueError("ticker24hr symbols are malformed or duplicated")
        result[symbol] = raw
    return result


def _ticker_evidence(raw: dict[str, object], symbol: str) -> tuple[Decimal, int, int]:
    volume = _finite_decimal(raw.get("quoteVolume"), f"{symbol} quoteVolume")
    open_time = _timestamp(raw.get("openTime"), f"{symbol} ticker openTime")
    close_time = _timestamp(raw.get("closeTime"), f"{symbol} ticker closeTime")
    if close_time < open_time:
        raise ValueError(f"{symbol} ticker timestamps are reversed")
    return volume, open_time, close_time


def _peer_return(payload: object, symbol: str, bar_open: int, interval_ms: int) -> Decimal:
    previous_open = bar_open - interval_ms
    if previous_open < 0 or not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"{symbol} peer kline response is not the exact two-bar window")
    expected_opens = [previous_open, bar_open]
    closes: list[Decimal] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, list) or len(raw) != 12:
            raise ValueError(f"{symbol} peer kline {index} is malformed")
        open_time = _timestamp(raw[0], f"{symbol} peer kline {index} openTime")
        close_time = _timestamp(raw[6], f"{symbol} peer kline {index} closeTime")
        if open_time != expected_opens[index] or close_time != open_time + interval_ms - 1:
            raise ValueError(f"{symbol} peer kline grid changed")
        closes.append(_finite_decimal(raw[4], f"{symbol} peer kline {index} close", positive=True))
    simple_return = closes[1] / closes[0] - Decimal(1)
    converted = float(simple_return)
    if simple_return <= -1 or not math.isfinite(converted):
        raise ValueError(f"{symbol} peer return is not a finite simple return")
    return simple_return


def _decimal_text(value: Decimal) -> str:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("derived decimal is not representable as a finite Double")
    return format(converted, ".17g")


def _panel_bytes(rows: list[list[object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(PANEL_COLUMNS)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def verify_and_derive(manifest_path: Path) -> tuple[bytes, dict[str, object]]:
    if manifest_path.is_symlink():
        raise ValueError("source manifest must be a regular non-symlink file")
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError("source manifest must be a regular non-symlink file")
    root = manifest_path.parent
    manifest, manifest_payload = _read_json_object(manifest_path, "source manifest")
    _exact_object(manifest, MANIFEST_KEYS, "source manifest")
    if (
        manifest["schemaId"] != SOURCE_MANIFEST_SCHEMA_ID
        or type(manifest["schemaVersion"]) is not int
        or manifest["schemaVersion"] != SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest["sourceId"] != SOURCE_ID
        or manifest["sourceLicenseRecord"] != SOURCE_LICENSE_RECORD
        or manifest["eligibilityPredicateId"] != ELIGIBILITY_PREDICATE_ID
    ):
        raise ValueError("source manifest identity is incompatible")
    quote = _string(manifest["quote"], "manifest quote")
    if not IDENTIFIER_PATTERN.fullmatch(quote):
        raise ValueError("manifest quote is not canonical")
    interval = _string(manifest["interval"], "manifest interval")
    if interval not in INTERVAL_MS:
        raise ValueError("manifest interval identity is incompatible")
    declared_interval_ms = _integer(manifest["intervalMs"], "manifest intervalMs")
    if declared_interval_ms != INTERVAL_MS[interval]:
        raise ValueError("manifest interval identity is incompatible")
    interval_ms = INTERVAL_MS[interval]
    bar_open = _timestamp(manifest["barOpenTime"], "manifest barOpenTime")
    bar_end = _timestamp(manifest["barEndTime"], "manifest barEndTime")
    if (
        bar_open < interval_ms
        or bar_open % interval_ms != 0
        or bar_open + interval_ms > TIMESTAMP_MAX
        or bar_end + interval_ms > TIMESTAMP_MAX
        or bar_end != bar_open + interval_ms
    ):
        raise ValueError("manifest bar window is incompatible")
    decision = _timestamp(manifest["decisionTime"], "manifest decisionTime")
    started_at = _timestamp(manifest["startedAtMs"], "manifest startedAtMs")
    completed_at = _timestamp(manifest["completedAtMs"], "manifest completedAtMs")
    max_skew = _integer(manifest["maxClockSkewMs"], "manifest maxClockSkewMs")
    if max_skew > 30_000:
        raise ValueError("manifest clock-skew bound is too large")
    commit = _string(manifest["codeCommit"], "manifest codeCommit")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("manifest codeCommit is malformed")
    artifacts = _exact_object(
        manifest["artifacts"],
        {"serverTimeBefore", "exchangeInfo", "ticker24hr", "peerKlines", "serverTimeAfter"},
        "manifest artifacts",
    )
    before_record, before_payload, _ = _verify_artifact(
        root,
        artifacts["serverTimeBefore"],
        label="serverTimeBefore",
        endpoint="/fapi/v1/time",
        params={},
    )
    exchange_record, exchange_payload, _ = _verify_artifact(
        root,
        artifacts["exchangeInfo"],
        label="exchangeInfo",
        endpoint="/fapi/v1/exchangeInfo",
        params={},
    )
    ticker_record, ticker_payload, _ = _verify_artifact(
        root,
        artifacts["ticker24hr"],
        label="ticker24hr",
        endpoint="/fapi/v1/ticker/24hr",
        params={},
    )
    population = _exact_object(
        manifest["population"],
        {"eligibleSymbols", "excluded"},
        "manifest population",
    )
    eligible, excluded = _symbol_metadata(exchange_payload, quote, bar_end)
    if population["eligibleSymbols"] != eligible or population["excluded"] != excluded:
        raise ValueError("manifest population disagrees with exchangeInfo")
    peer_values = artifacts["peerKlines"]
    if not isinstance(peer_values, list) or len(peer_values) != len(eligible):
        raise ValueError("peerKlines do not cover the complete eligible population")
    peer_records: list[dict[str, object]] = []
    peer_returns: dict[str, Decimal] = {}
    for index, (symbol, value) in enumerate(zip(eligible, peer_values)):
        label = f"peerKlines[{index}]"
        expected_params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": bar_open - interval_ms,
            "endTime": bar_end - 1,
            "limit": 2,
        }
        record, payload, _ = _verify_artifact(
            root, value, label=label, endpoint="/fapi/v1/klines", params=expected_params
        )
        peer_records.append(record)
        peer_returns[symbol] = _peer_return(payload, symbol, bar_open, interval_ms)
    after_record, after_payload, _ = _verify_artifact(
        root,
        artifacts["serverTimeAfter"],
        label="serverTimeAfter",
        endpoint="/fapi/v1/time",
        params={},
    )
    ordered_records = [before_record, exchange_record, ticker_record, *peer_records, after_record]
    if (
        started_at != ordered_records[0]["requestStartedAtMs"]
        or completed_at != ordered_records[-1]["responseCompletedAtMs"]
    ):
        raise ValueError("manifest acquisition bounds disagree with response records")
    for previous, current in zip(ordered_records, ordered_records[1:]):
        if previous["responseCompletedAtMs"] > current["requestStartedAtMs"]:
            raise ValueError("manifest response sequence overlaps or reverses")
    if not (completed_at <= decision < bar_end + interval_ms):
        raise ValueError("manifest decision is unavailable or outside the bar window")
    before_time = _server_time(before_payload, "serverTimeBefore")
    after_time = _server_time(after_payload, "serverTimeAfter")
    if before_time > after_time or bar_end > before_time:
        raise ValueError("provider clock does not prove that the peer bar was closed")
    for label, record, server_time in (
        ("serverTimeBefore", before_record, before_time),
        ("serverTimeAfter", after_record, after_time),
    ):
        if not (
            record["requestStartedAtMs"] - max_skew
            <= server_time
            <= record["responseCompletedAtMs"] + max_skew
        ):
            raise ValueError(f"{label} exceeds the declared clock-skew bound")
    registered_paths = {str(record["logicalPath"]) for record in ordered_records}
    if len(registered_paths) != len(ordered_records):
        raise ValueError("source manifest reuses a raw artifact path")
    raw_root = root / "raw"
    raw_entries = list(raw_root.iterdir())
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in raw_entries
        if path.is_file() and not path.is_symlink()
    }
    if registered_paths != actual_paths or any(
        path.is_symlink() or not path.is_file() for path in raw_entries
    ):
        raise ValueError("raw directory is incomplete or contains unregistered files")
    tickers = _ticker_rows(ticker_payload)
    ticker_evidence: dict[str, tuple[Decimal, int, int]] = {}
    for symbol in eligible:
        if symbol not in tickers:
            raise ValueError(f"ticker24hr is missing eligible symbol {symbol}")
        ticker_evidence[symbol] = _ticker_evidence(tickers[symbol], symbol)
    earliest_ticker_event = int(ticker_record["requestStartedAtMs"]) - max_skew
    latest_ticker_event = int(ticker_record["responseCompletedAtMs"]) + max_skew
    for symbol, (_, _, close_time) in ticker_evidence.items():
        if not earliest_ticker_event <= close_time <= latest_ticker_event:
            raise ValueError(
                f"{symbol} ticker event exceeds the declared collection window"
            )
    universe_event = max(close_time for _, _, close_time in ticker_evidence.values())
    if universe_event > int(ticker_record["responseCompletedAtMs"]) + max_skew:
        raise ValueError("ticker event exceeds the declared clock-skew bound")
    universe_availability = max(
        int(exchange_record["responseCompletedAtMs"]),
        int(ticker_record["responseCompletedAtMs"]),
        universe_event,
    )
    if universe_availability > decision:
        raise ValueError("complete universe was not available by the decision")
    manifest_sha = _sha256(manifest_payload)
    rows: list[list[object]] = []
    for ordinal, symbol in enumerate(eligible):
        volume, ticker_open, ticker_close = ticker_evidence[symbol]
        peer_availability = max(
            bar_end, int(peer_records[ordinal]["responseCompletedAtMs"])
        )
        if ticker_close > universe_availability or peer_availability > decision:
            raise ValueError("derived evidence is unavailable at the decision")
        rows.append(
            [
                PANEL_SCHEMA_ID,
                SOURCE_ID,
                manifest_sha,
                quote,
                interval_ms,
                bar_open,
                decision,
                universe_event,
                universe_availability,
                len(eligible),
                ordinal,
                symbol,
                _decimal_text(volume),
                1,
                ticker_open,
                ticker_close,
                1,
                bar_end,
                peer_availability,
                _decimal_text(peer_returns[symbol]),
            ]
        )
    panel = _panel_bytes(rows)
    receipt = {
        "schemaId": RECEIPT_SCHEMA_ID,
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "verifierPath": VERIFIER_PATH,
        "verifierSha256": _sha256(Path(__file__).resolve(strict=True).read_bytes()),
        "sourceManifestSha256": manifest_sha,
        "sourceManifestCodeCommit": commit,
        "panelSchemaId": PANEL_SCHEMA_ID,
        "panelSchemaVersion": PANEL_SCHEMA_VERSION,
        "panelSha256": _sha256(panel),
        "panelRows": len(rows),
        "outcomeUse": False,
        "modelUse": False,
        "orderUse": False,
        "liveAuthorizationUse": False,
    }
    return panel, receipt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verify", choices=["verify"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)
    if (args.panel_output is None) != (args.receipt_output is None):
        parser.error("--panel-output and --receipt-output must be provided together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        panel, receipt = verify_and_derive(args.manifest)
        if args.panel_output is not None:
            if args.panel_output.resolve() == args.receipt_output.resolve():
                raise ValueError("panel and receipt outputs must be different paths")
            manifest = args.manifest.resolve(strict=True)
            raw_root = manifest.parent / "raw"
            for output in (args.panel_output, args.receipt_output):
                resolved_output = output.resolve()
                if resolved_output == manifest or raw_root in resolved_output.parents:
                    raise ValueError("outputs cannot overwrite source-manifest or raw bytes")
            _write_bytes_atomic(args.panel_output, panel)
            _write_bytes_atomic(args.receipt_output, _json_bytes(receipt))
        print(json.dumps(receipt, allow_nan=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print(f"market-context verification failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
