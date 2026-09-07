#!/usr/bin/env python3
"""Fetch or verify a fixed-window, hash-bound public backtest dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_KIND = "backtest_data_manifest_v1"
GENERATOR_PATH = "scripts/fetch-backtest-data.py"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
CSV_FIELDS = [
    "openTimeMs",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeTimeMs",
    "quoteAssetVolume",
    "tradeCount",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
    "ignore",
]
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_INT64 = 2**63 - 1


class DatasetError(ValueError):
    """A deterministic data-contract violation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int(raw: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(raw, bool):
        raise DatasetError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{name} must be an integer") from exc
    if str(value) != str(raw).strip() and not isinstance(raw, int):
        raise DatasetError(f"{name} must use canonical integer syntax")
    if value < minimum or value > MAX_INT64:
        raise DatasetError(f"{name} is outside [{minimum}, {MAX_INT64}]")
    return value


def finite_decimal(raw: Any, name: str, *, strictly_positive: bool) -> str:
    if isinstance(raw, bool):
        raise DatasetError(f"{name} must be numeric")
    text = str(raw).strip()
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise DatasetError(f"{name} must be finite numeric evidence") from exc
    if not value.is_finite():
        raise DatasetError(f"{name} must be finite numeric evidence")
    if strictly_positive and value <= 0:
        raise DatasetError(f"{name} must be positive")
    if not strictly_positive and value < 0:
        raise DatasetError(f"{name} must be non-negative")
    return text


def normalize_rows(payload: Any, *, limit: int, end_time_ms: int) -> list[list[Any]]:
    if not isinstance(payload, list):
        raise DatasetError("Binance response must be a kline array")
    if len(payload) != limit:
        raise DatasetError(f"expected exactly {limit} klines, received {len(payload)}")

    rows: list[list[Any]] = []
    previous_close: int | None = None
    for index, raw_row in enumerate(payload):
        if not isinstance(raw_row, list) or len(raw_row) != len(CSV_FIELDS):
            raise DatasetError(f"kline {index} must contain exactly {len(CSV_FIELDS)} fields")
        open_time = parse_int(raw_row[0], f"kline {index} openTimeMs")
        close_time = parse_int(raw_row[6], f"kline {index} closeTimeMs")
        trade_count = parse_int(raw_row[8], f"kline {index} tradeCount")
        if close_time < open_time:
            raise DatasetError(f"kline {index} closes before it opens")
        if close_time > end_time_ms:
            raise DatasetError(f"kline {index} is not completed at endTimeMs")
        if previous_close is not None and open_time != previous_close + 1:
            raise DatasetError(f"kline {index} is not contiguous with its predecessor")

        normalized: list[Any] = [None] * len(CSV_FIELDS)
        normalized[0] = open_time
        normalized[1] = finite_decimal(raw_row[1], f"kline {index} open", strictly_positive=True)
        normalized[2] = finite_decimal(raw_row[2], f"kline {index} high", strictly_positive=True)
        normalized[3] = finite_decimal(raw_row[3], f"kline {index} low", strictly_positive=True)
        normalized[4] = finite_decimal(raw_row[4], f"kline {index} close", strictly_positive=True)
        normalized[5] = finite_decimal(raw_row[5], f"kline {index} volume", strictly_positive=False)
        normalized[6] = close_time
        normalized[7] = finite_decimal(raw_row[7], f"kline {index} quoteAssetVolume", strictly_positive=False)
        normalized[8] = trade_count
        normalized[9] = finite_decimal(raw_row[9], f"kline {index} takerBuyBaseVolume", strictly_positive=False)
        normalized[10] = finite_decimal(raw_row[10], f"kline {index} takerBuyQuoteVolume", strictly_positive=False)
        normalized[11] = str(raw_row[11])
        open_price = Decimal(normalized[1])
        high_price = Decimal(normalized[2])
        low_price = Decimal(normalized[3])
        close_price = Decimal(normalized[4])
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price) or low_price > high_price:
            raise DatasetError(f"kline {index} has incoherent OHLC values")
        rows.append(normalized)
        previous_close = close_time
    return rows


def encode_csv(rows: list[list[Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def canonical_rows_bytes(rows: list[list[Any]]) -> bytes:
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{5,24}", normalized):
        raise DatasetError(f"invalid Binance symbol: {symbol!r}")
    return normalized


def validate_interval(interval: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*(?:s|m|h|d|w|M)", interval):
        raise DatasetError(f"invalid Binance interval: {interval!r}")
    return interval


def validate_base_url(raw_url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(raw_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise DatasetError("base URL is malformed") from exc
    if parsed.username is not None or parsed.password is not None:
        raise DatasetError("base URL must not contain credentials")
    correct_path = parsed.path == "/api/v3/klines"
    loopback_http = (
        parsed.scheme == "http"
        and hostname in {"127.0.0.1", "::1", "localhost"}
        and correct_path
    )
    official_https = (
        parsed.scheme == "https"
        and hostname == "api.binance.com"
        and port is None
        and correct_path
    )
    if not official_https and not loopback_http:
        raise DatasetError("base URL must be the official Binance endpoint (loopback HTTP is test-only)")
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise DatasetError("base URL must be an absolute endpoint without query or fragment")
    return raw_url


def fetch_payload(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "trader-research-data/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DatasetError("Binance response exceeds the bounded payload size")
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def encode_manifest(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read JSON manifest {path}: {exc}") from exc


def manifest_entry(
    *,
    symbol: str,
    interval: str,
    limit: int,
    end_time_ms: int,
    base_url: str,
    rows: list[list[Any]],
    csv_name: str,
    csv_bytes: bytes,
) -> dict[str, Any]:
    return {
        "csvPath": csv_name,
        "csvSha256": sha256_bytes(csv_bytes),
        "endTimeMs": end_time_ms,
        "firstOpenTimeMs": rows[0][0],
        "interval": interval,
        "lastCloseTimeMs": rows[-1][6],
        "limit": limit,
        "request": {
            "endpoint": base_url,
            "parameters": {
                "endTime": end_time_ms,
                "interval": interval,
                "limit": limit,
                "symbol": symbol,
            },
        },
        "rows": len(rows),
        "sourceRowsSha256": sha256_bytes(canonical_rows_bytes(rows)),
        "symbol": symbol,
    }


def build_manifest(entries: list[dict[str, Any]], generator_sha256: str) -> dict[str, Any]:
    return {
        "entries": entries,
        "generator": {"path": GENERATOR_PATH, "sha256": generator_sha256},
        "kind": MANIFEST_KIND,
        "randomness": {"seed": None, "used": False},
        "schemaVersion": SCHEMA_VERSION,
        "source": expected_source_contract(),
    }


def expected_source_contract() -> dict[str, str]:
    return {
        "access": "public_read_only",
        "licenseName": "Binance API Terms of Use",
        "licenseUrl": "https://www.binance.com/en/terms",
        "provider": "Binance Spot REST API",
    }


def assert_expected_manifest(actual: dict[str, Any], expected_path: Path) -> None:
    expected = load_json(expected_path)
    if expected != actual:
        raise DatasetError(
            f"fetched evidence does not match expected manifest {expected_path}; no files written"
        )


def verify_manifest(path: Path, generator_path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise DatasetError("manifest must be a JSON object")
    if set(manifest) != {"entries", "generator", "kind", "randomness", "schemaVersion", "source"}:
        raise DatasetError("manifest fields do not match schema 1")
    if manifest.get("schemaVersion") != SCHEMA_VERSION or manifest.get("kind") != MANIFEST_KIND:
        raise DatasetError("unsupported backtest-data manifest schema")
    if manifest.get("randomness") != {"seed": None, "used": False}:
        raise DatasetError("this generator must record randomness as unused with a null seed")
    if manifest.get("source") != expected_source_contract():
        raise DatasetError("manifest source or license boundary is incompatible")
    generator = manifest.get("generator")
    if (
        not isinstance(generator, dict)
        or set(generator) != {"path", "sha256"}
        or generator.get("path") != GENERATOR_PATH
    ):
        raise DatasetError("manifest has an incompatible generator identity")
    if generator.get("sha256") != sha256_file(generator_path):
        raise DatasetError("manifest generator hash does not match current code")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DatasetError("manifest must contain at least one dataset entry")

    seen: set[tuple[str, str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise DatasetError("manifest dataset entry must be an object")
        if set(entry) != {
            "csvPath",
            "csvSha256",
            "endTimeMs",
            "firstOpenTimeMs",
            "interval",
            "lastCloseTimeMs",
            "limit",
            "request",
            "rows",
            "sourceRowsSha256",
            "symbol",
        }:
            raise DatasetError("manifest dataset entry fields do not match schema 1")
        symbol = validate_symbol(entry.get("symbol", ""))
        interval = validate_interval(entry.get("interval", ""))
        limit = parse_int(entry.get("limit"), "manifest limit", minimum=1)
        end_time_ms = parse_int(entry.get("endTimeMs"), "manifest endTimeMs")
        key = (symbol, interval, end_time_ms)
        if key in seen:
            raise DatasetError(f"duplicate manifest entry for {symbol} {interval} {end_time_ms}")
        seen.add(key)
        csv_name = entry.get("csvPath")
        if not isinstance(csv_name, str) or Path(csv_name).name != csv_name:
            raise DatasetError("manifest csvPath must be one safe relative filename")
        if csv_name != f"{symbol}-{interval}-{limit}.csv":
            raise DatasetError("manifest csvPath does not match its symbol, interval, and limit")
        request = entry.get("request")
        if not isinstance(request, dict) or set(request) != {"endpoint", "parameters"}:
            raise DatasetError("manifest entry lacks request provenance")
        endpoint = request.get("endpoint")
        if not isinstance(endpoint, str):
            raise DatasetError("manifest request endpoint is invalid")
        validate_base_url(endpoint)
        if request.get("parameters") != {
            "endTime": end_time_ms,
            "interval": interval,
            "limit": limit,
            "symbol": symbol,
        }:
            raise DatasetError("manifest request parameters do not match the dataset identity")
        csv_path = path.parent / csv_name
        try:
            csv_bytes = csv_path.read_bytes()
        except OSError as exc:
            raise DatasetError(f"cannot read manifested CSV {csv_path}: {exc}") from exc
        if sha256_bytes(csv_bytes) != entry.get("csvSha256"):
            raise DatasetError(f"CSV hash mismatch: {csv_name}")
        try:
            decoded_rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise DatasetError(f"invalid manifested CSV {csv_name}: {exc}") from exc
        if not decoded_rows or decoded_rows[0] != CSV_FIELDS:
            raise DatasetError(f"CSV schema mismatch: {csv_name}")
        rows = normalize_rows(decoded_rows[1:], limit=limit, end_time_ms=end_time_ms)
        if entry.get("rows") != len(rows) or len(rows) != limit:
            raise DatasetError(f"row-count metadata mismatch: {csv_name}")
        if entry.get("firstOpenTimeMs") != rows[0][0] or entry.get("lastCloseTimeMs") != rows[-1][6]:
            raise DatasetError(f"time-bound metadata mismatch: {csv_name}")
        if entry.get("sourceRowsSha256") != sha256_bytes(canonical_rows_bytes(rows)):
            raise DatasetError(f"source-row hash mismatch: {csv_name}")
    return manifest


def fetch(args: argparse.Namespace, generator_path: Path) -> int:
    end_time_ms = parse_int(args.end_time_ms, "endTimeMs")
    limit = parse_int(args.limit, "limit", minimum=1)
    if limit > 1000:
        raise DatasetError("limit cannot exceed Binance's 1000-row bound")
    interval = validate_interval(args.interval)
    base_url = validate_base_url(args.base_url)
    symbols = [validate_symbol(symbol) for symbol in args.symbols]
    if len(set(symbols)) != len(symbols):
        raise DatasetError("symbols must be unique")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.timeout_seconds > 60:
        raise DatasetError("timeout-seconds must be finite and inside (0, 60]")

    output_payloads: list[tuple[Path, bytes]] = []
    entries: list[dict[str, Any]] = []
    for symbol in symbols:
        parameters = {
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit),
            "endTime": str(end_time_ms),
        }
        url = f"{base_url}?{urllib.parse.urlencode(parameters)}"
        raw_payload = fetch_payload(url, timeout_seconds=args.timeout_seconds)
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{symbol}: Binance response is not JSON") from exc
        rows = normalize_rows(payload, limit=limit, end_time_ms=end_time_ms)
        csv_bytes = encode_csv(rows)
        csv_name = f"{symbol}-{interval}-{limit}.csv"
        output_payloads.append((args.data_dir / csv_name, csv_bytes))
        entries.append(
            manifest_entry(
                symbol=symbol,
                interval=interval,
                limit=limit,
                end_time_ms=end_time_ms,
                base_url=base_url,
                rows=rows,
                csv_name=csv_name,
                csv_bytes=csv_bytes,
            )
        )

    manifest = build_manifest(entries, sha256_file(generator_path))
    if args.expected_manifest is not None:
        assert_expected_manifest(manifest, args.expected_manifest)

    for path, payload in output_payloads:
        atomic_write(path, payload)
    manifest_path = args.data_dir / args.manifest_name
    manifest_bytes = encode_manifest(manifest)
    atomic_write(manifest_path, manifest_bytes)

    for entry in entries:
        print(
            "DATASET"
            f" symbol={entry['symbol']} rows={entry['rows']} end_time_ms={entry['endTimeMs']}"
            f" source_rows_sha256={entry['sourceRowsSha256']} csv_sha256={entry['csvSha256']}"
            " randomness=none"
        )
    print(f"MANIFEST path={manifest_path} sha256={sha256_bytes(manifest_bytes)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--end-time-ms")
    mode.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--limit", default="1000")
    parser.add_argument("--base-url", default="https://api.binance.com/api/v3/klines")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--manifest-name", default="backtest-data-manifest-v1.json")
    parser.add_argument("--expected-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generator_path = Path(__file__).resolve()
    try:
        if args.verify_manifest is not None:
            manifest = verify_manifest(args.verify_manifest, generator_path)
            print(
                f"VERIFIED path={args.verify_manifest} entries={len(manifest['entries'])}"
                f" sha256={sha256_file(args.verify_manifest)}"
            )
            return 0
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.json", args.manifest_name):
            raise DatasetError("manifest-name must be one safe JSON filename")
        return fetch(args, generator_path)
    except (DatasetError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
