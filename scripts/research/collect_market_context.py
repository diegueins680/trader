#!/usr/bin/env python3
"""Collect one prospective public Binance USDⓈ-M market-context bundle.

The collector is intentionally not an admission authority. It performs only
fixed public GET requests, preserves their exact bytes, and publishes a source
manifest after a complete bounded pass. The separate market_context_source.py
verifier must independently reconstruct any panel before research use.

No credential, model, outcome, order, deployment, or live-trading interface is
present in this module.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from market_context_source import (
    ELIGIBILITY_PREDICATE_ID,
    INTERVAL_MS,
    SOURCE_ID,
    SOURCE_LICENSE_RECORD,
    SOURCE_MANIFEST_SCHEMA_ID,
    SOURCE_MANIFEST_SCHEMA_VERSION,
)


BASE_URL = "https://fapi.binance.com"
COLLECTOR_PATH = "scripts/research/collect_market_context.py"
VERIFIER_PATH = "scripts/research/market_context_source.py"
COLLECTION_STATUS_SCHEMA_ID = "binance_usdm_market_context_collection_status_v1"
COLLECTION_STATUS_SCHEMA_VERSION = 1
USER_AGENT = "trader-market-context-collector/1"
REQUEST_WEIGHT_BUDGET = 900
REQUEST_WEIGHT_WINDOW_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_DEADLINE_SECONDS = 240
DEFAULT_MAX_CLOCK_SKEW_MS = 5_000
MAX_CLOCK_SKEW_MS = 30_000
COLLECTOR_INTERVALS = {"1h", "4h", "8h"}
REGISTERED_COLLECTION_START_MS = 1_800_489_600_000
REGISTERED_COLLECTION_END_MS = 1_832_022_000_000
MAX_RESPONSE_BYTES = {
    "/fapi/v1/time": 4_096,
    "/fapi/v1/exchangeInfo": 16 * 1024 * 1024,
    "/fapi/v1/ticker/24hr": 16 * 1024 * 1024,
    "/fapi/v1/klines": 64 * 1024,
}
REQUEST_WEIGHTS = {
    "/fapi/v1/time": 1,
    "/fapi/v1/exchangeInfo": 1,
    "/fapi/v1/ticker/24hr": 40,
    "/fapi/v1/klines": 1,
}
TIMESTAMP_MAX = 9_223_372_036_854_775_807
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Z0-9]+$")
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_]+$")
RATE_LIMIT_PATTERN = re.compile(r"\bbanned until (\d{10,})\b")
PROVENANCE_PATHS = (
    COLLECTOR_PATH,
    VERIFIER_PATH,
    "research-notes/market-prediction-2026-09-04/data-source-license-manifest.json",
)


class CollectionFailure(RuntimeError):
    """A sanitized, classified fail-closed collection error."""

    def __init__(
        self,
        failure_kind: str,
        message: str,
        *,
        endpoint: str | None = None,
        http_status: int | None = None,
        retry_after_seconds: int | None = None,
        banned_until_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.endpoint = endpoint
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        self.banned_until_ms = banned_until_ms


@dataclass(frozen=True)
class CapturedResponse:
    payload: bytes
    decoded: object
    request_started_at_ms: int
    response_completed_at_ms: int
    http_status: int
    used_weight_1m: int


class RequestWeightLimiter:
    """Conservatively combine process-local and shared-IP request weight."""

    def __init__(
        self,
        budget: int = REQUEST_WEIGHT_BUDGET,
        window_seconds: float = REQUEST_WEIGHT_WINDOW_SECONDS,
    ) -> None:
        if budget < 1 or window_seconds <= 0:
            raise ValueError("request-weight limits must be positive")
        self.budget = budget
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0

    def _expire(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            _, weight = self._events.popleft()
            self._used -= weight

    def wait(self, weight: int, deadline: float) -> None:
        if weight < 1 or weight > self.budget:
            raise ValueError("request weight does not fit the local budget")
        while True:
            now = time.monotonic()
            self._expire(now)
            if now >= deadline:
                raise CollectionFailure(
                    "deadline_exceeded", "collection deadline expired before request"
                )
            if self._used + weight <= self.budget:
                self._events.append((now, weight))
                self._used += weight
                return
            delay = self.window_seconds - (now - self._events[0][0])
            _bounded_sleep(max(0.01, delay), deadline)

    def observe_used_weight(self, used_weight: int) -> None:
        if type(used_weight) is not int or used_weight < 1:
            raise ValueError("observed request weight must be a positive integer")
        now = time.monotonic()
        self._expire(now)
        if used_weight > self._used:
            external_weight = used_weight - self._used
            self._events.append((now, external_weight))
            self._used += external_weight


RATE_LIMITER = RequestWeightLimiter()


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the fixed public host from redirecting requests elsewhere."""

    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


URL_OPENER = urllib.request.build_opener(NoRedirectHandler())


def _epoch_ms() -> int:
    value = int(time.time() * 1000)
    if value < 0 or value > TIMESTAMP_MAX:
        raise CollectionFailure("clock_invalid", "local epoch clock is outside Int64")
    return value


def _read_repository_commit() -> str | None:
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
    return commit if COMMIT_PATTERN.fullmatch(commit) else None


def _current_provenance_hashes() -> dict[str, str] | None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        return {
            path: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
            for path in PROVENANCE_PATHS
        }
    except OSError:
        return None


IMPORTED_CODE_COMMIT = _read_repository_commit()
IMPORTED_PROVENANCE_HASHES = _current_provenance_hashes()


def _repository_commit() -> str | None:
    return IMPORTED_CODE_COMMIT


def _provenance_tracked_clean(code_commit: str) -> bool:
    repository_root = Path(__file__).resolve().parents[2]
    if (
        code_commit != IMPORTED_CODE_COMMIT
        or IMPORTED_PROVENANCE_HASHES is None
        or _read_repository_commit() != code_commit
        or _current_provenance_hashes() != IMPORTED_PROVENANCE_HASHES
    ):
        return False
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", *PROVENANCE_PATHS],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", code_commit, "--", *PROVENANCE_PATHS],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return (
        tracked.returncode == 0
        and unchanged.returncode == 0
        and _read_repository_commit() == code_commit
        and _current_provenance_hashes() == IMPORTED_PROVENANCE_HASHES
    )


def _bounded_sleep(delay: float, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or delay >= remaining:
        raise CollectionFailure(
            "deadline_exceeded", "collection deadline expired while pacing requests"
        )
    time.sleep(max(0.0, delay))


def _require_publication_window(deadline: float, latest_safe_completion_ms: int) -> None:
    if time.monotonic() >= deadline:
        raise CollectionFailure(
            "deadline_exceeded", "collection deadline expired during publication"
        )
    if _epoch_ms() >= latest_safe_completion_ms:
        raise CollectionFailure(
            "deadline_exceeded", "publication crossed the causal decision window"
        )


def _require_publication_preconditions(
    deadline: float, latest_safe_completion_ms: int, code_commit: str
) -> None:
    if not _provenance_tracked_clean(code_commit):
        raise CollectionFailure(
            "provenance_invalid",
            "collector provenance changed during collection",
        )
    _require_publication_window(deadline, latest_safe_completion_ms)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(payload: bytes, endpoint: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CollectionFailure(
            "response_invalid", "public response is not strict UTF-8 JSON", endpoint=endpoint
        ) from error


def _positive_header(headers: object, name: str, endpoint: str) -> int:
    getter = getattr(headers, "get", None)
    raw = getter(name) if getter is not None else None
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise CollectionFailure(
            "response_invalid", f"public response lacks valid {name}", endpoint=endpoint
        ) from error
    if value < 1:
        raise CollectionFailure(
            "response_invalid", f"public response lacks positive {name}", endpoint=endpoint
        )
    return value


def _retry_after(headers: object) -> int | None:
    getter = getattr(headers, "get", None)
    raw = getter("Retry-After") if getter is not None else None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return math.ceil(value)


def _rate_limit_failure(
    endpoint: str,
    *,
    http_status: int | None,
    headers: object,
    payload: bytes,
) -> CollectionFailure:
    banned_until_ms = None
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError):
        decoded = None
    if isinstance(decoded, dict) and isinstance(decoded.get("msg"), str):
        match = RATE_LIMIT_PATTERN.search(decoded["msg"])
        if match is not None:
            banned_until_ms = int(match.group(1))
    retry_after = _retry_after(headers)
    return CollectionFailure(
        "provider_rate_limit",
        "Binance public request rate limited; collection circuit opened",
        endpoint=endpoint,
        http_status=http_status,
        retry_after_seconds=retry_after,
        banned_until_ms=banned_until_ms,
    )


def _read_bounded(response: object, endpoint: str) -> bytes:
    maximum = MAX_RESPONSE_BYTES[endpoint]
    headers = getattr(response, "headers", {})
    getter = getattr(headers, "get", None)
    declared = getter("Content-Length") if getter is not None else None
    declared_bytes = None
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except (TypeError, ValueError) as error:
            raise CollectionFailure(
                "response_invalid", "response Content-Length is invalid", endpoint=endpoint
            ) from error
        if declared_bytes < 0 or declared_bytes > maximum:
            raise CollectionFailure(
                "response_too_large", "public response exceeds byte limit", endpoint=endpoint
            )
    payload = response.read(maximum + 1)
    if not isinstance(payload, bytes) or len(payload) > maximum:
        raise CollectionFailure(
            "response_too_large", "public response exceeds byte limit", endpoint=endpoint
        )
    if declared_bytes is not None and declared_bytes != len(payload):
        raise CollectionFailure(
            "response_invalid",
            "response byte count disagrees with Content-Length",
            endpoint=endpoint,
        )
    return payload


def _run_before_deadline(
    action: Callable[[], object], deadline: float, endpoint: str
) -> object:
    """Bound a potentially trickled blocking transport operation end to end."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CollectionFailure(
            "deadline_exceeded",
            "collection deadline expired before transport",
            endpoint=endpoint,
        )
    result: list[tuple[bool, object]] = []
    completed = threading.Event()

    def run() -> None:
        try:
            result.append((True, action()))
        except BaseException as error:
            result.append((False, error))
        finally:
            completed.set()

    worker = threading.Thread(
        target=run,
        name="market-context-public-request",
        daemon=True,
    )
    worker.start()
    if not completed.wait(remaining) or time.monotonic() >= deadline:
        raise CollectionFailure(
            "deadline_exceeded",
            "collection deadline expired during transport",
            endpoint=endpoint,
        )
    succeeded, value = result[0]
    if not succeeded:
        if isinstance(value, BaseException):
            raise value
        raise CollectionFailure(
            "transport_error", "public endpoint transport failed", endpoint=endpoint
        )
    return value


def _request(
    endpoint: str,
    params: Mapping[str, object],
    *,
    deadline: float,
    limiter: RequestWeightLimiter,
) -> CapturedResponse:
    if endpoint not in REQUEST_WEIGHTS:
        raise ValueError("unsupported public market-context endpoint")
    limiter.wait(REQUEST_WEIGHTS[endpoint], deadline)
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{endpoint}" + (f"?{query}" if query else "")
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    request_started_at_ms = _epoch_ms()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CollectionFailure(
            "deadline_exceeded",
            "collection deadline expired before transport",
            endpoint=endpoint,
        )

    def receive() -> tuple[bytes, int, object]:
        try:
            with URL_OPENER.open(
                request, timeout=min(REQUEST_TIMEOUT_SECONDS, remaining)
            ) as response:
                status = getattr(response, "status", None)
                if type(status) is not int or status != 200:
                    raise CollectionFailure(
                        "http_error",
                        "public endpoint did not return HTTP 200",
                        endpoint=endpoint,
                    )
                headers = getattr(response, "headers", {})
                content_type_getter = getattr(headers, "get_content_type", None)
                if content_type_getter is not None:
                    content_type = content_type_getter()
                else:
                    getter = getattr(headers, "get", None)
                    raw_content_type = (
                        getter("Content-Type") if getter is not None else None
                    )
                    content_type = (
                        raw_content_type.split(";", 1)[0].strip().lower()
                        if isinstance(raw_content_type, str)
                        else None
                    )
                if content_type != "application/json":
                    raise CollectionFailure(
                        "response_invalid",
                        "public response Content-Type is not application/json",
                        endpoint=endpoint,
                    )
                payload = _read_bounded(response, endpoint)
                used_weight = _positive_header(
                    headers, "X-MBX-USED-WEIGHT-1M", endpoint
                )
                return payload, used_weight, headers
        except urllib.error.HTTPError as error:
            if error.code in {418, 429}:
                payload = error.read(65_537)[:65_536]
                raise _rate_limit_failure(
                    endpoint,
                    http_status=error.code,
                    headers=error.headers,
                    payload=payload,
                ) from error
            raise CollectionFailure(
                "http_error",
                f"public endpoint returned HTTP {error.code}",
                endpoint=endpoint,
            ) from error
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as error:
            raise CollectionFailure(
                "transport_error", "public endpoint transport failed", endpoint=endpoint
            ) from error

    transport_result = _run_before_deadline(receive, deadline, endpoint)
    if not isinstance(transport_result, tuple) or len(transport_result) != 3:
        raise CollectionFailure(
            "transport_error", "public endpoint transport failed", endpoint=endpoint
        )
    payload, used_weight, headers = transport_result
    if not isinstance(payload, bytes) or type(used_weight) is not int:
        raise CollectionFailure(
            "transport_error", "public endpoint transport failed", endpoint=endpoint
        )
    response_completed_at_ms = _epoch_ms()
    if response_completed_at_ms < request_started_at_ms:
        raise CollectionFailure(
            "clock_invalid", "local response clock moved backward", endpoint=endpoint
        )
    if time.monotonic() >= deadline:
        raise CollectionFailure(
            "deadline_exceeded", "collection deadline expired after response", endpoint=endpoint
        )
    decoded = _decode_json(payload, endpoint)
    if (
        isinstance(decoded, dict)
        and str(decoded.get("code")) == "-1003"
    ):
        raise _rate_limit_failure(
            endpoint,
            http_status=None,
            headers=headers,
            payload=payload,
        )
    limiter.observe_used_weight(used_weight)
    return CapturedResponse(
        payload=payload,
        decoded=decoded,
        request_started_at_ms=request_started_at_ms,
        response_completed_at_ms=response_completed_at_ms,
        http_status=200,
        used_weight_1m=used_weight,
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    before_replace: Callable[[], None] | None = None,
    after_replace: Callable[[], None] | None = None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        if after_replace is not None:
            after_replace()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(
    raw_dir: Path,
    logical_name: str,
    endpoint: str,
    params: Mapping[str, object],
    response: CapturedResponse,
) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", logical_name):
        raise CollectionFailure("collector_invalid", "raw logical filename is unsafe")
    logical_path = f"raw/{logical_name}"
    _write_bytes_atomic(raw_dir / logical_name, response.payload)
    return {
        "logicalPath": logical_path,
        "sha256": hashlib.sha256(response.payload).hexdigest(),
        "bytes": len(response.payload),
        "endpoint": endpoint,
        "params": dict(params),
        "requestStartedAtMs": response.request_started_at_ms,
        "responseCompletedAtMs": response.response_completed_at_ms,
        "httpStatus": response.http_status,
        "usedWeight1m": response.used_weight_1m,
    }


def _field_string(
    raw: object, field: str, label: str, endpoint: str
) -> str:
    if not isinstance(raw, dict):
        raise CollectionFailure(
            "response_invalid", f"{label} is not an object", endpoint=endpoint
        )
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise CollectionFailure(
            "response_invalid", f"{label} lacks {field}", endpoint=endpoint
        )
    return value


def _field_timestamp(
    raw: object, field: str, label: str, endpoint: str
) -> int:
    if not isinstance(raw, dict):
        raise CollectionFailure(
            "response_invalid", f"{label} is not an object", endpoint=endpoint
        )
    value = raw.get(field)
    if type(value) is not int or value < 0 or value > TIMESTAMP_MAX:
        raise CollectionFailure(
            "response_invalid", f"{label} has invalid {field}", endpoint=endpoint
        )
    return value


def _finite_decimal(
    raw: object, label: str, endpoint: str, *, positive: bool = False
) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise CollectionFailure(
            "response_invalid", f"{label} is not numeric", endpoint=endpoint
        )
    try:
        value = Decimal(str(raw))
    except InvalidOperation as error:
        raise CollectionFailure(
            "response_invalid", f"{label} is not numeric", endpoint=endpoint
        ) from error
    if not value.is_finite() or value < 0 or (positive and value == 0):
        raise CollectionFailure(
            "response_invalid", f"{label} is outside the finite domain", endpoint=endpoint
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise CollectionFailure(
            "response_invalid", f"{label} exceeds the output domain", endpoint=endpoint
        ) from error
    if not math.isfinite(converted):
        raise CollectionFailure(
            "response_invalid", f"{label} exceeds the output domain", endpoint=endpoint
        )
    return value


def _server_time(payload: object, endpoint: str) -> int:
    if not isinstance(payload, dict):
        raise CollectionFailure(
            "response_invalid", "server-time response is not an object", endpoint=endpoint
        )
    value = payload.get("serverTime")
    if type(value) is not int or value < 0 or value > TIMESTAMP_MAX:
        raise CollectionFailure(
            "response_invalid", "server-time response lacks a valid clock", endpoint=endpoint
        )
    return value


def _derive_population(
    exchange_info: object, quote: str, bar_end: int
) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(exchange_info, dict) or not isinstance(
        exchange_info.get("symbols"), list
    ):
        raise CollectionFailure(
            "response_invalid",
            "exchangeInfo lacks a symbols array",
            endpoint="/fapi/v1/exchangeInfo",
        )
    eligible: list[str] = []
    excluded: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(exchange_info["symbols"]):
        label = f"exchangeInfo symbol {index}"
        symbol = _field_string(raw, "symbol", label, "/fapi/v1/exchangeInfo")
        if not SYMBOL_PATTERN.fullmatch(symbol) or symbol in seen:
            raise CollectionFailure(
                "response_invalid",
                "exchangeInfo symbols are malformed or duplicated",
                endpoint="/fapi/v1/exchangeInfo",
            )
        seen.add(symbol)
        contract_type = _field_string(
            raw, "contractType", symbol, "/fapi/v1/exchangeInfo"
        )
        quote_asset = _field_string(
            raw, "quoteAsset", symbol, "/fapi/v1/exchangeInfo"
        )
        status = _field_string(raw, "status", symbol, "/fapi/v1/exchangeInfo")
        onboard_date = _field_timestamp(
            raw, "onboardDate", symbol, "/fapi/v1/exchangeInfo"
        )
        reasons: list[str] = []
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
                raise CollectionFailure(
                    "response_invalid",
                    "eligible exchangeInfo symbol is outside quote scope",
                    endpoint="/fapi/v1/exchangeInfo",
                )
            eligible.append(symbol)
    if not eligible:
        raise CollectionFailure(
            "response_invalid",
            "exchangeInfo produced an empty eligible population",
            endpoint="/fapi/v1/exchangeInfo",
        )
    return eligible, excluded


def _validate_tickers(
    payload: object,
    eligible: list[str],
    *,
    earliest_event_time: int,
    latest_event_time: int,
) -> int:
    if not isinstance(payload, list) or not payload:
        raise CollectionFailure(
            "response_invalid",
            "ticker response is not a non-empty array",
            endpoint="/fapi/v1/ticker/24hr",
        )
    seen: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(payload):
        symbol = _field_string(
            raw, "symbol", f"ticker row {index}", "/fapi/v1/ticker/24hr"
        )
        if not SYMBOL_PATTERN.fullmatch(symbol) or symbol in seen:
            raise CollectionFailure(
                "response_invalid",
                "ticker symbols are malformed or duplicated",
                endpoint="/fapi/v1/ticker/24hr",
            )
        seen[symbol] = raw
    missing = [symbol for symbol in eligible if symbol not in seen]
    if missing:
        raise CollectionFailure(
            "response_incomplete",
            "ticker response omits an eligible symbol",
            endpoint="/fapi/v1/ticker/24hr",
        )
    close_times: list[int] = []
    for symbol in eligible:
        raw = seen[symbol]
        _finite_decimal(
            raw.get("quoteVolume"),
            f"{symbol} quoteVolume",
            "/fapi/v1/ticker/24hr",
        )
        open_time = _field_timestamp(
            raw, "openTime", symbol, "/fapi/v1/ticker/24hr"
        )
        close_time = _field_timestamp(
            raw, "closeTime", symbol, "/fapi/v1/ticker/24hr"
        )
        if close_time < open_time:
            raise CollectionFailure(
                "response_invalid",
                f"ticker clocks for {symbol} are reversed",
                endpoint="/fapi/v1/ticker/24hr",
            )
        if not earliest_event_time <= close_time <= latest_event_time:
            raise CollectionFailure(
                "clock_invalid",
                f"ticker event for {symbol} is outside the collection window",
                endpoint="/fapi/v1/ticker/24hr",
            )
        close_times.append(close_time)
    return max(close_times)


def _validate_klines(
    payload: object, symbol: str, bar_open: int, interval_ms: int
) -> None:
    expected_opens = [bar_open - interval_ms, bar_open]
    if not isinstance(payload, list) or len(payload) != 2:
        raise CollectionFailure(
            "response_incomplete",
            f"kline response for {symbol} is not the exact two-bar window",
            endpoint="/fapi/v1/klines",
        )
    for index, row in enumerate(payload):
        if not isinstance(row, list) or len(row) != 12:
            raise CollectionFailure(
                "response_invalid",
                f"kline response for {symbol} has incompatible tuple shape",
                endpoint="/fapi/v1/klines",
            )
        if type(row[0]) is not int or type(row[6]) is not int:
            raise CollectionFailure(
                "response_invalid",
                f"kline response for {symbol} has invalid clocks",
                endpoint="/fapi/v1/klines",
            )
        if row[0] != expected_opens[index] or row[6] != row[0] + interval_ms - 1:
            raise CollectionFailure(
                "response_invalid",
                f"kline response for {symbol} is off the requested grid",
                endpoint="/fapi/v1/klines",
            )
        _finite_decimal(
            row[4],
            f"kline close for {symbol}",
            "/fapi/v1/klines",
            positive=True,
        )


def _prepare_output(output_dir: Path) -> tuple[Path, Path, Path]:
    if output_dir.exists() or output_dir.is_symlink():
        raise CollectionFailure(
            "output_exists", "output directory already exists; refusing overwrite"
        )
    parent = output_dir.parent.resolve(strict=True)
    if not parent.is_dir():
        raise CollectionFailure("output_invalid", "output parent is not a directory")
    resolved = parent / output_dir.name
    if output_dir.name in {"", ".", ".."}:
        raise CollectionFailure("output_invalid", "output directory name is unsafe")
    resolved.mkdir(mode=0o700)
    _fsync_directory(parent)
    raw_dir = resolved / "raw"
    raw_dir.mkdir(mode=0o700)
    _fsync_directory(resolved)
    return resolved, raw_dir, resolved / "collection-status.json"


def _file_sha256(path: str) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    return hashlib.sha256((repository_root / path).read_bytes()).hexdigest()


def _status_base(started_at_ms: int, code_commit: str) -> dict[str, object]:
    return {
        "schemaId": COLLECTION_STATUS_SCHEMA_ID,
        "schemaVersion": COLLECTION_STATUS_SCHEMA_VERSION,
        "sourceId": SOURCE_ID,
        "startedAtMs": started_at_ms,
        "codeCommit": code_commit,
        "provenanceTrackedClean": True,
        "collectorPath": COLLECTOR_PATH,
        "collectorSha256": _file_sha256(COLLECTOR_PATH),
        "verifierPath": VERIFIER_PATH,
        "verifierSha256": _file_sha256(VERIFIER_PATH),
        "runtime": {"python": platform.python_version()},
        "outcomeUse": False,
        "modelUse": False,
        "orderUse": False,
        "liveAuthorizationUse": False,
    }


def _write_status(
    path: Path,
    value: Mapping[str, object],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    _write_bytes_atomic(
        path, _json_bytes(dict(value)), before_replace=before_replace
    )


def _failure_status(
    base: Mapping[str, object],
    error: Exception,
    completed_paths: list[str],
) -> dict[str, object]:
    if isinstance(error, CollectionFailure):
        failure_kind = error.failure_kind
        message = str(error)
        endpoint = error.endpoint
        http_status = error.http_status
        retry_after = error.retry_after_seconds
        banned_until = error.banned_until_ms
    else:
        failure_kind = "collector_internal_failure"
        message = f"unexpected {type(error).__name__}; inspect local logs"
        endpoint = None
        http_status = None
        retry_after = None
        banned_until = None
    return {
        **base,
        "state": "partial_failure",
        "completedAtMs": _epoch_ms(),
        "completedArtifactPaths": completed_paths,
        "failureKind": failure_kind,
        "failureMessage": message,
        "failedEndpoint": endpoint,
        "httpStatus": http_status,
        "retryAfterSeconds": retry_after,
        "bannedUntilMs": banned_until,
        "sourceManifestPublished": False,
    }


def _cleanup_failure_status(
    base: Mapping[str, object],
    original_error: Exception,
    completed_paths: list[str],
) -> dict[str, object]:
    status = _failure_status(base, original_error, completed_paths)
    original_failure_kind = status["failureKind"]
    return {
        **status,
        "state": "cleanup_failure",
        "failureKind": "manifest_cleanup_failed",
        "failureMessage": (
            "source-manifest cleanup could not be confirmed; bundle state is indeterminate"
        ),
        "originalFailureKind": original_failure_kind,
        "sourceManifestPublished": None,
        "sourceManifestState": "indeterminate_after_cleanup_failure",
        "manifestCleanupRequired": True,
    }


def _remove_manifest(manifest_path: Path, output_dir: Path) -> None:
    manifest_path.unlink(missing_ok=True)
    _fsync_directory(output_dir)


def _validate_inputs(
    quote: str,
    interval: str,
    bar_open: int,
    max_clock_skew_ms: int,
    code_commit: str,
    deadline_seconds: int,
) -> tuple[int, int]:
    if not IDENTIFIER_PATTERN.fullmatch(quote):
        raise CollectionFailure("input_invalid", "quote must be canonical uppercase ASCII")
    if interval not in COLLECTOR_INTERVALS:
        raise CollectionFailure("input_invalid", "interval is outside the registration")
    interval_ms = INTERVAL_MS[interval]
    if (
        type(bar_open) is not int
        or bar_open < interval_ms
        or bar_open % interval_ms != 0
        or bar_open + 2 * interval_ms > TIMESTAMP_MAX
    ):
        raise CollectionFailure("input_invalid", "bar-open-time is not an aligned safe window")
    if (
        type(max_clock_skew_ms) is not int
        or max_clock_skew_ms < 0
        or max_clock_skew_ms > MAX_CLOCK_SKEW_MS
    ):
        raise CollectionFailure("input_invalid", "max-clock-skew-ms is outside the safe bound")
    if not COMMIT_PATTERN.fullmatch(code_commit):
        raise CollectionFailure("input_invalid", "code-commit must be a lowercase 40-hex commit")
    if type(deadline_seconds) is not int or deadline_seconds < 1:
        raise CollectionFailure("input_invalid", "deadline-seconds must be a positive integer")
    if deadline_seconds * 1000 + max_clock_skew_ms >= interval_ms:
        raise CollectionFailure(
            "input_invalid", "deadline and clock-skew budget must fit within one interval"
        )
    if not REGISTERED_COLLECTION_START_MS <= bar_open <= REGISTERED_COLLECTION_END_MS:
        raise CollectionFailure(
            "input_invalid", "bar-open-time is outside the registered dataset window"
        )
    return interval_ms, bar_open + interval_ms


def collect_bundle(
    output_dir: Path,
    *,
    quote: str,
    interval: str,
    bar_open: int,
    max_clock_skew_ms: int,
    deadline_seconds: int,
    limiter: RequestWeightLimiter | None = None,
) -> Path:
    code_commit = _repository_commit()
    if code_commit is None or not _provenance_tracked_clean(code_commit):
        raise CollectionFailure(
            "provenance_invalid",
            "collector, verifier, and source-license files must match a Git commit",
        )
    interval_ms, bar_end = _validate_inputs(
        quote,
        interval,
        bar_open,
        max_clock_skew_ms,
        code_commit,
        deadline_seconds,
    )
    started = _epoch_ms()
    next_bar_end = bar_end + interval_ms
    latest_safe_completion = next_bar_end - max_clock_skew_ms
    if started < bar_end or started >= latest_safe_completion:
        raise CollectionFailure(
            "input_invalid", "collection did not start inside the causal decision window"
        )
    deadline_budget = min(deadline_seconds, (latest_safe_completion - started) / 1000)
    if deadline_budget <= 0:
        raise CollectionFailure("deadline_exceeded", "no causal collection window remains")
    deadline = time.monotonic() + deadline_budget
    output, raw_dir, status_path = _prepare_output(output_dir)
    status_base = _status_base(started, code_commit)
    _write_status(
        status_path,
        {
            **status_base,
            "state": "collecting",
            "completedArtifactPaths": [],
            "sourceManifestPublished": False,
        },
    )
    completed_paths: list[str] = []
    active_limiter = limiter if limiter is not None else RATE_LIMITER
    previous_completion: int | None = None
    manifest_path = output / "source-manifest.json"
    manifest_written = False

    def mark_manifest_written() -> None:
        nonlocal manifest_written
        manifest_written = True

    def capture(
        logical_name: str, endpoint: str, params: Mapping[str, object]
    ) -> tuple[dict[str, object], object]:
        nonlocal previous_completion
        response = _request(
            endpoint, params, deadline=deadline, limiter=active_limiter
        )
        if (
            previous_completion is not None
            and response.request_started_at_ms < previous_completion
        ):
            raise CollectionFailure(
                "clock_invalid", "local request sequence moved backward", endpoint=endpoint
            )
        record = _artifact(raw_dir, logical_name, endpoint, params, response)
        completed_paths.append(str(record["logicalPath"]))
        previous_completion = response.response_completed_at_ms
        return record, response.decoded

    try:
        before_record, before_payload = capture(
            "server-time-before.json", "/fapi/v1/time", {}
        )
        before_time = _server_time(before_payload, "/fapi/v1/time")
        exchange_record, exchange_info = capture(
            "exchange-info.json", "/fapi/v1/exchangeInfo", {}
        )
        eligible, excluded = _derive_population(exchange_info, quote, bar_end)
        ticker_record, tickers = capture(
            "ticker-24hr.json", "/fapi/v1/ticker/24hr", {}
        )
        ticker_event_time = _validate_tickers(
            tickers,
            eligible,
            earliest_event_time=(
                int(ticker_record["requestStartedAtMs"]) - max_clock_skew_ms
            ),
            latest_event_time=(
                int(ticker_record["responseCompletedAtMs"]) + max_clock_skew_ms
            ),
        )
        peer_records: list[dict[str, object]] = []
        width = max(4, len(str(len(eligible) - 1)))
        for ordinal, symbol in enumerate(eligible):
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": bar_open - interval_ms,
                "endTime": bar_end - 1,
                "limit": 2,
            }
            record, klines = capture(
                f"peer-{ordinal:0{width}d}-{symbol}-klines.json",
                "/fapi/v1/klines",
                params,
            )
            _validate_klines(klines, symbol, bar_open, interval_ms)
            peer_records.append(record)
        after_record, after_payload = capture(
            "server-time-after.json", "/fapi/v1/time", {}
        )
        after_time = _server_time(after_payload, "/fapi/v1/time")
        if before_time > after_time or bar_end > before_time:
            raise CollectionFailure(
                "clock_invalid", "provider clock does not prove a completed feature bar"
            )
        for label, record, server_time in (
            ("serverTimeBefore", before_record, before_time),
            ("serverTimeAfter", after_record, after_time),
        ):
            if not (
                int(record["requestStartedAtMs"]) - max_clock_skew_ms
                <= server_time
                <= int(record["responseCompletedAtMs"]) + max_clock_skew_ms
            ):
                raise CollectionFailure(
                    "clock_invalid", f"{label} exceeds the clock-skew bound"
                )
        if (
            ticker_event_time
            > int(ticker_record["responseCompletedAtMs"]) + max_clock_skew_ms
        ):
            raise CollectionFailure(
                "clock_invalid", "ticker event exceeds the clock-skew bound"
            )
        conservative_decision = (
            int(after_record["responseCompletedAtMs"]) + max_clock_skew_ms
        )
        if conservative_decision > TIMESTAMP_MAX or conservative_decision >= next_bar_end:
            raise CollectionFailure(
                "deadline_exceeded", "clock-skew wait would cross the decision window"
            )
        while _epoch_ms() < conservative_decision:
            remaining_ms = conservative_decision - _epoch_ms()
            _bounded_sleep(min(0.1, remaining_ms / 1000), deadline)
        decision_time = _epoch_ms()
        if decision_time >= next_bar_end:
            raise CollectionFailure(
                "deadline_exceeded", "decision timestamp crossed the next bar boundary"
            )
        if ticker_event_time > decision_time:
            raise CollectionFailure(
                "clock_invalid", "ticker evidence is later than the decision"
            )
        manifest = {
            "schemaId": SOURCE_MANIFEST_SCHEMA_ID,
            "schemaVersion": SOURCE_MANIFEST_SCHEMA_VERSION,
            "sourceId": SOURCE_ID,
            "sourceLicenseRecord": SOURCE_LICENSE_RECORD,
            "quote": quote,
            "interval": interval,
            "intervalMs": interval_ms,
            "barOpenTime": bar_open,
            "barEndTime": bar_end,
            "decisionTime": decision_time,
            "maxClockSkewMs": max_clock_skew_ms,
            "eligibilityPredicateId": ELIGIBILITY_PREDICATE_ID,
            "population": {
                "eligibleSymbols": eligible,
                "excluded": excluded,
            },
            "startedAtMs": before_record["requestStartedAtMs"],
            "completedAtMs": after_record["responseCompletedAtMs"],
            "codeCommit": code_commit,
            "artifacts": {
                "serverTimeBefore": before_record,
                "exchangeInfo": exchange_record,
                "ticker24hr": ticker_record,
                "peerKlines": peer_records,
                "serverTimeAfter": after_record,
            },
        }
        manifest_payload = _json_bytes(manifest)
        _fsync_directory(raw_dir)
        _write_bytes_atomic(
            manifest_path,
            manifest_payload,
            before_replace=lambda: _require_publication_preconditions(
                deadline, latest_safe_completion, code_commit
            ),
            after_replace=mark_manifest_written,
        )
        _write_status(
            status_path,
            {
                **status_base,
                "state": "complete_unverified",
                "completedAtMs": _epoch_ms(),
                "completedArtifactPaths": completed_paths,
                "sourceManifestPublished": True,
                "sourceManifestSha256": hashlib.sha256(manifest_payload).hexdigest(),
                "eligiblePopulationCount": len(eligible),
            },
            before_replace=lambda: _require_publication_preconditions(
                deadline, latest_safe_completion, code_commit
            ),
        )
        return manifest_path
    except Exception as error:
        cleanup_error: OSError | None = None
        if manifest_written:
            try:
                _remove_manifest(manifest_path, output)
            except OSError as removal_error:
                cleanup_error = removal_error
        if cleanup_error is not None:
            try:
                _write_status(
                    status_path,
                    _cleanup_failure_status(status_base, error, completed_paths),
                )
            except Exception:
                pass
            raise CollectionFailure(
                "manifest_cleanup_failed",
                "source-manifest cleanup failed; bundle state is indeterminate",
            ) from cleanup_error
        try:
            _write_status(
                status_path, _failure_status(status_base, error, completed_paths)
            )
        except Exception:
            pass
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collect", choices=["collect"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--interval", required=True, choices=sorted(COLLECTOR_INTERVALS))
    parser.add_argument("--bar-open-time", required=True, type=int)
    parser.add_argument(
        "--max-clock-skew-ms", type=int, default=DEFAULT_MAX_CLOCK_SKEW_MS
    )
    parser.add_argument(
        "--deadline-seconds", type=int, default=DEFAULT_DEADLINE_SECONDS
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = collect_bundle(
            args.output_dir,
            quote=args.quote,
            interval=args.interval,
            bar_open=args.bar_open_time,
            max_clock_skew_ms=args.max_clock_skew_ms,
            deadline_seconds=args.deadline_seconds,
        )
        print(str(manifest))
        print(
            "collection complete; run the independent offline verifier before any research use"
        )
        return 0
    except Exception as error:
        if isinstance(error, CollectionFailure):
            print(
                f"market-context collection failed [{error.failure_kind}]: {error}",
                file=os.sys.stderr,
            )
        else:
            print(
                f"market-context collection failed [{type(error).__name__}]",
                file=os.sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
