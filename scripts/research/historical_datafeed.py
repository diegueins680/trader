"""Bounded public Binance history for pre-registered research snapshots.

Every fetch pins an inclusive ``startTime``/``endTime`` window and paginates
forward. Results are returned only after the complete response has passed
ordering, duplicate, progress, and (for klines) interval-grid checks.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any


BASE_URL = "https://fapi.binance.com"
CONTRACT_INTERVAL = "8h"
CONTRACT_INTERVAL_MS = 8 * 60 * 60 * 1000
MARK_PRICE_INTERVAL = "1h"
MARK_PRICE_INTERVAL_MS = 60 * 60 * 1000
KLINE_MAX_PAGE_LIMIT = 1500
KLINE_DEFAULT_PAGE_LIMIT = 499
FUNDING_MAX_PAGE_LIMIT = 1000
REQUEST_WEIGHT_BUDGET = 1200
REQUEST_WEIGHT_WINDOW_SECONDS = 60.0
FUNDING_REQUEST_BUDGET = 450
FUNDING_REQUEST_WINDOW_SECONDS = 5 * 60.0
MAX_PAGES_PER_SERIES = 512


class SnapshotIntegrityError(RuntimeError):
    """Raised when an API response cannot prove a complete bounded snapshot."""


def _bounded_sleep(delay: float, deadline: float | None) -> None:
    delay = max(0.0, delay)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or delay >= remaining:
            raise TimeoutError("historical snapshot acquisition deadline exceeded")
    time.sleep(delay)


class RequestWeightLimiter:
    """Keep this process well below Binance's shared one-minute IP budget."""

    def __init__(self, budget: int, window_seconds: float) -> None:
        if budget < 1 or window_seconds <= 0:
            raise ValueError("rate-limit budget and window must be positive")
        self.budget = budget
        self.window_seconds = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0

    def wait(self, weight: int, deadline: float | None = None) -> None:
        if weight < 1 or weight > self.budget:
            raise ValueError("request weight must fit within the limiter budget")
        while True:
            now = time.monotonic()
            cutoff = now - self.window_seconds
            while self._events and self._events[0][0] <= cutoff:
                _, expired_weight = self._events.popleft()
                self._used -= expired_weight
            if self._used + weight <= self.budget:
                self._events.append((now, weight))
                self._used += weight
                return
            delay = self.window_seconds - (now - self._events[0][0])
            _bounded_sleep(max(0.01, delay), deadline)

    def observe_used_weight(self, used_weight: int) -> None:
        """Conservatively account for IP-wide traffic reported by Binance."""
        if isinstance(used_weight, bool) or not isinstance(used_weight, int):
            raise TypeError("observed request weight must be an integer")
        if used_weight < 0:
            raise ValueError("observed request weight must be non-negative")
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            _, expired_weight = self._events.popleft()
            self._used -= expired_weight
        if used_weight > self._used:
            external_weight = used_weight - self._used
            self._events.append((now, external_weight))
            self._used += external_weight


RATE_LIMITER = RequestWeightLimiter(
    REQUEST_WEIGHT_BUDGET, REQUEST_WEIGHT_WINDOW_SECONDS
)
FUNDING_RATE_LIMITER = RequestWeightLimiter(
    FUNDING_REQUEST_BUDGET, FUNDING_REQUEST_WINDOW_SECONDS
)


def _request_weight(path: str, params: Mapping[str, object]) -> int:
    if path not in {
        "/fapi/v1/continuousKlines",
        "/fapi/v1/markPriceKlines",
    }:
        return 1
    try:
        limit = int(params.get("limit", 500))
    except (TypeError, ValueError) as error:
        raise ValueError("kline request limit must be an integer") from error
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


def _observe_response_weight(headers: object) -> None:
    getter = getattr(headers, "get", None)
    if getter is None:
        return
    raw_weight = getter("X-MBX-USED-WEIGHT-1M")
    if raw_weight is None:
        return
    try:
        used_weight = int(raw_weight)
    except (TypeError, ValueError):
        return
    if used_weight >= 0:
        RATE_LIMITER.observe_used_weight(used_weight)


def _get(
    path: str,
    *,
    tries: int = 4,
    deadline: float | None = None,
    **params: object,
) -> object:
    """Read one public Binance JSON response with bounded transient retries."""
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    for attempt in range(tries):
        if path == "/fapi/v1/fundingRate":
            FUNDING_RATE_LIMITER.wait(1, deadline)
        RATE_LIMITER.wait(_request_weight(path, params), deadline)
        timeout = 30.0
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("historical snapshot acquisition deadline exceeded")
            timeout = min(timeout, remaining)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                _observe_response_weight(response.headers)
                payload = json.load(response)
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        "historical snapshot acquisition deadline exceeded"
                    )
                return payload
        except urllib.error.HTTPError as error:
            _observe_response_weight(error.headers)
            retryable = error.code in {418, 429, 500, 502, 503, 504}
            if attempt == tries - 1 or not retryable:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay = 0.0
            _bounded_sleep(max(delay, 1.5 * (attempt + 1)), deadline)
        except (urllib.error.URLError, TimeoutError):
            if (
                attempt == tries - 1
                or (deadline is not None and time.monotonic() >= deadline)
            ):
                raise
            _bounded_sleep(1.5 * (attempt + 1), deadline)
    raise AssertionError("unreachable retry state")


def _validate_window(
    start_time: int, end_time: int, page_limit: int, maximum: int
) -> None:
    if isinstance(start_time, bool) or not isinstance(start_time, int):
        raise TypeError("start_time must be an integer millisecond timestamp")
    if isinstance(end_time, bool) or not isinstance(end_time, int):
        raise TypeError("end_time must be an integer millisecond timestamp")
    if start_time < 0 or end_time < 0:
        raise ValueError("snapshot timestamps must be non-negative")
    if start_time > end_time:
        raise ValueError("snapshot start_time must not exceed end_time")
    if isinstance(page_limit, bool) or not isinstance(page_limit, int):
        raise TypeError("page_limit must be an integer")
    if not 1 <= page_limit <= maximum:
        raise ValueError(f"page_limit must be between 1 and {maximum}")


def _response_rows(payload: object, path: str) -> list[object]:
    if isinstance(payload, Mapping):
        code = payload.get("code", "unknown")
        message = payload.get("msg", "unknown Binance API error")
        raise RuntimeError(f"Binance API error {code} from {path}: {message}")
    if not isinstance(payload, list):
        raise SnapshotIntegrityError(f"Binance response from {path} is not a list")
    return payload


def _store_record(
    records: dict[int, dict[str, object]],
    timestamp: int,
    record: dict[str, object],
    path: str,
) -> None:
    if timestamp in records:
        raise SnapshotIntegrityError(f"duplicate timestamp {timestamp} from {path}")
    records[timestamp] = record


def _string_field(
    raw_row: Sequence[object] | Mapping[str, object], key: int | str
) -> str:
    if isinstance(key, int) and isinstance(raw_row, Sequence):
        value = raw_row[key]
    elif isinstance(key, str) and isinstance(raw_row, Mapping):
        value = raw_row[key]
    else:
        raise TypeError(f"field {key} does not match the response row type")
    if not isinstance(value, str):
        raise TypeError(f"field {key} is not a string")
    return value


def _paginate_window(
    path: str,
    *,
    start_time: int,
    end_time: int,
    page_limit: int,
    maximum_page_limit: int,
    timestamp_key: str,
    parse_row: Callable[[object], dict[str, object]],
    request_params: Mapping[str, object],
    deadline: float | None = None,
) -> list[dict[str, object]]:
    """Collect a fixed inclusive window without ever returning a partial page set."""
    _validate_window(start_time, end_time, page_limit, maximum_page_limit)
    cursor = start_time
    records: dict[int, dict[str, object]] = {}
    pages = 0

    while cursor <= end_time:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("historical snapshot acquisition deadline exceeded")
        if pages >= MAX_PAGES_PER_SERIES:
            raise SnapshotIntegrityError(
                f"snapshot exceeded {MAX_PAGES_PER_SERIES} pages for {path}"
            )
        pages += 1
        payload = _get(
            path,
            deadline=deadline,
            **request_params,
            startTime=cursor,
            endTime=end_time,
            limit=page_limit,
        )
        rows = _response_rows(payload, path)
        if not rows:
            break

        page_timestamps: list[int] = []
        for raw_row in rows:
            record = parse_row(raw_row)
            timestamp = record[timestamp_key]
            if not isinstance(timestamp, int):
                raise SnapshotIntegrityError(
                    f"non-integer {timestamp_key} returned from {path}"
                )
            if not start_time <= timestamp <= end_time or timestamp < cursor:
                raise SnapshotIntegrityError(
                    f"timestamp {timestamp} is outside the requested page for {path}"
                )
            if page_timestamps and timestamp <= page_timestamps[-1]:
                raise SnapshotIntegrityError(
                    f"page timestamps are duplicate or unordered for {path}"
                )
            page_timestamps.append(timestamp)
            _store_record(records, timestamp, record, path)

        if not page_timestamps:
            raise SnapshotIntegrityError(f"pagination made no progress from {cursor} for {path}")
        latest = page_timestamps[-1]
        next_cursor = latest + 1
        if next_cursor <= cursor:
            raise SnapshotIntegrityError(
                f"pagination cursor did not advance from {cursor} for {path}"
            )
        cursor = next_cursor

    return [records[timestamp] for timestamp in sorted(records)]


def _parse_contract_kline(raw_row: object) -> dict[str, object]:
    if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
        raise SnapshotIntegrityError("continuous kline row is not an array")
    if len(raw_row) < 12:
        raise SnapshotIntegrityError("continuous kline row has fewer than 12 fields")
    try:
        return {
            "openTime": int(raw_row[0]),
            "open": _string_field(raw_row, 1),
            "high": _string_field(raw_row, 2),
            "low": _string_field(raw_row, 3),
            "close": _string_field(raw_row, 4),
            "volume": _string_field(raw_row, 5),
            "closeTime": int(raw_row[6]),
            "quoteVolume": _string_field(raw_row, 7),
            "trades": int(raw_row[8]),
            "takerBuyBaseVolume": _string_field(raw_row, 9),
            "takerBuyQuoteVolume": _string_field(raw_row, 10),
        }
    except (TypeError, ValueError) as error:
        raise SnapshotIntegrityError("continuous kline row has invalid fields") from error


def _parse_mark_price_kline(raw_row: object) -> dict[str, object]:
    if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
        raise SnapshotIntegrityError("mark-price kline row is not an array")
    if len(raw_row) < 7:
        raise SnapshotIntegrityError("mark-price kline row has fewer than 7 fields")
    try:
        return {
            "openTime": int(raw_row[0]),
            "open": _string_field(raw_row, 1),
            "high": _string_field(raw_row, 2),
            "low": _string_field(raw_row, 3),
            "close": _string_field(raw_row, 4),
            "closeTime": int(raw_row[6]),
        }
    except (TypeError, ValueError) as error:
        raise SnapshotIntegrityError("mark-price kline row has invalid fields") from error


def _parse_funding_event(raw_row: object) -> dict[str, object]:
    if not isinstance(raw_row, Mapping):
        raise SnapshotIntegrityError("funding event is not an object")
    required = ("fundingTime", "fundingRate", "markPrice")
    if any(field not in raw_row for field in required):
        raise SnapshotIntegrityError("funding event is missing required fields")
    try:
        record: dict[str, object] = {
            "fundingTime": int(raw_row["fundingTime"]),
            "fundingRate": _string_field(raw_row, "fundingRate"),
            "markPrice": _string_field(raw_row, "markPrice"),
        }
    except (TypeError, ValueError) as error:
        raise SnapshotIntegrityError("funding event has invalid fields") from error
    if "symbol" in raw_row:
        record["symbol"] = str(raw_row["symbol"])
    return record


def _validate_kline_grid(
    records: list[dict[str, object]],
    interval_ms: int,
    endpoint_name: str,
    start_time: int,
    end_time: int,
) -> None:
    timestamps = [int(record["openTime"]) for record in records]
    expected_first = ((start_time + interval_ms - 1) // interval_ms) * interval_ms
    expected_last = (end_time // interval_ms) * interval_ms
    if expected_first <= expected_last:
        if not timestamps:
            raise SnapshotIntegrityError(f"{endpoint_name} snapshot is empty")
        if timestamps[0] != expected_first:
            raise SnapshotIntegrityError(
                f"{endpoint_name} snapshot starts at {timestamps[0]}, expected {expected_first}"
            )
        if timestamps[-1] != expected_last:
            raise SnapshotIntegrityError(
                f"{endpoint_name} snapshot ends at {timestamps[-1]}, expected {expected_last}"
            )
    elif timestamps:
        raise SnapshotIntegrityError(
            f"{endpoint_name} returned rows outside an empty grid window"
        )
    for timestamp in timestamps:
        if timestamp % interval_ms != 0:
            raise SnapshotIntegrityError(
                f"{endpoint_name} timestamp {timestamp} is off the requested grid"
            )
    for record in records:
        expected_close = int(record["openTime"]) + interval_ms - 1
        if int(record["closeTime"]) != expected_close:
            raise SnapshotIntegrityError(
                f"{endpoint_name} at {record['openTime']} has an invalid close time"
            )
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous != interval_ms:
            raise SnapshotIntegrityError(
                f"{endpoint_name} snapshot has a gap after {previous}"
            )


def fetch_contract_klines(
    pair: str,
    start_time: int,
    end_time: int,
    *,
    page_limit: int = KLINE_DEFAULT_PAGE_LIMIT,
    deadline: float | None = None,
) -> list[dict[str, object]]:
    """Fetch an ascending, gap-free 8h PERPETUAL continuous-contract snapshot."""
    if not pair:
        raise ValueError("pair must not be empty")
    records = _paginate_window(
        "/fapi/v1/continuousKlines",
        start_time=start_time,
        end_time=end_time,
        page_limit=page_limit,
        maximum_page_limit=KLINE_MAX_PAGE_LIMIT,
        timestamp_key="openTime",
        parse_row=_parse_contract_kline,
        request_params={
            "pair": pair,
            "contractType": "PERPETUAL",
            "interval": CONTRACT_INTERVAL,
        },
        deadline=deadline,
    )
    _validate_kline_grid(
        records,
        CONTRACT_INTERVAL_MS,
        "continuous kline",
        start_time,
        end_time,
    )
    return records


def fetch_mark_price_klines(
    symbol: str,
    start_time: int,
    end_time: int,
    *,
    page_limit: int = KLINE_DEFAULT_PAGE_LIMIT,
    deadline: float | None = None,
) -> list[dict[str, object]]:
    """Fetch an ascending, gap-free 1h mark-price-kline snapshot."""
    if not symbol:
        raise ValueError("symbol must not be empty")
    records = _paginate_window(
        "/fapi/v1/markPriceKlines",
        start_time=start_time,
        end_time=end_time,
        page_limit=page_limit,
        maximum_page_limit=KLINE_MAX_PAGE_LIMIT,
        timestamp_key="openTime",
        parse_row=_parse_mark_price_kline,
        request_params={"symbol": symbol, "interval": MARK_PRICE_INTERVAL},
        deadline=deadline,
    )
    _validate_kline_grid(
        records,
        MARK_PRICE_INTERVAL_MS,
        "mark-price kline",
        start_time,
        end_time,
    )
    return records


def fetch_funding_events(
    symbol: str,
    start_time: int,
    end_time: int,
    *,
    page_limit: int = FUNDING_MAX_PAGE_LIMIT,
    deadline: float | None = None,
) -> list[dict[str, object]]:
    """Fetch ordered funding events, preserving rate, time, and charged mark price."""
    if not symbol:
        raise ValueError("symbol must not be empty")
    return _paginate_window(
        "/fapi/v1/fundingRate",
        start_time=start_time,
        end_time=end_time,
        page_limit=page_limit,
        maximum_page_limit=FUNDING_MAX_PAGE_LIMIT,
        timestamp_key="fundingTime",
        parse_row=_parse_funding_event,
        request_params={"symbol": symbol},
        deadline=deadline,
    )


def write_artifact_atomic(artifact: Any, path: str | os.PathLike[str]) -> None:
    """Atomically replace a JSON artifact without exposing a partial snapshot."""
    destination = os.fspath(path)
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                artifact,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
