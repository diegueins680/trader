"""Incremental, point-in-time data feed for the alpha-research harness.

Fetches Binance USDT-perp OHLCV plus the free derivatives-stats series (funding,
open interest, basis, taker buy/sell flow), caches them append-only under
``data/research/``, and serves bar-aligned panels.

Why a cache: the ``/futures/data`` stats endpoints only return ~30 days of
history, which is far too short to establish or refute an edge (see
``research-notes/data-acquisition-spec.md``). Running ``update_cache`` on a
schedule accumulates history going forward so the harness eventually has
months/years instead of days. The cache is also where you drop archival/paid
history (same schema) when you acquire it.

Dependencies: numpy, pandas, urllib (stdlib). No paid keys; all endpoints are
public.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://fapi.binance.com"
# Cache location: override with TRADER_RESEARCH_CACHE (e.g. for a standalone
# scheduled collector that must not depend on the repo checkout path), else
# default to data/research/ in the repo.
CACHE_DIR = os.environ.get("TRADER_RESEARCH_CACHE") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "research")
)
INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
STATS_RETENTION_MS = 30 * 86_400_000
STATS_RETENTION_SAFETY_MS = 300_000
STATS_PAGE_LIMIT = 500
FUNDING_FRESHNESS_MS = 9 * 3_600_000
COLLECTOR_STATE_DIR = ".collector"
COLLECTOR_LOCK_FILE = "collector.lock"
DERIVATIVE_OBSERVATION_DIR = ".observations"
DERIVATIVE_OBSERVATION_SCHEMA_ID = "binance_derivatives_first_seen_v2"
DERIVATIVE_FIELDS = ("funding", "oi", "basis", "taker")
DERIVATIVE_OBSERVATION_COLUMNS = (
    "schemaId",
    "symbol",
    "interval",
    "feature",
    "eventTime",
    "availabilityTime",
    "observed",
    "value",
)
TIMESTAMP_MAX = int(np.iinfo(np.int64).max)


class CacheWriterBusy(RuntimeError):
    """Raised when a non-blocking cache writer cannot acquire the shared lock."""


@contextmanager
def cache_writer_lock(*, blocking: bool = True):
    """Serialize every read/merge/replace transaction against the cache."""
    state_dir = os.path.join(CACHE_DIR, COLLECTOR_STATE_DIR)
    os.makedirs(state_dir, exist_ok=True)
    lock_path = os.path.join(state_dir, COLLECTOR_LOCK_FILE)
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(lock_handle.fileno(), operation)
        except BlockingIOError as error:
            raise CacheWriterBusy(f"research cache writer already active for {CACHE_DIR}") from error
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()} acquired={time.time():.3f}\n")
        lock_handle.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _get(path: str, tries: int = 4, **params):
    url = f"{BASE}{path}?" + "&".join(f"{k}={v}" for k, v in params.items())
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:  # transient — back off
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return []


def fetch_klines(sym: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """Closed klines, paginated past the 1000/req cap. Columns: openTime, open,
    high, low, close, volume."""
    out, end = [], None
    ms = INTERVAL_MS[interval]
    while len(out) < limit:
        p = dict(symbol=sym, interval=interval, limit=min(1000, limit - len(out)))
        if end is not None:
            p["endTime"] = end
        batch = _get("/fapi/v1/klines", **p)
        if not batch:
            break
        out = batch + out
        end = batch[0][0] - 1
        if len(batch) < p["limit"]:
            break
    rows = [
        (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        for k in out
    ]
    df = pd.DataFrame(rows, columns=["openTime", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("openTime").sort_values("openTime").reset_index(drop=True)
    # drop the still-forming last bar
    now = time.time() * 1000
    df = df[df["openTime"] + ms - 1 < now]
    return df.reset_index(drop=True)


def _stats(sym_or_pair_key: str, path: str, ts_key: str, val_key: str, **params):
    time.sleep(0.3)  # gentle pacing to avoid /futures/data rate limits
    rows = _get(path, **params)
    if isinstance(rows, dict):
        code = rows.get("code", "unknown")
        message = rows.get("msg", "unknown Binance stats error")
        raise RuntimeError(f"Binance stats error {code}: {message}")
    if not isinstance(rows, list):
        raise RuntimeError("Binance stats response is not a list")
    return [
        (int(r[ts_key]), float(r[val_key]))
        for r in rows
        if isinstance(r, dict) and ts_key in r and val_key in r
    ]


def _stats_window(
    sym_or_pair_key: str,
    path: str,
    ts_key: str,
    val_key: str,
    period_ms: int,
    start_time: int,
    end_time: int,
    page_limit: int = STATS_PAGE_LIMIT,
    **params,
) -> list[tuple[int, float]]:
    """Fetch a fixed stats snapshot, leaving missing intervals unavailable."""
    if period_ms <= 0:
        raise ValueError("stats period must be positive")
    if not 2 <= page_limit <= STATS_PAGE_LIMIT:
        raise ValueError("bounded stats page limit must be between 2 and 500")
    if start_time > end_time:
        return []

    values: dict[int, float] = {}
    chunk_start = int(start_time)
    snapshot_end = int(end_time)
    chunk_span = (page_limit - 1) * period_ms
    while chunk_start <= snapshot_end:
        chunk_end = min(snapshot_end, chunk_start + chunk_span)
        rows = _stats(
            sym_or_pair_key,
            path,
            ts_key,
            val_key,
            **params,
            limit=page_limit,
            startTime=chunk_start,
            endTime=chunk_end,
        )
        for timestamp, value in rows:
            if chunk_start <= timestamp <= chunk_end:
                values[timestamp] = value
        chunk_start = chunk_end + 1
    ordered = sorted(values.items())
    completed: list[tuple[int, float]] = []
    for timestamp, value in ordered:
        if completed:
            delta = timestamp - completed[-1][0]
            if delta % period_ms != 0:
                raise RuntimeError("Binance stats snapshot is off the requested grid")
            missing_time = completed[-1][0] + period_ms
            while missing_time < timestamp:
                completed.append((missing_time, float("nan")))
                missing_time += period_ms
        completed.append((timestamp, value))
    if completed:
        missing_time = completed[-1][0] + period_ms
        while missing_time <= snapshot_end:
            completed.append((missing_time, float("nan")))
            missing_time += period_ms
    return completed


def fetch_funding(sym, limit=1000, *, end_time=None):
    params = dict(symbol=sym, limit=limit)
    if end_time is not None:
        params["endTime"] = int(end_time)
    return _stats(
        sym,
        "/fapi/v1/fundingRate",
        "fundingTime",
        "fundingRate",
        **params,
    )


def fetch_oi(sym, period, limit=STATS_PAGE_LIMIT, *, start_time=None, end_time=None):
    if start_time is None and end_time is None:
        return _stats(
            sym,
            "/futures/data/openInterestHist",
            "timestamp",
            "sumOpenInterest",
            symbol=sym,
            period=period,
            limit=limit,
        )
    if start_time is None or end_time is None:
        raise ValueError("bounded OI fetch requires start_time and end_time")
    return _stats_window(
        sym,
        "/futures/data/openInterestHist",
        "timestamp",
        "sumOpenInterest",
        INTERVAL_MS[period],
        start_time,
        end_time,
        limit,
        symbol=sym,
        period=period,
    )


def fetch_basis(
    sym, period, limit=STATS_PAGE_LIMIT, *, start_time=None, end_time=None
):
    if start_time is None and end_time is None:
        return _stats(
            sym,
            "/futures/data/basis",
            "timestamp",
            "basisRate",
            pair=sym,
            contractType="PERPETUAL",
            period=period,
            limit=limit,
        )
    if start_time is None or end_time is None:
        raise ValueError("bounded basis fetch requires start_time and end_time")
    return _stats_window(
        sym,
        "/futures/data/basis",
        "timestamp",
        "basisRate",
        INTERVAL_MS[period],
        start_time,
        end_time,
        limit,
        pair=sym,
        contractType="PERPETUAL",
        period=period,
    )


def fetch_taker(
    sym, period, limit=STATS_PAGE_LIMIT, *, start_time=None, end_time=None
):
    if start_time is None and end_time is None:
        return _stats(
            sym,
            "/futures/data/takerlongshortRatio",
            "timestamp",
            "buySellRatio",
            symbol=sym,
            period=period,
            limit=limit,
        )
    if start_time is None or end_time is None:
        raise ValueError("bounded taker fetch requires start_time and end_time")
    return _stats_window(
        sym,
        "/futures/data/takerlongshortRatio",
        "timestamp",
        "buySellRatio",
        INTERVAL_MS[period],
        start_time,
        end_time,
        limit,
        symbol=sym,
        period=period,
    )


def align_pit(
    bar_close: np.ndarray,
    series: list[tuple[int, float]],
    *,
    max_age_ms: int | None = None,
) -> np.ndarray:
    """Point-in-time forward-fill: value at each bar is the latest observation
    with timestamp <= that bar's CLOSE. No future obs can affect an earlier bar."""
    series = sorted(series)
    out = np.full(len(bar_close), np.nan)
    j, last, last_timestamp = 0, np.nan, None
    for i, c in enumerate(bar_close):
        while j < len(series) and series[j][0] <= c:
            last_timestamp, last = series[j]
            j += 1
        if (
            max_age_ms is not None
            and last_timestamp is not None
            and c - last_timestamp > max_age_ms
        ):
            out[i] = np.nan
        else:
            out[i] = last
    return out


def _observation_path(symbol: str, interval: str, feature: str) -> str:
    """Path for one raw, first-seen derivatives observation ledger."""
    if feature not in DERIVATIVE_FIELDS:
        raise ValueError(f"unsupported derivatives feature: {feature}")
    if not symbol or not symbol.isalnum():
        raise ValueError("derivatives observation symbol must be alphanumeric")
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported derivatives observation interval: {interval}")
    return os.path.join(
        CACHE_DIR,
        DERIVATIVE_OBSERVATION_DIR,
        f"{symbol}_{interval}_{feature}_v2.csv",
    )


def _empty_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(DERIVATIVE_OBSERVATION_COLUMNS))


def _observation_frame(
    symbol: str,
    interval: str,
    feature: str,
    availability_time: int,
    series: list[tuple[int, float]],
) -> pd.DataFrame:
    """Tag finite API observations with their causal first-seen time.

    Binance's historical derivatives endpoints expose an event timestamp but
    no revision/publication timestamp. Consequently, a value first acquired by
    this collector is not usable before this fetch completed, even if the API
    returned an older event.
    """
    if availability_time < 0 or availability_time > TIMESTAMP_MAX:
        raise ValueError("derivatives availability time must be non-negative")
    rows = []
    for event_time, value in series:
        event_time = int(event_time)
        value = float(value)
        if event_time < 0 or event_time > availability_time:
            continue
        observed = int(np.isfinite(value))
        rows.append(
            {
                "schemaId": DERIVATIVE_OBSERVATION_SCHEMA_ID,
                "symbol": symbol,
                "interval": interval,
                "feature": feature,
                "eventTime": event_time,
                "availabilityTime": availability_time,
                "observed": observed,
                "value": value if observed else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=list(DERIVATIVE_OBSERVATION_COLUMNS))


def _validated_observation_frame(
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    feature: str,
) -> pd.DataFrame:
    if tuple(frame.columns) != DERIVATIVE_OBSERVATION_COLUMNS:
        raise ValueError("derivatives observation ledger has incompatible columns")
    if frame.empty:
        return _empty_observation_frame()
    expected_scope = {
        "schemaId": DERIVATIVE_OBSERVATION_SCHEMA_ID,
        "symbol": symbol,
        "interval": interval,
        "feature": feature,
    }
    for column, expected in expected_scope.items():
        if not frame[column].map(lambda value: value == expected).all():
            raise ValueError(f"derivatives observation ledger has mixed {column}")

    validated = frame.copy()
    for column in ("eventTime", "availabilityTime", "observed", "value"):
        validated[column] = pd.to_numeric(validated[column], errors="raise")
    numeric = validated.loc[
        :, ["eventTime", "availabilityTime", "observed", "value"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("derivatives observation ledger contains non-finite values")
    for column in ("eventTime", "availabilityTime"):
        values = validated[column].to_numpy(dtype=float)
        if (
            not np.equal(values, np.floor(values)).all()
            or np.any(values < 0)
            or np.any(values > TIMESTAMP_MAX)
        ):
            raise ValueError("derivatives observation ledger timestamps must be integers")
        validated[column] = validated[column].astype(np.int64)
    if not validated["observed"].isin([0, 1]).all():
        raise ValueError("derivatives observation ledger observed mask must be binary")
    if ((validated["observed"] == 0) & (validated["value"] != 0)).any():
        raise ValueError("unavailable derivatives observation has non-zero value")
    if (validated["eventTime"] < 0).any() or (
        validated["availabilityTime"] < validated["eventTime"]
    ).any():
        raise ValueError("derivatives observation ledger has incoherent timestamps")
    ambiguous = validated.duplicated(
        subset=["eventTime", "availabilityTime"], keep=False
    )
    if ambiguous.any():
        groups = validated.loc[ambiguous].groupby(
            ["eventTime", "availabilityTime"], sort=False
        )[["observed", "value"]]
        if any(len(states.drop_duplicates()) > 1 for _, states in groups):
            raise ValueError("derivatives observation ledger has an ambiguous revision")
    return validated


def merge_derivative_observations(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    symbol: str,
    interval: str,
    feature: str,
) -> pd.DataFrame:
    """Merge a first-seen ledger without backdating refetches or revisions."""
    old = _validated_observation_frame(existing, symbol, interval, feature)
    new = _validated_observation_frame(fresh, symbol, interval, feature)
    if old.empty and new.empty:
        return _empty_observation_frame()
    records = old.drop_duplicates().to_dict("records")
    latest_by_event = {}
    for record in records:
        event_time = int(record["eventTime"])
        incumbent = latest_by_event.get(event_time)
        if incumbent is None or int(record["availabilityTime"]) > int(
            incumbent["availabilityTime"]
        ):
            latest_by_event[event_time] = record
    for observation in new.drop_duplicates().sort_values(
        ["availabilityTime", "eventTime", "value"], kind="stable"
    ).to_dict("records"):
        event_time = int(observation["eventTime"])
        availability_time = int(observation["availabilityTime"])
        observed = int(observation["observed"])
        value = float(observation["value"])
        latest = latest_by_event.get(event_time)
        if latest is not None:
            latest_availability = int(latest["availabilityTime"])
            latest_observed = int(latest["observed"])
            latest_value = float(latest["value"])
            if availability_time < latest_availability:
                raise ValueError("derivatives observation would backdate a revision")
            if availability_time == latest_availability and (
                observed != latest_observed or value != latest_value
            ):
                raise ValueError("derivatives observation ledger has an ambiguous revision")
            if observed == latest_observed and value == latest_value:
                continue
        records.append(observation)
        latest_by_event[event_time] = observation
    merged = pd.DataFrame(records, columns=list(DERIVATIVE_OBSERVATION_COLUMNS))
    merged = _validated_observation_frame(merged, symbol, interval, feature)
    merged = merged.sort_values(
        ["availabilityTime", "eventTime", "value"], kind="stable"
    ).reset_index(drop=True)
    return merged.loc[:, list(DERIVATIVE_OBSERVATION_COLUMNS)]


def load_derivative_observations(
    symbol: str, interval: str, feature: str
) -> pd.DataFrame:
    path = _observation_path(symbol, interval, feature)
    if not os.path.exists(path):
        return _empty_observation_frame()
    return _validated_observation_frame(
        pd.read_csv(path), symbol, interval, feature
    )


def align_derivative_observations_v2(
    bar_close: np.ndarray,
    observations: pd.DataFrame,
    *,
    max_age_ms: int,
) -> pd.DataFrame:
    """Availability-time align a raw ledger and retain observed/fresh masks.

    The dense value is zero unless the selected observation is fresh. Event and
    availability witnesses remain populated for an observed-but-stale reading,
    so stale evidence cannot silently become a directional numeric feature.
    """
    if max_age_ms < 0:
        raise ValueError("derivatives freshness limit must be non-negative")
    closes = np.asarray(bar_close)
    if closes.ndim != 1:
        raise ValueError("derivatives bar closes must be a finite vector")
    numeric_closes = closes.astype(float)
    if not np.isfinite(numeric_closes).all() or not np.equal(
        numeric_closes, np.floor(numeric_closes)
    ).all() or np.any(numeric_closes < 0) or np.any(numeric_closes > TIMESTAMP_MAX):
        raise ValueError("derivatives bar closes must be a finite vector")
    closes = closes.astype(np.int64)
    if any(int(right) <= int(left) for left, right in zip(closes, closes[1:])):
        raise ValueError("derivatives bar closes must be strictly increasing")

    if tuple(observations.columns) != DERIVATIVE_OBSERVATION_COLUMNS:
        raise ValueError("derivatives observations have incompatible columns")
    if observations.empty:
        validated = _empty_observation_frame()
    else:
        validated = _validated_observation_frame(
            observations,
            str(observations.iloc[0]["symbol"]),
            str(observations.iloc[0]["interval"]),
            str(observations.iloc[0]["feature"]),
        )
    ordered = validated.sort_values(
        ["availabilityTime", "eventTime", "value"], kind="stable"
    ).reset_index(drop=True)
    values = np.zeros(len(closes), dtype=float)
    observed = np.zeros(len(closes), dtype=np.int8)
    fresh = np.zeros(len(closes), dtype=np.int8)
    event_times = np.full(len(closes), np.nan)
    availability_times = np.full(len(closes), np.nan)
    current_by_event: dict[int, tuple[int, int, float]] = {}
    cursor = 0
    for index, close_time in enumerate(closes):
        while (
            cursor < len(ordered)
            and int(ordered.at[cursor, "availabilityTime"]) <= int(close_time)
        ):
            event_time = int(ordered.at[cursor, "eventTime"])
            current_by_event[event_time] = (
                int(ordered.at[cursor, "availabilityTime"]),
                int(ordered.at[cursor, "observed"]),
                float(ordered.at[cursor, "value"]),
            )
            cursor += 1
        if not current_by_event:
            continue
        event_time = max(current_by_event)
        availability_time, is_observed, value = current_by_event[event_time]
        if not is_observed:
            continue
        observed[index] = 1
        event_times[index] = event_time
        availability_times[index] = availability_time
        if int(close_time) - event_time <= max_age_ms:
            fresh[index] = 1
            values[index] = value
    return pd.DataFrame(
        {
            "value": values,
            "observed": observed,
            "fresh": fresh,
            "eventTime": event_times,
            "availabilityTime": availability_times,
        }
    )


def _attach_derivative_v2_columns(
    frame: pd.DataFrame,
    feature: str,
    aligned: pd.DataFrame,
) -> None:
    if len(frame) != len(aligned):
        raise ValueError("derivatives v2 alignment length does not match bars")
    prefix = f"{feature}V2"
    for source, suffix in (
        ("value", "Value"),
        ("observed", "Observed"),
        ("fresh", "Fresh"),
        ("eventTime", "EventTime"),
        ("availabilityTime", "AvailabilityTime"),
    ):
        frame[prefix + suffix] = aligned[source].to_numpy()


def validate_derivative_v2_panel(
    frame: pd.DataFrame, interval: str
) -> dict[str, dict[str, int]]:
    """Validate additive v2 witnesses while permitting untouched legacy rows."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported derivatives panel interval: {interval}")
    if "openTime" not in frame:
        raise ValueError("derivatives panel has no openTime column")
    open_times_numeric = pd.to_numeric(frame["openTime"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(open_times_numeric).all() or not np.equal(
        open_times_numeric, np.floor(open_times_numeric)
    ).all() or np.any(open_times_numeric < 0) or np.any(
        open_times_numeric > TIMESTAMP_MAX - INTERVAL_MS[interval] + 1
    ):
        raise ValueError("derivatives panel open times must be finite integers")
    open_times = [int(value) for value in open_times_numeric]
    close_times = np.asarray(
        [value + INTERVAL_MS[interval] - 1 for value in open_times], dtype=object
    )
    coverage: dict[str, dict[str, int]] = {}
    for feature in DERIVATIVE_FIELDS:
        prefix = f"{feature}V2"
        names = {
            suffix: prefix + suffix
            for suffix in (
                "Value",
                "Observed",
                "Fresh",
                "EventTime",
                "AvailabilityTime",
            )
        }
        missing = [name for name in names.values() if name not in frame]
        if missing:
            raise ValueError(
                f"derivatives v2 panel is missing {feature} columns: "
                + ", ".join(missing)
            )
        numeric = {}
        for suffix, name in names.items():
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"derivatives v2 {feature} contains infinity")
            numeric[suffix] = values
        core_present = np.column_stack(
            [
                np.isfinite(numeric["Value"]),
                np.isfinite(numeric["Observed"]),
                np.isfinite(numeric["Fresh"]),
            ]
        )
        if np.any(core_present.any(axis=1) != core_present.all(axis=1)):
            raise ValueError(f"derivatives v2 {feature} core columns are partial")
        versioned = core_present.all(axis=1)
        observed = numeric["Observed"]
        fresh = numeric["Fresh"]
        if np.any(versioned & ~np.isin(observed, [0.0, 1.0])) or np.any(
            versioned & ~np.isin(fresh, [0.0, 1.0])
        ):
            raise ValueError(f"derivatives v2 {feature} masks must be binary")
        if np.any(versioned & (fresh > observed)):
            raise ValueError(f"derivatives v2 {feature} fresh mask exceeds observed")
        if np.any(versioned & (fresh == 0) & (numeric["Value"] != 0)):
            raise ValueError(
                f"derivatives v2 {feature} unavailable/stale value is non-zero"
            )
        has_event = np.isfinite(numeric["EventTime"])
        has_availability = np.isfinite(numeric["AvailabilityTime"])
        witnessed = versioned & (observed == 1)
        if np.any(witnessed & ~(has_event & has_availability)):
            raise ValueError(f"derivatives v2 {feature} observed row lacks timestamps")
        if np.any((~witnessed) & (has_event | has_availability)):
            raise ValueError(
                f"derivatives v2 {feature} unavailable row has timestamp evidence"
            )
        invalid_time = (
            (numeric["EventTime"] != np.floor(numeric["EventTime"]))
            | (
                numeric["AvailabilityTime"]
                != np.floor(numeric["AvailabilityTime"])
            )
            | (numeric["EventTime"] < 0)
            | (numeric["AvailabilityTime"] < 0)
            | (numeric["EventTime"] > TIMESTAMP_MAX)
            | (numeric["AvailabilityTime"] > TIMESTAMP_MAX)
        )
        if np.any(witnessed & invalid_time):
            raise ValueError(f"derivatives v2 {feature} timestamps must be integers")
        if np.any(
            witnessed
            & (
                (numeric["EventTime"] > numeric["AvailabilityTime"])
                | (numeric["AvailabilityTime"] > close_times)
            )
        ):
            raise ValueError(f"derivatives v2 {feature} timestamps are not causal")
        coverage[feature] = {
            "versioned": int(versioned.sum()),
            "observed": int(witnessed.sum()),
            "fresh": int((versioned & (fresh == 1)).sum()),
        }
    return coverage


def _cache_path(sym, interval):
    return os.path.join(CACHE_DIR, f"{sym}_{interval}.csv")


def merge_cache_frames(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping cache rows without erasing point-in-time observations.

    Fresh non-null values replace older values for the same bar, while an
    unavailable value in the latest API window falls back to the previously
    accumulated value. This matters because Binance kline history is deeper
    than several derivatives-stat endpoints: a refresh must not turn an older
    funding/OI/basis/taker observation back into NaN. Preserve the fresh
    frame's canonical column order and append legacy-only columns; pandas
    ``combine_first`` otherwise sorts columns alphabetically.
    """
    if existing.empty:
        return fresh.drop_duplicates("openTime", keep="last").sort_values("openTime").reset_index(drop=True)
    if fresh.empty:
        return existing.drop_duplicates("openTime", keep="last").sort_values("openTime").reset_index(drop=True)

    old_by_time = existing.drop_duplicates("openTime", keep="last").set_index("openTime")
    fresh_by_time = fresh.drop_duplicates("openTime", keep="last").set_index("openTime")
    merged_columns = list(fresh_by_time.columns)
    merged_columns.extend(
        column for column in old_by_time.columns if column not in fresh_by_time.columns
    )
    merged = fresh_by_time.combine_first(old_by_time)
    return merged.reindex(columns=merged_columns).sort_index().reset_index()


def write_cache_atomic(frame: pd.DataFrame, path: str) -> None:
    """Replace a cache CSV atomically without exposing a partial write."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _series_refresh_result(
    series: list[tuple[int, float]], snapshot_end: int, max_lag_ms: int
) -> dict[str, object]:
    bounded = [(timestamp, value) for timestamp, value in series if timestamp <= snapshot_end]
    finite = [
        (timestamp, value)
        for timestamp, value in bounded
        if np.isfinite(value)
    ]
    latest_observation = max(bounded, default=None, key=lambda item: item[0])
    latest_timestamp = max((timestamp for timestamp, _ in finite), default=None)
    lag_ms = snapshot_end - latest_timestamp if latest_timestamp is not None else None
    if latest_timestamp is None:
        status = "empty"
    elif latest_observation is not None and not np.isfinite(latest_observation[1]):
        status = "missing_tail"
    elif lag_ms > max_lag_ms:
        status = "stale"
    else:
        status = "ok"
    return {
        "status": status,
        "observations": len(series),
        "finite": len(finite),
        "latestTimestamp": int(latest_timestamp) if latest_timestamp is not None else None,
        "latestObservationTimestamp": (
            int(latest_observation[0]) if latest_observation is not None else None
        ),
        "lagMs": int(lag_ms) if lag_ms is not None else None,
    }


def _update_cache_unlocked(symbols, interval="1h", kline_limit=1500):
    """Fetch the latest window for each symbol and merge (dedup by openTime) into
    the append-only CSV cache. Legacy stats columns retain their historical
    event-time alignment. Additive v2 columns are rebuilt from raw first-seen
    ledgers and therefore never make a fetched value available before the
    collector actually observed it."""
    outcomes: dict[str, dict[str, object]] = {}
    os.makedirs(CACHE_DIR, exist_ok=True)
    ms = INTERVAL_MS[interval]
    period = interval if interval in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"} else "1h"
    for sym in symbols:
        k = fetch_klines(sym, interval, kline_limit)
        if k.empty:
            print(f"  {sym}: no klines")
            outcomes[sym] = {
                "status": "no_klines",
                "freshRows": 0,
                "freshLatestOpenTime": None,
                "series": {},
            }
            continue
        fresh_rows = len(k)
        fresh_latest_open_time = int(k["openTime"].iloc[-1])
        bclose = (k["openTime"] + ms - 1).to_numpy()
        snapshot_end = int(bclose[-1])
        # Stay just inside the documented retention boundary so the oldest
        # request remains valid while the fixed snapshot is collected.
        stats_start = max(
            int(bclose[0]),
            int(time.time() * 1000)
            - STATS_RETENTION_MS
            + STATS_RETENTION_SAFETY_MS,
        )
        fetchers = {
            "funding": lambda: fetch_funding(sym, end_time=snapshot_end),
            "oi": lambda: fetch_oi(
                sym, period, start_time=stats_start, end_time=snapshot_end
            ),
            "basis": lambda: fetch_basis(
                sym, period, start_time=stats_start, end_time=snapshot_end
            ),
            "taker": lambda: fetch_taker(
                sym, period, start_time=stats_start, end_time=snapshot_end
            ),
        }
        series_results: dict[str, dict[str, object]] = {}
        for column, fetch in fetchers.items():
            max_lag_ms = FUNDING_FRESHNESS_MS if column == "funding" else 2 * ms
            series: list[tuple[int, float]] | None = None
            try:
                series = fetch()
                series_results[column] = _series_refresh_result(
                    series, snapshot_end, max_lag_ms
                )
                k[column] = align_pit(bclose, series, max_age_ms=max_lag_ms)
            except Exception as e:  # each best-effort series is atomic
                print(f"  {sym}: {column} unavailable ({str(e)[:50]})")
                series_results[column] = {
                    "status": "error",
                    "error": str(e).replace("\n", " ")[:240],
                    "observations": 0,
                    "finite": 0,
                    "latestTimestamp": None,
                    "latestObservationTimestamp": None,
                    "lagMs": None,
                }
                k[column] = np.nan
            try:
                observations = load_derivative_observations(sym, interval, column)
                if series is not None:
                    first_seen_time = int(time.time() * 1000)
                    fresh_observations = _observation_frame(
                        sym,
                        interval,
                        column,
                        first_seen_time,
                        series,
                    )
                    observations = merge_derivative_observations(
                        observations,
                        fresh_observations,
                        sym,
                        interval,
                        column,
                    )
                    write_cache_atomic(
                        observations,
                        _observation_path(sym, interval, column),
                    )
                aligned_v2 = align_derivative_observations_v2(
                    bclose,
                    observations,
                    max_age_ms=max_lag_ms,
                )
                _attach_derivative_v2_columns(k, column, aligned_v2)
                series_results[column].update(
                    {
                        "observationSchema": DERIVATIVE_OBSERVATION_SCHEMA_ID,
                        "v2Status": "ok",
                        "v2Observations": len(observations),
                        "v2LatestAvailabilityTime": (
                            int(observations["availabilityTime"].max())
                            if not observations.empty
                            else None
                        ),
                    }
                )
            except Exception as error:
                source_status = series_results[column]["status"]
                print(
                    f"  {sym}: {column} v2 provenance unavailable "
                    f"({str(error)[:50]})"
                )
                unavailable = pd.DataFrame(
                    {
                        name: np.full(len(k), np.nan)
                        for name in (
                            "value",
                            "observed",
                            "fresh",
                            "eventTime",
                            "availabilityTime",
                        )
                    }
                )
                _attach_derivative_v2_columns(k, column, unavailable)
                series_results[column].update(
                    {
                        "status": "error",
                        "sourceStatus": source_status,
                        "observationSchema": DERIVATIVE_OBSERVATION_SCHEMA_ID,
                        "v2Status": "error",
                        "v2Error": str(error).replace("\n", " ")[:240],
                    }
                )
        path = _cache_path(sym, interval)
        if os.path.exists(path):
            old = pd.read_csv(path)
            k = merge_cache_frames(old, k)
        write_cache_atomic(k, path)
        print(f"  {sym}: cached {len(k)} bars -> {os.path.relpath(path)}")
        outcomes[sym] = {
            "status": "updated",
            "freshRows": fresh_rows,
            "freshLatestOpenTime": fresh_latest_open_time,
            "cacheRows": len(k),
            "series": series_results,
        }
    return outcomes


def update_cache(
    symbols,
    interval="1h",
    kline_limit=1500,
    *,
    acquire_lock: bool = True,
):
    """Refresh the cache and return explicit per-symbol acquisition evidence.

    Cache mutation takes the shared writer lock by default. A caller that owns
    ``cache_writer_lock`` across a larger transaction may opt out explicitly.
    """
    if acquire_lock:
        with cache_writer_lock():
            return _update_cache_unlocked(symbols, interval, kline_limit)
    return _update_cache_unlocked(symbols, interval, kline_limit)


def _load_panel_unlocked(symbols, interval):
    panel = {}
    for sym in symbols:
        path = _cache_path(sym, interval)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("openTime").reset_index(drop=True)
        lp = np.log(df["close"].to_numpy())
        df["ret"] = np.concatenate([[0.0], np.diff(lp)])
        df["fwd_ret"] = np.concatenate(
            [np.diff(lp), [np.nan]]
        )  # t -> t+1 (the label)
        panel[sym] = df
    return panel


def load_panel(symbols, interval="1h", refresh=True):
    """Load one cross-symbol snapshot, optionally refreshing it first."""
    if refresh:
        with cache_writer_lock():
            _update_cache_unlocked(symbols, interval)
            return _load_panel_unlocked(symbols, interval)
    try:
        with cache_writer_lock():
            return _load_panel_unlocked(symbols, interval)
    except OSError:
        if os.access(CACHE_DIR, os.W_OK):
            raise
        # A genuinely immutable mount cannot race a writer on that mount.
        return _load_panel_unlocked(symbols, interval)


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    print(f"Updating cache for {syms} (1h)…")
    update_cache(syms, "1h")
