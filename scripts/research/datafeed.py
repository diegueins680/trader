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

import json
import os
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


def align_pit(bar_close: np.ndarray, series: list[tuple[int, float]]) -> np.ndarray:
    """Point-in-time forward-fill: value at each bar is the latest observation
    with timestamp <= that bar's CLOSE. No future obs can affect an earlier bar."""
    series = sorted(series)
    out = np.full(len(bar_close), np.nan)
    j, last = 0, np.nan
    for i, c in enumerate(bar_close):
        while j < len(series) and series[j][0] <= c:
            last = series[j][1]
            j += 1
        out[i] = last
    return out


def _cache_path(sym, interval):
    return os.path.join(CACHE_DIR, f"{sym}_{interval}.csv")


def merge_cache_frames(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping cache rows without erasing point-in-time observations.

    Fresh non-null values replace older values for the same bar, while an
    unavailable value in the latest API window falls back to the previously
    accumulated value. This matters because Binance kline history is deeper
    than several derivatives-stat endpoints: a refresh must not turn an older
    funding/OI/basis/taker observation back into NaN.
    """
    if existing.empty:
        return fresh.drop_duplicates("openTime", keep="last").sort_values("openTime").reset_index(drop=True)
    if fresh.empty:
        return existing.drop_duplicates("openTime", keep="last").sort_values("openTime").reset_index(drop=True)

    old_by_time = existing.drop_duplicates("openTime", keep="last").set_index("openTime")
    fresh_by_time = fresh.drop_duplicates("openTime", keep="last").set_index("openTime")
    merged = fresh_by_time.combine_first(old_by_time)
    return merged.sort_index().reset_index()


def update_cache(symbols, interval="1h", kline_limit=1500):
    """Fetch the latest window for each symbol and merge (dedup by openTime) into
    the append-only CSV cache. Stats columns are stored already point-in-time
    aligned to the bars present at write time."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ms = INTERVAL_MS[interval]
    period = interval if interval in {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"} else "1h"
    for sym in symbols:
        k = fetch_klines(sym, interval, kline_limit)
        if k.empty:
            print(f"  {sym}: no klines")
            continue
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
        for column, fetch in fetchers.items():
            try:
                k[column] = align_pit(bclose, fetch())
            except Exception as e:  # each best-effort series is atomic
                print(f"  {sym}: {column} unavailable ({str(e)[:50]})")
                k[column] = np.nan
        path = _cache_path(sym, interval)
        if os.path.exists(path):
            old = pd.read_csv(path)
            k = merge_cache_frames(old, k)
        k.to_csv(path, index=False)
        print(f"  {sym}: cached {len(k)} bars -> {os.path.relpath(path)}")


def load_panel(symbols, interval="1h", refresh=True):
    """Return {sym: DataFrame} with ohlcv + funding/oi/basis/taker + derived
    columns (ret, fwd_ret). Refreshes the cache first unless refresh=False."""
    if refresh:
        update_cache(symbols, interval)
    panel = {}
    for sym in symbols:
        path = _cache_path(sym, interval)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).sort_values("openTime").reset_index(drop=True)
        lp = np.log(df["close"].to_numpy())
        df["ret"] = np.concatenate([[0.0], np.diff(lp)])
        df["fwd_ret"] = np.concatenate([np.diff(lp), [np.nan]])  # t -> t+1 (the label)
        panel[sym] = df
    return panel


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    print(f"Updating cache for {syms} (1h)…")
    update_cache(syms, "1h")
