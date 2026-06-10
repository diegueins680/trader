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
CACHE_DIR = os.path.abspath(
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
    if not isinstance(rows, list):  # error responses come back as a dict
        return []
    return [
        (int(r[ts_key]), float(r[val_key]))
        for r in rows
        if isinstance(r, dict) and ts_key in r and val_key in r
    ]


def fetch_funding(sym, limit=1000):
    return _stats(sym, "/fapi/v1/fundingRate", "fundingTime", "fundingRate", symbol=sym, limit=limit)


def fetch_oi(sym, period, limit=500):
    return _stats(sym, "/futures/data/openInterestHist", "timestamp", "sumOpenInterest", symbol=sym, period=period, limit=limit)


def fetch_basis(sym, period, limit=500):
    return _stats(sym, "/futures/data/basis", "timestamp", "basisRate", pair=sym, contractType="PERPETUAL", period=period, limit=limit)


def fetch_taker(sym, period, limit=500):
    return _stats(sym, "/futures/data/takerlongshortRatio", "timestamp", "buySellRatio", symbol=sym, period=period, limit=limit)


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
        try:
            k["funding"] = align_pit(bclose, fetch_funding(sym))
            k["oi"] = align_pit(bclose, fetch_oi(sym, period))
            k["basis"] = align_pit(bclose, fetch_basis(sym, period))
            k["taker"] = align_pit(bclose, fetch_taker(sym, period))
        except Exception as e:  # stats are best-effort; OHLCV still cached
            print(f"  {sym}: stats partial ({str(e)[:50]})")
            for c in ("funding", "oi", "basis", "taker"):
                if c not in k:
                    k[c] = np.nan
        path = _cache_path(sym, interval)
        if os.path.exists(path):
            old = pd.read_csv(path)
            k = (
                pd.concat([old, k])
                .drop_duplicates("openTime", keep="last")
                .sort_values("openTime")
                .reset_index(drop=True)
            )
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
