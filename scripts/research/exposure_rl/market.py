"""Causal market panel and observation features for exposure_env_v1.

Every feature at bar index ``t`` is a function of bars ``<= t`` and of funding
events with ``fundingTime <= closeTime[t]`` only. No scaler is fitted: all
normalisations are fixed, data-independent transforms, so there is no fitted
statistic that could carry later information into an earlier observation.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BAR_MS = 6 * 60 * 60 * 1000
BURN_IN = 60  # bars of history required before the first decision
BENCHMARK = "BTCUSDT"
FEATURE_VERSION = "exposure_features_v1"
MARKET_FEATURES = (
    "ret1_z",
    "ret4_z",
    "ret12_z",
    "ret28_z",
    "vol_ratio_14_56",
    "vol_level_56",
    "donchian20_position",
    "ema20_ema50_gap_atr",
    "rsi14",
    "adx14",
    "volume_log_ratio_56",
    "funding_last",
    "funding_mean_21",
    "btc_ret4_z",
    "btc_ret28_z",
)
CLIP = 5.0


@dataclass(frozen=True)
class Series:
    symbol: str
    open_time: np.ndarray  # int64, bar open (ms)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    funding_at_open: np.ndarray  # sum(rate*mark) for events at exactly open_time[b]
    funding_inside: np.ndarray  # sum(rate*mark) for events in (open_time[b], close_time[b]]
    features: np.ndarray  # [bars, len(MARKET_FEATURES)]; rows < BURN_IN are not used

    def __len__(self) -> int:
        return len(self.close)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        total = np.cumsum(np.insert(values, 0, 0.0))
        out[window - 1 :] = (total[window:] - total[:-window]) / window
    return out


def _rolling_extreme(values: np.ndarray, window: int, highest: bool) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        chunk = values[index - window + 1 : index + 1]
        out[index] = chunk.max() if highest else chunk.min()
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values))
    state = values[0]
    for index, value in enumerate(values):
        state = state + alpha * (value - state)
        out[index] = state
    return out


def _wilder(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(len(values))
    state = values[0]
    for index, value in enumerate(values):
        state = state + (value - state) / period
        out[index] = state
    return out


def _lagged_sum(returns: np.ndarray, horizon: int) -> np.ndarray:
    return _rolling_mean(returns, horizon) * horizon


def market_features(open_time, high, low, close, volume, funding_times, funding_rates, benchmark=None):
    """Return [bars, features]. ``benchmark`` is (open_time, close) of BTC or None."""
    bars = len(close)
    log_close = np.log(close)
    returns = np.zeros(bars)
    returns[1:] = np.diff(log_close)
    vol56 = np.sqrt(_rolling_mean(returns * returns, 56))
    vol14 = np.sqrt(_rolling_mean(returns * returns, 14))
    safe56 = np.where(vol56 > 1e-8, vol56, np.nan)
    columns = [
        returns / safe56,
        _lagged_sum(returns, 4) / (safe56 * 2.0),
        _lagged_sum(returns, 12) / (safe56 * np.sqrt(12.0)),
        _lagged_sum(returns, 28) / (safe56 * np.sqrt(28.0)),
        np.log(np.where(vol14 > 1e-8, vol14, np.nan) / safe56),
        np.log(safe56 / 0.02),
    ]
    top = _rolling_extreme(high, 20, True)
    bottom = _rolling_extreme(low, 20, False)
    width = np.where(top - bottom > 0, top - bottom, np.nan)
    columns.append((2.0 * close - top - bottom) / width)
    previous = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    atr = _wilder(true_range, 14)
    columns.append((_ema(close, 20) - _ema(close, 50)) / np.where(atr > 0, atr, np.nan))
    change = close - previous
    gain = _wilder(np.maximum(change, 0.0), 14)
    loss = _wilder(np.maximum(-change, 0.0), 14)
    columns.append((gain - loss) / np.where(gain + loss > 0, gain + loss, np.nan))
    up = high - np.concatenate(([high[0]], high[:-1]))
    down = np.concatenate(([low[0]], low[:-1])) - low
    plus = _wilder(np.where((up > down) & (up > 0), up, 0.0), 14)
    minus = _wilder(np.where((down > up) & (down > 0), down, 0.0), 14)
    spread = np.abs(plus - minus) / np.where(plus + minus > 0, plus + minus, np.nan)
    adx = _wilder(np.nan_to_num(spread), 14)
    columns.append(2.0 * adx - 0.5)
    mean_volume = _rolling_mean(volume, 56)
    columns.append(np.log(np.where(volume > 0, volume, np.nan) / np.where(mean_volume > 0, mean_volume, np.nan)))
    close_time = open_time + BAR_MS - 1
    # Number of funding events known at each bar close (fundingTime <= closeTime).
    known = np.searchsorted(funding_times, close_time, side="right")
    cumulative = np.concatenate(([0.0], np.cumsum(funding_rates)))
    last = np.where(known > 0, funding_rates[np.maximum(known, 1) - 1] if len(funding_rates) else 0.0, 0.0)
    first = np.maximum(known - 21, 0)
    mean21 = np.where(known > 0, (cumulative[known] - cumulative[first]) / np.maximum(known - first, 1), 0.0)
    columns.append(last * 2000.0)
    columns.append(mean21 * 2000.0)
    if benchmark is None:
        columns.append(columns[1].copy())
        columns.append(columns[3].copy())
    else:
        bench_time, bench_close = benchmark
        bench_returns = np.zeros(len(bench_close))
        bench_returns[1:] = np.diff(np.log(bench_close))
        bench_vol = np.sqrt(_rolling_mean(bench_returns * bench_returns, 56))
        bench_safe = np.where(bench_vol > 1e-8, bench_vol, np.nan)
        four = _lagged_sum(bench_returns, 4) / (bench_safe * 2.0)
        month = _lagged_sum(bench_returns, 28) / (bench_safe * np.sqrt(28.0))
        # Align on identical bar open times; a missing benchmark bar gives 0.
        position = np.searchsorted(bench_time, open_time)
        ok = (position < len(bench_time)) & (bench_time[np.minimum(position, len(bench_time) - 1)] == open_time)
        columns.append(np.where(ok, four[np.minimum(position, len(bench_time) - 1)], 0.0))
        columns.append(np.where(ok, month[np.minimum(position, len(bench_time) - 1)], 0.0))
    features = np.stack(columns, axis=1)
    return np.clip(np.nan_to_num(features, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)


def load_panel(panel_path: Path, funding_path: Path, expected: dict | None = None) -> dict[str, Series]:
    """Load a registered snapshot, verifying hashes when ``expected`` is given."""
    if expected is not None:
        if sha256_file(panel_path) != expected["panelSha256"] or sha256_file(funding_path) != expected["fundingSha256"]:
            raise ValueError("snapshot hash does not match the registered manifest")
    raw: dict[str, list[tuple]] = {}
    with Path(panel_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw.setdefault(row["symbol"], []).append(
                (int(row["openTime"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]))
            )
    events: dict[str, list[tuple]] = {symbol: [] for symbol in raw}
    with Path(funding_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            events[row["symbol"]].append((int(row["fundingTime"]), float(row["fundingRate"]), float(row["markPrice"])))
    arrays = {}
    for symbol, rows in raw.items():
        table = np.asarray(rows, dtype=np.float64)
        open_time = np.asarray([row[0] for row in rows], dtype=np.int64)
        if np.any(np.diff(open_time) != BAR_MS) or not np.isfinite(table).all() or np.any(table[:, 1:5] <= 0):
            raise ValueError(f"{symbol} panel is not a finite, positive, gap-free 6h grid")
        if np.any(table[:, 2] < np.maximum(table[:, 1], table[:, 4])) or np.any(table[:, 3] > np.minimum(table[:, 1], table[:, 4])):
            raise ValueError(f"{symbol} panel has inconsistent high/low")
        arrays[symbol] = (open_time, table)
    panel = {}
    bench = arrays.get(BENCHMARK)
    for symbol, (open_time, table) in arrays.items():
        funding = events[symbol]
        times = np.asarray([row[0] for row in funding], dtype=np.int64)
        rates = np.asarray([row[1] for row in funding], dtype=np.float64)
        marks = np.asarray([row[2] for row in funding], dtype=np.float64)
        if np.any(np.diff(times) <= 0) or not (np.isfinite(rates).all() and np.all(marks > 0)):
            raise ValueError(f"{symbol} funding events are invalid")
        # Funding stamps are a few ms after the nominal settlement; snap within 1 min.
        snapped = np.where(times % BAR_MS < 60_000, times - times % BAR_MS, times)
        bar = np.searchsorted(open_time, snapped, side="right") - 1
        inside = bar >= 0
        at_open = inside & (snapped == open_time[np.maximum(bar, 0)])
        cash = rates * marks
        f_open = np.zeros(len(open_time))
        f_inside = np.zeros(len(open_time))
        np.add.at(f_open, bar[at_open], cash[at_open])
        np.add.at(f_inside, bar[inside & ~at_open], cash[inside & ~at_open])
        features = market_features(
            open_time, table[:, 2], table[:, 3], table[:, 4], table[:, 5], times, rates,
            None if bench is None or symbol == BENCHMARK else (bench[0], bench[1][:, 4]),
        )
        panel[symbol] = Series(symbol, open_time, table[:, 1], table[:, 2], table[:, 3], table[:, 4], table[:, 5], f_open, f_inside, features)
    return dict(sorted(panel.items()))
