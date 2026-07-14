import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

test(
  "research stats pagination is fixed-snapshot and cache failures are isolated",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import datafeed

hour = datafeed.INTERVAL_MS["1h"]
timestamps = [index * hour for index in range(700)]
calls = []

def fake_get(path, **params):
    calls.append((path, dict(params)))
    value_key = {
        "/futures/data/openInterestHist": "sumOpenInterest",
        "/futures/data/basis": "basisRate",
        "/futures/data/takerlongshortRatio": "buySellRatio",
    }[path]
    rows = [
        {"timestamp": timestamp, value_key: str(index + 0.25)}
        for index, timestamp in enumerate(timestamps)
        if params["startTime"] <= timestamp <= params["endTime"]
    ]
    rows.reverse()
    if rows:
        rows.append(dict(rows[-1]))
    rows.append({"timestamp": params["endTime"] + hour, value_key: "999999"})
    return rows

datafeed._get = fake_get
datafeed.time.sleep = lambda _seconds: None
end_time = timestamps[-1]
oi = datafeed.fetch_oi("BTCUSDT", "1h", start_time=0, end_time=end_time)
basis = datafeed.fetch_basis("BTCUSDT", "1h", start_time=0, end_time=end_time)
taker = datafeed.fetch_taker("BTCUSDT", "1h", start_time=0, end_time=end_time)

for series in (oi, basis, taker):
    assert len(series) == 700
    assert [timestamp for timestamp, _value in series] == timestamps
    assert series[0][1] == 0.25
    assert series[-1][1] == 699.25

assert len(calls) == 6
for offset, expected_path in enumerate([
    "/futures/data/openInterestHist",
    "/futures/data/basis",
    "/futures/data/takerlongshortRatio",
]):
    first_path, first = calls[offset * 2]
    second_path, second = calls[offset * 2 + 1]
    assert first_path == expected_path
    assert second_path == expected_path
    assert first["startTime"] == 0
    assert first["endTime"] == 499 * hour
    assert second["startTime"] == 499 * hour + 1
    assert second["endTime"] == end_time
    assert first["limit"] == datafeed.STATS_PAGE_LIMIT
    assert second["limit"] == datafeed.STATS_PAGE_LIMIT
    assert first["period"] == "1h"
    assert second["period"] == "1h"

oi_params = calls[0][1]
basis_params = calls[2][1]
taker_params = calls[4][1]
assert oi_params["symbol"] == "BTCUSDT"
assert basis_params["pair"] == "BTCUSDT"
assert basis_params["contractType"] == "PERPETUAL"
assert taker_params["symbol"] == "BTCUSDT"

def missing_middle_get(path, **params):
    value_key = {
        "/futures/data/openInterestHist": "sumOpenInterest",
    }[path]
    return [
        {"timestamp": timestamp, value_key: str(index)}
        for index, timestamp in enumerate(timestamps)
        if params["startTime"] <= timestamp <= params["endTime"]
        and timestamp != 250 * hour
    ]

datafeed._get = missing_middle_get
missing_middle = datafeed.fetch_oi(
    "BTCUSDT", "1h", start_time=0, end_time=end_time
)
assert len(missing_middle) == 700
assert np.isnan(dict(missing_middle)[250 * hour])

def stale_tail_get(path, **params):
    value_key = {
        "/futures/data/openInterestHist": "sumOpenInterest",
    }[path]
    return [
        {"timestamp": timestamp, value_key: str(index)}
        for index, timestamp in enumerate(timestamps[:-1])
        if params["startTime"] <= timestamp <= params["endTime"]
    ]

datafeed._get = stale_tail_get
stale_tail = datafeed.fetch_oi(
    "BTCUSDT", "1h", start_time=0, end_time=end_time
)
assert len(stale_tail) == 700
assert stale_tail[-1][0] == end_time
assert np.isnan(stale_tail[-1][1])

short_calls = []
def empty_then_data(path, **params):
    short_calls.append(dict(params))
    if len(short_calls) == 1:
        return []
    return [{"timestamp": timestamps[-1], "sumOpenInterest": "42"}]

datafeed._get = empty_then_data
late_only = datafeed.fetch_oi(
    "BTCUSDT", "1h", start_time=0, end_time=end_time
)
assert len(short_calls) == 2
assert late_only == [(end_time, 42.0)]

missing_tail = datafeed._series_refresh_result(
    [(end_time - 2 * hour, 42.0), (end_time - hour, np.nan)],
    end_time,
    2 * hour,
)
assert missing_tail["status"] == "missing_tail"

def error_get(_path, **_params):
    if _params["startTime"] == 0:
        return [{"timestamp": 0, "sumOpenInterest": "1"}]
    return {"code": -1130, "msg": "invalid window"}

datafeed._get = error_get
try:
    datafeed.fetch_oi("BTCUSDT", "1h", start_time=0, end_time=end_time)
except RuntimeError as error:
    assert "-1130" in str(error)
else:
    raise AssertionError("a later-page API error must reject the complete series")

with tempfile.TemporaryDirectory() as cache_dir:
    datafeed.CACHE_DIR = cache_dir
    day = 86_400_000
    now = 40 * day
    bars = pd.DataFrame({
        "openTime": [0, now - 2 * hour, now - hour],
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1.0, 2.0, 3.0],
    })
    old = bars.copy()
    old["funding"] = [0.01, 0.02, 0.03]
    old["oi"] = [10.0, 20.0, 30.0]
    old["basis"] = [0.1, 0.2, 0.3]
    old["taker"] = [1.0, 1.1, 1.2]
    old.to_csv(datafeed._cache_path("BTCUSDT", "1h"), index=False)

    snapshot_end = now - 1
    expected_start = now - datafeed.STATS_RETENTION_MS + datafeed.STATS_RETENTION_SAFETY_MS
    observed_ends = []
    observed_starts = []
    datafeed.fetch_klines = lambda *_args, **_kwargs: bars.copy()
    def fail_funding(_sym, *, end_time):
        observed_ends.append(end_time)
        raise RuntimeError("funding page failed")
    def partial_oi(_sym, _period, *, start_time, end_time):
        observed_starts.append(start_time)
        observed_ends.append(end_time)
        return [(0, 100.0), (now - 2 * hour, np.nan), (now - hour, 300.0)]
    datafeed.fetch_funding = fail_funding
    datafeed.fetch_oi = partial_oi
    datafeed.fetch_basis = lambda _sym, _period, *, start_time, end_time: (
        observed_starts.append(start_time)
        or observed_ends.append(end_time)
        or [(0, 0.6)]
    )
    datafeed.fetch_taker = lambda _sym, _period, *, start_time, end_time: (
        observed_starts.append(start_time)
        or observed_ends.append(end_time)
        or [(0, 1.6)]
    )
    datafeed.time.time = lambda: now / 1000
    outcome = datafeed.update_cache(["BTCUSDT"], "1h", 3)

    refreshed = pd.read_csv(datafeed._cache_path("BTCUSDT", "1h"))
    assert refreshed["openTime"].tolist() == [0, now - 2 * hour, now - hour]
    assert refreshed["oi"].tolist() == [100.0, 20.0, 300.0]
    assert refreshed["funding"].tolist() == [0.01, 0.02, 0.03]
    assert refreshed["basis"].tolist() == [0.6, 0.2, 0.3]
    assert refreshed["taker"].tolist() == [1.6, 1.1, 1.2]
    assert observed_ends == [snapshot_end] * 4
    assert observed_starts == [expected_start, expected_start, expected_start]
    assert outcome["BTCUSDT"]["status"] == "updated"
    assert outcome["BTCUSDT"]["series"]["funding"]["status"] == "error"
    assert outcome["BTCUSDT"]["series"]["oi"]["status"] == "ok"
    assert outcome["BTCUSDT"]["series"]["basis"]["status"] == "stale"

    datafeed.fetch_klines = lambda *_args, **_kwargs: pd.DataFrame()
    empty_outcome = datafeed.update_cache(["ETHUSDT"], "1h", 3)
    assert empty_outcome["ETHUSDT"]["status"] == "no_klines"

    stale_bars = bars.iloc[[0, 1, 2]].copy()
    stale_bars["openTime"] = [now - 20 * hour, now - 10 * hour, now - hour]
    stale_funding_time = now - 19 * hour - 1
    datafeed.fetch_klines = lambda *_args, **_kwargs: stale_bars.copy()
    datafeed.fetch_funding = lambda _sym, *, end_time: [(stale_funding_time, 0.01)]
    datafeed.fetch_oi = lambda _sym, _period, *, start_time, end_time: [(end_time, 1.0)]
    datafeed.fetch_basis = lambda _sym, _period, *, start_time, end_time: [(end_time, 0.1)]
    datafeed.fetch_taker = lambda _sym, _period, *, start_time, end_time: [(end_time, 1.1)]
    stale_outcome = datafeed.update_cache(["ETHUSDT"], "1h", 3)
    stale_cache = pd.read_csv(datafeed._cache_path("ETHUSDT", "1h"))
    assert stale_outcome["ETHUSDT"]["series"]["funding"]["status"] == "stale"
    assert np.isfinite(stale_cache["funding"].iloc[0])
    assert stale_cache["funding"].iloc[1:].isna().all()

    atomic_path = os.path.join(cache_dir, "atomic.csv")
    old_bytes = b"openTime,close\n1,10\n"
    with open(atomic_path, "wb") as handle:
        handle.write(old_bytes)
    replacement = pd.DataFrame({"openTime": [1, 2], "close": [11.0, 12.0]})
    original_replace = datafeed.os.replace
    try:
        def fail_replace(_source, _target):
            raise OSError("simulated interrupted replace")
        datafeed.os.replace = fail_replace
        try:
            datafeed.write_cache_atomic(replacement, atomic_path)
        except OSError as error:
            assert "simulated" in str(error)
        else:
            raise AssertionError("failed replacement must propagate")
    finally:
        datafeed.os.replace = original_replace
    with open(atomic_path, "rb") as handle:
        assert handle.read() == old_bytes
    assert not [name for name in os.listdir(cache_dir) if name.endswith(".tmp")]

    datafeed.write_cache_atomic(replacement, atomic_path)
    atomically_written = pd.read_csv(atomic_path)
    assert atomically_written["openTime"].tolist() == [1, 2]
    assert atomically_written["close"].tolist() == [11.0, 12.0]

with tempfile.TemporaryDirectory() as read_only_cache:
    datafeed.CACHE_DIR = read_only_cache
    read_only_frame = pd.DataFrame({
        "openTime": [1, 2],
        "close": [10.0, 11.0],
    })
    read_only_frame.to_csv(datafeed._cache_path("BTCUSDT", "1h"), index=False)
    os.chmod(read_only_cache, 0o555)
    try:
        immutable_panel = datafeed.load_panel(["BTCUSDT"], "1h", refresh=False)
        assert immutable_panel["BTCUSDT"]["close"].tolist() == [10.0, 11.0]
        assert not os.path.exists(os.path.join(read_only_cache, ".collector"))
    finally:
        os.chmod(read_only_cache, 0o755)
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
      timeout: 30_000,
    });
    assert.equal(run.status, 0, run.stderr || run.stdout);
  },
);
