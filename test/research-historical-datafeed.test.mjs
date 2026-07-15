import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));

test("historical Binance snapshots are bounded, complete, and atomic", () => {
  const program = String.raw`
import json
import inspect
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
import historical_datafeed as feed

assert inspect.signature(feed.fetch_contract_klines).parameters["page_limit"].default == 499
assert inspect.signature(feed.fetch_mark_price_klines).parameters["page_limit"].default == 499
assert feed._request_weight("/fapi/v1/continuousKlines", {"limit": 99}) == 1
assert feed._request_weight("/fapi/v1/markPriceKlines", {"limit": 499}) == 2
assert feed._request_weight("/fapi/v1/markPriceKlines", {"limit": 1000}) == 5
assert feed._request_weight("/fapi/v1/markPriceKlines", {"limit": 1500}) == 10
assert feed._request_weight("/fapi/v1/fundingRate", {"limit": 1000}) == 1
assert feed.FUNDING_REQUEST_BUDGET == 450
assert feed.FUNDING_REQUEST_WINDOW_SECONDS == 300.0

clock = [0.0]
sleeps = []
original_monotonic = feed.time.monotonic
original_sleep = feed.time.sleep
try:
    feed.time.monotonic = lambda: clock[0]
    def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    feed.time.sleep = advance
    limiter = feed.RequestWeightLimiter(3, 60.0)
    limiter.wait(2)
    limiter.observe_used_weight(3)
    limiter.wait(1)
    funding_limiter = feed.RequestWeightLimiter(2, 300.0)
    funding_limiter.wait(1)
    funding_limiter.wait(1)
    funding_limiter.wait(1)
finally:
    feed.time.monotonic = original_monotonic
    feed.time.sleep = original_sleep
assert sleeps == [60.0, 300.0]

http_delays = []
original_urlopen = feed.urllib.request.urlopen
original_bounded_sleep = feed._bounded_sleep
try:
    feed._bounded_sleep = lambda delay, deadline: http_delays.append((delay, deadline))
    for status in (418, 429):
        attempts = []
        def rate_limited(*_args, **_kwargs):
            attempts.append(status)
            raise feed.urllib.error.HTTPError(
                "https://example.invalid",
                status,
                "rate limited",
                {"Retry-After": "2", "X-MBX-USED-WEIGHT-1M": "17"},
                None,
            )
        feed.urllib.request.urlopen = rate_limited
        try:
            feed._get("/fapi/v1/fundingRate", tries=2, symbol="BTCUSDT")
        except feed.urllib.error.HTTPError as error:
            assert error.code == status
        else:
            raise AssertionError("bounded HTTP retries must propagate the final error")
        assert attempts == [status, status]
finally:
    feed.urllib.request.urlopen = original_urlopen
    feed._bounded_sleep = original_bounded_sleep

late_clock = [0.0]
original_monotonic = feed.time.monotonic
original_rate_limiter = feed.RATE_LIMITER

class LateResponse:
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        late_clock[0] = 2.0
        return b"[]"

class OnTimeResponse(LateResponse):
    headers = {"X-MBX-USED-WEIGHT-1M": "23"}

    def read(self):
        return b"[]"

class RecordingLimiter:
    def __init__(self):
        self.calls = []

    def wait(self, weight, deadline=None):
        self.calls.append((weight, deadline))

try:
    feed.time.monotonic = lambda: late_clock[0]
    feed.RATE_LIMITER = feed.RequestWeightLimiter(100, 60.0)
    feed.urllib.request.urlopen = lambda *_args, **_kwargs: OnTimeResponse()
    assert feed._get(
        "/fapi/v1/continuousKlines", tries=1, limit=1
    ) == []
    assert feed.RATE_LIMITER._used == 23
    feed.urllib.request.urlopen = lambda *_args, **_kwargs: LateResponse()
    try:
        feed._get(
            "/fapi/v1/continuousKlines",
            tries=1,
            deadline=1.0,
            limit=1,
        )
    except TimeoutError as error:
        assert "deadline" in str(error)
    else:
        raise AssertionError("a response body completed after deadline must be discarded")
finally:
    feed.time.monotonic = original_monotonic
    feed.RATE_LIMITER = original_rate_limiter
    feed.urllib.request.urlopen = original_urlopen

original_funding_limiter = feed.FUNDING_RATE_LIMITER
original_rate_limiter = feed.RATE_LIMITER
recording_funding_limiter = RecordingLimiter()
try:
    feed.FUNDING_RATE_LIMITER = recording_funding_limiter
    feed.RATE_LIMITER = feed.RequestWeightLimiter(100, 60.0)
    feed.urllib.request.urlopen = lambda *_args, **_kwargs: OnTimeResponse()
    assert feed._get(
        "/fapi/v1/fundingRate", tries=1, symbol="BTCUSDT"
    ) == []
finally:
    feed.FUNDING_RATE_LIMITER = original_funding_limiter
    feed.RATE_LIMITER = original_rate_limiter
    feed.urllib.request.urlopen = original_urlopen
assert recording_funding_limiter.calls == [(1, None)]
assert http_delays == [(2.0, None), (2.0, None)]
assert feed.RATE_LIMITER._used >= 17

def deadline_limited(*_args, **_kwargs):
    raise feed.urllib.error.HTTPError(
        "https://example.invalid",
        429,
        "rate limited",
        {"Retry-After": "2"},
        None,
    )

feed.urllib.request.urlopen = deadline_limited
try:
    try:
        feed._get(
            "/fapi/v1/fundingRate",
            tries=4,
            deadline=feed.time.monotonic() + 0.5,
            symbol="BTCUSDT",
        )
    except TimeoutError as error:
        assert "deadline" in str(error)
    else:
        raise AssertionError("Retry-After must not overrun the acquisition deadline")
finally:
    feed.urllib.request.urlopen = original_urlopen

eight_hours = feed.CONTRACT_INTERVAL_MS
hour = feed.MARK_PRICE_INTERVAL_MS

def contract_row(timestamp, close):
    return [
        timestamp, str(close - 1), str(close + 1), str(close - 2), str(close),
        "10", timestamp + eight_hours - 1, "20", 3, "4", "5", "0",
    ]

def mark_row(timestamp, close):
    return [
        timestamp, str(close - 1), str(close + 1), str(close - 2), str(close),
        "0", timestamp + hour - 1, "0", 0, "0", "0", "0",
    ]

contract_rows = [contract_row(index * eight_hours, 100 + index) for index in range(5)]
mark_rows = [mark_row(index * hour, 200 + index) for index in range(5)]
funding_rows = [
    {
        "symbol": "BTCUSDT",
        "fundingTime": 100 + index * 100,
        "fundingRate": f"0.000{index + 1}",
        "markPrice": f"{30000 + index}.12345678",
    }
    for index in range(5)
]
funding_rows[0]["markPrice"] = ""
calls = []

def page(rows, timestamp, params):
    available = [row for row in rows if params["startTime"] <= timestamp(row) <= params["endTime"]]
    return available[:params["limit"]]

def fake_get(path, **params):
    calls.append((path, dict(params)))
    if path == "/fapi/v1/continuousKlines":
        return page(contract_rows, lambda row: row[0], params)
    if path == "/fapi/v1/markPriceKlines":
        return page(mark_rows, lambda row: row[0], params)
    if path == "/fapi/v1/fundingRate":
        return page(funding_rows, lambda row: row["fundingTime"], params)
    raise AssertionError(path)

feed._get = fake_get
contracts = feed.fetch_contract_klines(
    "BTCUSDT", 0, contract_rows[-1][0], page_limit=2
)
marks = feed.fetch_mark_price_klines(
    "BTCUSDT", 0, mark_rows[-1][0], page_limit=2
)
funding = feed.fetch_funding_events("BTCUSDT", 100, 550, page_limit=2)

assert [row["openTime"] for row in contracts] == [row[0] for row in contract_rows]
assert [row["openTime"] for row in marks] == [row[0] for row in mark_rows]
assert marks[0]["open"] == "199"
assert marks[0]["high"] == "201"
assert marks[0]["low"] == "198"
assert marks[0]["close"] == "200"
assert [row["fundingTime"] for row in funding] == [100, 200, 300, 400, 500]
assert funding[0] == {
    "fundingTime": 100,
    "fundingRate": "0.0001",
    "markPrice": "",
    "symbol": "BTCUSDT",
}

contract_calls = [params for path, params in calls if path == "/fapi/v1/continuousKlines"]
mark_calls = [params for path, params in calls if path == "/fapi/v1/markPriceKlines"]
funding_calls = [params for path, params in calls if path == "/fapi/v1/fundingRate"]
assert [params["startTime"] for params in contract_calls] == [
    0, eight_hours + 1, 3 * eight_hours + 1,
]
assert [params["startTime"] for params in mark_calls] == [0, hour + 1, 3 * hour + 1]
assert [params["startTime"] for params in funding_calls] == [100, 201, 401, 501]
assert all(params["endTime"] == contract_rows[-1][0] for params in contract_calls)
assert all(params["endTime"] == mark_rows[-1][0] for params in mark_calls)
assert all(params["endTime"] == 550 for params in funding_calls)
assert all(params["interval"] == "8h" for params in contract_calls)
assert all(params["contractType"] == "PERPETUAL" for params in contract_calls)
assert all(params["interval"] == "1h" for params in mark_calls)

def repeated_page(_path, **_params):
    return [funding_rows[0]]

feed._get = repeated_page
try:
    feed.fetch_funding_events("BTCUSDT", 100, 500, page_limit=2)
except feed.SnapshotIntegrityError as error:
    assert "outside the requested page" in str(error)
else:
    raise AssertionError("a repeated page must fail rather than loop or truncate")

def duplicate_page(_path, **_params):
    return [funding_rows[0], funding_rows[0]]

feed._get = duplicate_page
try:
    feed.fetch_funding_events("BTCUSDT", 100, 500, page_limit=2)
except feed.SnapshotIntegrityError as error:
    assert "duplicate or unordered" in str(error)
else:
    raise AssertionError("a duplicate response row must fail closed")

def unordered_page(_path, **_params):
    return [funding_rows[1], funding_rows[0]]

feed._get = unordered_page
try:
    feed.fetch_funding_events("BTCUSDT", 100, 500, page_limit=2)
except feed.SnapshotIntegrityError as error:
    assert "duplicate or unordered" in str(error)
else:
    raise AssertionError("an unordered response page must fail closed")

def gap_page(_path, **params):
    rows = [contract_rows[0], contract_rows[2]]
    return [row for row in rows if params["startTime"] <= row[0] <= params["endTime"]]

feed._get = gap_page
try:
    feed.fetch_contract_klines("BTCUSDT", 0, 2 * eight_hours, page_limit=2)
except feed.SnapshotIntegrityError as error:
    assert "gap" in str(error)
else:
    raise AssertionError("a missing kline interval must invalidate the snapshot")

def truncated_page(_path, **params):
    rows = mark_rows[:2]
    return [row for row in rows if params["startTime"] <= row[0] <= params["endTime"]]

feed._get = truncated_page
try:
    feed.fetch_mark_price_klines("BTCUSDT", 0, 2 * hour, page_limit=2)
except feed.SnapshotIntegrityError as error:
    assert "ends" in str(error)
else:
    raise AssertionError("missing trailing coverage must invalidate the snapshot")

partial_calls = []
def partial_then_error(path, **params):
    partial_calls.append(dict(params))
    if len(partial_calls) == 1:
        return [funding_rows[0]]
    return {"code": -1000, "msg": "later page failed"}

feed._get = partial_then_error
try:
    feed.fetch_funding_events("BTCUSDT", 100, 500, page_limit=2)
except RuntimeError as error:
    assert "-1000" in str(error)
else:
    raise AssertionError("a short first page must not hide a later-page error")
assert len(partial_calls) == 2
assert partial_calls[1]["startTime"] == 101
assert partial_calls[1]["endTime"] == 500

page_bound_calls = []
def one_event_per_page(_path, **params):
    page_bound_calls.append(dict(params))
    timestamp = params["startTime"]
    return [{
        "symbol": "BTCUSDT",
        "fundingTime": timestamp,
        "fundingRate": "0.0001",
        "markPrice": "100",
    }]

feed._get = one_event_per_page
original_page_bound = feed.MAX_PAGES_PER_SERIES
try:
    feed.MAX_PAGES_PER_SERIES = 2
    try:
        feed.fetch_funding_events("BTCUSDT", 100, 500, page_limit=1)
    except feed.SnapshotIntegrityError as error:
        assert "exceeded 2 pages" in str(error)
    else:
        raise AssertionError("a pathological response cannot exceed the page bound")
finally:
    feed.MAX_PAGES_PER_SERIES = original_page_bound
assert len(page_bound_calls) == 2

try:
    feed.fetch_funding_events(
        "BTCUSDT", 100, 500, page_limit=1, deadline=feed.time.monotonic() - 1
    )
except TimeoutError as error:
    assert "deadline" in str(error)
else:
    raise AssertionError("an expired pagination deadline must fail before requesting")

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "snapshot.json")
    old_bytes = b'{"state":"old"}\n'
    with open(path, "wb") as handle:
        handle.write(old_bytes)

    original_replace = feed.os.replace
    try:
        def fail_replace(_source, _destination):
            raise OSError("simulated interrupted replacement")
        feed.os.replace = fail_replace
        try:
            feed.write_artifact_atomic({"state": "new"}, path)
        except OSError as error:
            assert "interrupted" in str(error)
        else:
            raise AssertionError("failed atomic replacement must propagate")
    finally:
        feed.os.replace = original_replace

    with open(path, "rb") as handle:
        assert handle.read() == old_bytes
    assert not [name for name in os.listdir(directory) if name.endswith(".tmp")]

    feed.write_artifact_atomic({"rows": funding}, path)
    with open(path, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written == {"rows": funding}
`;

  const result = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
