import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repositoryRoot = fileURLToPath(new URL("..", import.meta.url));
const researchDir = fileURLToPath(
  new URL("../scripts/research/", import.meta.url),
);
const fixtureDir = fileURLToPath(
  new URL("./fixtures/market-context-source-v1/", import.meta.url),
);

function runPython(program) {
  return spawnSync("python3", ["-", researchDir, fixtureDir], {
    cwd: repositoryRoot,
    encoding: "utf8",
    input: program,
  });
}

const pythonFixtureHelpers = String.raw`
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.parse

sys.path.insert(0, sys.argv[1])
fixture = Path(sys.argv[2])
import collect_market_context as collector
import market_context_source as source

collector._repository_commit = lambda: "a" * 40
collector._provenance_tracked_clean = lambda: True

class Headers(dict):
    def get_content_type(self):
        raw = self.get("Content-Type")
        return raw.split(";", 1)[0].strip().lower() if raw else None

class Response:
    status = 200

    def __init__(self, payload, used_weight):
        self.payload = payload
        self.headers = Headers({
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(payload)),
            "X-MBX-USED-WEIGHT-1M": str(used_weight),
        })

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size=-1):
        return self.payload if size < 0 else self.payload[:size]

class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

class Clock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("fixture epoch clock exhausted")
        return self.values.pop(0)

def raw(name):
    return (fixture / "raw" / name).read_bytes()

def successful_responses(btc_payload=None):
    payloads = [
        raw("server-time-before.json"),
        raw("exchange-info.json"),
        raw("ticker-24hr.json"),
        btc_payload if btc_payload is not None else raw("BTCUSDT-klines.json"),
        raw("ETHUSDT-klines.json"),
        raw("DOGEUSDT-klines.json"),
        raw("server-time-after.json"),
    ]
    return [Response(payload, weight) for payload, weight in zip(
        payloads, [1, 2, 42, 43, 44, 45, 46]
    )]

success_clock = [
    10800090,
    10800100, 10800120,
    10800130, 10800140,
    10800150, 10800160,
    10800170, 10800180,
    10800190, 10800200,
    10800210, 10800220,
    10800230, 10800240,
    10800250, 10800260, 10800270, 10800280, 10800290,
]

def collect_at(output, responses, clock):
    collector.REGISTERED_COLLECTION_START_MS = 0
    opener = Opener(responses)
    collector.URL_OPENER = opener
    collector._epoch_ms = Clock(clock)
    manifest = collector.collect_bundle(
        output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
    return manifest, opener
`;

test("public market-context collector emits a deterministic independently verified bundle", () => {
  const result = runPython(`${pythonFixtureHelpers}
durable_directories = []
original_fsync_directory = collector._fsync_directory
original_write_status = collector._write_status

def record_fsync(path):
    original_fsync_directory(path)
    durable_directories.append(Path(path))

def require_publication_barriers(path, value, **kwargs):
    if value.get("state") == "complete_unverified":
        assert durable_directories[-1] == path.parent
        assert path.parent / "raw" in durable_directories
    original_write_status(path, value, **kwargs)

collector._fsync_directory = record_fsync
collector._write_status = require_publication_barriers
root = Path(tempfile.mkdtemp(prefix="trader-market-context-collector-"))
first_manifest, first_opener = collect_at(
    root / "first", successful_responses(), success_clock
)
second_manifest, second_opener = collect_at(
    root / "second", successful_responses(), success_clock
)

assert first_manifest.read_bytes() == second_manifest.read_bytes()
first_raw = sorted((first_manifest.parent / "raw").iterdir())
second_raw = sorted((second_manifest.parent / "raw").iterdir())
assert [path.name for path in first_raw] == [path.name for path in second_raw]
assert [path.read_bytes() for path in first_raw] == [path.read_bytes() for path in second_raw]

manifest = json.loads(first_manifest.read_text())
status = json.loads((first_manifest.parent / "collection-status.json").read_text())
assert manifest["schemaId"] == "binance_usdm_market_context_source_manifest_v1"
assert manifest["population"]["eligibleSymbols"] == [
    "BTCUSDT", "ETHUSDT", "DOGEUSDT"
]
assert manifest["decisionTime"] == 10800260
assert manifest["startedAtMs"] == 10800100
assert manifest["completedAtMs"] == 10800240
assert status["state"] == "complete_unverified"
assert status["sourceManifestPublished"] is True
assert status["eligiblePopulationCount"] == 3
assert status["provenanceTrackedClean"] is True
assert len(status["collectorSha256"]) == 64
assert len(status["verifierSha256"]) == 64
assert status["runtime"]["python"]
assert status["outcomeUse"] is False
assert status["modelUse"] is False
assert status["orderUse"] is False
assert status["liveAuthorizationUse"] is False

panel, receipt = source.verify_and_derive(first_manifest)
assert receipt["panelRows"] == 3
assert receipt["sourceManifestCodeCommit"] == "a" * 40
assert receipt["outcomeUse"] is False
assert len(panel.decode().splitlines()) == 4

expected_paths = [
    "/fapi/v1/time",
    "/fapi/v1/exchangeInfo",
    "/fapi/v1/ticker/24hr",
    "/fapi/v1/klines",
    "/fapi/v1/klines",
    "/fapi/v1/klines",
    "/fapi/v1/time",
]
requests = first_opener.requests
assert len(requests) == len(expected_paths)
for (request, timeout), expected_path in zip(requests, expected_paths):
    parsed = urllib.parse.urlparse(request.full_url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "fapi.binance.com"
    assert parsed.path == expected_path
    assert request.get_method() == "GET"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers == {
        "accept": "application/json",
        "user-agent": "trader-market-context-collector/1",
    }
    assert 0 < timeout <= collector.REQUEST_TIMEOUT_SECONDS

kline_query = urllib.parse.parse_qs(
    requests[3][0].full_url.split("?", 1)[1]
)
assert kline_query == {
    "symbol": ["BTCUSDT"],
    "interval": ["1h"],
    "startTime": ["3600000"],
    "endTime": ["10799999"],
    "limit": ["2"],
}
`);
  assert.equal(result.status, 0, result.stderr);
});

test("public market-context collector preserves fail-closed partial evidence", () => {
  const result = runPython(`${pythonFixtureHelpers}
root = Path(tempfile.mkdtemp(prefix="trader-market-context-failures-"))
collector.REGISTERED_COLLECTION_START_MS = 0

existing = root / "existing"
existing.mkdir()
opener = Opener([])
collector.URL_OPENER = opener
collector._epoch_ms = Clock([10800090])
try:
    collector.collect_bundle(
        existing,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "output_exists"
else:
    raise AssertionError("existing output must be rejected")
assert opener.requests == []

missing_weight = Response(raw("server-time-before.json"), 1)
del missing_weight.headers["X-MBX-USED-WEIGHT-1M"]
missing_output = root / "missing-weight"
collector.URL_OPENER = Opener([missing_weight])
collector._epoch_ms = Clock([10800090, 10800100, 10800110])
try:
    collector.collect_bundle(
        missing_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "response_invalid"
else:
    raise AssertionError("missing shared-IP weight must fail")
assert not (missing_output / "source-manifest.json").exists()
missing_status = json.loads((missing_output / "collection-status.json").read_text())
assert missing_status["state"] == "partial_failure"
assert missing_status["sourceManifestPublished"] is False
assert missing_status["completedArtifactPaths"] == []
assert all(missing_status[name] is False for name in (
    "outcomeUse", "modelUse", "orderUse", "liveAuthorizationUse"
))

rate_body = json.dumps({
    "code": -1003,
    "msg": "Way too many requests; IP(10.0.0.9) banned until 1234567890123.",
}).encode()
rate_error = urllib.error.HTTPError(
    "https://fapi.binance.com/fapi/v1/time",
    429,
    "rate limited",
    Headers({"Retry-After": "17", "X-MBX-USED-WEIGHT-1M": "99"}),
    BytesIO(rate_body),
)
rate_output = root / "rate-limit"
rate_opener = Opener([rate_error])
collector.URL_OPENER = rate_opener
collector._epoch_ms = Clock([10800090, 10800100, 10800110])
try:
    collector.collect_bundle(
        rate_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "provider_rate_limit"
else:
    raise AssertionError("HTTP 429 must open the collection circuit")
assert len(rate_opener.requests) == 1
rate_status = json.loads((rate_output / "collection-status.json").read_text())
assert rate_status["httpStatus"] == 429
assert rate_status["retryAfterSeconds"] == 17
assert rate_status["bannedUntilMs"] == 1234567890123
assert "10.0.0.9" not in json.dumps(rate_status)
assert not (rate_output / "source-manifest.json").exists()

invalid_klines = json.loads(raw("BTCUSDT-klines.json"))[:1]
invalid_bytes = (json.dumps(invalid_klines) + "\\n").encode()
invalid_output = root / "invalid-klines"
collector.URL_OPENER = Opener(successful_responses(invalid_bytes)[:4])
collector._epoch_ms = Clock([
    10800090,
    10800100, 10800120,
    10800130, 10800140,
    10800150, 10800160,
    10800170, 10800180,
    10800190,
])
try:
    collector.collect_bundle(
        invalid_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "response_incomplete"
else:
    raise AssertionError("partial kline response must fail")
assert not (invalid_output / "source-manifest.json").exists()
invalid_status = json.loads((invalid_output / "collection-status.json").read_text())
assert invalid_status["state"] == "partial_failure"
assert invalid_status["failedEndpoint"] == "/fapi/v1/klines"
assert invalid_status["completedArtifactPaths"][-1].endswith("BTCUSDT-klines.json")

status_failure_output = root / "status-failure"
collector.URL_OPENER = Opener(successful_responses())
collector._epoch_ms = Clock(success_clock)
original_write_status = collector._write_status

def fail_completion_status(path, value, **kwargs):
    if value.get("state") == "complete_unverified":
        raise OSError("injected completion-status failure")
    original_write_status(path, value, **kwargs)

collector._write_status = fail_completion_status
try:
    collector.collect_bundle(
        status_failure_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except OSError:
    pass
else:
    raise AssertionError("completion-status failure must fail the collection")
assert not (status_failure_output / "source-manifest.json").exists()
status_failure = json.loads(
    (status_failure_output / "collection-status.json").read_text()
)
assert status_failure["state"] == "partial_failure"
assert status_failure["failureKind"] == "collector_internal_failure"

cleanup_failure_output = root / "cleanup-failure"
collector.URL_OPENER = Opener(successful_responses())
collector._epoch_ms = Clock(success_clock)
original_remove_manifest = collector._remove_manifest

def fail_manifest_cleanup(_manifest_path, _output_dir):
    raise OSError("injected manifest cleanup failure")

collector._write_status = fail_completion_status
collector._remove_manifest = fail_manifest_cleanup
try:
    collector.collect_bundle(
        cleanup_failure_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "manifest_cleanup_failed"
else:
    raise AssertionError("failed manifest cleanup must be explicit")
assert (cleanup_failure_output / "source-manifest.json").is_file()
cleanup_status = json.loads(
    (cleanup_failure_output / "collection-status.json").read_text()
)
assert cleanup_status["state"] == "cleanup_failure"
assert cleanup_status["failureKind"] == "manifest_cleanup_failed"
assert cleanup_status["originalFailureKind"] == "collector_internal_failure"
assert cleanup_status["sourceManifestPublished"] is None
assert cleanup_status["sourceManifestState"] == "indeterminate_after_cleanup_failure"
assert cleanup_status["manifestCleanupRequired"] is True
assert "injected" not in json.dumps(cleanup_status)
collector._remove_manifest = original_remove_manifest

deadline_output = root / "publication-deadline"
collector.URL_OPENER = Opener(successful_responses())
collector._epoch_ms = Clock(success_clock)
collector._write_status = original_write_status
original_publication_window = collector._require_publication_window
publication_checks = 0

def expire_final_publication(_deadline, _latest_safe_completion):
    global publication_checks
    publication_checks += 1
    if publication_checks == 2:
        raise collector.CollectionFailure(
            "deadline_exceeded", "injected final publication deadline"
        )

collector._require_publication_window = expire_final_publication
try:
    collector.collect_bundle(
        deadline_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "deadline_exceeded"
else:
    raise AssertionError("expired final publication must fail")
assert publication_checks == 2
assert not (deadline_output / "source-manifest.json").exists()
deadline_status = json.loads(
    (deadline_output / "collection-status.json").read_text()
)
assert deadline_status["state"] == "partial_failure"
assert deadline_status["failureKind"] == "deadline_exceeded"
collector._require_publication_window = original_publication_window

abrupt_output = root / "abrupt-stop"
collector.URL_OPENER = Opener(successful_responses())
collector._epoch_ms = Clock(success_clock)

class AbruptStop(BaseException):
    pass

def interrupt_completion_status(path, value, **kwargs):
    if value.get("state") == "complete_unverified":
        raise AbruptStop()
    original_write_status(path, value, **kwargs)

collector._write_status = interrupt_completion_status
try:
    collector.collect_bundle(
        abrupt_output,
        quote="USDT",
        interval="1h",
        bar_open=7200000,
        max_clock_skew_ms=0,
        deadline_seconds=240,
        limiter=collector.RequestWeightLimiter(),
    )
except AbruptStop:
    pass
else:
    raise AssertionError("abrupt completion stop must escape the exception handler")
assert (abrupt_output / "source-manifest.json").is_file()
abrupt_status = json.loads(
    (abrupt_output / "collection-status.json").read_text()
)
assert abrupt_status["state"] == "collecting"
assert abrupt_status["sourceManifestPublished"] is False
`);
  assert.equal(result.status, 0, result.stderr);
});

test("collector CLI requires committed provenance before any public request", async () => {
  const source = await readFile(
    new URL("../scripts/research/collect_market_context.py", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(
    source,
    /BINANCE_API_KEY|BINANCE_API_SECRET|X-MBX-APIKEY|["']Authorization["']|\/fapi\/v1\/order/,
  );
  assert.match(source, /NoRedirectHandler/);
  assert.match(source, /_fsync_directory\(raw_dir\)/);
  assert.match(source, /_fsync_directory\(path\.parent\)/);
  assert.equal(
    source.match(/before_replace=lambda: _require_publication_window/g)?.length,
    2,
  );
  assert.match(source, /_provenance_tracked_clean/);
  assert.match(source, /The separate market_context_source\.py/);
  assert.doesNotMatch(source, /verify_and_derive\s*\(/);

  const result = runPython(`${pythonFixtureHelpers}
root = Path(tempfile.mkdtemp(prefix="trader-market-context-provenance-"))
collector._repository_commit = lambda: "a" * 40
collector._provenance_tracked_clean = lambda: False
opener = Opener([])
collector.URL_OPENER = opener
result = collector.main([
    "collect",
    "--output-dir", str(root / "blocked"),
    "--interval", "1h",
    "--bar-open-time", "7200000",
    "--max-clock-skew-ms", "0",
])
assert result == 1
assert opener.requests == []
assert not (root / "blocked").exists()

collector.REGISTERED_COLLECTION_START_MS = 1800489600000
try:
    collector._validate_inputs(
        "USDT", "1h", 1800486000000, 0, "a" * 40, 240
    )
except collector.CollectionFailure as error:
    assert error.failure_kind == "input_invalid"
else:
    raise AssertionError("pre-registration collection must fail")
assert collector._validate_inputs(
    "USDT", "8h", 1800489600000, 0, "a" * 40, 240
)[0] == 28800000

collector.time.monotonic = lambda: 10.0
collector._epoch_ms = lambda: 20
try:
    collector._require_publication_window(10.0, 30)
except collector.CollectionFailure as error:
    assert error.failure_kind == "deadline_exceeded"
else:
    raise AssertionError("expired monotonic deadline must block publication")
collector.time.monotonic = lambda: 9.0
try:
    collector._require_publication_window(10.0, 20)
except collector.CollectionFailure as error:
    assert error.failure_kind == "deadline_exceeded"
else:
    raise AssertionError("expired causal window must block publication")
`);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stderr, /provenance_invalid/);
});
