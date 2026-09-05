import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const RESEARCH_DIR = path.join(ROOT, "scripts", "research");
const COLLECTOR = path.join(RESEARCH_DIR, "collect_datafeed.py");
const RECEIPT_VERIFIER = path.join(RESEARCH_DIR, "verify_derivatives_receipt.py");
const INSTALLER = path.join(ROOT, "scripts", "install-research-datafeed-launchagent.sh");
const pythonProbe = spawnSync("python3", ["-c", "import sys, numpy, pandas; print(sys.executable)"], {
  encoding: "utf8",
});
const hasResearchPython = pythonProbe.status === 0;
const pythonPath = pythonProbe.stdout.trim();

test(
  "scheduled research collector serializes writers and records partial failure",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, sys.argv[1])
import collect_datafeed as collector
import verify_derivatives_receipt as receipt_verifier

try:
    os.environ["TRADER_RESEARCH_SYMBOLS"] = "btcusdt"
    collector._symbols_from_environment()
except ValueError as error:
    assert "uppercase USDT" in str(error)
else:
    raise AssertionError("invalid scheduled symbols must fail closed")

with tempfile.TemporaryDirectory() as cache_name:
    cache = Path(cache_name)
    os.environ["TRADER_RESEARCH_CACHE"] = str(cache)
    os.environ["TRADER_RESEARCH_SYMBOLS"] = "BTCUSDT ETHUSDT SOLUSDT"
    os.environ["TRADER_RESEARCH_MAX_RUN_SECONDS"] = "30"
    calls = []
    interval_ms = collector.feed.INTERVAL_MS["1h"]
    latest = int(time.time() * 1000) // interval_ms * interval_ms - interval_ms

    def fake_update(symbols, interval, *, acquire_lock):
        symbol = symbols[0]
        calls.append(symbol)
        assert interval == "1h"
        assert acquire_lock is False
        if symbol == "ETHUSDT":
            raise RuntimeError("simulated public endpoint failure")
        frame = pd.DataFrame({
            "openTime": [latest - 2 * interval_ms, latest - interval_ms, latest],
            "close": [10.0, 11.0, 12.0],
            "funding": [0.1, 0.1, 0.1],
            "oi": [100.0, 101.0, 102.0],
            "basis": [0.01, 0.02, 0.03],
            "taker": [1.0, 1.1, 1.2],
        })
        for field in ("funding", "oi", "basis", "taker"):
            frame[f"{field}V2Value"] = frame[field]
            frame[f"{field}V2Observed"] = 1
            frame[f"{field}V2Fresh"] = 1
            frame[f"{field}V2EventTime"] = frame["openTime"]
            frame[f"{field}V2AvailabilityTime"] = frame["openTime"]
            ledger = pd.DataFrame({
                "schemaId": collector.feed.DERIVATIVE_OBSERVATION_SCHEMA_ID,
                "symbol": symbol,
                "interval": interval,
                "feature": field,
                "eventTime": frame["openTime"],
                "availabilityTime": frame["openTime"],
                "observed": 1,
                "value": frame[field],
            })
            collector.feed.write_cache_atomic(
                ledger,
                collector.feed._observation_path(symbol, interval, field),
            )
        collector.feed.write_cache_atomic(
            frame, collector.feed._cache_path(symbol, interval)
        )
        series = {
            field: {
                "status": "ok",
                "observations": 3,
                "finite": 3,
                "latestTimestamp": latest,
                "latestObservationTimestamp": latest,
                "lagMs": interval_ms - 1,
                "trailingUnavailable": 0,
                "observationSchema": collector.feed.DERIVATIVE_OBSERVATION_SCHEMA_ID,
                "v2Status": "ok",
                "v2Observations": 3,
            }
            for field in ("funding", "oi", "basis", "taker")
        }
        if symbol == "SOLUSDT":
            series["basis"] = {
                "status": "error",
                "error": "simulated swallowed basis failure",
                "observations": 0,
                "finite": 0,
                "latestTimestamp": None,
            }
        return {
            symbol: {
                "status": "updated",
                "freshRows": 3,
                "freshLatestOpenTime": latest,
                "cacheRows": 3,
                "series": series,
            }
        }

    collector.feed.update_cache = fake_update
    collector._repository_commit = lambda: "a" * 40
    collector._provenance_tracked_clean = lambda: True
    result = collector.main()
    assert result == 1
    assert calls == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    status_path = cache / ".collector" / "last-run.json"
    status = json.loads(status_path.read_text())
    assert status["schemaVersion"] == 3
    assert status["state"] == "partial_failure"
    assert status["commit"] == "a" * 40
    assert status["artifactSchema"] == collector.ARTIFACT_SCHEMA_ID
    assert status["dataSourceLicenseManifest"] == collector.SOURCE_LICENSE_MANIFEST
    assert status["provenanceTrackedClean"] is True
    assert set(status["runtime"]) == {"python", "numpy", "pandas"}
    assert status["failedSymbols"] == ["ETHUSDT", "SOLUSDT"]
    assert status["results"]["BTCUSDT"]["status"] == "ok"
    assert status["results"]["BTCUSDT"]["rows"] == 3
    assert (
        status["results"]["BTCUSDT"]["artifactSchema"]
        == collector.ARTIFACT_SCHEMA_ID
    )
    assert (
        status["results"]["BTCUSDT"]["derivativesObservationSchema"]
        == collector.feed.DERIVATIVE_OBSERVATION_SCHEMA_ID
    )
    assert status["results"]["BTCUSDT"]["derivativesV2Coverage"]["funding"] == {
        "versioned": 3,
        "observed": 3,
        "fresh": 3,
    }
    assert set(status["results"]["BTCUSDT"]["artifacts"]) == {
        "cache",
        "observations",
    }
    assert len(status["results"]["BTCUSDT"]["artifacts"]["cache"]["sha256"]) == 64
    assert status["results"]["SOLUSDT"]["status"] == "degraded"
    assert status["results"]["SOLUSDT"]["issues"] == ["basis"]
    assert "simulated public endpoint failure" in status["results"]["ETHUSDT"]["error"]
    assert not list((cache / ".collector").glob("*.tmp"))

    verified_status = dict(status)
    verified_status["state"] = "pass"
    verified_status["symbols"] = ["BTCUSDT"]
    verified_status["results"] = {"BTCUSDT": status["results"]["BTCUSDT"]}
    verified_status["failedSymbols"] = []
    verified_path = cache / ".collector" / "verified-run.json"
    collector._write_json_atomic(verified_path, verified_status)
    verification = collector.verify_collection_artifacts(verified_path)
    assert verification["status"] == "verified"
    assert verification["statusSha256"] == collector._file_sha256(verified_path)
    assert verification["symbols"]["BTCUSDT"]["rows"] == 3
    assert verification["symbols"]["BTCUSDT"]["observations"] == {
        "funding": 3,
        "oi": 3,
        "basis": 3,
        "taker": 3,
    }

    with tempfile.TemporaryDirectory() as receipt_root_name:
        receipt_root = Path(receipt_root_name)
        archive = receipt_root / "archive"
        (archive / ".collector").mkdir(parents=True)
        (archive / ".observations").mkdir()
        archived_status_path = archive / ".collector" / "last-run.json"
        collector._write_json_atomic(archived_status_path, verified_status)

        status_artifacts = verified_status["results"]["BTCUSDT"]["artifacts"]
        shutil.copy2(status_artifacts["cache"]["path"], archive / "BTCUSDT_1h.csv")
        receipt_observations = {}
        for feature in collector.feed.DERIVATIVE_FIELDS:
            status_record = status_artifacts["observations"][feature]
            logical_path = f"BTCUSDT_1h_{feature}_v2.csv"
            shutil.copy2(
                status_record["path"], archive / ".observations" / logical_path
            )
            receipt_observations[feature] = {
                "logicalPath": logical_path,
                "rows": status_record["rows"],
                "sha256": status_record["sha256"],
            }
        archive_files = [path for path in archive.rglob("*") if path.is_file()]
        receipt = {
            "schemaVersion": 1,
            "receiptId": "test_derivatives_receipt",
            "receiptType": "metadata_only_collection_artifact_receipt",
            "collection": {
                "statusSchemaVersion": verified_status["schemaVersion"],
                "artifactSchema": verified_status["artifactSchema"],
                "derivativesObservationSchema": verified_status[
                    "derivativesObservationSchema"
                ],
                "featureAvailabilitySchema": verified_status[
                    "featureAvailabilitySchema"
                ],
                "interval": verified_status["interval"],
                "startedAt": verified_status["startedAt"],
                "finishedAt": verified_status["finishedAt"],
                "codeCommit": verified_status["commit"],
                "state": verified_status["state"],
                "failedSymbols": verified_status["failedSymbols"],
                "provenanceIssues": verified_status["provenanceIssues"],
                "provenanceTrackedClean": verified_status[
                    "provenanceTrackedClean"
                ],
            },
            "source": {
                "provider": "Binance USD-M Futures",
                "access": "public_read_only",
                "licenseManifest": collector.SOURCE_LICENSE_MANIFEST,
                "retrievalCommand": "python3 scripts/research/collect_datafeed.py",
                "verificationCommand": "collector verify-artifacts",
                "archivePolicy": "test archive remains outside Git",
            },
            "universe": {
                "symbols": ["BTCUSDT"],
                "stableOrder": "exact_fixed_collector_order",
            },
            "status": {
                "logicalPath": ".collector/last-run.json",
                "sha256": collector._file_sha256(archived_status_path),
            },
            "artifacts": {
                "BTCUSDT": {
                    "cache": {
                        "logicalPath": "BTCUSDT_1h.csv",
                        "rows": status_artifacts["cache"]["rows"],
                        "sha256": status_artifacts["cache"]["sha256"],
                    },
                    "observations": receipt_observations,
                }
            },
            "verification": {
                "result": "verified_in_place_and_relocated",
                "archiveFileCount": len(archive_files),
                "archiveApproximateMiB": round(
                    sum(path.stat().st_size for path in archive_files)
                    / (1024 * 1024),
                    1,
                ),
            },
            "outcomeBoundary": receipt_verifier.OUTCOME_BOUNDARY,
        }
        receipt_path = receipt_root / "receipt.json"
        collector._write_json_atomic(receipt_path, receipt)
        receipt_verification = receipt_verifier.verify_receipt(receipt_path, archive)
        assert receipt_verification["status"] == "verified"
        assert receipt_verification["artifactsVerified"] == 5
        assert receipt_verification["archiveFilesVerified"] == 6
        assert receipt_verification["outcomeAdmission"] == "acquisition_metadata_only"

        receipt_cli = subprocess.run(
            [
                sys.executable,
                sys.argv[3],
                "--receipt",
                str(receipt_path),
                "--archive",
                str(archive),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert receipt_cli.returncode == 0, receipt_cli.stderr
        assert json.loads(receipt_cli.stdout)["receiptId"] == receipt["receiptId"]

        wrong_boundary = json.loads(json.dumps(receipt))
        wrong_boundary["outcomeBoundary"]["returnsComputed"] = 0
        collector._write_json_atomic(receipt_path, wrong_boundary)
        try:
            receipt_verifier.verify_receipt(receipt_path, archive)
        except ValueError as error:
            assert "not acquisition-only" in str(error)
        else:
            raise AssertionError("a non-boolean outcome boundary must fail closed")

        wrong_hash = json.loads(json.dumps(receipt))
        wrong_hash["artifacts"]["BTCUSDT"]["cache"]["sha256"] = "0" * 64
        collector._write_json_atomic(receipt_path, wrong_hash)
        try:
            receipt_verifier.verify_receipt(receipt_path, archive)
        except ValueError as error:
            assert "disagrees with the collector status" in str(error)
        else:
            raise AssertionError("a changed receipt hash must fail verification")

        wrong_path = json.loads(json.dumps(receipt))
        wrong_path["artifacts"]["BTCUSDT"]["cache"]["logicalPath"] = (
            "../BTCUSDT_1h.csv"
        )
        collector._write_json_atomic(receipt_path, wrong_path)
        try:
            receipt_verifier.verify_receipt(receipt_path, archive)
        except ValueError as error:
            assert "normalized relative POSIX path" in str(error)
        else:
            raise AssertionError("a traversal path must invalidate a receipt")

        collector._write_json_atomic(receipt_path, receipt)
        extra_path = archive / "unbound.txt"
        extra_path.write_text("unbound")
        try:
            receipt_verifier.verify_receipt(receipt_path, archive)
        except ValueError as error:
            assert "archive inventory" in str(error)
        else:
            raise AssertionError("unbound archive bytes must invalidate a receipt")
        extra_path.unlink()

        receipt_path.write_text('{"schemaVersion":1,"schemaVersion":1}\n')
        try:
            receipt_verifier.verify_receipt(receipt_path, archive)
        except ValueError as error:
            assert "duplicate JSON key" in str(error)
        else:
            raise AssertionError("duplicate receipt keys must fail verification")

    rate_limit_calls = []
    def rate_limited_update(symbols, interval, *, acquire_lock):
        rate_limit_calls.append(symbols[0])
        assert interval == "1h"
        assert acquire_lock is False
        raise collector.feed.BinanceRateLimitError(
            http_status=429,
            banned_until_ms=1234567890123,
            retry_after_seconds=17,
        )
    collector.feed.update_cache = rate_limited_update
    assert collector.main() == 1
    rate_limited_status = json.loads(status_path.read_text())
    assert rate_limit_calls == ["BTCUSDT"]
    assert rate_limited_status["state"] == "partial_failure"
    assert rate_limited_status["failedSymbols"] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert rate_limited_status["results"]["BTCUSDT"] == {
        "status": "error",
        "failureKind": "provider_rate_limit",
        "error": (
            "Binance public API rate limited (HTTP 429, "
            "bannedUntilMs=1234567890123, retryAfterSeconds=17)"
        ),
        "httpStatus": 429,
        "bannedUntilMs": 1234567890123,
        "retryAfterSeconds": 17,
    }
    for symbol in ("ETHUSDT", "SOLUSDT"):
        assert rate_limited_status["results"][symbol] == {
            "status": "skipped",
            "failureKind": "provider_rate_limit_circuit_open",
            "error": "not attempted after Binance public API rate limit",
        }
    collector.feed.update_cache = fake_update

    malformed_refresh = json.loads(json.dumps(verified_status))
    malformed_refresh["results"]["BTCUSDT"]["refreshSeries"]["taker"][
        "lagMs"
    ] = 2 * interval_ms + 1
    collector._write_json_atomic(verified_path, malformed_refresh)
    try:
        collector.verify_collection_artifacts(verified_path)
    except ValueError as error:
        assert "refresh evidence is malformed" in str(error)
    else:
        raise AssertionError("out-of-bound refresh lag must fail verification")
    collector._write_json_atomic(verified_path, verified_status)

    cli_verification = subprocess.run(
        [
            sys.executable,
            sys.argv[2],
            "verify-artifacts",
            "--status",
            str(verified_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert cli_verification.returncode == 0, cli_verification.stderr
    assert json.loads(cli_verification.stdout)["status"] == "verified"

    artifact_lock_path = cache / ".collector" / "collector.lock"
    with artifact_lock_path.open("a+") as artifact_lock:
        fcntl.flock(artifact_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            collector.verify_collection_artifacts(verified_path)
        except ValueError as error:
            assert "writer is active" in str(error)
        else:
            raise AssertionError("artifact verification must not race a writer")

    with tempfile.TemporaryDirectory() as relocated_name:
        relocated = Path(relocated_name)
        shutil.copytree(cache, relocated, dirs_exist_ok=True)
        relocated_lock = relocated / ".collector" / "collector.lock"
        relocated_lock.unlink()
        relocated_verification = collector.verify_collection_artifacts(
            verified_path, cache_dir=relocated
        )
        assert relocated_verification["status"] == "verified"
        assert not relocated_lock.exists()

    cache_path = cache / "BTCUSDT_1h.csv"
    original_cache = cache_path.read_bytes()
    mutated = pd.read_csv(cache_path)
    mutated.loc[0, "fundingV2Value"] = 999.0
    mutated.to_csv(cache_path, index=False)
    try:
        collector.verify_collection_artifacts(verified_path)
    except ValueError as error:
        assert "sha256 mismatch" in str(error)
    else:
        raise AssertionError("changed cache bytes must fail artifact verification")

    verified_status["results"]["BTCUSDT"]["artifacts"]["cache"]["sha256"] = (
        collector._file_sha256(cache_path)
    )
    collector._write_json_atomic(verified_path, verified_status)
    try:
        collector.verify_collection_artifacts(verified_path)
    except ValueError as error:
        assert "disagrees with its ledger" in str(error)
    else:
        raise AssertionError("rehashing a changed cache must not bypass reconstruction")
    cache_path.write_bytes(original_cache)

    verified_status["results"]["BTCUSDT"]["artifacts"]["cache"]["sha256"] = (
        collector._file_sha256(cache_path)
    )
    collector._write_json_atomic(verified_path, verified_status)
    non_pass = dict(verified_status)
    non_pass["state"] = "partial_failure"
    rejected_path = cache / ".collector" / "rejected-run.json"
    collector._write_json_atomic(rejected_path, non_pass)
    try:
        collector.verify_collection_artifacts(rejected_path)
    except ValueError as error:
        assert "not a complete pass" in str(error)
    else:
        raise AssertionError("a degraded collector status must not verify")

    duplicate_status_path = cache / ".collector" / "duplicate-status.json"
    duplicate_status_path.write_text('{"schemaVersion":3,"schemaVersion":3}\n')
    try:
        collector.verify_collection_artifacts(duplicate_status_path)
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate manifest keys must fail verification")

    os.environ["TRADER_RESEARCH_SYMBOLS"] = "BTCUSDT"
    collector._provenance_tracked_clean = lambda: False
    assert collector.main() == 1
    dirty_implementation_status = json.loads(status_path.read_text())
    assert dirty_implementation_status["state"] == "partial_failure"
    assert dirty_implementation_status["failedSymbols"] == []
    assert dirty_implementation_status["provenanceIssues"] == [
        "provenance_files_differ_from_commit"
    ]
    collector._provenance_tracked_clean = lambda: True

    before = status_path.read_bytes()
    lock_path = cache / ".collector" / "collector.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = subprocess.run(
            [sys.executable, sys.argv[2]],
            env={
                **os.environ,
                "TRADER_RESEARCH_CACHE": str(cache),
                "TRADER_RESEARCH_SYMBOLS": "BTCUSDT",
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert locked.returncode == 0
    assert "already running" in locked.stdout
    assert status_path.read_bytes() == before

    direct_program = """
import sys
import pandas as pd
sys.path.insert(0, sys.argv[1])
import datafeed
datafeed.CACHE_DIR = sys.argv[2]
datafeed.fetch_klines = lambda *_args, **_kwargs: pd.DataFrame()
result = datafeed.update_cache(["XRPUSDT"], "1h")
assert result["XRPUSDT"]["status"] == "no_klines"
print("direct writer completed")
"""
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        direct = subprocess.Popen(
            [sys.executable, "-c", direct_program, sys.argv[1], str(cache)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.25)
        assert direct.poll() is None
    stdout, stderr = direct.communicate(timeout=10)
    assert direct.returncode == 0, stderr
    assert "direct writer completed" in stdout

    reader_program = """
import sys
sys.path.insert(0, sys.argv[1])
import datafeed
datafeed.CACHE_DIR = sys.argv[2]
panel = datafeed.load_panel(["BTCUSDT"], "1h", refresh=False)
assert len(panel["BTCUSDT"]) == 3
print("snapshot reader completed")
"""
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        reader = subprocess.Popen(
            [sys.executable, "-c", reader_program, sys.argv[1], str(cache)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.25)
        assert reader.poll() is None
    stdout, stderr = reader.communicate(timeout=10)
    assert reader.returncode == 0, stderr
    assert "snapshot reader completed" in stdout

    os.environ["TRADER_RESEARCH_SYMBOLS"] = "BTCUSDT"
    def stale_update(symbols, interval, *, acquire_lock):
        refreshes = fake_update(symbols, interval, acquire_lock=acquire_lock)
        refreshes[symbols[0]]["freshLatestOpenTime"] = 0
        return refreshes
    collector.feed.update_cache = stale_update
    stale_result = collector.main()
    assert stale_result == 1
    stale_status = json.loads(status_path.read_text())
    assert stale_status["state"] == "partial_failure"
    assert "stale" in stale_status["results"]["BTCUSDT"]["error"]

    def interrupt_update(_symbols, _interval, *, acquire_lock):
        assert acquire_lock is False
        raise collector.CollectorInterrupted("interrupted", "SIGTERM", 130)
    collector.feed.update_cache = interrupt_update
    interrupted_result = collector.main()
    assert interrupted_result == 130
    interrupted_status = json.loads(status_path.read_text())
    assert interrupted_status["state"] == "interrupted"
    assert interrupted_status["stopReason"] == "SIGTERM"

    collector.feed.update_cache = fake_update
    original_status_write = collector._write_json_atomic
    initial_write_interrupted = False
    def interrupt_initial_write(path, value):
        global initial_write_interrupted
        if value.get("state") == "running" and not initial_write_interrupted:
            initial_write_interrupted = True
            raise collector.CollectorInterrupted("interrupted", "SIGTERM", 130)
        original_status_write(path, value)
    collector._write_json_atomic = interrupt_initial_write
    initial_interrupt_result = collector.main()
    collector._write_json_atomic = original_status_write
    assert initial_interrupt_result == 130
    assert initial_write_interrupted
    assert json.loads(status_path.read_text())["state"] == "interrupted"

    final_write_interrupted = False
    def interrupt_final_write(path, value):
        global final_write_interrupted
        if value.get("state") == "pass" and not final_write_interrupted:
            final_write_interrupted = True
            raise collector.CollectorInterrupted("interrupted", "SIGTERM", 130)
        original_status_write(path, value)
    collector._write_json_atomic = interrupt_final_write
    final_interrupt_result = collector.main()
    collector._write_json_atomic = original_status_write
    assert final_interrupt_result == 130
    assert final_write_interrupted
    assert json.loads(status_path.read_text())["state"] == "interrupted"

    def slow_update(_symbols, _interval, *, acquire_lock):
        assert acquire_lock is False
        time.sleep(2)
        raise AssertionError("collector deadline did not fire")
    collector.feed.update_cache = slow_update
    os.environ["TRADER_RESEARCH_MAX_RUN_SECONDS"] = "1"
    timeout_result = collector.main()
    assert timeout_result == 124
    timeout_status = json.loads(status_path.read_text())
    assert timeout_status["state"] == "timeout"
    assert timeout_status["stopReason"] == "wall_clock_deadline"
`;
    const run = spawnSync(
      pythonPath,
      ["-c", program, RESEARCH_DIR, COLLECTOR, RECEIPT_VERIFIER],
      {
      encoding: "utf8",
      timeout: 30_000,
      },
    );
    assert.equal(run.status, 0, run.stderr || run.stdout);
  },
);

test(
  "research collector LaunchAgent is hourly, bounded, and secret-free",
  { skip: !hasResearchPython },
  async () => {
    const home = await fs.mkdtemp(path.join(os.tmpdir(), "research-datafeed-home-"));
    const cache = path.join(home, "cache");
    try {
      const environment = {
        ...process.env,
        HOME: home,
        TRADER_RESEARCH_LAUNCHD_LABEL: `test.trader.research-datafeed.${process.pid}`,
        TRADER_RESEARCH_CACHE: cache,
        TRADER_RESEARCH_SYMBOLS: "BTCUSDT ETHUSDT",
        TRADER_RESEARCH_COLLECT_MINUTE: "10",
        TRADER_RESEARCH_MAX_RUN_SECONDS: "3000",
        TRADER_RESEARCH_PYTHON: pythonPath,
      };
      const write = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: environment,
      });
      assert.equal(write.status, 0, write.stderr || write.stdout);

      const plistPath = path.join(
        home,
        "Library",
        "LaunchAgents",
        "ai.openclaw.trader.research-datafeed.plist",
      );
      const parseProgram = String.raw`
import json
import plistlib
import sys
with open(sys.argv[1], "rb") as handle:
    print(json.dumps(plistlib.load(handle), sort_keys=True))
`;
      const parsedRun = spawnSync(pythonPath, ["-c", parseProgram, plistPath], {
        encoding: "utf8",
      });
      assert.equal(parsedRun.status, 0, parsedRun.stderr);
      const plist = JSON.parse(parsedRun.stdout);
      assert.deepEqual(plist.ProgramArguments, [pythonPath, COLLECTOR]);
      assert.equal(plist.WorkingDirectory, ROOT.replace(/\/$/, ""));
      assert.equal(plist.RunAtLoad, true);
      assert.deepEqual(plist.StartCalendarInterval, { Minute: 10 });
      assert.equal(plist.ProcessType, "Background");
      assert.equal("KeepAlive" in plist, false);
      assert.equal(plist.EnvironmentVariables.TRADER_RESEARCH_CACHE, cache);
      assert.equal(plist.EnvironmentVariables.TRADER_RESEARCH_SYMBOLS, "BTCUSDT ETHUSDT");
      assert.equal(plist.EnvironmentVariables.TRADER_RESEARCH_MAX_RUN_SECONDS, "3000");
      assert.equal(plist.StandardOutPath, path.join(cache, ".collector", "launchd.stdout.log"));
      assert.equal(plist.StandardErrorPath, path.join(cache, ".collector", "launchd.stderr.log"));
      assert.deepEqual(Object.keys(plist.EnvironmentVariables).sort(), [
        "HOME",
        "PATH",
        "TRADER_RESEARCH_CACHE",
        "TRADER_RESEARCH_MAX_RUN_SECONDS",
        "TRADER_RESEARCH_SYMBOLS",
      ]);

      const relativePython = path.relative(ROOT, pythonPath);
      const relativePythonWrite = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: { ...environment, TRADER_RESEARCH_PYTHON: relativePython },
        cwd: ROOT,
      });
      assert.equal(
        relativePythonWrite.status,
        0,
        relativePythonWrite.stderr || relativePythonWrite.stdout,
      );
      const relativeParsedRun = spawnSync(pythonPath, ["-c", parseProgram, plistPath], {
        encoding: "utf8",
      });
      assert.equal(relativeParsedRun.status, 0, relativeParsedRun.stderr);
      assert.equal(path.isAbsolute(JSON.parse(relativeParsedRun.stdout).ProgramArguments[0]), true);

      const leadingZeroMinute = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: { ...environment, TRADER_RESEARCH_COLLECT_MINUTE: "08" },
      });
      assert.equal(
        leadingZeroMinute.status,
        0,
        leadingZeroMinute.stderr || leadingZeroMinute.stdout,
      );
      const reparsedRun = spawnSync(pythonPath, ["-c", parseProgram, plistPath], {
        encoding: "utf8",
      });
      assert.equal(reparsedRun.status, 0, reparsedRun.stderr);
      assert.deepEqual(JSON.parse(reparsedRun.stdout).StartCalendarInterval, { Minute: 8 });

      const plistBeforeInvalidWrite = await fs.readFile(plistPath);
      const invalidMinute = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: { ...environment, TRADER_RESEARCH_COLLECT_MINUTE: "60" },
      });
      assert.equal(invalidMinute.status, 1);
      assert.match(invalidMinute.stderr, /integer from 0 through 59/);
      assert.deepEqual(await fs.readFile(plistPath), plistBeforeInvalidWrite);

      const invalidLabel = spawnSync("bash", [INSTALLER, "print-plist-path"], {
        encoding: "utf8",
        env: { ...environment, TRADER_RESEARCH_LAUNCHD_LABEL: "../../unsafe" },
      });
      assert.equal(invalidLabel.status, 1);
      assert.match(invalidLabel.stderr, /invalid characters/);

      const customLabel = "test.trader.research-datafeed";
      const customCache = path.join(home, "custom-cache");
      const customEnvironment = {
        ...environment,
        TRADER_RESEARCH_CACHE: customCache,
        TRADER_RESEARCH_LAUNCHD_LABEL: customLabel,
        TRADER_RESEARCH_COLLECT_MINUTE: "10",
      };
      const customWrite = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: customEnvironment,
      });
      assert.equal(customWrite.status, 0, customWrite.stderr || customWrite.stdout);

      const cleanEnvironment = {
        ...process.env,
        HOME: home,
      };
      for (const name of [
        "TRADER_RESEARCH_CACHE",
        "TRADER_RESEARCH_SYMBOLS",
        "TRADER_RESEARCH_COLLECT_MINUTE",
        "TRADER_RESEARCH_MAX_RUN_SECONDS",
        "TRADER_RESEARCH_LAUNCHD_LABEL",
        "TRADER_RESEARCH_PYTHON",
      ]) {
        delete cleanEnvironment[name];
      }
      const configuredStatus = spawnSync("bash", [INSTALLER, "status"], {
        encoding: "utf8",
        env: cleanEnvironment,
      });
      assert.equal(
        configuredStatus.status,
        0,
        configuredStatus.stderr || configuredStatus.stdout,
      );
      assert.match(configuredStatus.stdout, new RegExp(`${customLabel}`));
      assert.match(configuredStatus.stdout, new RegExp(`${customCache}`));

      const customParsedRun = spawnSync(pythonPath, ["-c", parseProgram, plistPath], {
        encoding: "utf8",
      });
      assert.equal(customParsedRun.status, 0, customParsedRun.stderr);
      const customPlist = JSON.parse(customParsedRun.stdout);
      assert.equal(customPlist.Label, customLabel);
      assert.equal(customPlist.EnvironmentVariables.TRADER_RESEARCH_CACHE, customCache);

      const fakeBin = path.join(home, "bin");
      const launchctlLog = path.join(home, "launchctl.log");
      await fs.mkdir(fakeBin);
      const fakeLaunchctl = path.join(fakeBin, "launchctl");
      await fs.writeFile(
        fakeLaunchctl,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${LAUNCHCTL_LOG}"\nif [[ "$1" == "print" && "${FAIL_PRINT:-0}" == "1" ]]; then exit 42; fi\nif [[ "$1" == "bootout" && "${FAIL_BOOTOUT:-0}" == "1" ]]; then exit 42; fi\nif [[ "$1" == "bootstrap" && -n "${FAIL_BOOTSTRAP_ONCE_FILE:-}" && -f "${FAIL_BOOTSTRAP_ONCE_FILE}" ]]; then rm -f "${FAIL_BOOTSTRAP_ONCE_FILE}"; exit 42; fi\n',
      );
      await fs.chmod(fakeLaunchctl, 0o755);
      const lifecycleEnvironment = {
        ...customEnvironment,
        PATH: `${fakeBin}:${process.env.PATH}`,
        LAUNCHCTL_LOG: launchctlLog,
      };
      const install = spawnSync("bash", [INSTALLER, "install"], {
        encoding: "utf8",
        env: lifecycleEnvironment,
      });
      assert.equal(install.status, 0, install.stderr || install.stdout);
      const installCalls = (await fs.readFile(launchctlLog, "utf8")).trim().split("\n");
      assert.ok(installCalls.some((call) => call.startsWith("bootout ")));
      assert.ok(installCalls.some((call) => call.startsWith("bootstrap ")));
      assert.equal(installCalls.some((call) => call.startsWith("kickstart ")), false);

      const installedPlist = await fs.readFile(plistPath);
      const callsBeforeInvalidReinstall = await fs.readFile(launchctlLog, "utf8");
      const invalidReinstall = spawnSync("bash", [INSTALLER, "install"], {
        encoding: "utf8",
        env: { ...lifecycleEnvironment, TRADER_RESEARCH_COLLECT_MINUTE: "60" },
      });
      assert.equal(invalidReinstall.status, 1);
      assert.deepEqual(await fs.readFile(plistPath), installedPlist);
      assert.equal(await fs.readFile(launchctlLog, "utf8"), callsBeforeInvalidReinstall);

      const bootstrapFailureMarker = path.join(home, "fail-bootstrap-once");
      await fs.writeFile(bootstrapFailureMarker, "fail once\n");
      const failedReplacement = spawnSync("bash", [INSTALLER, "install"], {
        encoding: "utf8",
        env: {
          ...lifecycleEnvironment,
          TRADER_RESEARCH_COLLECT_MINUTE: "11",
          FAIL_BOOTSTRAP_ONCE_FILE: bootstrapFailureMarker,
        },
      });
      assert.equal(failedReplacement.status, 1);
      assert.match(failedReplacement.stderr, /restoring prior LaunchAgent/);
      assert.deepEqual(await fs.readFile(plistPath), installedPlist);

      const loadedWrite = spawnSync("bash", [INSTALLER, "write-plist"], {
        encoding: "utf8",
        env: { ...lifecycleEnvironment, TRADER_RESEARCH_COLLECT_MINUTE: "12" },
      });
      assert.equal(loadedWrite.status, 1);
      assert.match(loadedWrite.stderr, /use install/);
      assert.deepEqual(await fs.readFile(plistPath), installedPlist);

      const cleanLifecycleEnvironment = {
        ...cleanEnvironment,
        PATH: lifecycleEnvironment.PATH,
        LAUNCHCTL_LOG: launchctlLog,
      };
      const restart = spawnSync("bash", [INSTALLER, "restart"], {
        encoding: "utf8",
        env: cleanLifecycleEnvironment,
      });
      assert.equal(restart.status, 0, restart.stderr || restart.stdout);
      assert.match(await fs.readFile(launchctlLog, "utf8"), /kickstart -k/);

      const failedUninstall = spawnSync("bash", [INSTALLER, "uninstall"], {
        encoding: "utf8",
        env: { ...cleanLifecycleEnvironment, FAIL_BOOTOUT: "1" },
      });
      assert.equal(failedUninstall.status, 1);
      assert.match(failedUninstall.stderr, /definition retained/);
      await fs.access(plistPath);

      const unknownStateUninstall = spawnSync("bash", [INSTALLER, "uninstall"], {
        encoding: "utf8",
        env: { ...cleanLifecycleEnvironment, FAIL_PRINT: "1" },
      });
      assert.notEqual(unknownStateUninstall.status, 0);
      assert.match(unknownStateUninstall.stderr, /Unable to determine/);
      await fs.access(plistPath);

      const uninstall = spawnSync("bash", [INSTALLER, "uninstall"], {
        encoding: "utf8",
        env: cleanLifecycleEnvironment,
      });
      assert.equal(uninstall.status, 0, uninstall.stderr || uninstall.stdout);
      await assert.rejects(fs.access(plistPath));
      await fs.access(customCache);

      const source = await fs.readFile(INSTALLER, "utf8");
      assert.match(source, /launchctl bootstrap/);
      assert.match(source, /launchctl kickstart -k/);
      assert.match(source, /cache retained at/);
      assert.doesNotMatch(source, /BINANCE_API_KEY|BINANCE_API_SECRET/);
    } finally {
      await fs.rm(home, { recursive: true, force: true });
    }
  },
);
