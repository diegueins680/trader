import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

test(
  "historical funding runner seals inputs and leaves blocked holdout untouched",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import argparse
import json
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import campaign_runner as common
import funding_campaign as funding
import historical_datafeed as feed
import run_historical_funding_campaign as runner

INTERVAL = feed.CONTRACT_INTERVAL_MS
HOUR = feed.MARK_PRICE_INTERVAL_MS
committed_registration = runner._registration()
same_read_registration, committed_registration_sha = runner._registration_and_sha()
assert committed_registration["campaign"] == runner.CAMPAIGN_ID
assert same_read_registration == committed_registration
assert committed_registration_sha == common._file_digest(runner.REGISTRATION_PATH)
assert tuple(committed_registration["universe"]["symbols"]) == runner.REGISTERED_SYMBOLS
SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]
ROWS = 260
DEVELOPMENT_ROWS = 210
times = np.arange(ROWS, dtype=np.int64) * INTERVAL

trials = [
    {
        "id": spec.trial_id,
        "horizonHours": spec.horizon_hours,
        "horizonBars": spec.horizon_bars,
        "variant": spec.variant,
    }
    for spec in funding.campaign_specs(INTERVAL)
]
registration = {
    "campaign": runner.CAMPAIGN_ID,
    "registrationVersion": 1,
    "universe": {
        "interval": "8h",
        "intervalMilliseconds": INTERVAL,
        "symbols": SYMBOLS,
        "survivorshipLimitation": "synthetic survivor test",
    },
    "registeredData": {
        "startOpenTime": 0,
        "endOpenTime": int(times[-1]),
        "outcomeEndTimeExclusive": int(times[-1] + INTERVAL),
        "fundingWindowEndInclusive": int(times[-1] + INTERVAL - 1),
        "maximumFundingGapMilliseconds": INTERVAL + 60_000,
        "rows": ROWS,
        "developmentRows": DEVELOPMENT_ROWS,
        "developmentCutoffOpenTime": int(times[DEVELOPMENT_ROWS - 1]),
        "holdoutStartOpenTime": int(times[DEVELOPMENT_ROWS]),
        "holdoutBars": ROWS - DEVELOPMENT_ROWS,
        "holdoutReturnRows": ROWS - DEVELOPMENT_ROWS - 1,
    },
    "strategy": {
        "betaLookbackBars": 21,
        "costBpsPerUnitTurnover": 5.0,
        "fundingCrowdingZ": 2.0,
        "fundingZLookbackBars": 21,
        "grossExposure": 1.0,
        "signalDelayBars": 1,
        "topNPerSide": 1,
    },
    "trials": trials,
    "validation": {
        "bootstrapReplications": 100,
        "bootstrapSeed": 7,
        "developmentEvaluationRows": DEVELOPMENT_ROWS - 1 - 21,
        "featureWarmupRows": 21,
        "innerInitialTrain": 40,
        "innerFoldPolicy": "retain partial inner tail",
        "innerTestSize": 15,
        "labelHorizonBars": 1,
        "lifetimeTrialCount": 21,
        "newTrialCount": 6,
        "outerInitialTrain": 80,
        "outerFoldCount": 1,
        "outerFoldPolicy": "one complete synthetic outer fold",
        "outerTestSize": 107,
        "pairedComparisonFamilyWiseAlpha": 0.05,
        "pairedComparisonHypotheses": 3,
        "pboSlices": 2,
        "priorTrialCount": 15,
        "selectionRule": "synthetic expanding selection",
    },
    "promotion": {
        "currentCampaignDeflatedSharpeProbabilityMinimum": 0.95,
        "lifetimeBonferroniPsrProbabilityMinimum": 0.95,
        "maximumPbo": 0.2,
        "maximumRegimeLoss": 0.99,
        "maximumWorstFoldLoss": 0.99,
        "minimumActiveFraction": 0.0,
        "minimumOuterOosObservations": 1,
        "minimumRegimeObservations": 1,
        "minimumResolvedFundingFraction": 1.0,
        "minimumSymbols": 5,
        "requirePairedFundingImprovementSharpeCiAboveZero": True,
    },
}

snapshot = {}
for symbol_index, symbol in enumerate(SYMBOLS):
    market = 0.0004 * np.sin(np.arange(ROWS) * 0.09)
    residual = 0.0007 * np.sin(np.arange(ROWS) * (0.05 + symbol_index * 0.007) + symbol_index)
    close = 100.0 * np.exp(np.cumsum(market + residual))
    contracts = [
        {
            "openTime": int(open_time),
            "closeTime": int(open_time + INTERVAL - 1),
            "close": f"{close[row]:.12f}",
        }
        for row, open_time in enumerate(times)
    ]
    marks = []
    for open_time in range(0, int(times[-1] + INTERVAL), HOUR):
        bar = min(open_time // INTERVAL, ROWS - 1)
        marks.append({
            "openTime": open_time,
            "open": f"{close[bar]:.12f}",
            "high": f"{close[bar] * 1.001:.12f}",
            "low": f"{close[bar] * 0.999:.12f}",
            "close": f"{close[bar]:.12f}",
            "closeTime": open_time + HOUR - 1,
        })
    events = [
        {
            "symbol": symbol,
            "fundingTime": int(open_time + 7),
            "fundingRate": f"{0.00015 * np.sin(row * 0.13 + symbol_index):.12f}",
            "markPrice": "",
        }
        for row, open_time in enumerate(times)
    ]
    snapshot[symbol] = {"contract": contracts, "mark": marks, "funding": events}

panel = runner._contract_panel(snapshot, registration)
settlements, settlement_audit, coverage = runner._resolved_settlements(snapshot, registration)
development_panel = common._truncate_panel(
    panel, int(times[DEVELOPMENT_ROWS - 1])
)
development_end_close = int(times[DEVELOPMENT_ROWS - 1] + INTERVAL - 1)
development_settlement_audit = settlement_audit[
    pd.to_numeric(settlement_audit["fundingTime"], errors="raise")
    <= development_end_close
].copy()
assert len(settlements) == ROWS * len(SYMBOLS)
assert coverage["resolvedFraction"] == 1.0
assert coverage["fallbackMarkEvents"] == len(settlements)
assert coverage["maximumObservedGapMilliseconds"] == INTERVAL
assert set(settlement_audit["markSource"]) == {"containing_1h_mark_open"}

# Residual features use no values after the evaluated prefix.
close = runner._close_frame(panel)
full_momentum = runner._residual_momentum(close, 21, funding.HORIZON_HOURS)
prefix_momentum = runner._residual_momentum(close.iloc[:-10], 21, funding.HORIZON_HOURS)
for horizon in funding.HORIZON_HOURS:
    np.testing.assert_allclose(
        full_momentum[horizon].iloc[:-10],
        prefix_momentum[horizon],
        equal_nan=True,
        rtol=0,
        atol=1e-15,
    )

# Interrupted acquisition retains only hash-verified artifacts and resumes them.
acquisition_registration = {
    "universe": {"symbols": ["AAA"]},
    "registeredData": {
        "acquisitionMaxSeconds": 60,
        "startOpenTime": 0,
        "endOpenTime": 0,
        "fundingWindowEndInclusive": INTERVAL - 1,
    },
}
with tempfile.TemporaryDirectory() as acquisition_root:
    acquisition_root = common.Path(acquisition_root)
    calls = {"contract": 0, "mark": 0, "funding": 0}
    original_contract_fetch = feed.fetch_contract_klines
    original_mark_fetch = feed.fetch_mark_price_klines
    original_funding_fetch = feed.fetch_funding_events

    def acquire_contract(_symbol, _start, _end, *, deadline=None):
        assert deadline is not None
        calls["contract"] += 1
        return [{"kind": "contract"}]

    def acquire_mark(_symbol, _start, _end, *, deadline=None):
        assert deadline is not None
        calls["mark"] += 1
        return [{"kind": "mark"}]

    def acquire_funding(_symbol, _start, _end, *, deadline=None):
        assert deadline is not None
        calls["funding"] += 1
        if calls["funding"] == 1:
            raise RuntimeError("simulated interrupted funding request")
        return [{"kind": "funding"}]

    try:
        feed.fetch_contract_klines = acquire_contract
        feed.fetch_mark_price_klines = acquire_mark
        feed.fetch_funding_events = acquire_funding
        try:
            runner._load_snapshot(
                acquisition_root, acquisition_registration, "synthetic-sha", True
            )
        except RuntimeError as error:
            assert "interrupted" in str(error)
        else:
            raise AssertionError("an interrupted acquisition must propagate")

        progress = json.loads(
            (acquisition_root / "snapshot-progress.json").read_text()
        )
        assert set(progress["artifacts"]) == {"AAA:contract", "AAA:mark"}
        assert not (acquisition_root / "snapshot-manifest.json").exists()

        acquired_manifest, acquired_manifest_sha, acquired_snapshot = runner._load_snapshot(
            acquisition_root, acquisition_registration, "synthetic-sha", True
        )
    finally:
        feed.fetch_contract_klines = original_contract_fetch
        feed.fetch_mark_price_klines = original_mark_fetch
        feed.fetch_funding_events = original_funding_fetch

    assert calls == {"contract": 1, "mark": 1, "funding": 2}
    assert not (acquisition_root / "snapshot-progress.json").exists()
    assert len(acquired_manifest["artifacts"]) == 3
    assert acquired_manifest_sha == common._file_digest(
        acquisition_root / "snapshot-manifest.json"
    )
    assert acquired_snapshot["AAA"]["funding"] == [{"kind": "funding"}]

    feed.write_artifact_atomic(
        [{"kind": "tampered"}], acquisition_root / "AAA-funding-events.json"
    )
    try:
        runner._load_snapshot(
            acquisition_root, acquisition_registration, "synthetic-sha", False
        )
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("a changed sealed snapshot artifact must fail closed")

    with (acquisition_root / ".snapshot.lock").open("a+") as held_lock:
        runner.fcntl.flock(held_lock.fileno(), runner.fcntl.LOCK_EX)
        try:
            try:
                with runner._snapshot_lock(
                    acquisition_root, runner.time.monotonic() + 0.01
                ):
                    raise AssertionError("a held snapshot lock cannot be entered")
            except TimeoutError as error:
                assert "lock deadline" in str(error)
        finally:
            runner.fcntl.flock(held_lock.fileno(), runner.fcntl.LOCK_UN)

with tempfile.TemporaryDirectory() as deadline_root:
    deadline_root = common.Path(deadline_root)
    deadline_root.mkdir(exist_ok=True)
    deadline_clock = [0.0]
    original_monotonic = runner.time.monotonic
    original_contract_fetch = feed.fetch_contract_klines
    original_mark_fetch = feed.fetch_mark_price_klines
    original_funding_fetch = feed.fetch_funding_events

    def deadline_row(*_args, **_kwargs):
        return [{"onTime": True}]

    def late_funding(*_args, **_kwargs):
        deadline_clock[0] = 2.0
        return [{"tooLate": True}]

    try:
        runner.time.monotonic = lambda: deadline_clock[0]
        feed.fetch_contract_klines = deadline_row
        feed.fetch_mark_price_klines = deadline_row
        feed.fetch_funding_events = late_funding
        try:
            runner._acquire_snapshot(
                deadline_root, acquisition_registration, "synthetic-sha", 1.0
            )
        except TimeoutError as error:
            assert "deadline" in str(error)
        else:
            raise AssertionError("a late final fetch cannot seal the snapshot")
    finally:
        runner.time.monotonic = original_monotonic
        feed.fetch_contract_klines = original_contract_fetch
        feed.fetch_mark_price_klines = original_mark_fetch
        feed.fetch_funding_events = original_funding_fetch

    assert not (deadline_root / "snapshot-manifest.json").exists()
    deadline_progress = json.loads(
        (deadline_root / "snapshot-progress.json").read_text()
    )
    assert set(deadline_progress["artifacts"]) == {"AAA:contract", "AAA:mark"}

with tempfile.TemporaryDirectory() as root:
    root = common.Path(root)
    snapshot_dir = root / "snapshot"
    output_dir = root / "output"
    snapshot_dir.mkdir()
    common._write_json(snapshot_dir / "snapshot-manifest.json", {"synthetic": True})
    runner._registration_and_sha = lambda: (
        registration,
        "synthetic-registration-sha",
    )
    runner._registration_sha = lambda: "synthetic-registration-sha"
    runner._load_snapshot = lambda *_args, **_kwargs: (
        {"synthetic": True},
        "synthetic-snapshot-sha",
        snapshot,
    )
    runner.HOLDOUT_REGISTRY_DIR = root / "shared-holdouts"
    assert common.HOLDOUT_REGISTRY_VERSION == 3

    summary = runner.run(argparse.Namespace(
        snapshot_dir=str(snapshot_dir),
        output_dir=str(output_dir),
        acquire=False,
        open_final_holdout=True,
    ))
    assert summary["campaign"] == runner.CAMPAIGN_ID
    assert summary["status"] == "insufficient_evidence"
    assert len(summary["trials"]) == 6
    assert summary["data"]["trialReturnRows"] == DEVELOPMENT_ROWS - 1 - 21
    assert summary["data"]["settlements"]["resolvedFraction"] == 1.0
    assert summary["finalHoldout"]["status"] == "reserved"
    assert "symbolCount" in summary["finalHoldout"]["openBlockedBy"]
    assert not runner.HOLDOUT_REGISTRY_DIR.exists()
    assert summary["lifetimeMultipleTesting"]["lifetimeTrials"] == 21
    assert (
        summary["lifetimeMultipleTesting"]["adjustedProbability"]
        <= summary["lifetimeMultipleTesting"].get("singleTrialProbability", 1.0)
    )

    expected_files = [
        "campaign-manifest.json",
        "registered-development-panel.csv",
        "registered-development-settlements.csv",
        "summary.json",
        "trial-ledger.json",
        "trial-returns.csv",
        "trial-paths.csv",
        "diagnostic-trial-returns.csv",
        "pbo-trial-returns.csv",
        "nested-oos.csv",
        "outer-folds.csv",
        "inner-scores.csv",
        "final-selection-scores.csv",
        "final-selection-folds.csv",
        "stress-cost1_5x-nested-oos.csv",
        "stress-cost2x-nested-oos.csv",
        "stress-additionalDelay1bar-nested-oos.csv",
    ]
    for filename in expected_files:
        assert (output_dir / filename).is_file(), filename
    assert not (output_dir / "final-holdout-returns.csv").exists()
    trial_paths = pd.read_csv(output_dir / "trial-paths.csv")
    assert {"priceGross", "fundingCashflow", "gross", "turnover", "cost", "net"}.issubset(trial_paths)
    returns = pd.read_csv(output_dir / "trial-returns.csv")
    assert int(returns["openTime"].iloc[0]) == 21 * INTERVAL

    changed_panel = {symbol: frame.copy() for symbol, frame in panel.items()}
    changed_panel[SYMBOLS[0]].loc[DEVELOPMENT_ROWS + 1, "close"] *= 1.01
    try:
        runner._registered_input_manifest(
            output_dir,
            changed_panel,
            settlement_audit,
            development_panel,
            development_settlement_audit,
            "synthetic-snapshot-sha",
            registration,
            "synthetic-registration-sha",
            runner._implementation_sha(),
        )
    except ValueError as error:
        assert "registeredData" in str(error)
    else:
        raise AssertionError("registered panel values must be immutable")

    try:
        runner._registered_input_manifest(
            output_dir,
            panel,
            settlement_audit,
            development_panel,
            development_settlement_audit,
            "synthetic-snapshot-sha",
            registration,
            "synthetic-registration-sha",
            "stale-implementation-sha",
        )
    except ValueError as error:
        assert "implementation changed" in str(error)
    else:
        raise AssertionError("implementation provenance must remain frozen")

    success_registration = json.loads(json.dumps(registration))
    success_registration["promotion"]["minimumSymbols"] = len(SYMBOLS)
    runner._registration_and_sha = lambda: (
        success_registration,
        "synthetic-registration-sha",
    )
    common._bootstrap_ci = lambda *_args, **_kwargs: (1.0, 2.0)
    def passing_diagnostics(matrix, *_args, **_kwargs):
        return (
            {
                "deflatedSharpe": {"probability": 0.99},
                "pbo": {"probability": 0.01},
            },
            matrix.copy(),
            matrix.copy(),
        )
    common._diagnostics = passing_diagnostics
    runner._lifetime_multiple_testing = lambda *_args, **_kwargs: {
        "method": "synthetic",
        "lifetimeTrials": 21,
        "singleTrialProbability": 0.999,
        "adjustedProbability": 0.99,
    }
    success_output = root / "success-output"
    success_args = argparse.Namespace(
        snapshot_dir=str(snapshot_dir),
        output_dir=str(success_output),
        acquire=False,
        open_final_holdout=True,
    )
    success = runner.run(success_args)
    assert success["status"] == "final_holdout_passed"
    assert success["finalHoldout"]["status"] == "pass"
    assert success["finalHoldout"]["rows"] == ROWS - DEVELOPMENT_ROWS - 1
    assert success["promotionGates"]["pairedFundingImprovement"] is True
    holdout_returns = pd.read_csv(success_output / "final-holdout-returns.csv")
    assert {"priceGross", "fundingCashflow", "gross", "cost", "net"}.issubset(holdout_returns)
    stress_returns = pd.read_csv(success_output / "stress-cost2x-nested-oos.csv")
    assert {"priceGross", "fundingCashflow", "gross", "cost", "net"}.issubset(stress_returns)
    registry_records = list(runner.HOLDOUT_REGISTRY_DIR.glob("*.json"))
    assert len(registry_records) == 1
    registry_record = json.loads(registry_records[0].read_text())
    output_record = json.loads((success_output / "final-holdout-opened.json").read_text())
    assert registry_record["status"] == "completed"
    assert output_record["status"] == "completed"
    assert registry_record["registrationSha256"] == "synthetic-registration-sha"
    assert "campaignManifestSha256" in registry_record

    summary_before = (success_output / "summary.json").read_bytes()
    try:
        runner.run(argparse.Namespace(
            snapshot_dir=str(snapshot_dir),
            output_dir=str(success_output),
            acquire=False,
            open_final_holdout=False,
        ))
    except ValueError as error:
        assert "already consumed" in str(error)
    else:
        raise AssertionError("a completed output cannot be reverted to reserved")
    assert (success_output / "summary.json").read_bytes() == summary_before

    overlapping = common._holdout_window(
        [SYMBOLS[0]],
        "1h",
        int(times[DEVELOPMENT_ROWS] + HOUR),
        int(times[DEVELOPMENT_ROWS] + 2 * HOUR),
    )
    try:
        common._assert_holdout_available(
            runner.HOLDOUT_REGISTRY_DIR,
            overlapping,
            root / "overlap-output-record.json",
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("an overlapping interval cannot reuse the holdout")

    interrupted_window = common._holdout_window(
        ["ZZZ"], "8h", int(times[220]), int(times[225])
    )
    interrupted_marker = runner.HOLDOUT_REGISTRY_DIR / "interrupted.json"
    interrupted_output_dir = root / "interrupted-output"
    interrupted_output_dir.mkdir()
    interrupted_output_record = (
        interrupted_output_dir / "final-holdout-opened.json"
    )
    common._reserve_holdout(
        runner.HOLDOUT_REGISTRY_DIR,
        interrupted_marker,
        interrupted_window,
        interrupted_output_record,
        {
            "registryVersion": common.HOLDOUT_REGISTRY_VERSION,
            "status": "opening",
            "window": interrupted_window,
            "artifacts": {
                "outputDirectory": str(interrupted_output_dir.resolve()),
            },
        },
    )
    assert interrupted_output_record.exists()
    interrupted_output_record.unlink()
    try:
        common._assert_output_holdout_not_consumed(
            runner.HOLDOUT_REGISTRY_DIR, interrupted_output_dir
        )
    except ValueError as error:
        assert "registry evidence" in str(error)
    else:
        raise AssertionError("a global opening marker must guard a missing local record")
    try:
        common._assert_holdout_available(
            runner.HOLDOUT_REGISTRY_DIR,
            interrupted_window,
            root / "another-output.json",
        )
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("an interrupted opening must remain consumed")

    with (runner.HOLDOUT_REGISTRY_DIR / ".registry.lock").open("a+") as held_lock:
        common.fcntl.flock(held_lock.fileno(), common.fcntl.LOCK_EX)
        try:
            try:
                with common._holdout_registry_lock(
                    runner.HOLDOUT_REGISTRY_DIR, 0.01
                ):
                    raise AssertionError("a held registry lock cannot be entered")
            except TimeoutError as error:
                assert "lock deadline" in str(error)
        finally:
            common.fcntl.flock(held_lock.fileno(), common.fcntl.LOCK_UN)

# The maximum trailing funding-gap tolerance is inclusive.
boundary = {
    symbol: {kind: list(rows) for kind, rows in values.items()}
    for symbol, values in snapshot.items()
}
boundary_end = registration["registeredData"]["fundingWindowEndInclusive"]
maximum_gap = registration["registeredData"]["maximumFundingGapMilliseconds"]
boundary_last = dict(boundary[SYMBOLS[0]]["funding"][-1])
boundary_last["fundingTime"] = boundary_end - maximum_gap
boundary[SYMBOLS[0]]["funding"][-1] = boundary_last
runner._resolved_settlements(boundary, registration)

# A missing middle funding event violates the registered maximum-gap proof.
broken = {
    symbol: {kind: list(rows) for kind, rows in values.items()}
    for symbol, values in snapshot.items()
}
broken[SYMBOLS[0]]["funding"] = [
    row for index, row in enumerate(broken[SYMBOLS[0]]["funding"]) if index != 20
]
try:
    runner._resolved_settlements(broken, registration)
except ValueError as error:
    assert "maximum gap" in str(error)
else:
    raise AssertionError("a funding schedule gap must fail closed")
`;
    const result = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
      timeout: 60_000,
    });
    assert.equal(result.status, 0, result.stderr || result.stdout);
  },
);
