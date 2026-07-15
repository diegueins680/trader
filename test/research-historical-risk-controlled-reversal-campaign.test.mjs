import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(
  new URL("../scripts/research/", import.meta.url),
);
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" })
    .status === 0;

test(
  "risk-controlled historical runner is sealed and has fail-closed lifecycles",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import campaign_runner as common
import harness
import historical_datafeed as feed
import risk_controlled_reversal_campaign as core
import run_historical_risk_controlled_reversal_campaign as runner

registration = runner._registration()
assert registration["campaign"] == runner.CAMPAIGN_ID
assert runner.REGISTRATION_SHA256 == common._file_digest(runner.REGISTRATION_PATH)
assert registration["validation"]["lifetimeTrialCount"] == 39
assert registration["validation"]["bootstrapBlockBars"] == [21, 42, 63]
assert registration["riskPolicy"]["maximumTerminalTurnover"] == 1.25
assert runner.parse_args([]).predecessor_campaign_dir == (
    runner.PREDECESSOR_CAMPAIGN_DIRECTORY
)
assert not hasattr(runner.parse_args([]), "acquire")

tampered = copy.deepcopy(registration)
tampered["validation"]["lifetimeTrialCount"] = 38
try:
    runner._validate_registration(tampered)
except ValueError as error:
    assert "lifetimeTrialCount" in str(error)
else:
    raise AssertionError("changed lifetime accounting must fail closed")

mutations = (
    ("risk", "riskPolicy", "maximumTerminalTurnover", 1.3),
    (
        "predecessor pin",
        "predecessorEvidence",
        "campaignManifestSha256",
        "0" * 64,
    ),
    ("registry version", "holdoutPolicy", "registryVersion", 2),
    (
        "registry path",
        "holdoutPolicy",
        "registryDirectory",
        ".tmp/research/not-the-shared-registry",
    ),
    (
        "promotion",
        "promotion",
        "maximumChampionMeanTurnoverRatio",
        0.71,
    ),
    (
        "holdout rule",
        "holdoutPolicy",
        "chargeTerminalLiquidation",
        False,
    ),
)
for label, section, key, value in mutations:
    changed = copy.deepcopy(registration)
    changed[section][key] = value
    try:
        runner._validate_registration(changed)
    except ValueError:
        pass
    else:
        raise AssertionError(f"changed {label} must fail closed")

specs = core.campaign_specs(feed.CONTRACT_INTERVAL_MS)
assert [spec.trial_id for spec in specs if spec.champion_eligible] == [
    "resrev_24h_exit3_hysteresis",
    "resrev_72h_exit3_hysteresis",
    "resrev_168h_exit3_hysteresis",
]
assert len(specs) == 6

class FakeBreach(ValueError):
    def __init__(self, trial_id="resrev_24h_exit1_control"):
        self.trial_id = trial_id
        self.interval_left_close_time = 1_600_819_200_000 + feed.CONTRACT_INTERVAL_MS - 1
        self.outcome_close_time = self.interval_left_close_time + feed.CONTRACT_INTERVAL_MS
        self.reason = "maximum_activation_turnover"
        self.evidence = {
            "observed": 1.4,
            "limit": 1.25,
            "modeledImmediateLiquidationTurnover": 0.5,
            "modeledImmediateLiquidationCostFraction": 0.0005,
        }
        super().__init__(self.reason)

def fake_detail(index, treatment):
    rows = len(index)
    return pd.DataFrame(
        {
            "net": np.full(rows, 0.002 if treatment else 0.001),
            "gross": np.full(rows, 0.002 if treatment else 0.001),
            "turnover": np.full(rows, 0.1 if treatment else 0.2),
            "activationTurnover": np.full(rows, 0.1 if treatment else 0.2),
            "terminalTurnover": np.r_[np.zeros(max(0, rows - 1)), 0.5],
            "cost": np.full(rows, 0.0001),
            "active": np.full(rows, 2),
            "equity": np.cumprod(np.full(rows, 1.001)),
            "drawdown": np.zeros(rows),
            "endpointGrossLeverage": np.full(rows, 0.5),
            "maximumAbsoluteSymbolWeight": np.full(rows, 0.25),
            "shockEquityFraction": np.full(rows, 0.875),
            "shockedGrossLeverage": np.full(rows, 0.5),
            "shockMaintenanceCoverage": np.full(rows, 10.0),
            "weight_BTCUSDT": np.full(rows, 0.25),
            "weight_ETHUSDT": np.full(rows, -0.25),
        },
        index=index,
    )

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)

    # Predecessor evidence is both hash-pinned and semantically terminal. A
    # tampered byte must be rejected before that artifact reaches JSON parsing.
    predecessor_dir = root / "predecessor-integrity"
    predecessor_dir.mkdir()
    (predecessor_dir / ".campaign.lock").write_text("")
    predecessor_registration_path = (
        common.REPOSITORY_ROOT
        / "research-notes/registrations/residual-reversal-turnover-v1.json"
    )
    predecessor_registration_sha = common._file_digest(
        predecessor_registration_path
    )
    predecessor = copy.deepcopy(registration)
    predecessor_policy = predecessor["predecessorEvidence"]
    predecessor_policy["registrationSha256"] = predecessor_registration_sha
    predecessor_policy["implementationSha256"] = "d" * 64
    predecessor_policy["implementationArtifacts"] = {"runner.py": "e" * 64}
    expected_failure = {
        "reason": "portfolio_equity_exhausted",
        "trialId": "resrev_24h_rebalance_3bar",
        "closeTime": 10,
        "closeTimeSemantics": "interval_left_close",
        "outcomeCloseTime": 11,
    }
    expected_holdout = {
        "status": "reserved",
        "identitySha256": "f" * 64,
        "openRequested": False,
        "openBlockedBy": ["bankruptcyFree"],
        "rows": 1227,
    }
    predecessor_policy["requiredResult"] = {
        "status": "mechanically_invalid",
        "bankruptcyFree": False,
        "promotionGates": {"bankruptcyFree": False},
        "mechanicalFailure": expected_failure,
        "finalHoldout": expected_holdout,
    }
    predecessor_policy["requiredArtifactSet"] = [
        ".campaign.lock",
        "campaign-manifest.json",
        "mechanical-failure.json",
        "summary.json",
    ]
    predecessor_policy["forbiddenArtifacts"] = ["final-holdout-opened.json"]
    manifest = {
        "campaign": runner.V1.CAMPAIGN_ID,
        "registrationSha256": predecessor_registration_sha,
        "implementationSha256": "d" * 64,
        "implementationArtifacts": {"runner.py": "e" * 64},
    }
    manifest_path = predecessor_dir / "campaign-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = common._file_digest(manifest_path)
    failure = {
        "campaign": runner.V1.CAMPAIGN_ID,
        "registrationSha256": predecessor_registration_sha,
        "campaignManifestSha256": manifest_sha,
        "status": "mechanically_invalid",
        "bankruptcyFree": False,
        "mechanicalFailure": expected_failure,
        "finalHoldout": expected_holdout,
    }
    failure_path = predecessor_dir / "mechanical-failure.json"
    failure_path.write_text(json.dumps(failure))
    failure_sha = common._file_digest(failure_path)
    summary = {
        **failure,
        "promotionGates": {"bankruptcyFree": False},
        "evidence": {"mechanicalFailureSha256": failure_sha},
    }
    summary_path = predecessor_dir / "summary.json"
    summary_path.write_text(json.dumps(summary))
    predecessor_policy.update(
        {
            "campaignManifestSha256": manifest_sha,
            "mechanicalFailureSha256": failure_sha,
            "summarySha256": common._file_digest(summary_path),
        }
    )
    validated_predecessor = runner._validate_predecessor_evidence(
        predecessor_dir, predecessor
    )
    assert validated_predecessor["terminalStatus"] == "mechanically_invalid"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    parsed_paths = []
    original_json_parser = runner.V1._json_object_from_bytes
    runner.V1._json_object_from_bytes = lambda payload, path: (
        parsed_paths.append(path.name),
        original_json_parser(payload, path),
    )[1]
    try:
        try:
            runner._validate_predecessor_evidence(predecessor_dir, predecessor)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("tampered predecessor bytes must fail closed")
    finally:
        runner.V1._json_object_from_bytes = original_json_parser
    assert predecessor_registration_path.name in parsed_paths
    assert "campaign-manifest.json" not in parsed_paths

    synthetic = copy.deepcopy(registration)
    synthetic["validation"].update(
        {
            "featureWarmupRows": 0,
            "developmentEvaluationRows": 12,
            "outerInitialTrain": 4,
            "outerTestSize": 4,
            "outerFoldCount": 1,
            "nestedOuterOosRows": 12,
            "innerInitialTrain": 2,
            "innerTestSize": 2,
            "bootstrapReplications": 10,
            "pboSlices": 2,
        }
    )
    synthetic["promotion"].update(
        {
            "minimumNestedOuterOosObservations": 1,
            "minimumRegimeObservations": 1,
            "minimumPositiveOuterFolds": 1,
            "currentCampaignDeflatedSharpeProbabilityMinimum": 0.0,
            "lifetimeBonferroniPsrProbabilityMinimum": 0.0,
            "maximumPbo": 1.0,
        }
    )
    times = (
        core.REBALANCE_ANCHOR_OPEN_TIME
        + np.arange(12, dtype=np.int64) * feed.CONTRACT_INTERVAL_MS
    )
    index = pd.Index(times, name="openTime")
    details = {
        spec.trial_id: fake_detail(index, spec.champion_eligible)
        for spec in specs
    }
    matrix = pd.DataFrame(
        {name: detail["net"] for name, detail in details.items()}, index=index
    )
    outer_folds = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "selected_candidate": "resrev_24h_exit3_hysteresis",
                "train_start": 0,
                "train_stop": 0,
                "embargo_start": 0,
                "embargo_stop": 0,
                "test_start": 0,
                "test_stop": len(index),
                "selection_score": 1.0,
            }
        ]
    )
    nested = harness.NestedRollingResult(
        oos=pd.DataFrame({"net": np.full(len(index), 0.002)}),
        outer_folds=outer_folds,
        inner_scores=pd.DataFrame(
            {
                "outer_fold": [0],
                "candidate": ["resrev_24h_exit3_hysteresis"],
                "score": [1.0],
            }
        ),
    )

    def selected_frame():
        result = fake_detail(index, True).reset_index()
        result.insert(0, "selected_candidate", "resrev_24h_exit3_hysteresis")
        result.insert(0, "outer_fold", 0)
        result.insert(0, "row_position", np.arange(len(result)))
        return result

    original = {
        "registration": runner._registration_and_sha,
        "implementation": runner._implementation_artifacts,
        "predecessor": runner._validate_predecessor_evidence,
        "development": runner.V1._load_development_inputs,
        "trials": runner._trials_on_panel,
        "nested": runner._nested_selector,
        "champion": runner._final_champion,
        "selected": runner._selected_path,
        "fixed": runner._fixed_candidate_path,
        "stresses": runner._stress_paths,
        "bootstrap": runner._bootstrap_conjunction,
        "diagnostics": runner.C._diagnostics,
        "lifetime": runner._lifetime_multiple_testing,
        "regime_labels": runner.C._market_regime_labels,
        "regime": runner.C._regime_report,
        "inputs": runner._assert_inputs_unchanged,
        "reserve": runner.C._reserve_holdout,
        "snapshot": runner.V1._load_full_registered_inputs,
        "breach": runner.R.RiskConstraintBreach,
    }
    calls = []
    runner._registration_and_sha = lambda: (synthetic, "a" * 64)
    runner._implementation_artifacts = lambda: {}
    runner._validate_predecessor_evidence = lambda *_args: {"status": "validated"}
    runner.V1._load_development_inputs = lambda *_args: (
        {},
        [],
        pd.DataFrame(),
        {"resolvedFraction": 1.0},
        {"status": "validated"},
    )
    runner._trials_on_panel = lambda *_args: (matrix, details, specs)
    runner._nested_selector = lambda *_args: nested
    runner._final_champion = lambda *_args: (
        "resrev_24h_exit3_hysteresis",
        pd.DataFrame(
            {"candidate": ["resrev_24h_exit3_hysteresis"], "score": [1.0]}
        ),
        pd.DataFrame(
            {
                "fold": [0],
                "trainStart": [0],
                "trainStop": [1],
                "embargoStart": [1],
                "embargoStop": [2],
                "testStart": [2],
                "testStop": [12],
            }
        ),
    )
    runner._selected_path = lambda *_args, **_kwargs: selected_frame()
    runner._fixed_candidate_path = lambda *_args, **_kwargs: selected_frame()
    runner._stress_paths = lambda *_args, **_kwargs: {
        "cost2x": {
            "nestedOuterOos": selected_frame(),
            "finalChampion": selected_frame(),
        },
        "additionalDelay1bar": {
            "nestedOuterOos": selected_frame(),
            "finalChampion": selected_frame(),
        },
    }
    runner._bootstrap_conjunction = lambda *_args, **_kwargs: {
        "method": "circular_moving_block",
        "intervalsByBlockBars": {
            "21": [1.0, 2.0],
            "42": [1.0, 2.0],
            "63": [1.0, 2.0],
        },
        "allLowerBoundsAboveZero": True,
    }
    runner.C._diagnostics = lambda diagnostic_matrix, *_args, **_kwargs: (
        {
            "deflatedSharpe": {"probability": 1.0},
            "pbo": {"probability": 0.0},
        },
        diagnostic_matrix,
        diagnostic_matrix,
    )
    runner._lifetime_multiple_testing = lambda *_args: {"adjustedProbability": 1.0}
    runner.C._market_regime_labels = lambda *_args: pd.Series(
        "regime", index=index
    )
    runner.C._regime_report = lambda oos, *_args: (
        {"metrics": {"regime": {"observations": len(oos), "totalReturn": 0.01}}},
        True,
        oos.assign(regime="regime"),
    )
    input_checks = []
    runner._assert_inputs_unchanged = lambda *_args: input_checks.append("rehash")
    runner.C._reserve_holdout = lambda *_args, **_kwargs: calls.append("reserve")
    runner.V1._load_full_registered_inputs = lambda *_args: calls.append("snapshot")
    runner.R.RiskConstraintBreach = FakeBreach

    try:
        def arguments(name, open_requested):
            return argparse.Namespace(
                predecessor_campaign_dir=str(root / "predecessor"),
                source_campaign_dir=str(root / "source"),
                snapshot_dir=str(root / "snapshot"),
                output_dir=str(root / name),
                open_final_holdout=open_requested,
            )

        default = runner.run(arguments("default", False))
        assert default["status"] == "ready_for_final_holdout", default["promotionGates"]
        assert calls == []
        assert not (root / "default" / "evidence-index.json").exists()
        assert not (root / "default" / "final-holdout-opened.json").exists()

        synthetic["promotion"]["minimumSymbols"] = 11
        blocked = runner.run(arguments("blocked", True))
        assert blocked["status"] == "insufficient_evidence"
        assert blocked["finalHoldout"]["openBlockedBy"] == ["minimumSymbols"]
        assert calls == []
        assert (root / "blocked" / "evidence-index.json").is_file()
        assert not (root / "blocked" / "final-holdout-opened.json").exists()
        blocked_index = json.loads(
            (root / "blocked" / "evidence-index.json").read_text()
        )
        for artifact in (
            "primary-trial-returns",
            "primary-trial-paths",
            "final-champion-development",
            "stress-cost2x-nested-outer-oos",
            "stress-cost2x-final-champion",
            "stress-additional-delay1bar-nested-outer-oos",
            "stress-additional-delay1bar-final-champion",
        ):
            assert artifact in blocked_index["artifacts"]
        checks_before_repeat = len(input_checks)
        repeated_blocked = runner.run(arguments("blocked", True))
        assert repeated_blocked == blocked
        assert len(input_checks) == checks_before_repeat + 1
        unexpected = root / "blocked" / "unregistered.txt"
        unexpected.write_text("not indexed")
        try:
            runner.run(arguments("blocked", True))
        except ValueError as error:
            assert "artifact set changed" in str(error)
        else:
            raise AssertionError("unexpected terminal artifacts must fail closed")
        unexpected.unlink()

        unexpected_directory_output = root / "unexpected-directory"
        unexpected_directory_output.mkdir()
        for name in (
            "campaign-manifest.json",
            "risk-ledger.json",
            "risk-failure.json",
        ):
            (unexpected_directory_output / name).write_text("{}\n")
        (unexpected_directory_output / "unregistered").mkdir()
        try:
            runner._finalize_terminal(
                unexpected_directory_output,
                {"status": "risk_invalid"},
                "a" * 64,
                "b" * 64,
            )
        except ValueError as error:
            assert "unexpected artifact set" in str(error)
        else:
            raise AssertionError("unexpected terminal directories must fail closed")
        assert not (unexpected_directory_output / "evidence-index.json").exists()
        synthetic["promotion"]["minimumSymbols"] = 10

        interrupted_output = root / "interrupted"
        interrupted_output.mkdir()
        (interrupted_output / "final-holdout-opened.json").write_text(
            json.dumps({"status": "opening"})
        )
        try:
            runner.run(arguments("interrupted", False))
        except ValueError as error:
            assert "consumed or interrupted" in str(error)
        else:
            raise AssertionError("an opening record must remain consumed")
        assert calls == []

        shared_only_output = root / "shared-only-output"
        shared_only_registry = root / "shared-only-registry"
        shared_only_registry.mkdir()
        window = common._holdout_window(
            synthetic["universe"]["symbols"],
            feed.CONTRACT_INTERVAL,
            synthetic["registeredData"]["holdoutStartOpenTime"],
            synthetic["registeredData"]["endOpenTime"],
        )
        identity = common._json_digest(
            {
                "campaign": runner.CAMPAIGN_ID,
                "panelSha256": synthetic["registeredData"][
                    "fullPanelDigestSha256"
                ],
                "window": window,
            }
        )
        shared_record = {
            "registryVersion": common.HOLDOUT_REGISTRY_VERSION,
            "status": "opening",
            "campaign": runner.CAMPAIGN_ID,
            "registrationSha256": "a" * 64,
            "campaignManifestSha256": "b" * 64,
            "holdoutIdentitySha256": identity,
            "panelSha256": synthetic["registeredData"][
                "fullPanelDigestSha256"
            ],
            "window": window,
            "artifacts": {"outputDirectory": str(shared_only_output.resolve())},
        }
        (shared_only_registry / f"{identity}.json").write_text(
            json.dumps(shared_record)
        )
        saved_registry = runner.HOLDOUT_REGISTRY_DIR
        saved_override = runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE
        runner.HOLDOUT_REGISTRY_DIR = shared_only_registry
        runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = True
        try:
            try:
                runner.run(arguments("shared-only-output", False))
            except ValueError as error:
                assert "already consumed" in str(error)
            else:
                raise AssertionError("a shared-only opening must remain consumed")
        finally:
            runner.HOLDOUT_REGISTRY_DIR = saved_registry
            runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = saved_override
        assert not (shared_only_output / "campaign-manifest.json").exists()
        assert calls == []

        runner._trials_on_panel = lambda *_args: (_ for _ in ()).throw(FakeBreach())
        risk_invalid = runner.run(arguments("risk-invalid", True))
        assert risk_invalid["status"] == "risk_invalid"
        assert risk_invalid["riskFailure"]["reason"] == (
            "maximum_activation_turnover"
        )
        assert risk_invalid["riskFailure"][
            "immediateCloseLiquidationEvidence"
        ]["modeledImmediateLiquidationTurnover"] == 0.5
        assert calls == []
        assert not (root / "risk-invalid" / "final-holdout-opened.json").exists()
        risk_index = json.loads(
            (root / "risk-invalid" / "evidence-index.json").read_text()
        )
        assert "risk-failure" in risk_index["artifacts"]
    finally:
        runner._registration_and_sha = original["registration"]
        runner._implementation_artifacts = original["implementation"]
        runner._validate_predecessor_evidence = original["predecessor"]
        runner.V1._load_development_inputs = original["development"]
        runner._trials_on_panel = original["trials"]
        runner._nested_selector = original["nested"]
        runner._final_champion = original["champion"]
        runner._selected_path = original["selected"]
        runner._fixed_candidate_path = original["fixed"]
        runner._stress_paths = original["stresses"]
        runner._bootstrap_conjunction = original["bootstrap"]
        runner.C._diagnostics = original["diagnostics"]
        runner._lifetime_multiple_testing = original["lifetime"]
        runner.C._market_regime_labels = original["regime_labels"]
        runner.C._regime_report = original["regime"]
        runner._assert_inputs_unchanged = original["inputs"]
        runner.C._reserve_holdout = original["reserve"]
        runner.V1._load_full_registered_inputs = original["snapshot"]
        runner.R.RiskConstraintBreach = original["breach"]

    # Exercise the real strict reservation and shared-first completion order.
    holdout_registration = copy.deepcopy(registration)
    holdout_start = core.REBALANCE_ANCHOR_OPEN_TIME
    holdout_registration["registeredData"].update(
        {
            "holdoutStartOpenTime": holdout_start,
            "endOpenTime": holdout_start + 2 * feed.CONTRACT_INTERVAL_MS,
            "outcomeEndTimeExclusive": holdout_start + 3 * feed.CONTRACT_INTERVAL_MS,
            "holdoutReturnRows": 3,
        }
    )
    full_times = holdout_start + np.arange(4, dtype=np.int64) * feed.CONTRACT_INTERVAL_MS
    full_panel = {
        symbol: pd.DataFrame(
            {
                "openTime": full_times,
                "closeTime": full_times + feed.CONTRACT_INTERVAL_MS - 1,
                "close": np.full(len(full_times), 100.0 + number),
            }
        )
        for number, symbol in enumerate(registration["universe"]["symbols"])
    }
    original_inputs = runner._assert_inputs_unchanged
    original_loader = runner.V1._load_full_registered_inputs
    original_fixed = runner._fixed_candidate_path
    original_bootstrap = runner._bootstrap_conjunction
    original_registry = runner.HOLDOUT_REGISTRY_DIR
    original_registry_override = runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE
    original_breach = runner.R.RiskConstraintBreach
    runner._assert_inputs_unchanged = lambda *_args: None
    runner._bootstrap_conjunction = lambda *_args, **_kwargs: {
        "allLowerBoundsAboveZero": True,
        "intervalsByBlockBars": {"21": [1.0, 2.0], "42": [1.0, 2.0], "63": [1.0, 2.0]},
    }
    runner.R.RiskConstraintBreach = FakeBreach
    runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = True
    try:
        for outcome in ("success", "breach", "integrity", "execution"):
            output = root / f"holdout-{outcome}"
            output.mkdir()
            registry = root / f"registry-{outcome}"
            runner.HOLDOUT_REGISTRY_DIR = registry
            observations = []
            integrity_checks = []

            def check_holdout_inputs(*_args):
                integrity_checks.append("rehash")
                if outcome == "integrity" and len(integrity_checks) == 2:
                    raise ValueError("synthetic concurrent input change")

            runner._assert_inputs_unchanged = check_holdout_inputs

            def load_after_reservation(*_args):
                markers = list(registry.glob("*.json"))
                assert len(markers) == 1
                marker = json.loads(markers[0].read_text())
                local = json.loads((output / "final-holdout-opened.json").read_text())
                assert marker == local
                assert marker["status"] == "opening"
                assert not (output / "final-holdout-result.json").exists()
                observations.append("reserved-before-snapshot")
                if outcome == "execution":
                    raise ValueError("synthetic snapshot failure")
                return full_panel, [], {"resolvedFraction": 1.0}

            runner.V1._load_full_registered_inputs = load_after_reservation
            if outcome in {"success", "integrity"}:
                def successful_path(*_args, **_kwargs):
                    frame = fake_detail(pd.Index(full_times[:3], name="openTime"), True)
                    frame["net"] = 0.001
                    frame["active"] = 2
                    result = frame.reset_index()
                    result.insert(0, "selected_candidate", "resrev_24h_exit3_hysteresis")
                    result.insert(0, "outer_fold", 0)
                    result.insert(0, "row_position", np.arange(3))
                    return result
                runner._fixed_candidate_path = successful_path
            elif outcome == "breach":
                runner._fixed_candidate_path = lambda *_args, **_kwargs: (
                    _ for _ in ()
                ).throw(FakeBreach("resrev_24h_exit3_hysteresis"))

            result = runner._open_final_holdout(
                argparse.Namespace(open_final_holdout=True),
                output,
                root / "predecessor",
                root / "source",
                root / "snapshot",
                holdout_registration,
                "a" * 64,
                "c" * 64,
                "b" * 64,
                "resrev_24h_exit3_hysteresis",
                specs,
                common._periods_per_year(feed.CONTRACT_INTERVAL_MS),
            )
            assert observations == ["reserved-before-snapshot"]
            local = json.loads((output / "final-holdout-opened.json").read_text())
            marker = json.loads(next(registry.glob("*.json")).read_text())
            assert local == marker
            assert local["status"] == "completed"
            result_record = json.loads((output / "final-holdout-result.json").read_text())
            assert result_record["status"] == "evaluated"
            if outcome == "success":
                assert result["status"] == "pass"
                assert result_record["artifacts"]["returnsWritten"] is True
                assert (output / "final-holdout-returns.csv").is_file()
                terminal_summary = {
                    "status": "final_holdout_passed",
                    "finalHoldout": result,
                }
                runner._validate_completed_holdout_registry(
                    output, terminal_summary
                )
                marker_path = next(registry.glob("*.json"))
                marker_bytes = marker_path.read_bytes()
                marker_path.unlink()
                try:
                    runner._validate_completed_holdout_registry(
                        output, terminal_summary
                    )
                except ValueError as error:
                    assert "marker is missing" in str(error)
                else:
                    raise AssertionError("a missing shared marker must fail closed")
                marker_path.write_bytes(marker_bytes)
                changed_marker = json.loads(marker_path.read_text())
                changed_marker["candidate"] = "tampered"
                marker_path.write_text(json.dumps(changed_marker))
                try:
                    runner._validate_completed_holdout_registry(
                        output, terminal_summary
                    )
                except ValueError as error:
                    assert "records differ" in str(error)
                else:
                    raise AssertionError("a changed shared marker must fail closed")
                marker_path.write_bytes(marker_bytes)
            elif outcome == "breach":
                assert result["status"] == "fail"
                assert result["evaluationStatus"] == "risk_breach"
                assert result_record["artifacts"]["returnsWritten"] is False
                assert not (output / "final-holdout-returns.csv").exists()
            elif outcome == "integrity":
                assert result["status"] == "fail"
                assert result["evaluationStatus"] == "execution_error"
                assert result["failure"]["reason"] == (
                    "holdout_input_integrity_changed"
                )
                assert integrity_checks == ["rehash", "rehash"]
                assert result_record["artifacts"]["returnsWritten"] is False
                assert not (output / "final-holdout-returns.csv").exists()
            else:
                assert result["status"] == "fail"
                assert result["evaluationStatus"] == "execution_error"
                assert result["failure"]["reason"] == "holdout_execution_error"
                assert result_record["artifacts"]["returnsWritten"] is False
                assert not (output / "final-holdout-returns.csv").exists()
    finally:
        runner._assert_inputs_unchanged = original_inputs
        runner.V1._load_full_registered_inputs = original_loader
        runner._fixed_candidate_path = original_fixed
        runner._bootstrap_conjunction = original_bootstrap
        runner.HOLDOUT_REGISTRY_DIR = original_registry
        runner.TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = original_registry_override
        runner.R.RiskConstraintBreach = original_breach
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
    });
    assert.equal(run.status, 0, run.stderr);
  },
);
