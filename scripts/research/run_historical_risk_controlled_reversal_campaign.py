#!/usr/bin/env python3
"""Run the locked risk-controlled residual-reversal campaign.

Development runs consume only the predecessor campaign's hash-pinned CSV
artifacts. The raw snapshot is not read unless all development gates pass and
``--open-final-holdout`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import campaign_runner as C
import diagnostics
import funding_campaign as F
import harness as W
import historical_datafeed as feed
import risk_controlled_reversal_campaign as R
import run_historical_funding_campaign as H
import run_historical_reversal_campaign as V1


CAMPAIGN_ID = "residual_reversal_rank_hysteresis_risk_v1"
REGISTRATION_VERSION = 1
REGISTRATION_SHA256 = (
    "9491cb3ddb94ce346900872707cf393c62339cec410e21893f20cb2318fe701d"
)
REGISTRATION_PATH = (
    C.REPOSITORY_ROOT
    / "research-notes/registrations/residual-reversal-rank-hysteresis-risk-v1.json"
)
SOURCE_CAMPAIGN_ID = H.CAMPAIGN_ID
SOURCE_CAMPAIGN_DIRECTORY = ".tmp/research/historical-funding-campaign-v1"
PREDECESSOR_CAMPAIGN_DIRECTORY = ".tmp/research/historical-reversal-campaign-v1"
SNAPSHOT_DIRECTORY = ".tmp/research/historical-funding-snapshot-v1"
OUTPUT_DIRECTORY = ".tmp/research/historical-risk-controlled-reversal-campaign-v1"
HOLDOUT_REGISTRY_DIR = C.HOLDOUT_REGISTRY_DIR
REGISTERED_HOLDOUT_REGISTRY_DIR = (
    C.REPOSITORY_ROOT / ".tmp/research/edge-campaign-holdouts"
)
TEST_ONLY_ALLOW_REGISTRY_OVERRIDE = False

SOURCE_CAMPAIGN_MANIFEST_SHA256 = V1.SOURCE_CAMPAIGN_MANIFEST_SHA256
SOURCE_REGISTRATION_SHA256 = V1.SOURCE_REGISTRATION_SHA256
SNAPSHOT_MANIFEST_SHA256 = V1.SNAPSHOT_MANIFEST_SHA256
DEVELOPMENT_PANEL_SHA256 = V1.DEVELOPMENT_PANEL_SHA256
DEVELOPMENT_SETTLEMENTS_SHA256 = V1.DEVELOPMENT_SETTLEMENTS_SHA256
FULL_PANEL_DIGEST_SHA256 = V1.FULL_PANEL_DIGEST_SHA256
FULL_SETTLEMENTS_DIGEST_SHA256 = V1.FULL_SETTLEMENTS_DIGEST_SHA256
REGISTERED_SYMBOLS = V1.REGISTERED_SYMBOLS
REGISTERED_START_OPEN_TIME = V1.REGISTERED_START_OPEN_TIME
REGISTERED_END_OPEN_TIME = V1.REGISTERED_END_OPEN_TIME
REGISTERED_OUTCOME_END_EXCLUSIVE = V1.REGISTERED_OUTCOME_END_EXCLUSIVE
REGISTERED_DEVELOPMENT_ROWS = V1.REGISTERED_DEVELOPMENT_ROWS
REGISTERED_HOLDOUT_RETURN_ROWS = V1.REGISTERED_HOLDOUT_RETURN_ROWS

IMPLEMENTATION_FILES = (
    "campaign_runner.py",
    "diagnostics.py",
    "funding_campaign.py",
    "harness.py",
    "historical_datafeed.py",
    "reversal_campaign.py",
    "risk_controlled_reversal_campaign.py",
    "run_historical_funding_campaign.py",
    "run_historical_reversal_campaign.py",
    "run_historical_risk_controlled_reversal_campaign.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked risk-controlled residual-reversal campaign"
    )
    parser.add_argument(
        "--predecessor-campaign-dir",
        default=PREDECESSOR_CAMPAIGN_DIRECTORY,
        help="Pinned rejected predecessor campaign evidence directory",
    )
    parser.add_argument(
        "--source-campaign-dir",
        default=SOURCE_CAMPAIGN_DIRECTORY,
        help="Pinned predecessor campaign evidence directory",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=SNAPSHOT_DIRECTORY,
        help="Pinned raw snapshot directory, opened only after promotion",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIRECTORY,
        help="Registered risk-controlled campaign evidence directory",
    )
    parser.add_argument("--open-final-holdout", action="store_true")
    return parser.parse_args(argv)


def _read_json_object(path: Path) -> dict[str, object]:
    return V1._read_json_object(path)


def _registration() -> dict[str, object]:
    payload, _ = V1._read_pinned_bytes(REGISTRATION_PATH, REGISTRATION_SHA256)
    registration = V1._json_object_from_bytes(payload, REGISTRATION_PATH)
    _validate_registration(registration)
    return registration


def _registration_and_sha() -> tuple[dict[str, object], str]:
    registration = _registration()
    return registration, REGISTRATION_SHA256


def _implementation_artifacts() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: C._file_digest(root / name) for name in IMPLEMENTATION_FILES}


def _implementation_sha() -> str:
    return C._json_digest(_implementation_artifacts())


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"campaign registration has invalid {key} settings")
    return value


def _require_exact(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"registered {label} changed")


def _validate_registration(registration: Mapping[str, object]) -> None:
    """Fail closed if any locked campaign definition changes."""
    _require_exact(registration.get("campaign"), CAMPAIGN_ID, "campaign identity")
    _require_exact(
        registration.get("registrationVersion"),
        REGISTRATION_VERSION,
        "registration version",
    )
    _require_exact(
        registration.get("outputDirectory"), OUTPUT_DIRECTORY, "output directory"
    )
    universe = _require_mapping(registration, "universe")
    data = _require_mapping(registration, "registeredData")
    strategy = _require_mapping(registration, "strategy")
    risk = _require_mapping(registration, "riskPolicy")
    validation = _require_mapping(registration, "validation")
    promotion = _require_mapping(registration, "promotion")
    holdout = _require_mapping(registration, "holdoutPolicy")

    expected_universe = {
        "interval": feed.CONTRACT_INTERVAL,
        "intervalMilliseconds": feed.CONTRACT_INTERVAL_MS,
        "symbols": list(REGISTERED_SYMBOLS),
    }
    for key, expected in expected_universe.items():
        _require_exact(universe.get(key), expected, f"universe {key}")

    expected_data = {
        "sourceCampaign": SOURCE_CAMPAIGN_ID,
        "sourceCampaignDirectory": SOURCE_CAMPAIGN_DIRECTORY,
        "sourceCampaignManifest": V1.SOURCE_CAMPAIGN_MANIFEST,
        "sourceCampaignManifestSha256": SOURCE_CAMPAIGN_MANIFEST_SHA256,
        "sourceRegistrationSha256": SOURCE_REGISTRATION_SHA256,
        "snapshotDirectory": SNAPSHOT_DIRECTORY,
        "snapshotManifest": "snapshot-manifest.json",
        "snapshotManifestSha256": SNAPSHOT_MANIFEST_SHA256,
        "developmentPanel": "registered-development-panel.csv",
        "developmentPanelSha256": DEVELOPMENT_PANEL_SHA256,
        "developmentSettlements": "registered-development-settlements.csv",
        "developmentSettlementsSha256": DEVELOPMENT_SETTLEMENTS_SHA256,
        "fullPanelDigestSha256": FULL_PANEL_DIGEST_SHA256,
        "fullSettlementsDigestSha256": FULL_SETTLEMENTS_DIGEST_SHA256,
        "startOpenTime": REGISTERED_START_OPEN_TIME,
        "endOpenTime": REGISTERED_END_OPEN_TIME,
        "outcomeEndTimeExclusive": REGISTERED_OUTCOME_END_EXCLUSIVE,
        "rows": 6138,
        "developmentRows": REGISTERED_DEVELOPMENT_ROWS,
        "developmentCutoffOpenTime": 1742198400000,
        "holdoutStartOpenTime": 1742227200000,
        "holdoutBars": 1228,
        "holdoutReturnRows": REGISTERED_HOLDOUT_RETURN_ROWS,
    }
    for key, expected in expected_data.items():
        _require_exact(data.get(key), expected, f"registeredData {key}")

    expected_strategy = {
        "direction": "residual_reversal",
        "betaLookbackBars": 21,
        "costBpsPerUnitTurnover": 10.0,
        "grossExposure": 0.5,
        "entryRank": 1,
        "hysteresisExitRank": 3,
        "controlExitRank": 1,
        "decisionCadenceBars": 1,
        "rebalanceAnchorOpenTime": REGISTERED_START_OPEN_TIME,
        "signalDelayBars": 1,
        "topNPerSide": 1,
        "doubledCostBpsPerUnitTurnover": 20.0,
        "chargeInitialCashToTargetTurnover": True,
        "chargeTerminalLiquidation": True,
    }
    for key, expected in expected_strategy.items():
        _require_exact(strategy.get(key), expected, f"strategy {key}")

    expected_risk = {
        "minimumIntervalEquityFactorExclusive": 0.0,
        "minimumCumulativeEquity": 0.75,
        "maximumDrawdown": 0.20,
        "maximumEndpointGrossExposure": 1.0,
        "maximumAbsoluteSymbolExposure": 0.40,
        "maximumActivationTurnover": 1.25,
        "maximumTerminalTurnover": 1.25,
        "adverseShockFraction": 0.25,
        "minimumShockedEquityFraction": 0.50,
        "maintenanceMarginRate": 0.10,
        "liquidationReserveRate": 0.01,
        "minimumShockedMarginCoverage": 2.0,
    }
    for key, expected in expected_risk.items():
        _require_exact(risk.get(key), expected, f"riskPolicy {key}")

    expected_validation = {
        "bootstrapBlockBars": [21, 42, 63],
        "bootstrapReplications": 10000,
        "bootstrapSeed": 20260715,
        "developmentEvaluationRows": 4888,
        "featureWarmupRows": 21,
        "innerInitialTrain": 1222,
        "innerTestSize": 244,
        "labelHorizonBars": 1,
        "lifetimeTrialCount": 39,
        "newTrialCount": 6,
        "outerInitialTrain": 2444,
        "outerFoldCount": 7,
        "outerTestSize": 349,
        "pairedComparisonHypotheses": 3,
        "pairedComparisonFamilyWiseAlpha": 0.05,
        "pboSlices": 10,
        "currentCampaignTrialCount": 6,
        "priorTrialCount": 33,
    }
    for key, expected in expected_validation.items():
        _require_exact(validation.get(key), expected, f"validation {key}")

    expected_promotion = {
        "currentCampaignDeflatedSharpeProbabilityMinimum": 0.95,
        "lifetimeBonferroniPsrProbabilityMinimum": 0.95,
        "maximumPbo": 0.20,
        "maximumChampionMeanTurnoverRatio": 0.70,
        "maximumChampionOuterFoldTurnoverRatio": 0.85,
        "minimumPositiveOuterFolds": 5,
        "maximumRegimeLoss": 0.05,
        "maximumWorstOuterFoldLoss": 0.05,
        "minimumActiveFraction": 0.50,
        "minimumNestedOuterOosObservations": 2443,
        "minimumRegimeObservations": 100,
        "minimumResolvedFundingFraction": 1.0,
        "minimumSymbols": 10,
    }
    for key, expected in expected_promotion.items():
        _require_exact(promotion.get(key), expected, f"promotion {key}")

    expected_holdout = {
        "executionStateAtStart": "cash",
        "featureHistory": "full_registered_history_through_holdout_decision_time",
        "chargeInitialCashToFrozenTargetTurnover": True,
        "chargeTerminalLiquidation": True,
        "openOnlyAfterEveryDevelopmentGatePasses": True,
        "overlapAwareOneShotRegistry": True,
        "reservedByDefault": True,
        "registryVersion": C.HOLDOUT_REGISTRY_VERSION,
        "legacyRegistryCampaigns": sorted(C.LEGACY_HOLDOUT_IDENTITY_CAMPAIGNS),
        "registryDirectory": ".tmp/research/edge-campaign-holdouts",
        "outputBindingFormula": (
            "SHA-256 of canonical JSON containing holdoutIdentitySha256 and "
            "the absolute resolved outputDirectory"
        ),
    }
    for key, expected in expected_holdout.items():
        _require_exact(holdout.get(key), expected, f"holdoutPolicy {key}")

    specs = R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
    registered_trials = registration.get("trials")
    if not isinstance(registered_trials, list) or len(registered_trials) != 6:
        raise ValueError("registered six-trial ledger changed")
    expected_trials = [spec.to_dict() for spec in specs]
    if registered_trials != expected_trials:
        raise ValueError("registered trial ledger differs from implementation")
    canonical_payload, _ = V1._read_pinned_bytes(
        REGISTRATION_PATH, REGISTRATION_SHA256
    )
    canonical = V1._json_object_from_bytes(canonical_payload, REGISTRATION_PATH)
    if dict(registration) != canonical:
        raise ValueError("campaign registration differs from the locked document")


def _campaign_manifest(
    output_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_artifacts: Mapping[str, str],
    source_evidence: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    path = output_dir / "campaign-manifest.json"
    expected = {
        "campaign": CAMPAIGN_ID,
        "registrationVersion": REGISTRATION_VERSION,
        "registrationSha256": registration_sha,
        "implementationSha256": C._json_digest(implementation_artifacts),
        "implementationArtifacts": dict(implementation_artifacts),
        "sourceArtifacts": dict(source_evidence),
        "symbols": list(registration["universe"]["symbols"]),
        "trials": list(registration["trials"]),
        "strategy": dict(registration["strategy"]),
        "riskPolicy": dict(registration["riskPolicy"]),
        "registeredData": dict(registration["registeredData"]),
    }
    if path.exists():
        observed = _read_json_object(path)
        if observed != expected:
            raise ValueError("campaign manifest changed; use a new output directory")
        expected_payload = (
            json.dumps(expected, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if path.read_bytes() != expected_payload:
            raise ValueError("campaign manifest bytes changed; use a new directory")
    else:
        if any(output_dir.iterdir()):
            allowed = {".campaign.lock"}
            if any(item.name not in allowed for item in output_dir.iterdir()):
                raise ValueError(
                    "output directory has artifacts but no manifest; "
                    "use a new directory"
                )
        C._write_json_exclusive(path, expected)
    return expected, C._file_digest(path)


def _assert_inputs_unchanged(
    predecessor_dir: Path,
    source_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
) -> None:
    if registration_sha != REGISTRATION_SHA256:
        raise ValueError("campaign registration identity changed during this run")
    if C._file_digest(REGISTRATION_PATH) != registration_sha:
        raise ValueError("campaign registration changed during this run")
    if _implementation_sha() != implementation_sha:
        raise ValueError("campaign implementation changed during this run")
    _validate_predecessor_evidence(predecessor_dir, registration)
    data = _require_mapping(registration, "registeredData")
    V1._require_file_digest(
        H.REGISTRATION_PATH, data["sourceRegistrationSha256"]
    )
    for name_key, sha_key in (
        ("sourceCampaignManifest", "sourceCampaignManifestSha256"),
        ("developmentPanel", "developmentPanelSha256"),
        ("developmentSettlements", "developmentSettlementsSha256"),
    ):
        V1._require_file_digest(
            V1._artifact_path(source_dir, data[name_key]), data[sha_key]
        )


def _validate_predecessor_evidence(
    predecessor_dir: Path, registration: Mapping[str, object]
) -> dict[str, object]:
    """Verify the rejected campaign's bytes and terminal semantics first."""
    predecessor = _require_mapping(registration, "predecessorEvidence")
    _require_exact(
        predecessor.get("campaign"),
        V1.CAMPAIGN_ID,
        "predecessor campaign identity",
    )

    registration_path_value = predecessor.get("registration")
    if not isinstance(registration_path_value, str):
        raise ValueError("predecessor registration path is invalid")
    predecessor_registration_path = C.REPOSITORY_ROOT / registration_path_value
    predecessor_registration_payload, predecessor_registration_sha = (
        V1._read_pinned_bytes(
            predecessor_registration_path, predecessor["registrationSha256"]
        )
    )
    predecessor_registration = V1._json_object_from_bytes(
        predecessor_registration_payload, predecessor_registration_path
    )
    if predecessor_registration.get("campaign") != V1.CAMPAIGN_ID:
        raise ValueError("predecessor registration campaign changed")
    _require_exact(
        predecessor.get("directory"),
        PREDECESSOR_CAMPAIGN_DIRECTORY,
        "predecessor directory",
    )

    manifest_path = V1._artifact_path(
        predecessor_dir, predecessor["campaignManifest"]
    )
    failure_path = V1._artifact_path(
        predecessor_dir, predecessor["mechanicalFailure"]
    )
    summary_path = V1._artifact_path(predecessor_dir, predecessor["summary"])
    manifest_payload, manifest_sha = V1._read_pinned_bytes(
        manifest_path, predecessor["campaignManifestSha256"]
    )
    failure_payload, failure_sha = V1._read_pinned_bytes(
        failure_path, predecessor["mechanicalFailureSha256"]
    )
    summary_payload, summary_sha = V1._read_pinned_bytes(
        summary_path, predecessor["summarySha256"]
    )
    manifest = V1._json_object_from_bytes(manifest_payload, manifest_path)
    failure = V1._json_object_from_bytes(failure_payload, failure_path)
    summary = V1._json_object_from_bytes(summary_payload, summary_path)

    required_files = predecessor.get("requiredArtifactSet")
    forbidden_files = predecessor.get("forbiddenArtifacts")
    if not isinstance(required_files, list) or not isinstance(forbidden_files, list):
        raise ValueError("predecessor artifact policy is invalid")
    observed_files = sorted(path.name for path in predecessor_dir.iterdir())
    if observed_files != sorted(str(name) for name in required_files):
        raise ValueError("predecessor terminal artifact set changed")
    if any((predecessor_dir / str(name)).exists() for name in forbidden_files):
        raise ValueError("predecessor contains a forbidden holdout artifact")

    expected_manifest = {
        "campaign": V1.CAMPAIGN_ID,
        "registrationSha256": predecessor["registrationSha256"],
        "implementationSha256": predecessor["implementationSha256"],
        "implementationArtifacts": dict(predecessor["implementationArtifacts"]),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"predecessor manifest {key} changed")
    required_result = _require_mapping(predecessor, "requiredResult")
    expected_failure = _require_mapping(required_result, "mechanicalFailure")
    expected_holdout = _require_mapping(required_result, "finalHoldout")
    checks = {
        "failure campaign": failure.get("campaign") == V1.CAMPAIGN_ID,
        "failure manifest link": failure.get("campaignManifestSha256")
        == manifest_sha,
        "failure registration link": failure.get("registrationSha256")
        == predecessor["registrationSha256"],
        "failure status": failure.get("status") == required_result["status"],
        "failure bankruptcy": failure.get("bankruptcyFree")
        == required_result["bankruptcyFree"],
        "failure payload": failure.get("mechanicalFailure")
        == dict(expected_failure),
        "failure holdout": all(
            isinstance(failure.get("finalHoldout"), Mapping)
            and failure["finalHoldout"].get(key) == expected
            for key, expected in expected_holdout.items()
        ),
        "summary campaign": summary.get("campaign") == V1.CAMPAIGN_ID,
        "summary manifest link": summary.get("campaignManifestSha256")
        == manifest_sha,
        "summary registration link": summary.get("registrationSha256")
        == predecessor["registrationSha256"],
        "summary status": summary.get("status") == required_result["status"],
        "summary bankruptcy": summary.get("bankruptcyFree")
        == required_result["bankruptcyFree"],
        "summary gates": summary.get("promotionGates")
        == required_result["promotionGates"],
        "summary failure": summary.get("mechanicalFailure")
        == dict(expected_failure),
        "summary failure hash": isinstance(summary.get("evidence"), Mapping)
        and summary["evidence"].get("mechanicalFailureSha256") == failure_sha,
        "summary holdout": all(
            isinstance(summary.get("finalHoldout"), Mapping)
            and summary["finalHoldout"].get(key) == expected
            for key, expected in expected_holdout.items()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "predecessor terminal evidence violates registration: "
            + ", ".join(failed)
        )
    return {
        "directory": str(predecessor_dir),
        "registration": str(predecessor_registration_path),
        "registrationSha256": predecessor_registration_sha,
        "campaignManifest": manifest_path.name,
        "campaignManifestSha256": manifest_sha,
        "mechanicalFailure": failure_path.name,
        "mechanicalFailureSha256": failure_sha,
        "summary": summary_path.name,
        "summarySha256": summary_sha,
        "terminalStatus": summary["status"],
    }


def _terminal_evidence_index(
    output_dir: Path,
    status: str,
    artifacts: Mapping[str, Path],
    registration_sha: str,
    campaign_manifest_sha: str,
) -> dict[str, object]:
    path = output_dir / "evidence-index.json"
    if path.exists():
        raise ValueError("campaign already has immutable terminal evidence")
    indexed = {
        name: {
            "path": str(artifact.resolve()),
            "sha256": C._file_digest(artifact),
        }
        for name, artifact in sorted(artifacts.items())
    }
    record = {
        "campaign": CAMPAIGN_ID,
        "status": status,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "artifacts": indexed,
    }
    C._write_json_exclusive(path, record)
    return record


def _strategy_config(
    registration: Mapping[str, object],
    *,
    cost_multiplier: float = 1.0,
    additional_delay: int = 0,
) -> R.RiskControlledReversalConfig:
    strategy = _require_mapping(registration, "strategy")
    risk = _require_mapping(registration, "riskPolicy")
    return R.RiskControlledReversalConfig(
        interval_ms=feed.CONTRACT_INTERVAL_MS,
        rebalance_anchor_open_time=int(strategy["rebalanceAnchorOpenTime"]),
        gross_exposure=float(strategy["grossExposure"]),
        cost_per_turnover=float(strategy["costBpsPerUnitTurnover"])
        / 10_000
        * cost_multiplier,
        signal_delay_bars=int(strategy["signalDelayBars"]) + additional_delay,
        equity_floor=float(risk["minimumCumulativeEquity"]),
        maximum_drawdown=float(risk["maximumDrawdown"]),
        maximum_endpoint_gross_leverage=float(
            risk["maximumEndpointGrossExposure"]
        ),
        maximum_symbol_weight=float(risk["maximumAbsoluteSymbolExposure"]),
        maximum_activation_turnover=float(risk["maximumActivationTurnover"]),
        maximum_terminal_turnover=float(risk["maximumTerminalTurnover"]),
        adverse_shock_fraction=float(risk["adverseShockFraction"]),
        maintenance_margin_rate=float(risk["maintenanceMarginRate"]),
        liquidation_reserve_rate=float(risk["liquidationReserveRate"]),
        minimum_shock_equity_fraction=float(risk["minimumShockedEquityFraction"]),
        minimum_shock_maintenance_coverage=float(
            risk["minimumShockedMarginCoverage"]
        ),
        charge_terminal_liquidation=bool(strategy["chargeTerminalLiquidation"]),
    )


def _trial_inputs(
    panel: Mapping[str, pd.DataFrame], registration: Mapping[str, object]
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    strategy = _require_mapping(registration, "strategy")
    close = H._close_frame(panel)
    horizons = tuple(
        dict.fromkeys(
            spec.horizon_hours
            for spec in R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
        )
    )
    residual = H._residual_momentum(
        close, int(strategy["betaLookbackBars"]), horizons
    )
    return close, residual


def _adapt_close_time_detail(frame: pd.DataFrame) -> pd.DataFrame:
    adapted = frame.reset_index()
    if "closeTime" not in adapted:
        raise ValueError("risk-controlled path is not indexed by closeTime")
    adapted["openTime"] = (
        pd.to_numeric(adapted["closeTime"], errors="raise").astype(np.int64)
        - feed.CONTRACT_INTERVAL_MS
        + 1
    )
    return adapted.drop(columns="closeTime").set_index("openTime")


def _trials_on_panel(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: R.RiskControlledReversalConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    tuple[R.RiskControlledReversalSpec, ...],
]:
    close, residual = _trial_inputs(panel, registration)
    _, close_details, specs = R.run_trial_matrix(close, residual, settlements, config)
    details = {
        name: _adapt_close_time_detail(detail)
        for name, detail in close_details.items()
    }
    if set(details) != {spec.trial_id for spec in specs} or len(details) != 6:
        raise ValueError("core did not return the complete six-trial ledger")
    index = next(iter(details.values())).index
    if any(not detail.index.equals(index) for detail in details.values()):
        raise ValueError("primary trial return grids differ")
    matrix = pd.DataFrame(
        {name: detail["net"].to_numpy(dtype=float) for name, detail in details.items()},
        index=index,
    )
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("primary six-trial matrix contains non-finite returns")
    return matrix, details, specs


def _eligible_names(
    specs: Sequence[R.RiskControlledReversalSpec],
) -> list[str]:
    names = [spec.trial_id for spec in specs if spec.champion_eligible]
    if len(names) != 3 or any(
        spec.exit_rank != 3 for spec in specs if spec.champion_eligible
    ):
        raise ValueError("champion-eligible treatment family changed")
    return names


def _return_sharpe(values: pd.Series | np.ndarray) -> float:
    net = np.asarray(values, dtype=float)
    if net.ndim != 1 or len(net) < 2 or not np.isfinite(net).all():
        return float("-inf")
    standard_deviation = float(np.std(net, ddof=1))
    return (
        float(np.mean(net) / standard_deviation)
        if standard_deviation > 1e-15
        else float("-inf")
    )


def _nested_selector(
    matrix: pd.DataFrame,
    eligible_names: Sequence[str],
    registration: Mapping[str, object],
) -> W.NestedRollingResult:
    validation = _require_mapping(registration, "validation")
    frame = matrix.reset_index()
    candidates = {name: name for name in eligible_names}

    def evaluate(name: str, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"net": rows[name].to_numpy(dtype=float)})

    nested = W.nested_rolling_origin(
        frame,
        candidates,
        fit_candidate=lambda candidate, _train: candidate,
        evaluate_candidate=evaluate,
        score_candidate=lambda evaluated: _return_sharpe(evaluated["net"]),
        initial_train_size=int(validation["outerInitialTrain"]),
        outer_test_size=int(validation["outerTestSize"]),
        inner_initial_train_size=int(validation["innerInitialTrain"]),
        inner_test_size=int(validation["innerTestSize"]),
        label_horizon=int(validation["labelHorizonBars"]),
    )
    if (
        len(nested.outer_folds) != int(validation["outerFoldCount"])
        or len(nested.oos) != int(validation["nestedOuterOosRows"])
        or any(
            str(name) not in eligible_names
            for name in nested.outer_folds["selected_candidate"]
        )
    ):
        raise ValueError("nested rolling selection changed from registration")
    fold_sizes = (
        nested.outer_folds["test_stop"] - nested.outer_folds["test_start"]
    )
    if not np.all(fold_sizes == int(validation["outerTestSize"])):
        raise ValueError("nested selector produced a partial outer fold")
    return nested


def _final_champion(
    matrix: pd.DataFrame,
    eligible_names: Sequence[str],
    registration: Mapping[str, object],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    validation = _require_mapping(registration, "validation")
    splits = W.rolling_origin_splits(
        len(matrix),
        int(validation["innerInitialTrain"]),
        int(validation["innerTestSize"]),
        int(validation["labelHorizonBars"]),
    )
    if not splits or splits[-1].test_stop != len(matrix):
        raise ValueError("final rolling selection lost its registered partial tail")
    rows = []
    for name in eligible_names:
        validation_values = np.concatenate(
            [
                matrix[name].iloc[split.test_slice].to_numpy(dtype=float)
                for split in splits
            ]
        )
        rows.append(
            {
                "candidate": name,
                "score": _return_sharpe(validation_values),
                "folds": len(splits),
                "validationRows": len(validation_values),
            }
        )
    scores = pd.DataFrame(rows)
    finite = scores[np.isfinite(scores["score"])]
    if finite.empty:
        raise ValueError("all final treatment scores are non-finite")
    champion = str(finite.loc[finite["score"].idxmax(), "candidate"])
    folds = pd.DataFrame(
        [
            {
                "fold": split.fold,
                "trainStart": split.train_start,
                "trainStop": split.train_stop,
                "embargoStart": split.embargo_start,
                "embargoStop": split.embargo_stop,
                "testStart": split.test_start,
                "testStop": split.test_stop,
            }
            for split in splits
        ]
    )
    return champion, scores, folds


def _selected_path(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: R.RiskControlledReversalConfig,
    specs: Sequence[R.RiskControlledReversalSpec],
    matrix_index: pd.Index,
    selections: pd.DataFrame,
    *,
    path_id: str,
) -> pd.DataFrame:
    """Run one continuous cash-start path across frozen candidate choices."""
    specs_by_name = {spec.trial_id: spec for spec in specs}
    selected_by_position: dict[int, tuple[int, str]] = {}
    for fold in selections.to_dict("records"):
        fold_number = int(fold["outer_fold"])
        selected = str(fold["selected_candidate"])
        spec = specs_by_name.get(selected)
        if spec is None or not spec.champion_eligible:
            raise ValueError("selected path contains an ineligible reversal trial")
        for position in range(int(fold["test_start"]), int(fold["test_stop"])):
            if position in selected_by_position:
                raise ValueError("selected path folds overlap")
            selected_by_position[position] = (fold_number, selected)
    positions = np.asarray(sorted(selected_by_position), dtype=int)
    if len(positions) == 0 or (
        len(positions) > 1 and not np.all(np.diff(positions) == 1)
    ):
        raise ValueError("selected path folds do not form one contiguous path")
    if positions[0] < 0 or positions[-1] >= len(matrix_index):
        raise ValueError("selected path exceeds the registered matrix")
    evaluation_index = pd.Index(matrix_index[positions], name="openTime")

    close, residual = _trial_inputs(panel, registration)
    close_open_times = pd.Index(
        close.index.to_numpy(dtype=np.int64) - config.interval_ms + 1,
        name="openTime",
    )
    try:
        first_close_position = int(close_open_times.get_loc(evaluation_index[0]))
        last_close_position = int(close_open_times.get_loc(evaluation_index[-1]))
    except KeyError as error:
        raise ValueError(
            "selected evaluation time is absent from close grid"
        ) from error
    if last_close_position - first_close_position + 1 != len(evaluation_index):
        raise ValueError("selected evaluation times are not contiguous")
    close_slice = close.iloc[first_close_position : last_close_position + 2]
    if len(close_slice) != len(evaluation_index) + 1:
        raise ValueError("selected path lacks its terminal outcome close")

    selected_names = [selected_by_position[int(position)][1] for position in positions]
    selected_specs = {
        name: specs_by_name[name] for name in dict.fromkeys(selected_names)
    }
    targets = {
        name: R.weights_for_trial(
            residual[spec.horizon_hours], spec, config
        )
        for name, spec in selected_specs.items()
    }
    composite_targets = pd.DataFrame(
        0.0, index=close_slice.index.copy(), columns=close.columns.copy()
    )
    for local_row, (full_row, name) in enumerate(
        zip(range(first_close_position, last_close_position + 1), selected_names)
    ):
        composite_targets.iloc[local_row] = targets[name].iloc[full_row]
    composite_targets.iloc[-1] = composite_targets.iloc[-2]
    activations = np.zeros(len(close_slice), dtype=bool)
    activations[:-1] = True
    evaluated = _adapt_close_time_detail(
        R.evaluate_drifted_intervals(
            close_slice,
            composite_targets,
            activations,
            settlements,
            config,
            trial_id=path_id,
        )
    ).reset_index()
    if not np.array_equal(
        evaluated["openTime"].to_numpy(dtype=np.int64),
        evaluation_index.to_numpy(dtype=np.int64),
    ):
        raise ValueError("selected path return grid changed")
    evaluated.insert(0, "selected_candidate", selected_names)
    evaluated.insert(
        0,
        "outer_fold",
        [selected_by_position[int(position)][0] for position in positions],
    )
    evaluated.insert(0, "row_position", positions)
    return evaluated


def _fixed_candidate_path(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: R.RiskControlledReversalConfig,
    specs: Sequence[R.RiskControlledReversalSpec],
    matrix_index: pd.Index,
    candidate: str,
    start: int,
    stop: int,
    *,
    path_id: str,
) -> pd.DataFrame:
    selections = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "selected_candidate": candidate,
                "test_start": start,
                "test_stop": stop,
            }
        ]
    )
    return _selected_path(
        panel,
        settlements,
        registration,
        config,
        specs,
        matrix_index,
        selections,
        path_id=path_id,
    )


def _risk_failure(error: R.RiskConstraintBreach, path: str) -> dict[str, object]:
    evidence = getattr(error, "evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {"value": evidence}
    return {
        "reason": str(error.reason),
        "trialId": error.trial_id,
        "closeTime": int(error.interval_left_close_time),
        "closeTimeSemantics": "interval_left_close",
        "outcomeCloseTime": int(error.outcome_close_time),
        "path": path,
        "immediateCloseLiquidationEvidence": dict(evidence),
    }


def _bootstrap_conjunction(
    values: pd.Series | np.ndarray,
    periods_per_year: float,
    registration: Mapping[str, object],
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    validation = _require_mapping(registration, "validation")
    intervals: dict[str, list[float | None]] = {}
    passed = True
    for block in validation["bootstrapBlockBars"]:
        try:
            interval = R.circular_block_bootstrap_sharpe_ci(
                values,
                periods_per_year,
                block=int(block),
                n_boot=int(validation["bootstrapReplications"]),
                seed=int(validation["bootstrapSeed"]),
                alpha=alpha,
            )
        except (TypeError, ValueError):
            interval = (float("nan"), float("nan"))
        intervals[str(block)] = C._ci_json(interval)
        passed = passed and math.isfinite(interval[0]) and interval[0] > 0.0
    return {
        "method": "circular_moving_block",
        "alpha": alpha,
        "replications": int(validation["bootstrapReplications"]),
        "seed": int(validation["bootstrapSeed"]),
        "intervalsByBlockBars": intervals,
        "allLowerBoundsAboveZero": passed,
    }


def _matched_control(
    treatment: str, specs: Sequence[R.RiskControlledReversalSpec]
) -> R.RiskControlledReversalSpec:
    by_name = {spec.trial_id: spec for spec in specs}
    selected = by_name.get(treatment)
    if selected is None or not selected.champion_eligible or selected.exit_rank != 3:
        raise ValueError("selected champion is not a registered treatment")
    controls = [
        spec
        for spec in specs
        if spec.horizon_hours == selected.horizon_hours
        and not spec.champion_eligible
        and spec.exit_rank == 1
    ]
    if len(controls) != 1:
        raise ValueError("selected treatment lacks one matched rank-1 control")
    return controls[0]


def _paired_hysteresis_comparison(
    matrix: pd.DataFrame,
    champion: str,
    specs: Sequence[R.RiskControlledReversalSpec],
    registration: Mapping[str, object],
    periods_per_year: float,
) -> dict[str, object]:
    validation = _require_mapping(registration, "validation")
    hypotheses = int(validation["pairedComparisonHypotheses"])
    family_alpha = float(validation["pairedComparisonFamilyWiseAlpha"])
    comparison_alpha = family_alpha / hypotheses
    horizons = sorted({spec.horizon_hours for spec in specs})
    if hypotheses != len(horizons):
        raise ValueError("paired comparison family size changed")
    selected = next(spec for spec in specs if spec.trial_id == champion)
    rows = []
    for horizon in horizons:
        control = next(
            spec
            for spec in specs
            if spec.horizon_hours == horizon and spec.exit_rank == 1
        )
        treatment = next(
            spec
            for spec in specs
            if spec.horizon_hours == horizon and spec.exit_rank == 3
        )
        spread = matrix[treatment.trial_id] - matrix[control.trial_id]
        confidence = _bootstrap_conjunction(
            spread,
            periods_per_year,
            registration,
            alpha=comparison_alpha,
        )
        rows.append(
            {
                "horizonHours": horizon,
                "controlTrial": control.trial_id,
                "treatmentTrial": treatment.trial_id,
                "treatmentMinusControlMetrics": C._metrics(
                    spread, periods_per_year
                ),
                "simultaneousSharpeConfidence": confidence,
                "selectedChampionPair": horizon == selected.horizon_hours,
                "passed": bool(confidence["allLowerBoundsAboveZero"]),
            }
        )
    champion_row = next(row for row in rows if row["selectedChampionPair"])
    return {
        "method": "paired_exit3_minus_exit1_bonferroni_circular_bootstrap",
        "familyWiseAlpha": family_alpha,
        "hypotheses": hypotheses,
        "comparisonAlpha": comparison_alpha,
        "champion": champion,
        "championPassed": bool(champion_row["passed"]),
        "horizons": rows,
    }


def _turnover_comparison(
    details: Mapping[str, pd.DataFrame],
    evaluation_index: pd.Index,
    champion: str,
    specs: Sequence[R.RiskControlledReversalSpec],
    outer_folds: pd.DataFrame,
) -> dict[str, object]:
    control = _matched_control(champion, specs)
    treatment_values = details[champion].reindex(evaluation_index)[
        "turnover"
    ].to_numpy(dtype=float)
    control_values = details[control.trial_id].reindex(evaluation_index)[
        "turnover"
    ].to_numpy(dtype=float)

    def ratio(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.mean(right))
        return (
            float(np.mean(left)) / denominator
            if math.isfinite(denominator) and denominator > 0.0
            else float("inf")
        )

    mean_ratio = ratio(treatment_values, control_values)
    fold_rows = []
    for fold in outer_folds.to_dict("records"):
        start = int(fold["test_start"])
        stop = int(fold["test_stop"])
        value = ratio(treatment_values[start:stop], control_values[start:stop])
        fold_rows.append(
            {
                "outerFold": int(fold["outer_fold"]),
                "ratio": C._finite_number(value),
                "denominatorValid": math.isfinite(value),
            }
        )
    return {
        "champion": champion,
        "matchedControl": control.trial_id,
        "observations": len(evaluation_index),
        "championMeanTurnover": float(np.mean(treatment_values)),
        "controlMeanTurnover": float(np.mean(control_values)),
        "meanRatio": C._finite_number(mean_ratio),
        "meanDenominatorValid": math.isfinite(mean_ratio),
        "outerFolds": fold_rows,
    }


def _lifetime_multiple_testing(
    diagnostic_matrix: pd.DataFrame,
    champion: str,
    registration: Mapping[str, object],
) -> dict[str, object]:
    validation = _require_mapping(registration, "validation")
    trials = int(validation["lifetimeTrialCount"])
    try:
        result = diagnostics.deflated_sharpe_ratio(
            diagnostic_matrix,
            selected_trial=champion,
            periods_per_year=365.0,
            independent_trials=1,
        ).to_dict()
        single = float(result["probability"])
        adjusted = max(0.0, 1.0 - min(1.0, trials * (1.0 - single)))
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": trials,
            "singleTrialProbability": single,
            "adjustedProbability": adjusted,
            "underlyingProbabilisticSharpe": result,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": trials,
            "adjustedProbability": 0.0,
            "error": str(error),
        }


def _risk_ledger(paths: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    columns = {
        "minimumEquity": ("equity", min),
        "maximumDrawdown": ("drawdown", max),
        "maximumEndpointGrossLeverage": ("endpointGrossLeverage", max),
        "maximumAbsoluteSymbolWeight": ("maximumAbsoluteSymbolWeight", max),
        "maximumActivationTurnover": ("activationTurnover", max),
        "maximumTerminalTurnover": ("terminalTurnover", max),
        "minimumShockEquityFraction": ("shockEquityFraction", min),
        "minimumShockMaintenanceCoverage": ("shockMaintenanceCoverage", min),
    }
    ledger: dict[str, object] = {}
    for name, frame in paths.items():
        summary: dict[str, object] = {
            "status": "complete",
            "rows": len(frame),
        }
        for output, (column, reducer) in columns.items():
            if column not in frame:
                continue
            values = frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary[output] = float(reducer(finite)) if len(finite) else None
        ledger[name] = summary
    return ledger


def _stress_paths(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    specs: Sequence[R.RiskControlledReversalSpec],
    matrix_index: pd.Index,
    outer_folds: pd.DataFrame,
    champion: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    first = int(outer_folds["test_start"].min())
    stop = int(outer_folds["test_stop"].max())
    result = {}
    for label, config in (
        ("cost2x", _strategy_config(registration, cost_multiplier=2.0)),
        (
            "additionalDelay1bar",
            _strategy_config(registration, additional_delay=1),
        ),
    ):
        result[label] = {
            "nestedOuterOos": _selected_path(
                panel,
                settlements,
                registration,
                config,
                specs,
                matrix_index,
                outer_folds,
                path_id=f"{label}_nested_outer_oos",
            ),
            "finalChampion": _fixed_candidate_path(
                panel,
                settlements,
                registration,
                config,
                specs,
                matrix_index,
                champion,
                first,
                stop,
                path_id=f"{label}_{champion}",
            ),
        }
    return result


def _existing_terminal_result(output_dir: Path) -> dict[str, object] | None:
    index_path = output_dir / "evidence-index.json"
    if not index_path.exists():
        return None
    index = _read_json_object(index_path)
    if index.get("campaign") != CAMPAIGN_ID:
        raise ValueError("terminal evidence index campaign changed")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping) or "summary" not in artifacts:
        raise ValueError("terminal evidence index is incomplete")
    indexed_names = {".campaign.lock", "evidence-index.json"}
    for name, record in artifacts.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("terminal evidence index has an invalid artifact")
        path = Path(str(record.get("path", "")))
        if path.parent.resolve() != output_dir.resolve() or path.name in indexed_names:
            raise ValueError(
                "terminal evidence index path escaped its output directory"
            )
        if not path.is_file() or C._file_digest(path) != record.get("sha256"):
            raise ValueError("terminal evidence artifact changed")
        indexed_names.add(path.name)
    observed_names = {path.name for path in output_dir.iterdir()}
    if observed_names != indexed_names:
        raise ValueError("terminal evidence artifact set changed")
    manifest_record = artifacts.get("campaignManifest")
    if not isinstance(manifest_record, Mapping) or index.get(
        "campaignManifestSha256"
    ) != manifest_record.get("sha256"):
        raise ValueError("terminal evidence manifest cross-link changed")
    summary_record = artifacts["summary"]
    summary = _read_json_object(Path(str(summary_record["path"])))
    if (
        index.get("status") != summary.get("status")
        or summary.get("campaign") != CAMPAIGN_ID
        or summary.get("registrationSha256") != index.get("registrationSha256")
        or summary.get("campaignManifestSha256")
        != index.get("campaignManifestSha256")
    ):
        raise ValueError("terminal evidence summary cross-links changed")
    expected = _expected_terminal_artifact_names(summary)
    indexed_artifacts = {
        Path(str(record["path"])).name for record in artifacts.values()
    }
    if indexed_artifacts != expected:
        raise ValueError("terminal evidence contains an unexpected artifact set")
    return summary


def _expected_terminal_artifact_names(
    summary: Mapping[str, object],
) -> set[str]:
    status = summary.get("status")
    base = {"campaign-manifest.json", "summary.json", "risk-ledger.json"}
    if status == "risk_invalid":
        return base | {"risk-failure.json"}
    if status == "insufficient_evidence" and "developmentExecutionFailure" in summary:
        return base | {"development-execution-failure.json"}
    analysis = base | {
        "nested-outer-oos.csv",
        "nested-inner-scores.csv",
        "nested-outer-folds.csv",
        "final-selection.csv",
        "final-selection-folds.csv",
        "nested-outer-oos-regimes.csv",
        "diagnostic-dsr-matrix.csv",
        "diagnostic-pbo-matrix.csv",
        "primary-trial-returns.csv",
        "primary-trial-paths.csv",
        "final-champion-development.csv",
        "stress-cost2x-nested-outer-oos.csv",
        "stress-cost2x-final-champion.csv",
        "stress-additional-delay1bar-nested-outer-oos.csv",
        "stress-additional-delay1bar-final-champion.csv",
    }
    if status == "insufficient_evidence":
        return analysis
    if status in {"final_holdout_passed", "final_holdout_failed"}:
        holdout = summary.get("finalHoldout")
        if not isinstance(holdout, Mapping):
            raise ValueError("terminal holdout summary is invalid")
        result = analysis | {
            "final-holdout-opened.json",
            "final-holdout-result.json",
        }
        evidence = holdout.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("returnsSha256"):
            result.add("final-holdout-returns.csv")
        return result
    raise ValueError("evidence index cannot terminate a non-terminal campaign status")


def _validate_completed_holdout_registry(
    output_dir: Path, summary: Mapping[str, object]
) -> None:
    if summary.get("status") not in {
        "final_holdout_passed",
        "final_holdout_failed",
    }:
        return
    local_path = output_dir / "final-holdout-opened.json"
    local = _read_json_object(local_path)
    identity = local.get("holdoutIdentitySha256")
    if not isinstance(identity, str):
        raise ValueError("completed local holdout identity is invalid")
    marker = HOLDOUT_REGISTRY_DIR / f"{identity}.json"
    with C._holdout_registry_lock(HOLDOUT_REGISTRY_DIR):
        if not marker.is_file():
            raise ValueError("completed shared holdout marker is missing")
        shared = _read_json_object(marker)
        C._registry_window(marker, shared, strict_identity=True)
        if shared != local or shared.get("status") != "completed":
            raise ValueError("completed local and shared holdout records differ")
    final_holdout = summary.get("finalHoldout")
    if (
        not isinstance(final_holdout, Mapping)
        or final_holdout.get("identitySha256") != identity
    ):
        raise ValueError("completed holdout summary identity changed")


def _finalize_terminal(
    output_dir: Path,
    summary: Mapping[str, object],
    registration_sha: str,
    campaign_manifest_sha: str,
) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    C._write_json(summary_path, dict(summary))
    expected_names = _expected_terminal_artifact_names(summary)
    observed_names = {
        path.name
        for path in output_dir.iterdir()
        if path.name not in {".campaign.lock", "evidence-index.json"}
    }
    if observed_names != expected_names:
        raise ValueError("terminal output contains an unexpected artifact set")
    artifacts = {
        path.stem if path.name != "campaign-manifest.json" else "campaignManifest": path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name not in {".campaign.lock", "evidence-index.json"}
    }
    artifacts["summary"] = summary_path
    _terminal_evidence_index(
        output_dir,
        str(summary["status"]),
        artifacts,
        registration_sha,
        campaign_manifest_sha,
    )
    return dict(summary)


def _holdout_descriptor(
    args: argparse.Namespace,
    registration: Mapping[str, object],
    *,
    blocked_by: Sequence[str] = (),
) -> dict[str, object]:
    data = _require_mapping(registration, "registeredData")
    policy = _require_mapping(registration, "holdoutPolicy")
    _validate_registry_policy(policy)
    return _holdout_descriptor_value(
        args, registration, data, policy, blocked_by
    )


def _validate_registry_policy(policy: Mapping[str, object]) -> None:
    registered_registry = C.REPOSITORY_ROOT / str(policy["registryDirectory"])
    if (
        HOLDOUT_REGISTRY_DIR.resolve() != registered_registry.resolve()
        and not TEST_ONLY_ALLOW_REGISTRY_OVERRIDE
    ):
        raise ValueError(
            "official holdout evaluation must use the registered shared registry"
        )
    if int(policy["registryVersion"]) != C.HOLDOUT_REGISTRY_VERSION:
        raise ValueError("registered holdout registry version changed")


def _holdout_descriptor_value(
    args: argparse.Namespace,
    registration: Mapping[str, object],
    data: Mapping[str, object],
    policy: Mapping[str, object],
    blocked_by: Sequence[str],
) -> dict[str, object]:
    window = C._holdout_window(
        [str(symbol) for symbol in registration["universe"]["symbols"]],
        feed.CONTRACT_INTERVAL,
        int(data["holdoutStartOpenTime"]),
        int(data["endOpenTime"]),
    )
    identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": data["fullPanelDigestSha256"],
            "window": window,
        }
    )
    value: dict[str, object] = {
        "status": "reserved",
        "identitySha256": identity,
        "openRequested": bool(args.open_final_holdout),
        "executionStateAtStart": policy["executionStateAtStart"],
        "featureHistory": policy["featureHistory"],
        "chargeInitialCashToFrozenTargetTurnover": policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ],
        "chargeTerminalLiquidation": policy["chargeTerminalLiquidation"],
        "startOpenTime": int(data["holdoutStartOpenTime"]),
        "endOpenTime": int(data["endOpenTime"]),
        "outcomeEndTimeExclusive": int(data["outcomeEndTimeExclusive"]),
        "rows": int(data["holdoutReturnRows"]),
    }
    if blocked_by:
        value["openBlockedBy"] = list(blocked_by)
    return value


def _base_summary(
    registration: Mapping[str, object],
    registration_sha: str,
    campaign_manifest_sha: str,
    source_evidence: Mapping[str, object],
    predecessor_evidence: Mapping[str, object],
    settlement_coverage: Mapping[str, object],
) -> dict[str, object]:
    universe = _require_mapping(registration, "universe")
    data = _require_mapping(registration, "registeredData")
    return {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "symbols": list(universe["symbols"]),
        "interval": universe["interval"],
        "data": {
            "registeredRows": int(data["rows"]),
            "developmentRows": int(data["developmentRows"]),
            "sourceArtifacts": dict(source_evidence),
            "predecessorEvidence": dict(predecessor_evidence),
            "fullPanelDigestSha256": data["fullPanelDigestSha256"],
            "fullSettlementsDigestSha256": data["fullSettlementsDigestSha256"],
            "settlements": dict(settlement_coverage),
            "survivorshipLimitation": universe["survivorshipLimitation"],
        },
        "configuration": {
            "strategy": dict(registration["strategy"]),
            "riskPolicy": dict(registration["riskPolicy"]),
            "validation": dict(registration["validation"]),
        },
        "trials": list(registration["trials"]),
    }


def _primary_risk_invalid_result(
    args: argparse.Namespace,
    output_dir: Path,
    base: Mapping[str, object],
    error: R.RiskConstraintBreach,
    registration: Mapping[str, object],
    registration_sha: str,
    campaign_manifest_sha: str,
) -> dict[str, object]:
    failure = _risk_failure(error, "primary_development")
    failure_record = {
        "campaign": CAMPAIGN_ID,
        "status": "risk_invalid",
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "riskFailure": failure,
        "finalHoldout": _holdout_descriptor(
            args,
            registration,
            blocked_by=["everyPrimaryPathRiskSafeAndComplete"],
        ),
    }
    failure_path = output_dir / "risk-failure.json"
    C._write_json(failure_path, failure_record)
    ledger_path = output_dir / "risk-ledger.json"
    C._write_json(
        ledger_path,
        {"status": "risk_invalid", "primaryFailure": failure},
    )
    summary = {
        **dict(base),
        **failure_record,
        "promotionGates": {"everyPrimaryPathRiskSafeAndComplete": False},
        "evidence": {
            "riskFailure": str(failure_path.resolve()),
            "riskFailureSha256": C._file_digest(failure_path),
            "riskLedger": str(ledger_path.resolve()),
            "riskLedgerSha256": C._file_digest(ledger_path),
        },
    }
    return _finalize_terminal(
        output_dir, summary, registration_sha, campaign_manifest_sha
    )


def _derived_failure_result(
    args: argparse.Namespace,
    output_dir: Path,
    base: Mapping[str, object],
    error: R.RiskConstraintBreach,
    stage: str,
    registration: Mapping[str, object],
    registration_sha: str,
    campaign_manifest_sha: str,
) -> dict[str, object]:
    failure = _risk_failure(error, stage)
    path = output_dir / "development-execution-failure.json"
    C._write_json(
        path,
        {
            "campaign": CAMPAIGN_ID,
            "status": "insufficient_evidence",
            "registrationSha256": registration_sha,
            "campaignManifestSha256": campaign_manifest_sha,
            "developmentExecutionFailure": failure,
        },
    )
    ledger_path = output_dir / "risk-ledger.json"
    C._write_json(
        ledger_path,
        {"status": "derived_path_risk_breach", "failure": failure},
    )
    summary = {
        **dict(base),
        "status": "insufficient_evidence",
        "primaryPathsRiskSafeAndComplete": True,
        "derivedPathsRiskSafeAndComplete": False,
        "developmentExecutionFailure": failure,
        "promotionGates": {
            "everyPrimaryPathRiskSafeAndComplete": True,
            "allRegisteredStressPathsEvaluable": False,
        },
        "finalHoldout": _holdout_descriptor(
            args, registration, blocked_by=["allRegisteredStressPathsEvaluable"]
        ),
        "evidence": {
            "developmentExecutionFailure": str(path.resolve()),
            "developmentExecutionFailureSha256": C._file_digest(path),
            "riskLedger": str(ledger_path.resolve()),
            "riskLedgerSha256": C._file_digest(ledger_path),
        },
    }
    return _finalize_terminal(
        output_dir, summary, registration_sha, campaign_manifest_sha
    )


def _open_final_holdout(
    args: argparse.Namespace,
    output_dir: Path,
    predecessor_dir: Path,
    source_dir: Path,
    snapshot_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
    campaign_manifest_sha: str,
    champion: str,
    specs: Sequence[R.RiskControlledReversalSpec],
    periods_per_year: float,
) -> dict[str, object]:
    data = _require_mapping(registration, "registeredData")
    policy = _require_mapping(registration, "holdoutPolicy")
    _validate_registry_policy(policy)
    symbols = [str(symbol) for symbol in registration["universe"]["symbols"]]
    window = C._holdout_window(
        symbols,
        feed.CONTRACT_INTERVAL,
        int(data["holdoutStartOpenTime"]),
        int(data["endOpenTime"]),
    )
    identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": data["fullPanelDigestSha256"],
            "window": window,
        }
    )
    marker = HOLDOUT_REGISTRY_DIR / f"{identity}.json"
    output_record = output_dir / "final-holdout-opened.json"
    returns_path = output_dir / "final-holdout-returns.csv"
    result_path = output_dir / "final-holdout-result.json"
    stale = [
        path.name
        for path in (output_record, returns_path, result_path)
        if path.exists()
    ]
    if stale:
        raise ValueError(
            "final holdout output already contains immutable artifacts: "
            + ", ".join(stale)
        )
    opening_record = {
        "registryVersion": C.HOLDOUT_REGISTRY_VERSION,
        "status": "opening",
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "holdoutIdentitySha256": identity,
        "panelSha256": data["fullPanelDigestSha256"],
        "outputBindingSha256": C._json_digest(
            {
                "holdoutIdentitySha256": identity,
                "outputDirectory": str(output_dir.resolve()),
            }
        ),
        "candidate": champion,
        "window": window,
        "executionStateAtStart": policy["executionStateAtStart"],
        "featureHistory": policy["featureHistory"],
        "chargeInitialCashToFrozenTargetTurnover": policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ],
        "chargeTerminalLiquidation": policy["chargeTerminalLiquidation"],
        "artifacts": {
            "outputDirectory": str(output_dir.resolve()),
            "returns": str(returns_path.resolve()),
            "result": str(result_path.resolve()),
        },
    }
    _assert_inputs_unchanged(
        predecessor_dir,
        source_dir,
        registration,
        registration_sha,
        implementation_sha,
    )
    # Reservation is irreversible and intentionally precedes every snapshot read.
    C._reserve_holdout(
        HOLDOUT_REGISTRY_DIR,
        marker,
        window,
        output_record,
        opening_record,
        strict_identity=True,
    )

    artifacts = dict(opening_record["artifacts"])
    completed_returns: pd.DataFrame | None = None
    try:
        full_panel, full_settlements, _ = V1._load_full_registered_inputs(
            snapshot_dir, registration
        )
        full_index = pd.Index(C._common_times(full_panel)[:-1], name="openTime")
        holdout_index = full_index[
            (full_index >= int(data["holdoutStartOpenTime"]))
            & (full_index <= int(data["endOpenTime"]))
        ]
        if len(holdout_index) != int(data["holdoutReturnRows"]):
            raise ValueError("final holdout return rows changed from registration")
        start = int(full_index.get_loc(holdout_index[0]))
        stop = int(full_index.get_loc(holdout_index[-1])) + 1
        evaluated = _fixed_candidate_path(
            full_panel,
            full_settlements,
            registration,
            _strategy_config(registration),
            specs,
            full_index,
            champion,
            start,
            stop,
            path_id=champion,
        ).drop(columns=["row_position", "outer_fold", "selected_candidate"])
        metrics = C._metrics(
            evaluated["net"], periods_per_year, evaluated["active"]
        )
        confidence = _bootstrap_conjunction(
            evaluated["net"], periods_per_year, registration
        )
        passed = (
            float(metrics["totalReturn"]) > 0.0
            and float(metrics["maxDrawdown"])
            <= float(registration["promotion"]["maximumNestedOuterOosDrawdown"])
            and bool(confidence["allLowerBoundsAboveZero"])
        )
        final_holdout: dict[str, object] = {
            "status": "pass" if passed else "fail",
            "openRequested": True,
            "identitySha256": identity,
            "evaluationStatus": "completed_without_risk_breach",
            "candidate": champion,
            "executionStateAtStart": policy["executionStateAtStart"],
            "featureHistory": policy["featureHistory"],
            "chargeInitialCashToFrozenTargetTurnover": policy[
                "chargeInitialCashToFrozenTargetTurnover"
            ],
            "chargeTerminalLiquidation": policy["chargeTerminalLiquidation"],
            "startOpenTime": int(data["holdoutStartOpenTime"]),
            "endOpenTime": int(data["endOpenTime"]),
            "outcomeEndTimeExclusive": int(data["outcomeEndTimeExclusive"]),
            "rows": len(evaluated),
            "metrics": metrics,
            "sharpeConfidence": confidence,
        }
        completed_returns = evaluated
    except R.RiskConstraintBreach as error:
        final_holdout = {
            **_holdout_descriptor(args, registration),
            "status": "fail",
            "openRequested": True,
            "identitySha256": identity,
            "candidate": champion,
            "evaluationStatus": "risk_breach",
            "successRuleEvaluated": False,
            "failure": _risk_failure(error, "final_holdout"),
        }
        artifacts["returnsWritten"] = False
    except Exception as error:
        final_holdout = {
            **_holdout_descriptor(args, registration),
            "status": "fail",
            "openRequested": True,
            "identitySha256": identity,
            "candidate": champion,
            "evaluationStatus": "execution_error",
            "successRuleEvaluated": False,
            "failure": {
                "reason": "holdout_execution_error",
                "errorType": type(error).__name__,
                "message": str(error),
            },
        }
        artifacts["returnsWritten"] = False

    try:
        _assert_inputs_unchanged(
            predecessor_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
        )
    except Exception as error:
        completed_returns = None
        final_holdout = {
            **_holdout_descriptor(args, registration),
            "status": "fail",
            "openRequested": True,
            "identitySha256": identity,
            "candidate": champion,
            "evaluationStatus": "execution_error",
            "successRuleEvaluated": False,
            "failure": {
                "reason": "holdout_input_integrity_changed",
                "errorType": type(error).__name__,
                "message": str(error),
            },
        }
        artifacts["returnsWritten"] = False

    if completed_returns is not None:
        C._write_csv_atomic(completed_returns, returns_path, index=False)
        returns_sha = C._file_digest(returns_path)
        final_holdout["evidence"] = {
            "returns": str(returns_path.resolve()),
            "returnsSha256": returns_sha,
        }
        artifacts.update({"returnsSha256": returns_sha, "returnsWritten": True})

    result_record = {
        **opening_record,
        "status": "evaluated",
        "result": final_holdout,
        "artifacts": artifacts,
    }
    C._write_json(result_path, result_record)
    completion = {
        **opening_record,
        "status": "completed",
        "result": final_holdout,
        "artifacts": {
            **artifacts,
            "resultSha256": C._file_digest(result_path),
        },
    }
    V1._complete_holdout_records(marker, output_record, opening_record, completion)
    return final_holdout


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    with C._campaign_output_lock(output_dir):
        interrupted = [
            name
            for name in (
                "final-holdout-opened.json",
                "final-holdout-result.json",
                "final-holdout-returns.csv",
            )
            if (output_dir / name).exists()
        ]
        if interrupted and not (output_dir / "evidence-index.json").exists():
            raise ValueError(
                "final holdout was already consumed or interrupted; use its "
                "existing evidence: "
                + ", ".join(interrupted)
            )
        existing = _existing_terminal_result(output_dir)
        if existing is not None:
            registration, registration_sha = _registration_and_sha()
            _validate_registry_policy(
                _require_mapping(registration, "holdoutPolicy")
            )
            manifest = _read_json_object(output_dir / "campaign-manifest.json")
            if (
                manifest.get("registrationSha256") != registration_sha
                or manifest.get("implementationArtifacts")
                != _implementation_artifacts()
                or manifest.get("implementationSha256") != _implementation_sha()
            ):
                raise ValueError(
                    "terminal evidence does not match the current implementation"
                )
            _assert_inputs_unchanged(
                Path(args.predecessor_campaign_dir),
                Path(args.source_campaign_dir),
                registration,
                registration_sha,
                str(manifest["implementationSha256"]),
            )
            _validate_completed_holdout_registry(output_dir, existing)
            return existing
        C._assert_output_holdout_not_consumed(
            HOLDOUT_REGISTRY_DIR,
            output_dir,
            strict_identity=True,
        )
        return _run_locked(args, output_dir)


def _run_locked(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, object]:
    registration, registration_sha = _registration_and_sha()
    _validate_registry_policy(_require_mapping(registration, "holdoutPolicy"))
    implementation_artifacts = _implementation_artifacts()
    implementation_sha = C._json_digest(implementation_artifacts)
    predecessor_dir = Path(args.predecessor_campaign_dir)
    source_dir = Path(args.source_campaign_dir)
    snapshot_dir = Path(args.snapshot_dir)

    predecessor_evidence = _validate_predecessor_evidence(
        predecessor_dir, registration
    )
    source_registration_sha = V1._require_file_digest(
        H.REGISTRATION_PATH,
        registration["registeredData"]["sourceRegistrationSha256"],
    )
    (
        panel,
        settlements,
        _settlement_audit,
        settlement_coverage,
        funding_source_evidence,
    ) = V1._load_development_inputs(source_dir, registration)
    funding_source_evidence = {
        **dict(funding_source_evidence),
        "sourceRegistration": str(H.REGISTRATION_PATH),
        "sourceRegistrationSha256": source_registration_sha,
    }
    source_evidence = {
        "predecessor": dict(predecessor_evidence),
        "registeredDevelopment": dict(funding_source_evidence),
    }
    _, campaign_manifest_sha = _campaign_manifest(
        output_dir,
        registration,
        registration_sha,
        implementation_artifacts,
        source_evidence,
    )
    base = _base_summary(
        registration,
        registration_sha,
        campaign_manifest_sha,
        funding_source_evidence,
        predecessor_evidence,
        settlement_coverage,
    )
    config = _strategy_config(registration)
    try:
        raw_matrix, raw_details, specs = _trials_on_panel(
            panel, settlements, registration, config
        )
    except R.RiskConstraintBreach as error:
        _assert_inputs_unchanged(
            predecessor_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
        )
        return _primary_risk_invalid_result(
            args,
            output_dir,
            base,
            error,
            registration,
            registration_sha,
            campaign_manifest_sha,
        )

    validation = _require_mapping(registration, "validation")
    promotion = _require_mapping(registration, "promotion")
    warmup = int(validation["featureWarmupRows"])
    matrix = raw_matrix.iloc[warmup:].copy()
    details = {name: frame.reindex(matrix.index) for name, frame in raw_details.items()}
    if len(matrix) != int(validation["developmentEvaluationRows"]):
        raise ValueError("development evaluation row count changed")
    if any(frame.isna().any().any() for frame in details.values()):
        raise ValueError("primary trial detail is incomplete after warmup")
    eligible_names = _eligible_names(specs)

    nested = _nested_selector(matrix, eligible_names, registration)
    champion, final_scores, final_folds = _final_champion(
        matrix, eligible_names, registration
    )
    try:
        outer_oos = _selected_path(
            panel,
            settlements,
            registration,
            config,
            specs,
            matrix.index,
            nested.outer_folds,
            path_id="nested_outer_oos",
        )
        first_oos = int(nested.outer_folds["test_start"].min())
        stop_oos = int(nested.outer_folds["test_stop"].max())
        final_champion_path = _fixed_candidate_path(
            panel,
            settlements,
            registration,
            config,
            specs,
            matrix.index,
            champion,
            first_oos,
            stop_oos,
            path_id=f"final_champion_{champion}",
        )
        stresses = _stress_paths(
            panel,
            settlements,
            registration,
            specs,
            matrix.index,
            nested.outer_folds,
            champion,
        )
    except R.RiskConstraintBreach as error:
        _assert_inputs_unchanged(
            predecessor_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
        )
        return _derived_failure_result(
            args,
            output_dir,
            base,
            error,
            "derived_development",
            registration,
            registration_sha,
            campaign_manifest_sha,
        )

    periods_per_year = C._periods_per_year(feed.CONTRACT_INTERVAL_MS)
    outer_confidence = _bootstrap_conjunction(
        outer_oos["net"], periods_per_year, registration
    )
    final_champion_confidence = _bootstrap_conjunction(
        final_champion_path["net"], periods_per_year, registration
    )
    stress_report: dict[str, object] = {}
    stress_paths_for_ledger: dict[str, pd.DataFrame] = {}
    stress_confidence_passed = True
    for label, paths in stresses.items():
        path_report = {}
        for path_name, frame in paths.items():
            confidence = _bootstrap_conjunction(
                frame["net"], periods_per_year, registration
            )
            path_report[path_name] = {
                "metrics": C._metrics(
                    frame["net"], periods_per_year, frame["active"]
                ),
                "sharpeConfidence": confidence,
            }
            stress_confidence_passed = stress_confidence_passed and bool(
                confidence["allLowerBoundsAboveZero"]
            )
            stress_paths_for_ledger[f"{label}.{path_name}"] = frame
        stress_report[label] = path_report

    paired = _paired_hysteresis_comparison(
        matrix, champion, specs, registration, periods_per_year
    )
    turnover = _turnover_comparison(
        details, matrix.index, champion, specs, nested.outer_folds
    )
    diagnostic_report, diagnostic_matrix, pbo_matrix = C._diagnostics(
        matrix,
        champion,
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        int(validation["pboSlices"]),
        independent_trials=float(validation["currentCampaignTrialCount"]),
    )
    lifetime = _lifetime_multiple_testing(
        diagnostic_matrix, champion, registration
    )
    outer_metrics = C._metrics(
        outer_oos["net"], periods_per_year, outer_oos["active"]
    )
    fold_metrics = C._fold_metrics(outer_oos, periods_per_year)
    positive_folds = sum(
        float(metrics["totalReturn"]) > 0.0 for metrics in fold_metrics.values()
    )
    worst_fold = min(float(metrics["totalReturn"]) for metrics in fold_metrics.values())
    regime_labels = C._market_regime_labels(panel, feed.CONTRACT_INTERVAL_MS)
    regime_report, regime_passed, labelled_oos = C._regime_report(
        outer_oos,
        regime_labels,
        periods_per_year,
        int(promotion["minimumRegimeObservations"]),
        float(promotion["maximumRegimeLoss"]),
    )
    active_fraction = float(
        np.count_nonzero(outer_oos["active"].to_numpy(dtype=float) > 0)
        / len(outer_oos)
    )
    deflated = diagnostic_report.get("deflatedSharpe", {})
    pbo = diagnostic_report.get("pbo", {})
    deflated_probability = (
        float(deflated.get("probability", 0.0))
        if isinstance(deflated, Mapping)
        else 0.0
    )
    pbo_probability = (
        float(pbo.get("probability", 1.0)) if isinstance(pbo, Mapping) else 1.0
    )
    fold_ratios = [row["ratio"] for row in turnover["outerFolds"]]
    mean_turnover_ratio = turnover["meanRatio"]
    diagnostics_finite = (
        "errors" not in diagnostic_report
        and math.isfinite(deflated_probability)
        and math.isfinite(pbo_probability)
        and math.isfinite(float(lifetime.get("adjustedProbability", 0.0)))
    )
    stress_cost_pass = all(
        bool(value["sharpeConfidence"]["allLowerBoundsAboveZero"])
        for value in stress_report["cost2x"].values()
    )
    stress_delay_pass = all(
        bool(value["sharpeConfidence"]["allLowerBoundsAboveZero"])
        for value in stress_report["additionalDelay1bar"].values()
    )
    gates = {
        "minimumSymbols": len(registration["universe"]["symbols"])
        >= int(promotion["minimumSymbols"]),
        "fundingCoverage": float(settlement_coverage["resolvedFraction"])
        >= float(promotion["minimumResolvedFundingFraction"]),
        "everyPrimaryPathRiskSafeAndComplete": True,
        "nestedOuterOosObservations": len(outer_oos)
        >= int(promotion["minimumNestedOuterOosObservations"]),
        "minimumActiveFraction": active_fraction
        >= float(promotion["minimumActiveFraction"]),
        "maximumNestedOuterOosDrawdown": float(outer_metrics["maxDrawdown"])
        <= float(promotion["maximumNestedOuterOosDrawdown"]),
        "nestedOuterOosSharpeCiAboveZeroAllBlocks": bool(
            outer_confidence["allLowerBoundsAboveZero"]
        ),
        "cost2xSharpeCiAboveZeroAllBlocks": stress_cost_pass,
        "additionalDelaySharpeCiAboveZeroAllBlocks": stress_delay_pass,
        "matchedHysteresisImprovementSharpeCiAboveZeroAllBlocks": bool(
            paired["championPassed"]
        ),
        "championMeanTurnoverRatio": mean_turnover_ratio is not None
        and float(mean_turnover_ratio)
        <= float(promotion["maximumChampionMeanTurnoverRatio"]),
        "championEveryOuterFoldTurnoverRatio": all(
            value is not None
            and float(value)
            <= float(promotion["maximumChampionOuterFoldTurnoverRatio"])
            for value in fold_ratios
        ),
        "minimumPositiveOuterFolds": positive_folds
        >= int(promotion["minimumPositiveOuterFolds"]),
        "maximumWorstOuterFoldLoss": worst_fold
        >= -float(promotion["maximumWorstOuterFoldLoss"]),
        "regimeRobustness": regime_passed,
        "currentCampaignDeflatedSharpe": deflated_probability
        >= float(promotion["currentCampaignDeflatedSharpeProbabilityMinimum"]),
        "lifetimeBonferroniPsr": float(
            lifetime.get("adjustedProbability", 0.0)
        )
        >= float(promotion["lifetimeBonferroniPsrProbabilityMinimum"]),
        "pbo": pbo_probability <= float(promotion["maximumPbo"]),
        "championExitRank": next(
            spec.exit_rank for spec in specs if spec.trial_id == champion
        )
        == int(promotion["requireChampionExitRank"]),
        "allRegisteredStressPathsEvaluable": True,
        "allRegisteredStressConfidenceGates": stress_confidence_passed,
        "allDiagnosticsFinite": diagnostics_finite,
    }
    ready_for_holdout = all(gates.values())

    paths_for_ledger = {
        **details,
        "nestedOuterOos": outer_oos,
        "finalChampion": final_champion_path,
        **stress_paths_for_ledger,
    }
    risk_ledger = _risk_ledger(paths_for_ledger)
    risk_ledger_path = output_dir / "risk-ledger.json"
    C._write_json(risk_ledger_path, risk_ledger)
    C._write_csv_atomic(outer_oos, output_dir / "nested-outer-oos.csv", index=False)
    C._write_csv_atomic(
        nested.inner_scores, output_dir / "nested-inner-scores.csv", index=False
    )
    C._write_csv_atomic(
        nested.outer_folds, output_dir / "nested-outer-folds.csv", index=False
    )
    C._write_csv_atomic(final_scores, output_dir / "final-selection.csv", index=False)
    C._write_csv_atomic(
        final_folds, output_dir / "final-selection-folds.csv", index=False
    )
    C._write_csv_atomic(
        labelled_oos, output_dir / "nested-outer-oos-regimes.csv", index=False
    )
    C._write_csv_atomic(
        diagnostic_matrix, output_dir / "diagnostic-dsr-matrix.csv", index=True
    )
    C._write_csv_atomic(
        pbo_matrix, output_dir / "diagnostic-pbo-matrix.csv", index=True
    )
    C._write_csv_atomic(
        raw_matrix.reset_index(),
        output_dir / "primary-trial-returns.csv",
        index=False,
    )
    primary_paths = []
    for trial_id, frame in raw_details.items():
        path = frame.reset_index()
        path.insert(0, "trialId", trial_id)
        primary_paths.append(path)
    C._write_csv_atomic(
        pd.concat(primary_paths, ignore_index=True),
        output_dir / "primary-trial-paths.csv",
        index=False,
    )
    C._write_csv_atomic(
        final_champion_path,
        output_dir / "final-champion-development.csv",
        index=False,
    )
    stress_file_labels = {
        "cost2x": "cost2x",
        "additionalDelay1bar": "additional-delay1bar",
    }
    stress_path_labels = {
        "nestedOuterOos": "nested-outer-oos",
        "finalChampion": "final-champion",
    }
    for stress_label, paths in stresses.items():
        for path_label, frame in paths.items():
            filename = (
                f"stress-{stress_file_labels[stress_label]}-"
                f"{stress_path_labels[path_label]}.csv"
            )
            C._write_csv_atomic(frame, output_dir / filename, index=False)

    auditable_paths = [
        "risk-ledger.json",
        "nested-outer-oos.csv",
        "primary-trial-returns.csv",
        "primary-trial-paths.csv",
        "final-champion-development.csv",
        "stress-cost2x-nested-outer-oos.csv",
        "stress-cost2x-final-champion.csv",
        "stress-additional-delay1bar-nested-outer-oos.csv",
        "stress-additional-delay1bar-final-champion.csv",
    ]
    evidence_paths = {
        Path(name).stem: {
            "path": str((output_dir / name).resolve()),
            "sha256": C._file_digest(output_dir / name),
        }
        for name in auditable_paths
    }

    blocked_by = [name for name, passed in gates.items() if not passed]
    final_holdout = _holdout_descriptor(
        args, registration, blocked_by=blocked_by if args.open_final_holdout else ()
    )
    if args.open_final_holdout and ready_for_holdout:
        final_holdout = _open_final_holdout(
            args,
            output_dir,
            predecessor_dir,
            source_dir,
            snapshot_dir,
            registration,
            registration_sha,
            implementation_sha,
            campaign_manifest_sha,
            champion,
            specs,
            periods_per_year,
        )
    else:
        _assert_inputs_unchanged(
            predecessor_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
        )
    summary = {
        **base,
        "status": C._campaign_status(ready_for_holdout, final_holdout),
        "primaryPathsRiskSafeAndComplete": True,
        "derivedPathsRiskSafeAndComplete": True,
        "champion": champion,
        "nestedOuterOos": {
            "metrics": outer_metrics,
            "activeFraction": active_fraction,
            "sharpeConfidence": outer_confidence,
            "foldMetrics": fold_metrics,
        },
        "finalChampionDevelopmentPath": {
            "metrics": C._metrics(
                final_champion_path["net"],
                periods_per_year,
                final_champion_path["active"],
            ),
            "sharpeConfidence": final_champion_confidence,
        },
        "stressTests": stress_report,
        "pairedHysteresisComparison": paired,
        "turnoverComparison": turnover,
        "diagnostics": diagnostic_report,
        "lifetimeMultipleTesting": lifetime,
        "regimes": regime_report,
        "promotionGates": gates,
        "finalHoldout": final_holdout,
        "evidence": {
            "auditablePaths": evidence_paths,
        },
    }
    status = str(summary["status"])
    if status == "ready_for_final_holdout":
        C._write_json(output_dir / "summary.json", summary)
        return summary
    return _finalize_terminal(
        output_dir, summary, registration_sha, campaign_manifest_sha
    )


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"risk-controlled reversal campaign failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
