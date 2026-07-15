#!/usr/bin/env python3
"""Run the locked adaptive residual-reversal turnover campaign.

Development runs consume only the predecessor campaign's sealed development
CSV artifacts. The raw snapshot is not opened unless every development gate
passes and ``--open-final-holdout`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from io import BytesIO
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
import historical_datafeed as feed
import harness as W
import reversal_campaign as R
import run_historical_funding_campaign as H


CAMPAIGN_ID = "residual_reversal_turnover_v1"
REGISTRATION_VERSION = 1
REGISTRATION_PATH = (
    C.REPOSITORY_ROOT
    / "research-notes/registrations/residual-reversal-turnover-v1.json"
)
SOURCE_CAMPAIGN_ID = H.CAMPAIGN_ID
SOURCE_CAMPAIGN_DIRECTORY = ".tmp/research/historical-funding-campaign-v1"
SNAPSHOT_DIRECTORY = ".tmp/research/historical-funding-snapshot-v1"
OUTPUT_DIRECTORY = ".tmp/research/historical-reversal-campaign-v1"
SOURCE_CAMPAIGN_MANIFEST = "campaign-manifest.json"
SOURCE_CAMPAIGN_MANIFEST_SHA256 = (
    "686ddb5ae44c4b2e461ffedd5c2b8199da4399b75d14c9075636ac062a008776"
)
SOURCE_REGISTRATION_SHA256 = (
    "cedaea5af05c880af732ceb1a78c39d4056efc2b1065157bc7a1ce31ff684d9f"
)
SNAPSHOT_MANIFEST_SHA256 = (
    "0e970ef24bbda0a2ceff24af5b83bc1a70de60a96ab4940a729c505e9a9c801e"
)
DEVELOPMENT_PANEL_SHA256 = (
    "09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a"
)
DEVELOPMENT_SETTLEMENTS_SHA256 = (
    "2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899"
)
FULL_PANEL_DIGEST_SHA256 = (
    "11d0af89e1603fad91bceffae3847dfdaed819bf56a45fe6f9d6bec45c953c0a"
)
FULL_SETTLEMENTS_DIGEST_SHA256 = (
    "cc0daa4d86d8c2d64285cf12baae86dcb1d9cb5606525074cf023f77bd59c795"
)
REGISTERED_PANEL_COLUMNS = ("openTime", "closeTime", "close")
REGISTERED_SYMBOLS = H.REGISTERED_SYMBOLS
REGISTERED_START_OPEN_TIME = H.REGISTERED_START_OPEN_TIME
REGISTERED_END_OPEN_TIME = H.REGISTERED_END_OPEN_TIME
REGISTERED_OUTCOME_END_EXCLUSIVE = H.REGISTERED_OUTCOME_END_EXCLUSIVE
REGISTERED_DEVELOPMENT_ROWS = 4910
REGISTERED_HOLDOUT_RETURN_ROWS = 1227
HOLDOUT_REGISTRY_DIR = C.HOLDOUT_REGISTRY_DIR
IMPLEMENTATION_FILES = (
    "campaign_runner.py",
    "diagnostics.py",
    "funding_campaign.py",
    "harness.py",
    "historical_datafeed.py",
    "reversal_campaign.py",
    "run_historical_funding_campaign.py",
    "run_historical_reversal_campaign.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked adaptive residual-reversal turnover campaign"
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
        help="Registered reversal campaign evidence directory",
    )
    parser.add_argument("--open-final-holdout", action="store_true")
    return parser.parse_args(argv)


def _read_json_bytes(path: Path) -> tuple[bytes, object]:
    try:
        payload = path.read_bytes()
        return payload, json.loads(payload)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"JSON artifact is unreadable: {path}") from error


def _read_json_object(path: Path) -> dict[str, object]:
    _, value = _read_json_bytes(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _json_object_from_bytes(payload: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"JSON artifact is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _validate_registration(
    registration: dict[str, object],
) -> dict[str, object]:
    try:
        research_process = registration["researchProcess"]
        universe = registration["universe"]
        registered_data = registration["registeredData"]
        strategy = registration["strategy"]
        validation = registration["validation"]
        bankruptcy_policy = registration["bankruptcyPolicy"]
        promotion = registration["promotion"]
        holdout_policy = registration["holdoutPolicy"]
        trials = registration["trials"]
        if not isinstance(research_process, dict) or not all(
            isinstance(research_process.get(name), str)
            and bool(str(research_process[name]).strip())
            for name in (
                "hypothesisOrigin",
                "developmentContamination",
                "inferenceScope",
                "adaptationLimit",
            )
        ):
            raise TypeError
        if not all(
            isinstance(value, dict)
            for value in (
                universe,
                registered_data,
                strategy,
                validation,
                bankruptcy_policy,
                promotion,
                holdout_policy,
            )
        ):
            raise TypeError
        symbols = universe["symbols"]
        if (
            not isinstance(symbols, list)
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or len(symbols) != len(set(symbols))
        ):
            raise TypeError
        normalized_trials = [
            {
                "trial_id": row["id"],
                "horizon_hours": int(row["horizonHours"]),
                "horizon_bars": int(row["horizonBars"]),
                "rebalance_bars": int(row["rebalanceBars"]),
                "champion_eligible": row["championEligible"],
            }
            for row in trials
        ]
        expected_trials = [
            {
                "trial_id": spec.trial_id,
                "horizon_hours": spec.horizon_hours,
                "horizon_bars": spec.horizon_bars,
                "rebalance_bars": spec.rebalance_bars,
                "champion_eligible": spec.rebalance_bars == 3,
            }
            for spec in R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("campaign registration has an invalid schema") from error

    fixed_checks = {
        "campaign": registration.get("campaign") == CAMPAIGN_ID,
        "registrationVersion": registration.get("registrationVersion")
        == REGISTRATION_VERSION,
        "interval": universe.get("interval") == feed.CONTRACT_INTERVAL,
        "intervalMilliseconds": universe.get("intervalMilliseconds")
        == feed.CONTRACT_INTERVAL_MS,
        "symbols": tuple(symbols) == REGISTERED_SYMBOLS,
        "trialLedger": normalized_trials == expected_trials,
        "trialCount": len(expected_trials) == 6,
        "protocolStatus": research_process.get("protocolStatus")
        == "locked_after_disclosed_development_smoke",
        "protocolLockScope": research_process.get("protocolLockScope")
        == "locked_before_any_final_holdout_access",
        "preMergeDevelopmentSmoke": research_process.get(
            "preMergeDevelopmentSmoke"
        )
        == {
            "occurred": True,
            "authorizationStatus": "prohibited_pre_merge_execution",
            "dataScope": "registered_development_only",
            "finalHoldoutAccessed": False,
            "researchChoicesChangedAfterSmoke": False,
            "observations": [
                {
                    "trialId": "resrev_24h_rebalance_3bar",
                    "rebalancePhaseBars": 0,
                    "outcome": "portfolio_equity_exhausted",
                    "closeTime": 1_611_907_199_999,
                    "closeTimeSemantics": "interval_left_close",
                    "outcomeCloseTime": 1_611_935_999_999,
                },
                {
                    "trialId": "resrev_72h_rebalance_3bar",
                    "rebalancePhaseBars": 0,
                    "outcome": "completed_near_zero_equity",
                    "maximumTurnoverApprox": 208.51,
                },
                {
                    "trialId": "resrev_168h_rebalance_3bar",
                    "rebalancePhaseBars": 0,
                    "outcome": "completed_near_zero_equity",
                    "maximumTurnoverApprox": 234.69,
                },
                {
                    "trialScope": "all_three_rebalance_3bar_horizons",
                    "rebalancePhaseBars": [1, 2],
                    "outcome": "completed",
                    "maximumTurnoverApproxRange": [2.4, 3.24],
                },
            ],
            "postSmokeProtocolCorrections": [
                {
                    "correction": (
                        "Treat any primary-path equity exhaustion as a typed "
                        "campaign-level mechanical rejection."
                    ),
                    "classification": "conservative_fail_closed_policy",
                },
                {
                    "correction": (
                        "Carry the actual drifted portfolio state through frozen "
                        "outer-fold candidate changes instead of stitching "
                        "independently simulated states."
                    ),
                    "classification": "accounting_correctness_fix",
                },
                {
                    "correction": (
                        "Use full causal feature history but start final-holdout "
                        "execution from cash and charge the initial "
                        "cash-to-frozen-target turnover."
                    ),
                    "classification": "holdout_boundary_correctness_fix",
                },
                {
                    "correction": (
                        "Record equity exhaustion on derived development paths as "
                        "structured insufficient evidence and on a consumed "
                        "final-holdout path as a structured failed result with a "
                        "completed registry lifecycle, without broadening the "
                        "primary-path mechanically-invalid rule."
                    ),
                    "classification": "conservative_fail_closed_policy",
                },
                {
                    "correction": (
                        "Enforce the already stated rebalance-phase robustness rule "
                        "by confidence-gating both the frozen outer-fold composite "
                        "and frozen final-champion paths."
                    ),
                    "classification": "implementation_conformance_fix",
                },
            ],
            "outcomeUseForCorrections": (
                "The exposed development outcomes were not used to change the "
                "strategy, trial family, phases, thresholds, selection rule, or "
                "holdout success rule. The post-smoke corrections are fail-closed, "
                "implementation-conformance, or accounting-boundary corrections "
                "and cannot rescue the observed bankrupt path."
            ),
        },
        "sourceCampaign": registered_data.get("sourceCampaign")
        == SOURCE_CAMPAIGN_ID,
        "sourceCampaignDirectory": registered_data.get("sourceCampaignDirectory")
        == SOURCE_CAMPAIGN_DIRECTORY,
        "sourceCampaignManifest": registered_data.get("sourceCampaignManifest")
        == SOURCE_CAMPAIGN_MANIFEST,
        "sourceCampaignManifestSha256": registered_data.get(
            "sourceCampaignManifestSha256"
        )
        == SOURCE_CAMPAIGN_MANIFEST_SHA256,
        "sourceRegistrationSha256": registered_data.get(
            "sourceRegistrationSha256"
        )
        == SOURCE_REGISTRATION_SHA256,
        "snapshotDirectory": registered_data.get("snapshotDirectory")
        == SNAPSHOT_DIRECTORY,
        "snapshotManifest": registered_data.get("snapshotManifest")
        == "snapshot-manifest.json",
        "snapshotManifestSha256": registered_data.get("snapshotManifestSha256")
        == SNAPSHOT_MANIFEST_SHA256,
        "developmentPanel": registered_data.get("developmentPanel")
        == "registered-development-panel.csv",
        "developmentPanelSha256": registered_data.get("developmentPanelSha256")
        == DEVELOPMENT_PANEL_SHA256,
        "developmentSettlements": registered_data.get("developmentSettlements")
        == "registered-development-settlements.csv",
        "developmentSettlementsSha256": registered_data.get(
            "developmentSettlementsSha256"
        )
        == DEVELOPMENT_SETTLEMENTS_SHA256,
        "fullPanelDigestSha256": registered_data.get("fullPanelDigestSha256")
        == FULL_PANEL_DIGEST_SHA256,
        "fullSettlementsDigestSha256": registered_data.get(
            "fullSettlementsDigestSha256"
        )
        == FULL_SETTLEMENTS_DIGEST_SHA256,
        "startOpenTime": int(registered_data["startOpenTime"])
        == REGISTERED_START_OPEN_TIME,
        "endOpenTime": int(registered_data["endOpenTime"])
        == REGISTERED_END_OPEN_TIME,
        "outcomeEndTimeExclusive": int(
            registered_data["outcomeEndTimeExclusive"]
        )
        == REGISTERED_OUTCOME_END_EXCLUSIVE,
        "rows": int(registered_data["rows"]) == 6138,
        "developmentRows": int(registered_data["developmentRows"])
        == REGISTERED_DEVELOPMENT_ROWS,
        "rowSplit": int(registered_data["developmentRows"])
        + int(registered_data["holdoutBars"])
        == int(registered_data["rows"]),
        "developmentCutoff": int(registered_data["developmentCutoffOpenTime"])
        == int(registered_data["startOpenTime"])
        + (int(registered_data["developmentRows"]) - 1)
        * feed.CONTRACT_INTERVAL_MS,
        "holdoutStart": int(registered_data["holdoutStartOpenTime"])
        == int(registered_data["developmentCutoffOpenTime"])
        + feed.CONTRACT_INTERVAL_MS,
        "holdoutRows": int(registered_data["holdoutReturnRows"])
        == REGISTERED_HOLDOUT_RETURN_ROWS
        == int(registered_data["holdoutBars"]) - 1,
        "direction": strategy.get("direction") == "residual_reversal",
        "betaLookback": int(strategy["betaLookbackBars"]) == 21,
        "costBps": float(strategy["costBpsPerUnitTurnover"]) == 5.0,
        "fundingFilter": strategy.get("fundingFilter") == "none",
        "grossExposure": float(strategy["grossExposure"]) == 1.0,
        "rebalanceAnchor": int(strategy["rebalanceAnchorOpenTime"])
        == REGISTERED_START_OPEN_TIME,
        "rebalancePhase": int(strategy["rebalancePhaseBars"]) == 0,
        "signalDelay": int(strategy["signalDelayBars"]) == 1,
        "topN": int(strategy["topNPerSide"]) == 1,
        "primaryTrials": int(validation["primaryTrialCount"])
        == len(expected_trials)
        == 6,
        "phaseStressConfigurations": int(
            validation["phaseStressConfigurationCount"]
        )
        == 6,
        "newTrials": int(validation["newTrialCount"])
        == int(validation["primaryTrialCount"])
        + int(validation["phaseStressConfigurationCount"])
        == 12,
        "lifetimeTrials": int(validation["priorTrialCount"])
        + int(validation["newTrialCount"])
        == int(validation["lifetimeTrialCount"])
        == 33,
        "bootstrapBlock": int(validation["bootstrapBlockBars"]) == 3,
        "bootstrapReplications": int(validation["bootstrapReplications"]) == 2000,
        "bootstrapSeed": int(validation["bootstrapSeed"]) == 42,
        "featureWarmup": int(validation["featureWarmupRows"]) == 21,
        "developmentEvaluationRows": int(
            validation["developmentEvaluationRows"]
        )
        == 4888,
        "labelHorizon": int(validation["labelHorizonBars"]) == 1,
        "outerInitialTrain": int(validation["outerInitialTrain"]) == 2444,
        "outerTestSize": int(validation["outerTestSize"]) == 349,
        "outerFoldCount": int(validation["outerFoldCount"]) == 7,
        "innerInitialTrain": int(validation["innerInitialTrain"]) == 1222,
        "innerTestSize": int(validation["innerTestSize"]) == 244,
        "pairedHypotheses": int(validation["pairedComparisonHypotheses"]) == 3,
        "pairedAlpha": float(validation["pairedComparisonFamilyWiseAlpha"])
        == 0.05,
        "pboSlices": int(validation["pboSlices"]) == 10,
        "stressTests": validation["stressTests"]
        == [
            "cost1_5x",
            "cost2x",
            "additionalDelay1bar",
            "rebalancePhase1bar",
            "rebalancePhase2bar",
        ],
        "turnoverLimit": float(promotion["maximumChampionTurnoverRatio"])
        == 0.5,
        "minimumSymbols": int(promotion["minimumSymbols"]) == 10,
        "resolvedFunding": float(promotion["minimumResolvedFundingFraction"])
        == 1.0,
        "outerObservations": int(promotion["minimumOuterOosObservations"])
        == 1500,
        "activeFraction": float(promotion["minimumActiveFraction"]) == 0.25,
        "regimeObservations": int(promotion["minimumRegimeObservations"]) == 50,
        "regimeLoss": float(promotion["maximumRegimeLoss"]) == 0.05,
        "worstFoldLoss": float(promotion["maximumWorstFoldLoss"]) == 0.05,
        "dsrThreshold": float(
            promotion["currentCampaignDeflatedSharpeProbabilityMinimum"]
        )
        == 0.95,
        "lifetimePsrThreshold": float(
            promotion["lifetimeBonferroniPsrProbabilityMinimum"]
        )
        == 0.95,
        "pboThreshold": float(promotion["maximumPbo"]) == 0.2,
        "bankruptcyPolicy": bankruptcy_policy
        == {
            "pathScope": "all_six_primary_phase_zero_development_paths",
            "triggerExpression": "1 + netReturn <= 0",
            "campaignStatusOnTrigger": "mechanically_invalid",
            "requireEveryPrimaryDevelopmentPathBankruptcyFree": True,
            "allowReturnClipping": False,
            "allowAbsorbingBankruptcy": False,
            "allowPortfolioRestart": False,
            "allowTrialSubstitution": False,
            "allowParameterTuning": False,
            "failureOutput": "structured_mechanical_rejection_only",
            "holdoutDispositionOnTrigger": "reserved_unopened",
            "nonPrimaryDevelopmentDisposition": "structured_insufficient_evidence",
            "finalHoldoutDisposition": (
                "structured_failure_and_completed_registry"
            ),
        },
        "requireBankruptcyFree": promotion[
            "requireEveryPrimaryDevelopmentPathBankruptcyFree"
        ]
        is True,
        "championRebalance": int(promotion["requireChampionRebalanceBars"])
        == 3,
        "requireOuterCi": promotion["requireOuterOosSharpeCiAboveZero"] is True,
        "requireCostCi": promotion["requireCost2xSharpeCiAboveZero"] is True,
        "requireDelayCi": promotion[
            "requireAdditionalDelaySharpeCiAboveZero"
        ]
        is True,
        "requireStressExecution": promotion[
            "requireAllRegisteredStressPathsEvaluable"
        ]
        is True,
        "requirePairedCi": promotion[
            "requireMatchedRebalanceImprovementSharpeCiAboveZero"
        ]
        is True,
        "requirePhaseCi": promotion[
            "requireAllRebalancePhaseSharpeCiAboveZero"
        ]
        is True,
        "holdoutReserved": holdout_policy["reservedByDefault"] is True,
        "holdoutBlocksMechanicalFailure": holdout_policy[
            "blockOnMechanicallyInvalidDevelopment"
        ]
        is True,
        "holdoutStartsCash": holdout_policy["executionStateAtStart"] == "cash",
        "holdoutFeatureHistory": holdout_policy["featureHistory"]
        == "full_registered_history_through_holdout_decision_time",
        "holdoutEntryTurnover": holdout_policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ]
        is True,
        "holdoutGated": holdout_policy[
            "openOnlyAfterEveryDevelopmentGatePasses"
        ]
        is True,
        "holdoutRegistry": holdout_policy["overlapAwareOneShotRegistry"] is True,
    }
    failed = [name for name, passed in fixed_checks.items() if not passed]
    if failed:
        raise ValueError(
            "campaign registration violates fixed constraints: " + ", ".join(failed)
        )
    if (
        int(registered_data["outcomeEndTimeExclusive"])
        != int(registered_data["endOpenTime"]) + feed.CONTRACT_INTERVAL_MS
    ):
        raise ValueError("registered outcome end does not match the final bar")
    return registration


def _registration(
    path: Path = REGISTRATION_PATH,
) -> dict[str, object]:
    return _validate_registration(_read_json_object(path))


def _registration_and_sha(
    path: Path = REGISTRATION_PATH,
) -> tuple[dict[str, object], str]:
    payload, value = _read_json_bytes(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return _validate_registration(value), hashlib.sha256(payload).hexdigest()


def _registration_sha(path: Path = REGISTRATION_PATH) -> str:
    return C._file_digest(path)


def _implementation_artifacts() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: C._file_digest(root / name) for name in IMPLEMENTATION_FILES}


def _implementation_sha() -> str:
    return C._json_digest(_implementation_artifacts())


def _artifact_path(directory: Path, registered_name: object) -> Path:
    if (
        not isinstance(registered_name, str)
        or Path(registered_name).name != registered_name
    ):
        raise ValueError("registered source artifact name must be a plain filename")
    return directory / registered_name


def _read_pinned_bytes(path: Path, expected_sha: object) -> tuple[bytes, str]:
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"registered SHA-256 is invalid for {path.name}")
    if not path.is_file():
        raise ValueError(f"registered source artifact is missing: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"registered source artifact is unreadable: {path}") from error
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha:
        raise ValueError(f"registered source artifact hash mismatch: {path.name}")
    return payload, observed


def _require_file_digest(path: Path, expected_sha: object) -> str:
    _, observed = _read_pinned_bytes(path, expected_sha)
    return observed


def _validate_source_manifest(
    manifest: Mapping[str, object], registration: Mapping[str, object]
) -> None:
    registered_data = registration["registeredData"]
    if not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid source settings")
    source_data = manifest.get("registeredData")
    artifacts = manifest.get("artifacts")
    expected = {
        "campaign": manifest.get("campaign") == registered_data["sourceCampaign"],
        "registrationSha256": manifest.get("registrationSha256")
        == registered_data["sourceRegistrationSha256"],
        "snapshotManifestSha256": manifest.get("snapshotManifestSha256")
        == registered_data["snapshotManifestSha256"],
        "symbols": manifest.get("symbols") == registration["universe"]["symbols"],
        "sourceData": isinstance(source_data, Mapping),
        "artifacts": isinstance(artifacts, Mapping),
    }
    if isinstance(source_data, Mapping):
        expected.update(
            {
                "fullPanelDigest": source_data.get("panelSha256")
                == registered_data["fullPanelDigestSha256"],
                "fullSettlementsDigest": source_data.get("settlementsSha256")
                == registered_data["fullSettlementsDigestSha256"],
                "developmentRows": int(source_data.get("developmentRows", -1))
                == int(registered_data["developmentRows"]),
                "developmentCutoff": int(
                    source_data.get("developmentCutoffOpenTime", -1)
                )
                == int(registered_data["developmentCutoffOpenTime"]),
            }
        )
    if isinstance(artifacts, Mapping):
        expected.update(
            {
                "developmentPanel": artifacts.get("registeredDevelopmentPanel")
                == registered_data["developmentPanel"],
                "developmentPanelSha": artifacts.get("registeredPanelSha256")
                == registered_data["developmentPanelSha256"],
                "developmentSettlements": artifacts.get(
                    "registeredDevelopmentSettlements"
                )
                == registered_data["developmentSettlements"],
                "developmentSettlementsSha": artifacts.get(
                    "registeredSettlementsSha256"
                )
                == registered_data["developmentSettlementsSha256"],
            }
        )
    failed = [name for name, passed in expected.items() if not passed]
    if failed:
        raise ValueError(
            "source campaign manifest violates registered lineage: "
            + ", ".join(failed)
        )


def _panel_from_csv(
    payload: bytes, registration: Mapping[str, object]
) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(BytesIO(payload))
    expected_columns = ["symbol", *REGISTERED_PANEL_COLUMNS]
    if list(frame.columns) != expected_columns:
        raise ValueError("registered development panel columns changed")
    for column in REGISTERED_PANEL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["symbol", "openTime"]).any():
        raise ValueError("registered development panel has duplicate rows")
    registered_data = registration["registeredData"]
    universe = registration["universe"]
    if not isinstance(registered_data, Mapping) or not isinstance(universe, Mapping):
        raise ValueError("campaign registration has invalid panel settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    if set(frame["symbol"]) != set(symbols):
        raise ValueError("registered development panel symbols changed")
    panel = {
        symbol: frame[frame["symbol"] == symbol]
        .drop(columns="symbol")
        .sort_values("openTime")
        .reset_index(drop=True)
        for symbol in symbols
    }
    expected_rows = int(registered_data["developmentRows"])
    if any(len(symbol_frame) != expected_rows for symbol_frame in panel.values()):
        raise ValueError("registered development panel row count changed")
    common_times = C._common_times(panel)
    C._validate_market_grid(panel, common_times, feed.CONTRACT_INTERVAL_MS)
    if (
        common_times[0] != int(registered_data["startOpenTime"])
        or common_times[-1] != int(registered_data["developmentCutoffOpenTime"])
    ):
        raise ValueError("registered development panel time window changed")
    for symbol, symbol_frame in panel.items():
        open_times = symbol_frame["openTime"].to_numpy(dtype=np.int64)
        close_times = symbol_frame["closeTime"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            close_times, open_times + feed.CONTRACT_INTERVAL_MS - 1
        ):
            raise ValueError(f"{symbol} closeTime grid changed")
    return panel


def _settlements_from_csv(
    payload: bytes, registration: Mapping[str, object]
) -> tuple[list[F.FundingSettlement], pd.DataFrame, dict[str, object]]:
    audit = pd.read_csv(BytesIO(payload))
    expected_columns = [
        "symbol",
        "fundingTime",
        "fundingRate",
        "resolvedMarkPrice",
        "markSource",
        "markOpenTime",
    ]
    if list(audit.columns) != expected_columns:
        raise ValueError("registered development settlement columns changed")
    audit["symbol"] = audit["symbol"].astype(str)
    for column in ("fundingTime", "fundingRate", "resolvedMarkPrice"):
        audit[column] = pd.to_numeric(audit[column], errors="raise")
    universe = registration["universe"]
    registered_data = registration["registeredData"]
    if not isinstance(universe, Mapping) or not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid settlement settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    if set(audit["symbol"]) != set(symbols):
        raise ValueError("registered development settlement symbols changed")
    numeric = audit[["fundingRate", "resolvedMarkPrice"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(audit["resolvedMarkPrice"] <= 0):
        raise ValueError("registered development settlements are invalid")
    development_end_close = (
        int(registered_data["developmentCutoffOpenTime"])
        + feed.CONTRACT_INTERVAL_MS
        - 1
    )
    if (
        audit.empty
        or int(audit["fundingTime"].min()) < int(registered_data["startOpenTime"])
        or int(audit["fundingTime"].max()) > development_end_close
    ):
        raise ValueError("registered development settlement window changed")
    settlements = [
        F.FundingSettlement(
            symbol=str(row.symbol),
            funding_time=int(row.fundingTime),
            rate=float(row.fundingRate),
            resolved_mark_price=float(row.resolvedMarkPrice),
        )
        for row in audit.itertuples(index=False)
    ]
    gaps = []
    event_counts = {}
    for symbol in symbols:
        times = np.sort(
            audit.loc[audit["symbol"] == symbol, "fundingTime"].to_numpy(
                dtype=np.int64
            )
        )
        event_counts[symbol] = len(times)
        if len(times) > 1:
            gaps.extend(np.diff(times).tolist())
    coverage = {
        "returnedEvents": len(audit),
        "resolvedEvents": len(audit),
        "resolvedFraction": 1.0,
        "eventCountBySymbol": event_counts,
        "maximumObservedGapMilliseconds": int(max(gaps)) if gaps else 0,
        "resolutionScope": "hash-pinned predecessor development settlements",
    }
    return settlements, audit, coverage


def _load_development_inputs(
    source_dir: Path, registration: Mapping[str, object]
) -> tuple[
    dict[str, pd.DataFrame],
    list[F.FundingSettlement],
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
]:
    registered_data = registration["registeredData"]
    if not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid source settings")
    manifest_path = _artifact_path(
        source_dir, registered_data["sourceCampaignManifest"]
    )
    manifest_payload, manifest_sha = _read_pinned_bytes(
        manifest_path, registered_data["sourceCampaignManifestSha256"]
    )
    manifest = _json_object_from_bytes(manifest_payload, manifest_path)
    _validate_source_manifest(manifest, registration)
    panel_path = _artifact_path(source_dir, registered_data["developmentPanel"])
    settlements_path = _artifact_path(
        source_dir, registered_data["developmentSettlements"]
    )
    panel_payload, panel_sha = _read_pinned_bytes(
        panel_path, registered_data["developmentPanelSha256"]
    )
    settlements_payload, settlements_sha = _read_pinned_bytes(
        settlements_path, registered_data["developmentSettlementsSha256"]
    )
    panel = _panel_from_csv(panel_payload, registration)
    settlements, audit, coverage = _settlements_from_csv(
        settlements_payload, registration
    )
    source_evidence = {
        "directory": str(source_dir),
        "campaignManifest": manifest_path.name,
        "campaignManifestSha256": manifest_sha,
        "developmentPanel": panel_path.name,
        "developmentPanelSha256": panel_sha,
        "developmentSettlements": settlements_path.name,
        "developmentSettlementsSha256": settlements_sha,
    }
    return panel, settlements, audit, coverage, source_evidence


def _validated_campaign_manifest(
    path: Path, expected: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    payload, value = _read_json_bytes(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    if value != expected:
        raise ValueError("campaign manifest changed; use a new output directory")
    return value, hashlib.sha256(payload).hexdigest()


def _campaign_manifest(
    output_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_artifacts: Mapping[str, str],
    source_evidence: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    manifest_path = output_dir / "campaign-manifest.json"
    implementation_sha = C._json_digest(implementation_artifacts)
    expected = {
        "campaign": CAMPAIGN_ID,
        "registrationVersion": REGISTRATION_VERSION,
        "registrationSha256": registration_sha,
        "implementationSha256": implementation_sha,
        "implementationArtifacts": dict(implementation_artifacts),
        "sourceArtifacts": dict(source_evidence),
        "symbols": list(registration["universe"]["symbols"]),
        "trials": list(registration["trials"]),
        "registeredData": dict(registration["registeredData"]),
    }
    if manifest_path.exists():
        manifest, manifest_sha = _validated_campaign_manifest(
            manifest_path, expected
        )
        for record_name in (
            "summary.json",
            "mechanical-failure.json",
            "final-holdout-opened.json",
        ):
            record_path = output_dir / record_name
            if not record_path.exists():
                continue
            record = _read_json_object(record_path)
            if record.get("campaignManifestSha256") != manifest_sha:
                raise ValueError(
                    "campaign manifest bytes changed; use a new output directory"
                )
        return manifest, manifest_sha
    if (output_dir / "summary.json").exists():
        raise ValueError(
            "output directory has campaign artifacts but no manifest; "
            "use a new output directory"
        )
    C._write_json(manifest_path, expected)
    return _validated_campaign_manifest(manifest_path, expected)


def _price_precomputed_turnover(
    frame: pd.DataFrame, cost_per_turnover: float
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot price an empty evaluation path")
    weight_columns = sorted(
        column for column in frame.columns if str(column).startswith("weight_")
    )
    if not weight_columns or "turnover" not in frame:
        raise ValueError("evaluation path lacks weights or registered turnover")
    priced = frame.copy().reset_index(drop=True)
    gross = priced["gross"].to_numpy(dtype=float)
    turnover = priced["turnover"].to_numpy(dtype=float)
    weights = priced[weight_columns].to_numpy(dtype=float)
    if (
        not np.isfinite(gross).all()
        or not np.isfinite(turnover).all()
        or not np.isfinite(weights).all()
        or np.any(turnover < 0)
    ):
        raise ValueError("evaluation path contains invalid accounting values")
    priced["cost"] = cost_per_turnover * turnover
    priced["net"] = gross - priced["cost"]
    priced["active"] = np.count_nonzero(np.abs(weights) > 1e-12, axis=1)
    return priced


def _reprice_details(
    details: Mapping[str, pd.DataFrame],
    index: pd.Index,
    cost_per_turnover: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    repriced = {}
    for name, frame in details.items():
        path = frame.reindex(index).reset_index()
        if path.isna().any().any():
            raise ValueError(f"trial {name} is incomplete on the evaluation index")
        repriced[name] = _price_precomputed_turnover(
            path, cost_per_turnover
        ).set_index("openTime")
    matrix = pd.DataFrame(
        {name: frame["net"] for name, frame in repriced.items()}, index=index
    )
    return matrix, repriced


def _nested_input(
    matrix: pd.DataFrame, details: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    columns: dict[str, object] = {"openTime": matrix.index.to_numpy()}
    candidates = {}
    for name in matrix.columns:
        detail = details[name].reindex(matrix.index)
        weights = [column for column in detail if column.startswith("weight_")]
        gross_column = f"{name}__gross"
        turnover_column = f"{name}__turnover"
        detail_columns = [
            column
            for column in ("priceGross", "fundingCashflow")
            if column in detail
        ]
        input_details = [f"{name}__{column}" for column in detail_columns]
        input_weights = [f"{name}__{column}" for column in weights]
        columns[gross_column] = detail["gross"].to_numpy(dtype=float)
        columns[turnover_column] = detail["turnover"].to_numpy(dtype=float)
        for source, target in zip(detail_columns, input_details):
            columns[target] = detail[source].to_numpy(dtype=float)
        for source, target in zip(weights, input_weights):
            columns[target] = detail[source].to_numpy(dtype=float)
        candidates[name] = {
            "grossColumn": gross_column,
            "turnoverColumn": turnover_column,
            "inputWeightColumns": tuple(input_weights),
            "outputWeightColumns": tuple(weights),
            "inputDetailColumns": tuple(input_details),
            "outputDetailColumns": tuple(detail_columns),
        }
    return pd.DataFrame(columns), candidates


def _evaluate_candidate(
    candidate: Mapping[str, object], test: pd.DataFrame
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "gross": test[str(candidate["grossColumn"])].to_numpy(dtype=float),
            "turnover": test[str(candidate["turnoverColumn"])].to_numpy(
                dtype=float
            ),
        }
    )
    for source, target in zip(
        candidate.get("inputDetailColumns", ()),
        candidate.get("outputDetailColumns", ()),
    ):
        result[str(target)] = test[str(source)].to_numpy(dtype=float)
    for source, target in zip(
        candidate["inputWeightColumns"], candidate["outputWeightColumns"]
    ):
        result[str(target)] = test[str(source)].to_numpy(dtype=float)
    return result


def _score_frame(frame: pd.DataFrame, cost_per_turnover: float) -> float:
    values = _price_precomputed_turnover(frame, cost_per_turnover)[
        "net"
    ].to_numpy(dtype=float)
    if len(values) < 2:
        return float("-inf")
    standard_deviation = float(np.std(values, ddof=1))
    return (
        float(np.mean(values) / standard_deviation)
        if standard_deviation > 1e-15
        else float("-inf")
    )


def _run_nested_selector(
    frame: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, object]],
    sizes: Mapping[str, int],
    label_horizon: int,
    cost_per_turnover: float,
) -> W.NestedRollingResult:
    nested = W.nested_rolling_origin(
        frame,
        candidates,
        fit_candidate=lambda candidate, _train: candidate,
        evaluate_candidate=_evaluate_candidate,
        score_candidate=lambda validation: _score_frame(
            validation, cost_per_turnover
        ),
        initial_train_size=sizes["initialTrain"],
        outer_test_size=sizes["outerTest"],
        inner_initial_train_size=sizes["innerInitialTrain"],
        inner_test_size=sizes["innerTest"],
        label_horizon=label_horizon,
    )
    return nested


def _rolling_select_candidate(
    frame: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, object]],
    initial_train_size: int,
    test_size: int,
    label_horizon: int,
    cost_per_turnover: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    splits = W.rolling_origin_splits(
        len(frame), initial_train_size, test_size, label_horizon
    )
    if not splits:
        raise ValueError("not enough observations for final rolling selection")
    score_rows = []
    for name, candidate in candidates.items():
        validation_frames = []
        for split in splits:
            validation = frame.iloc[split.test_slice]
            evaluated = _evaluate_candidate(candidate, validation)
            evaluated.insert(0, "openTime", validation["openTime"].to_numpy())
            evaluated.insert(
                0,
                "row_position",
                np.arange(split.test_start, split.test_stop, dtype=int),
            )
            validation_frames.append(evaluated)
        combined = pd.concat(validation_frames, ignore_index=True)
        score_rows.append(
            {
                "candidate": name,
                "score": _score_frame(combined, cost_per_turnover),
                "folds": len(splits),
                "validationRows": len(combined),
            }
        )
    scores = pd.DataFrame(score_rows)
    finite = scores[np.isfinite(scores["score"])]
    if finite.empty:
        raise ValueError("all final rolling-selection scores are non-finite")
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


def _assert_inputs_unchanged(
    source_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
) -> None:
    registered_data = registration["registeredData"]
    if not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid source settings")
    if _registration_sha() != registration_sha:
        raise ValueError("campaign registration changed during this run")
    if _implementation_sha() != implementation_sha:
        raise ValueError("campaign implementation changed during this run")
    for name_key, sha_key in (
        ("sourceCampaignManifest", "sourceCampaignManifestSha256"),
        ("developmentPanel", "developmentPanelSha256"),
        ("developmentSettlements", "developmentSettlementsSha256"),
    ):
        _require_file_digest(
            _artifact_path(source_dir, registered_data[name_key]),
            registered_data[sha_key],
        )


def _strategy_config(
    registration: Mapping[str, object],
    *,
    cost_multiplier: float = 1.0,
    additional_delay: int = 0,
    rebalance_phase_bars: int = 0,
) -> R.ReversalCampaignConfig:
    strategy = registration["strategy"]
    if not isinstance(strategy, Mapping):
        raise ValueError("campaign registration has invalid strategy settings")
    return R.ReversalCampaignConfig(
        interval_ms=feed.CONTRACT_INTERVAL_MS,
        cost_per_turnover=float(strategy["costBpsPerUnitTurnover"])
        / 10_000
        * cost_multiplier,
        gross_exposure=float(strategy["grossExposure"]),
        top_n=int(strategy["topNPerSide"]),
        signal_delay_bars=int(strategy["signalDelayBars"]) + additional_delay,
        rebalance_anchor_open_time=int(strategy["rebalanceAnchorOpenTime"]),
        rebalance_phase_bars=rebalance_phase_bars,
    )


def _trial_inputs(
    panel: Mapping[str, pd.DataFrame], registration: Mapping[str, object]
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    strategy = registration["strategy"]
    if not isinstance(strategy, Mapping):
        raise ValueError("campaign registration has invalid feature settings")
    close = H._close_frame(panel)
    horizons = tuple(
        dict.fromkeys(
            spec.horizon_hours
            for spec in R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
        )
    )
    residual_momentum = H._residual_momentum(
        close, int(strategy["betaLookbackBars"]), horizons
    )
    return close, residual_momentum


def _trials_on_panel(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: R.ReversalCampaignConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], tuple[R.ReversalCampaignSpec, ...]]:
    close, residual_momentum = _trial_inputs(panel, registration)
    _, close_time_details, specs = R.run_trial_matrix(
        close, residual_momentum, settlements, config
    )
    details = {
        name: _adapt_close_time_detail(frame)
        for name, frame in close_time_details.items()
    }
    matrix = pd.DataFrame(
        {name: frame["net"] for name, frame in details.items()},
        index=next(iter(details.values())).index,
    )
    return matrix, details, specs


def _adapt_close_time_detail(frame: pd.DataFrame) -> pd.DataFrame:
    adapted = frame.reset_index()
    adapted["openTime"] = (
        pd.to_numeric(adapted["closeTime"], errors="raise").astype(np.int64)
        - feed.CONTRACT_INTERVAL_MS
        + 1
    )
    return adapted.drop(columns="closeTime").set_index("openTime")


def _activation_mask(
    close_index: pd.Index,
    spec: R.ReversalCampaignSpec,
    config: R.ReversalCampaignConfig,
) -> np.ndarray:
    close_times = close_index.to_numpy(dtype=np.int64)
    open_times = close_times - config.interval_ms + 1
    offsets = open_times - config.rebalance_anchor_open_time
    if np.any(offsets < 0) or np.any(offsets % config.interval_ms != 0):
        raise ValueError("stateful outer path is off the registered time grid")
    boundaries = offsets // config.interval_ms
    scheduled = (
        boundaries - config.rebalance_phase_bars
    ) % spec.rebalance_bars == 0
    activations = np.zeros(len(close_index), dtype=bool)
    delay = config.signal_delay_bars
    if delay < len(activations):
        activations[delay:] = scheduled[:-delay]
    return activations


def _stateful_outer_choices(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    config: R.ReversalCampaignConfig,
    specs: Sequence[R.ReversalCampaignSpec],
    matrix_index: pd.Index,
    outer_folds: pd.DataFrame,
) -> pd.DataFrame:
    specs_by_name = {spec.trial_id: spec for spec in specs}
    selected_by_position: dict[int, tuple[int, str]] = {}
    for fold in outer_folds.to_dict("records"):
        fold_number = int(fold["outer_fold"])
        selected = str(fold["selected_candidate"])
        if selected not in specs_by_name:
            raise ValueError("outer fold selected an unknown reversal trial")
        for position in range(int(fold["test_start"]), int(fold["test_stop"])):
            if position in selected_by_position:
                raise ValueError("outer evaluation folds overlap")
            selected_by_position[position] = (fold_number, selected)
    positions = np.asarray(sorted(selected_by_position), dtype=int)
    if len(positions) == 0 or (
        len(positions) > 1 and not np.all(np.diff(positions) == 1)
    ):
        raise ValueError("outer evaluation folds do not form one contiguous path")
    if positions[0] < 0 or positions[-1] >= len(matrix_index):
        raise ValueError("outer evaluation fold exceeds the development matrix")
    evaluation_index = pd.Index(matrix_index[positions], name="openTime")

    strategy = registration["strategy"]
    if not isinstance(strategy, Mapping):
        raise ValueError("campaign registration has invalid feature settings")
    close = H._close_frame(panel)
    close_open_times = pd.Index(
        close.index.to_numpy(dtype=np.int64) - config.interval_ms + 1,
        name="openTime",
    )
    try:
        first_close_position = int(close_open_times.get_loc(evaluation_index[0]))
        last_close_position = int(close_open_times.get_loc(evaluation_index[-1]))
    except KeyError as error:
        raise ValueError(
            "outer evaluation time is absent from the close grid"
        ) from error
    if last_close_position - first_close_position + 1 != len(evaluation_index):
        raise ValueError("outer evaluation times are not contiguous on the close grid")
    close_slice = close.iloc[first_close_position : last_close_position + 2]
    if len(close_slice) != len(evaluation_index) + 1:
        raise ValueError("outer evaluation lacks its final outcome close")

    selected_names = [selected_by_position[int(position)][1] for position in positions]
    selected_specs = {
        name: specs_by_name[name] for name in dict.fromkeys(selected_names)
    }
    horizons = tuple(
        dict.fromkeys(spec.horizon_hours for spec in selected_specs.values())
    )
    residual_momentum = H._residual_momentum(
        close, int(strategy["betaLookbackBars"]), horizons
    )
    targets = {
        name: R.weights_for_trial(
            residual_momentum[spec.horizon_hours], spec, config
        )
        for name, spec in selected_specs.items()
    }
    activations = {
        name: _activation_mask(close.index, spec, config)
        for name, spec in selected_specs.items()
    }
    composite_targets = pd.DataFrame(
        0.0, index=close_slice.index.copy(), columns=close.columns.copy()
    )
    composite_activations = np.zeros(len(close_slice), dtype=bool)
    for local_row, (full_row, name) in enumerate(
        zip(range(first_close_position, last_close_position + 1), selected_names)
    ):
        composite_targets.iloc[local_row] = targets[name].iloc[full_row]
        candidate_changed = local_row == 0 or name != selected_names[local_row - 1]
        composite_activations[local_row] = (
            candidate_changed or activations[name][full_row]
        )
    composite_targets.iloc[-1] = composite_targets.iloc[-2]
    try:
        evaluated = _adapt_close_time_detail(
            R.evaluate_drifted_intervals(
                close_slice,
                composite_targets,
                composite_activations,
                settlements,
                config,
            )
        ).reset_index()
    except R.PortfolioBankruptcyError as error:
        failure_open_time = error.interval_left_close_time - config.interval_ms + 1
        try:
            local_row = int(evaluation_index.get_loc(failure_open_time))
        except (KeyError, TypeError, ValueError) as lookup_error:
            raise ValueError(
                "stateful bankruptcy time is absent from the evaluation path"
            ) from lookup_error
        raise R.PortfolioBankruptcyError(
            error.interval_left_close_time, selected_names[local_row]
        ) from error
    if not np.array_equal(
        evaluated["openTime"].to_numpy(dtype=np.int64),
        evaluation_index.to_numpy(dtype=np.int64),
    ):
        raise ValueError("stateful outer evaluation time index changed")
    evaluated.insert(0, "selected_candidate", selected_names)
    evaluated.insert(
        0,
        "outer_fold",
        [selected_by_position[int(position)][0] for position in positions],
    )
    evaluated.insert(0, "row_position", positions)
    return evaluated


def _eligible_names(
    specs: Sequence[R.ReversalCampaignSpec], required_rebalance_bars: int
) -> list[str]:
    names = [
        spec.trial_id
        for spec in specs
        if spec.rebalance_bars == required_rebalance_bars
    ]
    if len(names) != 3:
        raise ValueError("registered champion-eligible trial family changed")
    return names


def _eligible_candidates(
    candidates: Mapping[str, Mapping[str, object]], eligible_names: Sequence[str]
) -> dict[str, Mapping[str, object]]:
    if any(name not in candidates for name in eligible_names):
        raise ValueError("registered champion-eligible candidate is absent")
    return {name: candidates[name] for name in eligible_names}


def _nested_sizes(registration: Mapping[str, object]) -> dict[str, int]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid validation settings")
    return {
        "initialTrain": int(validation["outerInitialTrain"]),
        "outerTest": int(validation["outerTestSize"]),
        "innerInitialTrain": int(validation["innerInitialTrain"]),
        "innerTest": int(validation["innerTestSize"]),
    }


def _lifetime_multiple_testing(
    diagnostic_matrix: pd.DataFrame,
    champion: str,
    registration: Mapping[str, object],
) -> dict[str, object]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid multiple-testing settings")
    lifetime_trials = int(validation["lifetimeTrialCount"])
    try:
        result = diagnostics.deflated_sharpe_ratio(
            diagnostic_matrix,
            selected_trial=champion,
            periods_per_year=365.0,
            independent_trials=1,
        ).to_dict()
        probability = float(result["probability"])
        adjusted = max(
            0.0, 1.0 - min(1.0, lifetime_trials * (1.0 - probability))
        )
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": lifetime_trials,
            "singleTrialProbability": probability,
            "adjustedProbability": adjusted,
            "underlyingProbabilisticSharpe": result,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {
            "method": "bonferroni_adjusted_probabilistic_sharpe_ratio",
            "priorTrials": int(validation["priorTrialCount"]),
            "newTrials": int(validation["newTrialCount"]),
            "lifetimeTrials": lifetime_trials,
            "adjustedProbability": 0.0,
            "error": str(error),
        }


def _matched_control(
    champion: str, specs: Sequence[R.ReversalCampaignSpec]
) -> R.ReversalCampaignSpec:
    by_name = {spec.trial_id: spec for spec in specs}
    selected = by_name.get(champion)
    if selected is None or selected.rebalance_bars != 3:
        raise ValueError("selected champion is not a registered 3-bar trial")
    controls = [
        spec
        for spec in specs
        if spec.horizon_hours == selected.horizon_hours
        and spec.rebalance_bars == 1
    ]
    if len(controls) != 1:
        raise ValueError("selected champion has no unique matched 1-bar control")
    return controls[0]


def _paired_rebalance_comparison(
    matrix: pd.DataFrame,
    champion: str,
    specs: Sequence[R.ReversalCampaignSpec],
    registration: Mapping[str, object],
    periods_per_year: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    validation = registration["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("campaign registration has invalid paired-test settings")
    hypotheses = int(validation["pairedComparisonHypotheses"])
    family_alpha = float(validation["pairedComparisonFamilyWiseAlpha"])
    comparison_alpha = family_alpha / hypotheses
    horizons = sorted({spec.horizon_hours for spec in specs})
    if hypotheses != len(horizons):
        raise ValueError("paired-comparison family size changed from registration")
    selected = next(spec for spec in specs if spec.trial_id == champion)
    rows = []
    for horizon_hours in horizons:
        control = next(
            spec
            for spec in specs
            if spec.horizon_hours == horizon_hours and spec.rebalance_bars == 1
        )
        slower = next(
            spec
            for spec in specs
            if spec.horizon_hours == horizon_hours and spec.rebalance_bars == 3
        )
        difference = matrix[slower.trial_id] - matrix[control.trial_id]
        interval = C._bootstrap_ci(
            difference,
            periods_per_year,
            feed.CONTRACT_INTERVAL_MS,
            bootstrap_reps,
            bootstrap_seed,
            alpha=comparison_alpha,
        )
        passed = math.isfinite(interval[0]) and interval[0] > 0
        rows.append(
            {
                "horizonHours": horizon_hours,
                "controlTrial": control.trial_id,
                "threeBarTrial": slower.trial_id,
                "threeBarMinusOneBarMetrics": C._metrics(
                    difference, periods_per_year
                ),
                "threeBarMinusOneBarSimultaneousSharpeCi": C._ci_json(interval),
                "selectedChampionPair": horizon_hours == selected.horizon_hours,
                "passed": passed,
            }
        )
    champion_row = next(row for row in rows if row["selectedChampionPair"])
    return {
        "method": "paired_3bar_minus_1bar_block_bootstrap",
        "estimand": "Sharpe(3bar net-return spread over matched 1bar)",
        "familyWiseAlpha": family_alpha,
        "comparisonAlpha": comparison_alpha,
        "comparisonConfidenceLevel": 1.0 - comparison_alpha,
        "successRule": (
            "the selected champion's simultaneous CI lower bound for "
            "Sharpe(3bar net-return spread over matched 1bar) exceeds zero"
        ),
        "champion": champion,
        "championPassed": bool(champion_row["passed"]),
        "anyHorizonPassed": any(bool(row["passed"]) for row in rows),
        "horizons": rows,
    }


def _champion_turnover_ratio(
    details: Mapping[str, pd.DataFrame],
    evaluation_index: pd.Index,
    champion: str,
    specs: Sequence[R.ReversalCampaignSpec],
) -> dict[str, object]:
    control = _matched_control(champion, specs)
    selected_turnover = details[champion].reindex(evaluation_index)[
        "turnover"
    ].to_numpy(dtype=float)
    control_turnover = details[control.trial_id].reindex(evaluation_index)[
        "turnover"
    ].to_numpy(dtype=float)
    selected_mean = float(np.mean(selected_turnover))
    control_mean = float(np.mean(control_turnover))
    ratio = (
        selected_mean / control_mean
        if math.isfinite(control_mean) and control_mean > 0
        else float("inf")
    )
    return {
        "champion": champion,
        "matchedControl": control.trial_id,
        "observations": len(evaluation_index),
        "championMeanTurnover": selected_mean,
        "controlMeanTurnover": control_mean,
        "ratio": C._finite_number(ratio),
        "denominatorValid": math.isfinite(control_mean) and control_mean > 0,
    }


def _stress_campaign(
    label: str,
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    matrix_index: pd.Index,
    outer_folds: pd.DataFrame,
    eligible_names: Sequence[str],
    periods_per_year: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[dict[str, object], pd.DataFrame | None, tuple[float, float]]:
    if label == "cost1_5x":
        config = _strategy_config(registration, cost_multiplier=1.5)
    elif label == "cost2x":
        config = _strategy_config(registration, cost_multiplier=2.0)
    elif label == "additionalDelay1bar":
        config = _strategy_config(registration, additional_delay=1)
    elif label == "rebalancePhase1bar":
        config = _strategy_config(registration, rebalance_phase_bars=1)
    elif label == "rebalancePhase2bar":
        config = _strategy_config(registration, rebalance_phase_bars=2)
    else:
        raise ValueError(f"unknown registered stress: {label}")
    specs = R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
    selected = set(outer_folds["selected_candidate"].astype(str))
    if not selected.issubset(set(eligible_names)):
        raise ValueError("stress path changed the frozen outer selections")
    try:
        oos = _stateful_outer_choices(
            panel,
            settlements,
            registration,
            config,
            specs,
            matrix_index,
            outer_folds,
        )
    except R.PortfolioBankruptcyError as error:
        return (
            {
                "status": "execution_failed",
                "failure": {
                    **_portfolio_bankruptcy_failure(error),
                    "path": "nested_outer_oos_stress",
                    "stress": label,
                },
            },
            None,
            (float("nan"), float("nan")),
        )
    interval = C._bootstrap_ci(
        oos["net"],
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        bootstrap_reps,
        bootstrap_seed,
    )
    return (
        {
            "status": "evaluated",
            "nestedOuterOos": {
                "metrics": C._metrics(oos["net"], periods_per_year, oos["active"]),
                "sharpeBootstrap95": C._ci_json(interval),
            }
        },
        oos,
        interval,
    )


def _phase_configuration_matrix(
    panel: Mapping[str, pd.DataFrame],
    settlements: Sequence[F.FundingSettlement],
    registration: Mapping[str, object],
    matrix_index: pd.Index,
    eligible_names: Sequence[str],
) -> pd.DataFrame:
    configurations = {}
    for phase_bars in (1, 2):
        config = _strategy_config(
            registration, rebalance_phase_bars=phase_bars
        )
        _, raw_details, _ = _trials_on_panel(
            panel, settlements, registration, config
        )
        phase_matrix, _ = _reprice_details(
            raw_details, matrix_index, config.cost_per_turnover
        )
        for name in eligible_names:
            configurations[f"{name}__rebalance_phase_{phase_bars}bar"] = (
                phase_matrix[name]
            )
    result = pd.DataFrame(configurations, index=matrix_index)
    if result.shape[1] != 6 or result.isna().any().any():
        raise ValueError("registered phase-stress configuration family changed")
    return result


def _load_full_registered_inputs(
    snapshot_dir: Path, registration: Mapping[str, object]
) -> tuple[
    dict[str, pd.DataFrame],
    list[F.FundingSettlement],
    dict[str, object],
]:
    registered_data = registration["registeredData"]
    if not isinstance(registered_data, Mapping):
        raise ValueError("campaign registration has invalid snapshot settings")
    source_registration, source_registration_sha = H._registration_and_sha()
    if source_registration_sha != registered_data["sourceRegistrationSha256"]:
        raise ValueError("source campaign registration hash changed")
    _, snapshot_manifest_sha, snapshot = H._load_snapshot(
        snapshot_dir, source_registration, source_registration_sha, False
    )
    if snapshot_manifest_sha != registered_data["snapshotManifestSha256"]:
        raise ValueError("source snapshot manifest hash changed")
    panel = H._contract_panel(snapshot, source_registration)
    settlements, audit, coverage = H._resolved_settlements(
        snapshot, source_registration
    )
    if (
        C._panel_digest(panel, REGISTERED_PANEL_COLUMNS)
        != registered_data["fullPanelDigestSha256"]
    ):
        raise ValueError("source full-panel digest changed")
    if H._frame_digest(audit) != registered_data["fullSettlementsDigestSha256"]:
        raise ValueError("source full-settlement digest changed")
    return panel, settlements, coverage


def _portfolio_bankruptcy_failure(
    error: R.PortfolioBankruptcyError,
) -> dict[str, object]:
    if not isinstance(error.trial_id, str) or not error.trial_id:
        raise ValueError("portfolio bankruptcy has no registered trial context")
    return {
        "reason": "portfolio_equity_exhausted",
        "trialId": error.trial_id,
        "closeTime": error.close_time,
        "closeTimeSemantics": "interval_left_close",
        "outcomeCloseTime": error.outcome_close_time,
    }


def _complete_holdout_records(
    marker: Path,
    output_record: Path,
    opening_record: Mapping[str, object],
    completion_record: Mapping[str, object],
) -> None:
    with C._holdout_registry_lock(marker.parent):
        for path in (marker, output_record):
            if _read_json_object(path) != opening_record:
                raise ValueError("final holdout opening record changed before completion")
        # Prefer a durable shared consumed marker if the second write fails.
        C._write_json(marker, completion_record)
        C._write_json(output_record, completion_record)


def _mechanically_invalid_result(
    args: argparse.Namespace,
    output_dir: Path,
    source_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
    campaign_manifest_sha: str,
    source_evidence: Mapping[str, object],
    settlement_coverage: Mapping[str, object],
    error: R.PortfolioBankruptcyError,
) -> dict[str, object]:
    registered_data = registration["registeredData"]
    universe = registration["universe"]
    strategy = registration["strategy"]
    bankruptcy_policy = registration["bankruptcyPolicy"]
    holdout_policy = registration["holdoutPolicy"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            registered_data,
            universe,
            strategy,
            bankruptcy_policy,
            holdout_policy,
        )
    ):
        raise ValueError("campaign registration has invalid failure settings")
    failure = _portfolio_bankruptcy_failure(error)
    holdout_start = int(registered_data["holdoutStartOpenTime"])
    holdout_end = int(registered_data["endOpenTime"])
    holdout_window = C._holdout_window(
        [str(symbol) for symbol in universe["symbols"]],
        feed.CONTRACT_INTERVAL,
        holdout_start,
        holdout_end,
    )
    holdout_identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": registered_data["fullPanelDigestSha256"],
            "window": holdout_window,
        }
    )
    final_holdout = {
        "status": "reserved",
        "identitySha256": holdout_identity,
        "openRequested": bool(args.open_final_holdout),
        "openBlockedBy": ["bankruptcyFree"],
        "executionStateAtStart": holdout_policy["executionStateAtStart"],
        "featureHistory": holdout_policy["featureHistory"],
        "chargeInitialCashToFrozenTargetTurnover": holdout_policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ],
        "startOpenTime": holdout_start,
        "endOpenTime": holdout_end,
        "outcomeEndTimeExclusive": int(holdout_window["outcomeEndTimeExclusive"]),
        "rows": int(registered_data["holdoutReturnRows"]),
    }
    failure_record = {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "status": bankruptcy_policy["campaignStatusOnTrigger"],
        "bankruptcyFree": False,
        "mechanicalFailure": failure,
        "finalHoldout": final_holdout,
    }
    failure_path = output_dir / "mechanical-failure.json"
    _assert_inputs_unchanged(
        source_dir, registration, registration_sha, implementation_sha
    )
    C._write_json(failure_path, failure_record)
    failure_sha = C._file_digest(failure_path)
    summary = {
        **failure_record,
        "symbols": [str(symbol) for symbol in universe["symbols"]],
        "interval": feed.CONTRACT_INTERVAL,
        "configuration": {"strategy": dict(strategy)},
        "data": {
            "registeredRows": int(registered_data["rows"]),
            "developmentRows": int(registered_data["developmentRows"]),
            "sourceArtifacts": dict(source_evidence),
            "fullPanelDigestSha256": registered_data["fullPanelDigestSha256"],
            "snapshotManifestSha256": registered_data["snapshotManifestSha256"],
            "settlements": dict(settlement_coverage),
            "survivorshipLimitation": universe["survivorshipLimitation"],
        },
        "trials": [
            spec.to_dict()
            for spec in R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
        ],
        "promotionGates": {"bankruptcyFree": False},
        "evidence": {
            "mechanicalFailure": str(failure_path.resolve()),
            "mechanicalFailureSha256": failure_sha,
        },
    }
    _assert_inputs_unchanged(
        source_dir, registration, registration_sha, implementation_sha
    )
    C._write_json(output_dir / "summary.json", summary)
    return summary


def _stateful_development_failure_result(
    args: argparse.Namespace,
    output_dir: Path,
    source_dir: Path,
    registration: Mapping[str, object],
    registration_sha: str,
    implementation_sha: str,
    campaign_manifest_sha: str,
    source_evidence: Mapping[str, object],
    settlement_coverage: Mapping[str, object],
    error: R.PortfolioBankruptcyError,
    stage: str,
) -> dict[str, object]:
    registered_data = registration["registeredData"]
    universe = registration["universe"]
    strategy = registration["strategy"]
    holdout_policy = registration["holdoutPolicy"]
    if not all(
        isinstance(value, Mapping)
        for value in (registered_data, universe, strategy, holdout_policy)
    ):
        raise ValueError("campaign registration has invalid failure settings")
    failure = {
        **_portfolio_bankruptcy_failure(error),
        "path": stage,
    }
    holdout_start = int(registered_data["holdoutStartOpenTime"])
    holdout_end = int(registered_data["endOpenTime"])
    holdout_window = C._holdout_window(
        [str(symbol) for symbol in universe["symbols"]],
        feed.CONTRACT_INTERVAL,
        holdout_start,
        holdout_end,
    )
    holdout_identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": registered_data["fullPanelDigestSha256"],
            "window": holdout_window,
        }
    )
    final_holdout = {
        "status": "reserved",
        "identitySha256": holdout_identity,
        "openRequested": bool(args.open_final_holdout),
        "openBlockedBy": ["statefulDevelopmentExecution"],
        "executionStateAtStart": holdout_policy["executionStateAtStart"],
        "featureHistory": holdout_policy["featureHistory"],
        "chargeInitialCashToFrozenTargetTurnover": holdout_policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ],
        "startOpenTime": holdout_start,
        "endOpenTime": holdout_end,
        "outcomeEndTimeExclusive": int(holdout_window["outcomeEndTimeExclusive"]),
        "rows": int(registered_data["holdoutReturnRows"]),
    }
    failure_record = {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "status": "insufficient_evidence",
        "bankruptcyFree": True,
        "statefulDevelopmentExecutionFree": False,
        "developmentExecutionFailure": failure,
        "finalHoldout": final_holdout,
    }
    failure_path = output_dir / "development-execution-failure.json"
    _assert_inputs_unchanged(
        source_dir, registration, registration_sha, implementation_sha
    )
    C._write_json(failure_path, failure_record)
    failure_sha = C._file_digest(failure_path)
    summary = {
        **failure_record,
        "symbols": [str(symbol) for symbol in universe["symbols"]],
        "interval": feed.CONTRACT_INTERVAL,
        "configuration": {"strategy": dict(strategy)},
        "data": {
            "registeredRows": int(registered_data["rows"]),
            "developmentRows": int(registered_data["developmentRows"]),
            "sourceArtifacts": dict(source_evidence),
            "fullPanelDigestSha256": registered_data["fullPanelDigestSha256"],
            "snapshotManifestSha256": registered_data["snapshotManifestSha256"],
            "settlements": dict(settlement_coverage),
            "survivorshipLimitation": universe["survivorshipLimitation"],
        },
        "trials": [
            spec.to_dict()
            for spec in R.campaign_specs(feed.CONTRACT_INTERVAL_MS)
        ],
        "promotionGates": {
            "bankruptcyFree": True,
            "statefulDevelopmentExecution": False,
        },
        "evidence": {
            "developmentExecutionFailure": str(failure_path.resolve()),
            "developmentExecutionFailureSha256": failure_sha,
        },
    }
    _assert_inputs_unchanged(
        source_dir, registration, registration_sha, implementation_sha
    )
    C._write_json(output_dir / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    with C._campaign_output_lock(output_dir):
        return _run_locked(args, output_dir)


def _run_locked(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, object]:
    registration, registration_sha = _registration_and_sha()
    implementation_artifacts = _implementation_artifacts()
    implementation_sha = C._json_digest(implementation_artifacts)
    source_dir = Path(args.source_campaign_dir)
    snapshot_dir = Path(args.snapshot_dir)
    (
        development_panel,
        development_settlements,
        _development_settlement_audit,
        settlement_coverage,
        source_evidence,
    ) = _load_development_inputs(source_dir, registration)
    manifest, campaign_manifest_sha = _campaign_manifest(
        output_dir,
        registration,
        registration_sha,
        implementation_artifacts,
        source_evidence,
    )

    registered_data = registration["registeredData"]
    validation = registration["validation"]
    promotion = registration["promotion"]
    universe = registration["universe"]
    strategy = registration["strategy"]
    holdout_policy = registration["holdoutPolicy"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            registered_data,
            validation,
            promotion,
            universe,
            strategy,
            holdout_policy,
        )
    ):
        raise ValueError("campaign registration has invalid runtime settings")
    symbols = [str(symbol) for symbol in universe["symbols"]]
    config = _strategy_config(registration)
    try:
        matrix_raw, details_raw, specs = _trials_on_panel(
            development_panel, development_settlements, registration, config
        )
    except R.PortfolioBankruptcyError as error:
        return _mechanically_invalid_result(
            args,
            output_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
            campaign_manifest_sha,
            source_evidence,
            settlement_coverage,
            error,
        )
    C._assert_output_holdout_not_consumed(HOLDOUT_REGISTRY_DIR, output_dir)
    warmup = int(validation["featureWarmupRows"])
    if len(matrix_raw) <= warmup:
        raise ValueError("development window is too short after feature warmup")
    evaluation_index = matrix_raw.index[warmup:]
    if len(evaluation_index) != int(validation["developmentEvaluationRows"]):
        raise ValueError("development evaluation rows changed from registration")
    matrix, details = _reprice_details(
        details_raw, evaluation_index, config.cost_per_turnover
    )
    periods_per_year = C._periods_per_year(feed.CONTRACT_INTERVAL_MS)
    sizes = _nested_sizes(registration)
    label_horizon = int(validation["labelHorizonBars"])
    nested_frame, all_candidates = _nested_input(matrix, details)
    required_rebalance_bars = int(promotion["requireChampionRebalanceBars"])
    eligible_names = _eligible_names(specs, required_rebalance_bars)
    candidates = _eligible_candidates(all_candidates, eligible_names)
    nested = _run_nested_selector(
        nested_frame,
        candidates,
        sizes,
        label_horizon,
        config.cost_per_turnover,
    )
    outer_lengths = (
        nested.outer_folds["test_stop"] - nested.outer_folds["test_start"]
    ).to_numpy(dtype=int)
    if (
        len(outer_lengths) != int(validation["outerFoldCount"])
        or not np.all(outer_lengths == sizes["outerTest"])
    ):
        raise ValueError("outer folds changed from the registered complete-fold policy")
    try:
        stateful_nested_oos = _stateful_outer_choices(
            development_panel,
            development_settlements,
            registration,
            config,
            specs,
            matrix.index,
            nested.outer_folds,
        )
    except R.PortfolioBankruptcyError as error:
        return _stateful_development_failure_result(
            args,
            output_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
            campaign_manifest_sha,
            source_evidence,
            settlement_coverage,
            error,
            "nested_outer_oos",
        )
    nested = replace(nested, oos=stateful_nested_oos)
    champion, final_selection_scores, final_selection_folds = (
        _rolling_select_candidate(
            nested_frame,
            candidates,
            sizes["innerInitialTrain"],
            sizes["innerTest"],
            label_horizon,
            config.cost_per_turnover,
        )
    )

    trial_metrics = {
        name: C._metrics(matrix[name], periods_per_year, details[name]["active"])
        for name in matrix.columns
    }
    nested_metrics = C._metrics(
        nested.oos["net"], periods_per_year, nested.oos["active"]
    )
    bootstrap_reps = int(validation["bootstrapReplications"])
    bootstrap_seed = int(validation["bootstrapSeed"])
    nested_ci = C._bootstrap_ci(
        nested.oos["net"],
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        bootstrap_reps,
        bootstrap_seed,
    )
    fold_metrics = C._fold_metrics(nested.oos, periods_per_year)
    worst_fold_return = min(
        float(metrics["totalReturn"]) for metrics in fold_metrics.values()
    )
    regime_report, _, labelled_nested_oos = C._regime_report(
        nested.oos,
        C._market_regime_labels(development_panel, feed.CONTRACT_INTERVAL_MS),
        periods_per_year,
        int(promotion["minimumRegimeObservations"]),
        float(promotion["maximumRegimeLoss"]),
    )
    try:
        phase_configuration_matrix = _phase_configuration_matrix(
            development_panel,
            development_settlements,
            registration,
            matrix.index,
            eligible_names,
        )
    except R.PortfolioBankruptcyError as error:
        return _stateful_development_failure_result(
            args,
            output_dir,
            source_dir,
            registration,
            registration_sha,
            implementation_sha,
            campaign_manifest_sha,
            source_evidence,
            settlement_coverage,
            error,
            "rebalance_phase_configuration_family",
        )
    diagnostic_configuration_matrix = pd.concat(
        [matrix, phase_configuration_matrix], axis=1
    )
    if diagnostic_configuration_matrix.shape[1] != int(
        validation["newTrialCount"]
    ):
        raise ValueError("registered diagnostic configuration count changed")
    selection_diagnostics, diagnostic_matrix, pbo_matrix = C._diagnostics(
        diagnostic_configuration_matrix,
        champion,
        periods_per_year,
        feed.CONTRACT_INTERVAL_MS,
        int(validation["pboSlices"]),
    )
    lifetime_multiple_testing = _lifetime_multiple_testing(
        diagnostic_matrix, champion, registration
    )
    paired_comparison = _paired_rebalance_comparison(
        matrix,
        champion,
        specs,
        registration,
        periods_per_year,
        bootstrap_reps,
        bootstrap_seed,
    )
    turnover_ratio = _champion_turnover_ratio(
        details, evaluation_index, champion, specs
    )

    stress_results: dict[str, object] = {}
    stress_paths: dict[str, pd.DataFrame] = {}
    stress_intervals: dict[str, tuple[float, float]] = {}
    phase_final_champion_intervals: dict[str, tuple[float, float]] = {}
    if final_selection_folds.empty:
        raise ValueError("final rolling selection produced no evaluation folds")
    final_selection_positions = np.concatenate(
        [
            np.arange(int(row.testStart), int(row.testStop), dtype=int)
            for row in final_selection_folds.itertuples(index=False)
        ]
    )
    if len(final_selection_positions) == 0 or (
        len(final_selection_positions) > 1
        and not np.all(np.diff(final_selection_positions) == 1)
    ):
        raise ValueError("final rolling-selection folds do not form one path")
    for label in validation["stressTests"]:
        result, path, interval = _stress_campaign(
            label,
            development_panel,
            development_settlements,
            registration,
            matrix.index,
            nested.outer_folds,
            eligible_names,
            periods_per_year,
            bootstrap_reps,
            bootstrap_seed,
        )
        stress_results[label] = result
        if path is not None:
            stress_paths[label] = path
        stress_intervals[label] = interval
        if label in ("rebalancePhase1bar", "rebalancePhase2bar"):
            phase_bars = 1 if label == "rebalancePhase1bar" else 2
            configuration = f"{champion}__rebalance_phase_{phase_bars}bar"
            frozen_final_returns = phase_configuration_matrix[configuration].iloc[
                final_selection_positions
            ]
            frozen_final_interval = C._bootstrap_ci(
                frozen_final_returns,
                periods_per_year,
                feed.CONTRACT_INTERVAL_MS,
                bootstrap_reps,
                bootstrap_seed,
            )
            phase_final_champion_intervals[label] = frozen_final_interval
            result["frozenFinalChampionSelectionOos"] = {
                "candidate": champion,
                "configuration": configuration,
                "metrics": C._metrics(frozen_final_returns, periods_per_year),
                "sharpeBootstrap95": C._ci_json(frozen_final_interval),
            }

    dsr_probability = selection_diagnostics.get("deflatedSharpe", {}).get(
        "probability", 0.0
    )
    pbo_probability = selection_diagnostics.get("pbo", {}).get(
        "probability", 1.0
    )
    active_fraction = float(nested_metrics["activeObservations"]) / float(
        nested_metrics["observations"]
    )
    champion_spec = next(spec for spec in specs if spec.trial_id == champion)
    turnover_value = turnover_ratio["ratio"]
    all_stress_paths_evaluable = all(
        stress_results[label].get("status") == "evaluated"
        for label in validation["stressTests"]
    )
    gates = {
        "bankruptcyFree": True,
        "symbolCount": len(development_panel)
        >= int(promotion["minimumSymbols"]),
        "resolvedFunding": float(settlement_coverage["resolvedFraction"])
        >= float(promotion["minimumResolvedFundingFraction"]),
        "outerOosObservations": int(nested_metrics["observations"])
        >= int(promotion["minimumOuterOosObservations"]),
        "outerOosActiveFraction": active_fraction
        >= float(promotion["minimumActiveFraction"]),
        "outerOosSharpeCiAboveZero": math.isfinite(nested_ci[0])
        and nested_ci[0] > 0,
        "worstOuterFoldLoss": worst_fold_return
        >= -float(promotion["maximumWorstFoldLoss"]),
        "regimeLoss": bool(regime_report["lossCapPassed"]),
        "regimeCoverage": bool(regime_report["observationCoveragePassed"]),
        "currentCampaignDeflatedSharpe": float(dsr_probability)
        >= float(promotion["currentCampaignDeflatedSharpeProbabilityMinimum"]),
        "lifetimeBonferroniPsr": float(
            lifetime_multiple_testing.get("adjustedProbability", 0.0)
        )
        >= float(promotion["lifetimeBonferroniPsrProbabilityMinimum"]),
        "pbo": float(pbo_probability) <= float(promotion["maximumPbo"]),
        "championRebalanceBars": champion_spec.rebalance_bars
        == required_rebalance_bars,
        "matchedRebalanceImprovement": bool(
            paired_comparison["championPassed"]
        ),
        "championTurnoverRatio": turnover_value is not None
        and float(turnover_value)
        <= float(promotion["maximumChampionTurnoverRatio"]),
        "allRegisteredStressPathsEvaluable": all_stress_paths_evaluable,
        "cost2xOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["cost2x"][0]
        )
        and stress_intervals["cost2x"][0] > 0,
        "additionalDelayOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["additionalDelay1bar"][0]
        )
        and stress_intervals["additionalDelay1bar"][0] > 0,
        "allRebalancePhaseOuterOosSharpeCiAboveZero": all(
            math.isfinite(stress_intervals[label][0])
            and stress_intervals[label][0] > 0
            for label in ("rebalancePhase1bar", "rebalancePhase2bar")
        ),
        "allRebalancePhaseFinalChampionSharpeCiAboveZero": all(
            math.isfinite(phase_final_champion_intervals[label][0])
            and phase_final_champion_intervals[label][0] > 0
            for label in ("rebalancePhase1bar", "rebalancePhase2bar")
        ),
    }
    ready_for_holdout = all(gates.values())

    holdout_start = int(registered_data["holdoutStartOpenTime"])
    holdout_end = int(registered_data["endOpenTime"])
    holdout_window = C._holdout_window(
        symbols, feed.CONTRACT_INTERVAL, holdout_start, holdout_end
    )
    holdout_identity = C._json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": registered_data["fullPanelDigestSha256"],
            "window": holdout_window,
        }
    )
    holdout_marker = HOLDOUT_REGISTRY_DIR / f"{holdout_identity}.json"
    output_holdout_record = output_dir / "final-holdout-opened.json"
    final_holdout: dict[str, object] = {
        "status": "reserved",
        "identitySha256": holdout_identity,
        "openRequested": bool(args.open_final_holdout),
        "executionStateAtStart": holdout_policy["executionStateAtStart"],
        "featureHistory": holdout_policy["featureHistory"],
        "chargeInitialCashToFrozenTargetTurnover": holdout_policy[
            "chargeInitialCashToFrozenTargetTurnover"
        ],
        "startOpenTime": holdout_start,
        "endOpenTime": holdout_end,
        "outcomeEndTimeExclusive": int(holdout_window["outcomeEndTimeExclusive"]),
        "rows": int(registered_data["holdoutReturnRows"]),
    }
    if args.open_final_holdout and not ready_for_holdout:
        final_holdout["openBlockedBy"] = [
            name for name, passed in gates.items() if not passed
        ]
    elif args.open_final_holdout:
        holdout_returns_path = output_dir / "final-holdout-returns.csv"
        holdout_result_path = output_dir / "final-holdout-result.json"
        opening_record = {
            "registryVersion": C.HOLDOUT_REGISTRY_VERSION,
            "status": "opening",
            "campaign": CAMPAIGN_ID,
            "registrationSha256": registration_sha,
            "campaignManifestSha256": campaign_manifest_sha,
            "holdoutIdentitySha256": holdout_identity,
            "candidate": champion,
            "window": holdout_window,
            "executionStateAtStart": holdout_policy["executionStateAtStart"],
            "featureHistory": holdout_policy["featureHistory"],
            "chargeInitialCashToFrozenTargetTurnover": holdout_policy[
                "chargeInitialCashToFrozenTargetTurnover"
            ],
            "artifacts": {
                "outputDirectory": str(output_dir.resolve()),
                "returns": str(holdout_returns_path.resolve()),
                "result": str(holdout_result_path.resolve()),
            },
        }
        _assert_inputs_unchanged(
            source_dir, registration, registration_sha, implementation_sha
        )
        # Reservation is intentionally irreversible and precedes all snapshot reads.
        C._reserve_holdout(
            HOLDOUT_REGISTRY_DIR,
            holdout_marker,
            holdout_window,
            output_holdout_record,
            opening_record,
        )
        full_panel, full_settlements, _ = _load_full_registered_inputs(
            snapshot_dir, registration
        )
        full_return_index = pd.Index(
            C._common_times(full_panel)[:-1], name="openTime"
        )
        holdout_index = full_return_index[
            (full_return_index >= holdout_start)
            & (full_return_index <= holdout_end)
        ]
        if len(holdout_index) != int(registered_data["holdoutReturnRows"]):
            raise ValueError("final holdout return rows changed from registration")
        holdout_fold = pd.DataFrame(
            [
                {
                    "outer_fold": 0,
                    "selected_candidate": champion,
                    "test_start": 0,
                    "test_stop": len(holdout_index),
                }
            ]
        )
        result_artifacts = dict(opening_record["artifacts"])
        try:
            evaluated_holdout = _stateful_outer_choices(
                full_panel,
                full_settlements,
                registration,
                config,
                specs,
                holdout_index,
                holdout_fold,
            ).drop(columns=["row_position", "outer_fold", "selected_candidate"])
        except R.PortfolioBankruptcyError as error:
            failure_open_time = (
                error.interval_left_close_time - feed.CONTRACT_INTERVAL_MS + 1
            )
            try:
                completed_rows = int(holdout_index.get_loc(failure_open_time))
            except (KeyError, TypeError, ValueError) as lookup_error:
                raise ValueError(
                    "holdout bankruptcy time is absent from the registered window"
                ) from lookup_error
            final_holdout = {
                **final_holdout,
                "status": "fail",
                "openRequested": True,
                "evaluationStatus": "portfolio_equity_exhausted",
                "successRuleEvaluated": False,
                "failure": {
                    **_portfolio_bankruptcy_failure(error),
                    "completedRowsBeforeFailure": completed_rows,
                },
            }
            result_artifacts["returnsWritten"] = False
        else:
            holdout_ci = C._bootstrap_ci(
                evaluated_holdout["net"],
                periods_per_year,
                feed.CONTRACT_INTERVAL_MS,
                bootstrap_reps,
                bootstrap_seed,
            )
            final_holdout = {
                "status": "pass"
                if math.isfinite(holdout_ci[0]) and holdout_ci[0] > 0
                else "fail",
                "openRequested": True,
                "identitySha256": holdout_identity,
                "executionStateAtStart": holdout_policy["executionStateAtStart"],
                "featureHistory": holdout_policy["featureHistory"],
                "chargeInitialCashToFrozenTargetTurnover": holdout_policy[
                    "chargeInitialCashToFrozenTargetTurnover"
                ],
                "startOpenTime": holdout_start,
                "endOpenTime": holdout_end,
                "outcomeEndTimeExclusive": int(
                    holdout_window["outcomeEndTimeExclusive"]
                ),
                "rows": len(evaluated_holdout),
                "metrics": C._metrics(
                    evaluated_holdout["net"],
                    periods_per_year,
                    evaluated_holdout["active"],
                ),
                "sharpeBootstrap95": C._ci_json(holdout_ci),
            }
            C._write_csv_atomic(evaluated_holdout, holdout_returns_path, index=False)
            returns_sha = C._file_digest(holdout_returns_path)
            final_holdout["evidence"] = {
                "returns": str(holdout_returns_path.resolve()),
                "returnsSha256": returns_sha,
            }
            result_artifacts.update(
                {"returnsSha256": returns_sha, "returnsWritten": True}
            )
        holdout_result_record = {
            **opening_record,
            "status": "evaluated",
            "result": final_holdout,
            "artifacts": result_artifacts,
        }
        C._write_json(holdout_result_path, holdout_result_record)
        holdout_completion_record = {
            **opening_record,
            "status": "completed",
            "result": final_holdout,
            "artifacts": {
                **holdout_result_record["artifacts"],
                "resultSha256": C._file_digest(holdout_result_path),
            },
        }
        _complete_holdout_records(
            holdout_marker,
            output_holdout_record,
            opening_record,
            holdout_completion_record,
        )

    _assert_inputs_unchanged(
        source_dir, registration, registration_sha, implementation_sha
    )
    summary = {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "campaignManifestSha256": campaign_manifest_sha,
        "status": C._campaign_status(ready_for_holdout, final_holdout),
        "bankruptcyFree": True,
        "statefulDevelopmentExecutionFree": all_stress_paths_evaluable,
        "symbols": symbols,
        "interval": feed.CONTRACT_INTERVAL,
        "configuration": {
            "strategy": dict(strategy),
            "nestedSizes": sizes,
            "labelHorizonBars": label_horizon,
            "eligibleTrials": eligible_names,
            "innerFoldPolicy": validation["innerFoldPolicy"],
            "outerFoldPolicy": validation["outerFoldPolicy"],
        },
        "data": {
            "registeredRows": int(registered_data["rows"]),
            "developmentRows": int(registered_data["developmentRows"]),
            "featureWarmupRows": warmup,
            "trialReturnRows": len(matrix),
            "diagnosticConfigurationCount": len(
                diagnostic_configuration_matrix.columns
            ),
            "sourceArtifacts": dict(source_evidence),
            "fullPanelDigestSha256": registered_data["fullPanelDigestSha256"],
            "snapshotManifestSha256": registered_data["snapshotManifestSha256"],
            "settlements": settlement_coverage,
            "survivorshipLimitation": universe["survivorshipLimitation"],
        },
        "trials": [spec.to_dict() for spec in specs],
        "champion": champion,
        "finalSelection": {
            "rule": validation["selectionRule"],
            "scores": C._json_records(final_selection_scores),
            "folds": C._json_records(final_selection_folds),
        },
        "championDevelopmentMetrics": trial_metrics[champion],
        "nestedOuterOos": {
            "metrics": nested_metrics,
            "activeFraction": active_fraction,
            "sharpeBootstrap95": C._ci_json(nested_ci),
            "foldMetrics": fold_metrics,
            "regimes": regime_report,
        },
        "selectionDiagnostics": selection_diagnostics,
        "lifetimeMultipleTesting": lifetime_multiple_testing,
        "matchedRebalanceComparison": paired_comparison,
        "championTurnover": turnover_ratio,
        "stress": stress_results,
        "promotionGates": gates,
        "finalHoldout": final_holdout,
    }

    C._write_csv_atomic(
        matrix, output_dir / "trial-returns.csv", index_label="openTime"
    )
    C._write_csv_atomic(
        diagnostic_configuration_matrix,
        output_dir / "diagnostic-configuration-returns.csv",
        index_label="openTime",
    )
    C._write_csv_atomic(
        diagnostic_matrix,
        output_dir / "diagnostic-trial-returns.csv",
        index_label="openTime",
    )
    C._write_csv_atomic(
        pbo_matrix, output_dir / "pbo-trial-returns.csv", index_label="openTime"
    )
    C._write_csv_atomic(
        labelled_nested_oos, output_dir / "nested-oos.csv", index=False
    )
    C._write_csv_atomic(
        nested.outer_folds, output_dir / "outer-folds.csv", index=False
    )
    C._write_csv_atomic(
        nested.inner_scores, output_dir / "inner-scores.csv", index=False
    )
    C._write_csv_atomic(
        final_selection_scores,
        output_dir / "final-selection-scores.csv",
        index=False,
    )
    C._write_csv_atomic(
        final_selection_folds,
        output_dir / "final-selection-folds.csv",
        index=False,
    )
    trial_paths = pd.concat(
        [
            frame.reindex(matrix.index).reset_index().assign(trial=name)
            for name, frame in details.items()
        ],
        ignore_index=True,
    )
    C._write_csv_atomic(trial_paths, output_dir / "trial-paths.csv", index=False)
    for label, path in stress_paths.items():
        C._write_csv_atomic(
            path, output_dir / f"stress-{label}-nested-oos.csv", index=False
        )
    C._write_json(
        output_dir / "trial-ledger.json",
        {
            "campaign": CAMPAIGN_ID,
            "trialCount": len(specs),
            "trials": [
                {
                    "specification": spec.to_dict(),
                    "championEligible": spec.trial_id in eligible_names,
                    "metrics": trial_metrics[spec.trial_id],
                    "finalSelectionScore": (
                        C._finite_number(
                            final_selection_scores.loc[
                                final_selection_scores["candidate"]
                                == spec.trial_id,
                                "score",
                            ].iloc[0]
                        )
                        if spec.trial_id in eligible_names
                        else None
                    ),
                }
                for spec in specs
            ],
        },
    )
    C._write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"historical reversal campaign failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
