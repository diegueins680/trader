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
  "historical reversal runner reuses sealed development inputs and preserves the holdout",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import copy
import inspect
import json
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import campaign_runner as common
import funding_campaign as funding
import historical_datafeed as feed
import reversal_campaign as reversal
import run_historical_reversal_campaign as runner

registration = runner._registration()
assert registration["campaign"] == runner.CAMPAIGN_ID
assert registration["registeredData"]["holdoutReturnRows"] == 1227
assert registration["validation"]["lifetimeTrialCount"] == 33
assert registration["strategy"]["rebalanceAnchorOpenTime"] == 1_600_819_200_000
assert runner.parse_args([]).source_campaign_dir == runner.SOURCE_CAMPAIGN_DIRECTORY
assert not hasattr(runner.parse_args([]), "acquire")

tampered_registration = copy.deepcopy(registration)
tampered_registration["validation"]["lifetimeTrialCount"] = 32
try:
    runner._validate_registration(tampered_registration)
except ValueError as error:
    assert "fixed constraints" in str(error)
else:
    raise AssertionError("a changed lifetime correction must fail closed")

# Candidate selection sees only the locked registered three-bar trials. One-bar
# trials remain present in the all-trial mapping for diagnostics and controls.
specs = reversal.campaign_specs(feed.CONTRACT_INTERVAL_MS)
all_candidates = {spec.trial_id: {"trial": spec.trial_id} for spec in specs}
eligible_names = runner._eligible_names(specs, 3)
eligible = runner._eligible_candidates(all_candidates, eligible_names)
assert tuple(eligible) == (
    "resrev_24h_rebalance_3bar",
    "resrev_72h_rebalance_3bar",
    "resrev_168h_rebalance_3bar",
)
assert len(all_candidates) == 6

# The turnover denominator and paired comparison are the selected champion's
# matched one-bar control on identical rows.
index = pd.Index(np.arange(90), name="openTime")
details = {
    spec.trial_id: pd.DataFrame(
        {"turnover": np.full(len(index), 0.25 if spec.rebalance_bars == 3 else 0.625)},
        index=index,
    )
    for spec in specs
}
champion = "resrev_72h_rebalance_3bar"
turnover = runner._champion_turnover_ratio(details, index, champion, specs)
assert turnover["matchedControl"] == "resrev_72h_rebalance_1bar"
assert abs(turnover["ratio"] - 0.4) < 1e-15

# Runner scoring preserves module-provided trade turnover rather than deriving
# a target-change proxy from the reported effective weights.
accounting = pd.DataFrame(
    {
        "gross": [0.01, 0.02],
        "turnover": [0.25, 0.0],
        "weight_AAA": [0.5, 0.5],
        "weight_BBB": [-0.5, -0.5],
    }
)
priced = runner._price_precomputed_turnover(accounting, 0.002)
np.testing.assert_allclose(priced["cost"], [0.0005, 0.0])

rng = np.random.default_rng(17)
matrix = pd.DataFrame(
    {spec.trial_id: rng.normal(0.0001, 0.002, len(index)) for spec in specs},
    index=index,
)
matrix[champion] = matrix["resrev_72h_rebalance_1bar"] + 0.0003
paired_registration = {
    "validation": {
        "pairedComparisonHypotheses": 3,
        "pairedComparisonFamilyWiseAlpha": 0.05,
    }
}
paired = runner._paired_rebalance_comparison(
    matrix,
    champion,
    specs,
    paired_registration,
    common._periods_per_year(feed.CONTRACT_INTERVAL_MS),
    100,
    9,
)
assert paired["champion"] == champion
champion_rows = [row for row in paired["horizons"] if row["selectedChampionPair"]]
assert len(champion_rows) == 1
assert champion_rows[0]["controlTrial"] == "resrev_72h_rebalance_1bar"

# A frozen-selection name change is executed as one continuous futures book.
# The off-schedule switch trades from A's drifted pretrade state, and the next
# row's weights include the actual switch cost in their equity denominator.
switch_open_times = (
    reversal.REBALANCE_ANCHOR_OPEN_TIME
    + np.arange(5, dtype=np.int64) * feed.CONTRACT_INTERVAL_MS
)
switch_closes = {
    "AAA": [100.0, 120.0, 120.0, 120.0, 120.0],
    "BBB": [100.0, 90.0, 90.0, 90.0, 90.0],
    "CCC": [100.0] * 5,
    "DDD": [100.0] * 5,
}
switch_panel = {
    symbol: pd.DataFrame(
        {
            "openTime": switch_open_times,
            "closeTime": switch_open_times + feed.CONTRACT_INTERVAL_MS - 1,
            "close": values,
        }
    )
    for symbol, values in switch_closes.items()
}
switch_settlements = [
    funding.FundingSettlement(
        symbol,
        int(switch_open_times[row] + 7),
        0.0,
        switch_closes[symbol][row],
    )
    for row in range(1, len(switch_open_times))
    for symbol in switch_closes
]
switch_config = reversal.ReversalCampaignConfig(
    interval_ms=feed.CONTRACT_INTERVAL_MS,
    rebalance_phase_bars=2,
    cost_per_turnover=0.01,
)
switch_index = pd.Index(switch_open_times[:3], name="openTime")
switch_folds = pd.DataFrame(
    [
        {
            "outer_fold": 0,
            "selected_candidate": "resrev_24h_rebalance_3bar",
            "test_start": 0,
            "test_stop": 1,
        },
        {
            "outer_fold": 1,
            "selected_candidate": "resrev_72h_rebalance_3bar",
            "test_start": 1,
            "test_stop": 3,
        },
    ]
)
original_weights_for_trial = runner.R.weights_for_trial

def controlled_targets(momentum, spec, _config):
    target = (
        np.array([0.5, -0.5, 0.0, 0.0])
        if spec.horizon_hours == 24
        else np.array([-0.5, 0.5, 0.0, 0.0])
    )
    return pd.DataFrame(
        np.tile(target, (len(momentum), 1)),
        index=momentum.index,
        columns=momentum.columns,
    )

runner.R.weights_for_trial = controlled_targets
try:
    switched = runner._stateful_outer_choices(
        switch_panel,
        switch_settlements,
        registration,
        switch_config,
        specs,
        switch_index,
        switch_folds,
    )
    partitioned_same_name = switch_folds.copy()
    partitioned_same_name["selected_candidate"] = "resrev_24h_rebalance_3bar"
    one_fold = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "selected_candidate": "resrev_24h_rebalance_3bar",
                "test_start": 0,
                "test_stop": 3,
            }
        ]
    )
    partitioned = runner._stateful_outer_choices(
        switch_panel,
        switch_settlements,
        registration,
        switch_config,
        specs,
        switch_index,
        partitioned_same_name,
    )
    unpartitioned = runner._stateful_outer_choices(
        switch_panel,
        switch_settlements,
        registration,
        switch_config,
        specs,
        switch_index,
        one_fold,
    )
finally:
    runner.R.weights_for_trial = original_weights_for_trial

pretrade = np.array([0.6 / 1.14, -0.45 / 1.14, 0.0, 0.0])
new_target = np.array([-0.5, 0.5, 0.0, 0.0])
expected_switch_turnover = np.abs(new_target - pretrade).sum()
assert abs(switched["turnover"].iloc[1] - expected_switch_turnover) < 1e-12
assert switched["turnover"].iloc[2] == 0.0
expected_post_cost_weights = new_target / (1.0 - 0.01 * expected_switch_turnover)
np.testing.assert_allclose(
    switched.filter(like="weight_").iloc[2],
    expected_post_cost_weights,
    rtol=0,
    atol=1e-12,
)
pd.testing.assert_frame_equal(
    partitioned.drop(columns=["outer_fold"]),
    unpartitioned.drop(columns=["outer_fold"]),
    check_exact=True,
)

# Build a small predecessor evidence bundle. Every source hash is checked
# before either CSV is parsed, and no raw-snapshot helper is involved.
with tempfile.TemporaryDirectory() as temporary:
    root = common.Path(temporary)
    source_dir = root / "source"
    source_dir.mkdir()
    interval = feed.CONTRACT_INTERVAL_MS
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    rows = 210
    times = (
        reversal.REBALANCE_ANCHOR_OPEN_TIME
        + np.arange(rows, dtype=np.int64) * interval
    )
    panel_rows = []
    settlement_rows = []
    for symbol_number, symbol in enumerate(symbols):
        for row, open_time in enumerate(times):
            panel_rows.append(
                {
                    "symbol": symbol,
                    "openTime": int(open_time),
                    "closeTime": int(open_time + interval - 1),
                    "close": 100.0 + symbol_number + row * 0.01,
                }
            )
            settlement_rows.append(
                {
                    "symbol": symbol,
                    "fundingTime": int(open_time + 7),
                    "fundingRate": 0.0001,
                    "resolvedMarkPrice": 100.0 + symbol_number,
                    "markSource": "event_mark_price",
                    "markOpenTime": np.nan,
                }
            )
    panel_path = source_dir / "registered-development-panel.csv"
    settlements_path = source_dir / "registered-development-settlements.csv"
    pd.DataFrame(panel_rows).to_csv(
        panel_path, index=False, lineterminator="\n", float_format="%.17g"
    )
    pd.DataFrame(settlement_rows).to_csv(
        settlements_path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="NA",
    )
    synthetic = copy.deepcopy(registration)
    synthetic["universe"]["symbols"] = symbols
    data = synthetic["registeredData"]
    data["startOpenTime"] = int(times[0])
    data["developmentRows"] = rows
    data["developmentCutoffOpenTime"] = int(times[-1])
    data["rows"] = 260
    data["holdoutBars"] = 50
    data["holdoutReturnRows"] = 49
    data["holdoutStartOpenTime"] = int(times[-1] + interval)
    data["endOpenTime"] = int(times[0] + 259 * interval)
    data["outcomeEndTimeExclusive"] = int(data["endOpenTime"] + interval)
    data["developmentPanelSha256"] = common._file_digest(panel_path)
    data["developmentSettlementsSha256"] = common._file_digest(settlements_path)
    data["fullPanelDigestSha256"] = "a" * 64
    data["fullSettlementsDigestSha256"] = "b" * 64
    source_manifest = {
        "campaign": data["sourceCampaign"],
        "registrationSha256": data["sourceRegistrationSha256"],
        "snapshotManifestSha256": data["snapshotManifestSha256"],
        "symbols": symbols,
        "registeredData": {
            "panelSha256": data["fullPanelDigestSha256"],
            "settlementsSha256": data["fullSettlementsDigestSha256"],
            "developmentRows": rows,
            "developmentCutoffOpenTime": int(times[-1]),
        },
        "artifacts": {
            "registeredDevelopmentPanel": data["developmentPanel"],
            "registeredPanelSha256": data["developmentPanelSha256"],
            "registeredDevelopmentSettlements": data["developmentSettlements"],
            "registeredSettlementsSha256": data["developmentSettlementsSha256"],
        },
    }
    manifest_path = source_dir / "campaign-manifest.json"
    common._write_json(manifest_path, source_manifest)
    data["sourceCampaignManifestSha256"] = common._file_digest(manifest_path)

    original_snapshot_loader = runner.H._load_snapshot
    runner.H._load_snapshot = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("development loading must not read the raw snapshot")
    )
    try:
        loaded_panel, settlements, audit, coverage, evidence = (
            runner._load_development_inputs(source_dir, synthetic)
        )
    finally:
        runner.H._load_snapshot = original_snapshot_loader
    assert set(loaded_panel) == set(symbols)
    assert all(len(frame) == rows for frame in loaded_panel.values())
    assert len(settlements) == rows * len(symbols)
    assert len(audit) == len(settlements)
    assert coverage["resolvedFraction"] == 1.0
    assert evidence["developmentPanelSha256"] == data["developmentPanelSha256"]

    # Parsing consumes the exact bytes that passed the digest check rather
    # than reopening a path that could have been replaced in between.
    pinned_panel, _ = runner._read_pinned_bytes(
        panel_path, data["developmentPanelSha256"]
    )
    original_panel_bytes = panel_path.read_bytes()
    try:
        panel_path.write_bytes(b"replaced after hashing\n")
        pinned_loaded_panel = runner._panel_from_csv(pinned_panel, synthetic)
    finally:
        panel_path.write_bytes(original_panel_bytes)
    assert all(len(frame) == rows for frame in pinned_loaded_panel.values())

    config = runner._strategy_config(synthetic)
    primary_raw, detail_raw, returned_specs = runner._trials_on_panel(
        loaded_panel, settlements, synthetic, config
    )
    evaluation_index = primary_raw.index[21:]
    primary, _ = runner._reprice_details(
        detail_raw, evaluation_index, config.cost_per_turnover
    )
    phase = runner._phase_configuration_matrix(
        loaded_panel,
        settlements,
        synthetic,
        evaluation_index,
        runner._eligible_names(returned_specs, 3),
    )
    assert primary.shape == (188, 6)
    assert phase.shape == (188, 6)
    assert pd.concat([primary, phase], axis=1).shape == (188, 12)
    cash_start_index = evaluation_index[:20]
    cash_start_fold = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "selected_candidate": "resrev_24h_rebalance_3bar",
                "test_start": 0,
                "test_stop": len(cash_start_index),
            }
        ]
    )
    original_residual_momentum = runner.H._residual_momentum

    def fixed_residual_momentum(close, _lookback, horizons):
        values = np.tile([-0.02, 0.02, -0.01, 0.01], (len(close), 1))
        return {
            horizon: pd.DataFrame(values, index=close.index, columns=close.columns)
            for horizon in horizons
        }

    runner.H._residual_momentum = fixed_residual_momentum
    try:
        cash_start = runner._stateful_outer_choices(
            loaded_panel,
            settlements,
            synthetic,
            config,
            returned_specs,
            cash_start_index,
            cash_start_fold,
        )
    finally:
        runner.H._residual_momentum = original_residual_momentum
    first_weights = cash_start.filter(like="weight_").iloc[0].to_numpy()
    assert np.abs(first_weights).sum() > 0
    assert abs(cash_start["turnover"].iloc[0] - np.abs(first_weights).sum()) < 1e-15
    assert len(cash_start) == len(cash_start_index)

    original_stateful_outer_choices = runner._stateful_outer_choices

    def bankrupt_stress(*_args, **_kwargs):
        raise reversal.PortfolioBankruptcyError(
            int(cash_start_index[0] + interval - 1),
            "resrev_24h_rebalance_3bar",
        )

    runner._stateful_outer_choices = bankrupt_stress
    try:
        stress_failure, stress_path, stress_interval = runner._stress_campaign(
            "cost1_5x",
            loaded_panel,
            settlements,
            synthetic,
            evaluation_index,
            cash_start_fold,
            runner._eligible_names(returned_specs, 3),
            common._periods_per_year(interval),
            100,
            7,
        )
    finally:
        runner._stateful_outer_choices = original_stateful_outer_choices
    assert stress_failure["status"] == "execution_failed"
    assert stress_failure["failure"]["path"] == "nested_outer_oos_stress"
    assert stress_path is None
    assert all(np.isnan(value) for value in stress_interval)

    synthetic["validation"].update(
        {
            "bootstrapReplications": 100,
            "bootstrapSeed": 7,
            "developmentEvaluationRows": 188,
            "innerInitialTrain": 40,
            "innerTestSize": 15,
            "outerInitialTrain": 80,
            "outerFoldCount": 1,
            "outerTestSize": 107,
            "pboSlices": 2,
        }
    )
    synthetic["promotion"].update(
        {
            "maximumRegimeLoss": 0.99,
            "maximumWorstFoldLoss": 0.99,
            "minimumActiveFraction": 0.0,
            "minimumOuterOosObservations": 1,
            "minimumRegimeObservations": 1,
            "minimumSymbols": 5,
        }
    )
    original_registration_and_sha = runner._registration_and_sha
    original_registration_sha = runner._registration_sha
    original_full_loader = runner._load_full_registered_inputs
    original_raw_snapshot_loader = runner.H._load_snapshot
    original_registry = runner.HOLDOUT_REGISTRY_DIR
    full_load_calls = []
    runner._registration_and_sha = lambda: (synthetic, "synthetic-registration")
    runner._registration_sha = lambda *_args, **_kwargs: "synthetic-registration"
    runner._load_full_registered_inputs = lambda *_args, **_kwargs: (
        full_load_calls.append(True),
        (_ for _ in ()).throw(
            AssertionError("a blocked campaign must not load the full snapshot")
        ),
    )[1]
    runner.H._load_snapshot = lambda *_args, **_kwargs: (
        full_load_calls.append(True),
        (_ for _ in ()).throw(
            AssertionError("development must not call the raw snapshot loader")
        ),
    )[1]
    runner.HOLDOUT_REGISTRY_DIR = root / "registry"
    try:
        default_summary = runner.run(
            runner.argparse.Namespace(
                source_campaign_dir=str(source_dir),
                snapshot_dir=str(root / "absent-snapshot"),
                output_dir=str(root / "default-output"),
                open_final_holdout=False,
            )
        )
        blocked_summary = runner.run(
            runner.argparse.Namespace(
                source_campaign_dir=str(source_dir),
                snapshot_dir=str(root / "absent-snapshot"),
                output_dir=str(root / "blocked-output"),
                open_final_holdout=True,
            )
        )
        original_trials_on_panel = runner._trials_on_panel
        original_registry_assert = runner.C._assert_output_holdout_not_consumed
        original_reserve = runner.C._reserve_holdout
        registry_calls = []

        def bankrupt_primary(*_args, **_kwargs):
            raise reversal.PortfolioBankruptcyError(
                1_611_907_199_999, "resrev_24h_rebalance_3bar"
            )

        runner._trials_on_panel = bankrupt_primary
        runner.C._assert_output_holdout_not_consumed = (
            lambda *_args, **_kwargs: registry_calls.append("assert")
        )
        runner.C._reserve_holdout = (
            lambda *_args, **_kwargs: registry_calls.append("reserve")
        )
        try:
            for open_requested in (False, True):
                mechanical_output = root / f"mechanical-{open_requested}"
                mechanical_args = runner.argparse.Namespace(
                    source_campaign_dir=str(source_dir),
                    snapshot_dir=str(root / "absent-snapshot"),
                    output_dir=str(mechanical_output),
                    open_final_holdout=open_requested,
                )
                mechanical = runner.run(mechanical_args)
                summary_path = mechanical_output / "summary.json"
                failure_path = mechanical_output / "mechanical-failure.json"
                first_summary_sha = common._file_digest(summary_path)
                first_failure_sha = common._file_digest(failure_path)
                repeated = runner.run(mechanical_args)
                assert repeated == mechanical
                assert common._file_digest(summary_path) == first_summary_sha
                assert common._file_digest(failure_path) == first_failure_sha
                assert mechanical["status"] == "mechanically_invalid"
                assert mechanical["bankruptcyFree"] is False
                assert mechanical["promotionGates"] == {"bankruptcyFree": False}
                assert mechanical["mechanicalFailure"] == {
                    "reason": "portfolio_equity_exhausted",
                    "trialId": "resrev_24h_rebalance_3bar",
                    "closeTime": 1_611_907_199_999,
                    "closeTimeSemantics": "interval_left_close",
                    "outcomeCloseTime": 1_611_935_999_999,
                }
                assert mechanical["finalHoldout"]["status"] == "reserved"
                assert mechanical["finalHoldout"]["openRequested"] is open_requested
                assert mechanical["finalHoldout"]["openBlockedBy"] == [
                    "bankruptcyFree"
                ]
                assert summary_path.is_file()
                assert failure_path.is_file()
                persisted = json.loads(summary_path.read_text())
                assert persisted == mechanical
                assert common._file_digest(failure_path) == mechanical["evidence"][
                    "mechanicalFailureSha256"
                ]
                assert {path.name for path in mechanical_output.iterdir()} == {
                    ".campaign.lock",
                    "campaign-manifest.json",
                    "mechanical-failure.json",
                    "summary.json",
                }
                campaign_manifest_path = mechanical_output / "campaign-manifest.json"
                campaign_manifest_path.write_bytes(
                    campaign_manifest_path.read_bytes() + b" "
                )
                try:
                    runner.run(mechanical_args)
                except ValueError as error:
                    assert "manifest bytes changed" in str(error)
                else:
                    raise AssertionError("manifest byte mutation must fail closed")
        finally:
            runner._trials_on_panel = original_trials_on_panel
            runner.C._assert_output_holdout_not_consumed = original_registry_assert
            runner.C._reserve_holdout = original_reserve

        original_stateful_outer_choices = runner._stateful_outer_choices

        def bankrupt_stateful_development(
            _panel,
            _settlements,
            _registration,
            _config,
            _specs,
            matrix_index,
            outer_folds,
        ):
            selected = str(outer_folds.iloc[0]["selected_candidate"])
            raise reversal.PortfolioBankruptcyError(
                int(matrix_index[int(outer_folds.iloc[0]["test_start"])] + interval - 1),
                selected,
            )

        runner._stateful_outer_choices = bankrupt_stateful_development
        try:
            stateful_failure_output = root / "stateful-failure-output"
            stateful_failure = runner.run(
                runner.argparse.Namespace(
                    source_campaign_dir=str(source_dir),
                    snapshot_dir=str(root / "absent-snapshot"),
                    output_dir=str(stateful_failure_output),
                    open_final_holdout=True,
                )
            )
        finally:
            runner._stateful_outer_choices = original_stateful_outer_choices
        assert stateful_failure["status"] == "insufficient_evidence"
        assert stateful_failure["bankruptcyFree"] is True
        assert stateful_failure["statefulDevelopmentExecutionFree"] is False
        assert stateful_failure["developmentExecutionFailure"]["path"] == (
            "nested_outer_oos"
        )
        assert stateful_failure["finalHoldout"]["status"] == "reserved"
        assert stateful_failure["finalHoldout"]["openBlockedBy"] == [
            "statefulDevelopmentExecution"
        ]
        assert not (stateful_failure_output / "final-holdout-opened.json").exists()
        assert registry_calls == []
        assert not (root / "registry").exists()
    finally:
        runner._registration_and_sha = original_registration_and_sha
        runner._registration_sha = original_registration_sha
        runner._load_full_registered_inputs = original_full_loader
        runner.H._load_snapshot = original_raw_snapshot_loader
        runner.HOLDOUT_REGISTRY_DIR = original_registry
    assert full_load_calls == []
    assert default_summary["data"]["diagnosticConfigurationCount"] == 12
    assert default_summary["champion"].endswith("_3bar")
    assert "allRebalancePhaseFinalChampionSharpeCiAboveZero" in default_summary[
        "promotionGates"
    ]
    for label in ("rebalancePhase1bar", "rebalancePhase2bar"):
        assert "frozenFinalChampionSelectionOos" in default_summary["stress"][label]
    assert default_summary["finalHoldout"]["status"] == "reserved"
    assert blocked_summary["finalHoldout"]["status"] == "reserved"
    assert "symbolCount" in blocked_summary["finalHoldout"]["openBlockedBy"]
    assert not (root / "default-output" / "final-holdout-returns.csv").exists()
    assert not (root / "blocked-output" / "final-holdout-opened.json").exists()

    # An all-gates pass reserves before loading raw inputs, starts the holdout
    # from cash, completes both registry records, and rejects every retry.
    success_registration = copy.deepcopy(synthetic)
    success_registration["promotion"].update(
        {
            "minimumSymbols": len(symbols),
            "currentCampaignDeflatedSharpeProbabilityMinimum": 0.0,
            "lifetimeBonferroniPsrProbabilityMinimum": 0.0,
            "maximumPbo": 1.0,
            "maximumChampionTurnoverRatio": 2.0,
        }
    )
    full_rows = int(success_registration["registeredData"]["rows"])
    full_times = (
        reversal.REBALANCE_ANCHOR_OPEN_TIME
        + np.arange(full_rows, dtype=np.int64) * interval
    )
    full_panel = {}
    full_settlements = []
    for symbol_number, symbol in enumerate(symbols):
        closes = 100.0 + symbol_number + np.arange(full_rows) * 0.01
        full_panel[symbol] = pd.DataFrame(
            {
                "openTime": full_times,
                "closeTime": full_times + interval - 1,
                "close": closes,
            }
        )
        full_settlements.extend(
            funding.FundingSettlement(
                symbol,
                int(full_times[row] + 7),
                0.0001,
                float(closes[row]),
            )
            for row in range(full_rows)
        )

    saved_registration_and_sha = runner._registration_and_sha
    saved_registration_sha = runner._registration_sha
    saved_full_loader = runner._load_full_registered_inputs
    saved_snapshot_loader = runner.H._load_snapshot
    saved_registry = runner.HOLDOUT_REGISTRY_DIR
    saved_bootstrap = runner.C._bootstrap_ci
    saved_market_regime_labels = runner.C._market_regime_labels
    saved_residual_momentum = runner.H._residual_momentum
    saved_stateful_outer_choices = runner._stateful_outer_choices

    def reservation_checking_loader(output, observations):
        def load(_snapshot_dir, loaded_registration):
            assert loaded_registration == success_registration
            markers = list(runner.HOLDOUT_REGISTRY_DIR.glob("*.json"))
            assert len(markers) == 1
            marker = json.loads(markers[0].read_text())
            local_path = output / "final-holdout-opened.json"
            assert local_path.is_file()
            local = json.loads(local_path.read_text())
            assert marker == local
            assert marker["status"] == "opening"
            assert marker["artifacts"]["outputDirectory"] == str(output.resolve())
            assert not (output / "final-holdout-returns.csv").exists()
            assert not (output / "final-holdout-result.json").exists()
            observations.append(marker)
            return full_panel, full_settlements, {"resolvedFraction": 1.0}

        return load

    runner._registration_and_sha = lambda: (
        success_registration,
        "synthetic-registration",
    )
    runner._registration_sha = lambda *_args, **_kwargs: "synthetic-registration"
    runner.H._load_snapshot = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("the patched full-input boundary must be used")
    )
    runner.C._bootstrap_ci = lambda *_args, **_kwargs: (1.0, 2.0)

    def alternating_regime_labels(panel, _interval_ms):
        grid = pd.Index(common._common_times(panel), name="openTime")
        return pd.Series(
            np.where(np.arange(len(grid)) % 2 == 0, "regime_a", "regime_b"),
            index=grid,
        )

    runner.C._market_regime_labels = alternating_regime_labels
    runner.H._residual_momentum = fixed_residual_momentum
    try:
        success_output = root / "success-output"
        success_args = runner.argparse.Namespace(
            source_campaign_dir=str(source_dir),
            snapshot_dir=str(root / "synthetic-snapshot"),
            output_dir=str(success_output),
            open_final_holdout=True,
        )
        success_openings = []
        runner.HOLDOUT_REGISTRY_DIR = root / "success-registry"
        runner._load_full_registered_inputs = reservation_checking_loader(
            success_output, success_openings
        )
        success = runner.run(success_args)
        assert len(success_openings) == 1, success["promotionGates"]
        assert all(success["promotionGates"].values())
        assert success["status"] == "final_holdout_passed"
        assert success["finalHoldout"]["status"] == "pass"

        returns_path = success_output / "final-holdout-returns.csv"
        result_path = success_output / "final-holdout-result.json"
        local_path = success_output / "final-holdout-opened.json"
        marker_path = next((root / "success-registry").glob("*.json"))
        returns = pd.read_csv(returns_path)
        result_record = json.loads(result_path.read_text())
        local_record = json.loads(local_path.read_text())
        marker_record = json.loads(marker_path.read_text())
        assert result_record["status"] == "evaluated"
        assert result_record["artifacts"]["returnsWritten"] is True
        assert local_record["status"] == "completed"
        assert marker_record == local_record
        assert local_record["result"] == success["finalHoldout"]
        assert common._file_digest(returns_path) == result_record["artifacts"][
            "returnsSha256"
        ]
        assert common._file_digest(result_path) == local_record["artifacts"][
            "resultSha256"
        ]
        weight_columns = [
            column for column in returns if column.startswith("weight_")
        ]
        entry_weight = float(returns.loc[0, weight_columns].abs().sum())
        assert entry_weight > 0.0
        assert abs(float(returns.loc[0, "turnover"]) - entry_weight) < 1e-12
        expected_cost = (
            success_registration["strategy"]["costBpsPerUnitTurnover"]
            / 10_000
            * entry_weight
        )
        assert abs(float(returns.loc[0, "cost"]) - expected_cost) < 1e-12

        protected = {
            path: path.read_bytes()
            for path in (
                returns_path,
                result_path,
                local_path,
                marker_path,
                success_output / "summary.json",
            )
        }
        retry_args = runner.argparse.Namespace(
            source_campaign_dir=str(source_dir),
            snapshot_dir=str(root / "synthetic-snapshot"),
            output_dir=str(success_output),
            open_final_holdout=False,
        )
        try:
            runner.run(retry_args)
        except ValueError as error:
            assert "already consumed" in str(error)
        else:
            raise AssertionError("completed holdout evidence must reject retries")
        assert len(success_openings) == 1
        for path, before in protected.items():
            assert path.read_bytes() == before

        # Bankruptcy after reservation is itself a completed one-shot result;
        # no partial or synthetic return series is written.
        failure_output = root / "holdout-failure-output"
        failure_args = runner.argparse.Namespace(
            source_campaign_dir=str(source_dir),
            snapshot_dir=str(root / "synthetic-snapshot"),
            output_dir=str(failure_output),
            open_final_holdout=True,
        )
        failure_openings = []
        runner.HOLDOUT_REGISTRY_DIR = root / "holdout-failure-registry"
        runner._load_full_registered_inputs = reservation_checking_loader(
            failure_output, failure_openings
        )

        def fail_only_on_holdout(
            panel,
            settlements,
            loaded_registration,
            config,
            trial_specs,
            matrix_index,
            outer_folds,
        ):
            if int(matrix_index[0]) >= int(
                loaded_registration["registeredData"]["holdoutStartOpenTime"]
            ):
                selected = str(outer_folds.iloc[0]["selected_candidate"])
                raise reversal.PortfolioBankruptcyError(
                    int(matrix_index[3] + interval - 1), selected
                )
            return saved_stateful_outer_choices(
                panel,
                settlements,
                loaded_registration,
                config,
                trial_specs,
                matrix_index,
                outer_folds,
            )

        runner._stateful_outer_choices = fail_only_on_holdout
        holdout_failure = runner.run(failure_args)
        assert len(failure_openings) == 1
        assert all(holdout_failure["promotionGates"].values())
        assert holdout_failure["status"] == "final_holdout_failed"
        assert holdout_failure["finalHoldout"]["status"] == "fail"
        assert holdout_failure["finalHoldout"]["successRuleEvaluated"] is False
        assert holdout_failure["finalHoldout"]["failure"][
            "completedRowsBeforeFailure"
        ] == 3
        assert not (failure_output / "final-holdout-returns.csv").exists()
        failure_result_path = failure_output / "final-holdout-result.json"
        failure_local_path = failure_output / "final-holdout-opened.json"
        failure_marker_path = next(
            (root / "holdout-failure-registry").glob("*.json")
        )
        failure_result = json.loads(failure_result_path.read_text())
        failure_local = json.loads(failure_local_path.read_text())
        failure_marker = json.loads(failure_marker_path.read_text())
        assert failure_result["status"] == "evaluated"
        assert failure_result["artifacts"]["returnsWritten"] is False
        assert failure_local["status"] == "completed"
        assert failure_marker == failure_local
        try:
            runner.run(failure_args)
        except ValueError as error:
            assert "already consumed" in str(error)
        else:
            raise AssertionError("failed holdout evidence must reject retries")
    finally:
        runner._registration_and_sha = saved_registration_and_sha
        runner._registration_sha = saved_registration_sha
        runner._load_full_registered_inputs = saved_full_loader
        runner.H._load_snapshot = saved_snapshot_loader
        runner.HOLDOUT_REGISTRY_DIR = saved_registry
        runner.C._bootstrap_ci = saved_bootstrap
        runner.C._market_regime_labels = saved_market_regime_labels
        runner.H._residual_momentum = saved_residual_momentum
        runner._stateful_outer_choices = saved_stateful_outer_choices

    # Tampering is rejected at the byte-hash boundary before pandas sees data.
    panel_path.write_text(panel_path.read_text() + "\n")
    original_read_csv = runner.pd.read_csv
    runner.pd.read_csv = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("a hash-invalid CSV must not be parsed")
    )
    try:
        try:
            runner._load_development_inputs(source_dir, synthetic)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("a changed development artifact must fail closed")
    finally:
        runner.pd.read_csv = original_read_csv

# The only full snapshot call is below the explicit all-gates branch and the
# irreversible overlap-aware registry reservation.
run_source = inspect.getsource(runner._run_locked)
blocked_branch = run_source.index(
    "if args.open_final_holdout and not ready_for_holdout"
)
open_branch = run_source.index("elif args.open_final_holdout")
reserve_call = run_source.index("C._reserve_holdout", open_branch)
full_load = run_source.index("_load_full_registered_inputs", reserve_call)
last_input_check = run_source.rfind(
    "_assert_inputs_unchanged", open_branch, reserve_call
)
single_trial = run_source.index("_stateful_outer_choices", full_load)
holdout_bootstrap = run_source.index("holdout_ci =", single_trial)
holdout_section = run_source[single_trial:holdout_bootstrap]
assert blocked_branch < open_branch < last_input_check < reserve_call < full_load
assert full_load < single_trial < holdout_bootstrap
assert "_trials_on_panel" not in holdout_section
assert "_reprice_path" not in holdout_section
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
    });
    assert.equal(run.status, 0, run.stderr);
  },
);
