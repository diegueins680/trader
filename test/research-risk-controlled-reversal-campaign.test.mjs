import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

function runResearchPython(program) {
  const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
    encoding: "utf8",
  });
  assert.equal(run.status, 0, run.stderr);
}

test(
  "risk-controlled reversal ledger, rank hysteresis, ties, and delays stay locked",
  { skip: !hasResearchPython },
  () => {
    runResearchPython(String.raw`
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import harness
import risk_controlled_reversal_campaign as R

INTERVAL = 8 * 3_600_000
config = R.RiskControlledReversalConfig(interval_ms=INTERVAL)
specs = R.campaign_specs(INTERVAL)
expected_ids = (
    "resrev_24h_exit1_control",
    "resrev_24h_exit3_hysteresis",
    "resrev_72h_exit1_control",
    "resrev_72h_exit3_hysteresis",
    "resrev_168h_exit1_control",
    "resrev_168h_exit3_hysteresis",
)
assert R.HORIZON_HOURS == (24, 72, 168)
assert R.EXIT_RANKS == (1, 3)
assert R.TRIAL_IDS == expected_ids
assert tuple(spec.trial_id for spec in specs) == expected_ids
assert tuple(spec.horizon_bars for spec in specs) == (3, 3, 9, 9, 21, 21)
assert tuple(spec.exit_rank for spec in specs) == (1, 3, 1, 3, 1, 3)
assert tuple(spec.champion_eligible for spec in specs) == (
    False, True, False, True, False, True
)
assert specs[0].to_dict() == {
    "id": "resrev_24h_exit1_control",
    "horizonBars": 3,
    "horizonHours": 24,
    "exitRank": 1,
    "role": "matched_control",
    "championEligible": False,
}
assert specs[1].to_dict() == {
    "id": "resrev_24h_exit3_hysteresis",
    "horizonBars": 3,
    "horizonHours": 24,
    "exitRank": 3,
    "role": "rank_hysteresis_treatment",
    "championEligible": True,
}
assert R.circular_block_bootstrap_sharpe_ci is harness.circular_block_bootstrap_sharpe_ci

times = pd.Index(
    R.REBALANCE_ANCHOR_OPEN_TIME
    + np.arange(6, dtype=np.int64) * INTERVAL
    + INTERVAL
    - 1,
    name="closeTime",
)
columns = ["A", "B", "C", "D", "E", "F", "G", "H"]
momentum = pd.DataFrame(
    [
        [-4.0, -3.0, -2.0, -1.0, 4.0, 3.0, 2.0, 1.0],
        [-2.0, -4.0, -3.0, -1.0, 2.0, 4.0, 3.0, 1.0],
        [-1.0, -4.0, -3.0, -2.0, 1.0, 4.0, 3.0, 2.0],
        [-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    ],
    index=times,
    columns=columns,
)
control = R.decision_weights_for_trial(momentum, specs[0], config)
treatment = R.decision_weights_for_trial(momentum, specs[1], config)

# Entry is rank one in stable column order for both paths.
np.testing.assert_allclose(control.iloc[0], [0.25, 0, 0, 0, -0.25, 0, 0, 0])
np.testing.assert_allclose(treatment.iloc[0], control.iloc[0])
# The treatment retains incumbents at rank three; the control immediately rotates.
np.testing.assert_allclose(treatment.iloc[1], treatment.iloc[0])
np.testing.assert_allclose(control.iloc[1], [0, 0.25, 0, 0, 0, -0.25, 0, 0])
# Rank four breaches hysteresis and replaces each side with the new rank one.
np.testing.assert_allclose(treatment.iloc[2], [0, 0.25, 0, 0, 0, -0.25, 0, 0])
# Exact ties honor stable column order, while a rank-two incumbent is retained.
np.testing.assert_allclose(treatment.iloc[3], treatment.iloc[2])
np.testing.assert_allclose(control.iloc[3], [0.25, 0, 0, 0, -0.25, 0, 0, 0])
# Losing either signed side flattens both and clears incumbents before re-entry.
np.testing.assert_allclose(treatment.iloc[4], 0.0)
np.testing.assert_allclose(treatment.iloc[5], [0.25, 0, 0, 0, -0.25, 0, 0, 0])

scores = R.scores_for_trial(momentum, specs[1])
np.testing.assert_allclose(scores.iloc[0], -momentum.iloc[0])
delay_one = R.weights_for_trial(momentum, specs[1], config)
delay_two = R.weights_for_trial(
    momentum, specs[1], replace(config, signal_delay_bars=2)
)
np.testing.assert_allclose(delay_one.iloc[0], 0.0)
np.testing.assert_allclose(delay_one.iloc[1:], treatment.iloc[:-1])
np.testing.assert_allclose(delay_two.iloc[:2], 0.0)
np.testing.assert_allclose(delay_two.iloc[2:], treatment.iloc[:-2])

for invalid_delay in (0, 3, True):
    try:
        R.weights_for_trial(
            momentum, specs[1], replace(config, signal_delay_bars=invalid_delay)
        )
    except ValueError as error:
        assert "signal_delay_bars" in str(error)
    else:
        raise AssertionError("unregistered signal delay was accepted")
for invalid_cost in (0.0005, 0.003):
    try:
        R.weights_for_trial(
            momentum, specs[1], replace(config, cost_per_turnover=invalid_cost)
        )
    except ValueError as error:
        assert "cost_per_turnover" in str(error)
    else:
        raise AssertionError("unregistered turnover cost was accepted")

assert config.gross_exposure == 0.5
assert config.registered_gross_exposure == 0.5
v2_config = replace(
    config,
    gross_exposure=0.25,
    registered_gross_exposure=0.25,
)
v2_decisions = R.decision_weights_for_trial(momentum, specs[0], v2_config)
np.testing.assert_allclose(
    v2_decisions.iloc[0], [0.125, 0, 0, 0, -0.125, 0, 0, 0]
)
for invalid_exposure in (
    replace(config, gross_exposure=0.25),
    replace(config, registered_gross_exposure=0.25),
    replace(config, gross_exposure=0.3, registered_gross_exposure=0.3),
):
    try:
        R.weights_for_trial(momentum, specs[1], invalid_exposure)
    except ValueError as error:
        assert "gross_exposure" in str(error)
    else:
        raise AssertionError("unregistered or mismatched gross exposure was accepted")
`);
  },
);

test(
  "drifted accounting, terminal costs, shock boundaries, and typed breaches reconcile",
  { skip: !hasResearchPython },
  () => {
    runResearchPython(String.raw`
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import risk_controlled_reversal_campaign as R

INTERVAL = 8 * 3_600_000
config = R.RiskControlledReversalConfig(interval_ms=INTERVAL)
times = pd.Index(
    R.REBALANCE_ANCHOR_OPEN_TIME
    + np.arange(4, dtype=np.int64) * INTERVAL
    + INTERVAL
    - 1,
    name="closeTime",
)
close = pd.DataFrame(
    {
        "AAA": [100.0, 110.0, 99.0, 108.9],
        "BBB": [100.0, 90.0, 99.0, 89.1],
    },
    index=times,
)
targets = pd.DataFrame(
    [[0.25, -0.25]] * len(times), index=times, columns=close.columns
)
settlements = [
    # Events on the left edge are excluded from (left, right].
    R.FundingSettlement("AAA", int(times[0]), 0.90, 100.0),
    R.FundingSettlement("BBB", int(times[0]), 0.90, 100.0),
    # Events exactly on the right edge are included.
    R.FundingSettlement("AAA", int(times[1]), 0.01, 110.0),
    R.FundingSettlement("BBB", int(times[1]), 0.02, 90.0),
    R.FundingSettlement("AAA", int(times[1]) + 1, 0.03, 99.0),
    R.FundingSettlement("BBB", int(times[1]) + 1, 0.04, 99.0),
]
detail = R.evaluate_drifted_intervals(
    close,
    targets,
    np.array([True, True, True, False], dtype=bool),
    settlements,
    config,
    trial_id="accounting_test",
)
assert list(detail.index) == list(times[:-1])
np.testing.assert_allclose(detail[["weight_AAA", "weight_BBB"]], targets.iloc[:-1])
np.testing.assert_allclose(detail["priceGross"], [0.05, -0.05, 0.05])
np.testing.assert_allclose(detail["fundingCashflow"], [0.00175, 0.00425, 0.0])
np.testing.assert_allclose(
    detail["gross"], detail["priceGross"] + detail["fundingCashflow"]
)
assert abs(detail["activationTurnover"].iloc[0] - 0.5) < 1e-15

first_factor = 1.0 + detail["net"].iloc[0]
first_endpoint = np.array([0.25 * 1.1, -0.25 * 0.9]) / first_factor
expected_second_activation = float(
    np.abs(np.array([0.25, -0.25]) - first_endpoint).sum()
)
assert abs(
    detail["activationTurnover"].iloc[1] - expected_second_activation
) < 1e-15

# Raw terminal turnover is normalized by endpoint equity for the risk cap.
# Charged turnover is normalized by interval-start equity for return accounting.
terminal = detail.iloc[-1]
preterminal_factor = (
    1.0
    + terminal["gross"]
    - config.cost_per_turnover * terminal["activationTurnover"]
)
assert abs(
    terminal["terminalChargedTurnover"]
    - preterminal_factor * terminal["terminalTurnover"]
) < 1e-15
assert abs(terminal["terminalChargedTurnover"] - 0.5) < 1e-15
assert abs(
    terminal["terminalChargedTurnover"] - terminal["terminalTurnover"]
) > 1e-3
np.testing.assert_allclose(
    detail["turnover"],
    detail["activationTurnover"] + detail["terminalChargedTurnover"],
)
np.testing.assert_allclose(
    detail["cost"], config.cost_per_turnover * detail["turnover"],
    rtol=0.0,
    atol=2e-16,
)
np.testing.assert_allclose(
    detail["net"], detail["gross"] - detail["cost"],
    rtol=0.0,
    atol=2e-16,
)
np.testing.assert_allclose(detail["equity"], np.cumprod(1.0 + detail["net"]))

# The shock diagnostic uses raw equity and independently compares it to 0.22G.
coverage_boundary_weights = np.array([-40.0 / 21.0])
shock_equity, shocked_gross, coverage = R._shock_state(
    coverage_boundary_weights, config
)
assert abs(shock_equity - 11.0 / 21.0) < 1e-15
assert abs(shocked_gross - 50.0 / 21.0) < 1e-15
assert abs(shock_equity - 0.22 * shocked_gross) < 1e-15
assert abs(coverage - 2.0) < 1e-15
assert R._shock_state(
    np.array([-np.nextafter(40.0 / 21.0, np.inf)]), config
)[2] < 2.0
assert R._shock_state(np.array([2.0]), config)[0] == 0.5
assert R._shock_state(
    np.array([np.nextafter(2.0, np.inf)]), config
)[0] < 0.5

# Every registered risk boundary is inclusive; one ULP beyond it fails typed.
boundary_weights = np.array([0.4, -0.4, 0.1, -0.1])
boundary_state = {
    "equity": 0.75,
    "drawdown": 0.20,
    "endpointGrossLeverage": 1.0,
    "maximumAbsoluteSymbolWeight": 0.40,
    "shockEquityFraction": 0.50,
    "shockedGrossLeverage": 1.0,
    "shockMaintenanceCoverage": 2.0,
}
R._check_endpoint_risk(
    int(times[0]), boundary_weights, 0.75, boundary_state, config, "boundary"
)
breaches = (
    ("equity", np.nextafter(0.75, -np.inf), "cumulative_equity_floor"),
    ("drawdown", np.nextafter(0.20, np.inf), "maximum_drawdown"),
    (
        "endpointGrossLeverage",
        np.nextafter(1.0, np.inf),
        "maximum_endpoint_gross_leverage",
    ),
    (
        "maximumAbsoluteSymbolWeight",
        np.nextafter(0.40, np.inf),
        "maximum_symbol_weight",
    ),
    (
        "shockEquityFraction",
        np.nextafter(0.50, -np.inf),
        "minimum_shock_equity_fraction",
    ),
    (
        "shockMaintenanceCoverage",
        np.nextafter(2.0, -np.inf),
        "minimum_shock_maintenance_coverage",
    ),
)
for field, value, reason in breaches:
    state = dict(boundary_state)
    state[field] = value
    try:
        R._check_endpoint_risk(
            int(times[0]),
            boundary_weights,
            float(state["equity"]),
            state,
            config,
            "boundary",
        )
    except R.RiskConstraintBreach as error:
        assert error.reason == reason
        assert error.trial_id == "boundary"
        assert error.interval_left_close_time == int(times[0])
        assert error.outcome_close_time == int(times[1])
        json.dumps(error.evidence, allow_nan=False)
    else:
        raise AssertionError("one-ULP risk breach was accepted: " + field)

# Activation turnover equal to 1.25 passes that check; one epsilon above fails it.
cap_columns = ["W0", "W1", "W2", "W3"]
cap_close = pd.DataFrame(100.0, index=times[:2], columns=cap_columns)
cap_settlements = [
    R.FundingSettlement(symbol, int(times[1]), 0.0, 100.0)
    for symbol in cap_columns
]
cap_targets = pd.DataFrame(
    [[0.3125, -0.3125, 0.3125, -0.3125]] * 2,
    index=times[:2],
    columns=cap_columns,
)
try:
    R.evaluate_drifted_intervals(
        cap_close,
        cap_targets,
        np.array([True, False], dtype=bool),
        cap_settlements,
        config,
        trial_id="activation_boundary",
    )
except R.RiskConstraintBreach as error:
    assert error.reason == "maximum_endpoint_gross_leverage"
else:
    raise AssertionError("the later endpoint gross check should reject this fixture")
above_cap_targets = cap_targets.copy()
above_cap_targets.iloc[:, 0] += 1e-12
try:
    R.evaluate_drifted_intervals(
        cap_close,
        above_cap_targets,
        np.array([True, False], dtype=bool),
        cap_settlements,
        config,
        trial_id="activation_breach",
    )
except R.RiskConstraintBreach as error:
    assert error.reason == "maximum_activation_turnover"
    assert error.trial_id == "activation_breach"
else:
    raise AssertionError("activation turnover above 1.25 was accepted")

# A favorable interval cannot hide a drawdown breach caused immediately by
# the activation cost at its left close.
recovery_close = pd.DataFrame(
    {
        "W0": [100.0, 60.2, 59.598],
        "W1": [100.0, 60.2, 59.598],
        "W2": [100.0, 139.8, 141.198],
        "W3": [100.0, 139.8, 141.198],
    },
    index=times[:3],
)
recovery_targets = pd.DataFrame(
    [
        [0.125, 0.125, -0.125, -0.125],
        [-0.125, -0.125, 0.125, 0.125],
        [-0.125, -0.125, 0.125, 0.125],
    ],
    index=times[:3],
    columns=recovery_close.columns,
)
recovery_settlements = [
    R.FundingSettlement(symbol, int(times[1]), 0.0, 100.0)
    for symbol in recovery_close.columns
]
try:
    R.evaluate_drifted_intervals(
        recovery_close,
        recovery_targets,
        np.array([True, True, False], dtype=bool),
        recovery_settlements,
        config,
        trial_id="post_activation_drawdown",
    )
except R.RiskConstraintBreach as error:
    assert error.reason == "maximum_drawdown"
    assert error.interval_left_close_time == int(times[1])
    assert error.evidence["evaluationStage"] == "post_activation"
    assert error.evidence["equityBeforeModeledLiquidation"] < 0.8
    assert error.evidence["observed"] > 0.20
else:
    raise AssertionError("same-bar recovery hid a post-activation drawdown breach")

# Endpoint failures carry exact interval context and immediate close evidence.
typed_close = pd.DataFrame(100.0, index=times[:2], columns=["AAA", "BBB"])
typed_targets = pd.DataFrame(
    [[0.5, 0.0], [0.5, 0.0]], index=times[:2], columns=typed_close.columns
)
typed_settlements = [
    R.FundingSettlement(symbol, int(times[1]), 0.0, 100.0)
    for symbol in typed_close.columns
]
try:
    R.evaluate_drifted_intervals(
        typed_close,
        typed_targets,
        np.array([True, False], dtype=bool),
        typed_settlements,
        config,
        trial_id="typed_test",
    )
except R.RiskConstraintBreach as error:
    assert error.reason == "maximum_symbol_weight"
    assert error.trial_id == "typed_test"
    assert error.interval_left_close_time == int(times[0])
    assert error.outcome_close_time == int(times[1])
    assert error.evidence["field"] == "maximumAbsoluteSymbolWeight"
    assert error.evidence["limit"] == 0.40
    assert error.evidence["modeledImmediateLiquidationTurnover"] > 0.5
    assert abs(error.evidence["modeledImmediateLiquidationCostEquity"] - 0.0005) < 1e-15
    json.dumps(error.evidence, allow_nan=False)
else:
    raise AssertionError("maximum symbol exposure breach was accepted")

# If terminal close cost itself breaches drawdown, evidence uses pre-close equity
# and reconciles the actual immediate liquidation rather than charging it twice.
terminal_close = pd.DataFrame(
    {"AAA": [100.0, 60.14], "BBB": [100.0, 60.14]},
    index=times[:2],
)
terminal_targets = pd.DataFrame(
    [[0.25, 0.25], [0.25, 0.25]],
    index=times[:2],
    columns=terminal_close.columns,
)
try:
    R.evaluate_drifted_intervals(
        terminal_close,
        terminal_targets,
        np.array([True, False], dtype=bool),
        typed_settlements,
        config,
        trial_id="terminal_drawdown",
    )
except R.RiskConstraintBreach as error:
    assert error.reason == "maximum_drawdown"
    evidence = error.evidence
    assert evidence["equityBeforeModeledLiquidation"] > 0.8
    assert evidence["equityAfterModeledLiquidation"] < 0.8
    assert abs(
        evidence["equityBeforeModeledLiquidation"]
        - evidence["modeledImmediateLiquidationCostEquity"]
        - evidence["equityAfterModeledLiquidation"]
    ) < 1e-15
else:
    raise AssertionError("terminal liquidation drawdown breach was accepted")

# Even pathological diagnostics remain strict-JSON evidence.
try:
    R._raise_breach(
        int(times[0]),
        "non_finite_state",
        np.array([np.inf]),
        np.nan,
        config,
        "json_safe",
        {"observed": np.inf, "nested": [np.nan, -np.inf]},
    )
except R.RiskConstraintBreach as error:
    assert error.evidence["observed"] is None
    assert error.evidence["nested"] == [None, None]
    assert error.evidence["equityBeforeModeledLiquidation"] is None
    assert error.evidence["modeledImmediateLiquidationTurnover"] is None
    assert error.evidence["modeledImmediateLiquidationCostFraction"] is None
    assert error.evidence["modeledImmediateLiquidationCostEquity"] is None
    json.dumps(error.evidence, allow_nan=False)
else:
    raise AssertionError("typed breach helper did not raise")
`);
  },
);
