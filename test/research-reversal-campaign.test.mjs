import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

test(
  "residual reversal campaign is fixed, causal, phased, and fully accounted",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import inspect
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from reversal_campaign import (
    HORIZON_HOURS,
    REBALANCE_ANCHOR_OPEN_TIME,
    REBALANCE_BARS,
    TRIAL_IDS,
    FundingSettlement,
    PortfolioBankruptcyError,
    ReversalCampaignConfig,
    campaign_specs,
    run_trial,
    run_trial_matrix,
    scores_for_trial,
    weights_for_trial,
)

INTERVAL = 8 * 3_600_000
config = ReversalCampaignConfig(
    interval_ms=INTERVAL,
    cost_per_turnover=0.001,
)
expected_ids = (
    "resrev_24h_rebalance_1bar",
    "resrev_24h_rebalance_3bar",
    "resrev_72h_rebalance_1bar",
    "resrev_72h_rebalance_3bar",
    "resrev_168h_rebalance_1bar",
    "resrev_168h_rebalance_3bar",
)
specs = campaign_specs(INTERVAL)
assert HORIZON_HOURS == (24, 72, 168)
assert REBALANCE_BARS == (1, 3)
assert TRIAL_IDS == expected_ids
assert tuple(spec.trial_id for spec in specs) == expected_ids
assert tuple(spec.horizon_bars for spec in specs) == (3, 3, 9, 9, 21, 21)
assert tuple(spec.rebalance_bars for spec in specs) == (1, 3, 1, 3, 1, 3)
assert config.top_n == 1
assert config.signal_delay_bars == 1
assert config.rebalance_anchor_open_time == REBALANCE_ANCHOR_OPEN_TIME
assert config.rebalance_phase_bars == 0

# No funding input can affect reversal scores or asset eligibility.
assert tuple(inspect.signature(scores_for_trial).parameters) == (
    "residual_momentum", "spec"
)

times = pd.Index(
    REBALANCE_ANCHOR_OPEN_TIME
    + np.arange(10, dtype=np.int64) * INTERVAL
    + INTERVAL
    - 1,
    name="closeTime",
)
momentum = pd.DataFrame(
    [
        [-2.0, 2.0],
        [9.0, -9.0],
        [8.0, -8.0],
        [4.0, -4.0],
        [-7.0, 7.0],
        [-6.0, 6.0],
        [-3.0, 3.0],
        [5.0, -5.0],
        [6.0, -6.0],
        [7.0, -7.0],
    ],
    index=times,
    columns=["AAA", "BBB"],
)
one_bar_spec = specs[0]
three_bar_spec = specs[1]

# Reversal is literal: the prior loser is long and the prior winner is short.
scores = scores_for_trial(momentum, one_bar_spec)
np.testing.assert_allclose(scores.iloc[0], [2.0, -2.0])
one_bar = weights_for_trial(momentum, one_bar_spec, config)
np.testing.assert_allclose(one_bar.iloc[0], [0.0, 0.0])
np.testing.assert_allclose(one_bar.iloc[1], [0.5, -0.5])
np.testing.assert_allclose(one_bar.iloc[2], [-0.5, 0.5])
executed_against_signal = one_bar.iloc[1:].to_numpy() * momentum.iloc[:-1].to_numpy()
assert (executed_against_signal[np.abs(one_bar.iloc[1:].to_numpy()) > 0] < 0).all()

# A three-bar signal is anchored to the registered row-zero open, delayed one
# bar, and then held until the following absolute scheduled signal activates.
three_bar = weights_for_trial(momentum, three_bar_spec, config)
np.testing.assert_allclose(three_bar.iloc[0], [0.0, 0.0])
np.testing.assert_allclose(three_bar.iloc[1:4], [[0.5, -0.5]] * 3)
np.testing.assert_allclose(three_bar.iloc[4:7], [[-0.5, 0.5]] * 3)
np.testing.assert_allclose(three_bar.iloc[7:10], [[0.5, -0.5]] * 3)

# All three absolute phases are explicit. They move only the three-bar decision
# rows; modulo-one controls are identical for every phase.
phase_one_config = replace(config, rebalance_phase_bars=1)
phase_two_config = replace(config, rebalance_phase_bars=2)
phase_one = weights_for_trial(momentum, three_bar_spec, phase_one_config)
phase_two = weights_for_trial(momentum, three_bar_spec, phase_two_config)
np.testing.assert_allclose(phase_one.iloc[:2], 0.0)
np.testing.assert_allclose(phase_one.iloc[2:5], [[-0.5, 0.5]] * 3)
np.testing.assert_allclose(phase_two.iloc[:3], 0.0)
np.testing.assert_allclose(phase_two.iloc[3:6], [[-0.5, 0.5]] * 3)
np.testing.assert_allclose(
    weights_for_trial(momentum, one_bar_spec, phase_one_config), one_bar
)
np.testing.assert_allclose(
    weights_for_trial(momentum, one_bar_spec, phase_two_config), one_bar
)

# Starting at a different row must not reset cadence to row zero. Original row
# three remains the next absolute decision, becoming effective at original row four.
offset = weights_for_trial(momentum.iloc[1:], three_bar_spec, config)
np.testing.assert_allclose(offset.iloc[:3], 0.0)
np.testing.assert_allclose(offset.iloc[3:6], [[-0.5, 0.5]] * 3)

# Current-close data cannot affect current effective weights, while changing a
# non-scheduled row cannot affect any target.
poisoned = momentum.copy()
poisoned.iloc[3] = [-40.0, 40.0]
poisoned_weights = weights_for_trial(poisoned, three_bar_spec, config)
np.testing.assert_allclose(poisoned_weights.iloc[:4], three_bar.iloc[:4])
np.testing.assert_allclose(poisoned_weights.iloc[4], [0.5, -0.5])
non_scheduled = momentum.copy()
non_scheduled.iloc[2] = [-100.0, 100.0]
np.testing.assert_allclose(
    weights_for_trial(non_scheduled, three_bar_spec, config),
    three_bar,
)

# Missing one side closes the book on the next scheduled activation.
one_sided = momentum.copy()
one_sided.iloc[3] = [1.0, 2.0]
one_sided_weights = weights_for_trial(one_sided, three_bar_spec, config)
np.testing.assert_allclose(one_sided_weights.iloc[3], [0.5, -0.5])
np.testing.assert_allclose(one_sided_weights.iloc[4:7], 0.0)

# Shared interval accounting charges turnover and includes funding cashflows;
# funding changes cashflow but never any trial's selected weights.
close = pd.DataFrame(100.0, index=times, columns=momentum.columns)
events = [
    FundingSettlement("AAA", int(times[2]), 0.01, 100.0),
    FundingSettlement("BBB", int(times[2]), 0.02, 100.0),
]
momentum_by_horizon = {24: momentum, 72: momentum * 0.8, 168: momentum * 0.6}
matrix, details, returned_specs = run_trial_matrix(
    close,
    momentum_by_horizon,
    events,
    config,
)
assert matrix.shape == (len(times) - 1, 6)
assert tuple(matrix.columns) == expected_ids
assert tuple(details) == expected_ids
assert returned_specs == specs
assert np.isfinite(matrix.to_numpy()).all()

detail = details[three_bar_spec.trial_id]
assert abs(detail["fundingCashflow"].iloc[1] - 0.005) < 1e-15
assert abs(detail["turnover"].iloc[1] - 1.0) < 1e-15
assert abs(detail["cost"].iloc[1] - 0.001) < 1e-15
assert abs(detail["net"].iloc[1] - 0.004) < 1e-15
# Flat prices do not imply constant weights: funding and cost changed equity,
# while the held contract notionals remained unchanged on row two.
np.testing.assert_allclose(
    detail[["weight_AAA", "weight_BBB"]].iloc[2],
    [0.5 / 1.004, -0.5 / 1.004],
)
assert detail["turnover"].iloc[2] == 0.0

zero_cost_matrix, zero_cost_details, _ = run_trial_matrix(
    close,
    momentum_by_horizon,
    events,
    replace(config, cost_per_turnover=0.0),
)
zero_cost_detail = zero_cost_details[three_bar_spec.trial_id]
assert abs(zero_cost_detail["gross"].iloc[1] - detail["gross"].iloc[1]) < 1e-15
assert abs(zero_cost_detail["net"].iloc[1] - detail["net"].iloc[1] - 0.001) < 1e-15
np.testing.assert_allclose(detail["cost"], config.cost_per_turnover * detail["turnover"])
np.testing.assert_allclose(detail["net"], detail["gross"] - detail["cost"])
assert not np.array_equal(
    zero_cost_detail[["weight_AAA", "weight_BBB"]].iloc[2].to_numpy(),
    detail[["weight_AAA", "weight_BBB"]].iloc[2].to_numpy(),
)
assert abs(
    zero_cost_detail["turnover"].iloc[4] - detail["turnover"].iloc[4]
) > 1e-12

changed_events = [
    FundingSettlement("AAA", int(times[2]), -0.07, 100.0),
    FundingSettlement("BBB", int(times[2]), 0.09, 100.0),
]
_, changed_details, _ = run_trial_matrix(
    close,
    momentum_by_horizon,
    changed_events,
    config,
)
# Funding cannot alter scheduled target selection. Actual weights may differ
# between activations because funding legitimately changes portfolio equity.
for spec in specs:
    activation_rows = range(1, len(times) - 1, spec.rebalance_bars)
    weight_columns = ["weight_AAA", "weight_BBB"]
    np.testing.assert_allclose(
        details[spec.trial_id][weight_columns].iloc[list(activation_rows)],
        changed_details[spec.trial_id][weight_columns].iloc[list(activation_rows)],
    )
assert not np.array_equal(
    detail["fundingCashflow"].to_numpy(),
    changed_details[three_bar_spec.trial_id]["fundingCashflow"].to_numpy(),
)
assert abs(
    detail["turnover"].iloc[4]
    - changed_details[three_bar_spec.trial_id]["turnover"].iloc[4]
) > 1e-12

# Boundary assignment is exactly (left, right], including multiple events.
boundary_events = [
    FundingSettlement("AAA", int(times[0]), 0.90, 100.0),
    FundingSettlement("BBB", int(times[0]), 0.90, 100.0),
    FundingSettlement("AAA", int(times[2]), 0.01, 100.0),
    FundingSettlement("BBB", int(times[2]), 0.00, 100.0),
    FundingSettlement("AAA", int(times[2]) + 7, 0.02, 100.0),
    FundingSettlement("BBB", int(times[2]) + 7, 0.00, 100.0),
    FundingSettlement("AAA", int(times[3]), 0.03, 100.0),
    FundingSettlement("BBB", int(times[3]), 0.00, 100.0),
]
boundary_detail = run_trial(
    close,
    {24: momentum},
    boundary_events,
    replace(config, cost_per_turnover=0.0),
    three_bar_spec,
)
assert abs(boundary_detail["fundingCashflow"].iloc[0]) < 1e-15
assert abs(boundary_detail["fundingCashflow"].iloc[1] - -0.005) < 1e-15
assert abs(
    boundary_detail["fundingCashflow"].iloc[2]
    - boundary_detail["weight_AAA"].iloc[2] * -0.05
) < 1e-15

# Non-flat prices drift executed weights between activations. There is no
# hidden rebalance until row four, whose turnover starts from the drifted book.
steady_momentum = pd.DataFrame(
    {"AAA": -2.0, "BBB": 2.0}, index=times, dtype=float
)
trending_close = pd.DataFrame(
    {
        "AAA": 100.0 * np.power(1.1, np.arange(len(times), dtype=float)),
        "BBB": 100.0,
    },
    index=times,
)
neutral_events = [
    FundingSettlement("AAA", int(times[2]), 0.0, 100.0),
    FundingSettlement("BBB", int(times[2]), 0.0, 100.0),
]
# Execution honors every phase as well as the target helper: delayed first
# activations occur at rows one, two, and three respectively.
phase_details = []
for phase, first_activation in ((0, 1), (1, 2), (2, 3)):
    phase_detail = run_trial(
        close,
        {24: steady_momentum},
        neutral_events,
        replace(config, cost_per_turnover=0.0, rebalance_phase_bars=phase),
        three_bar_spec,
    )
    np.testing.assert_allclose(
        phase_detail[["weight_AAA", "weight_BBB"]].iloc[:first_activation], 0.0
    )
    np.testing.assert_allclose(
        phase_detail[["weight_AAA", "weight_BBB"]].iloc[first_activation],
        [0.5, -0.5],
    )
    assert phase_detail["turnover"].iloc[first_activation] == 1.0
    phase_details.append(phase_detail)

drift_detail = run_trial(
    trending_close,
    {24: steady_momentum},
    neutral_events,
    replace(config, cost_per_turnover=0.0),
    three_bar_spec,
)
steady_targets = weights_for_trial(steady_momentum, three_bar_spec, config)
np.testing.assert_allclose(
    drift_detail[["weight_AAA", "weight_BBB"]].iloc[1], [0.5, -0.5]
)
np.testing.assert_allclose(
    drift_detail[["weight_AAA", "weight_BBB"]].iloc[2],
    [0.5 * 1.1 / 1.05, -0.5 / 1.05],
)
assert not np.array_equal(
    drift_detail[["weight_AAA", "weight_BBB"]].iloc[2].to_numpy(),
    steady_targets.iloc[2].to_numpy(),
)
np.testing.assert_allclose(drift_detail["turnover"].iloc[2:4], 0.0)
row_three_weights = drift_detail[["weight_AAA", "weight_BBB"]].iloc[3].to_numpy()
row_three_returns = (
    trending_close.iloc[4].to_numpy() / trending_close.iloc[3].to_numpy() - 1.0
)
pretrade_row_four = (
    row_three_weights
    * (1.0 + row_three_returns)
    / (1.0 + drift_detail["net"].iloc[3])
)
expected_turnover = np.abs(steady_targets.iloc[4].to_numpy() - pretrade_row_four).sum()
assert expected_turnover > 0.0
assert abs(drift_detail["turnover"].iloc[4] - expected_turnover) < 1e-15
np.testing.assert_allclose(
    drift_detail[["weight_AAA", "weight_BBB"]].iloc[4],
    steady_targets.iloc[4],
)
assert drift_detail["turnover"].iloc[-1] == 0.0  # no terminal liquidation charge

# When the final evaluated interval is itself an activation, its charged
# turnover is still only drifted-pretrade-to-target; open terminal positions
# are neither flattened nor added to that interval's turnover.
terminal_config = replace(config, cost_per_turnover=0.001)
terminal_detail = run_trial(
    trending_close.iloc[:6],
    {24: steady_momentum.iloc[:6]},
    neutral_events,
    terminal_config,
    three_bar_spec,
)
terminal_targets = weights_for_trial(
    steady_momentum.iloc[:6], three_bar_spec, terminal_config
)
terminal_prior_weights = terminal_detail[["weight_AAA", "weight_BBB"]].iloc[3].to_numpy()
terminal_prior_returns = (
    trending_close.iloc[4].to_numpy() / trending_close.iloc[3].to_numpy() - 1.0
)
terminal_pretrade = (
    terminal_prior_weights
    * (1.0 + terminal_prior_returns)
    / (1.0 + terminal_detail["net"].iloc[3])
)
terminal_expected_turnover = np.abs(
    terminal_targets.iloc[4].to_numpy() - terminal_pretrade
).sum()
assert abs(terminal_detail["turnover"].iloc[-1] - terminal_expected_turnover) < 1e-15
assert abs(
    terminal_detail["cost"].iloc[-1]
    - terminal_config.cost_per_turnover * terminal_expected_turnover
) < 1e-15
np.testing.assert_allclose(
    terminal_detail[["weight_AAA", "weight_BBB"]].iloc[-1],
    [0.5, -0.5],
)

# Insolvency fails closed with typed trial/timestamp evidence. It is never
# clipped, restarted, or converted into an artificial continuation path.
bankrupt_close = close.copy()
bankrupt_close.iloc[2] = [100.0, 300.0]
try:
    run_trial(
        bankrupt_close,
        {24: steady_momentum},
        neutral_events,
        replace(config, cost_per_turnover=0.0),
        three_bar_spec,
    )
except PortfolioBankruptcyError as error:
    assert error.close_time == int(times[1])
    assert error.interval_left_close_time == int(times[1])
    assert error.outcome_close_time == int(times[2])
    assert error.trial_id == three_bar_spec.trial_id
    assert three_bar_spec.trial_id in str(error)
else:
    raise AssertionError("portfolio bankruptcy must fail closed")

# Malformed campaign settings fail rather than silently changing the ledger.
bad_configs = [
    replace(config, interval_ms=4 * 3_600_000),
    replace(config, top_n=2),
    replace(config, signal_delay_bars=0),
    replace(config, rebalance_phase_bars=-1),
    replace(config, rebalance_phase_bars=3),
    replace(config, rebalance_phase_bars=True),
    replace(config, gross_exposure=np.nan),
    replace(config, cost_per_turnover=-0.001),
    replace(config, rebalance_anchor_open_time=REBALANCE_ANCHOR_OPEN_TIME + INTERVAL),
]
for bad_config in bad_configs:
    try:
        weights_for_trial(momentum, three_bar_spec, bad_config)
    except ValueError:
        pass
    else:
        raise AssertionError(f"malformed config must fail: {bad_config}")

try:
    weights_for_trial(momentum.iloc[::-1], three_bar_spec, config)
except ValueError as error:
    assert "unique and ordered" in str(error)
else:
    raise AssertionError("unordered residual momentum must fail")

try:
    weights_for_trial(momentum.set_axis(times + 1), three_bar_spec, config)
except ValueError as error:
    assert "interval boundaries" in str(error) or "absolute grid" in str(error)
else:
    raise AssertionError("off-grid close times must fail")
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
    });
    assert.equal(run.status, 0, run.stderr);
  },
);
