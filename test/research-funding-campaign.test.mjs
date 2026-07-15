import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

test(
  "funding campaign is fixed, causal, and accounts for every settlement",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import inspect
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from funding_campaign import (
    TRIAL_IDS,
    FundingCampaignConfig,
    FundingSettlement,
    campaign_specs,
    causal_funding_zscore,
    evaluate_intervals,
    run_trial_matrix,
    scores_for_trial,
    weights_for_trial,
)

INTERVAL = 8 * 3_600_000
config = FundingCampaignConfig(
    interval_ms=INTERVAL,
    funding_crowding_z=2.0,
    cost_per_turnover=0.001,
)
expected_ids = (
    "resmom_24h_base",
    "resmom_24h_funding_only",
    "resmom_72h_base",
    "resmom_72h_funding_only",
    "resmom_168h_base",
    "resmom_168h_funding_only",
)
specs = campaign_specs(INTERVAL)
assert TRIAL_IDS == expected_ids
assert tuple(spec.trial_id for spec in specs) == expected_ids
assert tuple(spec.horizon_bars for spec in specs) == (3, 3, 9, 9, 21, 21)
assert config.signal_delay_bars == 1
try:
    campaign_specs(4 * 3_600_000)
except ValueError as error:
    assert "exactly eight hours" in str(error)
else:
    raise AssertionError("the registered campaign interval must remain fixed")

times = pd.Index(np.arange(7, dtype=np.int64) * INTERVAL, name="closeTime")
momentum = pd.DataFrame(
    [
        [-2.0, 2.0],
        [2.0, -2.0],
        [-3.0, 3.0],
        [3.0, -3.0],
        [-4.0, 4.0],
        [4.0, -4.0],
        [-5.0, 5.0],
    ],
    index=times,
    columns=["AAA", "BBB"],
)
funding_z = pd.DataFrame(0.0, index=times, columns=momentum.columns)
base_spec = specs[0]
funding_spec = specs[1]

# The funding-only rule has no API path for OI, basis, or taker-flow features.
assert tuple(inspect.signature(scores_for_trial).parameters) == (
    "momentum", "funding_z", "spec", "config"
)
irrelevant_a = {
    "oi": pd.DataFrame(1.0, index=times, columns=momentum.columns),
    "basis": pd.DataFrame(2.0, index=times, columns=momentum.columns),
    "taker": pd.DataFrame(3.0, index=times, columns=momentum.columns),
}
irrelevant_b = {name: -frame * 1e9 for name, frame in irrelevant_a.items()}
left_scores = scores_for_trial(momentum, funding_z, funding_spec, config)
right_scores = scores_for_trial(momentum, funding_z, funding_spec, config)
np.testing.assert_allclose(left_scores, right_scores, equal_nan=True)
assert any(not irrelevant_a[name].equals(irrelevant_b[name]) for name in irrelevant_a)

# Directional crowding is the only extra eligibility gate, and missing z fails closed.
crowded = funding_z.copy()
crowded.loc[times[2], "AAA"] = -3.0  # short AAA pays crowded negative funding
crowded.loc[times[2], "BBB"] = 3.0   # long BBB pays crowded positive funding
crowded.loc[times[3], "AAA"] = np.nan
funding_scores = scores_for_trial(momentum, crowded, funding_spec, config)
assert funding_scores.loc[times[2]].isna().all()
assert np.isnan(funding_scores.loc[times[3], "AAA"])
assert np.isfinite(scores_for_trial(momentum, crowded, base_spec, config).loc[times[2]]).all()

# Current-close information changes next-bar weights, never current weights.
one_bar = weights_for_trial(momentum, funding_z, base_spec, config)
np.testing.assert_allclose(one_bar.iloc[0], [0.0, 0.0])
np.testing.assert_allclose(one_bar.iloc[1], [-0.5, 0.5])
np.testing.assert_allclose(one_bar.iloc[2], [0.5, -0.5])
poisoned_z = funding_z.copy()
poisoned_z.loc[times[1]] = [10.0, -10.0]
poisoned = weights_for_trial(momentum, poisoned_z, funding_spec, config)
baseline = weights_for_trial(momentum, funding_z, funding_spec, config)
np.testing.assert_allclose(poisoned.iloc[:2], baseline.iloc[:2])
np.testing.assert_allclose(poisoned.iloc[2], [0.0, 0.0])
assert not np.array_equal(poisoned.iloc[2].to_numpy(), baseline.iloc[2].to_numpy())

# Executed sides always agree with momentum direction; an all-positive row is not shorted.
all_positive = momentum.copy()
all_positive.loc[times[2]] = [1.0, 2.0]
directional_weights = weights_for_trial(all_positive, funding_z, base_spec, config)
np.testing.assert_allclose(directional_weights.loc[times[3]], [0.0, 0.0])
aligned = directional_weights * momentum.shift(1)
assert (aligned.to_numpy()[np.abs(directional_weights.to_numpy()) > 0] > 0).all()

# The frozen additional-delay stress recomputes weights two bars later; zero delay is illegal.
two_bar_config = replace(config, signal_delay_bars=2)
two_bar = weights_for_trial(momentum, funding_z, base_spec, two_bar_config)
np.testing.assert_allclose(two_bar.iloc[:2], 0.0)
np.testing.assert_allclose(two_bar.iloc[2:].to_numpy(), one_bar.iloc[1:-1].to_numpy())
try:
    weights_for_trial(momentum, funding_z, base_spec, replace(config, signal_delay_bars=0))
except ValueError as error:
    assert "signal_delay_bars" in str(error)
else:
    raise AssertionError("same-close execution must fail")

# Causal z-scores are invariant to observations after the evaluated prefix.
raw_rates = pd.DataFrame(
    {
        "AAA": [0.1, -0.2, 0.3, 0.0, 0.4, -0.1, 9.0],
        "BBB": [-0.1, 0.25, -0.35, 0.1, -0.2, 0.3, -9.0],
    },
    index=times,
)
full_z = causal_funding_zscore(raw_rates, 3, min_periods=2)
prefix_z = causal_funding_zscore(raw_rates.iloc[:-1], 3, min_periods=2)
np.testing.assert_allclose(full_z.iloc[:-1], prefix_z, equal_nan=True)

def funding_cashflow(weight, rate):
    closes = pd.DataFrame({"AAA": [100.0, 100.0]}, index=times[:2])
    weights = pd.DataFrame({"AAA": [weight, weight]}, index=times[:2])
    detail = evaluate_intervals(
        closes,
        weights,
        [FundingSettlement("AAA", int(times[1]), rate, 100.0)],
        replace(config, cost_per_turnover=0.0),
    )
    return detail["fundingCashflow"].iloc[0]

# Longs pay positive funding and shorts receive it; negative rates reverse both signs.
assert abs(funding_cashflow(1.0, 0.01) - -0.01) < 1e-15
assert abs(funding_cashflow(1.0, -0.01) - 0.01) < 1e-15
assert abs(funding_cashflow(-1.0, 0.01) - 0.01) < 1e-15
assert abs(funding_cashflow(-1.0, -0.01) - -0.01) < 1e-15
marked = evaluate_intervals(
    pd.DataFrame({"AAA": [100.0, 100.0]}, index=times[:2]),
    pd.DataFrame({"AAA": [0.5, 0.5]}, index=times[:2]),
    [FundingSettlement("AAA", int(times[1]), 0.01, 110.0)],
    replace(config, cost_per_turnover=0.0),
)
assert abs(marked["fundingCashflow"].iloc[0] - (-0.5 * 110.0 / 100.0 * 0.01)) < 1e-15

# Intervals are (left, right]: exact boundaries, boundary+7ms, and multiple events count.
boundary_times = pd.Index(np.arange(4, dtype=np.int64) * INTERVAL, name="closeTime")
flat_close = pd.DataFrame({"AAA": 100.0}, index=boundary_times)
long_weights = pd.DataFrame({"AAA": 1.0}, index=boundary_times)
events = [
    FundingSettlement("AAA", int(boundary_times[0]), 0.50, 100.0),
    FundingSettlement("AAA", int(boundary_times[1]), 0.01, 100.0),
    FundingSettlement("AAA", int(boundary_times[1]) + 7, 0.02, 100.0),
    FundingSettlement("AAA", int(boundary_times[2]), 0.03, 100.0),
    FundingSettlement("AAA", int(boundary_times[2]) + 7, 0.04, 100.0),
    FundingSettlement("AAA", int(boundary_times[3]), 0.05, 100.0),
    FundingSettlement("AAA", int(boundary_times[3]) + 7, 0.50, 100.0),
]
boundaries = evaluate_intervals(
    flat_close,
    long_weights,
    events,
    replace(config, cost_per_turnover=0.0),
)
np.testing.assert_allclose(boundaries["fundingCashflow"], [-0.01, -0.05, -0.09])

# Price, funding, turnover, costs, and net are recomputed from effective delayed weights.
close = pd.DataFrame(
    {
        "AAA": [100.0, 110.0, 99.0, 108.0, 102.0, 112.0, 105.0],
        "BBB": [100.0, 95.0, 104.0, 96.0, 107.0, 98.0, 109.0],
    },
    index=times,
)
neutral_events = [
    FundingSettlement(symbol, int(times[1]), 0.0, float(close.loc[times[1], symbol]))
    for symbol in close.columns
]
accounting = evaluate_intervals(close, one_bar, neutral_events, config)
np.testing.assert_allclose(accounting["gross"], accounting["priceGross"] + accounting["fundingCashflow"])
np.testing.assert_allclose(accounting["cost"], config.cost_per_turnover * accounting["turnover"])
np.testing.assert_allclose(accounting["net"], accounting["gross"] - accounting["cost"])
np.testing.assert_allclose(accounting["turnover"].iloc[:3], [0.0, 1.0, 2.0])
np.testing.assert_allclose(accounting[["weight_AAA", "weight_BBB"]], one_bar.iloc[:-1])

zero_cost = evaluate_intervals(
    close, one_bar, neutral_events, replace(config, cost_per_turnover=0.0)
)
high_cost = evaluate_intervals(
    close, one_bar, neutral_events, replace(config, cost_per_turnover=0.01)
)
np.testing.assert_allclose(zero_cost["gross"], high_cost["gross"])
np.testing.assert_allclose(
    high_cost["net"],
    zero_cost["net"] - 0.01 * high_cost["turnover"],
)
assert np.all(high_cost["net"] <= zero_cost["net"] + 1e-15)

# Every malformed resolved settlement input fails instead of becoming zero cashflow.
invalid_events = [
    FundingSettlement("AAA", int(times[1]), np.nan, 100.0),
    FundingSettlement("AAA", int(times[1]), None, 100.0),
    FundingSettlement("AAA", int(times[1]), 0.01, np.nan),
    FundingSettlement("AAA", int(times[1]), 0.01, None),
    FundingSettlement("AAA", int(times[1]), 0.01, 0.0),
]
for event in invalid_events:
    try:
        evaluate_intervals(close, one_bar, [event], config)
    except ValueError as error:
        assert "funding rate" in str(error) or "resolved mark price" in str(error)
    else:
        raise AssertionError("invalid funding settlement must fail closed")

try:
    evaluate_intervals(close, one_bar, [neutral_events[0]], config)
except ValueError as error:
    assert "absent for: BBB" in str(error)
else:
    raise AssertionError("an absent symbol funding schedule must fail closed")

try:
    evaluate_intervals(close, one_bar, [neutral_events[0], neutral_events[0]], config)
except ValueError as error:
    assert "duplicate" in str(error)
else:
    raise AssertionError("duplicate symbol/time settlements must fail closed")

matrix, details, returned_specs = run_trial_matrix(
    close,
    {24: momentum, 72: momentum * 0.8, 168: momentum * 0.6},
    funding_z,
    neutral_events,
    config,
)
assert matrix.shape == (len(times) - 1, 6)
assert tuple(matrix.columns) == expected_ids
assert tuple(details) == expected_ids
assert returned_specs == specs
assert np.isfinite(matrix.to_numpy()).all()
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
    });
    assert.equal(run.status, 0, run.stderr);
  },
);
