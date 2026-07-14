import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const hasResearchPython =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

test(
  "edge campaign is causal, complete, and cost monotone",
  { skip: !hasResearchPython },
  () => {
    const program = String.raw`
import sys
sys.path.insert(0, sys.argv[1])

from dataclasses import replace
import hashlib
import json
import numpy as np
import pandas as pd
import tempfile

from edge_campaign import (
    CampaignConfig,
    campaign_specs,
    prepare_panel,
    run_trial_matrix,
    scores_for_spec,
)
import run_edge_campaign as runner

assert runner.HOLDOUT_REGISTRY_DIR.is_absolute()

rng = np.random.default_rng(17)
n = 420
times = np.arange(n, dtype=np.int64) * 3_600_000
market = rng.normal(0.0001, 0.004, n)
panel = {}
for index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
    residual = rng.normal((index - 1.5) * 0.00003, 0.002, n)
    returns = (0.7 + index * 0.1) * market + residual
    close = 100 * np.exp(np.cumsum(returns))
    oi = 10_000 * np.exp(np.cumsum(rng.normal(0.0002, 0.002, n)))
    panel[symbol] = pd.DataFrame({
        "openTime": times,
        "close": close,
        "funding": 0.0001 * np.tanh(returns * 100),
        "oi": oi,
        "basis": 0.001 * np.tanh(returns * 80),
        "taker": 1 + np.tanh(returns * 120) * 0.1,
    })

config = CampaignConfig(
    interval_ms=3_600_000,
    beta_lookback_bars=48,
    feature_lookback_bars=48,
    oi_change_bars=6,
    rebalance_bars=4,
)
assert config.signal_delay_bars == 1
specs = campaign_specs(config.interval_ms)
assert len(specs) == 15
assert len({spec.name for spec in specs}) == 15

matrix, details, returned_specs = run_trial_matrix(panel, config)
assert matrix.shape == (n - 1, 15)
assert list(matrix.columns) == [spec.name for spec in specs]
assert np.isfinite(matrix.to_numpy()).all()
assert set(details) == set(matrix.columns)
assert returned_specs == specs

prepared = prepare_panel(panel, config)
poison_row = 250
for ablation, field in [
    ("funding_basis", "funding_z"),
    ("open_interest", "oi_change"),
    ("taker_flow", "taker_imbalance"),
]:
    specification = next(
        spec
        for spec in specs
        if spec.horizon_hours == 24 and spec.ablation == ablation
    )
    poisoned_values = getattr(prepared, field).copy()
    direction = np.sign(
        prepared.residual_momentum[specification.horizon_bars].iloc[poison_row]
    )
    if ablation == "funding_basis":
        poisoned_values.iloc[poison_row] = -direction * np.inf
        neutral_basis = prepared.basis_z.copy()
        neutral_basis.iloc[poison_row] = 0.0
        poisoned = replace(
            prepared,
            funding_z=poisoned_values,
            basis_z=neutral_basis,
        )
    elif ablation == "taker_flow":
        poisoned_values.iloc[poison_row] = direction * np.inf
        poisoned = replace(prepared, taker_imbalance=poisoned_values)
    else:
        poisoned_values.iloc[poison_row] = np.inf
        poisoned = replace(prepared, oi_change=poisoned_values)
    poisoned_scores = scores_for_spec(poisoned, specification, config)
    assert poisoned_scores.iloc[poison_row + config.signal_delay_bars].isna().all()

try:
    run_trial_matrix(panel, replace(config, signal_delay_bars=0))
except ValueError as error:
    assert "signal_delay_bars" in str(error)
else:
    raise AssertionError("same-close campaign execution must be rejected")

high_cost, _, _ = run_trial_matrix(panel, replace(config, cost_per_turnover=0.001))
assert np.all(high_cost.to_numpy() <= matrix.to_numpy() + 1e-15)
assert np.any(high_cost.to_numpy() < matrix.to_numpy() - 1e-15)

prefix = {symbol: frame.iloc[:360].copy() for symbol, frame in panel.items()}
prefix_matrix, _, _ = run_trial_matrix(prefix, config)
overlap = prefix_matrix.index[:-1]
np.testing.assert_allclose(
    prefix_matrix.loc[overlap].to_numpy(),
    matrix.loc[overlap].to_numpy(),
    rtol=0,
    atol=1e-15,
)

gapped = {symbol: frame.drop(index=120).reset_index(drop=True) for symbol, frame in panel.items()}
try:
    run_trial_matrix(gapped, config)
except ValueError as error:
    assert "exactly one interval" in str(error)
else:
    raise AssertionError("a shared timestamp gap must fail closed")

toy_frame = pd.DataFrame({
    "openTime": [1, 2, 3, 4],
    "candidate__gross": [0.0, 0.0, 0.0, 0.0],
    "candidate__weight_AAA": [0.5, 0.5, 0.5, 0.5],
    "candidate__weight_BBB": [-0.5, -0.5, -0.5, -0.5],
})
toy_candidates = {
    "candidate": {
        "grossColumn": "candidate__gross",
        "inputWeightColumns": (
            "candidate__weight_AAA",
            "candidate__weight_BBB",
        ),
        "outputWeightColumns": ("weight_AAA", "weight_BBB"),
    }
}
toy_folds = pd.DataFrame([
    {"outer_fold": 0, "test_start": 0, "test_stop": 2, "selected_candidate": "candidate"},
    {"outer_fold": 1, "test_start": 2, "test_stop": 4, "selected_candidate": "candidate"},
])
stitched = runner._evaluate_outer_choices(toy_frame, toy_candidates, toy_folds, 0.001)
np.testing.assert_allclose(stitched["turnover"], [1.0, 0.0, 0.0, 0.0])

diagnostic_rows = np.arange(17, dtype=float)
diagnostic_source = pd.DataFrame(
    {
        "trial_a": 0.001 + 0.004 * np.sin(diagnostic_rows * 0.7),
        "trial_b": 0.0005 + 0.003 * np.cos(diagnostic_rows * 0.4),
        "trial_c": -0.0002 + 0.005 * np.sin(diagnostic_rows * 0.5 + 1.0),
    },
    index=pd.Index(
        np.arange(len(diagnostic_rows), dtype=np.int64) * 86_400_000,
        name="openTime",
    ),
)
diagnostics_four = runner._diagnostics(
    diagnostic_source, "trial_a", 365.0, 86_400_000, 4
)
diagnostics_six = runner._diagnostics(
    diagnostic_source, "trial_a", 365.0, 86_400_000, 6
)
assert len(diagnostics_four) == 3
assert len(diagnostics_six) == 3
report_four, dsr_matrix_four, pbo_matrix_four = diagnostics_four
report_six, dsr_matrix_six, pbo_matrix_six = diagnostics_six
assert len(dsr_matrix_four) == len(diagnostic_source)
assert len(dsr_matrix_six) == len(diagnostic_source)
assert len(pbo_matrix_four) == 16
assert len(pbo_matrix_six) == 12
assert report_four["dsrObservations"] == len(diagnostic_source)
assert report_six["dsrObservations"] == len(diagnostic_source)
assert report_four["pboObservations"] == len(pbo_matrix_four)
assert report_six["pboObservations"] == len(pbo_matrix_six)
assert report_four["deflatedSharpe"]["observations"] == len(diagnostic_source)
assert report_six["deflatedSharpe"]["observations"] == len(diagnostic_source)
np.testing.assert_allclose(
    report_four["deflatedSharpe"]["probability"],
    report_six["deflatedSharpe"]["probability"],
    rtol=0,
    atol=0,
)

rare_regime_oos = pd.DataFrame(
    {
        "openTime": np.arange(6),
        "net": [0.01, 0.01, 0.01, 0.01, -0.10, -0.10],
        "active": [2, 2, 2, 2, 2, 2],
    }
)
rare_regime_labels = pd.Series(
    ["common", "common", "common", "common", "rare", "rare"],
    index=rare_regime_oos["openTime"],
)
_, rare_regime_passed, _ = runner._regime_report(
    rare_regime_oos,
    rare_regime_labels,
    periods_per_year=365.0,
    min_observations=3,
    max_loss=0.05,
)
assert rare_regime_passed is False

coverage_frame = pd.DataFrame(
    {
        "openTime": np.arange(5),
        "close": np.ones(5),
        "funding": [1.0, np.nan, 1.0, 1.0, np.inf],
        "oi": [1.0, 1.0, np.nan, 1.0, 1.0],
        "basis": [1.0, 1.0, 1.0, np.nan, 1.0],
        "taker": [1.0, 1.0, 1.0, 1.0, np.nan],
    }
)
coverage_panel = {"AAA": coverage_frame}
assert np.isclose(
    runner._minimum_joint_derivatives_coverage(coverage_panel), 0.2
)

pass_status = runner._campaign_status(
    True, {"status": "pass", "openRequested": True}
)
fail_status = runner._campaign_status(
    True, {"status": "fail", "openRequested": True}
)
assert pass_status != fail_status
assert "pass" in pass_status.lower()
assert "fail" in fail_status.lower()
assert runner._campaign_status(
    True, {"status": "reserved", "openRequested": False}
) == (
    "ready_for_final_holdout"
)
assert runner._campaign_status(
    False, {"status": "reserved", "openRequested": False}
) == (
    "insufficient_evidence"
)

assert runner._windows_overlap(
    runner._holdout_window(
        ["AAA"], "4h", 0, 8 * 3_600_000
    ),
    runner._holdout_window(
        ["AAA"], "1h", 9 * 3_600_000, 9 * 3_600_000
    ),
)

with tempfile.TemporaryDirectory() as reservation_root:
    reservation_path = runner.Path(reservation_root) / "reservation.json"
    original_link = runner.os.link
    try:
        def fail_link(*_args, **_kwargs):
            raise OSError("simulated interrupted install")
        runner.os.link = fail_link
        try:
            runner._write_json_exclusive(reservation_path, {"complete": True})
        except OSError as error:
            assert "simulated" in str(error)
        else:
            raise AssertionError("an interrupted reservation install must fail")
    finally:
        runner.os.link = original_link
    assert not reservation_path.exists()
    assert not list(reservation_path.parent.glob("*.tmp"))
    runner._write_json_exclusive(reservation_path, {"complete": True})
    assert json.loads(reservation_path.read_text()) == {"complete": True}
    try:
        runner._write_json_exclusive(reservation_path, {"complete": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("an exclusive reservation must not be replaced")
    assert json.loads(reservation_path.read_text()) == {"complete": True}

runner.feed.load_panel = lambda *_args, **_kwargs: panel
with tempfile.TemporaryDirectory() as output_dir:
    runner.HOLDOUT_REGISTRY_DIR = runner.Path(output_dir) / "holdout-registry"
    runner_argv = [
        "AAA", "BBB", "CCC", "DDD",
        "--output-dir", output_dir,
        "--beta-lookback-hours", "48",
        "--feature-lookback-hours", "48",
        "--oi-change-hours", "6",
        "--initial-train", "60",
        "--outer-test-size", "20",
        "--inner-initial-train", "30",
        "--inner-test-size", "10",
        "--pbo-slices", "2",
        "--bootstrap-reps", "100",
        "--min-symbols", "5",
        "--min-oos-observations", "20",
        "--open-final-holdout",
    ]
    args = runner.parse_args(runner_argv)
    summary = runner.run(args)
    assert summary["campaign"] == "residual_momentum_derivatives_ablation_v1"
    assert len(summary["trials"]) == 15
    assert summary["finalHoldout"]["status"] == "reserved"
    assert summary["finalHoldout"]["openRequested"] is True
    assert "symbolCount" in summary["finalHoldout"]["openBlockedBy"]
    assert summary["nestedOuterOos"]["metrics"]["observations"] > 0
    assert summary["configuration"]["derived"]["signalDelayBars"] == 1
    finite_scores = [
        row for row in summary["finalSelection"]["scores"] if row["score"] is not None
    ]
    assert summary["champion"] == max(finite_scores, key=lambda row: row["score"])["candidate"]
    assert "nestedOuterOos" in summary["stress"]["cost2x"]
    assert "nestedOuterOos" in summary["stress"]["additionalDelay1bar"]
    for filename in [
        "campaign-manifest.json",
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
        "stress-cost2x-nested-oos.csv",
        "stress-additionalDelay1bar-nested-oos.csv",
    ]:
        assert (runner.Path(output_dir) / filename).is_file()

    try:
        runner.run(runner.parse_args(runner_argv + ["--cost-bps", "6"]))
    except ValueError as error:
        assert "manifest mismatch" in str(error)
    else:
        raise AssertionError("registered parameters must be immutable")

    runner.HOLDOUT_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    runner._write_json(
        runner.HOLDOUT_REGISTRY_DIR
        / f"{summary['finalHoldout']['identitySha256']}.json",
        {
            "registryVersion": runner.HOLDOUT_REGISTRY_VERSION,
            "status": "completed",
            "window": runner._holdout_window(
                summary["symbolsAvailable"],
                summary["interval"],
                summary["finalHoldout"]["startOpenTime"],
                summary["finalHoldout"]["endOpenTime"],
            ),
        },
    )
    runner.feed.load_panel = lambda *_args, **_kwargs: {
        symbol: frame.iloc[:400].copy() for symbol, frame in panel.items()
    }
    alternate_output = runner.Path(output_dir) / "alternate"
    try:
        runner.run(
            runner.parse_args(
                runner_argv + ["--output-dir", str(alternate_output)]
            )
        )
    except ValueError as error:
        assert "already consumed" in str(error)
    else:
        raise AssertionError("an overlapping holdout must remain one-shot after a panel change")

runner.feed.load_panel = lambda *_args, **_kwargs: panel
runner._bootstrap_ci = lambda *_args, **_kwargs: (1.0, 2.0)
runner._diagnostics = lambda matrix, *_args, **_kwargs: (
    {
        "deflatedSharpe": {"probability": 0.99},
        "pbo": {"probability": 0.01},
    },
    matrix.copy(),
    matrix.copy(),
)
with tempfile.TemporaryDirectory() as completed_root:
    completed_root = runner.Path(completed_root)
    completed_output = completed_root / "output"
    runner.HOLDOUT_REGISTRY_DIR = completed_root / "registry"
    completed_args = runner.parse_args([
        "AAA", "BBB", "CCC", "DDD",
        "--output-dir", str(completed_output),
        "--beta-lookback-hours", "48",
        "--feature-lookback-hours", "48",
        "--oi-change-hours", "6",
        "--initial-train", "60",
        "--outer-test-size", "20",
        "--inner-initial-train", "30",
        "--inner-test-size", "10",
        "--pbo-slices", "2",
        "--bootstrap-reps", "100",
        "--min-symbols", "2",
        "--min-oos-observations", "1",
        "--min-active-fraction", "0",
        "--min-derivatives-coverage", "0",
        "--max-fold-loss", "0.99",
        "--max-regime-loss", "0.99",
        "--min-regime-observations", "1",
        "--open-final-holdout",
    ])
    completed_summary = runner.run(completed_args)
    assert completed_summary["status"] == "final_holdout_passed"
    assert completed_summary["finalHoldout"]["status"] == "pass"
    returns_path = completed_output / "final-holdout-returns.csv"
    result_path = completed_output / "final-holdout-result.json"
    output_record_path = completed_output / "final-holdout-opened.json"
    registry_paths = list(runner.HOLDOUT_REGISTRY_DIR.glob("*.json"))
    assert returns_path.is_file()
    assert result_path.is_file()
    assert output_record_path.is_file()
    assert len(registry_paths) == 1
    registry_record = json.loads(registry_paths[0].read_text())
    output_record = json.loads(output_record_path.read_text())
    assert registry_record["status"] == "completed"
    assert output_record["status"] == "completed"
    assert registry_record["window"]["symbols"] == ["AAA", "BBB", "CCC", "DDD"]
    assert registry_record["artifacts"]["returnsSha256"] == hashlib.sha256(
        returns_path.read_bytes()
    ).hexdigest()
    assert registry_record["artifacts"]["resultSha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
`;
    const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
      encoding: "utf8",
      timeout: 60_000,
    });
    assert.equal(run.status, 0, run.stderr || run.stdout);
  },
);
