#!/usr/bin/env python3
"""Run the pre-registered residual-momentum edge campaign.

The final chronological holdout is reserved by default. Pass
``--open-final-holdout`` only after the research specification is frozen.
Generated artifacts go under ``.tmp/research/edge-campaign`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import datafeed as feed
from campaign_runner import (
    HOLDOUT_REGISTRY_DIR,
    HOLDOUT_REGISTRY_VERSION,
    REPOSITORY_ROOT,
    _assert_holdout_available,
    _assert_output_holdout_not_consumed,
    _bootstrap_ci,
    _campaign_output_lock,
    _campaign_status,
    _ci_json,
    _common_times,
    _derived_nested_sizes,
    _diagnostics,
    _evaluate_nested_candidate,
    _evaluate_outer_choices,
    _file_digest,
    _finite_number,
    _fold_metrics,
    _holdout_registry_lock,
    _holdout_window,
    _json_digest,
    _json_records,
    _market_regime_labels,
    _metrics,
    _nested_input,
    _panel_on_times,
    _periods_per_year,
    _registry_window,
    _regime_report,
    _reprice_details,
    _reprice_path,
    _reserve_holdout,
    _rolling_select_candidate,
    _run_nested_selector,
    _score_frame,
    _truncate_panel,
    _validate_market_grid,
    _windows_overlap,
    _write_csv_atomic,
    _write_json,
    _write_json_exclusive,
)
from campaign_runner import _implementation_digest as _digest_implementation_files
from campaign_runner import _panel_digest as _digest_panel
from edge_campaign import CampaignConfig, campaign_specs, run_trial_matrix


CAMPAIGN_ID = "residual_momentum_derivatives_ablation_v1"
REGISTRATION_VERSION = 1
IMPLEMENTATION_FILES = (
    "campaign_runner.py",
    "datafeed.py",
    "diagnostics.py",
    "edge_campaign.py",
    "harness.py",
    "run_edge_campaign.py",
)
REGISTERED_PANEL_COLUMNS = ("openTime", "close", "funding", "oi", "basis", "taker")

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 15 causal residual-momentum ablations with nested validation"
    )
    parser.add_argument("symbols", nargs="*", help="Binance futures symbols")
    parser.add_argument("--interval", choices=sorted(feed.INTERVAL_MS), default="1h")
    parser.add_argument("--refresh", action="store_true", help="Refresh the local cache before running")
    parser.add_argument(
        "--output-dir",
        default=".tmp/research/edge-campaign",
        help="Generated report directory",
    )
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost per unit turnover")
    parser.add_argument("--rebalance-hours", type=float, default=8.0)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--beta-lookback-hours", type=float, default=168.0)
    parser.add_argument("--feature-lookback-hours", type=float, default=168.0)
    parser.add_argument("--oi-change-hours", type=float, default=24.0)
    parser.add_argument("--crowding-z", type=float, default=2.0)
    parser.add_argument("--initial-train", type=int, default=0, help="Outer train rows; 0 derives it")
    parser.add_argument("--outer-test-size", type=int, default=0, help="Outer test rows; 0 derives it")
    parser.add_argument("--inner-initial-train", type=int, default=0)
    parser.add_argument("--inner-test-size", type=int, default=0)
    parser.add_argument("--label-horizon", type=int, default=1)
    parser.add_argument("--final-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--open-final-holdout", action="store_true")
    parser.add_argument("--pbo-slices", type=int, default=10)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--min-symbols", type=int, default=5)
    parser.add_argument("--min-oos-observations", type=int, default=1500)
    parser.add_argument("--min-active-fraction", type=float, default=0.25)
    parser.add_argument("--min-derivatives-coverage", type=float, default=0.80)
    parser.add_argument("--max-fold-loss", type=float, default=0.05)
    parser.add_argument("--max-regime-loss", type=float, default=0.05)
    parser.add_argument("--min-regime-observations", type=int, default=50)
    return parser.parse_args(argv)


def _positive_bars(hours: float, interval_ms: int, name: str) -> int:
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return max(1, round(hours * 3_600_000 / interval_ms))


def validate_args(args: argparse.Namespace) -> None:
    finite_non_negative = {
        "--cost-bps": args.cost_bps,
        "--crowding-z": args.crowding_z,
    }
    for name, value in finite_non_negative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0")
    _positive_bars(args.rebalance_hours, feed.INTERVAL_MS[args.interval], "--rebalance-hours")
    _positive_bars(args.beta_lookback_hours, feed.INTERVAL_MS[args.interval], "--beta-lookback-hours")
    _positive_bars(args.feature_lookback_hours, feed.INTERVAL_MS[args.interval], "--feature-lookback-hours")
    _positive_bars(args.oi_change_hours, feed.INTERVAL_MS[args.interval], "--oi-change-hours")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.label_horizon < 1:
        raise ValueError("--label-horizon must be >= 1 for one-bar forward returns")
    if not 0.05 <= args.final_holdout_fraction <= 0.50:
        raise ValueError("--final-holdout-fraction must be between 0.05 and 0.50")
    if args.pbo_slices < 2 or args.pbo_slices % 2:
        raise ValueError("--pbo-slices must be an even integer >= 2")
    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be >= 100")
    if args.min_symbols < 2 or args.min_oos_observations < 1:
        raise ValueError("minimum symbols must be >= 2 and minimum observations >= 1")
    unit_interval = {
        "--min-active-fraction": args.min_active_fraction,
        "--min-derivatives-coverage": args.min_derivatives_coverage,
    }
    for name, value in unit_interval.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    for name, value in {
        "--max-fold-loss": args.max_fold_loss,
        "--max-regime-loss": args.max_regime_loss,
    }.items():
        if not math.isfinite(value) or value < 0 or value >= 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if args.min_regime_observations < 1:
        raise ValueError("--min-regime-observations must be >= 1")
    for name in ("initial_train", "outer_test_size", "inner_initial_train", "inner_test_size"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def _implementation_digest() -> str:
    return _digest_implementation_files(
        Path(__file__).resolve().parent, IMPLEMENTATION_FILES
    )


def _panel_digest(panel: Mapping[str, pd.DataFrame]) -> str:
    return _digest_panel(panel, REGISTERED_PANEL_COLUMNS)


def _registration_parameters(
    args: argparse.Namespace,
    symbols: Sequence[str],
    config: CampaignConfig,
) -> dict[str, object]:
    return {
        "symbolsRequested": list(symbols),
        "interval": args.interval,
        "strategy": {
            "costBps": args.cost_bps,
            "rebalanceHours": args.rebalance_hours,
            "topN": args.top_n,
            "grossExposure": config.gross_exposure,
            "betaLookbackHours": args.beta_lookback_hours,
            "featureLookbackHours": args.feature_lookback_hours,
            "oiChangeHours": args.oi_change_hours,
            "crowdingZ": args.crowding_z,
            "signalDelayBars": config.signal_delay_bars,
        },
        "validation": {
            "initialTrain": args.initial_train,
            "outerTestSize": args.outer_test_size,
            "innerInitialTrain": args.inner_initial_train,
            "innerTestSize": args.inner_test_size,
            "labelHorizon": args.label_horizon,
            "finalHoldoutFraction": args.final_holdout_fraction,
            "pboSlices": args.pbo_slices,
            "bootstrapReps": args.bootstrap_reps,
            "bootstrapSeed": args.bootstrap_seed,
        },
        "promotion": {
            "minSymbols": args.min_symbols,
            "minOosObservations": args.min_oos_observations,
            "minActiveFraction": args.min_active_fraction,
            "minDerivativesCoverage": args.min_derivatives_coverage,
            "maxFoldLoss": args.max_fold_loss,
            "maxRegimeLoss": args.max_regime_loss,
            "minRegimeObservations": args.min_regime_observations,
            "deflatedSharpeProbability": 0.95,
            "maxPbo": 0.20,
        },
    }


def _registered_inputs(
    output_dir: Path,
    panel: Mapping[str, pd.DataFrame],
    common_times: Sequence[int],
    parameters: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, pd.DataFrame], list[int], str]:
    manifest_path = output_dir / "campaign-manifest.json"
    implementation_sha = _implementation_digest()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("campaign manifest is unreadable") from error
        if not isinstance(manifest, dict):
            raise ValueError("campaign manifest must contain a JSON object")
        expected = {
            "campaign": CAMPAIGN_ID,
            "registrationVersion": REGISTRATION_VERSION,
            "parameters": dict(parameters),
            "symbolsAvailable": sorted(panel),
            "implementationSha256": implementation_sha,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"campaign manifest mismatch for {name}; use a new output directory"
                )
        registered = manifest.get("registeredData")
        if not isinstance(registered, dict):
            raise ValueError("campaign manifest has no registeredData object")
        try:
            start_time = int(registered["startOpenTime"])
            end_time = int(registered["endOpenTime"])
            registered_rows = int(registered["rows"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("campaign manifest registeredData is invalid") from error
        registered_times = [
            timestamp
            for timestamp in common_times
            if start_time <= timestamp <= end_time
        ]
        if (
            len(registered_times) != registered_rows
            or not registered_times
            or registered_times[0] != start_time
            or registered_times[-1] != end_time
        ):
            raise ValueError("registered market-data time grid changed")
        registered_panel = _panel_on_times(panel, registered_times)
        if _panel_digest(registered_panel) != registered.get("panelSha256"):
            raise ValueError("registered market-data values changed")
        return manifest, registered_panel, registered_times, _json_digest(manifest)

    if (output_dir / "summary.json").exists():
        raise ValueError(
            "output directory has campaign artifacts but no manifest; use a new directory"
        )
    development_count = max(
        3,
        math.floor(
            len(common_times)
            * (1 - float(parameters["validation"]["finalHoldoutFraction"]))
        ),
    )
    if development_count >= len(common_times):
        raise ValueError("registered data leaves no final holdout")
    registered_panel = _panel_on_times(panel, common_times)
    manifest = {
        "campaign": CAMPAIGN_ID,
        "registrationVersion": REGISTRATION_VERSION,
        "parameters": dict(parameters),
        "symbolsAvailable": sorted(panel),
        "implementationSha256": implementation_sha,
        "trials": [
            specification.to_dict()
            for specification in campaign_specs(feed.INTERVAL_MS[str(parameters["interval"])])
        ],
        "registeredData": {
            "startOpenTime": int(common_times[0]),
            "endOpenTime": int(common_times[-1]),
            "rows": len(common_times),
            "developmentRows": development_count,
            "developmentCutoffOpenTime": int(common_times[development_count - 1]),
            "holdoutStartOpenTime": int(common_times[development_count]),
            "holdoutBars": len(common_times) - development_count,
            "holdoutReturnRows": len(common_times) - development_count - 1,
            "panelSha256": _panel_digest(registered_panel),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest, registered_panel, list(common_times), _json_digest(manifest)


def _coverage(panel: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    fields = ("close", "funding", "oi", "basis", "taker")
    return {
        symbol: {
            field: int(
                np.isfinite(
                    pd.to_numeric(frame[field], errors="coerce").to_numpy(
                        dtype=float
                    )
                ).sum()
            )
            for field in fields
        }
        for symbol, frame in sorted(panel.items())
    }


def _minimum_joint_derivatives_coverage(
    panel: Mapping[str, pd.DataFrame],
) -> float:
    ratios = []
    for _, frame in sorted(panel.items()):
        if frame.empty:
            ratios.append(0.0)
            continue
        values = np.column_stack(
            [
                pd.to_numeric(frame[field], errors="coerce").to_numpy(
                    dtype=float
                )
                for field in ("funding", "oi", "basis", "taker")
            ]
        )
        ratios.append(float(np.isfinite(values).all(axis=1).mean()))
    return float(min(ratios)) if ratios else 0.0


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    output_dir = Path(args.output_dir)
    with _campaign_output_lock(output_dir):
        return _run_locked(args, output_dir)


def _run_locked(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, object]:
    symbols = list(args.symbols or DEFAULT_SYMBOLS)
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    interval_ms = feed.INTERVAL_MS[args.interval]
    config = CampaignConfig(
        interval_ms=interval_ms,
        beta_lookback_bars=_positive_bars(
            args.beta_lookback_hours, interval_ms, "--beta-lookback-hours"
        ),
        feature_lookback_bars=_positive_bars(
            args.feature_lookback_hours, interval_ms, "--feature-lookback-hours"
        ),
        oi_change_bars=_positive_bars(
            args.oi_change_hours, interval_ms, "--oi-change-hours"
        ),
        funding_basis_crowding_z=args.crowding_z,
        rebalance_bars=_positive_bars(
            args.rebalance_hours, interval_ms, "--rebalance-hours"
        ),
        top_n=args.top_n,
        cost_per_turnover=args.cost_bps / 10_000,
        # Signals use close-derived inputs, so activation starts one full bar later.
        signal_delay_bars=1,
    )
    panel = feed.load_panel(symbols, args.interval, refresh=args.refresh)
    if len(panel) < 2:
        raise ValueError("fewer than two requested symbols have cached data")
    if 2 * args.top_n > len(panel):
        raise ValueError("--top-n requires at least twice as many available symbols")
    common_times = _common_times(panel)
    if len(common_times) < 100:
        raise ValueError("fewer than 100 aligned panel rows are available")
    _validate_market_grid(panel, common_times, interval_ms)

    _assert_output_holdout_not_consumed(HOLDOUT_REGISTRY_DIR, output_dir)
    parameters = _registration_parameters(args, symbols, config)
    manifest, registered_panel, registered_times, registration_sha = _registered_inputs(
        output_dir, panel, common_times, parameters
    )
    registered_data = manifest["registeredData"]
    development_count = int(registered_data["developmentRows"])
    cutoff_time = int(registered_data["developmentCutoffOpenTime"])
    holdout_start_time = int(registered_data["holdoutStartOpenTime"])
    holdout_end_time = int(registered_data["endOpenTime"])
    holdout_window = _holdout_window(
        sorted(registered_panel), args.interval, holdout_start_time, holdout_end_time
    )
    holdout_identity = _json_digest(
        {
            "campaign": CAMPAIGN_ID,
            "panelSha256": registered_data["panelSha256"],
            "window": holdout_window,
        }
    )
    holdout_marker = HOLDOUT_REGISTRY_DIR / f"{holdout_identity}.json"
    output_holdout_record = output_dir / "final-holdout-opened.json"
    if args.open_final_holdout:
        _assert_holdout_available(
            HOLDOUT_REGISTRY_DIR, holdout_window, output_holdout_record
        )
    development_panel = _truncate_panel(registered_panel, cutoff_time)
    development_coverage = _coverage(development_panel)
    minimum_joint_derivatives_coverage = _minimum_joint_derivatives_coverage(
        development_panel
    )

    matrix_raw, details_raw, specs = run_trial_matrix(development_panel, config)
    warmup = max(
        config.beta_lookback_bars,
        config.feature_lookback_bars,
        config.oi_change_bars,
        max(spec.horizon_bars for spec in specs),
    )
    if len(matrix_raw) <= warmup + 20:
        raise ValueError("development window is too short after causal feature warmup")
    evaluation_index = matrix_raw.index[warmup:]
    matrix, details = _reprice_details(
        details_raw, evaluation_index, config.cost_per_turnover
    )
    periods_per_year = _periods_per_year(interval_ms)

    trial_metrics = {
        name: _metrics(
            matrix[name], periods_per_year, details[name]["active"]
        )
        for name in matrix.columns
    }
    nested_frame, candidates = _nested_input(matrix, details)
    sizes = _derived_nested_sizes(args, len(nested_frame))
    nested = _run_nested_selector(
        nested_frame,
        candidates,
        sizes,
        args.label_horizon,
        config.cost_per_turnover,
    )
    champion, final_selection_scores, final_selection_folds = (
        _rolling_select_candidate(
            nested_frame,
            candidates,
            sizes["innerInitialTrain"],
            sizes["innerTest"],
            args.label_horizon,
            config.cost_per_turnover,
        )
    )
    nested_metrics = _metrics(
        nested.oos["net"], periods_per_year, nested.oos["active"]
    )
    nested_ci = _bootstrap_ci(
        nested.oos["net"],
        periods_per_year,
        interval_ms,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    fold_metrics = _fold_metrics(nested.oos, periods_per_year)
    worst_fold_return = min(
        float(metrics["totalReturn"]) for metrics in fold_metrics.values()
    )
    regime_report, _, labelled_nested_oos = _regime_report(
        nested.oos,
        _market_regime_labels(development_panel, interval_ms),
        periods_per_year,
        args.min_regime_observations,
        args.max_regime_loss,
    )
    selection_diagnostics, diagnostic_matrix, pbo_matrix = _diagnostics(
        matrix, champion, periods_per_year, interval_ms, args.pbo_slices
    )

    stress_results: dict[str, object] = {}
    stress_paths: dict[str, pd.DataFrame] = {}
    stress_intervals: dict[str, tuple[float, float]] = {}
    for label, stress_config in {
        "cost1_5x": replace(config, cost_per_turnover=config.cost_per_turnover * 1.5),
        "cost2x": replace(config, cost_per_turnover=config.cost_per_turnover * 2.0),
        "additionalDelay1bar": replace(
            config, signal_delay_bars=config.signal_delay_bars + 1
        ),
    }.items():
        _, stress_details_raw, _ = run_trial_matrix(
            development_panel, stress_config
        )
        stress_matrix, stress_details = _reprice_details(
            stress_details_raw, matrix.index, stress_config.cost_per_turnover
        )
        stress_frame, stress_candidates = _nested_input(
            stress_matrix, stress_details
        )
        stress_oos = _evaluate_outer_choices(
            stress_frame,
            stress_candidates,
            nested.outer_folds,
            stress_config.cost_per_turnover,
        )
        stress_ci = _bootstrap_ci(
            stress_oos["net"],
            periods_per_year,
            interval_ms,
            args.bootstrap_reps,
            args.bootstrap_seed,
        )
        stress_intervals[label] = stress_ci
        stress_paths[label] = stress_oos
        stress_results[label] = {
            "championDevelopment": _metrics(
                stress_matrix[champion],
                periods_per_year,
                stress_details[champion]["active"],
            ),
            "nestedOuterOos": {
                "metrics": _metrics(
                    stress_oos["net"], periods_per_year, stress_oos["active"]
                ),
                "sharpeBootstrap95": _ci_json(stress_ci),
            },
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
    gates = {
        "symbolCount": len(registered_panel) >= args.min_symbols,
        "derivativesJointCoverage": minimum_joint_derivatives_coverage
        >= args.min_derivatives_coverage,
        "outerOosObservations": int(nested_metrics["observations"])
        >= args.min_oos_observations,
        "outerOosActiveFraction": active_fraction >= args.min_active_fraction,
        "outerOosSharpeCiAboveZero": math.isfinite(nested_ci[0]) and nested_ci[0] > 0,
        "worstOuterFoldLoss": worst_fold_return >= -args.max_fold_loss,
        "regimeLoss": bool(regime_report["lossCapPassed"]),
        "regimeCoverage": bool(regime_report["observationCoveragePassed"]),
        "deflatedSharpeProbability": float(dsr_probability) >= 0.95,
        "pbo": float(pbo_probability) <= 0.20,
        "cost2xOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["cost2x"][0]
        )
        and stress_intervals["cost2x"][0] > 0,
        "additionalDelayOuterOosSharpeCiAboveZero": math.isfinite(
            stress_intervals["additionalDelay1bar"][0]
        )
        and stress_intervals["additionalDelay1bar"][0] > 0,
    }
    ready_for_holdout = all(gates.values())

    final_holdout: dict[str, object] = {
        "status": "reserved",
        "identitySha256": holdout_identity,
        "openRequested": args.open_final_holdout,
        "startOpenTime": int(holdout_start_time),
        "endOpenTime": int(holdout_end_time),
        "outcomeEndTimeExclusive": int(holdout_window["outcomeEndTimeExclusive"]),
        "rows": int(registered_data["holdoutReturnRows"]),
    }
    holdout_completion_record: dict[str, object] | None = None
    evaluated_holdout: pd.DataFrame | None = None
    if args.open_final_holdout and not ready_for_holdout:
        final_holdout["openBlockedBy"] = [
            name for name, passed in gates.items() if not passed
        ]
    elif args.open_final_holdout:
        holdout_returns_path = output_dir / "final-holdout-returns.csv"
        holdout_result_path = output_dir / "final-holdout-result.json"
        opening_record = {
            "registryVersion": HOLDOUT_REGISTRY_VERSION,
            "status": "opening",
            "campaign": CAMPAIGN_ID,
            "registrationSha256": registration_sha,
            "holdoutIdentitySha256": holdout_identity,
            "candidate": champion,
            "window": holdout_window,
            "artifacts": {
                "outputDirectory": str(output_dir.resolve()),
                "returns": str(holdout_returns_path.resolve()),
                "result": str(holdout_result_path.resolve()),
            },
        }
        _reserve_holdout(
            HOLDOUT_REGISTRY_DIR,
            holdout_marker,
            holdout_window,
            output_holdout_record,
            opening_record,
        )
        full_matrix_raw, full_details, _ = run_trial_matrix(
            registered_panel, config
        )
        full_frame, full_candidates = _nested_input(
            full_matrix_raw, full_details
        )
        holdout_frame = full_frame[
            (full_frame["openTime"] >= holdout_start_time)
            & (full_frame["openTime"] <= holdout_end_time)
        ]
        evaluated_holdout = _evaluate_nested_candidate(
            full_candidates[champion], holdout_frame
        )
        evaluated_holdout.insert(
            0, "openTime", holdout_frame["openTime"].to_numpy()
        )
        evaluated_holdout = _reprice_path(
            evaluated_holdout, config.cost_per_turnover
        )
        holdout_metrics = _metrics(
            evaluated_holdout["net"],
            periods_per_year,
            evaluated_holdout["active"],
        )
        holdout_ci = _bootstrap_ci(
            evaluated_holdout["net"],
            periods_per_year,
            interval_ms,
            args.bootstrap_reps,
            args.bootstrap_seed,
        )
        final_holdout = {
            "status": "pass"
            if ready_for_holdout and math.isfinite(holdout_ci[0]) and holdout_ci[0] > 0
            else "fail",
            "openRequested": True,
            "identitySha256": holdout_identity,
            "startOpenTime": int(holdout_start_time),
            "endOpenTime": int(holdout_end_time),
            "outcomeEndTimeExclusive": int(
                holdout_window["outcomeEndTimeExclusive"]
            ),
            "rows": len(evaluated_holdout),
            "metrics": holdout_metrics,
            "sharpeBootstrap95": _ci_json(holdout_ci),
        }
        _write_csv_atomic(evaluated_holdout, holdout_returns_path, index=False)
        returns_sha = _file_digest(holdout_returns_path)
        final_holdout["evidence"] = {
            "returns": str(holdout_returns_path.resolve()),
            "returnsSha256": returns_sha,
        }
        holdout_result_record = {
            **opening_record,
            "status": "evaluated",
            "result": final_holdout,
            "artifacts": {
                **opening_record["artifacts"],
                "returnsSha256": returns_sha,
            },
        }
        _write_json(holdout_result_path, holdout_result_record)
        holdout_completion_record = {
            **opening_record,
            "status": "completed",
            "result": final_holdout,
            "artifacts": {
                **holdout_result_record["artifacts"],
                "resultSha256": _file_digest(holdout_result_path),
            },
        }

    summary = {
        "campaign": CAMPAIGN_ID,
        "registrationSha256": registration_sha,
        "status": _campaign_status(ready_for_holdout, final_holdout),
        "symbolsRequested": symbols,
        "symbolsAvailable": sorted(registered_panel),
        "interval": args.interval,
        "configuration": {
            "registeredParameters": parameters,
            "derived": {
                "rebalanceBars": config.rebalance_bars,
                "betaLookbackBars": config.beta_lookback_bars,
                "featureLookbackBars": config.feature_lookback_bars,
                "oiChangeBars": config.oi_change_bars,
                "signalDelayBars": config.signal_delay_bars,
            },
            "nestedSizes": sizes,
        },
        "data": {
            "registeredRows": len(registered_times),
            "developmentRows": development_count,
            "featureWarmupRows": warmup,
            "trialReturnRows": len(matrix),
            "coverage": development_coverage,
            "minimumJointDerivativesCoverage": minimum_joint_derivatives_coverage,
            "panelSha256": registered_data["panelSha256"],
        },
        "trials": [spec.to_dict() for spec in specs],
        "champion": champion,
        "finalSelection": {
            "rule": "expanding_rolling_origin_inner_oos_sharpe",
            "scores": _json_records(final_selection_scores),
            "folds": _json_records(final_selection_folds),
        },
        "championDevelopmentMetrics": trial_metrics[champion],
        "nestedOuterOos": {
            "metrics": nested_metrics,
            "activeFraction": active_fraction,
            "sharpeBootstrap95": _ci_json(nested_ci),
            "foldMetrics": fold_metrics,
            "regimes": regime_report,
        },
        "selectionDiagnostics": selection_diagnostics,
        "stress": stress_results,
        "promotionGates": gates,
        "finalHoldout": final_holdout,
    }

    _write_csv_atomic(
        matrix, output_dir / "trial-returns.csv", index_label="openTime"
    )
    _write_csv_atomic(
        diagnostic_matrix,
        output_dir / "diagnostic-trial-returns.csv",
        index_label="openTime",
    )
    _write_csv_atomic(
        pbo_matrix, output_dir / "pbo-trial-returns.csv", index_label="openTime"
    )
    _write_csv_atomic(
        labelled_nested_oos, output_dir / "nested-oos.csv", index=False
    )
    _write_csv_atomic(
        nested.outer_folds, output_dir / "outer-folds.csv", index=False
    )
    _write_csv_atomic(
        nested.inner_scores, output_dir / "inner-scores.csv", index=False
    )
    _write_csv_atomic(
        final_selection_scores,
        output_dir / "final-selection-scores.csv",
        index=False,
    )
    _write_csv_atomic(
        final_selection_folds,
        output_dir / "final-selection-folds.csv",
        index=False,
    )
    trial_paths = pd.concat(
        [
            frame.reindex(matrix.index)
            .reset_index()
            .assign(trial=name)
            for name, frame in details.items()
        ],
        ignore_index=True,
    )
    _write_csv_atomic(trial_paths, output_dir / "trial-paths.csv", index=False)
    for label, path in stress_paths.items():
        _write_csv_atomic(
            path, output_dir / f"stress-{label}-nested-oos.csv", index=False
        )
    _write_json(
        output_dir / "trial-ledger.json",
        {
            "campaign": summary["campaign"],
            "trialCount": len(specs),
            "trials": [
                {
                    "specification": spec.to_dict(),
                    "metrics": trial_metrics[spec.name],
                    "finalSelectionScore": _finite_number(
                        final_selection_scores.loc[
                            final_selection_scores["candidate"] == spec.name,
                            "score",
                        ].iloc[0]
                    ),
                }
                for spec in specs
            ],
        },
    )
    _write_json(output_dir / "summary.json", summary)
    if holdout_completion_record is not None:
        _write_json(output_holdout_record, holdout_completion_record)
        _write_json(holdout_marker, holdout_completion_record)
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except (OSError, TypeError, ValueError) as error:
        print(f"edge campaign failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
