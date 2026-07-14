#!/usr/bin/env python3
"""Run the pre-registered residual-momentum edge campaign.

The final chronological holdout is reserved by default. Pass
``--open-final-holdout`` only after the research specification is frozen.
Generated artifacts go under ``.tmp/research/edge-campaign`` by default.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

import datafeed as feed
import diagnostics
from edge_campaign import CampaignConfig, campaign_specs, run_trial_matrix
import harness as H


CAMPAIGN_ID = "residual_momentum_derivatives_ablation_v1"
REGISTRATION_VERSION = 1
HOLDOUT_REGISTRY_VERSION = 3
IMPLEMENTATION_FILES = (
    "diagnostics.py",
    "edge_campaign.py",
    "harness.py",
    "run_edge_campaign.py",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_REGISTRY_DIR = Path(
    os.environ.get(
        "TRADER_EDGE_HOLDOUT_REGISTRY",
        str(REPOSITORY_ROOT / ".tmp/research/edge-campaign-holdouts"),
    )
).expanduser()

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


def _common_times(panel: Mapping[str, pd.DataFrame]) -> list[int]:
    common: set[int] | None = None
    for frame in panel.values():
        times = set(pd.to_numeric(frame["openTime"], errors="raise").astype(np.int64))
        common = times if common is None else common.intersection(times)
    return sorted(common or [])


def _truncate_panel(
    panel: Mapping[str, pd.DataFrame], cutoff_time: int
) -> dict[str, pd.DataFrame]:
    return {
        symbol: frame[pd.to_numeric(frame["openTime"]) <= cutoff_time].copy()
        for symbol, frame in panel.items()
    }


def _panel_on_times(
    panel: Mapping[str, pd.DataFrame], times: Sequence[int]
) -> dict[str, pd.DataFrame]:
    allowed = set(times)
    return {
        symbol: frame[
            pd.to_numeric(frame["openTime"], errors="raise").isin(allowed)
        ].copy()
        for symbol, frame in panel.items()
    }


def _validate_market_grid(
    panel: Mapping[str, pd.DataFrame], common_times: Sequence[int], interval_ms: int
) -> None:
    timestamps = np.asarray(common_times, dtype=np.int64)
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) == interval_ms):
        raise ValueError("aligned market data must be exactly one interval apart")
    aligned = _panel_on_times(panel, common_times)
    for symbol, frame in aligned.items():
        if len(frame) != len(common_times) or frame["openTime"].duplicated().any():
            raise ValueError(f"{symbol} does not have one row per aligned timestamp")
        close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError(f"{symbol} close prices must be finite and positive")


def _implementation_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _panel_digest(panel: Mapping[str, pd.DataFrame]) -> str:
    columns = ("openTime", "close", "funding", "oi", "basis", "taker")
    digest = hashlib.sha256()
    for symbol, frame in sorted(panel.items()):
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{symbol} is missing registered columns: {', '.join(sorted(missing))}"
            )
        canonical = frame.loc[:, columns].sort_values("openTime")
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            canonical.to_csv(
                index=False,
                lineterminator="\n",
                na_rep="NA",
                float_format="%.17g",
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _periods_per_year(interval_ms: int) -> float:
    return 365.0 * 86_400_000 / interval_ms


def _metrics(
    values: pd.Series | np.ndarray,
    periods_per_year: float,
    active: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    net = np.asarray(values, dtype=float)
    if net.ndim != 1 or len(net) == 0 or not np.isfinite(net).all():
        raise ValueError("metrics require a non-empty finite return series")
    equity = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = 1 - equity / np.maximum(peak, 1e-12)
    std = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * np.mean(net) / std) if std > 0 else 0.0
    active_observations = (
        int(np.count_nonzero(np.asarray(active, dtype=float) > 0))
        if active is not None
        else int(np.count_nonzero(np.abs(net) > 1e-15))
    )
    return {
        "observations": len(net),
        "activeObservations": active_observations,
        "totalReturn": float(equity[-1] - 1),
        "meanReturn": float(np.mean(net)),
        "annualizedSharpe": sharpe,
        "maxDrawdown": float(np.max(drawdown)),
    }


def _reprice_path(frame: pd.DataFrame, cost_per_turnover: float) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot price an empty evaluation path")
    weight_columns = sorted(
        column for column in frame.columns if str(column).startswith("weight_")
    )
    if not weight_columns:
        raise ValueError("evaluation path has no portfolio weights")
    priced = frame.copy().reset_index(drop=True)
    gross = priced["gross"].to_numpy(dtype=float)
    weights = priced[weight_columns].to_numpy(dtype=float)
    if not np.isfinite(gross).all() or not np.isfinite(weights).all():
        raise ValueError("evaluation path contains non-finite gross returns or weights")
    previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    turnover = np.abs(weights - previous).sum(axis=1)
    priced["turnover"] = turnover
    priced["net"] = gross - cost_per_turnover * turnover
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
        repriced[name] = _reprice_path(path, cost_per_turnover).set_index(
            "openTime"
        )
    matrix = pd.DataFrame(
        {name: frame["net"] for name, frame in repriced.items()}, index=index
    )
    return matrix, repriced


def _score_frame(frame: pd.DataFrame, cost_per_turnover: float) -> float:
    values = _reprice_path(frame, cost_per_turnover)["net"].to_numpy(dtype=float)
    if len(values) < 2:
        return float("-inf")
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / std) if std > 1e-15 else float("-inf")


def _nested_input(
    matrix: pd.DataFrame, details: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    frame = pd.DataFrame({"openTime": matrix.index.to_numpy()})
    candidates = {}
    for name in matrix.columns:
        detail = details[name].reindex(matrix.index)
        weight_columns = [column for column in detail if column.startswith("weight_")]
        gross_column = f"{name}__gross"
        renamed_weights = [f"{name}__{column}" for column in weight_columns]
        frame[gross_column] = detail["gross"].to_numpy(dtype=float)
        for source, target in zip(weight_columns, renamed_weights):
            frame[target] = detail[source].to_numpy(dtype=float)
        candidates[name] = {
            "grossColumn": gross_column,
            "inputWeightColumns": tuple(renamed_weights),
            "outputWeightColumns": tuple(weight_columns),
        }
    return frame, candidates


def _evaluate_nested_candidate(
    candidate: Mapping[str, object], test: pd.DataFrame
) -> pd.DataFrame:
    gross = test[str(candidate["grossColumn"])].to_numpy(dtype=float)
    input_columns = list(candidate["inputWeightColumns"])
    output_columns = list(candidate["outputWeightColumns"])
    result = pd.DataFrame({"gross": gross})
    for source, target in zip(input_columns, output_columns):
        result[str(target)] = test[str(source)].to_numpy(dtype=float)
    return result


def _run_nested_selector(
    frame: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, object]],
    sizes: Mapping[str, int],
    label_horizon: int,
    cost_per_turnover: float,
) -> H.NestedRollingResult:
    nested = H.nested_rolling_origin(
        frame,
        candidates,
        fit_candidate=lambda candidate, _train: candidate,
        evaluate_candidate=_evaluate_nested_candidate,
        score_candidate=lambda validation: _score_frame(
            validation, cost_per_turnover
        ),
        initial_train_size=sizes["initialTrain"],
        outer_test_size=sizes["outerTest"],
        inner_initial_train_size=sizes["innerInitialTrain"],
        inner_test_size=sizes["innerTest"],
        label_horizon=label_horizon,
    )
    return replace(nested, oos=_reprice_path(nested.oos, cost_per_turnover))


def _rolling_select_candidate(
    frame: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, object]],
    initial_train_size: int,
    test_size: int,
    label_horizon: int,
    cost_per_turnover: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    splits = H.rolling_origin_splits(
        len(frame), initial_train_size, test_size, label_horizon
    )
    if not splits:
        raise ValueError("not enough observations for final rolling selection")
    score_rows = []
    for name, candidate in candidates.items():
        validation_frames = []
        for split in splits:
            validation = frame.iloc[split.test_slice]
            evaluated = _evaluate_nested_candidate(candidate, validation)
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


def _evaluate_outer_choices(
    frame: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, object]],
    outer_folds: pd.DataFrame,
    cost_per_turnover: float,
) -> pd.DataFrame:
    evaluations = []
    for fold in outer_folds.to_dict("records"):
        name = str(fold["selected_candidate"])
        start = int(fold["test_start"])
        stop = int(fold["test_stop"])
        test = frame.iloc[start:stop]
        evaluated = _evaluate_nested_candidate(candidates[name], test)
        evaluated.insert(0, "selected_candidate", name)
        evaluated.insert(0, "outer_fold", int(fold["outer_fold"]))
        evaluated.insert(0, "openTime", test["openTime"].to_numpy())
        evaluated.insert(0, "row_position", np.arange(start, stop, dtype=int))
        evaluations.append(evaluated)
    combined = pd.concat(evaluations, ignore_index=True)
    positions = combined["row_position"].to_numpy(dtype=int)
    if len(positions) > 1 and not np.all(np.diff(positions) == 1):
        raise ValueError("outer evaluation folds do not form one contiguous path")
    return _reprice_path(combined, cost_per_turnover)


def _derived_nested_sizes(args: argparse.Namespace, observations: int) -> dict[str, int]:
    initial = args.initial_train or max(60, observations // 2)
    outer_test = args.outer_test_size or max(12, observations // 10)
    inner_initial = args.inner_initial_train or max(30, initial // 2)
    inner_test = args.inner_test_size or max(6, outer_test // 2)
    return {
        "initialTrain": initial,
        "outerTest": outer_test,
        "innerInitialTrain": inner_initial,
        "innerTest": inner_test,
    }


def _bootstrap_ci(
    values: pd.Series | np.ndarray,
    periods_per_year: float,
    interval_ms: int,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    block = max(2, round(86_400_000 / interval_ms))
    lo, hi = H.block_bootstrap_sharpe_ci(
        values,
        periods_per_year,
        block=block,
        n_boot=reps,
        seed=seed,
    )
    return float(lo), float(hi)


def _ci_json(interval: tuple[float, float]) -> list[float | None]:
    return [float(value) if math.isfinite(value) else None for value in interval]


def _finite_number(value: object) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    safe = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict("records")


def _fold_metrics(
    oos: pd.DataFrame, periods_per_year: float
) -> dict[str, dict[str, object]]:
    return {
        str(fold): _metrics(group["net"], periods_per_year, group["active"])
        for fold, group in oos.groupby("outer_fold", sort=True)
    }


def _market_regime_labels(
    panel: Mapping[str, pd.DataFrame], interval_ms: int
) -> pd.Series:
    times = pd.Index(_common_times(panel), name="openTime")
    closes = pd.DataFrame(
        {
            symbol: pd.to_numeric(
                frame.set_index("openTime")["close"].reindex(times), errors="coerce"
            )
            for symbol, frame in sorted(panel.items())
        },
        index=times,
    )
    market = closes.pct_change(fill_method=None).mean(axis=1, skipna=False)
    day = max(2, round(86_400_000 / interval_ms))
    week = max(day, 7 * day)
    volatility = market.rolling(day, min_periods=day).std(ddof=1)
    causal_median = volatility.expanding(min_periods=day).median().shift(1)
    trend = (1 + market).rolling(week, min_periods=week).apply(np.prod, raw=True) - 1
    labels = pd.Series(index=times, dtype=object)
    known = volatility.notna() & causal_median.notna() & trend.notna()
    volatility_label = pd.Series(
        np.where(
            volatility.loc[known] >= causal_median.loc[known],
            "high_vol",
            "low_vol",
        ),
        index=times[known],
    )
    trend_label = pd.Series(
        np.where(trend.loc[known] >= 0, "_up", "_down"),
        index=times[known],
    )
    labels.loc[known] = volatility_label + trend_label
    return labels


def _regime_report(
    oos: pd.DataFrame,
    labels: pd.Series,
    periods_per_year: float,
    min_observations: int,
    max_loss: float,
) -> tuple[dict[str, object], bool, pd.DataFrame]:
    labelled = oos.copy()
    labelled["regime"] = labelled["openTime"].map(labels)
    if labelled["regime"].isna().any():
        raise ValueError("nested OOS rows are missing causal market-regime labels")
    metrics = {
        str(regime): _metrics(group["net"], periods_per_year, group["active"])
        for regime, group in labelled.groupby("regime", sort=True)
    }
    loss_cap_passed = bool(metrics) and all(
        float(values["totalReturn"]) >= -max_loss for values in metrics.values()
    )
    observation_coverage_passed = len(metrics) >= 2 and all(
        int(values["observations"]) >= min_observations
        for values in metrics.values()
    )
    eligible = sorted(
        regime
        for regime, values in metrics.items()
        if int(values["observations"]) >= min_observations
    )
    passed = loss_cap_passed and observation_coverage_passed
    return (
        {
            "minimumObservations": min_observations,
            "maximumAllowedLoss": max_loss,
            "observedRegimes": sorted(metrics),
            "eligibleRegimes": eligible,
            "lossCapPassed": loss_cap_passed,
            "observationCoveragePassed": observation_coverage_passed,
            "metrics": metrics,
        },
        passed,
        labelled,
    )


def _diagnostics(
    matrix: pd.DataFrame,
    champion: str,
    periods_per_year: float,
    interval_ms: int,
    requested_slices: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    aggregation_bars = max(1, round(86_400_000 / interval_ms))
    aggregated = diagnostics.compound_return_matrix(matrix, aggregation_bars)
    remainder = len(aggregated.matrix) % requested_slices
    dsr_matrix = aggregated.matrix.copy()
    pbo_matrix = dsr_matrix.iloc[remainder:].copy()
    report: dict[str, object] = {
        "aggregation": aggregated.to_dict(),
        "periodsPerYear": periods_per_year / aggregation_bars,
        "requestedSlices": requested_slices,
        "pboRowsDroppedFromStart": remainder,
        "observations": len(dsr_matrix),
        "dsrObservations": len(dsr_matrix),
        "pboObservations": len(pbo_matrix),
    }
    errors = {}
    try:
        report["deflatedSharpe"] = diagnostics.deflated_sharpe_ratio(
            dsr_matrix,
            selected_trial=champion,
            periods_per_year=periods_per_year / aggregation_bars,
        ).to_dict()
    except (KeyError, TypeError, ValueError) as error:
        errors["deflatedSharpe"] = str(error)
    if len(pbo_matrix) < requested_slices:
        errors["pbo"] = "not enough aligned observations for requested CSCV slices"
    else:
        try:
            report["pbo"] = diagnostics.cscv_pbo(
                pbo_matrix, n_slices=requested_slices
            ).to_dict()
        except (TypeError, ValueError) as error:
            errors["pbo"] = str(error)
    if errors:
        report["errors"] = errors
    return report, dsr_matrix, pbo_matrix


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_csv_atomic(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdout_window(
    symbols: Sequence[str], interval: str, start_time: int, end_time: int
) -> dict[str, object]:
    interval_ms = feed.INTERVAL_MS.get(interval)
    if not symbols or interval_ms is None or start_time > end_time:
        raise ValueError("final holdout window is invalid")
    return {
        "symbols": sorted(set(symbols)),
        "interval": interval,
        "startOpenTime": int(start_time),
        "endOpenTime": int(end_time),
        "outcomeEndTimeExclusive": int(end_time) + interval_ms,
    }


def _registry_window(path: Path, record: object) -> dict[str, object]:
    try:
        if not isinstance(record, dict):
            raise TypeError
        if record.get("registryVersion") != HOLDOUT_REGISTRY_VERSION:
            raise ValueError
        if record.get("status") not in {"opening", "completed"}:
            raise ValueError
        window = record["window"]
        if not isinstance(window, dict):
            raise TypeError
        symbols = window["symbols"]
        interval = window["interval"]
        start_time = int(window["startOpenTime"])
        end_time = int(window["endOpenTime"])
        outcome_end_time = int(window["outcomeEndTimeExclusive"])
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or not isinstance(interval, str)
            or not interval
            or start_time > end_time
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"holdout registry entry {path.name} is invalid") from error
    canonical = _holdout_window(symbols, interval, start_time, end_time)
    if canonical["outcomeEndTimeExclusive"] != outcome_end_time:
        raise ValueError(f"holdout registry entry {path.name} is invalid")
    return canonical


def _windows_overlap(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    if not set(left["symbols"]).intersection(right["symbols"]):
        return False
    return int(left["startOpenTime"]) < int(
        right["outcomeEndTimeExclusive"]
    ) and int(right["startOpenTime"]) < int(left["outcomeEndTimeExclusive"])


def _assert_holdout_available(
    registry_dir: Path,
    window: Mapping[str, object],
    output_record: Path,
) -> None:
    if output_record.exists():
        raise ValueError("final holdout was already consumed for this output directory")
    if not registry_dir.exists():
        return
    for path in sorted(registry_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"holdout registry entry {path.name} is unreadable") from error
        registered_window = _registry_window(path, record)
        if _windows_overlap(window, registered_window):
            raise ValueError(
                "final holdout overlaps an already consumed symbol/time window"
            )


@contextmanager
def _holdout_registry_lock(registry_dir: Path) -> Iterator[None]:
    registry_dir.mkdir(parents=True, exist_ok=True)
    with (registry_dir / ".registry.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reserve_holdout(
    registry_dir: Path,
    marker: Path,
    window: Mapping[str, object],
    output_record: Path,
    opening_record: Mapping[str, object],
) -> None:
    with _holdout_registry_lock(registry_dir):
        _assert_holdout_available(registry_dir, window, output_record)
        _write_json_exclusive(marker, opening_record)


def _campaign_status(
    ready_for_holdout: bool, final_holdout: Mapping[str, object]
) -> str:
    holdout_status = final_holdout.get("status")
    if holdout_status == "pass":
        return "final_holdout_passed"
    if holdout_status == "fail":
        return "final_holdout_failed"
    if holdout_status != "reserved":
        raise ValueError("final holdout status is invalid")
    return "ready_for_final_holdout" if ready_for_holdout else "insufficient_evidence"


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        _write_json(output_holdout_record, opening_record)
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

    matrix.to_csv(output_dir / "trial-returns.csv", index_label="openTime")
    diagnostic_matrix.to_csv(
        output_dir / "diagnostic-trial-returns.csv", index_label="openTime"
    )
    pbo_matrix.to_csv(
        output_dir / "pbo-trial-returns.csv", index_label="openTime"
    )
    labelled_nested_oos.to_csv(output_dir / "nested-oos.csv", index=False)
    nested.outer_folds.to_csv(output_dir / "outer-folds.csv", index=False)
    nested.inner_scores.to_csv(output_dir / "inner-scores.csv", index=False)
    final_selection_scores.to_csv(
        output_dir / "final-selection-scores.csv", index=False
    )
    final_selection_folds.to_csv(
        output_dir / "final-selection-folds.csv", index=False
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
    trial_paths.to_csv(output_dir / "trial-paths.csv", index=False)
    for label, path in stress_paths.items():
        path.to_csv(output_dir / f"stress-{label}-nested-oos.csv", index=False)
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
