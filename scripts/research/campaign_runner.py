"""Shared validation, evidence, and holdout machinery for research campaigns.

Campaign-specific runners own their trial grid, feature construction, input
registration, and promotion gates. This module owns the evaluation mechanics
that must remain identical across campaigns, especially the overlap-aware
one-shot final-holdout registry.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

import diagnostics
import harness as H


HOLDOUT_REGISTRY_VERSION = 3
HOLDOUT_REGISTRY_LOCK_TIMEOUT_SECONDS = 30.0
CAMPAIGN_OUTPUT_LOCK_TIMEOUT_SECONDS = 3600.0
LEGACY_HOLDOUT_PANEL_FIELDS = {
    "residual_momentum_derivatives_ablation_v1": "panelSha256",
    "residual_momentum_funding_only_v1": "panelSha256",
    "residual_reversal_turnover_v1": "fullPanelDigestSha256",
}
LEGACY_HOLDOUT_IDENTITY_CAMPAIGNS = frozenset(LEGACY_HOLDOUT_PANEL_FIELDS)
LEGACY_HOLDOUT_MANIFEST_DIGEST_MODES = {
    "residual_momentum_funding_only_v1": "canonical-json",
    "residual_reversal_turnover_v1": "raw-bytes",
}
STRICT_HOLDOUT_MANIFEST_PANEL_FIELDS = (
    "panelSha256",
    "fullPanelDigestSha256",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHARED_HOLDOUT_REGISTRY_RELATIVE = Path(
    ".tmp/research/edge-campaign-holdouts"
)


def _git_worktree_roots(repository_root: Path) -> tuple[Path, ...]:
    """Return Git's canonical and linked worktrees, including stale paths."""
    command = [
        "git",
        "--no-optional-locks",
        "-C",
        str(repository_root.resolve()),
    ]
    try:
        top_level = subprocess.run(
            [*command, "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
        worktrees = subprocess.run(
            [*command, "worktree", "list", "--porcelain", "-z"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("unable to resolve Git worktree metadata") from error
    if top_level.returncode != 0 or worktrees.returncode != 0:
        raise ValueError("unable to resolve Git worktree metadata")
    current_root = Path(os.fsdecode(top_level.stdout).strip()).resolve()
    roots = tuple(
        Path(os.fsdecode(field.removeprefix(b"worktree "))).resolve()
        for field in worktrees.stdout.split(b"\0")
        if field.startswith(b"worktree ")
    )
    if not roots or current_root != repository_root.resolve():
        raise ValueError("Git worktree metadata does not include this checkout")
    if current_root not in roots:
        # Git reports the metadata directory as the primary worktree for a
        # separate-git-dir checkout. It cannot identify that primary checkout
        # clone-wide once linked worktrees exist, so strict sharing is unsafe.
        if len(roots) != 1:
            raise ValueError(
                "separate-git-dir with linked worktrees is unsupported"
            )
        return (current_root,)
    canonical_git_path = roots[0] / ".git"
    if len(roots) > 1 and not canonical_git_path.is_dir():
        raise ValueError("separate-git-dir with linked worktrees is unsupported")
    if not canonical_git_path.exists():
        raise ValueError("canonical Git worktree is not a non-bare checkout")
    return roots


def _shared_repository_root(repository_root: Path) -> Path:
    """Resolve the checkout shared by every linked worktree in one clone."""
    return _git_worktree_roots(repository_root)[0]


def _git_common_directory(repository_root: Path) -> Path:
    """Return the absolute Git common directory for this clone."""
    command = [
        "git",
        "--no-optional-locks",
        "-C",
        str(repository_root.resolve()),
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("unable to resolve Git common directory") from error
    if result.returncode != 0:
        raise ValueError("unable to resolve Git common directory")
    common_directory = Path(os.fsdecode(result.stdout).strip()).resolve()
    if not common_directory.is_dir():
        raise ValueError("Git common directory is not a directory")
    return common_directory


def _configured_shared_holdout_registry(
    shared_root: Path, environment: Mapping[str, str] | None = None
) -> Path:
    # Production imports always use the clone-canonical registry. Tests may
    # inject an explicit mapping to exercise override rejection paths.
    source: Mapping[str, str] = {} if environment is None else environment
    override = source.get("TRADER_EDGE_HOLDOUT_REGISTRY")
    if override is None:
        return (shared_root / SHARED_HOLDOUT_REGISTRY_RELATIVE).resolve()
    configured = Path(override).expanduser()
    if not configured.is_absolute():
        raise ValueError("TRADER_EDGE_HOLDOUT_REGISTRY must be absolute")
    return configured.resolve()


def _assert_shared_registry_reconciled(
    repository_root: Path,
    shared_root: Path,
    shared_registry: Path,
) -> None:
    """Reject hidden markers in every active worktree before shared use."""
    canonical_registry = shared_registry.resolve()
    roots = _git_worktree_roots(repository_root)
    if roots[0] != shared_root.resolve():
        raise ValueError("shared repository root changed during reconciliation")
    for root in roots:
        local_registry = (root / SHARED_HOLDOUT_REGISTRY_RELATIVE).resolve()
        if local_registry == canonical_registry:
            continue
        if local_registry.is_dir() and any(local_registry.glob("*.json")):
            raise ValueError(
                "legacy worktree-local holdout markers require reconciliation: "
                f"{local_registry}"
            )


try:
    SHARED_REPOSITORY_ROOT = _shared_repository_root(REPOSITORY_ROOT)
    GIT_COMMON_DIR = _git_common_directory(REPOSITORY_ROOT)
    SHARED_REPOSITORY_RESOLUTION_ERROR: str | None = None
except ValueError as error:
    # Legacy research imports retain their former local behavior outside Git;
    # strict campaigns must reject this fallback before opening a holdout.
    SHARED_REPOSITORY_ROOT = REPOSITORY_ROOT
    GIT_COMMON_DIR = (REPOSITORY_ROOT / ".git").resolve()
    SHARED_REPOSITORY_RESOLUTION_ERROR = str(error)
CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR = (
    SHARED_REPOSITORY_ROOT / SHARED_HOLDOUT_REGISTRY_RELATIVE
).resolve()
SHARED_HOLDOUT_REGISTRY_DIR = _configured_shared_holdout_registry(
    SHARED_REPOSITORY_ROOT
)
HOLDOUT_REGISTRY_DIR = SHARED_HOLDOUT_REGISTRY_DIR

# Registry entries need enough interval metadata to compare outcome windows.
# Keep this independent of a campaign's acquisition feed; 8h is used by the
# bounded historical-funding campaign but not by the rolling stats collector.
HOLDOUT_INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


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


def _implementation_digest(
    root: Path, implementation_files: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for name in implementation_files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _panel_digest(
    panel: Mapping[str, pd.DataFrame], columns: Sequence[str]
) -> str:
    canonical_columns = tuple(columns)
    if not canonical_columns or len(set(canonical_columns)) != len(canonical_columns):
        raise ValueError("registered panel columns must be non-empty and unique")
    digest = hashlib.sha256()
    for symbol, frame in sorted(panel.items()):
        missing = set(canonical_columns).difference(frame.columns)
        if missing:
            raise ValueError(
                f"{symbol} is missing registered columns: {', '.join(sorted(missing))}"
            )
        canonical = frame.loc[:, canonical_columns].sort_values("openTime")
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
    columns = {"openTime": matrix.index.to_numpy()}
    candidates = {}
    for name in matrix.columns:
        detail = details[name].reindex(matrix.index)
        weight_columns = [column for column in detail if column.startswith("weight_")]
        gross_column = f"{name}__gross"
        detail_columns = [
            column
            for column in ("priceGross", "fundingCashflow")
            if column in detail.columns
        ]
        renamed_details = [f"{name}__{column}" for column in detail_columns]
        renamed_weights = [f"{name}__{column}" for column in weight_columns]
        columns[gross_column] = detail["gross"].to_numpy(dtype=float)
        for source, target in zip(detail_columns, renamed_details):
            columns[target] = detail[source].to_numpy(dtype=float)
        for source, target in zip(weight_columns, renamed_weights):
            columns[target] = detail[source].to_numpy(dtype=float)
        candidate = {
            "grossColumn": gross_column,
            "inputWeightColumns": tuple(renamed_weights),
            "outputWeightColumns": tuple(weight_columns),
        }
        if detail_columns:
            candidate["inputDetailColumns"] = tuple(renamed_details)
            candidate["outputDetailColumns"] = tuple(detail_columns)
        candidates[name] = candidate
    return pd.DataFrame(columns), candidates


def _evaluate_nested_candidate(
    candidate: Mapping[str, object], test: pd.DataFrame
) -> pd.DataFrame:
    gross = test[str(candidate["grossColumn"])].to_numpy(dtype=float)
    input_columns = list(candidate["inputWeightColumns"])
    output_columns = list(candidate["outputWeightColumns"])
    result = pd.DataFrame({"gross": gross})
    for source, target in zip(
        candidate.get("inputDetailColumns", ()),
        candidate.get("outputDetailColumns", ()),
    ):
        result[str(target)] = test[str(source)].to_numpy(dtype=float)
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


def _derived_nested_sizes(args: Any, observations: int) -> dict[str, int]:
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
    alpha: float = 0.05,
) -> tuple[float, float]:
    block = max(2, round(86_400_000 / interval_ms))
    lo, hi = H.block_bootstrap_sharpe_ci(
        values,
        periods_per_year,
        block=block,
        n_boot=reps,
        alpha=alpha,
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
    independent_trials: float | None = None,
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
            independent_trials=independent_trials,
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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
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
            handle.write(
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
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
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            _fsync_directory(path.parent)


def _write_csv_atomic(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holdout_window(
    symbols: Sequence[str], interval: str, start_time: int, end_time: int
) -> dict[str, object]:
    interval_ms = HOLDOUT_INTERVAL_MS.get(interval)
    if not symbols or interval_ms is None or start_time > end_time:
        raise ValueError("final holdout window is invalid")
    return {
        "symbols": sorted(set(symbols)),
        "interval": interval,
        "startOpenTime": int(start_time),
        "endOpenTime": int(end_time),
        "outcomeEndTimeExclusive": int(end_time) + interval_ms,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _registry_window(
    path: Path,
    record: object,
    *,
    strict_identity: bool = False,
) -> dict[str, object]:
    try:
        if not isinstance(record, dict):
            raise TypeError
        if record.get("registryVersion") != HOLDOUT_REGISTRY_VERSION:
            raise ValueError
        if record.get("status") not in {"opening", "completed"}:
            raise ValueError
        if strict_identity:
            identity = record["holdoutIdentitySha256"]
            registration_sha = record["registrationSha256"]
            manifest_sha = record["campaignManifestSha256"]
            panel_sha = record["panelSha256"]
            output_binding_sha = record["outputBindingSha256"]
            campaign = record["campaign"]
            if (
                not _is_sha256(identity)
                or path.name != f"{identity}.json"
                or not _is_sha256(registration_sha)
                or not _is_sha256(manifest_sha)
                or not _is_sha256(panel_sha)
                or not _is_sha256(output_binding_sha)
                or not isinstance(campaign, str)
                or not campaign
            ):
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
    if strict_identity:
        output_directory = _registry_output_directory(path, record)
        expected_identity = _json_digest(
            {
                "campaign": campaign,
                "panelSha256": panel_sha,
                "window": canonical,
            }
        )
        if identity != expected_identity:
            raise ValueError(f"holdout registry entry {path.name} is invalid")
        expected_output_binding = _json_digest(
            {
                "holdoutIdentitySha256": identity,
                "outputDirectory": str(output_directory),
            }
        )
        if output_binding_sha != expected_output_binding:
            raise ValueError(f"holdout registry entry {path.name} is invalid")
        _validate_strict_registry_manifest(
            path,
            output_directory,
            campaign,
            registration_sha,
            manifest_sha,
            panel_sha,
        )
    return canonical


def _registry_output_directory(path: Path, record: object) -> Path:
    try:
        if not isinstance(record, Mapping):
            raise TypeError
        artifacts = record["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise TypeError
        output_directory = artifacts["outputDirectory"]
        if not isinstance(output_directory, str) or not output_directory:
            raise TypeError
        output_path = Path(output_directory)
        if not output_path.is_absolute():
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"holdout registry entry {path.name} is invalid") from error
    return output_path.resolve()


def _validate_strict_registry_manifest(
    path: Path,
    output_directory: Path,
    campaign: str,
    registration_sha: str,
    manifest_sha: str,
    panel_sha: str,
) -> None:
    manifest_path = output_directory / "campaign-manifest.json"
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("campaign") != campaign
            or manifest.get("registrationSha256") != registration_sha
            or hashlib.sha256(manifest_payload).hexdigest() != manifest_sha
        ):
            raise ValueError
        registered_data = manifest["registeredData"]
        if not isinstance(registered_data, Mapping) or not any(
            registered_data.get(field) == panel_sha
            for field in STRICT_HOLDOUT_MANIFEST_PANEL_FIELDS
        ):
            raise ValueError
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"holdout registry entry {path.name} is invalid") from error


def _legacy_registry_window(
    path: Path,
    record: Mapping[str, object],
) -> dict[str, object]:
    campaign = record.get("campaign")
    panel_field = LEGACY_HOLDOUT_PANEL_FIELDS.get(campaign)
    if panel_field is None:
        raise ValueError(f"holdout registry entry {path.name} is invalid")
    canonical = _registry_window(path, record)
    output_directory = _registry_output_directory(path, record)
    manifest_path = output_directory / "campaign-manifest.json"
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        if not isinstance(manifest, Mapping) or manifest.get("campaign") != campaign:
            raise ValueError
        registered_data = manifest["registeredData"]
        if not isinstance(registered_data, Mapping):
            raise TypeError
        panel_sha = registered_data[panel_field]
        identity = record["holdoutIdentitySha256"]
        registration_sha = record["registrationSha256"]
        if (
            not _is_sha256(panel_sha)
            or not _is_sha256(identity)
            or not _is_sha256(registration_sha)
        ):
            raise ValueError
        if path.name != f"{identity}.json":
            raise ValueError
        expected_identity = _json_digest(
            {
                "campaign": campaign,
                "panelSha256": panel_sha,
                "window": canonical,
            }
        )
        if identity != expected_identity:
            raise ValueError
        if campaign == "residual_momentum_derivatives_ablation_v1":
            expected_registration_sha = _json_digest(manifest)
        else:
            expected_registration_sha = manifest["registrationSha256"]
        if registration_sha != expected_registration_sha:
            raise ValueError
        manifest_digest_mode = LEGACY_HOLDOUT_MANIFEST_DIGEST_MODES.get(campaign)
        if manifest_digest_mode is not None:
            manifest_sha = record["campaignManifestSha256"]
            expected_manifest_sha = (
                _json_digest(manifest)
                if manifest_digest_mode == "canonical-json"
                else hashlib.sha256(manifest_payload).hexdigest()
            )
            if not _is_sha256(manifest_sha) or manifest_sha != expected_manifest_sha:
                raise ValueError
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"holdout registry entry {path.name} is invalid") from error
    return canonical


def _registry_window_for_scan(
    path: Path,
    record: object,
    *,
    strict_identity: bool,
) -> dict[str, object]:
    """Validate strict markers while retaining overlap safety for legacy v3."""
    legacy_schema = (
        strict_identity
        and isinstance(record, Mapping)
        and record.get("campaign") in LEGACY_HOLDOUT_IDENTITY_CAMPAIGNS
        and "panelSha256" not in record
    )
    if legacy_schema:
        return _legacy_registry_window(path, record)
    return _registry_window(path, record, strict_identity=strict_identity)


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
    *,
    strict_identity: bool = False,
) -> None:
    if output_record.is_symlink():
        raise ValueError("final holdout output record path is unsafe")
    if output_record.exists():
        raise ValueError("final holdout was already consumed for this output directory")
    if not registry_dir.exists():
        return
    for path in sorted(registry_dir.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"holdout registry entry {path.name} is unsafe")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"holdout registry entry {path.name} is unreadable") from error
        registered_window = _registry_window_for_scan(
            path, record, strict_identity=strict_identity
        )
        if (
            strict_identity
            and _registry_output_directory(path, record)
            == output_record.parent.resolve()
        ):
            raise ValueError(
                "final holdout was already consumed for this output directory; "
                "use its existing registry evidence"
            )
        if _windows_overlap(window, registered_window):
            raise ValueError(
                "final holdout overlaps an already consumed symbol/time window"
            )


def _open_lock_file(path: Path) -> Any:
    """Open a regular lock file without following a precreated symlink."""
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0:
        raise RuntimeError("lock files require O_NOFOLLOW support")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | no_follow
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"lock path is not a regular file: {path}")
        handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        descriptor = None
        return handle
    except OSError as error:
        raise ValueError(f"lock path is unsafe or unavailable: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _is_official_shared_registry(registry_dir: Path) -> bool:
    return registry_dir.resolve() == SHARED_HOLDOUT_REGISTRY_DIR.resolve()


def _assert_official_shared_registry_resolved(registry_dir: Path) -> None:
    if (
        _is_official_shared_registry(registry_dir)
        and SHARED_REPOSITORY_RESOLUTION_ERROR is not None
    ):
        raise ValueError("official shared holdout registry cannot be resolved")


@contextmanager
def _campaign_output_lock(
    output_dir: Path,
    timeout_seconds: float = CAMPAIGN_OUTPUT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("campaign output lock timeout must be positive and finite")
    deadline = time.monotonic() + timeout_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    with _open_lock_file(output_dir / ".campaign.lock") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("campaign output lock deadline exceeded")
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _holdout_registry_lock(
    registry_dir: Path,
    timeout_seconds: float = HOLDOUT_REGISTRY_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("holdout registry lock timeout must be positive and finite")
    deadline = time.monotonic() + timeout_seconds
    registry_dir.mkdir(parents=True, exist_ok=True)
    with _open_lock_file(registry_dir / ".registry.lock") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("holdout registry lock deadline exceeded")
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assert_output_holdout_not_consumed(
    registry_dir: Path,
    output_dir: Path,
    *,
    strict_identity: bool = False,
) -> None:
    if strict_identity:
        _assert_official_shared_registry_resolved(registry_dir)
    output_record = output_dir / "final-holdout-opened.json"
    if output_record.is_symlink():
        raise ValueError("final holdout output record path is unsafe")
    if output_record.exists():
        if not output_record.is_file():
            raise ValueError("final holdout output record path is unsafe")
        try:
            record = json.loads(output_record.read_text(encoding="utf-8"))
            status = record["status"]
        except (KeyError, json.JSONDecodeError, OSError, TypeError) as error:
            raise ValueError("final holdout output record is invalid") from error
        if status not in {"opening", "completed"}:
            raise ValueError("final holdout output record has an invalid status")
        raise ValueError(
            "final holdout was already consumed for this output directory; "
            "use its existing evidence"
        )

    resolved_output = str(output_dir.resolve())
    if not registry_dir.exists():
        return
    with _holdout_registry_lock(registry_dir):
        for path in sorted(registry_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"holdout registry entry {path.name} is unsafe"
                )
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, TypeError) as error:
                raise ValueError(
                    f"holdout registry entry {path.name} is invalid"
                ) from error
            _registry_window_for_scan(
                path, record, strict_identity=strict_identity
            )
            artifacts = record.get("artifacts") if isinstance(record, Mapping) else None
            if not isinstance(artifacts, Mapping):
                if strict_identity:
                    raise ValueError(
                        f"holdout registry entry {path.name} is invalid"
                    )
                continue
            registered_output: object = artifacts.get("outputDirectory")
            if strict_identity:
                registered_output = str(_registry_output_directory(path, record))
            if registered_output != resolved_output:
                continue
            if record.get("status") not in {"opening", "completed"}:
                raise ValueError(f"holdout registry entry {path.name} is invalid")
            raise ValueError(
                "final holdout was already consumed for this output directory; "
                "use its existing registry evidence"
            )


def _reserve_holdout(
    registry_dir: Path,
    marker: Path,
    window: Mapping[str, object],
    output_record: Path,
    opening_record: Mapping[str, object],
    *,
    strict_identity: bool = False,
) -> None:
    if marker.parent.resolve() != registry_dir.resolve():
        raise ValueError("holdout marker escaped its locked registry")
    with _holdout_registry_lock(registry_dir):
        if strict_identity:
            _assert_official_shared_registry_resolved(registry_dir)
        if (
            _is_official_shared_registry(registry_dir)
            and SHARED_REPOSITORY_RESOLUTION_ERROR is None
        ):
            _assert_shared_registry_reconciled(
                REPOSITORY_ROOT,
                SHARED_REPOSITORY_ROOT,
                CANONICAL_SHARED_HOLDOUT_REGISTRY_DIR,
            )
        if strict_identity:
            recorded_window = _registry_window(
                marker, dict(opening_record), strict_identity=True
            )
            if (
                recorded_window != dict(window)
                or _registry_output_directory(marker, opening_record)
                != output_record.parent.resolve()
            ):
                raise ValueError(
                    f"holdout registry entry {marker.name} is invalid"
                )
        _assert_holdout_available(
            registry_dir,
            window,
            output_record,
            strict_identity=strict_identity,
        )
        _write_json_exclusive(marker, opening_record)
        _write_json_exclusive(output_record, opening_record)


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
