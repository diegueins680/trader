"""Causal residual-momentum campaign used by the edge research runner.

The campaign deliberately contains fifteen specifications: three momentum
horizons crossed with five economically distinct derivatives-data ablations.
It produces one dollar-neutral portfolio return series per specification so
the complete trial family can be evaluated for selection bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


ABLATIONS = ("base", "funding_basis", "open_interest", "taker_flow", "all")
DEFAULT_HORIZON_HOURS = (24, 72, 168)


@dataclass(frozen=True)
class CampaignSpec:
    """One pre-registered strategy specification."""

    name: str
    horizon_hours: int
    horizon_bars: int
    ablation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignConfig:
    interval_ms: int
    beta_lookback_bars: int = 168
    feature_lookback_bars: int = 168
    oi_change_bars: int = 24
    funding_basis_crowding_z: float = 2.0
    rebalance_bars: int = 8
    top_n: int = 1
    gross_exposure: float = 1.0
    cost_per_turnover: float = 0.0005
    signal_delay_bars: int = 1


@dataclass(frozen=True)
class PreparedPanel:
    times: pd.Index
    forward_returns: pd.DataFrame
    residual_momentum: Mapping[int, pd.DataFrame]
    funding_z: pd.DataFrame
    basis_z: pd.DataFrame
    oi_change: pd.DataFrame
    taker_imbalance: pd.DataFrame


def campaign_specs(interval_ms: int) -> list[CampaignSpec]:
    """Return the fixed 3 x 5 campaign grid for an interval."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    specs = []
    for hours in DEFAULT_HORIZON_HOURS:
        bars = max(1, round(hours * 3_600_000 / interval_ms))
        for ablation in ABLATIONS:
            specs.append(
                CampaignSpec(
                    name=f"resmom_{hours}h_{ablation}",
                    horizon_hours=hours,
                    horizon_bars=bars,
                    ablation=ablation,
                )
            )
    return specs


def _require_panel_columns(panel: Mapping[str, pd.DataFrame]) -> None:
    required = {"openTime", "close", "funding", "oi", "basis", "taker"}
    if len(panel) < 2:
        raise ValueError("edge campaign requires at least two symbols")
    for symbol, frame in panel.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{symbol} is missing columns: {', '.join(sorted(missing))}")
        if frame["openTime"].duplicated().any():
            raise ValueError(f"{symbol} contains duplicate openTime values")


def _aligned_field(panel: Mapping[str, pd.DataFrame], field: str, times: pd.Index) -> pd.DataFrame:
    columns = {}
    for symbol, frame in sorted(panel.items()):
        series = frame.set_index("openTime")[field].reindex(times)
        columns[symbol] = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(columns, index=times, dtype=float)


def _past_zscore(values: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Standardize each row from observations strictly before that row."""
    lookback = max(2, lookback)
    past = values.shift(1)
    minimum = min(lookback, max(5, lookback // 3))
    mean = past.rolling(lookback, min_periods=minimum).mean()
    std = past.rolling(lookback, min_periods=minimum).std(ddof=0)
    return (values - mean) / std.where(std > 1e-12)


def _finite_mask(values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(values.to_numpy(dtype=float)),
        index=values.index,
        columns=values.columns,
    )


def _rolling_beta(asset_returns: pd.DataFrame, market_returns: pd.Series, lookback: int) -> pd.DataFrame:
    """Estimate causal rolling betas against an equal-weight market return."""
    lookback = max(2, lookback)
    minimum = min(lookback, max(5, lookback // 3))
    market_mean = market_returns.rolling(lookback, min_periods=minimum).mean()
    market_var = market_returns.rolling(lookback, min_periods=minimum).var(ddof=0)
    betas = {}
    for symbol in asset_returns:
        values = asset_returns[symbol]
        covariance = (
            (values * market_returns).rolling(lookback, min_periods=minimum).mean()
            - values.rolling(lookback, min_periods=minimum).mean() * market_mean
        )
        betas[symbol] = covariance / market_var.where(market_var > 1e-16)
    return pd.DataFrame(betas, index=asset_returns.index)


def prepare_panel(panel: Mapping[str, pd.DataFrame], config: CampaignConfig) -> PreparedPanel:
    """Align a panel and build only point-in-time campaign features."""
    _require_panel_columns(panel)
    if config.interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    common_times: set[int] | None = None
    for frame in panel.values():
        observed = set(pd.to_numeric(frame["openTime"], errors="raise").astype(np.int64))
        common_times = observed if common_times is None else common_times.intersection(observed)
    times = pd.Index(sorted(common_times or []), name="openTime")
    if len(times) < 3:
        raise ValueError("edge campaign has fewer than three aligned observations")
    spacing = np.diff(times.to_numpy(dtype=np.int64))
    if not np.all(spacing == config.interval_ms):
        raise ValueError("aligned openTime values must be exactly one interval apart")

    close = _aligned_field(panel, "close", times)
    close_values = close.to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or np.any(close_values <= 0):
        raise ValueError("close prices must be finite and positive")
    simple_returns = close.pct_change(fill_method=None)
    forward_returns = close.shift(-1).divide(close) - 1
    market_returns = simple_returns.mean(axis=1, skipna=False)
    beta = _rolling_beta(simple_returns, market_returns, config.beta_lookback_bars)

    residual_momentum = {}
    for spec in campaign_specs(config.interval_ms):
        horizon = spec.horizon_bars
        if horizon in residual_momentum:
            continue
        asset_momentum = close.divide(close.shift(horizon)) - 1
        market_momentum = (1 + market_returns).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True) - 1
        residual_momentum[horizon] = asset_momentum.subtract(beta.mul(market_momentum, axis=0))

    funding = _aligned_field(panel, "funding", times)
    basis = _aligned_field(panel, "basis", times)
    oi = _aligned_field(panel, "oi", times)
    taker = _aligned_field(panel, "taker", times)
    funding_z = _past_zscore(funding, config.feature_lookback_bars)
    basis_z = _past_zscore(basis, config.feature_lookback_bars)
    oi_denominator = oi.shift(max(1, config.oi_change_bars)).abs().where(lambda values: values > 1e-12)
    oi_change = oi.divide(oi_denominator) - 1
    taker_imbalance = taker - 1.0
    return PreparedPanel(
        times=times,
        forward_returns=forward_returns,
        residual_momentum=residual_momentum,
        funding_z=funding_z,
        basis_z=basis_z,
        oi_change=oi_change,
        taker_imbalance=taker_imbalance,
    )


def scores_for_spec(prepared: PreparedPanel, spec: CampaignSpec, config: CampaignConfig) -> pd.DataFrame:
    """Apply one ablation as a causal eligibility gate over residual momentum."""
    base = prepared.residual_momentum[spec.horizon_bars]
    direction = np.sign(base)
    eligible = _finite_mask(base)
    if spec.ablation in {"funding_basis", "all"}:
        crowding = direction * (prepared.funding_z + prepared.basis_z) / 2
        eligible &= _finite_mask(crowding) & (
            crowding <= config.funding_basis_crowding_z
        )
    if spec.ablation in {"open_interest", "all"}:
        eligible &= _finite_mask(prepared.oi_change) & (prepared.oi_change > 0)
    if spec.ablation in {"taker_flow", "all"}:
        aligned_flow = direction * prepared.taker_imbalance
        eligible &= _finite_mask(aligned_flow) & (aligned_flow > 0)
    if config.signal_delay_bars < 1:
        raise ValueError("signal_delay_bars must be >= 1 for close-derived features")
    return base.where(eligible).shift(config.signal_delay_bars)


def portfolio_returns(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
    config: CampaignConfig,
) -> pd.DataFrame:
    """Backtest an equal long/short cross-sectional book after turnover costs."""
    if not scores.index.equals(forward_returns.index) or list(scores.columns) != list(forward_returns.columns):
        raise ValueError("scores and forward_returns must have identical axes")
    top_n = max(1, config.top_n)
    side_weight = max(0.0, config.gross_exposure) / (2 * top_n)
    score_values = scores.to_numpy(dtype=float)
    return_values = forward_returns.to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise ValueError("forward returns must be finite for every evaluated row")
    previous = np.zeros(len(scores.columns), dtype=float)
    rows = []
    for row_number, timestamp in enumerate(scores.index):
        row_scores = score_values[row_number]
        row_returns = return_values[row_number]
        valid = np.flatnonzero(np.isfinite(row_scores))
        rebalance = row_number % max(1, config.rebalance_bars) == 0
        if rebalance and valid.size >= 2 * top_n:
            ranked = valid[np.argsort(row_scores[valid], kind="stable")]
            weights = np.zeros_like(previous)
            weights[ranked[:top_n]] = -side_weight
            weights[ranked[-top_n:]] = side_weight
        elif rebalance:
            weights = np.zeros_like(previous)
        else:
            weights = previous.copy()

        gross = float(np.dot(weights, row_returns))
        turnover = float(np.abs(weights - previous).sum())
        net = gross - max(0.0, config.cost_per_turnover) * turnover
        if not np.isfinite(net):
            raise ValueError(f"non-finite portfolio return at openTime={timestamp}")
        active = int(np.count_nonzero(np.abs(weights) > 1e-12))
        rows.append((timestamp, gross, turnover, net, active, *weights.tolist()))
        previous = weights
    columns = ["openTime", "gross", "turnover", "net", "active"] + [
        f"weight_{symbol}" for symbol in scores.columns
    ]
    return pd.DataFrame(rows, columns=columns).set_index("openTime")


def run_trial_matrix(
    panel: Mapping[str, pd.DataFrame],
    config: CampaignConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[CampaignSpec]]:
    """Run every pre-registered spec and return one aligned net-return matrix."""
    prepared = prepare_panel(panel, config)
    details = {}
    specs = campaign_specs(config.interval_ms)
    evaluation_returns = prepared.forward_returns.iloc[:-1]
    for spec in specs:
        scores = scores_for_spec(prepared, spec, config).iloc[:-1]
        details[spec.name] = portfolio_returns(scores, evaluation_returns, config)
    matrix = pd.DataFrame(
        {name: frame["net"] for name, frame in details.items()},
        index=evaluation_returns.index,
    )
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("trial matrix contains non-finite returns")
    return matrix, details, specs
