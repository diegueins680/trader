"""Pure mechanics for the fixed residual-reversal research campaign.

The campaign reverses residual momentum without using funding as a feature.
Execution holds futures notionals between scheduled activations, so reported
weights drift with prices and net equity rather than hiding a rebalance on
every bar. Terminal liquidation is deliberately not charged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from funding_campaign import CAMPAIGN_INTERVAL_MS, FundingSettlement


HORIZON_HOURS: Final[tuple[int, ...]] = (24, 72, 168)
REBALANCE_BARS: Final[tuple[int, ...]] = (1, 3)
REBALANCE_ANCHOR_OPEN_TIME: Final[int] = 1_600_819_200_000
TRIAL_IDS: Final[tuple[str, ...]] = tuple(
    f"resrev_{hours}h_rebalance_{rebalance_bars}bar"
    for hours in HORIZON_HOURS
    for rebalance_bars in REBALANCE_BARS
)


class PortfolioBankruptcyError(ValueError):
    """A trial's net interval loss exhausted its modeled portfolio equity."""

    def __init__(
        self, interval_left_close_time: int, trial_id: str | None = None
    ) -> None:
        self.interval_left_close_time = int(interval_left_close_time)
        self.outcome_close_time = self.interval_left_close_time + CAMPAIGN_INTERVAL_MS
        # Retain the disclosed pre-merge exception field as an explicit alias.
        self.close_time = self.interval_left_close_time
        self.trial_id = trial_id
        trial_context = f" for trial {trial_id}" if trial_id is not None else ""
        super().__init__(
            f"portfolio equity exhausted{trial_context} over closeTime interval "
            f"({self.interval_left_close_time}, {self.outcome_close_time}]"
        )


@dataclass(frozen=True)
class ReversalCampaignSpec:
    """One member of the fixed three-horizon by two-cadence ledger."""

    trial_id: str
    horizon_hours: int
    horizon_bars: int
    rebalance_bars: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReversalCampaignConfig:
    """Portfolio and accounting rules shared by all six trials."""

    interval_ms: int
    rebalance_anchor_open_time: int = REBALANCE_ANCHOR_OPEN_TIME
    rebalance_phase_bars: int = 0
    top_n: int = 1
    gross_exposure: float = 1.0
    cost_per_turnover: float = 0.0005
    signal_delay_bars: int = 1


def campaign_specs(interval_ms: int) -> tuple[ReversalCampaignSpec, ...]:
    """Return the exact 24/72/168 hour by 1/3 bar trial ledger."""
    _positive_integer(interval_ms, "interval_ms")
    if interval_ms != CAMPAIGN_INTERVAL_MS:
        raise ValueError("reversal campaign interval_ms must be exactly eight hours")

    specifications = tuple(
        ReversalCampaignSpec(
            trial_id=f"resrev_{hours}h_rebalance_{rebalance_bars}bar",
            horizon_hours=hours,
            horizon_bars=hours * 3_600_000 // interval_ms,
            rebalance_bars=rebalance_bars,
        )
        for hours in HORIZON_HOURS
        for rebalance_bars in REBALANCE_BARS
    )
    if tuple(spec.trial_id for spec in specifications) != TRIAL_IDS:
        raise AssertionError("fixed reversal campaign ledger changed")
    return specifications


def scores_for_trial(
    residual_momentum: pd.DataFrame,
    spec: ReversalCampaignSpec,
) -> pd.DataFrame:
    """Negate finite, non-zero residual momentum for one fixed trial."""
    _validate_spec(spec, CAMPAIGN_INTERVAL_MS)
    momentum = _numeric_frame("residual_momentum", residual_momentum, allow_nan=True)
    eligible = np.isfinite(momentum) & (momentum != 0.0)
    return -momentum.where(eligible)


def weights_for_trial(
    residual_momentum: pd.DataFrame,
    spec: ReversalCampaignSpec,
    config: ReversalCampaignConfig,
) -> pd.DataFrame:
    """Build the delayed target path on the spec's absolute UTC grid.

    Binance close times are one millisecond before an interval boundary. Each
    close is mapped back to its bar's absolute open time, and scheduled from
    the registered open-time anchor regardless of which row begins an
    evaluated slice. The resulting target becomes effective after
    ``signal_delay_bars`` and is held until the next target becomes effective.
    """
    _validate_config(config)
    _validate_spec(spec, config.interval_ms)
    scores = scores_for_trial(residual_momentum, spec)
    boundary_numbers = _boundary_numbers(
        scores.index,
        config.interval_ms,
        config.rebalance_anchor_open_time,
    )
    scheduled = _scheduled_mask(boundary_numbers, spec, config)

    target = pd.DataFrame(
        np.nan,
        index=scores.index.copy(),
        columns=scores.columns.copy(),
        dtype=float,
    )
    side_weight = config.gross_exposure / 2.0
    values = scores.to_numpy(dtype=float)
    for row_number in np.flatnonzero(scheduled):
        row = values[row_number]
        short_candidates = np.flatnonzero(np.isfinite(row) & (row < 0.0))
        long_candidates = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        weights = np.zeros(len(scores.columns), dtype=float)
        if short_candidates.size and long_candidates.size:
            ranked_shorts = short_candidates[
                np.argsort(row[short_candidates], kind="stable")
            ]
            ranked_longs = long_candidates[
                np.argsort(row[long_candidates], kind="stable")
            ]
            weights[ranked_shorts[0]] = -side_weight
            weights[ranked_longs[-1]] = side_weight
        target.iloc[row_number] = weights

    delayed = target.shift(config.signal_delay_bars).ffill().fillna(0.0)
    return delayed.astype(float)


def evaluate_drifted_intervals(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    activation_mask: np.ndarray,
    settlements: Sequence[FundingSettlement],
    config: ReversalCampaignConfig,
) -> pd.DataFrame:
    """Account for a futures target path without implicit interim rebalancing.

    On activation rows, turnover is measured from the pre-trade drifted weights
    to the scheduled target. Between activations, signed contract notionals are
    held: each notional moves with its asset price and its weight is divided by
    portfolio equity after price PnL, funding, and activation cost. Funding
    events are included exactly on ``(left close, right close]``. No terminal
    liquidation turnover or cost is added.
    """
    _validate_config(config)
    close_values = _numeric_frame("close", close, allow_nan=False)
    if len(close_values) < 2:
        raise ValueError("close must contain at least two observations")
    if np.any(close_values.to_numpy(dtype=float) <= 0):
        raise ValueError("close prices must be positive")
    _boundary_numbers(
        close_values.index,
        config.interval_ms,
        config.rebalance_anchor_open_time,
    )

    targets = _numeric_frame("target_weights", target_weights, allow_nan=False)
    _require_identical_axes(close_values, targets, "close", "target_weights")
    activations = np.asarray(activation_mask)
    if activations.dtype != np.bool_ or activations.shape != (len(close_values),):
        raise ValueError("activation_mask must contain one boolean per close row")

    close_times = close_values.index.to_numpy(dtype=np.int64, copy=True)
    funding_coefficients = _funding_coefficients(
        close_values,
        close_times,
        tuple(settlements),
    )
    closes = close_values.to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)
    price_returns = closes[1:] / closes[:-1] - 1.0
    drifted_weights = np.zeros(close_values.shape[1], dtype=float)
    rows: list[tuple[object, ...]] = []
    for row_number, close_time in enumerate(close_times[:-1]):
        if activations[row_number]:
            effective_weights = target_values[row_number].copy()
            turnover = float(np.abs(effective_weights - drifted_weights).sum())
        else:
            effective_weights = drifted_weights.copy()
            turnover = 0.0

        price_gross = float(np.dot(effective_weights, price_returns[row_number]))
        funding_cashflow = float(
            np.dot(effective_weights, funding_coefficients[row_number])
        )
        gross = price_gross + funding_cashflow
        cost = config.cost_per_turnover * turnover
        net = gross - cost
        decomposition = (price_gross, funding_cashflow, gross, turnover, cost, net)
        if not np.isfinite(decomposition).all():
            raise ValueError(f"non-finite interval accounting at closeTime={close_time}")
        equity_factor = 1.0 + net
        if equity_factor <= 0.0:
            raise PortfolioBankruptcyError(int(close_time))

        active = int(np.count_nonzero(np.abs(effective_weights) > 1e-12))
        rows.append((close_time, *decomposition, active, *effective_weights.tolist()))
        drifted_weights = (
            effective_weights * (1.0 + price_returns[row_number]) / equity_factor
        )
        if not np.isfinite(drifted_weights).all():
            raise ValueError(f"non-finite drifted weights at closeTime={close_time}")

    columns = [
        "closeTime",
        "priceGross",
        "fundingCashflow",
        "gross",
        "turnover",
        "cost",
        "net",
        "active",
        *(f"weight_{symbol}" for symbol in close_values.columns),
    ]
    return pd.DataFrame(rows, columns=columns).set_index("closeTime")


def run_trial(
    close: pd.DataFrame,
    residual_momentum_by_horizon: Mapping[int, pd.DataFrame],
    settlements: Sequence[FundingSettlement],
    config: ReversalCampaignConfig,
    spec: ReversalCampaignSpec,
) -> pd.DataFrame:
    """Evaluate one fixed trial while retaining its full drifted state path."""
    _validate_config(config)
    _validate_spec(spec, config.interval_ms)
    try:
        residual_momentum = residual_momentum_by_horizon[spec.horizon_hours]
    except KeyError as exc:
        raise ValueError(
            f"missing residual momentum for {spec.horizon_hours}h horizon"
        ) from exc
    targets = weights_for_trial(residual_momentum, spec, config)
    boundary_numbers = _boundary_numbers(
        targets.index,
        config.interval_ms,
        config.rebalance_anchor_open_time,
    )
    scheduled = _scheduled_mask(boundary_numbers, spec, config)
    activations = np.zeros(len(targets), dtype=bool)
    delay = config.signal_delay_bars
    if delay < len(activations):
        activations[delay:] = scheduled[:-delay]
    try:
        return evaluate_drifted_intervals(
            close,
            targets,
            activations,
            settlements,
            config,
        )
    except PortfolioBankruptcyError as exc:
        if exc.trial_id is not None:
            raise
        raise PortfolioBankruptcyError(exc.close_time, spec.trial_id) from exc


def run_trial_matrix(
    close: pd.DataFrame,
    residual_momentum_by_horizon: Mapping[int, pd.DataFrame],
    settlements: Sequence[FundingSettlement],
    config: ReversalCampaignConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    tuple[ReversalCampaignSpec, ...],
]:
    """Evaluate all six trials with drift-aware futures accounting."""
    _validate_config(config)
    specs = campaign_specs(config.interval_ms)
    events = tuple(settlements)
    details: dict[str, pd.DataFrame] = {}
    for spec in specs:
        details[spec.trial_id] = run_trial(
            close,
            residual_momentum_by_horizon,
            events,
            config,
            spec,
        )

    matrix = pd.DataFrame(
        {trial_id: detail["net"] for trial_id, detail in details.items()}
    )
    if tuple(matrix.columns) != TRIAL_IDS:
        raise AssertionError("trial matrix does not match the fixed reversal ledger")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("trial matrix contains non-finite returns")
    return matrix, details, specs


def _scheduled_mask(
    boundary_numbers: np.ndarray,
    spec: ReversalCampaignSpec,
    config: ReversalCampaignConfig,
) -> np.ndarray:
    return (
        boundary_numbers - config.rebalance_phase_bars
    ) % spec.rebalance_bars == 0


def _funding_coefficients(
    close: pd.DataFrame,
    close_times: np.ndarray,
    settlements: Sequence[FundingSettlement],
) -> np.ndarray:
    """Resolve settlement cashflow coefficients for each close interval."""
    result = np.zeros((len(close_times) - 1, len(close.columns)), dtype=float)
    symbols = {symbol: position for position, symbol in enumerate(close.columns)}
    close_values = close.to_numpy(dtype=float)
    observed_symbols: set[object] = set()
    observed_keys: set[tuple[str, int]] = set()
    for event_number, event in enumerate(settlements):
        if not isinstance(event, FundingSettlement):
            raise TypeError(f"settlement {event_number} must be a FundingSettlement")
        if event.symbol not in symbols:
            raise ValueError(f"funding settlement has unknown symbol: {event.symbol}")
        if isinstance(event.funding_time, bool) or not isinstance(
            event.funding_time, (int, np.integer)
        ):
            raise ValueError("funding_time must be an integer millisecond timestamp")
        event_key = (event.symbol, int(event.funding_time))
        if event_key in observed_keys:
            raise ValueError("duplicate symbol/funding_time settlement")
        observed_keys.add(event_key)
        rate = _finite_number(event.rate, "funding rate")
        mark = _finite_number(event.resolved_mark_price, "resolved mark price")
        if mark <= 0:
            raise ValueError("resolved mark price must be positive")

        interval = int(np.searchsorted(close_times, event.funding_time, side="left") - 1)
        if interval < 0 or interval >= len(result):
            continue
        observed_symbols.add(event.symbol)
        symbol_position = symbols[event.symbol]
        result[interval, symbol_position] += (
            -mark / close_values[interval, symbol_position] * rate
        )

    missing_symbols = set(symbols).difference(observed_symbols)
    if missing_symbols:
        missing = ", ".join(sorted(str(symbol) for symbol in missing_symbols))
        raise ValueError(f"funding settlement schedule is absent for: {missing}")
    return result


def _boundary_numbers(
    index: pd.Index,
    interval_ms: int,
    rebalance_anchor_open_time: int,
) -> np.ndarray:
    values = index.to_numpy()
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(
            "residual_momentum index must contain integer millisecond closeTime values"
        )
    close_times = values.astype(np.int64, copy=True)
    if len(close_times) > 1 and not np.all(np.diff(close_times) == interval_ms):
        raise ValueError("residual_momentum closeTime values must be one interval apart")
    if not np.all((close_times + 1) % interval_ms == 0):
        raise ValueError(
            "residual_momentum closeTime values must end one millisecond before "
            "absolute interval boundaries"
        )
    open_times = close_times - interval_ms + 1
    if np.any(open_times < rebalance_anchor_open_time):
        raise ValueError("residual_momentum closeTime values precede the fixed anchor")
    offsets = open_times - rebalance_anchor_open_time
    if not np.all(offsets % interval_ms == 0):
        raise ValueError("residual_momentum closeTime values are off the absolute grid")
    return offsets // interval_ms


def _validate_config(config: ReversalCampaignConfig) -> None:
    if not isinstance(config, ReversalCampaignConfig):
        raise TypeError("config must be a ReversalCampaignConfig")
    campaign_specs(config.interval_ms)
    if isinstance(config.rebalance_anchor_open_time, bool) or not isinstance(
        config.rebalance_anchor_open_time, (int, np.integer)
    ):
        raise ValueError(
            "rebalance_anchor_open_time must be an integer millisecond timestamp"
        )
    if config.rebalance_anchor_open_time != REBALANCE_ANCHOR_OPEN_TIME:
        raise ValueError(
            "rebalance_anchor_open_time must match the fixed campaign anchor"
        )
    if config.rebalance_anchor_open_time % config.interval_ms != 0:
        raise ValueError("rebalance_anchor_open_time must lie on an interval boundary")
    if isinstance(config.rebalance_phase_bars, bool) or not isinstance(
        config.rebalance_phase_bars, (int, np.integer)
    ):
        raise ValueError("rebalance_phase_bars must be an integer")
    if config.rebalance_phase_bars not in (0, 1, 2):
        raise ValueError("rebalance_phase_bars must be exactly 0, 1, or 2")
    _positive_integer(config.top_n, "top_n")
    if config.top_n != 1:
        raise ValueError("top_n must be exactly one for the fixed reversal campaign")
    _positive_integer(config.signal_delay_bars, "signal_delay_bars")
    exposure = _finite_number(config.gross_exposure, "gross_exposure")
    cost = _finite_number(config.cost_per_turnover, "cost_per_turnover")
    if exposure < 0:
        raise ValueError("gross_exposure must be non-negative")
    if cost < 0:
        raise ValueError("cost_per_turnover must be non-negative")


def _validate_spec(spec: ReversalCampaignSpec, interval_ms: int) -> None:
    if not isinstance(spec, ReversalCampaignSpec):
        raise TypeError("spec must be a ReversalCampaignSpec")
    if spec not in campaign_specs(interval_ms):
        raise ValueError("spec is not a member of the fixed reversal campaign")


def _numeric_frame(
    name: str,
    frame: pd.DataFrame,
    *,
    allow_nan: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and ordered")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    try:
        values = frame.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        requirement = "finite or missing" if allow_nan else "finite"
        raise ValueError(f"{name} values must be {requirement}")
    return pd.DataFrame(values, index=frame.index.copy(), columns=frame.columns.copy())


def _require_identical_axes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> None:
    if not left.index.equals(right.index) or list(left.columns) != list(right.columns):
        raise ValueError(f"{left_name} and {right_name} must have identical axes")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite_number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
