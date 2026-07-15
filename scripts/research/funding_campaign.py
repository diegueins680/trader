"""Pure, causal mechanics for the fixed funding-only research campaign.

This module deliberately does not acquire or repair market data.  In
particular, every funding settlement must arrive with a mark price resolved by
the caller from information available at the settlement time.  Missing marks
or rates fail closed rather than being treated as zero cashflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final, Literal, Mapping, Sequence

import numpy as np
import pandas as pd


TrialVariant = Literal["base", "funding_only"]

CAMPAIGN_INTERVAL_MS: Final[int] = 8 * 3_600_000
HORIZON_HOURS: Final[tuple[int, ...]] = (24, 72, 168)
TRIAL_VARIANTS: Final[tuple[TrialVariant, ...]] = ("base", "funding_only")
TRIAL_IDS: Final[tuple[str, ...]] = tuple(
    f"resmom_{hours}h_{variant}"
    for hours in HORIZON_HOURS
    for variant in TRIAL_VARIANTS
)


@dataclass(frozen=True)
class FundingCampaignSpec:
    """One member of the fixed six-trial campaign."""

    trial_id: str
    horizon_hours: int
    horizon_bars: int
    variant: TrialVariant

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FundingCampaignConfig:
    """Portfolio rules shared by all six pre-registered trials."""

    interval_ms: int
    funding_crowding_z: float = 2.0
    top_n: int = 1
    gross_exposure: float = 1.0
    cost_per_turnover: float = 0.0005
    signal_delay_bars: int = 1


@dataclass(frozen=True)
class FundingSettlement:
    """A funding event with its point-in-time mark already resolved upstream.

    When the exchange omits ``markPrice``, the resolver must use the open of
    the mark-price candle containing ``funding_time``.  A candle close would
    include information that was unavailable at the settlement.
    """

    symbol: str
    funding_time: int
    rate: float
    resolved_mark_price: float


def campaign_specs(interval_ms: int) -> tuple[FundingCampaignSpec, ...]:
    """Return the exact 24/72/168 hour by base/funding-only trial ledger."""
    _positive_integer(interval_ms, "interval_ms")
    if interval_ms != CAMPAIGN_INTERVAL_MS:
        raise ValueError("funding campaign interval_ms must be exactly eight hours")
    specifications = []
    for hours in HORIZON_HOURS:
        horizon_ms = hours * 3_600_000
        if horizon_ms % interval_ms != 0:
            raise ValueError("interval_ms must divide every fixed campaign horizon")
        bars = horizon_ms // interval_ms
        for variant in TRIAL_VARIANTS:
            specifications.append(
                FundingCampaignSpec(
                    trial_id=f"resmom_{hours}h_{variant}",
                    horizon_hours=hours,
                    horizon_bars=bars,
                    variant=variant,
                )
            )
    result = tuple(specifications)
    if tuple(spec.trial_id for spec in result) != TRIAL_IDS:
        raise AssertionError("fixed funding campaign ledger changed")
    return result


def causal_funding_zscore(
    funding_rates: pd.DataFrame,
    lookback_bars: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Standardize each funding rate against observations strictly before it."""
    rates = _numeric_frame("funding_rates", funding_rates, allow_nan=False)
    _positive_integer(lookback_bars, "lookback_bars")
    if min_periods is None:
        minimum = lookback_bars
    else:
        _positive_integer(min_periods, "min_periods")
        if min_periods > lookback_bars:
            raise ValueError("min_periods cannot exceed lookback_bars")
        minimum = min_periods

    past = rates.shift(1)
    mean = past.rolling(lookback_bars, min_periods=minimum).mean()
    standard_deviation = past.rolling(
        lookback_bars, min_periods=minimum
    ).std(ddof=0)
    return (rates - mean) / standard_deviation.where(standard_deviation > 1e-12)


def scores_for_trial(
    momentum: pd.DataFrame,
    funding_z: pd.DataFrame,
    spec: FundingCampaignSpec,
    config: FundingCampaignConfig,
) -> pd.DataFrame:
    """Apply the trial's sole eligibility rule to residual momentum.

    ``base`` requires only finite, non-zero momentum. ``funding_only`` adds
    exactly one gate: directional funding z-score must not exceed the fixed
    crowding threshold. Missing funding z-scores therefore make that asset
    ineligible; no OI, basis, or taker-flow input is accepted by this API.
    """
    _validate_config(config)
    _validate_spec(spec, config.interval_ms)
    momentum_values = _numeric_frame("momentum", momentum, allow_nan=True)
    funding_values = _numeric_frame("funding_z", funding_z, allow_nan=True)
    _require_identical_axes(momentum_values, funding_values, "momentum", "funding_z")

    finite_momentum = np.isfinite(momentum_values)
    direction = np.sign(momentum_values)
    eligible = finite_momentum & (direction != 0)
    if spec.variant == "funding_only":
        directional_funding = direction * funding_values
        eligible &= np.isfinite(funding_values) & (
            directional_funding <= config.funding_crowding_z
        )
    return momentum_values.where(eligible)


def weights_for_trial(
    momentum: pd.DataFrame,
    funding_z: pd.DataFrame,
    spec: FundingCampaignSpec,
    config: FundingCampaignConfig,
) -> pd.DataFrame:
    """Build dollar-neutral weights with the configured causal bar delay."""
    scores = scores_for_trial(momentum, funding_z, spec, config)
    target = _target_weights(scores, config)
    delayed = target.shift(config.signal_delay_bars).fillna(0.0)
    return delayed.astype(float)


def evaluate_intervals(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    settlements: Sequence[FundingSettlement],
    config: FundingCampaignConfig,
) -> pd.DataFrame:
    """Account for price, funding, turnover, and costs over close intervals.

    The row at ``closeTime[t]`` uses the weight already effective at that
    close. Every settlement satisfying
    ``closeTime[t] < funding_time <= closeTime[t + 1]`` is included as
    ``-weight * resolved_mark_price / close[t] * rate``.
    """
    _validate_config(config)
    close_values = _numeric_frame("close", close, allow_nan=False)
    if len(close_values) < 2:
        raise ValueError("close must contain at least two observations")
    if np.any(close_values.to_numpy(dtype=float) <= 0):
        raise ValueError("close prices must be positive")
    close_times = _close_times(close_values.index, config.interval_ms)

    weight_values = _numeric_frame("weights", weights, allow_nan=False)
    _require_identical_axes(close_values, weight_values, "close", "weights")
    events = tuple(settlements)
    funding_cashflows = _funding_cashflows(
        close_values,
        weight_values,
        close_times,
        events,
    )

    closes = close_values.to_numpy(dtype=float)
    effective_weights = weight_values.to_numpy(dtype=float)
    price_returns = closes[1:] / closes[:-1] - 1.0
    previous = np.zeros(close_values.shape[1], dtype=float)
    rows: list[tuple[object, ...]] = []
    for row_number, close_time in enumerate(close_times[:-1]):
        current = effective_weights[row_number]
        price_gross = float(np.dot(current, price_returns[row_number]))
        funding_cashflow = float(funding_cashflows[row_number])
        gross = price_gross + funding_cashflow
        turnover = float(np.abs(current - previous).sum())
        cost = config.cost_per_turnover * turnover
        net = gross - cost
        decomposition = (price_gross, funding_cashflow, gross, turnover, cost, net)
        if not np.isfinite(decomposition).all():
            raise ValueError(f"non-finite interval accounting at closeTime={close_time}")
        active = int(np.count_nonzero(np.abs(current) > 1e-12))
        rows.append((close_time, *decomposition, active, *current.tolist()))
        previous = current

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


def run_trial_matrix(
    close: pd.DataFrame,
    momentum_by_horizon: Mapping[int, pd.DataFrame],
    funding_z: pd.DataFrame,
    settlements: Sequence[FundingSettlement],
    config: FundingCampaignConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    tuple[FundingCampaignSpec, ...],
]:
    """Evaluate all six trials and return their complete aligned net matrix."""
    specs = campaign_specs(config.interval_ms)
    events = tuple(settlements)
    details: dict[str, pd.DataFrame] = {}
    for spec in specs:
        try:
            momentum = momentum_by_horizon[spec.horizon_hours]
        except KeyError as exc:
            raise ValueError(
                f"missing momentum for {spec.horizon_hours}h horizon"
            ) from exc
        weights = weights_for_trial(momentum, funding_z, spec, config)
        details[spec.trial_id] = evaluate_intervals(
            close,
            weights,
            events,
            config,
        )

    matrix = pd.DataFrame(
        {trial_id: detail["net"] for trial_id, detail in details.items()}
    )
    if tuple(matrix.columns) != TRIAL_IDS:
        raise AssertionError("trial matrix does not match the fixed campaign ledger")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("trial matrix contains non-finite returns")
    return matrix, details, specs


def _target_weights(
    scores: pd.DataFrame,
    config: FundingCampaignConfig,
) -> pd.DataFrame:
    side_weight = config.gross_exposure / (2 * config.top_n)
    values = scores.to_numpy(dtype=float)
    target = np.zeros_like(values)
    for row_number, row in enumerate(values):
        short_candidates = np.flatnonzero(np.isfinite(row) & (row < 0.0))
        long_candidates = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if (
            short_candidates.size < config.top_n
            or long_candidates.size < config.top_n
        ):
            continue
        ranked_shorts = short_candidates[
            np.argsort(row[short_candidates], kind="stable")
        ]
        ranked_longs = long_candidates[
            np.argsort(row[long_candidates], kind="stable")
        ]
        target[row_number, ranked_shorts[: config.top_n]] = -side_weight
        target[row_number, ranked_longs[-config.top_n :]] = side_weight
    return pd.DataFrame(target, index=scores.index.copy(), columns=scores.columns.copy())


def _funding_cashflows(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    close_times: np.ndarray,
    settlements: Sequence[FundingSettlement],
) -> np.ndarray:
    result = np.zeros(len(close_times) - 1, dtype=float)
    symbols = {symbol: position for position, symbol in enumerate(close.columns)}
    close_values = close.to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)
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
        result[interval] += (
            -weight_values[interval, symbol_position]
            * mark
            / close_values[interval, symbol_position]
            * rate
        )
    missing_symbols = set(symbols).difference(observed_symbols)
    if missing_symbols:
        missing = ", ".join(sorted(str(symbol) for symbol in missing_symbols))
        raise ValueError(f"funding settlement schedule is absent for: {missing}")
    return result


def _validate_config(config: FundingCampaignConfig) -> None:
    if not isinstance(config, FundingCampaignConfig):
        raise TypeError("config must be a FundingCampaignConfig")
    campaign_specs(config.interval_ms)
    _positive_integer(config.top_n, "top_n")
    _positive_integer(config.signal_delay_bars, "signal_delay_bars")
    crowding = _finite_number(config.funding_crowding_z, "funding_crowding_z")
    exposure = _finite_number(config.gross_exposure, "gross_exposure")
    cost = _finite_number(config.cost_per_turnover, "cost_per_turnover")
    if crowding < 0:
        raise ValueError("funding_crowding_z must be non-negative")
    if exposure < 0:
        raise ValueError("gross_exposure must be non-negative")
    if cost < 0:
        raise ValueError("cost_per_turnover must be non-negative")


def _validate_spec(spec: FundingCampaignSpec, interval_ms: int) -> None:
    if not isinstance(spec, FundingCampaignSpec):
        raise TypeError("spec must be a FundingCampaignSpec")
    if spec not in campaign_specs(interval_ms):
        raise ValueError("spec is not a member of the fixed funding campaign")


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


def _close_times(index: pd.Index, interval_ms: int) -> np.ndarray:
    values = index.to_numpy()
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError("close index must contain integer millisecond closeTime values")
    times = values.astype(np.int64, copy=True)
    if not np.all(np.diff(times) == interval_ms):
        raise ValueError("closeTime values must be exactly one interval apart")
    return times


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
