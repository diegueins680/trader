"""Pure mechanics for the locked risk-controlled reversal campaign.

The campaign changes one alpha-execution mechanism: exit-rank-three rank
hysteresis versus a fixed exit-rank-one matched control. All portfolios share
the same lower exposure, close-to-close risk constraints, funding accounting,
and charged terminal liquidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

from funding_campaign import CAMPAIGN_INTERVAL_MS, FundingSettlement
from harness import circular_block_bootstrap_sharpe_ci
import reversal_campaign as Base


__all__ = [
    "EXIT_RANKS",
    "HORIZON_HOURS",
    "REBALANCE_ANCHOR_OPEN_TIME",
    "TRIAL_IDS",
    "FundingSettlement",
    "RiskConstraintBreach",
    "RiskControlledReversalConfig",
    "RiskControlledReversalSpec",
    "campaign_specs",
    "circular_block_bootstrap_sharpe_ci",
    "decision_weights_for_trial",
    "evaluate_drifted_intervals",
    "run_trial",
    "run_trial_matrix",
    "scores_for_trial",
    "weights_for_trial",
]


HORIZON_HOURS: Final[tuple[int, ...]] = (24, 72, 168)
EXIT_RANKS: Final[tuple[int, ...]] = (1, 3)
REBALANCE_ANCHOR_OPEN_TIME: Final[int] = 1_600_819_200_000
TRIAL_IDS: Final[tuple[str, ...]] = tuple(
    f"resrev_{hours}h_exit{exit_rank}_{'control' if exit_rank == 1 else 'hysteresis'}"
    for hours in HORIZON_HOURS
    for exit_rank in EXIT_RANKS
)


class RiskConstraintBreach(ValueError):
    """A path crossed one of the campaign's locked execution-risk limits."""

    def __init__(
        self,
        interval_left_close_time: int,
        reason: str,
        evidence: Mapping[str, object],
        trial_id: str | None = None,
    ) -> None:
        self.interval_left_close_time = int(interval_left_close_time)
        self.outcome_close_time = self.interval_left_close_time + CAMPAIGN_INTERVAL_MS
        self.reason = str(reason)
        self.evidence = _json_safe_mapping(evidence)
        self.trial_id = trial_id
        context = f" for trial {trial_id}" if trial_id else ""
        super().__init__(
            f"risk constraint {self.reason} breached{context} over closeTime "
            f"interval ({self.interval_left_close_time}, {self.outcome_close_time}]"
        )


@dataclass(frozen=True)
class RiskControlledReversalSpec:
    """One horizon crossed with the treatment or its matched control."""

    trial_id: str
    horizon_hours: int
    horizon_bars: int
    exit_rank: int
    champion_eligible: bool

    def to_dict(self) -> dict[str, object]:
        treatment = self.exit_rank == 3
        return {
            "id": self.trial_id,
            "horizonBars": self.horizon_bars,
            "horizonHours": self.horizon_hours,
            "exitRank": self.exit_rank,
            "role": "rank_hysteresis_treatment" if treatment else "matched_control",
            "championEligible": self.champion_eligible,
        }


@dataclass(frozen=True)
class RiskControlledReversalConfig:
    """Locked accounting and close-to-close risk parameters."""

    interval_ms: int
    rebalance_anchor_open_time: int = REBALANCE_ANCHOR_OPEN_TIME
    gross_exposure: float = 0.5
    cost_per_turnover: float = 0.001
    signal_delay_bars: int = 1
    equity_floor: float = 0.75
    maximum_drawdown: float = 0.20
    maximum_endpoint_gross_leverage: float = 1.0
    maximum_symbol_weight: float = 0.40
    maximum_activation_turnover: float = 1.25
    maximum_terminal_turnover: float = 1.25
    adverse_shock_fraction: float = 0.25
    maintenance_margin_rate: float = 0.10
    liquidation_reserve_rate: float = 0.01
    minimum_shock_equity_fraction: float = 0.50
    minimum_shock_maintenance_coverage: float = 2.0
    charge_terminal_liquidation: bool = True


def campaign_specs(interval_ms: int) -> tuple[RiskControlledReversalSpec, ...]:
    """Return the exact three-horizon treatment/control ledger."""
    if isinstance(interval_ms, bool) or not isinstance(
        interval_ms, (int, np.integer)
    ):
        raise ValueError("interval_ms must be a positive integer")
    if int(interval_ms) != CAMPAIGN_INTERVAL_MS:
        raise ValueError("risk-controlled campaign interval must be exactly eight hours")
    specs = tuple(
        RiskControlledReversalSpec(
            trial_id=(
                f"resrev_{hours}h_exit{exit_rank}_"
                f"{'control' if exit_rank == 1 else 'hysteresis'}"
            ),
            horizon_hours=hours,
            horizon_bars=hours * 3_600_000 // int(interval_ms),
            exit_rank=exit_rank,
            champion_eligible=exit_rank == 3,
        )
        for hours in HORIZON_HOURS
        for exit_rank in EXIT_RANKS
    )
    if tuple(spec.trial_id for spec in specs) != TRIAL_IDS:
        raise AssertionError("fixed risk-controlled trial ledger changed")
    return specs


def scores_for_trial(
    residual_momentum: pd.DataFrame,
    spec: RiskControlledReversalSpec,
) -> pd.DataFrame:
    """Return reversal scores while retaining missing/ineligible observations."""
    _validate_spec(spec, CAMPAIGN_INTERVAL_MS)
    momentum = Base._numeric_frame(
        "residual_momentum", residual_momentum, allow_nan=True
    )
    eligible = np.isfinite(momentum) & (momentum != 0.0)
    return -momentum.where(eligible)


def decision_weights_for_trial(
    residual_momentum: pd.DataFrame,
    spec: RiskControlledReversalSpec,
    config: RiskControlledReversalConfig,
) -> pd.DataFrame:
    """Build every-bar target decisions with fixed rank hysteresis."""
    _validate_config(config)
    _validate_spec(spec, config.interval_ms)
    momentum = Base._numeric_frame(
        "residual_momentum", residual_momentum, allow_nan=True
    )
    Base._boundary_numbers(
        momentum.index,
        config.interval_ms,
        config.rebalance_anchor_open_time,
    )
    values = momentum.to_numpy(dtype=float)
    decisions = np.zeros_like(values, dtype=float)
    incumbent_long: int | None = None
    incumbent_short: int | None = None
    side_weight = config.gross_exposure / 2.0

    for row_number, row in enumerate(values):
        negative = np.flatnonzero(np.isfinite(row) & (row < 0.0))
        positive = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if not negative.size or not positive.size:
            incumbent_long = None
            incumbent_short = None
            continue
        long_ranked = negative[np.argsort(row[negative], kind="stable")]
        short_ranked = positive[np.argsort(-row[positive], kind="stable")]
        incumbent_long = _retained_or_extreme(
            incumbent_long, long_ranked, spec.exit_rank
        )
        incumbent_short = _retained_or_extreme(
            incumbent_short, short_ranked, spec.exit_rank
        )
        decisions[row_number, incumbent_long] = side_weight
        decisions[row_number, incumbent_short] = -side_weight

    return pd.DataFrame(
        decisions,
        index=momentum.index.copy(),
        columns=momentum.columns.copy(),
    )


def weights_for_trial(
    residual_momentum: pd.DataFrame,
    spec: RiskControlledReversalSpec,
    config: RiskControlledReversalConfig,
) -> pd.DataFrame:
    """Delay the every-bar decisions by the single registered signal lag."""
    decisions = decision_weights_for_trial(residual_momentum, spec, config)
    return decisions.shift(config.signal_delay_bars).fillna(0.0).astype(float)


def evaluate_drifted_intervals(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    activation_mask: np.ndarray,
    settlements: Sequence[FundingSettlement],
    config: RiskControlledReversalConfig,
    *,
    trial_id: str | None = None,
) -> pd.DataFrame:
    """Evaluate one continuous futures book and fail on any risk breach."""
    _validate_config(config)
    close_values = Base._numeric_frame("close", close, allow_nan=False)
    if len(close_values) < 2:
        raise ValueError("close must contain at least two observations")
    if np.any(close_values.to_numpy(dtype=float) <= 0.0):
        raise ValueError("close prices must be positive")
    Base._boundary_numbers(
        close_values.index,
        config.interval_ms,
        config.rebalance_anchor_open_time,
    )
    targets = Base._numeric_frame(
        "target_weights", target_weights, allow_nan=False
    )
    Base._require_identical_axes(close_values, targets, "close", "target_weights")
    activations = np.asarray(activation_mask)
    if activations.dtype != np.bool_ or activations.shape != (len(close_values),):
        raise ValueError("activation_mask must contain one boolean per close row")

    close_times = close_values.index.to_numpy(dtype=np.int64, copy=True)
    funding = Base._funding_coefficients(
        close_values, close_times, tuple(settlements)
    )
    closes = close_values.to_numpy(dtype=float)
    target_values = targets.to_numpy(dtype=float)
    price_returns = closes[1:] / closes[:-1] - 1.0
    drifted_weights = np.zeros(close_values.shape[1], dtype=float)
    equity = 1.0
    peak_equity = 1.0
    rows: list[tuple[object, ...]] = []

    for row_number, close_time in enumerate(close_times[:-1]):
        if activations[row_number]:
            effective_weights = target_values[row_number].copy()
            activation_turnover = float(
                np.abs(effective_weights - drifted_weights).sum()
            )
        else:
            effective_weights = drifted_weights.copy()
            activation_turnover = 0.0

        if not np.isfinite(activation_turnover):
            _raise_breach(
                close_time,
                "non_finite_state",
                drifted_weights,
                equity,
                config,
                trial_id,
                {"field": "activationTurnover"},
            )
        if activation_turnover > config.maximum_activation_turnover:
            _raise_breach(
                close_time,
                "maximum_activation_turnover",
                drifted_weights,
                equity,
                config,
                trial_id,
                {
                    "observed": activation_turnover,
                    "limit": config.maximum_activation_turnover,
                },
            )

        activation_cost = config.cost_per_turnover * activation_turnover
        if activations[row_number]:
            activation_factor = 1.0 - activation_cost
            if not np.isfinite(activation_factor) or activation_factor <= 0.0:
                _raise_breach(
                    close_time,
                    "activation_equity_exhausted",
                    effective_weights,
                    equity,
                    config,
                    trial_id,
                    {"activationEquityFactor": activation_factor},
                )
            post_activation_equity = equity * activation_factor
            post_activation_weights = effective_weights / activation_factor
            post_activation_drawdown = 1.0 - post_activation_equity / max(
                peak_equity, post_activation_equity
            )
            (
                post_activation_shock_equity,
                post_activation_shocked_gross,
                post_activation_shock_coverage,
            ) = _shock_state(post_activation_weights, config)
            post_activation_state = {
                "equity": post_activation_equity,
                "drawdown": post_activation_drawdown,
                "endpointGrossLeverage": float(
                    np.abs(post_activation_weights).sum()
                ),
                "maximumAbsoluteSymbolWeight": float(
                    np.max(np.abs(post_activation_weights))
                ),
                "shockEquityFraction": post_activation_shock_equity,
                "shockedGrossLeverage": post_activation_shocked_gross,
                "shockMaintenanceCoverage": post_activation_shock_coverage,
            }
            _check_endpoint_risk(
                close_time,
                post_activation_weights,
                post_activation_equity,
                post_activation_state,
                config,
                trial_id,
                evaluation_stage="post_activation",
            )

        price_gross = float(np.dot(effective_weights, price_returns[row_number]))
        funding_cashflow = float(np.dot(effective_weights, funding[row_number]))
        gross = price_gross + funding_cashflow
        interval_factor = 1.0 + gross - activation_cost
        if not np.isfinite(
            (price_gross, funding_cashflow, gross, activation_cost, interval_factor)
        ).all():
            _raise_breach(
                close_time,
                "non_finite_state",
                effective_weights,
                equity,
                config,
                trial_id,
                {"field": "intervalAccounting"},
            )
        if interval_factor <= 0.0:
            _raise_breach(
                close_time,
                "interval_equity_exhausted",
                effective_weights,
                equity,
                config,
                trial_id,
                {"intervalEquityFactor": interval_factor},
            )

        next_equity = equity * interval_factor
        endpoint_weights = (
            effective_weights * (1.0 + price_returns[row_number]) / interval_factor
        )
        if not np.isfinite(endpoint_weights).all() or not np.isfinite(next_equity):
            _raise_breach(
                close_time,
                "non_finite_state",
                effective_weights,
                equity,
                config,
                trial_id,
                {"field": "endpointState"},
            )

        peak_after_interval = max(peak_equity, next_equity)
        drawdown = 1.0 - next_equity / peak_after_interval
        endpoint_gross = float(np.abs(endpoint_weights).sum())
        maximum_symbol = float(np.max(np.abs(endpoint_weights)))
        shock_equity_fraction, shocked_gross, shock_coverage = _shock_state(
            endpoint_weights, config
        )
        risk_state = {
            "equity": next_equity,
            "drawdown": drawdown,
            "endpointGrossLeverage": endpoint_gross,
            "maximumAbsoluteSymbolWeight": maximum_symbol,
            "shockEquityFraction": shock_equity_fraction,
            "shockedGrossLeverage": shocked_gross,
            "shockMaintenanceCoverage": shock_coverage,
        }
        _check_endpoint_risk(
            close_time,
            endpoint_weights,
            next_equity,
            risk_state,
            config,
            trial_id,
        )

        terminal_turnover = 0.0
        terminal_charged_turnover = 0.0
        is_terminal = row_number == len(close_times) - 2
        if is_terminal and config.charge_terminal_liquidation:
            equity_before_terminal_liquidation = next_equity
            terminal_turnover = endpoint_gross
            if terminal_turnover > config.maximum_terminal_turnover:
                _raise_breach(
                    close_time,
                    "maximum_terminal_turnover",
                    endpoint_weights,
                    next_equity,
                    config,
                    trial_id,
                    {
                        "observed": terminal_turnover,
                        "limit": config.maximum_terminal_turnover,
                    },
                )
            terminal_charged_turnover = interval_factor * terminal_turnover
            terminal_factor = 1.0 - config.cost_per_turnover * terminal_turnover
            if terminal_factor <= 0.0 or not np.isfinite(terminal_factor):
                _raise_breach(
                    close_time,
                    "terminal_liquidation_equity_exhausted",
                    endpoint_weights,
                    next_equity,
                    config,
                    trial_id,
                    {"terminalEquityFactor": terminal_factor},
                )
            next_equity *= terminal_factor
            interval_factor *= terminal_factor
            drawdown = 1.0 - next_equity / max(peak_after_interval, next_equity)
            if next_equity < config.equity_floor:
                _raise_breach(
                    close_time,
                    "cumulative_equity_floor",
                    endpoint_weights,
                    equity_before_terminal_liquidation,
                    config,
                    trial_id,
                    {
                        "observed": next_equity,
                        "limit": config.equity_floor,
                        "equityAfterModeledLiquidation": next_equity,
                    },
                )
            if drawdown > config.maximum_drawdown:
                _raise_breach(
                    close_time,
                    "maximum_drawdown",
                    endpoint_weights,
                    equity_before_terminal_liquidation,
                    config,
                    trial_id,
                    {
                        "observed": drawdown,
                        "limit": config.maximum_drawdown,
                        "equityAfterModeledLiquidation": next_equity,
                    },
                )

        total_turnover = activation_turnover + terminal_charged_turnover
        cost = config.cost_per_turnover * total_turnover
        net = interval_factor - 1.0
        active = int(np.count_nonzero(np.abs(effective_weights) > 1e-12))
        rows.append(
            (
                close_time,
                price_gross,
                funding_cashflow,
                gross,
                activation_turnover,
                terminal_turnover,
                terminal_charged_turnover,
                total_turnover,
                cost,
                net,
                active,
                next_equity,
                drawdown,
                endpoint_gross,
                maximum_symbol,
                shock_equity_fraction,
                shocked_gross,
                shock_coverage,
                *effective_weights.tolist(),
            )
        )
        equity = next_equity
        peak_equity = max(peak_after_interval, next_equity)
        drifted_weights = (
            np.zeros_like(endpoint_weights) if is_terminal else endpoint_weights
        )

    columns = [
        "closeTime",
        "priceGross",
        "fundingCashflow",
        "gross",
        "activationTurnover",
        "terminalTurnover",
        "terminalChargedTurnover",
        "turnover",
        "cost",
        "net",
        "active",
        "equity",
        "drawdown",
        "endpointGrossLeverage",
        "maximumAbsoluteSymbolWeight",
        "shockEquityFraction",
        "shockedGrossLeverage",
        "shockMaintenanceCoverage",
        *(f"weight_{symbol}" for symbol in close_values.columns),
    ]
    return pd.DataFrame(rows, columns=columns).set_index("closeTime")


def run_trial(
    close: pd.DataFrame,
    residual_momentum_by_horizon: Mapping[int, pd.DataFrame],
    settlements: Sequence[FundingSettlement],
    config: RiskControlledReversalConfig,
    spec: RiskControlledReversalSpec,
) -> pd.DataFrame:
    """Evaluate one fixed treatment or matched control path."""
    _validate_config(config)
    _validate_spec(spec, config.interval_ms)
    try:
        momentum = residual_momentum_by_horizon[spec.horizon_hours]
    except KeyError as error:
        raise ValueError(
            f"missing residual momentum for {spec.horizon_hours}h horizon"
        ) from error
    targets = weights_for_trial(momentum, spec, config)
    activations = np.zeros(len(targets), dtype=bool)
    if config.signal_delay_bars < len(activations):
        activations[config.signal_delay_bars :] = True
    try:
        return evaluate_drifted_intervals(
            close,
            targets,
            activations,
            settlements,
            config,
            trial_id=spec.trial_id,
        )
    except RiskConstraintBreach as error:
        if error.trial_id is not None:
            raise
        raise RiskConstraintBreach(
            error.interval_left_close_time,
            error.reason,
            error.evidence,
            spec.trial_id,
        ) from error


def run_trial_matrix(
    close: pd.DataFrame,
    residual_momentum_by_horizon: Mapping[int, pd.DataFrame],
    settlements: Sequence[FundingSettlement],
    config: RiskControlledReversalConfig,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    tuple[RiskControlledReversalSpec, ...],
]:
    """Evaluate every registered path before any candidate selection."""
    specs = campaign_specs(config.interval_ms)
    events = tuple(settlements)
    details: dict[str, pd.DataFrame] = {}
    for spec in specs:
        details[spec.trial_id] = run_trial(
            close, residual_momentum_by_horizon, events, config, spec
        )
    matrix = pd.DataFrame(
        {trial_id: detail["net"] for trial_id, detail in details.items()}
    )
    if tuple(matrix.columns) != TRIAL_IDS:
        raise AssertionError("trial matrix does not match the fixed ledger")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("trial matrix contains non-finite returns")
    return matrix, details, specs


def _retained_or_extreme(
    incumbent: int | None, ranked: np.ndarray, exit_rank: int
) -> int:
    retained = ranked[:exit_rank]
    if incumbent is not None and np.any(retained == incumbent):
        return incumbent
    return int(ranked[0])


def _shock_state(
    endpoint_weights: np.ndarray,
    config: RiskControlledReversalConfig,
) -> tuple[float, float, float]:
    shock_returns = np.where(
        endpoint_weights > 0.0,
        -config.adverse_shock_fraction,
        np.where(endpoint_weights < 0.0, config.adverse_shock_fraction, 0.0),
    )
    shocked_notionals = endpoint_weights * (1.0 + shock_returns)
    shocked_equity = float(1.0 + np.dot(endpoint_weights, shock_returns))
    shocked_gross = float(np.abs(shocked_notionals).sum())
    requirement = (
        config.maintenance_margin_rate + config.liquidation_reserve_rate
    ) * shocked_gross
    coverage = (
        shocked_equity / requirement
        if requirement > 0.0
        else config.minimum_shock_maintenance_coverage
    )
    return shocked_equity, shocked_gross, float(coverage)


def _check_endpoint_risk(
    close_time: int,
    endpoint_weights: np.ndarray,
    equity: float,
    state: Mapping[str, float],
    config: RiskControlledReversalConfig,
    trial_id: str | None,
    *,
    evaluation_stage: str = "outcome_endpoint",
) -> None:
    checks = (
        ("cumulative_equity_floor", state["equity"] < config.equity_floor, "equity", config.equity_floor),
        ("maximum_drawdown", state["drawdown"] > config.maximum_drawdown, "drawdown", config.maximum_drawdown),
        (
            "maximum_endpoint_gross_leverage",
            state["endpointGrossLeverage"]
            > config.maximum_endpoint_gross_leverage,
            "endpointGrossLeverage",
            config.maximum_endpoint_gross_leverage,
        ),
        (
            "maximum_symbol_weight",
            state["maximumAbsoluteSymbolWeight"] > config.maximum_symbol_weight,
            "maximumAbsoluteSymbolWeight",
            config.maximum_symbol_weight,
        ),
        (
            "minimum_shock_equity_fraction",
            state["shockEquityFraction"]
            < config.minimum_shock_equity_fraction,
            "shockEquityFraction",
            config.minimum_shock_equity_fraction,
        ),
        (
            "minimum_shock_maintenance_coverage",
            state["shockMaintenanceCoverage"]
            < config.minimum_shock_maintenance_coverage,
            "shockMaintenanceCoverage",
            config.minimum_shock_maintenance_coverage,
        ),
    )
    for reason, breached, field, limit in checks:
        if breached:
            _raise_breach(
                close_time,
                reason,
                endpoint_weights,
                equity,
                config,
                trial_id,
                {
                    "evaluationStage": evaluation_stage,
                    "field": field,
                    "observed": state[field],
                    "limit": limit,
                },
            )


def _raise_breach(
    close_time: int,
    reason: str,
    liquidation_weights: np.ndarray,
    equity: float,
    config: RiskControlledReversalConfig,
    trial_id: str | None,
    evidence: Mapping[str, object],
) -> None:
    liquidation_turnover = float(np.abs(liquidation_weights).sum())
    liquidation_cost_fraction = config.cost_per_turnover * liquidation_turnover
    raise RiskConstraintBreach(
        int(close_time),
        reason,
        {
            **dict(evidence),
            "equityBeforeModeledLiquidation": float(equity),
            "modeledImmediateLiquidationTurnover": liquidation_turnover,
            "modeledImmediateLiquidationCostFraction": liquidation_cost_fraction,
            "modeledImmediateLiquidationCostEquity": float(
                equity * liquidation_cost_fraction
            ),
        },
        trial_id,
    )


def _json_safe_mapping(evidence: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_safe_value(value) for key, value in evidence.items()}


def _json_safe_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _validate_config(config: RiskControlledReversalConfig) -> None:
    if not isinstance(config, RiskControlledReversalConfig):
        raise TypeError("config must be a RiskControlledReversalConfig")
    campaign_specs(config.interval_ms)
    expected = {
        "rebalance_anchor_open_time": REBALANCE_ANCHOR_OPEN_TIME,
        "gross_exposure": 0.5,
        "equity_floor": 0.75,
        "maximum_drawdown": 0.20,
        "maximum_endpoint_gross_leverage": 1.0,
        "maximum_symbol_weight": 0.40,
        "maximum_activation_turnover": 1.25,
        "maximum_terminal_turnover": 1.25,
        "adverse_shock_fraction": 0.25,
        "maintenance_margin_rate": 0.10,
        "liquidation_reserve_rate": 0.01,
        "minimum_shock_equity_fraction": 0.50,
        "minimum_shock_maintenance_coverage": 2.0,
        "charge_terminal_liquidation": True,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"{name} changed from the locked campaign")
    if config.rebalance_anchor_open_time % config.interval_ms != 0:
        raise ValueError("rebalance anchor must lie on the absolute interval grid")
    if config.cost_per_turnover not in (0.001, 0.002):
        raise ValueError("cost_per_turnover must be the base or doubled stress cost")
    if isinstance(config.signal_delay_bars, (bool, np.bool_)) or not isinstance(
        config.signal_delay_bars, (int, np.integer)
    ):
        raise ValueError("signal_delay_bars must be the base or delayed stress value")
    if config.signal_delay_bars not in (1, 2):
        raise ValueError("signal_delay_bars must be the base or delayed stress value")


def _validate_spec(spec: RiskControlledReversalSpec, interval_ms: int) -> None:
    if not isinstance(spec, RiskControlledReversalSpec):
        raise TypeError("spec must be a RiskControlledReversalSpec")
    if spec not in campaign_specs(interval_ms):
        raise ValueError("spec is not a member of the fixed risk-controlled ledger")
