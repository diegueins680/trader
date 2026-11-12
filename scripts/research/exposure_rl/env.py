"""exposure_env_v1: bounded target-exposure control of one USD-M perpetual.

Timeline for a decision at the close of bar ``t`` (all information <= close t):

1. observation(t)      features of bars <= t plus account state marked at close t
2. proposal            raw policy output in [-1, 1] (untrusted)
3. shield              permitted integer target exposure, or no intent
4. submission          simulated intent, submitted after close t
5. fill                at open of bar t+1 (base) or close of bar t+1 (delay stress)
6. funding             events at exactly open(t+1) settle on the pre-fill position,
                       events in (open(t+1), close(t+1)] on the position then held
7. settlement          equity marked at close t+1; reward = (E[t+1] - E[t]) / E[t]

Price dynamics and funding are exogenous (price-taker replay of 6h OHLCV). The
only action-dependent dynamics are the account's own position, costs and
equity. This is model-based counterfactual evidence under those assumptions; it
supports no queue-position, intrabar or market-impact claim.

Episode ends: ``terminated`` only when equity reaches the insolvency floor (the
position is then liquidated at cost; no bootstrap). ``truncated`` on the
training time limit or at the end of the permitted data range (no liquidation;
the caller bootstraps from the returned final observation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import shield as S
from .market import BURN_IN, MARKET_FEATURES, Series

ENV_VERSION = "exposure_env_v1"
STATE_FEATURES = ("exposure", "drawdown", "risk_off", "bars_since_fill")
OBS_DIM = len(MARKET_FEATURES) + len(STATE_FEATURES)
REWARD_SCALE = 100.0  # learner reward = REWARD_SCALE * one-step net equity return
PEAK_WINDOW = 120  # bars (30 days) over which the drawdown peak is measured
EQUITY_FLOOR = 0.5  # insolvency floor as a fraction of initial equity
INITIAL_EQUITY = 1.0
BASE_COST = 0.0008  # per unit traded notional: 5 bp taker + 1.5 bp slippage + 1.5 bp half-spread


@dataclass(frozen=True)
class Execution:
    cost: float = BASE_COST
    fill_at_close: bool = False  # one-bar execution delay stress
    fill_fraction: float = 1.0  # partial-fill stress
    miss_probability: float = 0.0  # missed-fill stress
    funding_multiplier: float = 1.0
    stale_probability: float = 0.0  # account snapshot not fresh -> reduce-only
    corrupt_probability: float = 0.0  # invalid snapshot -> recovery required
    vol_cost_slope: float = 0.0  # extra cost fraction per unit of |ret1_z| (spread widening)
    shield_enabled: bool = True  # False only for labelled diagnostic ablations


STRESSES = {
    "base": Execution(),
    "cost1_5x": Execution(cost=1.5 * BASE_COST),
    "cost2x": Execution(cost=2.0 * BASE_COST),
    "severe25bp_delay": Execution(cost=0.0025, fill_at_close=True),
    "delay1bar": Execution(fill_at_close=True),
    "partial50pct": Execution(fill_fraction=0.5),
    "missed10pct": Execution(miss_probability=0.10),
    "funding2x": Execution(funding_multiplier=2.0),
    "stale5pct": Execution(stale_probability=0.05),
    "corrupt1pct": Execution(corrupt_probability=0.01),
    "spread_widening": Execution(vol_cost_slope=0.5),
}


def _away_from_zero(values: np.ndarray) -> np.ndarray:
    return (np.sign(values) * np.ceil(np.abs(values) - 1e-9)).astype(np.int64)


def usable_symbols(panel: dict[str, Series], ranges: dict[str, tuple[int, int]]) -> list[str]:
    """Symbols with at least one decision after burn-in inside their range."""
    return [s for s in panel if s in ranges and ranges[s][1] - max(ranges[s][0], BURN_IN) >= 2]


class ExposureEnv:
    """Vectorised environment over ``lanes`` independent single-symbol accounts."""

    def __init__(
        self,
        panel: dict[str, Series],
        ranges: dict[str, tuple[int, int]],
        lanes: int,
        seed: int,
        execution: Execution = Execution(),
        episode_length: int | None = 256,
    ) -> None:
        self.symbols = usable_symbols(panel, ranges)
        if not self.symbols:
            raise ValueError("no symbol has a usable decision range")
        self.execution = execution
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        offsets, low, high = [], [], []
        total = 0
        for symbol in self.symbols:
            series = panel[symbol]
            first, stop = ranges[symbol]
            first = max(first, BURN_IN)
            if not 0 <= first < stop <= len(series):
                raise ValueError(f"{symbol} decision range is outside the series")
            offsets.append(total)
            low.append(total + first)  # first decision bar
            high.append(total + stop - 1)  # last bar whose close may be an outcome
            total += len(series)
        cat = lambda name: np.concatenate([getattr(panel[s], name) for s in self.symbols])
        self.open, self.close = cat("open"), cat("close")
        self.f_open, self.f_inside = cat("funding_at_open"), cat("funding_inside")
        self.features = np.concatenate([panel[s].features for s in self.symbols])
        self.low, self.high, self.offsets = np.asarray(low), np.asarray(high), np.asarray(offsets)
        self.lanes = lanes
        self.sym = np.zeros(lanes, dtype=np.int64)
        self.t = np.zeros(lanes, dtype=np.int64)
        self.equity = np.ones(lanes)
        self.units = np.zeros(lanes)
        self.history = np.ones((lanes, PEAK_WINDOW))  # trailing equity marks
        self.cursor = np.zeros(lanes, dtype=np.int64)
        self.age = np.zeros(lanes)
        self.steps = np.zeros(lanes, dtype=np.int64)
        self.active = np.ones(lanes, dtype=bool)

    # -- episode management -------------------------------------------------
    def _reset_lanes(self, mask: np.ndarray, sequential: bool = False) -> None:
        index = np.flatnonzero(mask)
        if not len(index):
            return
        if sequential:
            self.sym[index] = index % len(self.symbols)
            self.t[index] = self.low[self.sym[index]]
        else:
            span = (self.high - self.low).astype(np.float64)
            self.sym[index] = self.rng.choice(len(self.symbols), size=len(index), p=span / span.sum())
            room = self.high[self.sym[index]] - self.low[self.sym[index]]
            need = 1 if self.episode_length is None else min(self.episode_length, 16)
            latest = np.maximum(room - need, 0)
            self.t[index] = self.low[self.sym[index]] + (self.rng.random(len(index)) * (latest + 1)).astype(np.int64)
        self.equity[index] = INITIAL_EQUITY
        self.units[index] = 0.0
        self.history[index] = INITIAL_EQUITY
        self.cursor[index] = 0
        self.age[index] = 0.0
        self.steps[index] = 0
        self.active[index] = True

    def reset(self, sequential: bool = False) -> np.ndarray:
        """Random starts for training, or one lane per symbol from its range start."""
        if sequential and self.lanes != len(self.symbols):
            raise ValueError("sequential evaluation needs one lane per symbol")
        self._reset_lanes(np.ones(self.lanes, dtype=bool), sequential)
        return self.observe()

    def reset_at(self, symbol_index: np.ndarray, bar: np.ndarray) -> np.ndarray:
        """Start every lane flat at the given (symbol, decision bar)."""
        self._reset_lanes(np.ones(self.lanes, dtype=bool))
        self.sym = np.asarray(symbol_index, dtype=np.int64).copy()
        self.t = np.asarray(bar, dtype=np.int64).copy()
        if np.any(self.t < self.low[self.sym]) or np.any(self.t >= self.high[self.sym]):
            raise ValueError("start state is outside the permitted decision range")
        return self.observe()

    # -- observation --------------------------------------------------------
    def snapshot(self):
        exposure = _away_from_zero(self.units * self.close[self.t] / self.equity * S.SCALE)
        # Drawdown from the trailing PEAK_WINDOW-bar equity peak (no permanent lockout).
        peak = np.maximum(self.history.max(axis=1), self.equity)
        drawdown = np.ceil(np.clip(1.0 - self.equity / peak, 0.0, 1.0) * S.SCALE - 1e-9).astype(np.int64)
        return exposure, drawdown

    def observe(self) -> np.ndarray:
        exposure, drawdown = self.snapshot()
        state = np.stack(
            [
                np.clip(exposure / S.CAP, -2.0, 2.0),
                np.clip(drawdown / S.DD_LIMIT, 0.0, 2.0),
                (drawdown >= S.DD_LIMIT).astype(np.float64),
                np.clip(self.age / 28.0, 0.0, 1.0),
            ],
            axis=1,
        )
        return np.concatenate([self.features[self.t], state], axis=1)

    # -- transition ---------------------------------------------------------
    def step(self, raw: np.ndarray, auto_reset: bool = True):
        ex = self.execution
        raw = np.asarray(raw, dtype=np.float64).reshape(self.lanes)
        exposure, drawdown = self.snapshot()
        fresh = self.rng.random(self.lanes) >= ex.stale_probability if ex.stale_probability else np.ones(self.lanes, dtype=bool)
        seen_exposure = exposure
        if ex.corrupt_probability:
            corrupt = self.rng.random(self.lanes) < ex.corrupt_probability
            seen_exposure = np.where(corrupt, S.SNAPSHOT_BOUND + 1, exposure)
        if ex.shield_enabled:
            kind, target, reasons = S.shield_batch(seen_exposure, drawdown, fresh, np.full(self.lanes, S.CAP), raw)
            target = np.where(kind == S.RECOVERY_REQUIRED, exposure, target)
        else:  # diagnostic only: raw proposal executed without projection
            target = np.trunc(np.clip(np.nan_to_num(raw), -1.0, 1.0) * S.CAP).astype(np.int64)
            kind = np.full(self.lanes, S.PERMITTED, dtype=np.int64)
            reasons = np.zeros(self.lanes, dtype=np.int64)
        intent = (target != exposure) & (kind != S.RECOVERY_REQUIRED)
        t = self.t
        b = np.where(self.active, t + 1, t)  # frozen lanes re-read their last bar
        equity, units = self.equity, self.units
        price, open_, close = self.close[t], self.open[b], self.close[b]
        funding_open = units * self.f_open[b] * ex.funding_multiplier
        equity_open = equity + units * (open_ - price) - funding_open
        filled = intent.copy()
        if ex.miss_probability:
            filled &= self.rng.random(self.lanes) >= ex.miss_probability
        rate = ex.cost * (1.0 + ex.vol_cost_slope * np.abs(self.features[t, 0]))
        weight = target / S.SCALE
        if ex.fill_at_close:
            funding_inside = units * self.f_inside[b] * ex.funding_multiplier
            equity_fill = equity_open + units * (close - open_) - funding_inside
            wanted = weight * equity_fill / close
            delta = np.where(filled, ex.fill_fraction * (wanted - units), 0.0)
            cost = np.abs(delta) * close * rate
            new_units = units + delta
            new_equity = equity_fill - cost
            gross = units * (close - price)
        else:
            wanted = weight * equity_open / open_
            delta = np.where(filled, ex.fill_fraction * (wanted - units), 0.0)
            cost = np.abs(delta) * open_ * rate
            new_units = units + delta
            funding_inside = new_units * self.f_inside[b] * ex.funding_multiplier
            new_equity = equity_open - cost + new_units * (close - open_) - funding_inside
            gross = units * (open_ - price) + new_units * (close - open_)
        funding = funding_open + funding_inside
        terminated = new_equity <= EQUITY_FLOOR * INITIAL_EQUITY
        liquidation = np.where(terminated, np.abs(new_units) * close * rate, 0.0)
        new_equity = new_equity - liquidation
        new_units = np.where(terminated, 0.0, new_units)
        reward = (new_equity - equity) / equity
        turnover = np.abs(delta) * np.where(ex.fill_at_close, close, open_) / equity
        self.steps = self.steps + 1
        time_limit = np.zeros(self.lanes, dtype=bool) if self.episode_length is None else self.steps >= self.episode_length
        exhausted = b >= self.high[self.sym]
        truncated = (time_limit | exhausted) & ~terminated
        info = {
            "symbol": self.sym.copy(),
            "bar": b.copy(),
            "proposed": raw.copy(),
            "kind": kind,
            "reasons": reasons,
            "exposure_before": exposure,
            "permitted": target,
            "intent": intent,
            "filled": filled & (delta != 0.0),
            "gross": gross,
            "cost": cost + liquidation,
            "funding": funding,
            "turnover": turnover,
            "equity_before": equity.copy(),
            "equity_after": new_equity.copy(),
            "net_return": reward.copy(),
            "active": self.active.copy(),
            # Cost, as a return on equity, of closing the remaining position at this close.
            "closing_cost_return": np.abs(new_units) * close * rate / np.maximum(new_equity, 1e-12),
        }
        live = self.active
        # Lanes that already finished (evaluation mode) are frozen, not re-traded.
        self.equity = np.where(live, new_equity, equity)
        self.units = np.where(live, new_units, units)
        self.t = np.where(live, b, t)
        rows = np.flatnonzero(live)
        self.history[rows, self.cursor[rows]] = self.equity[rows]
        self.cursor[rows] = (self.cursor[rows] + 1) % PEAK_WINDOW
        self.age = np.where(info["filled"] & live, 0.0, self.age + 1.0)
        info["exposure_after"] = self.snapshot()[0]
        final_obs = self.observe()
        done = terminated | truncated
        if auto_reset:
            self._reset_lanes(done)
            next_obs = self.observe()
        else:
            self.active = self.active & ~done
            next_obs = final_obs
        return next_obs, REWARD_SCALE * reward, terminated, truncated, final_obs, info
