"""Version-1 offline inventory replay. No network, credential or order interface.

This is a deliberately limited close-price simulator, not an execution model
certified for production. Source data must be admitted separately by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np

ENVIRONMENT = "sequential_replay_v1"
OBSERVATION = "close_inventory_v1"
ACTIONS = np.array([-0.25, 0.0, 0.25])
FEATURE_COUNT = 12


def _integer(value) -> bool:
    return type(value) is int or (isinstance(value, np.integer) and value.dtype.kind in "iu")


def _real_series(value) -> bool:
    """Inspect representation only; never scan future market values at admission."""
    return (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value) and
            value.ndim == 1 and value.dtype.kind in "iuf")


def _feature_vector(value) -> bool:
    return _real_series(value) and value.shape == (6,) and bool(np.isfinite(value).all())


def market_features(prices: np.ndarray, t: int) -> np.ndarray | None:
    """Read only the trailing prefix, including the completed decision bar."""
    if not _real_series(prices) or not _integer(t) or not 24 <= t < len(prices):
        return None
    t = int(t)
    p = np.asarray(prices[t - 24:t + 1], dtype=float)
    if not np.isfinite(p).all() or np.any(p <= 0):
        return None
    r = p[1:] / p[:-1] - 1
    x = np.array([r[-1], p[-1] / p[-4] - 1, p[-1] / p[-7] - 1,
                  p[-1] / p[0] - 1, np.std(r[-6:]), np.std(r)])
    return x if np.isfinite(x).all() else None


@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    std: np.ndarray
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        snapshots = {}
        for name in ("mean", "std", "low", "high"):
            value = getattr(self, name)
            if not _real_series(value) or value.shape != (6,):
                raise ValueError("invalid scale parameter shape or type")
            # Immutable backing bytes prevent caller aliasing and write-flag reactivation.
            with np.errstate(over="ignore", invalid="ignore"):
                snapshot = np.frombuffer(np.asarray(value, dtype=float).tobytes(), dtype=float)
            if not _feature_vector(snapshot):
                raise ValueError("non-finite scale parameter")
            snapshots[name] = snapshot
        if np.any(snapshots["std"] <= 0) or np.any(snapshots["low"] > snapshots["high"]):
            raise ValueError("invalid scale deviation or support bounds")
        for name, value in snapshots.items():
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, prefixes: list[np.ndarray]) -> Scale:
        if (not isinstance(prefixes, (list, tuple)) or not prefixes or
            any(not _real_series(p) or len(p) < 25 for p in prefixes)):
            raise ValueError("incomplete training prefixes")
        rows = [market_features(p, t) for p in prefixes for t in range(24, len(p))]
        if not rows or any(x is None for x in rows):
            raise ValueError("incomplete training observations")
        x = np.array(rows)
        values = [x.mean(0), np.maximum(x.std(0), 1e-8), x.min(0), x.max(0)]
        return cls(*values)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not _feature_vector(x):
            raise ValueError("invalid observation")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = (np.asarray(x, dtype=float) - self.mean) / self.std
        if not _feature_vector(result):
            raise ValueError("non-finite normalized observation")
        return result

    def supported(self, x: np.ndarray) -> bool:
        return bool(_feature_vector(x) and np.all(x >= self.low) and np.all(x <= self.high))


@dataclass(frozen=True)
class Execution:
    cost_multiplier: float = 1.0
    extra_delay: int = 0
    fill_fraction: float = 1.0
    miss_every: int = 0
    impact_bps: float = 0.0
    funding_multiplier: float = 1.0
    risk_penalty: float = 0.01

    def __post_init__(self) -> None:
        vals = (self.cost_multiplier, self.fill_fraction, self.impact_bps,
                self.funding_multiplier, self.risk_penalty)
        if not all(_finite_real(v) and v >= 0 for v in vals) or not 0 < self.fill_fraction <= 1:
            raise ValueError("invalid execution assumptions")
        if type(self.extra_delay) is not int or self.extra_delay not in (0, 1):
            raise ValueError("invalid delay")
        if type(self.miss_every) is not int or self.miss_every < 0:
            raise ValueError("invalid missed-fill schedule")


def _finite_real(value) -> bool:
    """Accept real numeric scalars without coercing booleans, arrays or complex values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def shield(action: float, *, enabled: bool, valid: bool,
           ownership: bool = True, elapsed_ms: float = 0.0) -> tuple[float | None, str]:
    """A rejected proposal is absent, never an executable instruction to flatten."""
    if enabled is not True:
        return None, "disabled"
    if valid is not True or ownership is not True:
        return None, "invalid_observation_or_position"
    if not _finite_real(elapsed_ms) or not 0 <= elapsed_ms <= 20:
        return None, "timeout"
    if not _finite_real(action) or action not in (-0.25, 0.0, 0.25):
        return None, "invalid_action"
    return float(action), "research_proposal_only"


class Replay:
    """Inventory units and cash equity; transitions are bounded by [start, stop).

    funding[t] is sum(mark * rate) for events in (close[t-1], close[t]].
    Policy observes no funding outcome and no price beyond its decision time.
    """
    def __init__(self, prices: np.ndarray, funding: np.ndarray, start: int, stop: int,
                 horizon: int, scale: Scale, execution: Execution = Execution(),
                 *, enabled: bool = False) -> None:
        if not _real_series(prices) or not _real_series(funding):
            raise ValueError("invalid market series representation")
        if not all(_integer(v) for v in (start, stop, horizon)):
            raise ValueError("episode indices and horizon must be integers")
        start, stop, horizon = int(start), int(stop), int(horizon)
        if horizon not in (1, 3, 6) or not 24 <= start < stop - 1 <= len(prices) - 1:
            raise ValueError("invalid episode boundaries")
        if len(prices) != len(funding):
            raise ValueError("funding shape mismatch")
        self.prices, self.funding = prices, funding
        self.start, self.stop, self.t = start, stop, start
        self.horizon, self.scale, self.execution = horizon, scale, execution
        self.enabled = enabled
        self.units, self.equity, self.peak = 0.0, 1.0, 1.0
        self.pending: tuple[int, float] | None = None
        self.done, self.failure = False, None
        self.fills, self.proposals, self.rejections, self.modifications = 0, 0, 0, 0
        self.rows: list[dict] = []
        self.actions: list[float] = []

    def observation(self) -> np.ndarray | None:
        x = market_features(self.prices, self.t)
        if x is None or not isfinite(self.equity) or self.equity <= 0:
            return None
        p = self.prices[self.t]
        target, delay = (0.0, 0.0) if self.pending is None else (self.pending[1], self.pending[0] - self.t)
        state = [self.units * p / self.equity, 1 - self.equity / self.peak,
                 self.equity, target, delay / 2, (self.stop - 1 - self.t) / (self.stop - 1 - self.start)]
        try:
            normalized = self.scale.transform(x)
        except ValueError:
            return None
        obs = np.r_[normalized, state]
        return obs if np.isfinite(obs).all() else None

    def supported(self) -> bool:
        x = market_features(self.prices, self.t)
        return x is not None and self.scale.supported(x)

    def _trade(self, target: float, terminal: bool = False) -> dict:
        p, e, cfg = self.prices[self.t], self.equity, self.execution
        old = self.units
        desired = target * e / p
        # Terminal accounting always closes all units; sensitivity applies to entries/rebalances.
        new = desired if terminal else old + cfg.fill_fraction * (desired - old)
        if not terminal and cfg.miss_every and self.proposals % cfg.miss_every == 0:
            new = old
        turnover = abs(new - old) * p / e
        if not terminal and turnover > 0.50 + 1e-12:
            # Reject, never project an unsafe increase or silently ignore a breach.
            self.failure = "turnover_limit"
            self.rejections += 1
            return {"turnover": 0.0, "fee": 0.0, "spread": 0.0, "slippage": 0.0, "impact": 0.0}
        if not terminal and abs(new - desired) > 1e-15:
            self.modifications += 1
        cash = abs(new - old) * p
        terms = {"turnover": cash, "fee": cash * 0.0005 * cfg.cost_multiplier,
                 "spread": cash * 0.00005 * cfg.cost_multiplier,
                 "slippage": cash * 0.00045 * cfg.cost_multiplier,
                 "impact": cash * cfg.impact_bps * 1e-4 * np.sqrt(turnover)}
        self.equity -= sum(terms[k] for k in ("fee", "spread", "slippage", "impact"))
        self.units = new
        self.fills += int(cash > 1e-15)
        return terms

    def _risk(self) -> str | None:
        if not isfinite(self.equity) or self.equity <= 0:
            return "equity_exhausted"
        if self.equity < 0.8:
            return "capital_floor"
        if 1 - self.equity / self.peak > 0.15:
            return "drawdown_limit"
        if abs(self.units * self.prices[self.t] / self.equity) > 0.35:
            return "endpoint_exposure"
        return None

    def step(self, action: float, *, valid: bool = True, ownership: bool = True,
             elapsed_ms: float = 0.0) -> tuple[np.ndarray | None, float, bool]:
        if self.done:
            raise ValueError("episode already terminated")
        proposal, reason = shield(action, enabled=self.enabled, valid=valid,
                                  ownership=ownership, elapsed_ms=elapsed_ms)
        if proposal is not None and self.observation() is None:
            proposal, reason = None, "invalid_observation_or_position"
        if proposal is None:
            self.rejections += 1
            self.failure, self.done = reason, True
            # No actual or simulated new order; caller records incomplete path.
            return None, 0.0, True
        before, penalty = self.equity, 0.0
        self.proposals += 1
        self.actions.append(proposal)
        if self.pending is None:
            self.pending = (self.t + 1 + self.execution.extra_delay, proposal)
        else:
            self.rejections += 1
        end = min(self.t + self.horizon, self.stop - 1)
        while self.t < end:
            left, old_equity = self.t, self.equity
            p0, p1 = self.prices[left], self.prices[left + 1]
            f = self.funding[left + 1]
            if not all(_finite_real(v) for v in (p0, p1, f)) or min(p0, p1) <= 0:
                self.failure, self.done = "invalid_market_transition", True
                break
            exposure = self.units * p0 / old_equity
            x = market_features(self.prices, left)
            if x is None:
                self.failure, self.done = "invalid_observation", True
                break
            penalty += exposure**2 * x[5]**2
            gross, funding = self.units * (p1 - p0), -self.units * f * self.execution.funding_multiplier
            self.equity += gross + funding
            self.t += 1
            self.peak = max(self.peak, self.equity)
            self.failure = self._risk()
            costs = dict.fromkeys(("turnover", "fee", "spread", "slippage", "impact"), 0.0)
            if self.failure is None and self.pending is not None and self.pending[0] <= self.t:
                costs = self._trade(self.pending[1])
                self.pending = None
                self.failure = self.failure or self._risk()
            terminal = self.failure is not None or self.t == self.stop - 1
            if terminal:
                self.pending = None
                if isfinite(self.equity) and self.equity > 0:
                    liquidation = self._trade(0.0, terminal=True)
                    costs = {k: costs[k] + liquidation[k] for k in costs}
                self.done = True
                self.failure = self.failure or self._risk()
            self.rows.append({"t": self.t, "net": self.equity / old_equity - 1,
                              "equity": self.equity, "gross": gross, "funding": funding,
                              "exposure": exposure, "drawdown": 1 - self.equity / self.peak,
                              "rewardPenalty": 100 * self.execution.risk_penalty * exposure**2 * x[5]**2,
                              **costs})
            if self.done:
                break
        reward = 100 * ((self.equity - before) / before - self.execution.risk_penalty * penalty)
        if not isfinite(reward):
            raise ValueError("non-finite reward; terminal failure must be recorded")
        return (None if self.done else self.observation()), reward, self.done


def _admit_training_transition(env: Replay, left: int, nxt: np.ndarray | None,
                               reward: float, done: bool) -> None:
    """Incomplete market paths are not terminal learning targets or cash samples."""
    accounted_risk = ("capital_floor", "drawdown_limit", "endpoint_exposure", "turnover_limit")
    valid = (env.failure in (None, *accounted_risk) and _finite_real(reward) and
             _finite_real(env.equity) and env.equity > 0 and
             left < env.t <= min(left + env.horizon, env.stop - 1))
    if done is True:
        valid = (valid and nxt is None and env.units == 0 and env.pending is None and
                 (env.failure in accounted_risk or env.t == env.stop - 1))
    else:
        valid = (valid and done is False and env.failure is None and
                 env.t == left + env.horizon and env.t < env.stop - 1 and
                 _real_series(nxt) and nxt.shape == (FEATURE_COUNT,) and np.isfinite(nxt).all())
    if not valid:
        raise ValueError(f"incomplete training transition: {env.failure or 'invalid successor state'}")


def collect(prices: dict[str, np.ndarray], funding: dict[str, np.ndarray], scale: Scale,
            horizon: int, seed: int, count: int, policy=None, execution: Execution = Execution()) -> dict:
    """Training-only prefixes physically bound every replay and episode."""
    if (not all(_integer(v) for v in (horizon, seed, count)) or
        horizon not in (1, 3, 6) or seed < 0 or count <= 0):
        raise ValueError("invalid collection horizon, seed or transition budget")
    if (not isinstance(prices, dict) or not isinstance(funding, dict) or not prices or
        set(prices) != set(funding) or any(not isinstance(s, str) or not s for s in prices)):
        raise ValueError("invalid collection symbol coverage")
    for symbol, p in prices.items():
        f = funding[symbol]
        if not _real_series(p) or not _real_series(f) or len(p) != len(f) or len(p) <= 120:
            raise ValueError("invalid collection series or insufficient episode history")
    horizon, seed, count = int(horizon), int(seed), int(count)
    rng = np.random.default_rng(seed)
    symbols = sorted(prices)
    rows, episodes, env = [], [], None
    while len(rows) < count:
        if env is None or env.done:
            if env is not None:
                episodes.append({"return": env.equity - 1, "failure": env.failure})
            symbol = symbols[int(rng.integers(len(symbols)))]
            p = prices[symbol]
            start = int(rng.integers(24, len(p) - 96))
            env = Replay(p, funding[symbol], start, start + 97, horizon, scale, execution, enabled=True)
        s = env.observation()
        if s is None:
            raise ValueError("invalid training observation")
        probs = np.full(3, 1 / 3) if policy is None else policy(s)
        if not _real_series(probs) or probs.shape != (3,) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
            raise ValueError("invalid behavior probabilities")
        a = int(rng.choice(3, p=probs))
        left = env.t
        nxt, reward, done = env.step(float(ACTIONS[a]))
        _admit_training_transition(env, left, nxt, reward, done)
        rows.append((s, a, reward, np.zeros(FEATURE_COUNT) if nxt is None else nxt, done, probs[a]))
    return {"s": np.array([r[0] for r in rows]), "a": np.array([r[1] for r in rows]),
            "r": np.array([r[2] for r in rows]), "next": np.array([r[3] for r in rows]),
            "done": np.array([r[4] for r in rows]), "prob": np.array([r[5] for r in rows]),
            "episodes": episodes}
