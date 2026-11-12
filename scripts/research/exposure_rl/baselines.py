"""Non-RL policies on the same observations, and the simulated behaviour mixture.

Feature indices follow ``market.MARKET_FEATURES`` then ``env.STATE_FEATURES``.
"""

from __future__ import annotations

import numpy as np

from .env import BASE_COST, OBS_DIM
from .market import MARKET_FEATURES

F = {name: index for index, name in enumerate(MARKET_FEATURES)}
EXPOSURE = len(MARKET_FEATURES)  # state feature: exposure / CAP
BEHAVIOR_STD = 0.3
LOG_2PI = float(np.log(2.0 * np.pi))


class Constant:
    def __init__(self, value: float, name: str):
        self.value, self.name = float(value), name

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(len(obs), self.value)


class RegimeRule:
    """Fixed, unfitted regime-switch rule built from the observation features.

    Trend when ADX14 > 25 and the EMA20/EMA50 gap exceeds half an ATR; otherwise
    RSI14 reversion (<=35 long, >=65 short); otherwise a 20-bar Donchian
    breakout; otherwise flat. It mirrors the indicator families of the
    production ``ta_regime_switch`` rule but is not that rule.
    """

    name = "regime_rule"

    def act(self, obs: np.ndarray) -> np.ndarray:
        adx = (obs[:, F["adx14"]] + 0.5) / 2.0
        gap = obs[:, F["ema20_ema50_gap_atr"]]
        rsi = obs[:, F["rsi14"]]
        channel = obs[:, F["donchian20_position"]]
        trend = (adx > 0.25) & (np.abs(gap) > 0.5)
        reversion = np.where(rsi <= -0.3, 1.0, np.where(rsi >= 0.3, -1.0, 0.0))
        breakout = np.where(channel >= 0.9, 1.0, np.where(channel <= -0.9, -1.0, 0.0))
        return np.where(trend, np.sign(gap), np.where(reversion != 0.0, reversion, breakout))


class RidgeForecast:
    """Ridge forecast of the next-bar return mapped to a cost-aware target.

    Fitted only on the training region. The target is ``forecast / (2*cost)``
    clipped to [-1, 1]; forecasts smaller than one side's cost hold the current
    exposure. No coefficient is tuned on evaluation data.
    """

    name = "ridge_forecast"

    def __init__(self, features: np.ndarray, next_returns: np.ndarray, penalty: float = 10.0):
        x = np.concatenate([features, np.ones((len(features), 1))], axis=1)
        gram = x.T @ x + penalty * np.eye(x.shape[1])
        gram[-1, -1] -= penalty
        self.weights = np.linalg.solve(gram, x.T @ next_returns)

    def forecast(self, obs: np.ndarray) -> np.ndarray:
        market = obs[:, : len(MARKET_FEATURES)]
        return market @ self.weights[:-1] + self.weights[-1]

    def act(self, obs: np.ndarray) -> np.ndarray:
        forecast = self.forecast(obs)
        target = np.clip(forecast / (2.0 * BASE_COST), -1.0, 1.0)
        hold = np.clip(obs[:, EXPOSURE], -1.0, 1.0)
        return np.where(np.abs(forecast) < BASE_COST, hold, target)


def ridge_training_set(panel, ranges):
    """(features[t], close[t+1]/close[t]-1) for decisions whose outcome is inside the range."""
    xs, ys = [], []
    for symbol, (first, stop) in ranges.items():
        series = panel[symbol]
        first = max(first, 60)
        if stop - first < 2:
            continue
        xs.append(series.features[first : stop - 1])
        ys.append(series.close[first + 1 : stop] / series.close[first : stop - 1] - 1.0)
    return np.concatenate(xs), np.concatenate(ys)


class BehaviorMixture:
    """Known-propensity simulated behaviour policy for the offline dataset.

    Each episode draws one component: 0 = regime rule + Gaussian noise,
    1 = ridge forecast + Gaussian noise, 2 = mean-reverting random walk,
    3 = uniform. The logged action is the pre-clip sample; ``logp`` is its
    exact conditional density given the episode's component.
    """

    WEIGHTS = np.array([0.35, 0.35, 0.15, 0.15])

    def __init__(self, rule, ridge, lanes: int, seed: int):
        self.rule, self.ridge = rule, ridge
        self.rng = np.random.default_rng(seed)
        self.component = self.rng.choice(4, size=lanes, p=self.WEIGHTS)
        self.walk = np.zeros(lanes)

    def start_episodes(self, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if count:
            self.component[mask] = self.rng.choice(4, size=count, p=self.WEIGHTS)
            self.walk[mask] = 0.0

    def sample(self, obs: np.ndarray):
        lanes = len(obs)
        noise = self.rng.standard_normal(lanes)
        mean = np.where(self.component == 0, self.rule.act(obs), self.ridge.act(obs))
        gaussian = mean + BEHAVIOR_STD * noise
        self.walk = 0.9 * self.walk + 0.4 * self.rng.standard_normal(lanes)
        uniform = self.rng.uniform(-1.0, 1.0, lanes)
        sample = np.where(self.component <= 1, gaussian, np.where(self.component == 2, self.walk, uniform))
        logp = np.where(
            self.component <= 1,
            -0.5 * noise * noise - np.log(BEHAVIOR_STD) - 0.5 * LOG_2PI,
            np.where(self.component == 3, np.log(0.5), np.nan),  # walk density depends on hidden state
        )
        return sample, logp, mean, self.component.copy()


def collect_dataset(env, behavior: BehaviorMixture, steps: int) -> dict:
    """Simulator-generated offline dataset. Labelled simulated; not exchange logs."""
    obs = env.reset()
    behavior.start_episodes(np.ones(env.lanes, dtype=bool))
    keys = ("obs", "act", "rew", "next_obs", "term", "trunc", "logp", "mean", "component", "episode", "step", "symbol", "bar")
    rows = {key: [] for key in keys}
    episode = np.arange(env.lanes)
    step_index = np.zeros(env.lanes, dtype=np.int64)
    next_episode = env.lanes
    collected = 0
    while collected < steps:
        sample, logp, mean, component = behavior.sample(obs)
        start_bar = env.t.copy()
        symbol = env.sym.copy()
        next_obs, reward, terminated, truncated, final_obs, _ = env.step(np.clip(sample, -1.0, 1.0))
        for key, value in zip(keys, (obs, sample, reward, final_obs, terminated, truncated, logp, mean, component, episode.copy(), step_index.copy(), symbol, start_bar)):
            rows[key].append(value)
        done = terminated | truncated
        step_index = np.where(done, 0, step_index + 1)
        fresh = int(done.sum())
        episode = np.where(done, next_episode + np.cumsum(done) - 1, episode)
        next_episode += fresh
        behavior.start_episodes(done)
        obs = next_obs
        collected += env.lanes
    data = {key: np.concatenate(value) for key, value in rows.items()}
    data["act"] = np.clip(data["act"], -1.0, 1.0)  # learners see the proposal the environment received
    data["raw"] = np.concatenate(rows["act"])
    assert data["obs"].shape[1] == OBS_DIM
    return data
