"""Descriptive economic and OPE diagnostics; no promotion decision from scores."""
from __future__ import annotations

import math
import numpy as np
from sequential_env import ACTIONS, FEATURE_COUNT, Execution, Replay, _finite_real, _real_series, collect, market_features


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def sharpe(r: np.ndarray) -> float | None:
    return finite(np.mean(r) / np.std(r, ddof=1) * np.sqrt(1095)) if len(r) > 1 and np.std(r) > 1e-14 else None


def economic(env: Replay) -> dict:
    rows = env.rows
    if not rows:
        return {"status": "failed", "reason": env.failure, "observations": 0}
    r = np.array([x["net"] for x in rows])
    equity = np.r_[1.0, [x["equity"] for x in rows]]
    dd = 1 - equity / np.maximum.accumulate(equity)
    duration, max_duration = 0, 0
    for x in dd:
        duration = duration + 1 if x > 1e-12 else 0
        max_duration = max(duration, max_duration)
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    ann = finite(equity[-1]**(1095 / len(r)) - 1) if equity[-1] > 0 else None
    action_counts = [env.actions.count(float(a)) for a in ACTIONS]
    probs = np.array(action_counts) / max(1, len(env.actions))
    wins, losses = r[r > 0], r[r < 0]
    gross, funding = sum(x["gross"] for x in rows), sum(x["funding"] for x in rows)
    costs = {k: sum(x[k] for x in rows) for k in ("fee", "spread", "slippage", "impact")}
    if not np.isclose(equity[-1], 1 + gross + funding - sum(costs.values()), atol=1e-10, rtol=1e-10):
        raise ValueError("equity ledger does not reconcile")
    return {"status": "failed" if env.failure else "complete", "reason": env.failure,
            "observations": len(r), "firstOutcomeIndex": rows[0]["t"], "lastOutcomeIndex": rows[-1]["t"],
            "grossPnlOverInitialEquity": gross, "netReturn": equity[-1] - 1,
            "annualizedReturn": ann, "sharpe": sharpe(r),
            "sortino": finite(r.mean() / downside * np.sqrt(1095)) if downside > 0 else None,
            "calmar": finite(ann / max(dd)) if ann is not None and max(dd) > 0 else None,
            "maxDrawdown": float(max(dd)), "drawdownDurationBars": max_duration,
            "expectedShortfall95": float(-np.mean(np.sort(r)[:max(1, math.ceil(len(r) * 0.05))])),
            "turnoverOverInitialEquity": sum(x["turnover"] for x in rows), "fillsIncludingLiquidation": env.fills,
            "meanAbsoluteExposure": float(np.mean([abs(x["exposure"]) for x in rows])),
            "barWinRate": float(np.mean(r > 0)), "averageWinningBar": finite(wins.mean()) if len(wins) else None,
            "averageLosingBar": finite(losses.mean()) if len(losses) else None,
            "barProfitFactor": finite(wins.sum() / -losses.sum()) if len(losses) else None,
            "fundingPnlOverInitialEquity": funding, "costsOverInitialEquity": costs,
            "actionCounts": action_counts, "actionEntropy": float(-sum(p * np.log(p) for p in probs if p > 0)),
            "shieldRejectionRate": env.rejections / max(1, env.proposals),
            "fillModificationRate": env.modifications / max(1, env.proposals),
            "constraintFailure": env.failure in ("drawdown_limit", "capital_floor", "endpoint_exposure", "turnover_limit", "equity_exhausted")}


class Baselines:
    names = ("cash", "constant_long", "constant_short", "historical_mean", "last_return",
             "momentum", "reversal", "ridge_optimizer", "logistic", "contextual_bandit",
             "behavior_uniform", "behavior_clone")

    def __init__(self, prices, funding, scale, horizon: int):
        self.scale, self.horizon = scale, horizon
        x, y = [], []
        for p in prices.values():
            for t in range(24, len(p) - horizon - 1):
                x.append(np.r_[scale.transform(market_features(p, t)), 1.0])
                y.append(p[t + 1 + horizon] / p[t + 1] - 1)
        x, y = np.array(x), np.array(y)
        self.mean = float(y.mean())
        self.ridge = np.linalg.solve(x.T @ x + np.eye(7), x.T @ y)
        self.logistic = np.zeros(7)
        for _ in range(200):
            probs = 1 / (1 + np.exp(-np.clip(x @ self.logistic, -30, 30)))
            self.logistic -= 0.05 * (x.T @ (probs - (y > 0)) / len(y) + 0.001 * self.logistic)
        logs = collect(prices, funding, scale, horizon, 20260917, 4096)
        state = np.c_[logs["s"], np.ones(len(logs["s"]))]
        self.bandit = np.empty((13, 3))
        for a in range(3):
            idx = logs["a"] == a
            self.bandit[:, a] = np.linalg.solve(state[idx].T @ state[idx] + np.eye(13), state[idx].T @ logs["r"][idx])
        counts = np.bincount(logs["a"], minlength=3)
        self.clone_action = float(ACTIONS[int(np.argmax(counts))])
        self.behavior_counts = counts.tolist()

    @staticmethod
    def _valid_observation(obs: np.ndarray | None) -> bool:
        return (_real_series(obs) and obs.shape == (FEATURE_COUNT,) and
                bool(np.isfinite(obs).all()))

    def _valid_parameters(self, name: str) -> bool:
        fields = {"ridge_optimizer": ("ridge", (7,)), "logistic": ("logistic", (7,)),
                  "contextual_bandit": ("bandit", (13, 3))}
        if name in fields:
            field, shape = fields[name]
            value = getattr(self, field, None)
            return (isinstance(value, np.ndarray) and not np.ma.isMaskedArray(value) and
                    value.shape == shape and value.dtype.kind in "iuf" and bool(np.isfinite(value).all()))
        if name == "historical_mean":
            return _finite_real(getattr(self, "mean", None))
        if name == "behavior_clone":
            value = getattr(self, "clone_action", None)
            return _finite_real(value) and value in (-0.25, 0.0, 0.25)
        return True

    def forecast(self, name: str, obs: np.ndarray | None) -> float | None:
        if (not isinstance(name, str) or name not in
            ("historical_mean", "last_return", "momentum", "reversal", "ridge_optimizer") or
            not self._valid_observation(obs) or not self._valid_parameters(name)):
            return None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if name == "historical_mean":
                    value = self.mean
                elif name == "ridge_optimizer":
                    value = np.r_[obs[:6], 1.0] @ self.ridge
                else:
                    raw = obs[:6] * self.scale.std + self.scale.mean
                    value = raw[0] if name == "last_return" else raw[1] * (-1 if name == "reversal" else 1)
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None
        return float(value) if _finite_real(value) else None

    def action(self, name: str, obs: np.ndarray | None, rng: np.random.Generator) -> float | None:
        if (not isinstance(name, str) or name not in self.names or
            not self._valid_observation(obs) or not self._valid_parameters(name)):
            return None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                target = self._action(name, obs, rng)
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            return None
        return float(target) if _finite_real(target) and target in (-0.25, 0.0, 0.25) else None

    def _action(self, name: str, obs: np.ndarray, rng: np.random.Generator) -> float | None:
        if name in ("cash", "constant_long", "constant_short"):
            return {"cash": 0.0, "constant_long": 0.25, "constant_short": -0.25}[name]
        if name == "behavior_clone":
            return self.clone_action
        if name == "behavior_uniform":
            return float(rng.choice(ACTIONS))
        if name == "contextual_bandit":
            values = np.r_[obs, 1.0] @ self.bandit
            return float(ACTIONS[np.argmax(values)]) if np.isfinite(values).all() else None
        if name == "logistic":
            logit = np.r_[obs[:6], 1.0] @ self.logistic
            if not _finite_real(logit):
                return None
            prob = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
            return 0.25 if prob > 0.55 else -0.25 if prob < 0.45 else 0.0
        mu = self.forecast(name, obs)
        if mu is None:
            return None
        scores = ACTIONS * mu - 0.001 * np.abs(ACTIONS - obs[6])
        if not np.isfinite(scores).all():
            return None
        # Cash first on equal economic utility; no accidental bullish default.
        best = np.flatnonzero(scores >= max(scores) - 1e-14)
        return 0.0 if 1 in best else float(ACTIONS[int(best[0])])


def replay_policy(p, funding, start, stop, horizon, scale, action, execution=Execution()):
    env = Replay(p, funding, start, stop, horizon, scale, execution, enabled=True)
    latencies, ood, proposals = [], 0, 0
    while not env.done:
        obs = env.observation()
        proposals += 1
        if not env.supported():
            # Unsupported state: request only a neutral target at a later valid
            # close. This reduces inventory and supplies no new direction.
            ood += 1
            target = 0.0
            ms = 0.0
        else:
            target, ms = action(obs)
        latencies.append(ms)
        env.step(target, elapsed_ms=ms)
    result = economic(env)
    result.update({"oodObservationRate": ood / max(1, proposals),
                   "latencyP50Ms": finite(np.median(latencies)), "latencyP99Ms": finite(np.quantile(latencies, 0.99)),
                   "forecastMetrics": "not_applicable_to_policy_values"})
    return env, result


def ope_estimates(rewards, actions, behavior_prob, target_prob, q, v, gamma: float) -> dict:
    """Finite-horizon sequential IS/PDIS/WIS/DR, no weight clipping.

    q[i,t] is Q_hat(s_t, logged a_t); v includes final zero bootstrap.
    Exact logged propensities are mandatory. ESS is trajectory-weight ESS.
    """
    inputs = (rewards, actions, behavior_prob, target_prob, q, v)
    if any(np.ma.isMaskedArray(x) for x in inputs):
        raise ValueError("masked OPE input")
    arrays = [np.asarray(x) for x in inputs]
    r, a, b, pi, q, v = arrays
    if (r.ndim != 2 or 0 in r.shape or any(x.shape != r.shape for x in (a, b, pi, q)) or
        v.shape != (r.shape[0], r.shape[1] + 1)):
        raise ValueError("OPE shape mismatch or empty episodes")
    if (any(x.dtype.kind not in "iuf" for x in arrays) or
        not all(np.isfinite(x).all() for x in arrays)):
        raise ValueError("non-numeric or non-finite OPE input")
    if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= len(ACTIONS)):
        raise ValueError("invalid logged OPE action")
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.integer, np.floating)) or
        not np.isfinite(gamma) or not 0 <= gamma <= 1):
        raise ValueError("invalid OPE discount")
    if np.any(b <= 0) or np.any(b > 1) or np.any(pi < 0) or np.any(pi > 1):
        raise ValueError("invalid OPE probabilities")
    if np.any(v[:, -1] != 0):
        raise ValueError("nonzero terminal OPE bootstrap")
    # Finite inputs can still overflow ratios, accumulated weights, moments or
    # bootstrap means. Reject the batch; never clip weights to manufacture ESS.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _ope_estimates(r.astype(float), b.astype(float), pi.astype(float),
                                  q.astype(float), v.astype(float), float(gamma))
    except FloatingPointError as exc:
        raise ValueError("non-finite OPE arithmetic") from exc


def _ope_estimates(r, b, pi, q, v, gamma):
    weights = np.cumprod(pi / b, axis=1)
    discount = gamma ** np.arange(r.shape[1])
    returns = r @ discount
    w = weights[:, -1]
    is_values = w * returns
    pdis_values = np.sum(weights * r * discount, axis=1)
    dr_values = v[:, 0] + np.sum(weights * (r + gamma * v[:, 1:] - q) * discount, axis=1)
    ess = float(w.sum()**2 / (w @ w)) if w @ w > 0 else 0.0
    result = {"ordinaryIS": float(is_values.mean()), "perDecisionIS": float(pdis_values.mean()),
              "weightedIS": float(w @ returns / w.sum()) if w.sum() else None,
              "doublyRobust": float(dr_values.mean()), "effectiveSampleSize": ess,
              "maxTrajectoryWeight": float(max(w)), "nonzeroTrajectories": int(np.count_nonzero(w)),
              "episodes": len(w), "horizonDecisions": r.shape[1], "weightClipping": "none",
              "reliable": False, "reason": "Conditional simulator OPE; no live behavior support. ESS and estimator agreement are additional necessary gates."}
    rng = np.random.default_rng(20260917)
    draws = rng.integers(len(w), size=(1000, len(w)))
    result["conditionalBootstrap95"] = {k: np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist()
                                         for k, values in (("IS", is_values), ("PDIS", pdis_values), ("DR", dr_values))}
    return result


def short_ope(prices, funding, scale, horizon, start, stop, net, seed, episodes=200):
    rng = np.random.default_rng(seed)
    symbols = sorted(prices)
    all_r, all_a, all_b, all_pi, all_q, all_v, direct = [], [], [], [], [], [], []
    failed = 0
    for _ in range(episodes):
        sym = symbols[int(rng.integers(len(symbols)))]
        t = int(rng.integers(start, stop - 6 * horizon))
        env = Replay(prices[sym], funding[sym], t, t + 6 * horizon + 1, horizon, scale, enabled=True)
        replay = Replay(prices[sym], funding[sym], t, t + 6 * horizon + 1, horizon, scale, enabled=True)
        rs, ac, pi, qs, vs = [], [], [], [], []
        for j in range(6):
            obs = env.observation()
            out = net.forward(obs)
            chosen = int(np.argmax(out)) if env.supported() else 1
            a = int(rng.integers(3))
            vs.append(float(out[chosen]))
            qs.append(float(out[a]))
            pi.append(float(a == chosen))
            ac.append(a)
            _, reward, done = env.step(float(ACTIONS[a]))
            rs.append(reward)
            if done and j < 5:
                break
        value = 0.0
        for j in range(6):
            if replay.done:
                break
            a = int(np.argmax(net.forward(replay.observation()))) if replay.supported() else 1
            _, reward, _ = replay.step(float(ACTIONS[a]))
            value += 0.99**(horizon * j) * reward
        if len(rs) != 6 or env.failure or replay.failure:
            failed += 1
            continue
        all_r.append(rs); all_a.append(ac); all_b.append([1 / 3] * 6)
        all_pi.append(pi); all_q.append(qs); all_v.append(vs + [0.0]); direct.append(value)
    if failed:
        return {"status": "invalid", "failedEpisodes": failed, "reason": "No failed episodes may be silently excluded from OPE."}
    result = ope_estimates(np.array(all_r), np.array(all_a), np.array(all_b), np.array(all_pi),
                           np.array(all_q), np.array(all_v), 0.99**horizon)
    result.update({"directSimulatorValue": float(np.mean(direct)),
                   "valueModel": "policy network outputs used only as a DR control variate; PPO logits are not calibrated Q estimates",
                   "uncertaintyScope": "bootstrap of randomized behavior conditional on sampled fixed historical episodes; not a market-generalization interval",
                   "liveStateActionSupport": "unavailable", "simulatedActionSupport": [1/3, 1/3, 1/3]})
    return result
