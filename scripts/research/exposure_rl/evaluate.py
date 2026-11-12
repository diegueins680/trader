"""Frozen-policy evaluation, economic metrics and dependence-aware statistics."""

from __future__ import annotations

import math

import numpy as np

from . import shield as S
from .env import ExposureEnv, Execution, usable_symbols
from .market import BAR_MS

BARS_PER_YEAR = 365.25 * 4.0
FLEET = ("ARBUSDT", "BCHUSDT", "DOGEUSDT", "ETHUSDT", "NEARUSDT", "SOLUSDT")


def rollout(policy, panel, ranges, execution: Execution = Execution(), seed: int = 0) -> dict:
    """Run one account per symbol from flat over its whole range; liquidate at the end.

    Returns per-symbol arrays aligned to bar open times. The policy is frozen:
    no weights, scalers or calibration are updated here.
    """
    env = ExposureEnv(panel, ranges, len(usable_symbols(panel, ranges)), seed, execution, episode_length=None)
    obs = env.reset(sequential=True)
    logs: dict[str, list] = {}
    while env.active.any():
        if hasattr(policy, "act_at"):  # schedule policies need the decision bar
            local = env.t - env.offsets[env.sym]
            action = policy.act_at([env.symbols[i] for i in env.sym], [panel[env.symbols[i]].open_time[j] for i, j in zip(env.sym, local)])
        else:
            action = policy.act(obs)
        obs, _, terminated, truncated, _, info = env.step(action, auto_reset=False)
        info["terminated"], info["truncated"] = terminated, truncated
        for key, value in info.items():
            logs.setdefault(key, []).append(np.asarray(value))
    table = {key: np.stack(value) for key, value in logs.items()}
    result = {}
    for lane, symbol in enumerate(env.symbols):
        live = table["active"][:, lane]
        row = {key: value[live, lane] for key, value in table.items() if key != "active"}
        net = row["net_return"].copy()
        closing = float(row["closing_cost_return"][-1]) if not row["terminated"][-1] else 0.0
        net[-1] = (1.0 + net[-1]) * (1.0 - closing) - 1.0  # terminal liquidation at cost
        bars = row["bar"] - env.offsets[lane]
        result[symbol] = {
            "open_time": panel[symbol].open_time[bars],
            "net": net,
            "gross": row["gross"] / row["equity_before"],
            "cost": row["cost"] / row["equity_before"] + np.where(np.arange(len(net)) == len(net) - 1, closing, 0.0),
            "funding": row["funding"] / row["equity_before"],
            "turnover": row["turnover"],
            "exposure": row["exposure_after"] / S.SCALE,
            "permitted": row["permitted"],
            "proposed": row["proposed"],
            "kind": row["kind"],
            "reasons": row["reasons"],
            "intent": row["intent"],
            "filled": row["filled"],
            "terminated": bool(row["terminated"][-1]),
        }
    return result


def portfolio_series(result: dict, symbols=FLEET):
    """First-order fleet aggregate: per-bar sum of symbol account returns.

    Each account sizes against the same notional equity with a 16.66% cap, so
    the sum is the return on that equity ignoring second-order compounding
    interactions between accounts. Symbols not yet listed contribute zero.
    """
    times = np.unique(np.concatenate([result[s]["open_time"] for s in symbols if s in result]))
    total = np.zeros(len(times))
    for symbol in symbols:
        if symbol in result:
            total[np.searchsorted(times, result[symbol]["open_time"])] += result[symbol]["net"]
    return times, total


def max_drawdown(returns: np.ndarray) -> tuple[float, int]:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = 1.0 - equity / peak
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value > 0 else 0
        longest = max(longest, current)
    return float(drawdown.max(initial=0.0)), longest


def metrics(returns: np.ndarray) -> dict:
    n = len(returns)
    if n < 2:
        return {"bars": n, "insufficient": True}
    mean, std = float(returns.mean()), float(returns.std(ddof=1))
    depth, duration = max_drawdown(returns)
    tail = np.sort(returns)[: max(1, int(math.floor(0.05 * n)))]
    return {
        "bars": n,
        "totalReturn": float(np.prod(1.0 + returns) - 1.0),
        "annualizedMean": mean * BARS_PER_YEAR,
        "annualizedVol": std * math.sqrt(BARS_PER_YEAR),
        "sharpe": None if std == 0 else mean / std * math.sqrt(BARS_PER_YEAR),
        "maxDrawdown": depth,
        "maxDrawdownBars": duration,
        "expectedShortfall95": float(tail.mean()),
        "worstBar": float(returns.min()),
    }


def activity(result: dict) -> dict:
    """Trading activity and shield intervention summary across symbols."""
    turnover = sum(float(r["turnover"].sum()) for r in result.values())
    bars = sum(len(r["net"]) for r in result.values())
    kinds = np.concatenate([r["kind"] for r in result.values()])
    reasons = np.concatenate([r["reasons"] for r in result.values()])
    exposure = np.concatenate([r["exposure"] for r in result.values()])
    fills = np.concatenate([r["filled"] for r in result.values()])
    holds = []
    for r in result.values():
        sign = np.sign(np.round(r["exposure"], 4))
        change = np.flatnonzero(np.diff(sign) != 0)
        if len(change) > 1:
            holds.extend(np.diff(change)[sign[change[:-1] + 1] != 0].tolist())
    return {
        "decisions": bars,
        "fills": int(fills.sum()),
        "annualTurnover": turnover / max(bars / len(result), 1) * BARS_PER_YEAR / len(result),
        "meanAbsExposure": float(np.abs(exposure).mean()),
        "maxAbsExposure": float(np.abs(exposure).max()),
        "longShare": float((exposure > 0).mean()),
        "shortShare": float((exposure < 0).mean()),
        "medianHoldingBars": None if not holds else float(np.median(holds)),
        "permittedRate": float((kinds == S.PERMITTED).mean()),
        "abstainRate": float((kinds == S.ABSTAIN).mean()),
        "rejectedRate": float((kinds == S.REJECTED).mean()),
        "recoveryRate": float((kinds == S.RECOVERY_REQUIRED).mean()),
        "projectedRate": float(((reasons & (S.R_CAP | S.R_HEADROOM | S.R_RISK_OFF)) != 0).mean()),
        "riskOffRate": float(((reasons & S.R_RISK_OFF) != 0).mean()),
        "grossReturn": sum(float(r["gross"].sum()) for r in result.values()),
        "costPaid": sum(float(r["cost"].sum()) for r in result.values()),
        "fundingPaid": sum(float(r["funding"].sum()) for r in result.values()),
    }


def stationary_bootstrap_mean(values: np.ndarray, rng, draws: int = 2000, mean_block: float = 28.0) -> np.ndarray:
    """Politis-Romano stationary bootstrap of the sample mean."""
    n = len(values)
    out = np.empty(draws)
    p = 1.0 / mean_block
    for draw in range(draws):
        restart = rng.random(n) < p
        restart[0] = True
        starts = rng.integers(0, n, size=n)
        index = np.empty(n, dtype=np.int64)
        position = 0
        for i in range(n):
            position = starts[i] if restart[i] else (position + 1) % n
            index[i] = position
        out[draw] = values[index].mean()
    return out


def paired_difference(candidate: np.ndarray, baseline: np.ndarray, seed: int = 0, draws: int = 2000) -> dict:
    """Annualised mean per-bar return difference with a 95% stationary-bootstrap interval."""
    diff = candidate - baseline
    if len(diff) < 60:
        return {"insufficient": True, "bars": len(diff)}
    boot = stationary_bootstrap_mean(diff, np.random.default_rng(seed), draws) * BARS_PER_YEAR
    return {
        "bars": len(diff),
        "annualizedDifference": float(diff.mean() * BARS_PER_YEAR),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "probabilityPositive": float((boot > 0).mean()),
    }


def span_days(times: np.ndarray) -> float:
    return float((times[-1] - times[0] + BAR_MS) / 86_400_000)
