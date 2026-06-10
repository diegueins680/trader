"""Honest large-sample evaluation harness for trading signals.

Design principle: an edge must survive (1) out-of-sample walk-forward, (2)
realistic costs, (3) a bootstrap confidence interval that respects sample size,
and (4) a multiple-testing haircut. The point of this module is to make it hard
to fool yourself — short, lucky windows produce wide CIs and get flagged, not
celebrated.

A "signal" is a function ``fit_predict(train_df, test_row) -> float`` predicting
the next-bar return (sign/magnitude). Helpers below provide OLS-linear and
momentum signals; plug in your own.

Dependencies: numpy, pandas (stdlib otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


# --------------------------------------------------------------------------- #
# Signals: (feature_cols) -> a fit_predict closure over an expanding window     #
# --------------------------------------------------------------------------- #
def ols_signal(feature_cols):
    """Expanding-window OLS of fwd_ret on standardized features. Returns a
    z-scored prediction (pred / in-sample fitted std) so the deadband is
    comparable across symbols."""

    def fit_predict(train: pd.DataFrame, x_row: pd.Series):
        X = train[feature_cols].to_numpy(float)
        y = train["fwd_ret"].to_numpy(float)
        ok = np.all(np.isfinite(np.column_stack([X, y])), axis=1)
        X, y = X[ok], y[ok]
        if len(y) < max(50, 5 * len(feature_cols)):
            return 0.0
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        Z = (X - mu) / sd
        A = np.hstack([np.ones((len(Z), 1)), Z])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        xz = (x_row[feature_cols].to_numpy(float) - mu) / sd
        if not np.all(np.isfinite(xz)):
            return 0.0
        pred = beta[0] + xz @ beta[1:]
        fitted_sd = np.std(A @ beta) or 1e-9
        return float(pred / fitted_sd)

    return fit_predict


def momentum_signal(col="mom_score"):
    """Z-score the precomputed momentum column against the train distribution so
    its scale is comparable to the OLS signal's (and the deadband applies)."""

    def fit_predict(train, x_row):
        m = train[col].to_numpy(float)
        m = m[np.isfinite(m)]
        if len(m) < 50:
            return 0.0
        mu, sd = m.mean(), (m.std() or 1e-9)
        x = x_row.get(col, np.nan)
        if not np.isfinite(x):
            return 0.0
        return float((x - mu) / sd)

    return fit_predict


# --------------------------------------------------------------------------- #
# Walk-forward, cost-aware                                                       #
# --------------------------------------------------------------------------- #
def walk_forward(df, fit_predict, min_train=2000, cost=0.0005, deadband=0.3):
    """Expanding-window OOS backtest. Position = sign(score) when |score| beats
    the deadband, else flat. Net return charges ``cost`` per unit position
    change. Returns a DataFrame of OOS rows with score/position/net_ret."""
    df = df.reset_index(drop=True)
    n = len(df)
    rows = []
    pos_prev = 0.0
    for t in range(min_train, n):
        if not np.isfinite(df.at[t, "fwd_ret"]):
            continue
        score = fit_predict(df.iloc[:t], df.iloc[t])
        pos = np.sign(score) if abs(score) > deadband else 0.0
        net = pos * df.at[t, "fwd_ret"] - cost * abs(pos - pos_prev)
        rows.append((df.at[t, "openTime"], score, pos, df.at[t, "fwd_ret"], net))
        pos_prev = pos
    return pd.DataFrame(rows, columns=["openTime", "score", "pos", "fwd_ret", "net"])


# --------------------------------------------------------------------------- #
# Metrics with honesty built in                                                 #
# --------------------------------------------------------------------------- #
def sharpe(net, ppy):
    net = np.asarray(net, float)
    return float(np.sqrt(ppy) * net.mean() / (net.std() + 1e-12)) if len(net) else 0.0


def block_bootstrap_sharpe_ci(net, ppy, block=24, n_boot=2000, alpha=0.05, seed=0):
    """Moving-block bootstrap CI for the annualized sharpe (respects
    autocorrelation). Wide CI on short samples is the point."""
    net = np.asarray(net, float)
    m = len(net)
    if m < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(m / block))
    starts_pool = np.arange(0, m - block + 1)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_pool, size=nblocks)
        sample = np.concatenate([net[s : s + block] for s in starts])[:m]
        out[b] = sharpe(sample, ppy)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def deflated_sharpe_prob(sr_ann, n_obs, ppy, n_trials=1):
    """Probability the true sharpe > 0 given an observed annualized sharpe, the
    sample size, and the number of strategies tried (multiple-testing haircut).
    Uses the standard SR standard-error ~ sqrt((1+SR_p^2/2)/(T-1)) and a
    Šidák-style adjustment for n_trials. Returns a 0..1 probability."""
    T = max(2, n_obs)
    sr_p = sr_ann / np.sqrt(ppy)  # per-period sharpe
    se = np.sqrt((1 + 0.5 * sr_p**2) / (T - 1))
    z = sr_p / (se + 1e-12)
    # one-sided normal CDF
    from math import erf, sqrt

    p_single = 0.5 * (1 + erf(z / sqrt(2)))
    # haircut for having tried n_trials independent strategies
    return float(p_single ** max(1, n_trials))


def summarize(net, ppy, label, n_trials=1):
    net = np.asarray(net, float)
    n = len(net)
    sr = sharpe(net, ppy)
    lo, hi = block_bootstrap_sharpe_ci(net, ppy)
    dsr = deflated_sharpe_prob(sr, n, ppy, n_trials)
    tot = float(np.prod(1 + net) - 1)
    pos = net[net != 0]
    hit = float((pos > 0).mean()) if len(pos) else 0.0
    flags = []
    if n < 1500:
        flags.append(f"SMALL SAMPLE (n={n}; sharpe CI is unreliable below ~1500)")
    if not np.isnan(lo) and lo < 0 < hi:
        flags.append("sharpe CI straddles 0 (edge not distinguishable from noise)")
    if dsr < 0.95:
        flags.append(f"deflated P(SR>0)={dsr:.2f} (<0.95 after {n_trials} trials)")
    print(
        f"  {label:18} n={n:5d}  net={100 * tot:+7.2f}%  sharpe={sr:+6.2f} "
        f"[{lo:+.2f},{hi:+.2f}]  hit={100 * hit:4.1f}%  P(SR>0)={dsr:.2f}"
    )
    for f in flags:
        print(f"        ⚠ {f}")
    return dict(n=n, net=tot, sharpe=sr, ci=(lo, hi), hit=hit, dsr=dsr, flags=flags)


def regime_report(oos: pd.DataFrame, df: pd.DataFrame, ppy):
    """Split OOS net returns by the contemporaneous market regime (trend
    up/down by 24-bar return sign) and report sharpe in each — a real edge holds
    across regimes rather than just shorting a downtrend."""
    j = df.set_index("openTime")
    up, down = [], []
    for _, r in oos.iterrows():
        ot = r["openTime"]
        if ot not in j.index:
            continue
        i = j.index.get_loc(ot)
        if i < 24:
            continue
        trend = j["close"].iloc[i] / j["close"].iloc[i - 24] - 1
        (up if trend >= 0 else down).append(r["net"])
    print(f"  by regime: up-trend sharpe={sharpe(up, ppy):+.2f} (n={len(up)})  "
          f"down-trend sharpe={sharpe(down, ppy):+.2f} (n={len(down)})")


def cross_sectional(panel, score_col, cost=0.0005, top=2):
    """Market-neutral cross-section: each bar, long the top-`top` symbols by
    score and short the bottom-`top`, equal weight. Returns net per-bar series.
    `score_col` must already exist in each symbol frame (a precomputed signal)."""
    # align on common openTimes
    common = None
    for df in panel.values():
        s = set(df["openTime"])
        common = s if common is None else (common & s)
    times = sorted(common or [])
    rows = []
    prev_w = {}
    for ot in times:
        scores, fwds = {}, {}
        for sym, df in panel.items():
            r = df[df["openTime"] == ot]
            if r.empty:
                continue
            sc, fw = r["openTime"].index[0], None
            scores[sym] = float(r[score_col].iloc[0]) if np.isfinite(r[score_col].iloc[0]) else None
            fwds[sym] = float(r["fwd_ret"].iloc[0]) if np.isfinite(r["fwd_ret"].iloc[0]) else None
        valid = [s for s in scores if scores[s] is not None and fwds[s] is not None]
        if len(valid) < 2 * top:
            continue
        ranked = sorted(valid, key=lambda s: scores[s])
        w = {s: 0.0 for s in valid}
        for s in ranked[:top]:
            w[s] = -1.0 / top
        for s in ranked[-top:]:
            w[s] = +1.0 / top
        gross = sum(w[s] * fwds[s] for s in valid)
        turn = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) for s in set(w) | set(prev_w))
        rows.append((ot, gross - cost * turn))
        prev_w = w
    return pd.DataFrame(rows, columns=["openTime", "net"])
