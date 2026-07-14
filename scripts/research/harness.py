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

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

PERIODS = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


# --------------------------------------------------------------------------- #
# Nested rolling-origin validation                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RollingOriginSplit:
    """Positional bounds for one expanding-window chronological split.

    All ``stop`` fields are exclusive. The embargo is always exactly the label
    horizon, so no training label can reach into the following evaluation
    window.
    """

    fold: int
    train_start: int
    train_stop: int
    embargo_start: int
    embargo_stop: int
    test_start: int
    test_stop: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_stop)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_stop)


@dataclass(frozen=True)
class NestedRollingResult:
    """Auditable outputs from :func:`nested_rolling_origin`.

    ``oos`` contains every outer-test row exactly once. ``outer_folds`` records
    the selected candidate and positional boundaries for each outer fold.
    ``inner_scores`` contains every candidate score considered for each outer
    fold.
    """

    oos: pd.DataFrame
    outer_folds: pd.DataFrame
    inner_scores: pd.DataFrame


def _require_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def rolling_origin_splits(
    n_obs: int,
    initial_train_size: int,
    test_size: int,
    label_horizon: int = 1,
    *,
    allow_partial: bool = True,
) -> list[RollingOriginSplit]:
    """Build non-overlapping expanding-window splits with a label embargo.

    The first test begins after ``initial_train_size`` eligible training rows
    plus an embargo of ``label_horizon`` rows. Later test windows are adjacent,
    while their training windows expand and continue to exclude the most recent
    ``label_horizon`` rows. With ``allow_partial=True`` the final short test
    window is retained, giving complete OOS coverage from the first test onward.
    """

    _require_int("n_obs", n_obs, 0)
    _require_int("initial_train_size", initial_train_size, 1)
    _require_int("test_size", test_size, 1)
    _require_int("label_horizon", label_horizon, 0)
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be a bool")

    splits = []
    test_start = initial_train_size + label_horizon
    fold = 0
    while test_start < n_obs:
        test_stop = min(test_start + test_size, n_obs)
        if not allow_partial and test_stop - test_start < test_size:
            break
        train_stop = test_start - label_horizon
        splits.append(
            RollingOriginSplit(
                fold=fold,
                train_start=0,
                train_stop=train_stop,
                embargo_start=train_stop,
                embargo_stop=test_start,
                test_start=test_start,
                test_stop=test_stop,
            )
        )
        fold += 1
        test_start = test_stop
    return splits


def _evaluation_frame(output: Any, expected_length: int) -> pd.DataFrame:
    if isinstance(output, pd.DataFrame):
        frame = output.copy().reset_index(drop=True)
    elif isinstance(output, pd.Series):
        name = output.name if output.name is not None else "prediction"
        frame = output.rename(name).reset_index(drop=True).to_frame()
    else:
        values = np.asarray(output)
        if values.ndim != 1:
            raise ValueError(
                "candidate evaluation must be a one-dimensional array or a pandas object"
            )
        frame = pd.DataFrame({"prediction": values})
    if len(frame) != expected_length:
        raise ValueError(
            "candidate evaluation must return one row per evaluation row "
            f"(expected {expected_length}, got {len(frame)})"
        )
    if frame.columns.has_duplicates:
        raise ValueError("candidate evaluation returned duplicate column names")
    return frame


def _decorate_evaluation(
    evaluation: pd.DataFrame,
    source: pd.DataFrame,
    positions: range,
    *,
    time_col: str | None,
    fold_col: str,
    fold: int,
    selected_candidate: str | None = None,
) -> pd.DataFrame:
    metadata_columns = {"row_position", fold_col}
    if time_col is not None:
        metadata_columns.add(time_col)
    if selected_candidate is not None:
        metadata_columns.add("selected_candidate")
    conflicts = metadata_columns.intersection(evaluation.columns)
    if conflicts:
        names = ", ".join(sorted(str(name) for name in conflicts))
        raise ValueError(f"candidate evaluation uses reserved columns: {names}")

    metadata = {
        "row_position": list(positions),
        fold_col: np.full(len(evaluation), fold, dtype=int),
    }
    if time_col is not None:
        metadata[time_col] = source[time_col].to_numpy(copy=True)
    if selected_candidate is not None:
        metadata["selected_candidate"] = np.full(
            len(evaluation), selected_candidate, dtype=object
        )
    return pd.concat([pd.DataFrame(metadata), evaluation], axis=1)


def nested_rolling_origin(
    df: pd.DataFrame,
    candidates: Mapping[str, Any],
    fit_candidate: Callable[[Any, pd.DataFrame], Any],
    evaluate_candidate: Callable[[Any, pd.DataFrame], Any],
    score_candidate: Callable[[pd.DataFrame], float],
    *,
    initial_train_size: int,
    outer_test_size: int,
    inner_initial_train_size: int,
    inner_test_size: int,
    label_horizon: int = 1,
    time_col: str | None = "openTime",
    maximize: bool = True,
) -> NestedRollingResult:
    """Select candidates on inner folds and evaluate frozen outer folds.

    For every outer fold, each candidate is freshly fit on each expanding inner
    training slice and evaluated on its embargoed validation slice. The
    chronological inner outputs are concatenated and passed to
    ``score_candidate``. The winning candidate is then fit once on the full,
    embargoed outer training slice and evaluated once on the untouched outer
    test slice.

    ``evaluate_candidate`` must return one result per supplied evaluation row as
    a Series, one-dimensional array, or DataFrame. This supports both fitted
    models and precomputed causal candidate-return columns. Scores must be finite;
    ties are resolved by candidate insertion order.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    callbacks = (fit_candidate, evaluate_candidate, score_candidate)
    if not all(callable(callback) for callback in callbacks):
        raise TypeError("fit_candidate, evaluate_candidate, and score_candidate must be callable")
    if not isinstance(maximize, bool):
        raise TypeError("maximize must be a bool")
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time column {time_col!r} is missing")
        if not df[time_col].is_monotonic_increasing:
            raise ValueError(f"time column {time_col!r} must be sorted ascending")
        if df[time_col].duplicated().any():
            raise ValueError(f"time column {time_col!r} must be unique")

    candidate_items = list(candidates.items())
    for candidate_name, _ in candidate_items:
        if not isinstance(candidate_name, str) or not candidate_name:
            raise ValueError("candidate names must be non-empty strings")

    data = df.reset_index(drop=True).copy()
    outer_splits = rolling_origin_splits(
        len(data), initial_train_size, outer_test_size, label_horizon
    )
    if not outer_splits:
        raise ValueError("not enough observations for an outer test fold")

    oos_frames = []
    outer_rows = []
    inner_score_rows = []
    for outer in outer_splits:
        inner_splits = rolling_origin_splits(
            outer.train_stop,
            inner_initial_train_size,
            inner_test_size,
            label_horizon,
        )
        if not inner_splits:
            raise ValueError(
                f"outer fold {outer.fold} has no inner validation fold; "
                "reduce inner_initial_train_size or inner_test_size"
            )

        scored_candidates = []
        for candidate_name, candidate in candidate_items:
            validation_frames = []
            for inner in inner_splits:
                train = data.iloc[inner.train_slice].copy()
                validation = data.iloc[inner.test_slice].copy()
                model = fit_candidate(candidate, train)
                evaluated = _evaluation_frame(
                    evaluate_candidate(model, validation.copy()), len(validation)
                )
                validation_frames.append(
                    _decorate_evaluation(
                        evaluated,
                        validation,
                        range(inner.test_start, inner.test_stop),
                        time_col=time_col,
                        fold_col="inner_fold",
                        fold=inner.fold,
                    )
                )
            candidate_validation = pd.concat(validation_frames, ignore_index=True)
            try:
                score = float(score_candidate(candidate_validation.copy()))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"score for candidate {candidate_name!r} must be a scalar"
                ) from error
            inner_score_rows.append(
                {
                    "outer_fold": outer.fold,
                    "candidate": candidate_name,
                    "score": score,
                    "inner_folds": len(inner_splits),
                    "validation_rows": len(candidate_validation),
                }
            )
            scored_candidates.append((candidate_name, candidate, score))

        finite_candidates = [
            entry for entry in scored_candidates if np.isfinite(entry[2])
        ]
        if not finite_candidates:
            raise ValueError(
                f"all candidate scores are non-finite in outer fold {outer.fold}"
            )
        choose = max if maximize else min
        selected_name, selected_candidate, selection_score = choose(
            finite_candidates, key=lambda entry: entry[2]
        )

        outer_train = data.iloc[outer.train_slice].copy()
        outer_test = data.iloc[outer.test_slice].copy()
        frozen_model = fit_candidate(selected_candidate, outer_train)
        outer_evaluation = _evaluation_frame(
            evaluate_candidate(frozen_model, outer_test.copy()), len(outer_test)
        )
        oos_frames.append(
            _decorate_evaluation(
                outer_evaluation,
                outer_test,
                range(outer.test_start, outer.test_stop),
                time_col=time_col,
                fold_col="outer_fold",
                fold=outer.fold,
                selected_candidate=selected_name,
            )
        )
        outer_rows.append(
            {
                "outer_fold": outer.fold,
                "train_start": outer.train_start,
                "train_stop": outer.train_stop,
                "embargo_start": outer.embargo_start,
                "embargo_stop": outer.embargo_stop,
                "test_start": outer.test_start,
                "test_stop": outer.test_stop,
                "selected_candidate": selected_name,
                "selection_score": selection_score,
                "inner_folds": len(inner_splits),
            }
        )

    return NestedRollingResult(
        oos=pd.concat(oos_frames, ignore_index=True),
        outer_folds=pd.DataFrame(outer_rows),
        inner_scores=pd.DataFrame(inner_score_rows),
    )


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


def multiple_testing_sharpe_proxy(sr_ann, n_obs, ppy, n_trials=1):
    """Cheap Sidak-style Sharpe probability proxy.

    This is not the formal Deflated Sharpe Ratio: it does not use cross-trial
    Sharpe dispersion or return skew/kurtosis. Use
    ``diagnostics.deflated_sharpe_ratio`` with a complete aligned trial-return
    matrix for that statistic.
    """
    T = max(2, n_obs)
    sr_p = sr_ann / np.sqrt(ppy)  # per-period sharpe
    se = np.sqrt((1 + 0.5 * sr_p**2) / (T - 1))
    z = sr_p / (se + 1e-12)
    # one-sided normal CDF
    from math import erf, sqrt

    p_single = 0.5 * (1 + erf(z / sqrt(2)))
    # haircut for having tried n_trials independent strategies
    return float(p_single ** max(1, n_trials))


def deflated_sharpe_prob(sr_ann, n_obs, ppy, n_trials=1):
    """Deprecated compatibility name for ``multiple_testing_sharpe_proxy``.

    Despite the historical name, the returned value is not a formal DSR.
    """

    return multiple_testing_sharpe_proxy(sr_ann, n_obs, ppy, n_trials)


def summarize(net, ppy, label, n_trials=1):
    net = np.asarray(net, float)
    n = len(net)
    sr = sharpe(net, ppy)
    lo, hi = block_bootstrap_sharpe_ci(net, ppy)
    probability_proxy = multiple_testing_sharpe_proxy(sr, n, ppy, n_trials)
    tot = float(np.prod(1 + net) - 1)
    pos = net[net != 0]
    hit = float((pos > 0).mean()) if len(pos) else 0.0
    flags = []
    if n < 1500:
        flags.append(f"SMALL SAMPLE (n={n}; sharpe CI is unreliable below ~1500)")
    if not np.isnan(lo) and lo < 0 < hi:
        flags.append("sharpe CI straddles 0 (edge not distinguishable from noise)")
    if probability_proxy < 0.95:
        flags.append(
            f"proxy P(SR>0)={probability_proxy:.2f} (<0.95 after {n_trials} trials)"
        )
    print(
        f"  {label:18} n={n:5d}  net={100 * tot:+7.2f}%  sharpe={sr:+6.2f} "
        f"[{lo:+.2f},{hi:+.2f}]  hit={100 * hit:4.1f}%  "
        f"proxy_P(SR>0)={probability_proxy:.2f}"
    )
    for f in flags:
        print(f"        ⚠ {f}")
    return dict(
        n=n,
        net=tot,
        sharpe=sr,
        ci=(lo, hi),
        hit=hit,
        multiple_testing_proxy=probability_proxy,
        flags=flags,
    )


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
