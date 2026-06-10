"""Demo + smoke test for the research harness.

Loads (and refreshes) a small symbol panel, runs a funding+basis OLS signal and a
momentum signal through the cost-aware walk-forward, and prints rigorous reports
with bootstrap sharpe CIs, a regime split, a multiple-testing haircut, and a
cross-sectional book.

On the currently-available ~30-day window this will (correctly) FLAG small-sample
unreliability — that is the intended lesson: the harness refuses to bless an edge
the data can't support. Re-run after the cache has accumulated months of history
(via scheduled ``datafeed.update_cache``) for a verdict you can act on.

Usage:  python3 scripts/research/run_example.py [SYM ...]
"""

from __future__ import annotations

import sys

import numpy as np

import datafeed as feed
import harness as H


def add_features(df):
    df = df.copy()
    r = df["ret"].to_numpy(float)
    n = len(r)
    df["funding_d"] = np.concatenate([[np.nan], np.diff(df["funding"].to_numpy(float))])
    oi = df["oi"].to_numpy(float)
    df["doi"] = np.concatenate([[np.nan], np.diff(oi)]) / np.where(np.abs(oi) < 1e-9, np.nan, oi)
    df["taker_imb"] = df["taker"].to_numpy(float) - 1.0
    df["mom_score"] = np.array([np.sum(r[max(0, i - 5) : i + 1]) for i in range(n)])
    return df


def main():
    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    interval = "1h"
    ppy = H.PERIODS[interval]
    print(f"Loading panel {syms} @ {interval} (refreshing cache)…")
    panel = {s: add_features(d) for s, d in feed.load_panel(syms, interval).items()}
    if not panel:
        print("No data — check connectivity / cache.")
        return

    # number of (signal x symbol) configurations we are scanning, for the haircut
    n_trials = 2 * len(panel)
    min_train = 250  # small window today; raise toward 2000+ once cache has history

    print("\n=== per-symbol walk-forward (cost 5bps/turn) ===")
    for sym, df in panel.items():
        print(f"\n{sym}:")
        exo = H.walk_forward(df, H.ols_signal(["funding", "basis"]), min_train=min_train)
        H.summarize(exo["net"], ppy, "exo(funding+basis)", n_trials=n_trials)
        H.regime_report(exo, df, ppy)
        mom = H.walk_forward(df, H.momentum_signal(), min_train=min_train)
        H.summarize(mom["net"], ppy, "momentum-only", n_trials=n_trials)

    # cross-sectional book on the funding+basis score (precompute a simple score)
    print("\n=== cross-sectional (long top / short bottom by -funding+basis) ===")
    for sym, df in panel.items():
        f = df["funding"].to_numpy(float)
        b = df["basis"].to_numpy(float)
        # standardize within symbol; signal = +basis - funding (per the IC signs)
        def z(x):
            return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
        df["xs_score"] = z(b) - z(f)
    xs = H.cross_sectional(panel, "xs_score")
    if len(xs):
        H.summarize(xs["net"], ppy, "cross-sectional", n_trials=n_trials)
    else:
        print("  (not enough overlapping bars for a cross-section yet)")

    print(
        "\nNote: small-sample flags above are expected on the ~30-day window. "
        "Schedule datafeed.update_cache to accumulate history, then re-run."
    )


if __name__ == "__main__":
    main()
