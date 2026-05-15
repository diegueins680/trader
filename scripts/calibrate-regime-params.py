#!/usr/bin/env python3
"""
scripts/calibrate-regime-params.py

Grid-search calibration for Candidate 2 regime-confidence hyperparameters.
Sweeps w_adx, t_trend, t_range over conservative ranges and ranks results
by Sharpe ratio using walk-forward validation (7-fold).

Usage:
    python3 scripts/calibrate-regime-params.py \
        --data data/BTCUSDT-4h-1000.csv \
        --output artifacts/research/regime-calibration-btcusdt-4h.csv

Requires: trader-hs binary supporting --regime-adx-weight, --regime-trend-threshold,
          --regime-range-threshold (commit f400cff8 or later).
"""

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

DEFAULT_BINARY = Path(__file__).parent.parent / "haskell" / "dist-newstyle" / "build" / "x86_64-osx" / "ghc-9.4.8" / "trader-0.1.0.0" / "x" / "trader-hs" / "build" / "trader-hs" / "trader-hs"

# Grid bounds (conservative to avoid overfit)
# t_range dropped per t-range-redundancy-memo.md (commit 24885260)
W_ADX_VALUES = [round(x, 2) for x in [0.20, 0.30, 0.40, 0.50, 0.60]]
T_TREND_VALUES = [round(x, 2) for x in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]


def map_to_main_flags(w_adx: float, t_trend: float) -> dict:
    """Map Candidate 2 conceptual params to main-branch regime-parameter-bank multipliers.

    Mapping logic:
    - t_trend: lower = easier trend detection → higher trend-open-mult, lower mr-open-mult
    - w_adx: higher = stronger ADX regime conviction → higher size multipliers
    """
    trend_open_mult = max(0.5, round(1.0 + (0.75 - t_trend) * 4.0, 2))
    mr_open_mult = max(0.5, round(1.0 + t_trend * 2.0, 2))
    trend_size_mult = max(0.5, round(1.0 + w_adx * 2.5, 2))
    mr_size_mult = max(0.5, round(1.0 + (0.60 - w_adx) * 1.5, 2))
    return {
        "regime_trend_open_mult": trend_open_mult,
        "regime_mr_open_mult": mr_open_mult,
        "regime_trend_size_mult": trend_size_mult,
        "regime_mr_size_mult": mr_size_mult,
        "regime_high_vol_open_mult": 1.0,
        "regime_high_vol_size_mult": 1.0,
    }


def run_backtest(binary: Path, data: Path, w_adx: float, t_trend: float, mode: str = "candidate2") -> dict:
    """Run a single backtest and return parsed metrics."""
    cmd = [
        str(binary),
        "--data", str(data),
        "--price-column", "close",
        "--bars", "1000",
        "--method", "ta_trend",
        "--vol-conf-gate", "vol_conf_v1_default",
        "--walk-forward-folds", "7",
        "--json",
    ]
    if mode == "candidate2":
        cmd += [
            "--regime-adx-weight", str(w_adx),
            "--regime-trend-threshold", str(t_trend),
            "--regime-range-threshold", "0.45",
        ]
    elif mode == "main":
        mults = map_to_main_flags(w_adx, t_trend)
        cmd += [
            "--regime-parameter-bank",
            "--regime-trend-open-mult", str(mults["regime_trend_open_mult"]),
            "--regime-mr-open-mult", str(mults["regime_mr_open_mult"]),
            "--regime-trend-size-mult", str(mults["regime_trend_size_mult"]),
            "--regime-mr-size-mult", str(mults["regime_mr_size_mult"]),
            "--regime-high-vol-open-mult", str(mults["regime_high_vol_open_mult"]),
            "--regime-high-vol-size-mult", str(mults["regime_high_vol_size_mult"]),
        ]
    else:
        return {"error": f"unknown mode: {mode}", "sharpe": float("-inf")}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "sharpe": float("-inf")}
    except Exception as e:
        return {"error": str(e), "sharpe": float("-inf")}

    if result.returncode != 0:
        # Some errors go to stderr; some to stdout
        err_text = result.stderr.strip() or result.stdout.strip()
        return {"error": f"exit {result.returncode}: {err_text[:200]}", "sharpe": float("-inf")}

    # Parse JSON from last non-empty line (the binary may print progress then JSON)
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"error": "no output", "sharpe": float("-inf")}

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"error": f"json decode: {e}", "sharpe": float("-inf")}

    # The JSON structure wraps metrics under backtest.metrics in current binary.
    # Some fields (sharpe) are duplicated at backtest level; most are not.
    # Prefer backtest.metrics when available, fall back to top-level keys.
    metrics = payload
    if isinstance(payload, dict):
        if "backtest" in payload and isinstance(payload["backtest"], dict):
            if "metrics" in payload["backtest"] and isinstance(payload["backtest"]["metrics"], dict):
                metrics = payload["backtest"]["metrics"]
            else:
                metrics = payload["backtest"]
        elif "metrics" in payload and isinstance(payload["metrics"], dict):
            metrics = payload["metrics"]
        elif "result" in payload and isinstance(payload["result"], dict):
            metrics = payload["result"]

    return {
        "sharpe": float(metrics.get("sharpe", -9999) or -9999),
        "max_drawdown": float(metrics.get("maxDrawdown", -9999) or -9999),
        "win_rate": float(metrics.get("winRate", -9999) or -9999),
        "closed_trades": int(metrics.get("closedTrades", -1) or -1),
        "avg_trade": float(metrics.get("avgTrade", -9999) or -9999),
        "profit_factor": float(metrics.get("profitFactor", -9999) or -9999),
        "raw": metrics,
    }


def run_baseline(binary: Path, data: Path) -> dict:
    """Run baseline with default regime parameters (hardcoded defaults)."""
    cmd = [
        str(binary),
        "--data", str(data),
        "--price-column", "close",
        "--bars", "1000",
        "--method", "ta_trend",
        "--vol-conf-gate", "vol_conf_v1_default",
        "--walk-forward-folds", "7",
        "--json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return {"error": str(e), "sharpe": float("-inf")}

    if result.returncode != 0:
        err_text = result.stderr.strip() or result.stdout.strip()
        return {"error": f"exit {result.returncode}: {err_text[:200]}", "sharpe": float("-inf")}

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return {"error": "no output", "sharpe": float("-inf")}

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"error": f"json decode: {e}", "sharpe": float("-inf")}

    # The JSON structure wraps metrics under backtest.metrics in current binary.
    # Some fields (sharpe) are duplicated at backtest level; most are not.
    # Prefer backtest.metrics when available, fall back to top-level keys.
    metrics = payload
    if isinstance(payload, dict):
        if "backtest" in payload and isinstance(payload["backtest"], dict):
            if "metrics" in payload["backtest"] and isinstance(payload["backtest"]["metrics"], dict):
                metrics = payload["backtest"]["metrics"]
            else:
                metrics = payload["backtest"]
        elif "metrics" in payload and isinstance(payload["metrics"], dict):
            metrics = payload["metrics"]
        elif "result" in payload and isinstance(payload["result"], dict):
            metrics = payload["result"]

    return {
        "sharpe": float(metrics.get("sharpe", -9999) or -9999),
        "max_drawdown": float(metrics.get("maxDrawdown", -9999) or -9999),
        "win_rate": float(metrics.get("winRate", -9999) or -9999),
        "closed_trades": int(metrics.get("closedTrades", -1) or -1),
        "avg_trade": float(metrics.get("avgTrade", -9999) or -9999),
        "profit_factor": float(metrics.get("profitFactor", -9999) or -9999),
        "raw": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Candidate 2 regime-confidence grid search")
    parser.add_argument("--data", required=True, help="Path to CSV price data")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="Path to trader-hs binary")
    parser.add_argument("--output", default="artifacts/research/regime-calibration-results.csv", help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print grid size and exit without running backtests")
    parser.add_argument("--quick", action="store_true", help="Run a reduced grid for quick validation")
    parser.add_argument("--mode", choices=["candidate2", "main"], default="candidate2",
                        help="Backtest mode: candidate2 (uses --regime-adx-weight) or main (uses --regime-*-mult)")
    args = parser.parse_args()

    data_path = Path(args.data)
    binary_path = Path(args.binary)
    out_path = Path(args.output)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    if not binary_path.exists():
        print(f"ERROR: binary not found: {binary_path}", file=sys.stderr)
        sys.exit(1)

    # Grid definition
    if args.quick:
        w_adx_vals = [0.30, 0.40, 0.50]
        t_trend_vals = [0.50, 0.60]
    else:
        w_adx_vals = W_ADX_VALUES
        t_trend_vals = T_TREND_VALUES

    grid = list(product(w_adx_vals, t_trend_vals))
    print(f"Grid size: {len(grid)} combinations")
    print(f"Binary: {binary_path}")
    print(f"Data: {data_path}")

    if args.dry_run:
        if args.mode == "main":
            sample = map_to_main_flags(0.20, 0.70)
            print(f"Main-mode sample mapping (w_adx=0.20, t_trend=0.70): {sample}")
        print("Dry run — exiting without executing backtests.")
        sys.exit(0)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run baseline first
    print("\n[1/2] Running baseline (default regime params)...")
    baseline = run_baseline(binary_path, data_path)
    if "error" in baseline:
        print(f"BASELINE ERROR: {baseline['error']}", file=sys.stderr)
        # Continue anyway; script should still produce grid results
    else:
        print(f"Baseline Sharpe: {baseline['sharpe']:.4f}, maxDD: {baseline['max_drawdown']:.4f}, winRate: {baseline['win_rate']:.2%}, trades: {baseline['closed_trades']}")

    # Run grid
    print(f"\n[2/2] Running grid search ({len(grid)} combinations)...")
    results = []
    for i, (w_adx, t_trend) in enumerate(grid, start=1):
        print(f"  [{i}/{len(grid)}] w_adx={w_adx} t_trend={t_trend} ... ", end="", flush=True)
        if args.mode == "main":
            mults = map_to_main_flags(w_adx, t_trend)
            print(f"(mults: trend_open={mults['regime_trend_open_mult']}, mr_open={mults['regime_mr_open_mult']}) ", end="", flush=True)
        res = run_backtest(binary_path, data_path, w_adx, t_trend, mode=args.mode)
        if "error" in res:
            print(f"ERROR: {res['error']}")
        else:
            print(f"Sharpe={res['sharpe']:.4f} maxDD={res['max_drawdown']:.4f} trades={res['closed_trades']}")
        results.append({
            "w_adx": w_adx,
            "t_trend": t_trend,
            "sharpe": res.get("sharpe", -9999),
            "max_drawdown": res.get("max_drawdown", -9999),
            "win_rate": res.get("win_rate", -9999),
            "closed_trades": res.get("closed_trades", -1),
            "avg_trade": res.get("avg_trade", -9999),
            "profit_factor": res.get("profit_factor", -9999),
            "error": res.get("error", ""),
        })

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Write CSV
    fieldnames = [
        "rank", "w_adx", "t_trend",
        "sharpe", "max_drawdown", "win_rate",
        "closed_trades", "avg_trade", "profit_factor", "error",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(results, start=1):
            row = dict(r)
            row["rank"] = rank
            writer.writerow(row)

    print(f"\nResults written to: {out_path}")

    # Top 10 summary
    print("\n=== TOP 10 PARAMETER SETS ===")
    print(f"{'Rank':>4} {'w_adx':>6} {'t_trend':>8} {'Sharpe':>8} {'maxDD':>8} {'winRate':>8} {'trades':>7}")
    for rank, r in enumerate(results[:10], start=1):
        print(f"{rank:>4} {r['w_adx']:>6.2f} {r['t_trend']:>8.2f} {r['sharpe']:>8.4f} {r['max_drawdown']:>8.4f} {r['win_rate']:>8.2%} {r['closed_trades']:>7}")

    # Baseline reminder
    if "error" not in baseline:
        print(f"\nBaseline Sharpe: {baseline['sharpe']:.4f}")
        top_sharpe = results[0]["sharpe"] if results else float("-inf")
        delta = top_sharpe - baseline["sharpe"]
        print(f"Top calibrated Sharpe: {top_sharpe:.4f} (delta: {delta:+.4f})")
        if delta >= 0.10:
            print("PASS: Top parameter set improves Sharpe by >= 0.10 vs baseline.")
        else:
            print("FAIL: No parameter set improves Sharpe by >= 0.10 vs baseline.")


if __name__ == "__main__":
    main()
