#!/usr/bin/env python3
"""Run the locked 5-row vol/conf scorecard and emit a markdown report."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BINARY = Path(__file__).parent.parent / "haskell" / "dist-newstyle" / "build" / "x86_64-osx" / "ghc-9.4.8" / "trader-0.1.0.0" / "x" / "trader-hs" / "build" / "trader-hs" / "trader-hs"
DATA = Path(__file__).parent.parent / "data" / "BTCUSDT-4h-1000.csv"
PRESETS = [
    "disabled",
    "vol_conf_v1_default",
    "vol_conf_v1_high_vol_tighter",
    "vol_conf_v1_high_vol_looser",
    "vol_conf_v1_conf_stricter",
]


def run_backtest(preset: str, data_path: Path, bars: int) -> dict:
    cmd = [
        str(BINARY),
        "--data", str(data_path),
        "--price-column", "close",
        "--bars", str(bars),
        "--method", "ta_trend",
        "--vol-conf-gate", preset,
        "--walk-forward-folds", "7",
        "--json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "sharpe": float("-inf")}
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

    bt = payload.get("backtest", payload)
    if isinstance(bt, dict):
        if "metrics" in bt and isinstance(bt["metrics"], dict):
            metrics = bt["metrics"]
        else:
            metrics = bt
    else:
        metrics = payload

    closed = int(metrics.get("closedTrades", -1) or -1)
    pos_changes = int(metrics.get("positionChanges", -1) or -1)
    retention = (closed / pos_changes * 100) if pos_changes > 0 else float("nan")

    def _f(key: str) -> float:
        v = metrics.get(key)
        return float(v) if v is not None else -9999

    return {
        "sharpe": _f("sharpe"),
        "max_drawdown": _f("maxDrawdown"),
        "avg_trade": _f("avgTrade"),
        "closed_trades": closed,
        "trade_retention_pct": retention,
        "win_rate": _f("winRate"),
        "profit_factor": _f("profitFactor"),
        "total_return": _f("totalReturn"),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked 5-row vol/conf scorecard")
    parser.add_argument("--data", type=Path, default=DATA, help="Path to CSV data file")
    parser.add_argument("--bars", type=int, default=1000, help="Number of bars to use")
    args = parser.parse_args()

    data_path = args.data
    bars = args.bars

    if not BINARY.exists():
        print(f"ERROR: binary not found: {BINARY}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"ERROR: data not found: {data_path}", file=sys.stderr)
        return 1

    rows = []
    for preset in PRESETS:
        print(f"Running {preset} ... ", end="", flush=True)
        res = run_backtest(preset, data_path, bars)
        if res.get("error"):
            print(f"ERROR: {res['error']}")
        else:
            print(f"Sharpe={res['sharpe']:.4f} maxDD={res['max_drawdown']:.4f} trades={res['closed_trades']}")
        rows.append((preset, res))

    # Emit markdown table
    print("\n| preset | sharpe | max_drawdown | avg_trade | closed_trades | trade_retention_pct |")
    print("|--------|--------|--------------|-----------|---------------|---------------------|")
    for preset, res in rows:
        if res.get("error"):
            print(f"| {preset} | ERROR | - | - | - | - |")
        else:
            print(f"| {preset} | {res['sharpe']:.4f} | {res['max_drawdown']:.4f} | {res['avg_trade']:.6f} | {res['closed_trades']} | {res['trade_retention_pct']:.1f}% |")

    # Find best Sharpe
    valid = [(p, r) for p, r in rows if not r.get("error")]
    if valid:
        best_preset, best_res = max(valid, key=lambda x: x[1]["sharpe"])
        print(f"\nBest preset by Sharpe: {best_preset} ({best_res['sharpe']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
