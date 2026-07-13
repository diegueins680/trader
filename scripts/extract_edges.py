#!/usr/bin/env python3
"""
Extract edge values from backtest JSON for threshold calibration.

Usage:
    python scripts/extract_edges.py --backtest-json backtest.json --output edges.csv
    python scripts/extract_edges.py --backtest-json backtest.json \
        --series realized-return --output returns.csv

The default output contains one ex-ante decision edge per line, which can be
fed into threshold calibration. Realized trade returns are outcomes and are
available only through the separate ``realized-return`` series.
"""

import argparse
import json
import math
import sys


def _positive_finite(raw) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) and value > 0 else None


def _finite_number(raw) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _backtest_payload(data):
    nested = data.get("backtest") if isinstance(data, dict) else None
    return nested if isinstance(nested, dict) else data


def extract_decision_edges(data) -> list[float]:
    """Extract ex-ante decision/gate edge observations only."""
    payload = _backtest_payload(data)
    if not isinstance(payload, dict):
        return []
    edges = []
    for trace in payload.get("decisionTraces", []):
        if not isinstance(trace, dict):
            continue
        edge = _positive_finite(trace.get("edge", trace.get("entryEdge")))
        if edge is not None:
            edges.append(edge)

    telemetry = payload.get("gateTelemetry", {})
    recent = telemetry.get("recentRejections", []) if isinstance(telemetry, dict) else []
    for rejection in recent:
        if not isinstance(rejection, dict):
            continue
        edge = _positive_finite(rejection.get("edge"))
        if edge is not None:
            edges.append(edge)
    return edges


def extract_realized_returns(data) -> list[float]:
    """Extract signed realized trade returns as a distinct outcome series."""
    payload = _backtest_payload(data)
    if not isinstance(payload, dict):
        return []
    returns = []
    for trade in payload.get("trades", []):
        if not isinstance(trade, dict):
            continue
        reported = _finite_number(trade.get("return", trade.get("pnlPercent")))
        if reported is not None:
            returns.append(reported)
            continue
        entry_price = _positive_finite(trade.get("entryPrice"))
        exit_price = _positive_finite(trade.get("exitPrice"))
        if entry_price is None or exit_price is None:
            continue
        price_return = (exit_price - entry_price) / entry_price
        side = trade.get("side")
        side_multiplier = -1 if side == -1 or str(side).strip().upper() == "SHORT" else 1
        returns.append(side_multiplier * price_return)
    return returns


def extract_edges(backtest_path: str, series: str = "decision-edge") -> list[float]:
    """Extract one homogeneous series from backtest JSON output."""
    with open(backtest_path) as f:
        data = json.load(f)

    if series == "decision-edge":
        return extract_decision_edges(data)
    if series == "realized-return":
        return extract_realized_returns(data)
    raise ValueError(f"Unknown series: {series}")


def main():
    parser = argparse.ArgumentParser(description="Extract edge values from backtest")
    parser.add_argument("--backtest-json", required=True, help="Path to backtest JSON output")
    parser.add_argument("--output", default="edges.csv", help="Output CSV path")
    parser.add_argument(
        "--series",
        choices=("decision-edge", "realized-return"),
        default="decision-edge",
        help="Homogeneous series to export (default: ex-ante decision-edge)",
    )
    args = parser.parse_args()

    edges = extract_edges(args.backtest_json, args.series)
    
    if not edges:
        print(f"WARNING: No {args.series} values found in backtest output", file=sys.stderr)
        sys.exit(1)
    
    with open(args.output, "w") as f:
        for edge in edges:
            f.write(f"{edge:.8f}\n")
    
    print(f"Extracted {len(edges)} {args.series} values to {args.output}")
    print(f"  Min:    {min(edges):.8f}")
    print(f"  Max:    {max(edges):.8f}")
    print(f"  Mean:   {sum(edges)/len(edges):.8f}")
    
    # Compute percentiles
    sorted_edges = sorted(edges)
    n = len(sorted_edges)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int((p / 100.0) * (n - 1))
        print(f"  P{p:02d}:   {sorted_edges[idx]:.8f}")


if __name__ == "__main__":
    main()
