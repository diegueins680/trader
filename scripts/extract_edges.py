#!/usr/bin/env python3
"""
Extract edge values from backtest JSON for threshold calibration.

Usage:
    python scripts/extract_edges.py --backtest-json backtest.json --output edges.csv

The output CSV contains one edge value per line, which can be fed into
threshold calibration to compute data-driven open thresholds.
"""

import argparse
import json
import sys
from pathlib import Path


def extract_edges(backtest_path: str) -> list[float]:
    """Extract edge values from backtest JSON output."""
    with open(backtest_path) as f:
        data = json.load(f)
    
    edges = []
    
    # Try to extract from trades
    trades = data.get("trades", [])
    for trade in trades:
        # Edge can be computed from entry/exit prices
        entry_price = trade.get("entryPrice", 0)
        exit_price = trade.get("exitPrice", 0)
        if entry_price > 0 and exit_price > 0:
            edge = abs(exit_price - entry_price) / entry_price
            edges.append(edge)
    
    # Try to extract from decision trace if available
    decision_traces = data.get("decisionTraces", [])
    for trace in decision_traces:
        edge = trace.get("edge", trace.get("entryEdge", 0))
        if edge > 0:
            edges.append(edge)
    
    # Try to extract from gate telemetry if available
    telemetry = data.get("gateTelemetry", {})
    recent = telemetry.get("recentRejections", [])
    for rejection in recent:
        edge = rejection.get("edge", 0)
        if edge > 0:
            edges.append(edge)
    
    return edges


def main():
    parser = argparse.ArgumentParser(description="Extract edge values from backtest")
    parser.add_argument("--backtest-json", required=True, help="Path to backtest JSON output")
    parser.add_argument("--output", default="edges.csv", help="Output CSV path")
    args = parser.parse_args()
    
    edges = extract_edges(args.backtest_json)
    
    if not edges:
        print("WARNING: No edge values found in backtest output", file=sys.stderr)
        print("The backtest may have produced zero trades or lacks edge metadata", file=sys.stderr)
        sys.exit(1)
    
    with open(args.output, "w") as f:
        for edge in edges:
            f.write(f"{edge:.8f}\n")
    
    print(f"Extracted {len(edges)} edge values to {args.output}")
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
