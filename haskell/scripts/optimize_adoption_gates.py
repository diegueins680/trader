#!/usr/bin/env python3
"""Recommend relaxed live-adoption gates from a top-combos JSON payload.

The optimizer is intentionally evidence-first: it only relaxes thresholds from
combos that already carry all required readings (freshness, minEdge,
tradeCount, walk-forward Sharpe mean, and walk-forward Sharpe std). It never
turns missing evidence into a pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_MIN_EDGE = 0.0018
DEFAULT_MIN_TRADES = 20
DEFAULT_MIN_WF_SHARPE_MEAN = 0.3
DEFAULT_MAX_WF_SHARPE_STD = 1.5


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def combo_fields(combo: dict[str, Any]) -> dict[str, Any]:
    params = combo.get("params") if isinstance(combo.get("params"), dict) else {}
    metrics = combo.get("metrics") if isinstance(combo.get("metrics"), dict) else {}
    wf = metrics.get("walkForwardSummary") if isinstance(metrics.get("walkForwardSummary"), dict) else {}
    live = metrics.get("live") if isinstance(metrics.get("live"), dict) else combo.get("live")
    live = live if isinstance(live, dict) else {}
    return {
        "uuid": combo.get("uuid"),
        "rank": combo.get("rank"),
        "symbol": params.get("binanceSymbol") or params.get("symbol") or combo.get("symbol"),
        "interval": params.get("interval") or combo.get("interval"),
        "method": params.get("method") or combo.get("method"),
        "createdAtMs": finite_int(combo.get("createdAtMs")),
        "backtestRefreshedAtMs": finite_int(combo.get("backtestRefreshedAtMs")),
        "minEdge": finite_float(params.get("minEdge")),
        "tradeCount": finite_int(metrics.get("tradeCount")),
        "wfSharpeMean": finite_float(wf.get("sharpeMean")),
        "wfSharpeStd": finite_float(wf.get("sharpeStd")),
        "totalReturn": finite_float(metrics.get("totalReturn")),
        "finalEquity": finite_float(metrics.get("finalEquity") or combo.get("finalEquity")),
        "liveFinalEquity": finite_float(live.get("finalEquity")),
        "liveOperationCount": finite_int(live.get("operationCount")) or 0,
    }


def pnl_score(row: dict[str, Any]) -> float:
    if row["liveOperationCount"] > 0 and row["liveFinalEquity"] is not None:
        return row["liveFinalEquity"] - 1
    if row["totalReturn"] is not None:
        return row["totalReturn"]
    if row["finalEquity"] is not None:
        return row["finalEquity"] - 1
    return 0.0


def freshness_ts(row: dict[str, Any]) -> int | None:
    return row["backtestRefreshedAtMs"] or row["createdAtMs"]


def is_fresh(row: dict[str, Any], now_ms: int, max_age_days: float, allow_missing: bool) -> bool:
    ts = freshness_ts(row)
    if ts is None:
        return allow_missing
    return max(0, now_ms - ts) <= int(max_age_days * 86_400_000)


def has_full_evidence(row: dict[str, Any], now_ms: int, max_age_days: float, allow_missing_freshness: bool) -> bool:
    return (
        is_fresh(row, now_ms, max_age_days, allow_missing_freshness)
        and row["minEdge"] is not None
        and row["tradeCount"] is not None
        and row["wfSharpeMean"] is not None
        and row["wfSharpeStd"] is not None
    )


def passes(row: dict[str, Any], gates: dict[str, float | int], now_ms: int, max_age_days: float, allow_missing_freshness: bool) -> bool:
    return (
        has_full_evidence(row, now_ms, max_age_days, allow_missing_freshness)
        and row["minEdge"] >= gates["minEdgeFloor"]
        and row["tradeCount"] >= gates["minTradeCount"]
        and row["wfSharpeMean"] >= gates["minWfSharpeMean"]
        and (gates["maxWfSharpeStd"] <= 0 or row["wfSharpeStd"] <= gates["maxWfSharpeStd"])
    )


def optimize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    gates: dict[str, float | int] = {
        "minEdgeFloor": args.min_edge_floor,
        "minTradeCount": args.min_trade_count,
        "minWfSharpeMean": args.min_wf_sharpe_mean,
        "maxWfSharpeStd": args.max_wf_sharpe_std,
    }
    full = [row for row in rows if has_full_evidence(row, args.now_ms, args.max_age_days, args.allow_missing_freshness)]
    strict = [row for row in full if passes(row, gates, args.now_ms, args.max_age_days, args.allow_missing_freshness)]
    selected = sorted(strict, key=pnl_score, reverse=True)[: args.target_count]
    relaxed = False
    if len(selected) < args.target_count:
        selected = sorted(full, key=pnl_score, reverse=True)[: args.target_count]
        if selected:
            relaxed = True
            gates = {
                "minEdgeFloor": min(args.min_edge_floor, min(row["minEdge"] for row in selected)),
                "minTradeCount": min(args.min_trade_count, min(row["tradeCount"] for row in selected)),
                "minWfSharpeMean": min(args.min_wf_sharpe_mean, min(row["wfSharpeMean"] for row in selected)),
                "maxWfSharpeStd": 0
                if args.max_wf_sharpe_std <= 0
                else max(args.max_wf_sharpe_std, max(row["wfSharpeStd"] for row in selected)),
            }
            strict = [row for row in full if passes(row, gates, args.now_ms, args.max_age_days, args.allow_missing_freshness)]

    def public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "uuid": row["uuid"],
            "rank": row["rank"],
            "symbol": row["symbol"],
            "interval": row["interval"],
            "method": row["method"],
            "pnlScore": pnl_score(row),
            "minEdge": row["minEdge"],
            "tradeCount": row["tradeCount"],
            "wfSharpeMean": row["wfSharpeMean"],
            "wfSharpeStd": row["wfSharpeStd"],
            "freshnessMs": freshness_ts(row),
        }

    cli_args = [
        "--adoption-min-edge-floor",
        f"{gates['minEdgeFloor']:.12g}",
        "--adoption-min-trade-count",
        str(gates["minTradeCount"]),
        "--adoption-min-wf-sharpe-mean",
        f"{gates['minWfSharpeMean']:.12g}",
        "--adoption-max-wf-sharpe-std",
        f"{gates['maxWfSharpeStd']:.12g}",
    ]
    return {
        "inputCombos": len(rows),
        "fullyEvidencedCombos": len(full),
        "passingCombos": len(strict),
        "relaxed": relaxed,
        "targetCount": args.target_count,
        "gates": gates,
        "cliArgs": cli_args,
        "selected": [public_row(row) for row in selected],
        "topPassing": [public_row(row) for row in sorted(strict, key=pnl_score, reverse=True)[: args.top]],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-combos", required=True, type=Path, help="top-combos JSON path.")
    parser.add_argument("--target-count", type=int, default=1, help="Minimum number of fully evidenced combos to admit.")
    parser.add_argument("--top", type=int, default=10, help="Top passing combos to include in the report.")
    parser.add_argument("--min-edge-floor", type=float, default=DEFAULT_MIN_EDGE)
    parser.add_argument("--min-trade-count", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--min-wf-sharpe-mean", type=float, default=DEFAULT_MIN_WF_SHARPE_MEAN)
    parser.add_argument("--max-wf-sharpe-std", type=float, default=DEFAULT_MAX_WF_SHARPE_STD)
    parser.add_argument("--max-age-days", type=float, default=14)
    parser.add_argument("--allow-missing-freshness", action="store_true")
    parser.add_argument(
        "--now-ms",
        type=int,
        default=int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000),
        help="Evaluation time in epoch milliseconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_count <= 0:
        raise SystemExit("--target-count must be > 0")
    if args.top <= 0:
        raise SystemExit("--top must be > 0")
    payload = json.loads(args.top_combos.read_text())
    combos = payload.get("combos", payload if isinstance(payload, list) else [])
    rows = [combo_fields(combo) for combo in combos if isinstance(combo, dict)]
    report = optimize(rows, args)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
