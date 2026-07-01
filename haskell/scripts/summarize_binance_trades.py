#!/usr/bin/env python3
"""Summarize and freshness-check a /binance/trades JSON snapshot.

The report intentionally omits order ids and origin IPs. It is meant for
deployment review: enough aggregate PnL/commission context to validate the
export path without leaking sensitive execution metadata into logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def finite_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def finite_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iso_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.UTC).isoformat()


def summarize(payload: dict[str, Any], now_ms: int, max_age_hours: float) -> dict[str, Any]:
    trades = payload.get("trades")
    if not isinstance(trades, list):
        raise ValueError("snapshot missing trades array")
    fetched_at = finite_int(payload.get("fetchedAtMs"))
    if fetched_at is None:
        raise ValueError("snapshot missing fetchedAtMs")
    age_ms = max(0, now_ms - fetched_at)
    max_age_ms = int(max_age_hours * 3_600_000)
    by_symbol: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "realizedPnl": 0.0, "commission": 0.0, "quoteQty": 0.0})
    min_time: int | None = None
    max_time: int | None = None
    for raw in trades:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "UNKNOWN").upper()
        row = by_symbol[symbol]
        row["count"] += 1
        row["realizedPnl"] += finite_float(raw.get("realizedPnl"))
        row["commission"] += finite_float(raw.get("commission"))
        row["quoteQty"] += finite_float(raw.get("quoteQty"))
        t = finite_int(raw.get("time"))
        if t is not None:
            min_time = t if min_time is None else min(min_time, t)
            max_time = t if max_time is None else max(max_time, t)
    total_realized = sum(row["realizedPnl"] for row in by_symbol.values())
    total_commission = sum(row["commission"] for row in by_symbol.values())
    return {
        "market": payload.get("market"),
        "testnet": payload.get("testnet"),
        "allSymbols": payload.get("allSymbols"),
        "fetchedAtMs": fetched_at,
        "fetchedAt": iso_ms(fetched_at),
        "ageMs": age_ms,
        "stale": age_ms > max_age_ms,
        "maxAgeHours": max_age_hours,
        "tradesCount": sum(row["count"] for row in by_symbol.values()),
        "minTradeTime": iso_ms(min_time),
        "maxTradeTime": iso_ms(max_time),
        "totalRealizedPnl": total_realized,
        "totalCommission": total_commission,
        "netRealizedAfterCommission": total_realized - total_commission,
        "bySymbol": dict(sorted(by_symbol.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw /binance/trades JSON snapshot.")
    parser.add_argument("--max-age-hours", type=float, default=24, help="Freshness budget before the snapshot is marked stale.")
    parser.add_argument("--fail-stale", action="store_true", help="Exit nonzero if the snapshot is stale.")
    parser.add_argument("--now-ms", type=int, default=int(dt.datetime.now(tz=dt.UTC).timestamp() * 1000))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_age_hours <= 0:
        raise SystemExit("--max-age-hours must be > 0")
    payload = json.loads(args.input.read_text())
    if not isinstance(payload, dict):
        raise SystemExit("snapshot root must be an object")
    report = summarize(payload, args.now_ms, args.max_age_hours)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.fail_stale and report["stale"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
