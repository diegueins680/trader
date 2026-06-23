#!/usr/bin/env python3
"""Analyze trade-log NDJSON by method, symbol, side, gate, and exit reason."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TradeRecord:
    line_no: int
    timestamp: str
    symbol: str
    side: str
    method: str
    vol_conf_gate: str
    exit_reason: str
    pnl_pct: float
    pnl: float
    trade_id: str


@dataclass
class GroupStats:
    key: str
    trades: int = 0
    total_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    wins: int = 0

    def add(self, trade: TradeRecord) -> None:
        self.trades += 1
        self.total_pnl_pct += trade.pnl_pct
        self.total_pnl += trade.pnl
        if trade.pnl_pct > 0:
            self.wins += 1

    @property
    def avg_pnl_pct(self) -> float:
        return self.total_pnl_pct / self.trades if self.trades else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


def maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.endswith("%"):
            value = value[:-1]
            scale = 0.01
        else:
            scale = 1.0
    else:
        scale = 1.0
    try:
        result = float(value) * scale
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def first_value(payload: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def normalized_text(payload: dict[str, Any], names: Iterable[str], default: str) -> str:
    value = first_value(payload, names)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def parse_trade(payload: dict[str, Any], line_no: int) -> TradeRecord | None:
    pnl_pct = maybe_float(first_value(payload, ["pnlPercent", "pnl_pct", "return", "returnPct"]))
    if pnl_pct is None:
        return None
    pnl = maybe_float(first_value(payload, ["pnl", "pnl_quote", "realizedPnl"])) or 0.0
    return TradeRecord(
        line_no=line_no,
        timestamp=normalized_text(payload, ["timestamp", "time", "closedAt"], ""),
        symbol=normalized_text(payload, ["symbol"], "unknown").upper(),
        side=normalized_text(payload, ["side"], "unknown").upper(),
        method=normalized_text(payload, ["method", "signal_method"], "unknown"),
        vol_conf_gate=normalized_text(payload, ["volConfGate", "vol_conf_gate"], "unknown"),
        exit_reason=normalized_text(payload, ["exitReason", "exit_reason"], "unknown"),
        pnl_pct=pnl_pct,
        pnl=pnl,
        trade_id=normalized_text(payload, ["tradeId", "trade_id"], ""),
    )


def load_trades(path: Path, dedupe: bool = True) -> tuple[list[TradeRecord], int, int]:
    records: list[TradeRecord] = []
    seen: set[tuple[Any, ...]] = set()
    skipped = 0
    duplicate_count = 0
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(payload, dict):
            skipped += 1
            continue
        record = parse_trade(payload, line_no)
        if record is None:
            skipped += 1
            continue
        key = dedupe_key(record)
        if dedupe and key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        records.append(record)
    return records, skipped, duplicate_count


def dedupe_key(record: TradeRecord) -> tuple[Any, ...]:
    if record.trade_id:
        return ("trade_id", record.trade_id)
    return (
        "logical",
        record.timestamp,
        record.symbol,
        record.side,
        record.method,
        record.vol_conf_gate,
        record.exit_reason,
        round(record.pnl_pct, 12),
        round(record.pnl, 12),
    )


def add_group(groups: dict[str, GroupStats], key: str, trade: TradeRecord) -> None:
    groups.setdefault(key, GroupStats(key)).add(trade)


def summarize_dimension(records: list[TradeRecord], dimension: str) -> list[GroupStats]:
    groups: dict[str, GroupStats] = {}
    for trade in records:
        if dimension == "method":
            key = trade.method
        elif dimension == "symbol":
            key = trade.symbol
        elif dimension == "side":
            key = trade.side
        elif dimension == "exitReason":
            key = trade.exit_reason
        elif dimension == "methodSymbol":
            key = f"{trade.method}|{trade.symbol}"
        elif dimension == "methodVolGateSide":
            key = f"{trade.method}|{trade.vol_conf_gate}|{trade.side}"
        else:
            raise ValueError(f"unknown dimension: {dimension}")
        add_group(groups, key, trade)
    return list(groups.values())


def row_json(row: GroupStats) -> dict[str, Any]:
    return {
        "key": row.key,
        "trades": row.trades,
        "totalPnlPct": row.total_pnl_pct,
        "avgPnlPct": row.avg_pnl_pct,
        "winRate": row.win_rate,
        "totalPnl": row.total_pnl,
    }


def ranked(rows: list[GroupStats], top: int, reverse: bool, positive: bool) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.total_pnl_pct > 0] if positive else [row for row in rows if row.total_pnl_pct < 0]
    return [
        row_json(row)
        for row in sorted(rows, key=lambda row: (row.total_pnl_pct, row.trades), reverse=reverse)[:top]
    ]


def build_report(records: list[TradeRecord], skipped: int, duplicates: int, min_group_trades: int, top: int) -> dict[str, Any]:
    dimensions = ["method", "symbol", "side", "exitReason", "methodSymbol", "methodVolGateSide"]
    groups = {dimension: summarize_dimension(records, dimension) for dimension in dimensions}
    method_rows = groups["method"]
    symbol_rows = groups["symbol"]
    exit_rows = groups["exitReason"]

    def favorable(rows: list[GroupStats]) -> list[str]:
        return [
            row.key
            for row in sorted(rows, key=lambda row: (row.total_pnl_pct, row.trades), reverse=True)
            if row.trades >= min_group_trades and row.total_pnl_pct > 0
        ]

    def unfavorable(rows: list[GroupStats]) -> list[str]:
        return [
            row.key
            for row in sorted(rows, key=lambda row: (row.total_pnl_pct, -row.trades))
            if row.trades >= min_group_trades and row.total_pnl_pct < 0
        ]

    return {
        "summary": {
            "trades": len(records),
            "skippedLines": skipped,
            "duplicateTrades": duplicates,
            "totalPnlPct": sum(trade.pnl_pct for trade in records),
            "totalPnl": sum(trade.pnl for trade in records),
            "minGroupTrades": min_group_trades,
        },
        "recommendations": {
            "favorMethods": favorable(method_rows),
            "avoidMethods": unfavorable(method_rows),
            "favorSymbols": favorable(symbol_rows),
            "avoidSymbols": unfavorable(symbol_rows),
            "reviewExitReasons": unfavorable(exit_rows),
        },
        "groups": {
            dimension: {
                "gainers": ranked(rows, top, True, True),
                "losers": ranked(rows, top, False, False),
            }
            for dimension, rows in groups.items()
        },
    }


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_money(value: float) -> str:
    return f"{value:.4f}"


def render_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| group | trades | total % | avg % | win % | pnl |", "|---|---:|---:|---:|---:|---:|"]
    if not rows:
        lines.append("| none | 0 | 0.00% | 0.00% | 0.00% | 0.0000 |")
    for row in rows:
        lines.append(
            "| {key} | {trades} | {total} | {avg} | {win} | {pnl} |".format(
                key=row["key"],
                trades=row["trades"],
                total=fmt_pct(row["totalPnlPct"]),
                avg=fmt_pct(row["avgPnlPct"]),
                win=fmt_pct(row["winRate"]),
                pnl=fmt_money(row["totalPnl"]),
            )
        )
    lines.append("")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    recs = report["recommendations"]
    lines = [
        "# Position History Analysis",
        "",
        f"- Trades analyzed: {summary['trades']}",
        f"- Duplicate trades ignored: {summary['duplicateTrades']}",
        f"- Skipped lines: {summary['skippedLines']}",
        f"- Total return contribution: {fmt_pct(summary['totalPnlPct'])}",
        f"- Total PnL: {fmt_money(summary['totalPnl'])}",
        "",
        "## Recommendations",
        "",
        f"- Favor methods: {', '.join(recs['favorMethods']) or 'none'}",
        f"- Avoid methods: {', '.join(recs['avoidMethods']) or 'none'}",
        f"- Favor symbols: {', '.join(recs['favorSymbols']) or 'none'}",
        f"- Avoid symbols: {', '.join(recs['avoidSymbols']) or 'none'}",
        f"- Review exit reasons: {', '.join(recs['reviewExitReasons']) or 'none'}",
        "",
    ]
    groups = report["groups"]
    for dimension in ["method", "symbol", "exitReason", "methodVolGateSide"]:
        lines.extend(render_table(f"Biggest Gains By {dimension}", groups[dimension]["gainers"]))
        lines.extend(render_table(f"Biggest Losses By {dimension}", groups[dimension]["losers"]))
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Trade-log NDJSON path.")
    parser.add_argument("--output", type=Path, help="Optional output path. Defaults to stdout.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    parser.add_argument("--top", type=int, default=10, help="Rows per gain/loss table.")
    parser.add_argument("--min-group-trades", type=int, default=5, help="Minimum trades before a group can be recommended.")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not drop duplicate trade ids or identical logical rows.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top <= 0:
        raise SystemExit("--top must be > 0")
    if args.min_group_trades <= 0:
        raise SystemExit("--min-group-trades must be > 0")
    records, skipped, duplicates = load_trades(args.input, dedupe=not args.no_dedupe)
    report = build_report(records, skipped, duplicates, args.min_group_trades, args.top)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.json else render_markdown(report)
    if args.output:
        args.output.write_text(output)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
