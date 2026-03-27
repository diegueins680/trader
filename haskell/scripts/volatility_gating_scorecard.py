#!/usr/bin/env python3
"""Render a volatility-gating markdown scorecard from existing backtest JSON outputs.

This is a narrow research support tool: it does not run backtests. It consumes already
produced JSON artifacts and emits a 5-row-ready markdown table with the firm contract
applied relative to the baseline row.

Accepted input shapes per file:
1. {"backtest": {"metrics": {...}}}
2. {"metrics": {...}}
3. {...metrics fields directly at top level...}

Metric mapping:
- Sharpe -> metrics.sharpe
- Max drawdown -> metrics.maxDrawdown
- Expectancy -> metrics.expectancy, else metrics.avgTradeReturn
- Trade count -> metrics.tradeCount
- Closed trades -> metrics.roundTrips, else metrics.tradeCount
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ScoreRow:
    label: str
    path: Path
    sharpe: float
    max_drawdown: float
    expectancy: float
    trade_count: int
    closed_trades: int


@dataclass(frozen=True)
class Contract:
    min_sharpe_delta: float
    max_drawdown_regression: float
    min_trade_retention: float
    min_expectancy_uplift: float
    min_closed_trades: int


@dataclass(frozen=True)
class EvaluatedRow:
    row: ScoreRow
    trade_retention: float | None
    verdict: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Input row in the rendered table. Use once per row; the first row is baseline unless --baseline-label is set.",
    )
    parser.add_argument(
        "--baseline-label",
        help="Explicit label to treat as baseline. Defaults to the first --row.",
    )
    parser.add_argument(
        "--min-sharpe-delta",
        type=float,
        default=0.10,
        help="Required Sharpe improvement vs baseline.",
    )
    parser.add_argument(
        "--max-dd-regression",
        type=float,
        default=0.02,
        help="Maximum allowed max-drawdown regression vs baseline, expressed as a fraction (0.02 = 2 percentage points).",
    )
    parser.add_argument(
        "--min-trade-retention",
        type=float,
        default=0.60,
        help="Minimum retained trade count vs baseline unless expectancy improves enough.",
    )
    parser.add_argument(
        "--min-expectancy-uplift",
        type=float,
        default=0.10,
        help="Minimum required expectancy improvement vs baseline when trade retention is below the threshold.",
    )
    parser.add_argument(
        "--min-closed-trades",
        type=int,
        default=50,
        help="Minimum closed-trade count required for a non-inconclusive row.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to stdout.",
    )
    return parser.parse_args()


def parse_row_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"invalid --row '{raw}': expected LABEL=PATH")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"invalid --row '{raw}': empty label")
    path = Path(path_str.strip())
    if not path:
        raise ValueError(f"invalid --row '{raw}': empty path")
    return label, path


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def maybe_int(value: Any) -> int | None:
    as_float = maybe_float(value)
    if as_float is None:
        return None
    return int(round(as_float))


def as_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def find_metrics(payload: Any) -> dict[str, Any]:
    candidates: list[dict[str, Any] | None] = []
    top = as_mapping(payload)
    if top is not None:
        backtest = as_mapping(top.get("backtest"))
        if backtest is not None:
            candidates.append(as_mapping(backtest.get("metrics")))
            candidates.append(backtest)
        candidates.append(as_mapping(top.get("metrics")))
        candidates.append(top)
    metric_keys = {"sharpe", "maxDrawdown", "tradeCount", "roundTrips", "avgTradeReturn", "expectancy"}
    for candidate in candidates:
        if candidate is not None and metric_keys.intersection(candidate.keys()):
            return candidate
    raise ValueError("unable to locate metrics in payload")


def require_float(metrics: dict[str, Any], keys: Iterable[str], *, context: str) -> float:
    for key in keys:
        value = maybe_float(metrics.get(key))
        if value is not None:
            return value
    joined = "/".join(keys)
    raise ValueError(f"{context}: missing numeric {joined}")


def require_int(metrics: dict[str, Any], keys: Iterable[str], *, context: str) -> int:
    for key in keys:
        value = maybe_int(metrics.get(key))
        if value is not None:
            return value
    joined = "/".join(keys)
    raise ValueError(f"{context}: missing numeric {joined}")


def load_row(label: str, path: Path) -> ScoreRow:
    payload = load_json(path)
    metrics = find_metrics(payload)
    context = f"{label} ({path})"
    return ScoreRow(
        label=label,
        path=path,
        sharpe=require_float(metrics, ["sharpe"], context=context),
        max_drawdown=require_float(metrics, ["maxDrawdown"], context=context),
        expectancy=require_float(metrics, ["expectancy", "avgTradeReturn"], context=context),
        trade_count=require_int(metrics, ["tradeCount"], context=context),
        closed_trades=require_int(metrics, ["roundTrips", "tradeCount"], context=context),
    )


def expectancy_uplift_ok(candidate: ScoreRow, baseline: ScoreRow, min_uplift: float) -> bool:
    required_gain = max(abs(baseline.expectancy) * min_uplift, 1e-12)
    return candidate.expectancy >= baseline.expectancy + required_gain


def evaluate_rows(rows: list[ScoreRow], baseline_label: str | None, contract: Contract) -> list[EvaluatedRow]:
    if not rows:
        raise ValueError("at least one --row is required")

    baseline = None
    if baseline_label is None:
        baseline = rows[0]
    else:
        for row in rows:
            if row.label == baseline_label:
                baseline = row
                break
        if baseline is None:
            raise ValueError(f"--baseline-label '{baseline_label}' does not match any row label")

    evaluated: list[EvaluatedRow] = []
    for row in rows:
        if row.label == baseline.label:
            evaluated.append(
                EvaluatedRow(
                    row=row,
                    trade_retention=1.0,
                    verdict="baseline",
                    reasons=("reference",),
                )
            )
            continue

        reasons: list[str] = []
        retention = None if baseline.trade_count <= 0 else row.trade_count / baseline.trade_count

        if row.sharpe < baseline.sharpe + contract.min_sharpe_delta:
            reasons.append(f"sharpe<{baseline.sharpe + contract.min_sharpe_delta:.3f}")
        if row.max_drawdown > baseline.max_drawdown + contract.max_drawdown_regression:
            reasons.append(f"maxDD>{(baseline.max_drawdown + contract.max_drawdown_regression) * 100:.2f}%")
        retention_ok = retention is not None and retention >= contract.min_trade_retention
        expectancy_ok = expectancy_uplift_ok(row, baseline, contract.min_expectancy_uplift)
        if not (retention_ok or expectancy_ok):
            reasons.append(
                f"retention<{contract.min_trade_retention * 100:.1f}% and expectancy<{baseline.expectancy + max(abs(baseline.expectancy) * contract.min_expectancy_uplift, 1e-12):.6f}"
            )
        if row.closed_trades < contract.min_closed_trades:
            reasons.append(f"closedTrades<{contract.min_closed_trades}")

        verdict = "pass" if not reasons else "fail"
        evaluated.append(
            EvaluatedRow(
                row=row,
                trade_retention=retention,
                verdict=verdict,
                reasons=tuple(reasons),
            )
        )
    return evaluated


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_retention(value: float | None) -> str:
    return "n/a" if value is None else pct(value, 1)


def render_markdown(evaluated: list[EvaluatedRow], contract: Contract) -> str:
    lines: list[str] = []
    lines.append("## Scorecard")
    lines.append("")
    lines.append(
        "| Row | Sharpe | Max drawdown | Expectancy/trade | Trade-count retention | Closed-trade count | Contract |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for item in evaluated:
        verdict = item.verdict if item.verdict == "baseline" else f"{item.verdict} ({'; '.join(item.reasons)})"
        lines.append(
            "| {label} | {sharpe:.3f} | {max_dd} | {expectancy} | {retention} | {closed_trades} | {verdict} |".format(
                label=item.row.label,
                sharpe=item.row.sharpe,
                max_dd=pct(item.row.max_drawdown),
                expectancy=pct(item.row.expectancy),
                retention=fmt_retention(item.trade_retention),
                closed_trades=item.row.closed_trades,
                verdict=verdict,
            )
        )

    lines.append("")
    lines.append("### Source artifacts")
    lines.append("")
    for item in evaluated:
        lines.append(f"- `{item.row.label}` → `{item.row.path}`")

    lines.append("")
    lines.append("### Firm contract applied")
    lines.append("")
    lines.append(f"- Sharpe improvement vs baseline: `>= +{contract.min_sharpe_delta:.2f}`")
    lines.append(f"- Max drawdown regression vs baseline: `<= +{pct(contract.max_drawdown_regression)}`")
    lines.append(
        f"- Trade-count retention: `>= {contract.min_trade_retention * 100:.0f}%` of baseline unless expectancy/trade improves by `>= {contract.min_expectancy_uplift * 100:.0f}%`"
    )
    lines.append(f"- Closed-trade minimum: `>= {contract.min_closed_trades}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        row_specs = [parse_row_arg(raw) for raw in args.row]
        rows = [load_row(label, path) for label, path in row_specs]
        evaluated = evaluate_rows(
            rows,
            args.baseline_label,
            Contract(
                min_sharpe_delta=args.min_sharpe_delta,
                max_drawdown_regression=args.max_dd_regression,
                min_trade_retention=args.min_trade_retention,
                min_expectancy_uplift=args.min_expectancy_uplift,
                min_closed_trades=args.min_closed_trades,
            ),
        )
        output = render_markdown(
            evaluated,
            Contract(
                min_sharpe_delta=args.min_sharpe_delta,
                max_drawdown_regression=args.max_dd_regression,
                min_trade_retention=args.min_trade_retention,
                min_expectancy_uplift=args.min_expectancy_uplift,
                min_closed_trades=args.min_closed_trades,
            ),
        )
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output)
        else:
            sys.stdout.write(output)
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
