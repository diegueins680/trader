#!/usr/bin/env python3
"""Replay a local trading day from persisted bot-state snapshots.

This script is intentionally dependency-free so it can run anywhere the repo can
run Python 3. It scans one tenant's `bot-state-*.json` snapshots, reconstructs
completed trades and same-day open positions for a local calendar date, and
flags order-evidence gaps where the saved order event only shows an ack-like
`NEW` status with zero executed quantity.

Regime labels use explicit 24-bar diagnostics:
- `high-vol` when realized per-bar volatility >= 1.5%
- `trend-up` / `trend-down` when efficiency >= 0.45 and |z| >= 1.0
- `chop` when efficiency <= 0.25
- `range-drift` otherwise
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ACK_ONLY_STATUSES = {
    "new",
    "pendingcancel",
}

INTERVAL_UNITS_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    # Keep month semantics consistent with the repo's coarse interval handling.
    "M": 30 * 86_400_000,
}


@dataclass(frozen=True)
class BotSnapshot:
    path: Path
    status: dict[str, Any]

    @property
    def symbol(self) -> str:
        return str(self.status.get("symbol") or self.path.stem.replace("bot-state-", ""))

    @property
    def updated_at_ms(self) -> int:
        raw = self.status.get("updatedAtMs")
        return int(raw) if isinstance(raw, (int, float)) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Local trade date to replay (YYYY-MM-DD).")
    parser.add_argument(
        "--timezone",
        default="America/Guayaquil",
        help="IANA timezone used to define the local trading day.",
    )
    parser.add_argument(
        "--tenant-dir",
        help=(
            "Explicit tenant snapshot directory. "
            "Defaults to the tenant with the latest updatedAtMs under haskell/.tmp/bot/tenants."
        ),
    )
    parser.add_argument(
        "--bot-root",
        default="haskell/.tmp/bot/tenants",
        help="Root directory containing tenant bot-state snapshots.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def load_snapshot(path: Path) -> BotSnapshot | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    status = payload.get("status")
    if not isinstance(status, dict):
        return None
    return BotSnapshot(path=path, status=status)


def choose_tenant_dir(bot_root: Path) -> Path:
    best_dir: Path | None = None
    best_updated = -1
    for tenant_dir in sorted(p for p in bot_root.iterdir() if p.is_dir()):
        for snap_path in tenant_dir.glob("bot-state-*.json"):
            snapshot = load_snapshot(snap_path)
            if snapshot is None:
                continue
            if snapshot.updated_at_ms > best_updated:
                best_updated = snapshot.updated_at_ms
                best_dir = tenant_dir
    if best_dir is None:
        raise FileNotFoundError(f"no tenant snapshots found under {bot_root}")
    return best_dir


def interval_ms(interval: str | None) -> int | None:
    if not interval or len(interval) < 2:
        return None
    unit = interval[-1]
    if unit not in INTERVAL_UNITS_MS:
        return None
    try:
        amount = int(interval[:-1])
    except ValueError:
        return None
    return amount * INTERVAL_UNITS_MS[unit]


def infer_open_times(snapshot: BotSnapshot) -> list[int]:
    prices = snapshot.status.get("prices") or []
    open_times = snapshot.status.get("openTimes") or []
    if isinstance(open_times, list) and len(open_times) == len(prices):
        return [int(x) for x in open_times]
    ms = interval_ms(snapshot.status.get("interval"))
    last_kline = snapshot.status.get("fetchedLastKline") or {}
    last_open_time = last_kline.get("openTime")
    if ms is None or last_open_time is None or not prices:
        return []
    last_open_time = int(last_open_time)
    return [last_open_time - (len(prices) - 1 - i) * ms for i in range(len(prices))]


def local_iso(ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ms / 1000, tz).isoformat()


def safe_pct(value: float | None) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return value * 100.0


def classify_regime(prices: list[float], idx: int, window_bars: int = 24) -> dict[str, Any]:
    lo = max(1, idx - window_bars + 1)
    window = prices[lo - 1 : idx + 1]
    if len(window) < 3:
        return {
            "label": "insufficient",
            "lookbackBars": len(window),
            "netReturnPct": 0.0,
            "realizedVolPct": 0.0,
            "efficiency": 0.0,
            "zScore": 0.0,
        }
    returns = []
    for prev, cur in zip(window, window[1:]):
        if prev:
            returns.append(cur / prev - 1)
    if not returns:
        return {
            "label": "insufficient",
            "lookbackBars": len(window),
            "netReturnPct": 0.0,
            "realizedVolPct": 0.0,
            "efficiency": 0.0,
            "zScore": 0.0,
        }
    net = window[-1] / window[0] - 1 if window[0] else 0.0
    path = sum(abs(ret) for ret in returns)
    efficiency = 0.0 if path == 0 else abs(net) / path
    mean = sum(returns) / len(returns)
    variance = sum((ret - mean) ** 2 for ret in returns) / max(1, len(returns) - 1)
    realized_vol = math.sqrt(variance)
    z_score = 0.0 if realized_vol == 0 else net / (realized_vol * math.sqrt(len(returns)))
    if realized_vol >= 0.015:
        label = "high-vol"
    elif efficiency >= 0.45 and abs(z_score) >= 1.0:
        label = "trend-up" if net > 0 else "trend-down"
    elif efficiency <= 0.25:
        label = "chop"
    else:
        label = "range-drift"
    return {
        "label": label,
        "lookbackBars": len(window),
        "netReturnPct": net * 100.0,
        "realizedVolPct": realized_vol * 100.0,
        "efficiency": efficiency,
        "zScore": z_score,
    }


def infer_trade_side(snapshot: BotSnapshot, entry_idx: int, exit_idx: int) -> str:
    positions = snapshot.status.get("positions") or []
    for idx in range(entry_idx, min(exit_idx + 1, len(positions))):
        pos = positions[idx]
        if pos > 0:
            return "long"
        if pos < 0:
            return "short"
    return "unknown"


def float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def is_ack_only_order(order_event: dict[str, Any]) -> bool:
    order = order_event.get("order") or {}
    if not order.get("sent"):
        return False
    executed_qty = float_or_none(order.get("executedQty"))
    if executed_qty is not None and executed_qty > 0:
        return False
    status = str(order.get("status") or "").strip().lower()
    return status in ACK_ONLY_STATUSES


def order_event_at_ms(order_event: dict[str, Any]) -> int | None:
    at_ms = order_event.get("atMs")
    return int(at_ms) if isinstance(at_ms, (int, float)) else None


def open_side_to_order_side(side: Any) -> str | None:
    side_text = str(side or "").strip().lower()
    if side_text == "long":
        return "BUY"
    if side_text == "short":
        return "SELL"
    return None


def order_matches_open_side(order_event: dict[str, Any], open_side: Any) -> bool:
    expected = open_side_to_order_side(open_side)
    if expected is None:
        return False
    op_side = str(order_event.get("opSide") or (order_event.get("order") or {}).get("side") or "").strip().upper()
    return op_side == expected


def is_adoption_no_order(order_event: dict[str, Any], open_side: Any) -> bool:
    order = order_event.get("order") or {}
    if order.get("sent") is not False:
        return False
    side_text = str(open_side or "").strip().lower()
    if not side_text:
        return False
    message = str(order.get("message") or "").strip().lower()
    return f"already {side_text}" in message


def pick_preferred_order_event(order_events: list[dict[str, Any]], entry_idx: int) -> dict[str, Any] | None:
    if not order_events:
        return None
    same_index = [event for event in order_events if event.get("index") == entry_idx]
    candidates = same_index or order_events
    return max(candidates, key=lambda event: order_event_at_ms(event) or -1)


def build_open_position_provenance(
    snapshot: BotSnapshot,
    open_trade: dict[str, Any],
    entry_idx: int,
    start_ms: int,
    end_ms: int,
    tz: ZoneInfo,
) -> dict[str, Any]:
    matching_orders = [
        order_event for order_event in snapshot.status.get("orders") or [] if order_matches_open_side(order_event, open_trade.get("side"))
    ]
    sent_orders = [order_event for order_event in matching_orders if (order_event.get("order") or {}).get("sent")]
    same_day_sent = [
        order_event
        for order_event in sent_orders
        if ((at_ms := order_event_at_ms(order_event)) is not None and start_ms <= at_ms < end_ms)
    ]
    prior_sent = [
        order_event
        for order_event in sent_orders
        if ((at_ms := order_event_at_ms(order_event)) is not None and at_ms < start_ms)
    ]
    adoption_orders = [
        order_event
        for order_event in matching_orders
        if is_adoption_no_order(order_event, open_trade.get("side"))
        and ((at_ms := order_event_at_ms(order_event)) is not None and start_ms <= at_ms < end_ms)
    ]

    supporting_order = pick_preferred_order_event(same_day_sent, entry_idx)
    adoption_order = pick_preferred_order_event(adoption_orders, entry_idx)
    if supporting_order is None:
        supporting_order = pick_preferred_order_event(prior_sent, entry_idx)

    details: dict[str, Any] = {}
    if supporting_order is not None:
        supporting_order_at = order_event_at_ms(supporting_order)
        supporting_order_payload = supporting_order.get("order") or {}
        details.update(
            {
                "supportingOrderAtLocal": local_iso(supporting_order_at, tz) if supporting_order_at is not None else None,
                "supportingOrderMessage": supporting_order_payload.get("message"),
                "supportingOrderStatus": supporting_order_payload.get("status"),
            }
        )
    if adoption_order is not None:
        adoption_at = order_event_at_ms(adoption_order)
        adoption_payload = adoption_order.get("order") or {}
        details.update(
            {
                "adoptionEventAtLocal": local_iso(adoption_at, tz) if adoption_at is not None else None,
                "adoptionMessage": adoption_payload.get("message"),
            }
        )

    if same_day_sent:
        details.update(
            {
                "kind": "entered_today",
                "provenance": "same_day_order_evidence",
                "provenanceDetail": "Matched saved opening order evidence on the target day.",
            }
        )
        return details

    if prior_sent:
        details.update(
            {
                "kind": "carried_in",
                "provenance": "prior_day_order_evidence",
                "provenanceDetail": (
                    "Same-day no-order carry event matched prior-day opening order evidence."
                    if adoption_order is not None
                    else "Latest saved opening order evidence predates the target day."
                ),
            }
        )
        return details

    if adoption_order is not None:
        details.update(
            {
                "kind": "ambiguous",
                "provenance": "adoption_without_saved_entry_order",
                "provenanceDetail": "Same-day no-order carry event had no saved opening order evidence.",
            }
        )
        return details

    details.update(
        {
            "kind": "entered_today",
            "provenance": "entry_index_only",
            "provenanceDetail": "No saved opening order evidence contradicted the same-day entry index.",
        }
    )
    return details


def snapshot_range_local(snapshots: list[BotSnapshot], tz: ZoneInfo) -> dict[str, str] | None:
    updated = [snap.updated_at_ms for snap in snapshots if snap.updated_at_ms > 0]
    if not updated:
        return None
    return {
        "firstUpdatedAtLocal": local_iso(min(updated), tz),
        "lastUpdatedAtLocal": local_iso(max(updated), tz),
    }


def build_report(date_str: str, tz_name: str, tenant_dir: Path) -> dict[str, Any]:
    tz = ZoneInfo(tz_name)
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_local = datetime.combine(target_date, datetime.min.time(), tz)
    end_local = start_local + timedelta(days=1)
    start_ms = int(start_local.timestamp() * 1000)
    end_ms = int(end_local.timestamp() * 1000)

    snapshots = []
    for snap_path in sorted(tenant_dir.glob("bot-state-*.json")):
        snapshot = load_snapshot(snap_path)
        if snapshot is not None:
            snapshots.append(snapshot)
    if not snapshots:
        raise FileNotFoundError(f"no bot-state snapshots found in {tenant_dir}")

    completed_trades: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    carried_open_positions: list[dict[str, Any]] = []
    order_events: list[dict[str, Any]] = []
    ambiguous_open_positions: list[dict[str, Any]] = []

    for snapshot in snapshots:
        prices = snapshot.status.get("prices") or []
        equity_curve = snapshot.status.get("equityCurve") or []
        open_times = infer_open_times(snapshot)
        latest_signal = snapshot.status.get("latestSignal") or {}
        latest_regimes = latest_signal.get("regimes") or {}

        for trade in snapshot.status.get("trades") or []:
            entry_idx = int(trade.get("entryIndex", -1))
            exit_idx = int(trade.get("exitIndex", -1))
            if exit_idx < 0 or exit_idx >= len(open_times):
                continue
            exit_ms = open_times[exit_idx]
            if exit_ms < start_ms or exit_ms >= end_ms:
                continue
            entry_ms = open_times[entry_idx] if 0 <= entry_idx < len(open_times) else None
            entry_price = prices[entry_idx] if 0 <= entry_idx < len(prices) else None
            exit_price = prices[exit_idx] if 0 <= exit_idx < len(prices) else None
            completed_trades.append(
                {
                    "symbol": snapshot.symbol,
                    "interval": snapshot.status.get("interval"),
                    "side": infer_trade_side(snapshot, entry_idx, exit_idx),
                    "entryIndex": entry_idx,
                    "exitIndex": exit_idx,
                    "entryTimeLocal": local_iso(entry_ms, tz) if entry_ms is not None else None,
                    "exitTimeLocal": local_iso(exit_ms, tz),
                    "entryPrice": entry_price,
                    "exitPrice": exit_price,
                    "returnPct": safe_pct(float_or_none(trade.get("return"))),
                    "holdingBars": trade.get("holdingPeriods"),
                    "exitReason": trade.get("exitReason"),
                    "entryHighVolProb": trade.get("entryHighVolProb"),
                    "entryRegime": classify_regime(prices, entry_idx),
                    "exitRegime": classify_regime(prices, exit_idx),
                    "latestRegimes": latest_regimes,
                    "latestVolatility": latest_signal.get("volatility"),
                }
            )

        open_trade = snapshot.status.get("openTrade") or None
        if isinstance(open_trade, dict):
            entry_idx = int(open_trade.get("entryIndex", -1))
            if 0 <= entry_idx < len(open_times):
                entry_ms = open_times[entry_idx]
                if start_ms <= entry_ms < end_ms:
                    entry_equity = float_or_none(open_trade.get("entryEquity"))
                    current_equity = float_or_none(equity_curve[-1]) if equity_curve else None
                    mtm_pct = None
                    if entry_equity and current_equity:
                        mtm_pct = (current_equity / entry_equity - 1.0) * 100.0
                    position_row = {
                        "symbol": snapshot.symbol,
                        "interval": snapshot.status.get("interval"),
                        "side": open_trade.get("side"),
                        "entryIndex": entry_idx,
                        "entryTimeLocal": local_iso(entry_ms, tz),
                        "entryPrice": open_trade.get("entryPrice"),
                        "currentPrice": prices[-1] if prices else None,
                        "holdingBars": open_trade.get("holdingPeriods"),
                        "markToMarketPct": mtm_pct,
                        "trail": open_trade.get("trail"),
                        "size": open_trade.get("size"),
                        "entryRegime": classify_regime(prices, entry_idx),
                        "latestRegimes": latest_regimes,
                        "latestVolatility": latest_signal.get("volatility"),
                    }
                    provenance = build_open_position_provenance(snapshot, open_trade, entry_idx, start_ms, end_ms, tz)
                    kind = provenance.pop("kind")
                    position_row.update(provenance)
                    if kind == "entered_today":
                        open_positions.append(position_row)
                    elif kind == "carried_in":
                        carried_open_positions.append(position_row)
                    else:
                        ambiguous_open_positions.append(position_row)

        for order_event in snapshot.status.get("orders") or []:
            at_ms = order_event.get("atMs")
            if not isinstance(at_ms, (int, float)):
                continue
            at_ms = int(at_ms)
            if at_ms < start_ms or at_ms >= end_ms:
                continue
            ack_only = is_ack_only_order(order_event)
            order = order_event.get("order") or {}
            order_events.append(
                {
                    "symbol": snapshot.symbol,
                    "atLocal": local_iso(at_ms, tz),
                    "openTimeLocal": local_iso(int(order_event.get("openTime")), tz)
                    if isinstance(order_event.get("openTime"), (int, float))
                    else None,
                    "index": order_event.get("index"),
                    "opSide": order_event.get("opSide"),
                    "price": order_event.get("price"),
                    "sent": order.get("sent"),
                    "status": order.get("status"),
                    "message": order.get("message"),
                    "executedQty": order.get("executedQty"),
                    "quantity": order.get("quantity"),
                    "ackOnly": ack_only,
                }
            )

    completed_trades.sort(key=lambda row: (row["exitTimeLocal"] or "", row["symbol"]))
    open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))
    carried_open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))
    order_events.sort(key=lambda row: (row["atLocal"] or "", row["symbol"]))
    ambiguous_open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))

    active_symbols = {row["symbol"] for row in completed_trades} | {row["symbol"] for row in open_positions}
    fill_evidence_gaps = [row for row in order_events if row["ackOnly"] and row["symbol"] in active_symbols]

    compound = 1.0
    for trade in completed_trades:
        trade_return = trade["returnPct"]
        if trade_return is not None:
            compound *= 1.0 + trade_return / 100.0

    return {
        "date": date_str,
        "timezone": tz_name,
        "tenantDir": str(tenant_dir),
        "windowLocal": {
            "start": start_local.isoformat(),
            "endExclusive": end_local.isoformat(),
        },
        "snapshotRangeLocal": snapshot_range_local(snapshots, tz),
        "summary": {
            "completedTrades": len(completed_trades),
            "completedCompoundPct": (compound - 1.0) * 100.0 if completed_trades else 0.0,
            "completedAveragePct": (
                sum(trade["returnPct"] for trade in completed_trades if trade["returnPct"] is not None) / len(completed_trades)
                if completed_trades
                else 0.0
            ),
            "openPositionsEnteredToday": len(open_positions),
            "openPositionsCarriedIn": len(carried_open_positions),
            "sameDayOrderEvents": len(order_events),
            "ackOnlyOrderEvents": sum(1 for row in order_events if row["ackOnly"]),
            "fillEvidenceGaps": len(fill_evidence_gaps),
            "ambiguousOpenPositionOrigins": len(ambiguous_open_positions),
        },
        "completedTrades": completed_trades,
        "openPositionsEnteredToday": open_positions,
        "openPositionsCarriedIn": carried_open_positions,
        "orderEvents": order_events,
        "anomalies": {
            "fillEvidenceGaps": fill_evidence_gaps,
            "missingCompletedTradeSide": [row["symbol"] for row in completed_trades if row["side"] == "unknown"],
            "missingOpenTimeReconstruction": [
                snap.symbol for snap in snapshots if not infer_open_times(snap) and (snap.status.get("trades") or snap.status.get("openTrade"))
            ],
            "ambiguousOpenPositionOrigins": ambiguous_open_positions,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report["summary"]
    lines.append(f"# Daily Bot Review: {report['date']} ({report['timezone']})")
    lines.append("")
    lines.append(f"- Tenant: `{report['tenantDir']}`")
    lines.append(f"- Window: `{report['windowLocal']['start']}` to `{report['windowLocal']['endExclusive']}`")
    snapshot_range = report.get("snapshotRangeLocal")
    if snapshot_range:
        lines.append(
            "- Snapshot updates: "
            f"`{snapshot_range['firstUpdatedAtLocal']}` to `{snapshot_range['lastUpdatedAtLocal']}`"
        )
    lines.append(
        "- Summary: "
        f"`completed={summary['completedTrades']}` "
        f"`compound={summary['completedCompoundPct']:.5f}%` "
        f"`open_entered_today={summary['openPositionsEnteredToday']}` "
        f"`open_carried_in={summary['openPositionsCarriedIn']}` "
        f"`order_events={summary['sameDayOrderEvents']}` "
        f"`ack_only={summary['ackOnlyOrderEvents']}` "
        f"`fill_gaps={summary['fillEvidenceGaps']}` "
        f"`ambiguous_open_origin={summary['ambiguousOpenPositionOrigins']}`"
    )
    lines.append("")

    lines.append("## Completed Trades")
    if report["completedTrades"]:
        for trade in report["completedTrades"]:
            entry_regime = trade["entryRegime"]["label"]
            exit_regime = trade["exitRegime"]["label"]
            lines.append(
                "- "
                f"`{trade['symbol']}` `{trade['interval']}` `{trade['side']}` "
                f"entry `{trade['entryTimeLocal']}` @ `{trade['entryPrice']}` "
                f"exit `{trade['exitTimeLocal']}` @ `{trade['exitPrice']}` "
                f"pnl `{trade['returnPct']:.5f}%` "
                f"hold `{trade['holdingBars']}` bars "
                f"exitReason `{trade['exitReason']}` "
                f"regime `{entry_regime}->{exit_regime}`"
            )
    else:
        lines.append("- No completed trades closed on the target local date.")
    lines.append("")

    lines.append("## Open Positions Entered Today")
    if report["openPositionsEnteredToday"]:
        for trade in report["openPositionsEnteredToday"]:
            lines.append(
                "- "
                f"`{trade['symbol']}` `{trade['interval']}` `{trade['side']}` "
                f"entry `{trade['entryTimeLocal']}` @ `{trade['entryPrice']}` "
                f"current `{trade['currentPrice']}` "
                f"mtm `{(trade['markToMarketPct'] or 0):.5f}%` "
                f"hold `{trade['holdingBars']}` bars "
                f"regime `{trade['entryRegime']['label']}` "
                f"provenance `{trade['provenance']}`"
            )
    else:
        lines.append("- No open positions were entered on the target local date.")
    lines.append("")

    lines.append("## Open Positions Carried In")
    if report["openPositionsCarriedIn"]:
        for trade in report["openPositionsCarriedIn"]:
            lines.append(
                "- "
                f"`{trade['symbol']}` `{trade['interval']}` `{trade['side']}` "
                f"adopted `{trade['entryTimeLocal']}` @ `{trade['entryPrice']}` "
                f"prior_order `{trade.get('supportingOrderAtLocal')}` "
                f"provenance `{trade['provenance']}`"
            )
    else:
        lines.append("- No open positions were classified as carried in from a prior day.")
    lines.append("")

    lines.append("## Same-Day Order Events")
    if report["orderEvents"]:
        for event in report["orderEvents"]:
            gap = " fill-gap" if event["ackOnly"] and event["symbol"] in {
                row["symbol"] for row in report["anomalies"]["fillEvidenceGaps"]
            } else ""
            lines.append(
                "- "
                f"`{event['symbol']}` `{event['atLocal']}` `{event['opSide']}` "
                f"status `{event['status']}` qty `{event['executedQty']}` "
                f"sent `{event['sent']}` "
                f"message `{event['message']}`{gap}"
            )
    else:
        lines.append("- No order events touched the target local date.")
    lines.append("")

    lines.append("## Anomalies")
    if report["anomalies"]["fillEvidenceGaps"]:
        for event in report["anomalies"]["fillEvidenceGaps"]:
            lines.append(
                "- "
                f"`{event['symbol']}` has ack-only order evidence at `{event['atLocal']}` "
                f"while the same-day replay also shows a completed/open trade. "
                "Saved artifacts need stronger fill provenance."
            )
    else:
        lines.append("- No ack-only fill-evidence gaps were detected for active same-day trades.")
    if report["anomalies"]["missingCompletedTradeSide"]:
        lines.append(
            "- "
            "Could not infer completed-trade side from the saved positions vector for: "
            + ", ".join(f"`{symbol}`" for symbol in report["anomalies"]["missingCompletedTradeSide"])
        )
    if report["anomalies"]["missingOpenTimeReconstruction"]:
        lines.append(
            "- "
            "Missing open-time reconstruction for snapshots with live trade state: "
            + ", ".join(f"`{symbol}`" for symbol in report["anomalies"]["missingOpenTimeReconstruction"])
        )
    if report["anomalies"]["ambiguousOpenPositionOrigins"]:
        for trade in report["anomalies"]["ambiguousOpenPositionOrigins"]:
            lines.append(
                "- "
                f"`{trade['symbol']}` open position at `{trade['entryTimeLocal']}` has "
                f"ambiguous provenance: `{trade.get('adoptionMessage')}`. "
                "Review should not treat it as a confirmed same-day entry."
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    tenant_dir = Path(args.tenant_dir) if args.tenant_dir else choose_tenant_dir(Path(args.bot_root))
    report = build_report(args.date, args.timezone, tenant_dir)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
