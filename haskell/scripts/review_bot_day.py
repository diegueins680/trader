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
import re
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

AUTH_FAILURE_CODES = {
    -2015,
    -2014,
    -1022,
    -1021,
    401,
    403,
}

AUTH_FAILURE_PHRASES = (
    "invalid api-key",
    "invalid api key",
    "permissions for action",
    "signature for this request",
    "signature is not valid",
    "timestamp for this request is outside of the recvwindow",
    "recvwindow",
    "api-key format invalid",
)

HTTP_STATUS_RE = re.compile(r"http(?:/\d(?:\.\d)?)?\s+(\d{3})", re.IGNORECASE)
BINANCE_CODE_RE = re.compile(r"binance code\s+(-?\d+)(?:\s*[:/]\s*(.*))?", re.IGNORECASE)

DIRECTIONALITY_CHOP_EFFICIENCY_MAX = 0.25
DIRECTIONALITY_MR_EFFICIENCY_MAX = 0.40
DIRECTIONALITY_REGIME_HYSTERESIS = 0.05

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
        "--end-local",
        help=(
            "Optional local cutoff inside the target day (ISO-8601 local time, "
            "for example 2026-04-03T23:37). When omitted, the review runs through local midnight."
        ),
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


def parse_local_dt(value: str, tz: ZoneInfo) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid --end-local value {value!r}: expected ISO-8601 local datetime") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


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


def normalized_entry_source(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if raw in {"signal", "adopted"}:
        return raw
    return None


def normalized_regimes(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    trend = float_or_none(value.get("trend"))
    mr = float_or_none(value.get("mr"))
    high_vol = float_or_none(value.get("highVol"))
    if trend is None and mr is None and high_vol is None:
        return None
    return {
        "trend": trend if trend is not None else 0.0,
        "mr": mr if mr is not None else 0.0,
        "highVol": high_vol if high_vol is not None else 0.0,
    }


def build_directionality_snapshot(
    prices: list[float],
    idx: int,
    regimes: dict[str, Any] | None,
    regime_hysteresis: float = DIRECTIONALITY_REGIME_HYSTERESIS,
) -> dict[str, Any] | None:
    regime = classify_regime(prices, idx)
    if regime["label"] == "insufficient":
        return None
    normalized = normalized_regimes(regimes)
    regime_leader = None
    regime_gap = None
    mr_dominant = False
    if normalized is not None:
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            regime_leader = ranked[0][0]
        if len(ranked) >= 2:
            regime_gap = ranked[0][1] - ranked[1][1]
        mr_dominant = regime_leader == "mr" and regime_gap is not None and regime_gap >= max(0.0, regime_hysteresis)

    efficiency = regime["efficiency"]
    reason = None
    if efficiency <= DIRECTIONALITY_CHOP_EFFICIENCY_MAX:
        reason = "NON_DIRECTIONAL_CHOP"
    elif efficiency <= DIRECTIONALITY_MR_EFFICIENCY_MAX and mr_dominant:
        reason = "NON_DIRECTIONAL_MR"

    return {
        "lookbackBars": regime["lookbackBars"],
        "netReturnPct": regime["netReturnPct"],
        "realizedVolPct": regime["realizedVolPct"],
        "efficiency": efficiency,
        "zScore": regime["zScore"],
        "label": regime["label"],
        "trendProb": normalized["trend"] if normalized is not None else None,
        "mrProb": normalized["mr"] if normalized is not None else None,
        "highVolProb": normalized["highVol"] if normalized is not None else None,
        "regimeLeader": regime_leader,
        "regimeGap": regime_gap,
        "nonDirectional": reason is not None,
        "reason": reason,
    }


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


def position_direction(value: Any) -> int:
    numeric = float_or_none(value)
    if numeric is None:
        return 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


def position_side(value: Any) -> str | None:
    direction = position_direction(value)
    if direction > 0:
        return "long"
    if direction < 0:
        return "short"
    return None


def normalized_order_side(value: Any) -> str | None:
    raw = str(value or "").strip().upper()
    if raw in {"BUY", "SELL"}:
        return raw
    return None


def open_side_to_order_side(side: Any) -> str | None:
    side_text = str(side or "").strip().lower()
    if side_text == "long":
        return "BUY"
    if side_text == "short":
        return "SELL"
    return None


def order_flow_role_from_side(order_side: str | None, reference_side: str | None) -> str | None:
    expected_open_side = open_side_to_order_side(reference_side)
    if order_side is None or expected_open_side is None:
        return None
    if order_side == expected_open_side:
        return "entry_or_add"
    return "exit_or_flatten"


def extract_http_status_code(message: str) -> int | None:
    match = HTTP_STATUS_RE.search(message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_labeled_binance_error(message: str) -> tuple[int | None, str | None]:
    match = BINANCE_CODE_RE.search(message)
    if match is None:
        return None, None
    try:
        code = int(match.group(1))
    except ValueError:
        code = None
    summary = (match.group(2) or "").strip() or None
    return code, summary


def classify_binance_auth_failure(message: Any) -> dict[str, Any] | None:
    text = str(message or "").strip()
    if not text:
        return None
    http_code = extract_http_status_code(text)
    code, summary = extract_labeled_binance_error(text)
    summary_text = summary or text
    lower_summary = summary_text.lower()
    if code not in AUTH_FAILURE_CODES and http_code not in AUTH_FAILURE_CODES and not any(
        phrase in lower_summary for phrase in AUTH_FAILURE_PHRASES
    ):
        return None
    return {
        "authFailure": True,
        "authFailureSummary": summary_text,
        "authFailureCode": code,
        "authFailureHttpCode": http_code,
    }


def classify_order_event_flow_role(
    snapshot: BotSnapshot,
    order_event: dict[str, Any],
    order_idx: int | None,
) -> dict[str, Any]:
    order_side = normalized_order_side(order_event.get("opSide") or (order_event.get("order") or {}).get("side"))
    if order_side is None:
        return {
            "flowRole": "unknown",
            "flowRoleEvidence": "missing_order_side",
            "contextSide": None,
        }

    message = str((order_event.get("order") or {}).get("message") or "").strip().lower()
    message_side_contexts = [
        ("message_already_flat", None, "exit_or_flatten"),
        ("message_already_long", "long", None),
        ("message_already_short", "short", None),
    ]
    for evidence, side, fixed_role in message_side_contexts:
        marker = evidence.replace("message_", "").replace("_", " ")
        if marker in message:
            role = fixed_role if fixed_role is not None else order_flow_role_from_side(order_side, side)
            if role is not None:
                return {
                    "flowRole": role,
                    "flowRoleEvidence": evidence,
                    "contextSide": side,
                }

    for trade in snapshot.status.get("trades") or []:
        entry_idx = int(trade.get("entryIndex", -1))
        exit_idx = int(trade.get("exitIndex", -1))
        if order_idx not in {entry_idx, exit_idx}:
            continue
        trade_side = infer_trade_side(snapshot, entry_idx, exit_idx)
        role = order_flow_role_from_side(order_side, trade_side)
        if role is not None:
            return {
                "flowRole": role,
                "flowRoleEvidence": "completed_trade_entry_side" if order_idx == entry_idx else "completed_trade_exit_side",
                "contextSide": trade_side,
            }

    open_trade = snapshot.status.get("openTrade")
    if isinstance(open_trade, dict):
        open_trade_side = str(open_trade.get("side") or "").strip().lower()
        role = order_flow_role_from_side(order_side, open_trade_side)
        if role is not None:
            return {
                "flowRole": role,
                "flowRoleEvidence": "open_trade_side",
                "contextSide": open_trade_side,
            }

    positions = snapshot.status.get("positions") or []
    if order_idx is not None:
        position_side_contexts = [
            ("position_at_index", position_side(positions[order_idx]) if 0 <= order_idx < len(positions) else None),
            ("position_before_index", position_side(positions[order_idx - 1]) if 0 < order_idx <= len(positions) else None),
            (
                "position_after_index",
                position_side(positions[order_idx + 1]) if 0 <= order_idx < len(positions) - 1 else None,
            ),
        ]
        for evidence, side in position_side_contexts:
            role = order_flow_role_from_side(order_side, side)
            if role is not None:
                return {
                    "flowRole": role,
                    "flowRoleEvidence": evidence,
                    "contextSide": side,
                }

    return {
        "flowRole": "entry_or_add",
        "flowRoleEvidence": "default_without_close_context",
        "contextSide": None,
    }


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
    entry_ms: int | None,
    start_ms: int,
    end_ms: int,
    tz: ZoneInfo,
    entry_source: str | None,
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

    if entry_source == "adopted":
        details.update(
            {
                "kind": "carried_in",
                "provenance": "startup_adopted_position",
                "provenanceDetail": "Bot snapshot marks the position as adopted from pre-start exchange state.",
            }
        )
        return details

    if entry_ms is not None and entry_ms < start_ms:
        details.update(
            {
                "kind": "carried_in",
                "provenance": "entry_index_before_window",
                "provenanceDetail": "Position was already open before the target review window began.",
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


def last_open_index_before(open_times: list[int], end_ms: int) -> int | None:
    for idx in range(len(open_times) - 1, -1, -1):
        if open_times[idx] < end_ms:
            return idx
    return None


def open_position_entry_index(positions: list[Any], cutoff_idx: int) -> int | None:
    side = position_direction(positions[cutoff_idx])
    if side == 0:
        return None
    entry_idx = cutoff_idx
    while entry_idx > 0 and position_direction(positions[entry_idx - 1]) == side:
        entry_idx -= 1
    return entry_idx


def build_open_position_at_window_end(
    snapshot: BotSnapshot,
    cutoff_idx: int,
    start_ms: int,
    end_ms: int,
    tz: ZoneInfo,
) -> tuple[str, dict[str, Any]] | None:
    positions = snapshot.status.get("positions") or []
    prices = snapshot.status.get("prices") or []
    equity_curve = snapshot.status.get("equityCurve") or []
    open_times = infer_open_times(snapshot)
    if not open_times or cutoff_idx >= len(open_times) or cutoff_idx >= len(positions):
        return None
    side_direction = position_direction(positions[cutoff_idx])
    if side_direction == 0:
        return None
    entry_idx = open_position_entry_index(positions, cutoff_idx)
    if entry_idx is None or entry_idx >= len(open_times):
        return None

    side = "long" if side_direction > 0 else "short"
    entry_ms = open_times[entry_idx]
    matching_open_trade = snapshot.status.get("openTrade")
    if not isinstance(matching_open_trade, dict):
        matching_open_trade = None
    if matching_open_trade is not None:
        matching_entry = int(matching_open_trade.get("entryIndex", -1))
        matching_side = str(matching_open_trade.get("side") or "").strip().lower()
        if matching_entry != entry_idx or matching_side != side:
            matching_open_trade = None
    entry_source = normalized_entry_source(matching_open_trade.get("entrySource") if matching_open_trade is not None else None)

    entry_equity = None
    if matching_open_trade is not None:
        entry_equity = float_or_none(matching_open_trade.get("entryEquity"))
    if entry_equity is None and entry_idx < len(equity_curve):
        entry_equity = float_or_none(equity_curve[entry_idx])

    current_equity = float_or_none(equity_curve[cutoff_idx]) if cutoff_idx < len(equity_curve) else None
    mtm_pct = None
    if entry_equity and current_equity:
        mtm_pct = (current_equity / entry_equity - 1.0) * 100.0

    position_row = {
        "symbol": snapshot.symbol,
        "interval": snapshot.status.get("interval"),
        "side": side,
        "entryIndex": entry_idx,
        "entryTimeLocal": local_iso(entry_ms, tz),
        "entryPrice": matching_open_trade.get("entryPrice") if matching_open_trade is not None else prices[entry_idx],
        "currentPrice": prices[cutoff_idx] if cutoff_idx < len(prices) else None,
        "holdingBars": cutoff_idx - entry_idx,
        "markToMarketPct": mtm_pct,
        "trail": matching_open_trade.get("trail") if matching_open_trade is not None else None,
        "size": matching_open_trade.get("size") if matching_open_trade is not None else None,
        "entrySource": entry_source,
        "entryRegime": classify_regime(prices, entry_idx),
        "latestRegimes": snapshot.status.get("latestSignal", {}).get("regimes") if snapshot.updated_at_ms <= end_ms else None,
        "latestVolatility": snapshot.status.get("latestSignal", {}).get("volatility") if snapshot.updated_at_ms <= end_ms else None,
    }
    provenance = build_open_position_provenance(
        snapshot,
        {"side": side},
        entry_idx,
        entry_ms,
        start_ms,
        end_ms,
        tz,
        entry_source,
    )
    kind = provenance.pop("kind")
    position_row.update(provenance)
    return kind, position_row


def build_report(date_str: str, tz_name: str, tenant_dir: Path, end_local_text: str | None = None) -> dict[str, Any]:
    tz = ZoneInfo(tz_name)
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_local = datetime.combine(target_date, datetime.min.time(), tz)
    day_end_local = start_local + timedelta(days=1)
    if end_local_text:
        requested_end_local = parse_local_dt(end_local_text, tz)
        if requested_end_local < start_local or requested_end_local > day_end_local:
            raise ValueError(
                f"--end-local must stay within {start_local.isoformat()} and {day_end_local.isoformat()}"
            )
        end_local = requested_end_local
    else:
        end_local = day_end_local
    start_ms = int(start_local.timestamp() * 1000)
    end_ms = int(end_local.timestamp() * 1000)

    snapshots = []
    for snap_path in sorted(tenant_dir.glob("bot-state-*.json")):
        snapshot = load_snapshot(snap_path)
        if snapshot is not None:
            snapshots.append(snapshot)
    if not snapshots:
        raise FileNotFoundError(f"no bot-state snapshots found in {tenant_dir}")

    snapshots_updated_after_window = [
        {"symbol": snap.symbol, "updatedAtLocal": local_iso(snap.updated_at_ms, tz)}
        for snap in snapshots
        if snap.updated_at_ms > end_ms
    ]
    completed_trades: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    carried_open_positions: list[dict[str, Any]] = []
    order_events: list[dict[str, Any]] = []
    ambiguous_open_positions: list[dict[str, Any]] = []
    adopted_trade_closures: list[dict[str, Any]] = []

    for snapshot in snapshots:
        prices = snapshot.status.get("prices") or []
        equity_curve = snapshot.status.get("equityCurve") or []
        open_times = infer_open_times(snapshot)
        latest_signal = snapshot.status.get("latestSignal") or {}
        latest_regimes = latest_signal.get("regimes") or {}
        cutoff_idx = last_open_index_before(open_times, end_ms) if open_times else None

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
            entry_source = normalized_entry_source(trade.get("entrySource"))
            trade_row = {
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
                "entrySource": entry_source,
                "provenance": "startup_adopted_position" if entry_source == "adopted" else "signal_entry",
                "entryRegime": classify_regime(prices, entry_idx),
                "exitRegime": classify_regime(prices, exit_idx),
                "latestRegimes": latest_regimes if snapshot.updated_at_ms <= end_ms else None,
                "latestVolatility": latest_signal.get("volatility") if snapshot.updated_at_ms <= end_ms else None,
            }
            completed_trades.append(trade_row)
            if entry_source == "adopted":
                adopted_trade_closures.append(trade_row)

        if cutoff_idx is not None:
            position_at_window_end = build_open_position_at_window_end(snapshot, cutoff_idx, start_ms, end_ms, tz)
            if position_at_window_end is not None:
                kind, position_row = position_at_window_end
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
            order_idx_raw = order_event.get("index")
            order_idx = int(order_idx_raw) if isinstance(order_idx_raw, (int, float)) else None
            directionality = None
            if isinstance(order_event.get("directionality"), dict):
                directionality = order_event.get("directionality")
            elif order_idx is not None:
                directionality = build_directionality_snapshot(prices, order_idx, latest_regimes)
            auth_failure = classify_binance_auth_failure(order.get("message")) if order.get("sent") is not True else None
            flow_role = classify_order_event_flow_role(snapshot, order_event, order_idx)
            order_row = {
                "symbol": snapshot.symbol,
                "atLocal": local_iso(at_ms, tz),
                "openTimeLocal": local_iso(int(order_event.get("openTime")), tz)
                if isinstance(order_event.get("openTime"), (int, float))
                else None,
                "index": order_idx,
                "opSide": order_event.get("opSide"),
                "price": order_event.get("price"),
                "sent": order.get("sent"),
                "status": order.get("status"),
                "message": order.get("message"),
                "executedQty": order.get("executedQty"),
                "quantity": order.get("quantity"),
                "ackOnly": ack_only,
                "authFailure": auth_failure is not None,
                "authFailureSummary": auth_failure["authFailureSummary"] if auth_failure else None,
                "authFailureCode": auth_failure["authFailureCode"] if auth_failure else None,
                "authFailureHttpCode": auth_failure["authFailureHttpCode"] if auth_failure else None,
                "directionality": directionality,
                "nonDirectionalVeto": bool(directionality and directionality.get("nonDirectional")),
                "nonDirectionalReason": directionality.get("reason") if directionality else None,
                "flowRole": flow_role["flowRole"],
                "flowRoleEvidence": flow_role["flowRoleEvidence"],
                "contextSide": flow_role["contextSide"],
            }
            order_events.append(order_row)

    completed_trades.sort(key=lambda row: (row["exitTimeLocal"] or "", row["symbol"]))
    open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))
    carried_open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))
    order_events.sort(key=lambda row: (row["atLocal"] or "", row["symbol"]))
    ambiguous_open_positions.sort(key=lambda row: (row["entryTimeLocal"] or "", row["symbol"]))
    adopted_trade_closures.sort(key=lambda row: (row["exitTimeLocal"] or "", row["symbol"]))

    active_symbols = (
        {row["symbol"] for row in completed_trades}
        | {row["symbol"] for row in open_positions}
        | {row["symbol"] for row in carried_open_positions}
    )
    fill_evidence_gaps = [row for row in order_events if row["ackOnly"] and row["symbol"] in active_symbols]
    non_directional_order_attempts = [
        row for row in order_events if row["nonDirectionalVeto"] and row["flowRole"] == "entry_or_add"
    ]
    non_directional_exit_or_flatten_events = [
        row for row in order_events if row["nonDirectionalVeto"] and row["flowRole"] == "exit_or_flatten"
    ]
    non_directional_unknown_role_events = [
        row for row in order_events if row["nonDirectionalVeto"] and row["flowRole"] == "unknown"
    ]
    auth_failure_order_events = [row for row in order_events if row["authFailure"]]
    auth_failure_entry_or_add_events = [row for row in auth_failure_order_events if row["flowRole"] == "entry_or_add"]
    auth_failure_exit_or_flatten_events = [
        row for row in auth_failure_order_events if row["flowRole"] == "exit_or_flatten"
    ]
    auth_failure_unknown_role_events = [row for row in auth_failure_order_events if row["flowRole"] == "unknown"]

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
            "authFailureOrderEvents": len(auth_failure_order_events),
            "authFailureEntryOrAddEvents": len(auth_failure_entry_or_add_events),
            "authFailureExitOrFlattenEvents": len(auth_failure_exit_or_flatten_events),
            "authFailureUnknownRoleEvents": len(auth_failure_unknown_role_events),
            "nonDirectionalOrderAttempts": len(non_directional_order_attempts),
            "nonDirectionalExitOrFlattenEvents": len(non_directional_exit_or_flatten_events),
            "nonDirectionalUnknownRoleEvents": len(non_directional_unknown_role_events),
            "fillEvidenceGaps": len(fill_evidence_gaps),
            "ambiguousOpenPositionOrigins": len(ambiguous_open_positions),
            "snapshotsUpdatedAfterWindow": len(snapshots_updated_after_window),
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
            "adoptedTradeClosures": adopted_trade_closures,
            "authFailureEntryOrAddEvents": auth_failure_entry_or_add_events,
            "authFailureExitOrFlattenEvents": auth_failure_exit_or_flatten_events,
            "authFailureUnknownRoleEvents": auth_failure_unknown_role_events,
            "nonDirectionalOrderAttempts": non_directional_order_attempts,
            "nonDirectionalExitOrFlattenEvents": non_directional_exit_or_flatten_events,
            "nonDirectionalUnknownRoleEvents": non_directional_unknown_role_events,
            "ambiguousOpenPositionOrigins": ambiguous_open_positions,
            "snapshotsUpdatedAfterWindow": snapshots_updated_after_window,
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
        f"`auth_failures={summary['authFailureOrderEvents']}` "
        f"`auth_entry_or_add={summary['authFailureEntryOrAddEvents']}` "
        f"`auth_exit_or_flatten={summary['authFailureExitOrFlattenEvents']}` "
        f"`auth_unknown={summary['authFailureUnknownRoleEvents']}` "
        f"`non_directional_attempts={summary['nonDirectionalOrderAttempts']}` "
        f"`non_directional_exit_or_flatten={summary['nonDirectionalExitOrFlattenEvents']}` "
        f"`non_directional_unknown={summary['nonDirectionalUnknownRoleEvents']}` "
        f"`fill_gaps={summary['fillEvidenceGaps']}` "
        f"`ambiguous_open_origin={summary['ambiguousOpenPositionOrigins']}` "
        f"`snapshots_after_window={summary['snapshotsUpdatedAfterWindow']}`"
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
                f"regime `{entry_regime}->{exit_regime}` "
                f"source `{trade.get('entrySource') or 'unknown'}`"
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
            directionality = event.get("directionality") or {}
            directionality_suffix = ""
            if directionality:
                efficiency = float_or_none(directionality.get("efficiency"))
                directionality_suffix = (
                    " "
                    f"regime `{directionality.get('label')}` "
                    f"eff `{efficiency:.5f}` "
                    f"veto `{directionality.get('reason') or 'allow'}`"
                ) if efficiency is not None else f" regime `{directionality.get('label')}` veto `{directionality.get('reason') or 'allow'}`"
            auth_suffix = ""
            if event["authFailure"]:
                auth_suffix = f" auth `{event['authFailureSummary']}`"
            lines.append(
                "- "
                f"`{event['symbol']}` `{event['atLocal']}` `{event['opSide']}` "
                f"status `{event['status']}` qty `{event['executedQty']}` "
                f"sent `{event['sent']}` "
                f"message `{event['message']}` "
                f"flow `{event['flowRole']}` via `{event['flowRoleEvidence']}`{gap}{auth_suffix}{directionality_suffix}"
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
    if report["anomalies"]["adoptedTradeClosures"]:
        for trade in report["anomalies"]["adoptedTradeClosures"]:
            lines.append(
                "- "
                f"`{trade['symbol']}` closed on `{trade['exitTimeLocal']}` but its saved entry provenance is `adopted`. "
                "Treat it as a carry-in position that exited today, not a confirmed same-day fresh entry."
            )
    if report["anomalies"]["authFailureEntryOrAddEvents"]:
        for event in report["anomalies"]["authFailureEntryOrAddEvents"]:
            lines.append(
                "- "
                f"`{event['symbol']}` failed `{event['opSide']}` entry/add flow at `{event['atLocal']}` because Binance auth/IP/signature validation failed: "
                f"`{event['authFailureSummary']}`."
            )
    if report["anomalies"]["authFailureExitOrFlattenEvents"]:
        for event in report["anomalies"]["authFailureExitOrFlattenEvents"]:
            lines.append(
                "- "
                f"`{event['symbol']}` failed `{event['opSide']}` exit/flatten flow at `{event['atLocal']}` because Binance auth/IP/signature validation failed: "
                f"`{event['authFailureSummary']}`."
            )
    if report["anomalies"]["authFailureUnknownRoleEvents"]:
        for event in report["anomalies"]["authFailureUnknownRoleEvents"]:
            lines.append(
                "- "
                f"`{event['symbol']}` failed at `{event['atLocal']}` with Binance auth/IP/signature validation, but its flow role stayed "
                f"`{event['flowRole']}` via `{event['flowRoleEvidence']}`."
            )
    if report["anomalies"]["nonDirectionalOrderAttempts"]:
        for event in report["anomalies"]["nonDirectionalOrderAttempts"]:
            directionality = event.get("directionality") or {}
            efficiency = float_or_none(directionality.get("efficiency"))
            if efficiency is not None:
                lines.append(
                    "- "
                    f"`{event['symbol']}` attempted `{event['opSide']}` at `{event['atLocal']}` in "
                    f"`{directionality.get('label')}` with efficiency `{efficiency:.5f}`. "
                    f"The new low-directionality gate would veto it as `{event['nonDirectionalReason']}` "
                    f"(flow `{event['flowRole']}` via `{event['flowRoleEvidence']}`)."
                )
            else:
                lines.append(
                    "- "
                    f"`{event['symbol']}` attempted `{event['opSide']}` at `{event['atLocal']}` in "
                    f"`{directionality.get('label')}`. "
                    f"The new low-directionality gate would veto it as `{event['nonDirectionalReason']}` "
                    f"(flow `{event['flowRole']}` via `{event['flowRoleEvidence']}`)."
                )
    if report["anomalies"]["nonDirectionalExitOrFlattenEvents"]:
        for event in report["anomalies"]["nonDirectionalExitOrFlattenEvents"]:
            directionality = event.get("directionality") or {}
            efficiency = float_or_none(directionality.get("efficiency"))
            if efficiency is not None:
                lines.append(
                    "- "
                    f"`{event['symbol']}` hit a non-directional veto at `{event['atLocal']}` with "
                    f"flow `{event['flowRole']}` via `{event['flowRoleEvidence']}` in "
                    f"`{directionality.get('label')}` (efficiency `{efficiency:.5f}`), but this was an "
                    "exit/flatten event and should not count as a fresh entry attempt."
                )
            else:
                lines.append(
                    "- "
                    f"`{event['symbol']}` hit a non-directional veto at `{event['atLocal']}` with "
                    f"flow `{event['flowRole']}` via `{event['flowRoleEvidence']}`, but this was an "
                    "exit/flatten event and should not count as a fresh entry attempt."
                )
    if report["anomalies"]["nonDirectionalUnknownRoleEvents"]:
        for event in report["anomalies"]["nonDirectionalUnknownRoleEvents"]:
            lines.append(
                "- "
                f"`{event['symbol']}` hit a non-directional veto at `{event['atLocal']}` but its order-flow role is "
                f"`{event['flowRole']}` via `{event['flowRoleEvidence']}`. Review this event before using it as an entry-attempt signal."
            )
    if report["anomalies"]["snapshotsUpdatedAfterWindow"]:
        lines.append(
            "- "
            f"`{len(report['anomalies']['snapshotsUpdatedAfterWindow'])}` snapshot(s) were updated after the review window end. "
            "Open-position membership is reconstructed from the saved positions vector at the cutoff, and post-window latest-signal fields are omitted to reduce leakage."
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    tenant_dir = Path(args.tenant_dir) if args.tenant_dir else choose_tenant_dir(Path(args.bot_root))
    report = build_report(args.date, args.timezone, tenant_dir, args.end_local)
    if args.format == "json":
        json.dump(report, sys.stdout, indent=2, sort_keys=False)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
