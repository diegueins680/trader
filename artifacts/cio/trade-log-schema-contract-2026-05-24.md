# Trade-Log Schema Contract — 2026-05-24

**Owner:** trader-firm-cio  
**Manager:** trader-firm-ceo  
**Reference:** `haskell/app/Trader/Trading.hs` (Trade, BacktestCostAttribution, ExitReason), `haskell/app/Trader/Metrics.hs` (BacktestMetrics), Research field semantics request, Risk validation invariants  
**Status:** ACTIVE — replaces the missing `artifacts/research/trade-log-schema-fields-2026-05-23.md`

---

## 1. Purpose

This contract defines the exact JSON serialization format for every trade emitted by the backtester and live engine. It is the single source of truth for:
- Risk validation (RISK-AC-001)
- Execution wire format
- Research reproducibility
- External integrations (journal, webhook, NDJSON archive)

---

## 2. Top-Level JSON Object (per trade)

Each trade is one JSON object. In NDJSON mode one object per line. No outer array.

| # | Field Name | Haskell Type | JSON Type | Nullability | NaN / Inf Handling | Example Value |
|---|-----------|--------------|-----------|-------------|-------------------|---------------|
| 1 | `entry_index` | `Int` | `number` (integer) | NOT NULL | N/A | `42` |
| 2 | `exit_index` | `Int` | `number` (integer) | NOT NULL | N/A | `55` |
| 3 | `entry_equity` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `10000.00` |
| 4 | `exit_equity` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `10019.20` |
| 5 | `return` | `Double` | `number` (decimal) | NOT NULL | `0.0` if NaN/Inf | `0.00192` |
| 6 | `holding_periods` | `Int` | `number` (integer) | NOT NULL | N/A | `13` |
| 7 | `entry_high_vol_prob` | `Maybe Double` | `number` / `null` | NULLABLE | `null` if NaN/Inf | `0.12` |
| 8 | `entry_source` | `TradeEntrySource` | `string` | NOT NULL | N/A | `"signal"` |
| 9 | `exit_reason` | `Maybe ExitReason` | `string` / `null` | NULLABLE | N/A | `"TRAILING_STOP"` |
| 10 | `entry_ip` | `Maybe Text` | `string` / `null` | NULLABLE | N/A | `"192.168.1.42"` |
| 11 | `exit_ip` | `Maybe Text` | `string` / `null` | NULLABLE | N/A | `null` |
| 12 | `fee_cost` | `Double` | `number` (decimal) | NOT NULL | `0.0` if NaN/Inf | `0.15` |
| 13 | `entry_price` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `89129.50` |
| 14 | `exit_price` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `89012.30` |
| 15 | `position_size` | `Double` | `number` (decimal) | NOT NULL | `0.0` if NaN/Inf | `1.0` |
| 16 | `symbol` | `String` | `string` | NOT NULL | N/A | `"BTCUSDT"` |
| 17 | `timestamp_ms` | `Int64` | `number` (integer) | NOT NULL | N/A | `1716547200000` |
| 18 | `method` | `Method` | `string` | NOT NULL | N/A | `"ta_trend"` |
| 19 | `vol_conf_gate` | `VolConfGatePreset` | `string` | NOT NULL | N/A | `"vol_conf_v1_high_vol_looser"` |
| 20 | `trailing_stop` | `Maybe Double` | `number` / `null` | NULLABLE | `null` if NaN/Inf | `0.005` |
| 21 | `cost_attribution` | `BacktestCostAttribution` | `object` | NOT NULL | see §4 | `{...}` |
| 22 | `sharpe` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `2.663` |
| 23 | `max_drawdown` | `Double` | `number` (decimal) | NOT NULL | `null` if NaN/Inf | `0.0726` |
| 24 | `trade_count` | `Int` | `number` (integer) | NOT NULL | N/A | `13` |

---

## 3. Enum Values

### 3.1 `entry_source`
| Value | Meaning |
|-------|---------|
| `"signal"` | Entry triggered by primary signal |
| `"adopted"` | Entry adopted from prior open position |
| `"post_direction_gates"` | Entry after direction gates passed |

### 3.2 `exit_reason`
| Value | Meaning |
|-------|---------|
| `"SIGNAL"` | Exit by opposing signal |
| `"STOP_LOSS"` | Hard stop-loss hit |
| `"TRAILING_STOP"` | Trailing stop triggered |
| `"TAKE_PROFIT"` | Take-profit level reached |
| `"MAX_DRAWDOWN"` | Max drawdown circuit breaker |
| `"MAX_DAILY_LOSS"` | Daily loss limit breached |
| `"MAX_WEEKLY_LOSS"` | Weekly loss limit breached |
| `"LIQUIDATION"` | Liquidation (live only) |
| `"EOD"` | End-of-day close (not a round-trip) |

**Note:** `null` means the trade is still open or the reason was not recorded.

---

## 4. `cost_attribution` Object

Embedded in every trade record for full cost transparency.

| Field Name | Haskell Type | JSON Type | Nullability | Example Value |
|-----------|--------------|-----------|-------------|---------------|
| `gross_equity_curve` | `[Double]` | `array` of `number` | NOT NULL | `[10000.0, 10005.2, ...]` |
| `net_equity_curve` | `[Double]` | `array` of `number` | NOT NULL | `[10000.0, 10003.1, ...]` |
| `realized_fee_cost` | `Double` | `number` | NOT NULL | `0.15` |
| `realized_slippage_cost` | `Double` | `number` | NOT NULL | `0.05` |
| `realized_spread_cost` | `Double` | `number` | NOT NULL | `0.02` |
| `realized_funding_cost` | `Double` | `number` | NOT NULL | `0.00` |
| `realized_total_cost` | `Double` | `number` | NOT NULL | `0.22` |
| `consistency_residual` | `Double` | `number` | NOT NULL | `0.0001` |

---

## 5. Complete Example NDJSON Record

```json
{
  "entry_index": 42,
  "exit_index": 55,
  "entry_equity": 10000.00,
  "exit_equity": 10019.20,
  "return": 0.00192,
  "holding_periods": 13,
  "entry_high_vol_prob": 0.12,
  "entry_source": "signal",
  "exit_reason": "TRAILING_STOP",
  "entry_ip": "192.168.1.42",
  "exit_ip": null,
  "fee_cost": 0.15,
  "entry_price": 89129.50,
  "exit_price": 89012.30,
  "position_size": 1.0,
  "symbol": "BTCUSDT",
  "timestamp_ms": 1716547200000,
  "method": "ta_trend",
  "vol_conf_gate": "vol_conf_v1_high_vol_looser",
  "trailing_stop": 0.005,
  "cost_attribution": {
    "gross_equity_curve": [10000.0, 10005.2, 10010.1, 10019.4],
    "net_equity_curve": [10000.0, 10003.1, 10007.8, 10019.2],
    "realized_fee_cost": 0.15,
    "realized_slippage_cost": 0.05,
    "realized_spread_cost": 0.02,
    "realized_funding_cost": 0.00,
    "realized_total_cost": 0.22,
    "consistency_residual": 0.0001
  },
  "sharpe": 2.663,
  "max_drawdown": 0.0726,
  "trade_count": 13
}
```

---

## 6. Numeric Precision Rules

- **Prices & equity:** 2 decimal places (matching quote asset precision).
- **Returns, Sharpe, drawdowns:** 6 decimal places.
- **Costs:** 2 decimal places.
- **Percentages stored as ratios:** e.g. `0.0726` for 7.26% (not `7.26`).

---

## 7. Nullability & Edge Cases

| Condition | Action |
|-----------|--------|
| NaN or ±Inf in any `Double` field | Serialize as `null` (if nullable) or `0.0` (if not nullable). Document in `consistency_residual`. |
| Missing `exit_reason` (open trade) | `null` |
| Missing `entry_ip` / `exit_ip` | `null` |
| Missing `trailing_stop` (not used) | `null` |
| Empty `gross_equity_curve` / `net_equity_curve` | `[]` (empty array) |

---

## 8. Validation Invariants (for Risk)

1. `entry_index < exit_index` (unless EOD close).
2. `exit_equity == entry_equity * (1 + return)` within `consistency_residual` tolerance.
3. `exit_reason` must be one of the enum values in §3.2 or `null`.
4. `cost_attribution.realized_total_cost` must equal the sum of the four realized cost components.
5. `timestamp_ms` must be a valid Unix epoch millisecond timestamp (≥ 0).

---

## 9. Versioning

- **Contract version:** `v1.0`
- **Effective date:** 2026-05-24
- **Next review:** 2026-05-31 or upon schema change
- **Change process:** Research proposes field changes → CIO approves → Execution updates serialization → Risk updates invariants
