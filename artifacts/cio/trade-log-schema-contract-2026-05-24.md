# Trade-log schema contract

- Owner: trader-firm-cio
- Effective version: backtest `1.1`; live event `1.2`
- Last reconciled with production: 2026-09-07

## Scope

`--trade-log FILE` appends one JSON object per line. There is no outer array.
The file can contain two record kinds:

- backtest closed-trade records, which retain schema `1.1`; and
- live applied-execution event records, whose schema advances additively to
  `1.2` with a native pre-execution risk-state snapshot.

Readers must dispatch by `schemaVersion` or `schema_version` and `method`.
Version 1.2 does not change or remove any version 1.1 field. In particular, a
live row is an applied OPEN/CLOSE event marker; its zero quantity, P&L, and fee
compatibility fields are not a claim that a completed round trip earned zero.

Canonical implementation:

- `haskell/app/Main.hs` selects the record and supplies live decision state.
- `haskell/app/Trader/TradeLogRiskState.hs` owns the version 1.2 live encoder
  and non-finite handling.

## Version 1.1 common fields

The writer preserves these fields in both record kinds. Camel-case numeric
fields are the current typed interface; snake-case decimal fields are legacy
string compatibility aliases.

| Field | JSON type | Backtest meaning | Live event meaning |
|---|---|---|---|
| `timestamp` | string | last input-bar time for the emitting run, or empty when unavailable | processing time in UTC, second precision |
| `symbol` | string | explicit symbol or symbol inferred from the data filename | configured live symbol |
| `side` | string | `LONG` or `SHORT` | `OPEN` or `CLOSE` event type |
| `entryPrice` | number | price at trade entry index | event price |
| `exitPrice` | number | price at trade exit index | event price |
| `quantity` | number | absolute simulated exposure at entry | compatibility marker `0.0` |
| `pnl` | number | exit equity minus entry equity | compatibility marker `0.0` |
| `pnlPercent` | number | recorded net trade return | compatibility marker `0.0` |
| `fees` | number | recorded trade fee cost | compatibility marker `0.0` |
| `method` | string | selected backtest method | `live` |
| `volConfGate` | string | selected volatility-confidence preset | `live` |
| `exitReason` | string or null | canonical simulated exit reason | `OPEN` or `CLOSE` event type |
| `entry_price` | string | decimal compatibility alias | decimal compatibility alias |
| `exit_price` | string | decimal compatibility alias | decimal compatibility alias |
| `quantity_text` | string | decimal compatibility alias | `"0.0"` |
| `pnl_quote` | string | decimal compatibility alias | `"0.0"` |
| `pnl_pct` | string | decimal compatibility alias | `"0.0"` |
| `fee_quote` | string | decimal compatibility alias | `"0.0"` |
| `signal_method` | string | compatibility method alias | `live` |
| `vol_conf_gate` | string | compatibility preset alias | `live` |
| `regime_filter` | null | reserved | reserved |
| `slippage_estimate` | null | reserved | reserved |
| `latency_ms` | null | reserved | reserved |
| `trade_id` | string | generated UUID | generated UUID |
| `schemaVersion` | string | `1.1` | `1.2` |
| `schema_version` | string | `1.1` | `1.2` |

The generated UUID intentionally makes emitted files non-deterministic. Tests
bind field semantics and safe encoding rather than claiming byte-identical
logs.

## Version 1.2 live risk state

Live rows add `closeReason`/`close_reason` aliases and identical `riskState` and
`risk_state` objects. The close reason is null on OPEN and otherwise carries
the canonical reason used to create the closed trade; the legacy `exitReason`
event marker is unchanged. The snapshot is
captured at the `pre_execution_decision` boundary from the same values passed
to the canonical live halt decision. It therefore describes risk evidence
before the current order's execution cost, not post-fill accounting.

| Field | JSON type | Meaning |
|---|---|---|
| `phase` | string | always `pre_execution_decision` |
| `asOfMs` | integer | processing/availability time of the decision |
| `marketEventTimeMs` | integer | causal completed-bar event time used by the decision |
| `equity` | number or null | marked-to-market equity evaluated by risk |
| `peakEquity` | number or null | peak including the current evaluated equity |
| `dayKey` | integer | UTC epoch-day bucket |
| `dayStartEquity` | number or null | equity reference for daily loss |
| `weekKey` | integer | seven-day epoch bucket; not an ISO week number |
| `weekStartEquity` | number or null | equity reference for weekly loss |
| `drawdown` | number or null | non-negative drawdown supplied to the halt decision |
| `dailyLoss` | number or null | non-negative daily loss supplied to the halt decision |
| `weeklyLoss` | number or null | non-negative weekly loss supplied to the halt decision |
| `expectancy` | number or null | evaluated recent mean trade return, or null when not evaluated/unavailable |
| `expectancyLookback` | integer | configured non-negative trade lookback |
| `expectancyObservations` | integer | finite preceding trade returns used when evaluated |
| `expectancyRequired` | boolean | whether a minimum-expectancy risk gate is configured |
| `expectancyAvailable` | boolean | whether this snapshot contains finite expectancy evidence |
| `haltReason` | string or null | halt state at the decision boundary |
| `finite` | boolean | false if any required metric or supplied expectancy was non-finite |
| `valid` | boolean | false if a source value is outside its risk domain or required expectancy is unavailable |

`expectancy=null` is not zero and must never be interpreted as bullish,
bearish, or sufficient risk evidence. If a numeric input is NaN, infinite, or
outside its risk domain, the corresponding field is `null`. Non-finite input
sets `finite=false`; any domain error or unavailable required expectancy sets
`valid=false`. Consumers must abstain or fail closed unless both flags are true;
they must not impute an actionable value.

## Compatibility and limitations

- Version 1.1 readers may ignore the two additive risk-state objects after
  accepting version 1.2 through an explicit compatible-version policy.
- Backtest rows remain 1.1. The simulator's exact intrabar daily/weekly risk
  decision state is not present in `BacktestSummary`; manufacturing it later
  from close-only trade rows would be an approximation and is prohibited by
  this contract.
- Live event markers do not replace the richer bot status, order journal, or
  exchange evidence. They must not be used alone to infer fills, quantity,
  round-trip P&L, fees, or actual exit cause.
- No field in this contract authorizes orders, promotion, deployment, or live
  trading.

## Required invariants

1. Every live applied OPEN/CLOSE row is schema 1.2 and contains equal
   `riskState` and `risk_state` objects.
2. All version 1.1 keys and meanings remain present in version 1.2.
3. `closeReason` and `close_reason` are equal, null on OPEN, and preserve the
   canonical close cause when a CLOSE row is emitted.
4. Risk values come from the same pre-execution boundary as the canonical halt
   decision; event time and processing time remain distinct.
5. A non-finite risk value is serialized as `null` and makes `finite=false` and
   `valid=false`; an out-of-domain value is also null and invalid.
6. Missing expectancy remains explicitly unavailable and is never zero-filled;
   it makes the snapshot invalid when the expectancy gate is required.
7. Backtest rows retain schema 1.1 until exact state is carried natively by the
   simulator.

## Verification

`Trader.Test.TradeLogRiskState` checks exact finite values, non-finite
neutralization, version markers, both risk aliases, and preservation of every
legacy v1.1 key in the version 1.2 encoder. It also binds the additive close
reason aliases and the flat-to-open / position-to-flat transition classifier
that carries the canonical close cause. The repository-wide Haskell wrapper
compiles the production call site that supplies the canonical risk inputs.
