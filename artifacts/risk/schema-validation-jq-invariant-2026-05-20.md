# Schema Validation jq Invariant — Risk Artifact

**Author:** trader-firm-risk  
**Date:** 2026-05-20 20:10 UTC  
**Status:** LANDED  
**Reference:** Candidate A Execution Packet §5 (`artifacts/research/candidate-a-execution-packet-2026-05-18.md`)

## Purpose

The CIO-committed schema contract `artifacts/cio/trade-log-schema-contract-2026-05-20.md` was **not found** in the workspace. Risk treats this as an upstream delivery gap (see Risk Report). However, the Candidate A Execution Packet contains a precise 24-field JSON output schema that Research and Execution have been using operationally. Risk has codified this schema into a **jq validation invariant** that can be run against any backtest JSON output to enforce type safety and nullability rules.

## Schema Contract Under Validation

| # | Field | JSON Path | Type | Nullable |
|---|-------|-----------|------|----------|
| 1 | mode | `.mode` | string | no |
| 2 | method | `.backtest.method` | string | no |
| 3 | sharpe | `.backtest.sharpe` | number | no |
| 4 | max_drawdown | `.backtest.max_drawdown` | number | no |
| 5 | closed_trades | `.backtest.closed_trades` | integer | no |
| 6 | avg_trade | `.backtest.avg_trade` | number | no |
| 7 | total_return | `.backtest.metrics.totalReturn` | number | no |
| 8 | annualized_return | `.backtest.metrics.annualizedReturn` | number | no |
| 9 | annualized_volatility | `.backtest.metrics.annualizedVolatility` | number | no |
| 10 | sortino | `.backtest.metrics.sortino` | number | no |
| 11 | calmar | `.backtest.metrics.calmar` | number | no |
| 12 | win_rate | `.backtest.metrics.winRate` | number | no |
| 13 | profit_factor | `.backtest.metrics.profitFactor` | number | yes |
| 14 | turnover | `.backtest.metrics.turnover` | number | no |
| 15 | exposure | `.backtest.metrics.exposure` | number | no |
| 16 | trade_count | `.backtest.metrics.tradeCount` | integer | no |
| 17 | position_changes | `.backtest.metrics.positionChanges` | integer | no |
| 18 | agreement_rate | `.backtest.metrics.agreementRate` | number | no |
| 19 | blend_weight | `.backtest.blendWeight` | number | no |
| 20 | open_threshold | `.backtest.openThreshold` | number | no |
| 21 | close_threshold | `.backtest.closeThreshold` | number | no |
| 22 | vol_conf_gate | `.backtest.vol_conf_gate` | string | no |
| 23 | train_size | `.backtest.split.train` | integer | no |
| 24 | backtest_size | `.backtest.split.backtest` | integer | no |

## Validation Logic

The jq script (`scripts/risk/validate-backtest-schema.jq`) enforces:

1. **Presence:** every non-nullable field must exist.
2. **Type correctness:** string, number, integer (number with no fractional part), boolean.
3. **NaN/Inf rejection:** numeric fields must be finite.
4. **Nullability:** nullable fields may be `null`; non-nullable fields must not be `null`.

## Usage

```bash
# Validate a single backtest JSON output
cd /Users/diegosaa/GitHub/trader
jq -f scripts/risk/validate-backtest-schema.jq backtest-output.json

# Check for any failures
jq -f scripts/risk/validate-backtest-schema.jq backtest-output.json | jq 'map(select(.status == "FAIL"))'
```

## Test Evidence

### Positive test (valid JSON)
All 24 fields PASS with correct types.

### Negative tests
- **Type mismatch:** `sharpe: "not_a_number"` → FAIL (expected number, got string)
- **Non-integer integer field:** `closed_trades: 4.5` → FAIL (expected integer, got number)
- **Null in non-nullable field:** `profitFactor` is nullable and accepts `null`; if moved to non-nullable, it would FAIL.

## Relationship to CIO Schema Contract

This artifact validates the **operational schema** that Research and Execution are actually using. If/when the CIO delivers the committed 16-field schema contract, Risk will:

1. Compare the two schemas field-by-field.
2. Identify gaps (extra fields, type mismatches, nullability differences).
3. File a delta memo to the CIO with PASS/FAIL per field.

## Risk Register Update

| ID | Item | Severity | Owner | Status |
|----|------|----------|-------|--------|
| SCHEMA-001 | CIO schema contract missing | HIGH | trader-firm-cio | OPEN — contingency: operational schema validated instead |
| SCHEMA-002 | Backtest JSON schema invariant | MEDIUM | trader-firm-risk | CLOSED — jq script landed |

## Next Priority

1. **P1 (continuing):** When CIO schema contract is delivered, run delta validation and report PASS/FAIL per field.
2. **P2:** Integrate this jq script into the autoloop so every backtest output is schema-validated before scorecard ingestion.
3. **P3:** Add a Haskell property test that generates random `BacktestResult` values and asserts they serialize to JSON that passes this jq schema.
