# Interim Backtest Pass/Fail Contract Spec v1

**Author:** trader-firm-research  
**Date:** 2026-05-18 12:30 UTC  
**Status:** RATIFIED by CIO 2026-05-15 15:46 UTC  
**Replaces:** 2026-03-18 contract spec (not locatable in workspace)  
**Next review:** When B5 clears and post-B5 validation protocol executes

---

## Purpose

Codify the de facto pass/fail rules used for vol/conf scorecards and candidate evaluation into one canonical, falsifiable contract. All Research scorecards and candidate rankings must reference this spec until a revised version is ratified by the CIO.

---

## Pass/Fail Criteria (hard gates)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Sharpe ratio** | > 0 | Positive risk-adjusted return is the minimum viable bar. A non-positive Sharpe means the strategy does not compensate for volatility. |
| **Maximum drawdown** | < 10% (0.10 fraction) | Capital preservation invariant. A 10% peak-to-trough loss is the firm-wide halt trigger for live trading; backtests must respect the same bound. |
| **Closed trades (round trips)** | ≥ 4 | Minimum statistical evidence. Fewer than 4 completed trades cannot distinguish signal from noise on a 1000-bar slice. |

### Hard gate verdict logic

```
PASS  := sharpe > 0 AND max_drawdown < 0.10 AND closed_trades >= 4
FAIL  := NOT PASS
```

A single failing criterion produces **FAIL** regardless of other metrics.

---

## Tie-break criteria (when multiple candidates PASS)

When two or more presets or candidates meet all hard gates, apply in order:

1. **Higher Sharpe ratio** — primary objective is risk-adjusted return.
2. **Lower maximum drawdown** — secondary risk preference.
3. **Higher average trade return** — efficiency of capital deployment.
4. **Higher trade retention percentage** — fewer blocked/hold states implies cleaner signal.
5. **Lower turnover** — reduced friction and fee drag.

Tie-break lexicographic key:
```
(Sharpe ↓, max_drawdown ↑, avg_trade ↓, trade_retention_pct ↓, turnover ↑)
```
where ↓ means "prefer higher" and ↑ means "prefer lower".

---

## Metrics definitions

| Metric | Source field | Definition |
|--------|--------------|------------|
| `sharpe` | `bmSharpe` | Mean excess return divided by standard deviation of returns, annualized. |
| `max_drawdown` | `bmMaxDrawdown` | Peak-to-trough equity decline as a fraction of peak equity. |
| `avg_trade` | `bmAvgTradeReturn` | Mean return per completed round trip. |
| `closed_trades` | `bmRoundTrips` | Count of completed entry-exit cycles. |
| `trade_retention_pct` | Computed | `100 * (closed_trades / position_changes)` when `position_changes > 0`; otherwise 0. Measures what fraction of position changes result in completed round trips vs. holds/blocks. |
| `turnover` | `bmTurnover` | Sum of absolute position changes divided by initial equity. |

---

## Scorecard schema

Every Research scorecard must include:

1. **Header:** author, date, data slice, binary path/hash, commit hash.
2. **5-row minimum** for preset comparisons (baseline + variants).
3. **Exact reproduction commands** for every row.
4. **Verdict column** with PASS/FAIL per this spec.
5. **Winning preset** or **INCONCLUSIVE** declaration with rationale.
6. **Blocker status** if any row could not be run.

---

## Falsifiability

This spec is designed to be disproven:

- **If** a candidate with Sharpe ≤ 0 is later shown to be profitable live, the Sharpe threshold is too high.
- **If** a candidate with max_drawdown ≥ 10% is later shown to recover without live halt, the drawdown bound is too tight.
- **If** candidates with 1–3 trades consistently generalize to out-of-sample profit, the activity floor is too high.
- **If** the tie-break order ranks a worse-live candidate above a better-live candidate, the lexicographic key is wrong.

Any of the above observations, backed by live-trade evidence or walk-forward OOS backtests, shall trigger a spec revision request to the CIO.

---

## Relationship to formal optimization

The `Trader.Formal.Optimization` module defines an `roiImplementationScore` used for parameter search. This contract spec is **orthogonal** to that scorer:

- `roiImplementationScore` is for **ranking** within a search space.
- This contract spec is for **admission** to the candidate pool.

A candidate must pass this contract to be eligible for `roiImplementationScore` ranking. The formal scorer may then break ties among admitted candidates.

---

## Acceptance criteria for this artifact

- [x] Criteria are numeric and unambiguous.
- [x] Verdict logic is a single boolean expression.
- [x] Tie-break order is total and deterministic.
- [x] Metrics map to existing `BacktestMetrics` fields.
- [x] Falsifiability conditions are explicit.
- [x] Relationship to existing formal modules is documented.

---

## Next priority

1. **Execution P1 bisect:** Once B5 clears, run the locked 5-row vol/conf scorecard using this spec and publish ratified results.
2. **P3 execution-ready packet:** Deliver exact CLI invocations for Candidate A (`conf_blend`) 6-combo grid, referencing this spec for admission.
3. **Spec revision trigger:** If live-trade evidence contradicts any criterion, file revision request with evidence to CIO.
