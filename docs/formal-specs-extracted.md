# Extracted Formal Specifications

Date extracted: 2026-06-21

This document consolidates the formal and quasi-formal specifications implemented by the codebase and described by the repository documentation. It separates implemented executable contracts from documented intended contracts and known parity gaps.

Status legend:

- `Implemented`: enforced directly by code and covered by tests or executable verification helpers.
- `Documented`: described by docs, comments, or audit artifacts; may not be fully enforced end to end.
- `Gap`: documentation explicitly says implementation parity is incomplete.

## Sources

Primary implemented sources:

- `haskell/app/Trader/SignalGates.hs`
- `haskell/app/Trader/VolConfGate.hs`
- `haskell/app/Trader/Trading.hs`
- `haskell/app/Trader/OrderExecution.hs`
- `haskell/app/Trader/RoiScore.hs`
- `haskell/app/Trader/Formal/Execution.hs`
- `haskell/app/Trader/Formal/Risk.hs`
- `haskell/app/Trader/Formal/Optimization.hs`
- `haskell/app/Trader/Formal/CloseTiming.hs`
- `haskell/app/Trader/Predictors/Conformal.hs`
- `haskell/app/Trader/Predictors/Quantile.hs`
- `haskell/app/Trader/ThresholdCalibration.hs`
- `haskell/app/Trader/GateTelemetry.hs`
- `haskell/app/Trader/MarketDataIntegrity.hs`
- `haskell/app/Trader/BotStartSemantics.hs`
- `haskell/app/Trader/TopCombosStore.hs`
- `haskell/app/Trader/TopComboScoring.hs`
- `haskell/app/Trader/CostCalibration.hs`

Primary documentation sources:

- `FORMAL_METHODS.md`
- `haskell/app/Trader/Formal/GateTelemetry.md`
- `haskell/app/Trader/Formal/ThresholdCalibration.md`
- `docs/audits/strategy-decision-flow-spec.md`
- `docs/audits/p0-item-2-data-integrity-and-leakage.md`
- `README.md`
- `CHANGELOG.md`

## Spec Index

| ID | Specification | Status |
|---|---|---|
| S1 | Fresh-entry threshold, headroom, spike, and fee-buffer gates | Implemented |
| S2 | Directionality gate | Implemented |
| S3 | Volatility-confidence gate | Implemented |
| S4 | Conformal prediction interval admissibility | Implemented |
| S5 | Quantile prediction interval admissibility | Implemented |
| S6 | Backtest cost attribution | Implemented |
| S7 | Order execution and position-state reconciliation | Implemented |
| S8 | Risk halt decision | Implemented |
| S9 | ROI scoring, optimizer tie-breaks, public optimizer surface | Implemented |
| S10 | Close-timing analysis and max-hold retune acceptance | Implemented |
| S11 | Threshold calibration | Implemented |
| S12 | Gate telemetry | Implemented |
| S13 | Market-data freshness and continuation helpers | Implemented helper, broader data-QA gap |
| S14 | Bot startup, adoption evidence, and start queue safety | Implemented |
| S15 | Cost calibration and venue cost floors | Implemented |
| S16 | Top-combo scoring, freshness, pruning, and merge safety | Implemented |
| S17 | Strategy decision flow across latest-signal, live bot, and backtest | Documented with parity gaps |
| S18 | Data integrity and leakage checklist | Documented with implementation gaps |

## S1. Fresh-Entry Gates

Status: Implemented.

Primary code:

- `Trader.SignalGates`
- `Trader.Trading.mkEntryGateState`
- `Trader.Trading.simulateEnsembleVWithHLChecked`

Documentation:

- `FORMAL_METHODS.md`
- `README.md`

Definitions:

- `threshold = normalizeSignalOpenThreshold(openThreshold)`
- `headroomMultiple = sgcEntryEdgeHeadroomMultiple`, default `1.5`
- `requiredHeadroom = headroomMultiple * threshold`
- `requiredEdge = requiredHeadroom + roundTripFeeFloor`
- `spikeCap = min(sgcEntryEdgeSpikeMultiple * threshold, sgcEntryEdgeSpikeCredibleCap)`

Implemented defaults in `defaultSignalGateConfig`:

- `sgcEntryEdgeHeadroomMultiple = 1.5`
- `sgcEntryEdgeSpikeMultiple = 1000.0`
- `sgcEntryEdgeSpikeCredibleCap = 5.0`
- `sgcEntryEdgeSpikeConsecutiveLimit = 3`

Formal clauses:

1. A fresh entry is admissible only if `openThreshold` is finite and `>= 0`.
2. `normalizeSignalEntryEdge(rawEdge)` preserves finite non-negative edges and returns `Nothing` for non-finite edges. Negative finite raw edges normalize to `Just 0`.
3. `signalEntryHeadroomOk openThreshold edge` is exactly the zero-fee specialization of `signalEntryFeeBufferOk openThreshold 0 edge`.
4. `signalEntryFeeBufferOk` requires a finite non-negative fee floor and an explicit finite edge satisfying `edge >= requiredEdge`.
5. Missing edge, negative threshold, non-finite threshold, negative fee floor, non-finite fee floor, or non-finite edge fails closed.
6. With fixed threshold and edge, admissibility is monotone non-increasing as fee floor rises.
7. With fixed threshold and fee floor, admissibility is monotone non-increasing as edge falls.
8. The spike gate admits only explicit finite non-negative edge values satisfying `edge <= spikeCap`.
9. `signalEntryHeadroomThresholdCap edge = normalizedEdge / 1.5`; malformed or non-finite edge evidence yields cap `0`.
10. `mkEntryGateState` applies spike, headroom, and fee-buffer checks only when opening from flat, reuses one normalized edge sample for all three checks, and combines them conjunctively.
11. In `mkEntryGateState`, malformed per-side fee becomes `NaN`, which is rejected by the fee-buffer gate. This prevents bad fees from reopening a blocked entry.
12. Fresh-entry sizing must prove max-position cap and min-position floor are finite and non-negative before entry can open. Invalid sizing evidence maps to no-entry sizing.
13. Valid zero cap is a legal no-entry boundary. Valid zero floor is admissible. Equality at the min-size floor is admissible. Tightening a valid cap cannot increase realized fresh-entry exposure.
14. Documentation also describes the implemented infeasible-threshold relation: because headroom requires `edge >= 1.5 * openThreshold` and the default spike veto requires `edge <= min(1000 * openThreshold, 5.0)`, open thresholds above `10/3` are infeasible, with equality at `10/3` admissible only at `edge == 5.0`.

Bounded obligations:

- `testSignalGateEntryHeadroomSpecializesFeeBuffer`
- `testSignalGateEntryFeeBufferFailsClosed`
- `testNormalizeSignalEntryEdgeFailClosedRegression`
- `testTradingEntryGateFailClosedMonotone`
- `testTradingEntryGateMalformedNoReopen`
- `testBacktestEntryGateUsesRoundTripFeeBuffer`
- `testBacktestFreshEntrySizingBoundsFailClosed`
- `testSignalGateEntryEdgeSpike*`

## S2. Directionality Gate

Status: Implemented.

Primary code:

- `Trader.SignalGates`

Documentation:

- `README.md`

Implemented defaults:

- `lookbackBars = 24`
- `chopEfficiencyMax = 0.08`
- `mrEfficiencyMax = 0.35`
- `weakBandZMin = 0.5`
- `regimeMassTolerance = 1e-3`

The README now matches the implemented default `NON_DIRECTIONAL_CHOP` boundary at `<= 0.08`.

Definitions:

- Per-bar return path uses additive simple returns.
- `efficiency = abs(sum returns) / sum(abs returns)`, with zero path mapped to `0`.
- `zScore = sum returns / (stdev returns * sqrt(n))`, with tiny stdev mapped to `0`.

Formal clauses:

1. A directionality window is valid only when required prices exist, are finite, and prior price is non-zero.
2. Malformed windows produce `NON_DIRECTIONAL_MALFORMED`.
3. Efficiency must be finite and within `[0,1]`, with small tolerance.
4. If `efficiency <= chopEfficiencyMax`, the gate reports `NON_DIRECTIONAL_CHOP`.
5. If `efficiency <= mrEfficiencyMax`, the weak-band path applies.
6. In the weak band, regime hysteresis must be finite and non-negative.
7. Present regime probabilities must each be finite and in `[0,1]`, have positive total mass, and sum to `1 +/- 1e-3`.
8. If valid regime probabilities are MR-dominant by the hysteresis gap, the gate reports `NON_DIRECTIONAL_MR`.
9. If no MR veto applies, the requested side must be confirmed by signed z-score: long requires `zScore >= weakBandZMin`; short requires `zScore <= -weakBandZMin`.
10. Missing chosen side, opposite-signed z-score, zero z-score, or non-finite z-score fails the weak-band confirmation.
11. Absent regime probabilities do not by themselves make the state malformed; the gate falls back to signed weak-band z-score.

Bounded obligations:

- `testSignalDirectionalityLiveSemanticsRegression`
- `testSignalDirectionalityPredictionAwareLiveSemantics`
- `testSignalGatePredictionSanityInvariant`
- `testSignalGatePredictionAwareWeakBand`

## S3. Volatility-Confidence Gate

Status: Implemented.

Primary code:

- `Trader.VolConfGate`
- `Trader.Formal.Optimization`

Documentation:

- `FORMAL_METHODS.md`
- `README.md`

Implemented defaults:

- Volatility evidence maximum: `2.0`
- Low-vol threshold: `0.5`
- High-vol threshold: default `1.2`, tighter `1.0`, looser `1.4`
- Weak confidence threshold: default `0.60`, stricter `0.65`
- Strong confidence threshold: `0.80`
- Size multipliers are clamped to `[0,1]`.

Formal clauses:

1. Disabled preset always returns `AllowEntry` with size multiplier `1.0`.
2. For enabled presets, volatility evidence is admissible only when present, finite, `>= 0`, and `<= volatilityEvidenceMax`.
3. Missing, negative, non-finite, or above-range volatility maps to `AllowExitOnly` with size multiplier `0`.
4. Provided confidence evidence is admissible only when finite and in `[0,1]`.
5. Missing confidence maps to weak confidence. Malformed provided confidence maps to `AllowExitOnly 0`.
6. Volatility buckets are low when `vol < lowVolThreshold`, medium when `lowVolThreshold <= vol < highVolThreshold`, and high when `vol >= highVolThreshold`.
7. Confidence buckets are weak when `conf < weakThreshold`, medium when `weakThreshold <= conf < strongThreshold`, and strong when `conf >= strongThreshold`.
8. Low/medium volatility plus weak confidence holds flat; high volatility plus weak confidence blocks; high volatility plus medium confidence is exit-only; strong confidence can allow entry with the configured multiplier.
9. `applyVolConfGateBehavior` preserves reduce-only semantics for `Block` and `AllowExitOnly`; those states cannot open or flip exposure.
10. Tightening confidence or high-volatility requirements cannot reopen malformed or weak evidence on the same bounded witness.

Bounded obligations:

- `testVolConfGateMalformedInputsFailClosed`
- `Trader.Formal.Optimization.verifyFormalOptimization` asserts the VolConf canonicalization, malformed-volatility, malformed-confidence, conservative-input, and bounded-output report fields against the same production semantics.

## S4. Conformal Prediction Intervals

Status: Implemented.

Primary code:

- `Trader.Predictors.Conformal`

Documentation:

- `FORMAL_METHODS.md`
- `README.md`

Definitions:

- `alpha` is clamped into `[1e-6, 0.999999]`.
- Valid residual evidence is a non-empty list of finite values `>= 0`.
- Radius is the conformal quantile selected from sorted residuals.

Formal clauses:

1. Empty calibration evidence fails closed.
2. If any residual is negative, `NaN`, or infinite, the entire residual sample fails closed; malformed residuals are not filtered out.
3. Failed evidence returns `ConformalModel { cmCount = 0, cmRadius = Infinity }`.
4. `predictInterval` returns `(-Infinity, Infinity, Nothing)` when the model is unavailable, radius is malformed, or point forecast is non-finite.
5. Valid zero residuals are admissible and can produce a zero-width interval.
6. `sigmaFromInterval` returns `Nothing` for non-positive or non-finite interval width.
7. For finite `mu` and valid evidence, interval width is exactly `2 * cmRadius`.
8. Increasing the selected residual quantile cannot narrow the emitted interval.

Bounded obligations:

- `testConformalCalibrationResidualsFailClosed`

## S5. Quantile Prediction Intervals

Status: Implemented.

Primary code:

- `Trader.Predictors.Quantile`

Documentation:

- `FORMAL_METHODS.md`

Formal clauses:

1. Quantile prediction requires all three linear heads to expose the same positive feature dimension.
2. The query feature vector must have exactly that dimension and all query features must be finite.
3. Model weights and biases must be finite.
4. Raw `q10`, `q50`, and `q90` predictions must be finite.
5. Ordered evidence is required: `q10 <= q90`. Inverted evidence fails closed and is not repaired by sorting.
6. Once admissible, `q50` may be clamped into `[q10,q90]`, but the raw q50 is still returned.
7. Equality `q10 == q90` is admissible but yields no sigma.
8. `sigmaFromQ1090` requires positive finite width and returns `width / (2 * Phi^-1(0.9))`.
9. Widening ordered bounds cannot narrow the emitted interval or decrease positive sigma.

Bounded obligations:

- `testQuantileSanitizesMalformedInputs`

## S6. Backtest Cost Attribution

Status: Implemented.

Primary code:

- `Trader.Trading.costAttributionFromTotals`

Documentation:

- `FORMAL_METHODS.md`
- `README.md`

Definitions:

- Net curve is the realized simulation equity path after costs.
- Cumulative realized costs are fee, slippage, spread, and funding buckets.
- Gross attribution curve is `net + cumulativeRealizedCosts`.

Formal clauses:

1. Cost buckets are attributed across fee, slippage, spread, and funding.
2. Non-finite net-curve values are sanitized to zero for attribution output.
3. Cost totals are aligned to the net curve length; missing cost points use the last available cumulative cost or zero.
4. `gross[i] = net[i] + cumulativeCosts[i]`.
5. Final residual is `finalGross - finalTotalCost - finalNet`.
6. The expected accounting identity is `gross - realized costs = net`, up to finite residual drift.
7. The gross curve is not a no-cost counterfactual replay. Consumers needing no-cost performance must run a distinct no-cost simulation.

Bounded obligations:

- `testBacktestCostAttributionGrossNetConsistency`
- `testBacktestCostAttributionNonFiniteComponentsRegression`

## S7. Order Execution And Position State

Status: Implemented.

Primary code:

- `Trader.OrderExecution`
- `Trader.Formal.Execution`

Formal clauses:

1. Unsent orders apply no quantity.
2. Non-live orders apply the positive finite fallback quantity.
3. Live orders trust positive finite explicit executed quantity first, even if status is terminal.
4. If live order has no executed quantity, no-fill statuses yield `Nothing`.
5. Fill-implying statuses use the positive finite fallback quantity.
6. Unknown or empty statuses with no executed quantity yield `Nothing`.
7. Applied fractions convert base-unit fill evidence into position-fraction state by scaling intended fraction by `appliedBase / requestedBase`.
8. If requested base quantity is absent, applied fraction falls back to all-or-nothing on intended fraction.
9. Executed fill updates signed exposure algebraically: `newSigned = currentSigned + signedFill`.
10. `closeQty` is the fill portion reducing existing opposite exposure; `openQty` is the remaining fill portion.
11. Sanitized fill quantity is conserved: `closeQty + openQty ~= sanitizedQty`.
12. Reduce-only execution can reduce or close current exposure but can never increase exposure, flip direction, or reopen.
13. Negative, zero, non-finite, or tiny quantities sanitize to zero.
14. Extremely large finite magnitudes are capped before signed exposure arithmetic.

Bounded obligations:

- `verifyFormalExecution`
- `testFormalExecutionInvariants`
- `testOrderExecutionFillSanitizationInvariant`
- `testOrderExecutionCorruptedInputInvariant`

## S8. Risk Halt Decision

Status: Implemented.

Primary code:

- `Trader.Trading.specRiskHalt`
- `Trader.Formal.Risk`

Formal clauses:

1. Previous daily-loss halt resets when day changes.
2. Previous weekly-loss halt resets when week changes.
3. Other previous halt reasons persist.
4. If no previous halt remains, risk checks run in fixed priority order:
   `RISK_LIMIT_NON_FINITE`, `DRAWDOWN_LIMIT_INVALID`, `POSITION_SIZE_INVALID`, `EXPECTANCY_INVALID`, `VOL_TARGET_INVALID`, `LEVERAGE_INVALID`, daily loss, weekly loss, drawdown, negative expectancy, position size, loss streak.
5. Any configured numeric risk limit that is `NaN` or infinite yields `RISK_LIMIT_NON_FINITE`.
6. Drawdown limit must be finite and strictly inside `(0,1)`.
7. Position size evidence must be finite, non-negative, and `<= 10`.
8. If minimum expectancy is configured, expectancy evidence must be present and finite.
9. Vol target evidence must be finite, non-negative, and `<= 10`.
10. Leverage evidence must be finite, non-negative, and `<= 150`.
11. Negative configured limits are sanitized to zero before comparisons, making them maximally restrictive rather than disabling halts.
12. Daily loss, weekly loss, and drawdown halt on `metric >= limit`.
13. Negative expectancy halts when `expectancy < minExpectancy`.
14. Position-size halt occurs when `positionSize > maxPositionSize`.
15. Loss-streak halt occurs only when limit is positive and `consecutiveLosses > limit`.
16. Increasing risk metrics cannot remove a halt on the bounded verification domain.

Bounded obligations:

- `verifyFormalRisk`
- `testFormalRiskInvariants`
- `testFormalRiskNoFalsePositiveWitness`
- `testFormalRiskNegativeLimitSanitization`
- `testFormalRiskPositionSizeHalt`
- `testFormalRiskLossStreakHalt`

## S9. ROI Scoring And Optimizer Tie-Breaks

Status: Implemented.

Primary code:

- `Trader.RoiScore`
- `Trader.Formal.Optimization`
- `Trader.Optimization`

Documentation:

- `FORMAL_METHODS.md`
- `README.md`
- `CHANGELOG.md`

ROI score definition:

```text
score =
  annualizedReturn
  - penaltyMaxDrawdown * (maxDrawdown + tailLoss)
  - penaltyTurnover * turnover
  + expectancyReward
  + paybackReward
  - sparseEvidencePenalty
```

Formal clauses:

1. Inputs are sanitized: non-finite returns, drawdown, tail loss, turnover, expectancy, exposure, and penalties map to zero; negative risk/turnover/exposure terms are clamped to zero where applicable.
2. Activity count is `max(roundTrips, tradeCount)`.
3. Higher annualized return is preferred.
4. Higher drawdown, tail loss, and turnover are penalized.
5. Positive expectancy reward requires at least one completed round trip and exposure above the configured minimum exposure floor.
6. Negative expectancy is still penalized.
7. Payback reward requires completed round trips at or above the activity floor, exposure above the floor, positive expectancy, and positive finite payback duration.
8. Invalid, zero, negative, or non-finite payback durations are equivalent to missing payback.
9. Low or zero activity and low or zero exposure incur evidence penalties.
10. With other inputs fixed, score is monotone non-decreasing in annualized return, expectancy, valid exposure, and valid payback quality.
11. With other inputs fixed, score is monotone non-increasing in drawdown, tail loss, turnover, and sparse-evidence penalties.
12. Candidate tie-break order after normalization is: higher final equity, lower turnover, higher round trips, non-inverted thresholds preferred over inverted, then lexicographically larger `(openThreshold, closeThreshold)`.
13. Public optimizer/trading constructors remain part of the stable surface expected by optimizer and metrics consumers.

Bounded obligations:

- `verifyFormalOptimization`
- `testOptimizerActivityCountInvariant`
- `testOptimizerPublicSurfaceRegression`
- `testMetricsConsumesTradingPublicResults`
- `testOptimizer*` ROI and quality-budget regressions

## S10. Close Timing And Max-Hold Retunes

Status: Implemented.

Primary code:

- `Trader.Formal.CloseTiming`

Documentation:

- `README.md`
- `CHANGELOG.md`

Definitions:

- For a trade opened at `ta` and closed at `tc`, the optimal close search window is `[ta, ta + 2 * (tc - ta)]`.
- The chosen optimal point maximizes PNL; ties choose the earliest timestamp/index.
- Timing ratio is normalized to `[0,2]`.

Formal clauses:

1. Observations with `tc <= ta` are invalid for timestamp-based close timing.
2. PNL candidates must be finite.
3. Per-trade samples require finite positive entry price and finite non-zero side.
4. Per-combo summaries use robust percentiles and MAD over normalized timing ratios.
5. Minimum close-timing sample count is `5`.
6. Minimum positive-lift support count is `3`.
7. Runtime live max-PNL evidence ignores non-positive holding periods and returns the 75th percentile only after positive-lift support reaches the floor.
8. Runtime learned max-hold can narrow an existing cap but cannot widen it.
9. Offline recommended max-hold acceptance requires: recommendation present and positive, ordered support, finite positive lift stats, recommendation inside analyzed hold domain, exact bucket evidence, permitted application direction, and a one-bar deadband versus current cap.
10. A recommendation with insufficient support, non-finite lift, zero/negative median lift, outside-domain value, missing exact bucket evidence, or deadband-only change fails closed.
11. `--max-hold-bars 0` is documented as valid and disables forced max-hold exits.

Bounded obligations:

- `testLiveMaxPnlCloseTimingRecommendation`
- `testOptimizerCloseTimingRecommendationRequiresAcceptedEvidence`
- `testOptimizerCloseTimingMetricsRecordAppliedRecommendation`
- `testMaxHoldBarsZeroDisablesForcedExit`

## S11. Threshold Calibration

Status: Implemented.

Primary code:

- `Trader.ThresholdCalibration`

Documentation:

- `haskell/app/Trader/Formal/ThresholdCalibration.md`
- `CHANGELOG.md`

Formal clauses:

1. Empty edge input yields `Nothing`.
2. If any edge is negative, `NaN`, or infinite, the whole distribution is rejected.
3. Percentiles are computed over sorted non-negative edge evidence.
4. `thresholdAtPercentile` clamps outside `[0,100]` and linearly interpolates between stored anchors.
5. Percentile threshold is monotone in percentile.
6. Config validation requires finite positive headroom divisor, finite non-negative fee floor, non-negative minimum sample size, percentiles in `[0,100]`, and aggressive percentile `<=` conservative percentile.
7. `headroomThreshold = suggestedThreshold / headroomDivisor`; default divisor is `1.5`.
8. `feeBufferThreshold = suggestedThreshold + feeFloor`; default fee floor is `0.001`.
9. Confidence interval is ordered because it is `threshold +/- 1.96 * standardError`.
10. Recommendation is one of `INSUFFICIENT_SAMPLE`, `CONSERVATIVE`, `AGGRESSIVE`, or `BALANCED`.

Bounded obligations:

- `testThresholdCalibration*`

## S12. Gate Telemetry

Status: Implemented.

Primary code:

- `Trader.GateTelemetry`

Documentation:

- `haskell/app/Trader/Formal/GateTelemetry.md`

Formal clauses:

1. Telemetry is observational: recording a rejection cannot affect gate decisions because telemetry is written after the decision and not read by gate logic.
2. Empty telemetry has zero counts, no recent rejections, and no binding gate.
3. Rejection histogram counts by `(gate, reason)`.
4. Recent rejection history is bounded by `maxRecent`.
5. Binding gate is the gate with the highest rejection count.
6. Diagnosis is derived from candidate/rejection counts and is one of the documented telemetry diagnosis classes.

Bounded obligations:

- `testGateTelemetryEmptyInvariant`
- `testGateTelemetryAccumulationInvariant`
- `testGateTelemetryBindingGateIdentification`
- `testGateTelemetryHistogramSorting`

## S13. Market Data Freshness And Continuity

Status: Implemented helper; broader live data-QA gate is documented as a gap.

Primary code:

- `Trader.MarketDataIntegrity`

Documentation:

- `docs/audits/p0-item-2-data-integrity-and-leakage.md`
- `README.md`

Implemented helper clauses:

1. Interval strings must parse through `parseIntervalSeconds`; invalid intervals produce `MARKET_DATA_INTERVAL_INVALID`.
2. Last close time is `lastOpenTimeMs + intervalMs`.
3. Data age is `nowMs - lastCloseTimeMs`.
4. Data is stale when `ageMs > intervalMs`.
5. Stale reason includes age, budget, and last close time.
6. Continuation expects each next open time to equal the previous expected open plus interval.
7. First mismatch yields `MARKET_DATA_GAP` with expected, actual, and interval values.
8. `MARKET_DATA_GAP` and `STALE_MARKET_DATA` are transient market-data errors for queued bot starts.

Documented broader data-QA checklist:

1. Symbol canonicalization before HTTP request.
2. Timestamp parse and unit normalization.
3. Strict monotonicity and duplicate rejection.
4. Missing-bar continuity.
5. Closed-bar completeness; open candles must be discarded for live trading.
6. Stale-data freshness.
7. Finite OHLCV values.
8. OHLC relationship invariants.
9. Non-negative volume where required.
10. Feature causality: features at `t` use only data `<= t`, target at `t+1`.
11. Missing OHLCV must be surfaced or blocked in live trading rather than hidden by synthetic fallback.
12. Live trade placement should hard-fail on failed data QA.

Gap:

- The P0 data-integrity audit states that the broader checklist is not yet fully enforced across all live-loaded series and venue loaders.

Bounded obligations:

- `testMarketDataFreshnessAndContinuationInvariant`
- `testQueuedBotStartIgnoresTransientMarketDataErrors`

## S14. Bot Startup And Adoption Evidence

Status: Implemented.

Primary code:

- `Trader.BotStartSemantics`

Documentation:

- `README.md`
- `CHANGELOG.md`

Formal clauses:

1. Live-adopted combo `maxPositionSize` is capped to `[0, adoptionMaxPositionSizeCap]`, default cap `0.25`.
2. Negative, non-finite, or negative-cap adoption sizing collapses to zero.
3. Adoption requires trade-count evidence: missing `tradeCount` fails closed.
4. Adoption trade-count predicate is monotone in trade count and defaults to floor `20`.
5. Adoption requires walk-forward mean Sharpe evidence: missing or non-finite Sharpe fails closed.
6. Walk-forward Sharpe passes at equality: `sharpe >= 0.3` by default.
7. Disabled symbols compare after uppercase/whitespace normalization.
8. Bot startup ROI acceptable iff final equity is finite and `> 1.0`.
9. Disabled startup backtest guard always allows.
10. Enabled guard with missing final equity yields no verdict, not abort.
11. Enabled guard with acceptable final equity yields `BacktestAllow`.
12. Enabled guard with unacceptable final equity yields `BacktestAbort` only when trade count is at or above the minimum evidence floor.
13. Under-min-trades, zero-trade, unknown-trade, or missing final-equity windows yield `BacktestNoVerdict`.
14. Default startup backtest minimum trades is `3`.
15. Startup guard never prunes combos; it can block starts but pruning belongs to scheduled refresh/optimizer logic.
16. Orphan-position symbols are prioritized before regular start symbols, with stable deduplication.
17. Queued starts ignore transient market data errors but block when order errors reach configured positive max-order-errors.
18. Position-origin persistence requires trading enabled, live mode, a switched position, and an order actually sent.

Bounded obligations:

- `testCapAdoptedMaxPositionSizeBoundsLiveExposure`
- `testAdoptionMinTradeCountMatchesOptimizerProductionGate`
- `testComboTradeCountMeetsAdoptionFloor*`
- `testComboWalkForwardSharpeMeetsAdoptionFloor*`
- `testBotStartupBacktest*`
- `testPrioritizeOrphanBotStartSymbols`
- `testDisabledBotStartSymbols`
- `testQueuedBotStart*`

## S15. Cost Calibration And Venue Floors

Status: Implemented.

Primary code:

- `Trader.CostCalibration`

Documentation:

- `README.md`
- `CHANGELOG.md`

Implemented constants:

- `venueTakerFeeFloor = 5.0e-4`
- `venueSlippageFloor = 5.0e-5`
- `venueSpreadFloor = 1.0e-4`
- `venueRoundTripCostFloor = 2 * (fee + slippage) + spread = 1.2e-3`
- `minEdgeCostMultiplier = 1.5`
- `venueMinEdgeFloor = minEdgeCostMultiplier * venueRoundTripCostFloor`

Calibration defaults:

- Minimum observations: `8`
- Shrinkage pseudo-count: `16`
- Window: `64`
- Floor factor: `0.25`
- Max per-side slippage: `0.01`
- Outlier bound: `0.05`

Formal clauses:

1. Observed slippage requires side `BUY` or `SELL`, positive finite decision price, positive finite executed quantity, and positive finite cumulative quote.
2. BUY slippage is positive when average fill exceeds decision price; SELL slippage is positive when average fill is below decision price.
3. Unknown side, non-positive or non-finite fields, and measurements exceeding the outlier bound yield `Nothing`.
4. Before minimum observations, calibrated slippage equals the sanitized configured prior.
5. Configured non-finite or negative prior sanitizes to zero.
6. After enough observations, use the median of the most recent finite observations within the configured window.
7. Realized estimate is shrunk toward prior with `w = n / (n + shrinkageObs)`.
8. Calibrated value is floored at `floorFactor * prior` and capped at `maxPerSide`.
9. Venue min-edge floor requires deployed/sampled min-edge to beat round-trip venue floors by the cost multiplier.

Bounded obligations:

- `testObservedSlippageFractionSemantics`
- `testCalibratedSlippageShrinkage`
- `testCostCalibrationConfigurableRoiKnobs`
- `testVenueRoundTripCostFloorMatchesVenueCosts`
- `testVenueMinEdgeFloorClearsRoundTripCost`
- `testVenueMinEdgeFloorMatchesProductionRegressionEvidence`

## S16. Top-Combo Scoring, Refresh, And Pruning

Status: Implemented.

Primary code:

- `Trader.TopComboScoring`
- `Trader.TopCombosStore`
- `Trader.BotStartSemantics`

Documentation:

- `README.md`
- `CHANGELOG.md`

Formal clauses:

1. Top-combo scoring config validates finite/non-negative/positive/unit-bounded parameters as appropriate.
2. Live annualized return is clamped to configured floor and ceiling before scoring.
3. Live blend weight is `operationCount / (operationCount + shrinkageOps)` for positive operation count; zero operations yield weight `0`.
4. Live quarantine applies when operations meet the minimum and final equity is at or below the quarantine max final equity.
5. Validated annualized return is clamped to `[0, validatedAnnualizedReturnCap]`.
6. Walk-forward multiplier depends on deployable verdict: deployable, below-floor, or missing evidence.
7. Drawdown multiplier is `1 / (1 + scale * max(0, drawdown))`.
8. Equity term is `max(equityLogFloor, log(max(equityFloor, finalEquity)))`.
9. Freshness scoring is disabled when half-life `<= 0`; otherwise missing/non-finite/old age decays toward the freshness floor multiplier.
10. Positive scores are multiplied by freshness multiplier; non-positive scores are unchanged.
11. Scheduled refresh includes stale combos even when rank is low.
12. Refreshed non-zero-trade performance below final equity floor is pruned through a tombstone so stale replicas cannot resurrect it.
13. Zero-trade refresh windows update freshness but do not prune the combo.
14. Live adoption requires recent backtest/discovery freshness evidence; missing or stale freshness fails closed.
15. Merge logic deduplicates source/null-equivalent combos and preserves live stats through recalculation/backtest/merge paths.

Bounded obligations:

- `testSelectCombosForBacktestRefreshIncludesEveryStaleCombo`
- `testLiveComboFreshnessRequiresRecentBacktestEvidence`
- `testPrunedBacktestTombstonePreventsStaleResurrection`
- `testKeepAllUpdateKeepsUnprofitableComboStamped`
- `testTopComboBacktestPrunesRoiLosers`
- `testMerge*`
- `testLiveBlendShrinkageRanking`
- `testLiveQuarantineThresholds`
- `testLiveFamilyQuarantineAcrossUuidChurn`

## S17. Strategy Decision Flow

Status: Documented with parity gaps.

Primary documentation:

- `docs/audits/strategy-decision-flow-spec.md`

Canonical intended flow:

1. Prediction inputs: bar context, model predictions, confidence, regime, conformal, quantile, volatility context.
2. Raw directional proposals: derive open direction from open threshold and close direction from close threshold.
3. Entry-only edge gates: spike, headroom, and later fee buffer.
4. Post-direction gates: volatility, vol-target readiness, trend, cloud, price action, signal-to-noise, non-directionality, regime edge, MTF consensus, cross-asset, meta-label, funding/OI.
5. Side eligibility and positioning semantics.
6. Sizing pipeline: base size, confidence/vol-conf, volatility targeting, SNR, risk-per-trade, regime, pairs, funding, Kelly, cap, and floor.
7. Vol-confidence behavior application.
8. Execution intent: hold, open, close, flip, rebalance.
9. Exit ladder: signal exits, bracket exits, flip exits, max hold, risk halts, live-only order/auth halts.
10. State persistence: signal, gate reason, size, orders, open/closed trades, halt/cooldown/exposure counters.

Documented implementation state:

1. Latest-signal has the richest stateless prediction/gate/sizing path.
2. Live bot delegates upstream signal generation to latest-signal and adds stateful hold, cooldown, exposure, halt, execution, reconciliation, and persistence behavior.
3. Backtest reimplements a similar but not identical path inside `Trader.Trading`.
4. Backtest does not yet route through the same shared post-direction gate reducer as latest-signal.
5. Existing tests cover helper invariants, but not the full blocked-entry/hold/flip/halt path matrix end to end.

Gap:

- The audit explicitly states implementation parity is not fully achieved across latest-signal, live bot, and backtest.

## S18. Data Integrity And Leakage

Status: Documented with implementation gaps.

Primary documentation:

- `docs/audits/p0-item-2-data-integrity-and-leakage.md`

Documented current positives:

1. CSV and exchange decoders reject non-finite numeric payloads in parsed fields.
2. CLI/args symbol normalization exists for Binance, Coinbase, and Poloniex.
3. `Trader.Predictors.Features` is structurally no-lookahead for supervised labels and feature indexing when input bars are already closed and correctly ordered.
4. Predictor training preserves time order by using tail calibration rather than shuffling calibration into training.

Documented required P0 live-trading conditions:

1. Live trading should be blocked on stale, incomplete/open, gapped, duplicate, or out-of-order bars.
2. Venue loaders should validate strict monotonicity, continuity, closed-bar completeness, OHLC invariants, and non-negative volume where relevant.
3. Feature construction should not hide live data defects by fabricating OHLCV values in trading paths.
4. Cache TTL is not sufficient as a market-data freshness guarantee.
5. Open current candles must be discarded before strategy input.
6. Feature causality requires features at `t` to use only data `<= t`, with supervised target at `t+1`.

Gap:

- The audit verdict is partial for Binance live paths and still fail for full P0 repo-wide live-trading safety because the complete QA gate is not uniformly enforced after price loading across all venue paths.
