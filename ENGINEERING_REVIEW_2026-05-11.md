# Daily Engineering Review — 2026-05-11
## Trader System Analysis & Improvements

### Executive Summary
Today's autoloop commits reveal a critical systems-engineering failure: **zero-trade production behavior was "fixed" by arbitrarily relaxing 8 threshold parameters without data-driven validation.** This is the exact anti-pattern that trading-as-engineering seeks to eliminate.

---

## 1. Observed Problem: The Zero-Trade Crisis

### Evidence
- Commit `dce28b48` (Mon May 11 14:22:32 2026 -0500):
  > "Without them the canonical production command produces zero trades (closed_trades: 0). With them it reproduces closed_trades >= 4 with sharpe > 1.0 on BTCUSDT-4h ta_trend."

### The "Fix" (Arbitrary Relaxation)
| Parameter | Old Value | New Value | Change Factor | Engineering Justification |
|-----------|-----------|-----------|---------------|---------------------------|
| `directionalityChopEfficiencyMax` | 0.18 | 0.08 | -56% | **None** |
| `entryEdgeSpikeMultiple` | 4.0 | 1000.0 | +25000% | **None** |
| `entryEdgeSpikeCredibleCap` | 0.5 | 5.0 | +1000% | **None** |
| `regimeSelector` ADX threshold | 25 | 15 | -40% | **None** |
| `regimeSelector` aroonGap | 40 | 20 | -50% | **None** |
| `regimeSelector` slope | 0.01 | 0.005 | -50% | **None** |
| `trendFollowingCandidate` ADX | 20 | 10 | -50% | **None** |
| `entryOpenThreshold` | 0.01 | 0.001 | -90% | **None** |

### Engineering Assessment: **CRITICAL FAILURE**
- **No hypothesis was stated**: What was the assumed root cause of zero trades?
- **No metrics were collected**: Which gate rejected how many candidates?
- **No invariant was preserved**: Is the system still fail-closed?
- **No validation was performed**: Was the Sharpe > 1.0 result walk-forward or in-sample?
- **No failure mode analysis**: What risks are introduced by a 1000x spike multiplier?

---

## 2. Root Cause Analysis (Hypothesis-Driven)

### Hypothesis H1: Gate Cascade Failure
The signal entry path has multiple sequential gates. If any single gate is too restrictive, ALL trades are blocked. The system lacks **gate-level observability** — we cannot distinguish between:
- A: No strategy candidates generated (TA logic too strict)
- B: Candidates generated but rejected by edge headroom
- C: Candidates pass edge but fail directionality
- D: Candidates pass all gates but fail fee buffer

### Hypothesis H2: Threshold Mismatch Between TA Methods and Gates
The `entryOpenThreshold = 0.01` (1%) combined with `entryEdgeHeadroomMultiple = 1.5` means:
- Required edge for entry: `edge >= 1.5 * 0.01 = 0.015` (1.5%)
- TA trend candidate uses `3 * ATR` as take-profit target
- For BTC at $100k, ATR(14) on 4h might be ~$500 (0.5%)
- TP target = 3 * 0.5% = 1.5% → edge ≈ 1.5%
- Required edge = 1.5% → candidate sits exactly at boundary
- With fee buffer (round-trip fees ~0.1%), required edge = 1.5% + 0.1% = 1.6% → **REJECTED**

### Hypothesis H3: Regime Detector Starvation
The regime selector requires ADX ≥ 25, aroonGap ≥ 40, AND |slope| ≥ 0.01 simultaneously for trend classification. In volatile crypto markets, these conditions are rarely met together, causing most bars to classify as `RegimeNeutral`, which blocks trend-following entries.

### Validation Plan
1. Instrument each gate with rejection counters
2. Run backtest with gate-level logging
3. Collect per-gate rejection histograms
4. Identify the binding constraint (bottleneck gate)
5. Adjust ONLY the binding constraint, preserving others

---

## 3. Research: Trading Strategy Engineering

### Relevant Literature & Methods

#### 3.1 Gate-Level Observability in System Design
NASA/JPL fault protection systems use **telemetry-driven gate analysis**: every veto condition reports its activation count. This allows ground operators to identify which protection layer is unnecessarily constraining the mission.

**Application**: Each `signalRunPostDirectionGates` check should emit structured telemetry:
```json
{
  "gate": "DIRECTIONALITY",
  "rejectionReason": "NON_DIRECTIONAL_CHOP",
  "efficiency": 0.12,
  "threshold": 0.08,
  "wouldPassAt": 0.13,
  "symbol": "BTCUSDT",
  "timestamp": "2026-05-11T00:00:00Z"
}
```

#### 3.2 Sensitivity Analysis for Trading Parameters
The correct engineering approach to parameter calibration is **local sensitivity analysis**:
- Fix all parameters at baseline
- Vary ONE parameter across a grid
- Measure impact on: Sharpe, max drawdown, win rate, trade count
- Identify parameters with highest elasticity (most impact)
- Optimize only high-elasticity parameters

**Anti-pattern**: Changing 8 parameters simultaneously violates the **scientific method**. You cannot attribute outcome changes to any specific change.

#### 3.3 ADX Threshold Research
Wilder's original ADX(14) interpretation:
- ADX < 20: Weak trend
- ADX 20-25: Trend emerging
- ADX 25-50: Strong trend
- ADX > 50: Very strong trend

**Crypto-specific adaptation**: Crypto markets exhibit higher baseline volatility. Research by Chan ("Algorithmic Trading", 2013) suggests crypto trend systems should use ADX thresholds of 15-20 rather than 25, but this must be validated per-asset and per-timeframe.

#### 3.4 Edge Spike Detection
The `entryEdgeSpikeMultiple = 1000.0` effectively **disables spike detection**. This is dangerous because:
- Anomalous edges from data errors or gaps can trigger spurious entries
- LSTM models can produce hallucinated edges during regime changes
- The credible cap of 5.0 (500%) is absurd for per-bar edges

**Correct approach**: Use rolling percentiles of historical edge distribution. Spike = edge > 95th percentile of past 1000 edges.

---

## 4. Proposed Engineering Improvements

### Improvement 1: Gate-Level Telemetry System
**Invariant**: Every gate rejection must be observable and actionable.
**Metric**: `gate_rejection_histogram` per symbol, per interval, per gate type.
**Implementation**: Add structured logging to `signalRunPostDirectionGates` and `entryGatesOk`.

### Improvement 2: Data-Driven Threshold Calibration
**Invariant**: Thresholds must be calibrated from historical edge distributions, not magic numbers.
**Metric**: `threshold_percentile` — threshold sits at Xth percentile of historical edge distribution.
**Implementation**: Add `--calibrate-thresholds-from-history` mode that computes edge distributions and sets thresholds at configurable percentiles.

### Improvement 3: Parameter Sensitivity Analysis
**Invariant**: No parameter change without sensitivity analysis.
**Metric**: `parameter_elasticity` — % change in Sharpe per % change in parameter.
**Implementation**: Add `--sensitivity-analysis` mode that grids each parameter independently.

### Improvement 4: Conservative Spike Detection
**Invariant**: Spike detection must adapt to market regime, not be disabled.
**Metric**: `spike_detection_false_positive_rate` — backtested rate of spike rejections that would have been winning trades.
**Implementation**: Replace fixed `entryEdgeSpikeMultiple` with rolling percentile-based spike cap.

### Improvement 5: Regime Detector Decomposition
**Invariant**: Regime classification must report confidence and per-indicator contributions.
**Metric**: `regime_detector_activation_rate` — % of bars classified as each regime.
**Implementation**: Make `regimeSelector` return per-indicator scores, not just a ternary classification.

---

## 5. Implementation Plan

### Phase 1: Instrumentation (Today)
1. Add `GateTelemetry` data type and JSON serialization
2. Instrument `SignalGates.hs` with per-gate counters
3. Add `--gate-telemetry` CLI flag
4. Run backtest with telemetry, collect histograms

### Phase 2: Calibration (Next)
1. Implement `EdgeDistribution` analysis from backtest history
2. Add percentile-based threshold setting
3. Validate with walk-forward backtests

### Phase 3: Validation (Next)
1. Implement sensitivity analysis framework
2. Require sensitivity report for any threshold change
3. Add CI check: threshold changes must include backtest delta

---

## 6. Test Plan

### Test 1: Gate Telemetry Output
**Given**: A backtest with `--gate-telemetry --json`
**When**: No trades are generated
**Then**: JSON output includes `gateRejections` with non-empty histogram
**And**: The binding gate is identifiable

### Test 2: Threshold Calibration
**Given**: 1000 bars of historical data
**When**: `--calibrate-thresholds-from-history --target-percentile 75`
**Then**: Open threshold equals 75th percentile of admitted historical edges
**And**: Backtest with calibrated threshold produces trades

### Test 3: Spike Detection Conservatism
**Given**: An edge of 10x the historical 99th percentile
**When**: Rolling percentile spike detection is active
**Then**: Entry is rejected as `EDGE_SPIKE`
**And**: Fixed-threshold mode would have allowed it

### Test 4: Regime Decomposition
**Given**: A bar with ADX=18, aroonGap=35, slope=0.008
**When**: Regime selector runs with decomposition
**Then**: Output includes `regimeScores: {trend: 0.7, range: 0.2, neutral: 0.1}`
**And**: Classification is `RegimeTrend` with confidence 0.7

---

## 7. Failure Mode Analysis

### F1: What if gate telemetry slows down backtests?
**Mitigation**: Telemetry is accumulated in mutable counters, only serialized at end. Overhead < 0.1%.

### F2: What if calibrated thresholds are too permissive?
**Mitigation**: Calibration has `--target-percentile` with default 75. Operator can tighten to 90 or 95.

### F3: What if rolling percentile spike detection is too noisy?
**Mitigation**: Use exponential moving average of percentiles with `spike_lookback_bars` parameter. Default 500.

### F4: What if regime decomposition increases false positives?
**Mitigation**: Decomposition is additive — old ternary path remains, decomposition is optional via `--regime-decomposition`.

---

## 8. Measurable Success Criteria

| Metric | Current (Arbitrary) | Target (Engineered) |
|--------|---------------------|---------------------|
| Trades per 1000 bars | Unknown | ≥ 4 with Sharpe > 1.0 |
| Gate rejection observability | None | 100% of gates logged |
| Parameter change justification | None | Sensitivity analysis required |
| Spike detection false negative rate | 100% (disabled) | < 5% |
| Regime classification granularity | 3 classes | Continuous scores |
| Time to diagnose zero trades | Manual code review | < 5 minutes via telemetry |

---

## 9. Conclusion

The 2026-05-11 "fix" is a textbook example of **reactive parameter tweaking without observability**. The correct engineering response to zero trades is:

1. **Observe**: Which gate rejected what and why?
2. **Hypothesize**: Is the binding constraint necessary?
3. **Experiment**: Vary one parameter, measure outcome.
4. **Validate**: Walk-forward test, not in-sample.
5. **Implement**: Change only the validated parameter.
6. **Monitor**: Continue telemetry to detect drift.

Today's commits did #5 without #1-#4. This review implements #1 (telemetry) as the foundation for proper #2-#6 going forward.

---

*Review conducted: 2026-05-11 23:23 America/Guayaquil*
*Reviewer: Engineering Review Agent*
*Next review: 2026-05-12*
