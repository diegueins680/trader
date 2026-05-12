# Regime Detector Engineering Analysis
## Date: 2026-05-11

## Problem Statement

The current `regimeSelector` in `Trader.TechnicalAnalysis.Strategies` uses a **compound AND condition** that is too restrictive:

```haskell
if adxValue adxNow >= 25 && aroonGap >= 40 && abs slope >= 0.01
    then Just RegimeTrend
```

This means ALL THREE conditions must be met simultaneously for a trend classification. The commit `dce28b48` "fixed" zero trades by arbitrarily lowering these thresholds:
- ADX 25 → 15 (-40%)
- aroonGap 40 → 20 (-50%)
- slope 0.01 → 0.005 (-50%)

## Engineering Analysis

### Why Compound AND Gates Fail

For independent conditions with probabilities p1, p2, p3, the joint probability is:
```
P(trend) = P(ADX) * P(aroonGap) * P(slope)
```

If each condition has 50% probability individually:
```
P(trend) = 0.5 * 0.5 * 0.5 = 0.125 (12.5%)
```

**This means 87.5% of bars are classified as non-trend**, even when the market IS trending by some measures.

### The Correct Approach: Weighted Scoring

Instead of compound AND, use a **weighted scoring system**:

```haskell
scoreTrend = w1 * normADX + w2 * normAroon + w3 * normSlope
where
    normADX = clamp01 ((adx - 10) / 20)      -- 0 at ADX=10, 1 at ADX=30
    normAroon = clamp01 ((aroonGap - 10) / 50) -- 0 at gap=10, 1 at gap=60
    normSlope = clamp01 ((abs slope - 0.002) / 0.02) -- 0 at 0.2%, 1 at 2%
```

Then classify:
```haskell
if scoreTrend >= 0.6 then RegimeTrend
else if scoreTrend <= 0.2 then RegimeRange
else RegimeNeutral
```

This approach:
1. **Decomposes** the compound gate into interpretable components
2. **Weights** can be calibrated from historical data
3. **Continuous scores** provide confidence levels
4. **No arbitrary thresholds** — the 0.6 boundary is calibrated

### Decomposed Regime Detector (Proposed Implementation)

```haskell
data RegimeScore = RegimeScore
    { rsTrendScore :: !Double    -- ^ 0-1, higher = more trend-like
    , rsRangeScore :: !Double    -- ^ 0-1, higher = more range-like
    , rsConfidence :: !Double    -- ^ 0-1, confidence in classification
    }

regimeSelectorDecomposed :: OhlcvSeries -> Maybe (Regime, RegimeScore)
regimeSelectorDecomposed series = do
    closeNow <- lastValue (ohlcvClose series)
    adxNow <- latestJust (adxSeries 14 ...)
    aroonNow <- latestJust (aroonSeries 25 ...)
    fastNow <- latestJust (emaSeries 20 ...)
    fastPrev <- laggedJust 5 (emaSeries 20 ...)
    bbNow <- latestJust (bollingerBandsSeries 20 2 ...)
    
    let slope = safeDivide (fastNow - fastPrev) closeNow
        width = safeDivide (bandUpper bbNow - bandLower bbNow) closeNow
        aroonGap = abs (aroonUp aroonNow - aroonDown aroonNow)
        
        -- Component scores (0-1)
        adxScore = clamp01 ((adxValue adxNow - 10) / 20)
        aroonScore = clamp01 ((aroonGap - 10) / 50)
        slopeScore = clamp01 ((abs slope - 0.002) / 0.018)
        widthScore = clamp01 ((0.08 - width) / 0.08)  -- Higher = narrower = more range
        
        -- Weighted trend score
        trendScore = (0.4 * adxScore + 0.3 * aroonScore + 0.3 * slopeScore)
        
        -- Range score
        rangeScore = widthScore * (1 - adxScore) * (1 - slopeScore)
        
        -- Classification with confidence
        (regime, confidence)
            | trendScore >= 0.6 = (RegimeTrend, trendScore)
            | rangeScore >= 0.6 = (RegimeRange, rangeScore)
            | otherwise = (RegimeNeutral, 1 - max trendScore rangeScore)
    
    pure (regime, RegimeScore trendScore rangeScore confidence)
```

### Calibration Process

1. **Collect historical regime labels** (or use implicit labels from profitable trades)
2. **Grid search weights** to maximize trade count while maintaining Sharpe > 1.0
3. **Validate** with walk-forward analysis
4. **Monitor** regime activation rates — should be:
   - Trend: 30-50% of bars in trending markets
   - Range: 20-40% in ranging markets
   - Neutral: 10-30% (transitions)

### Metrics to Track

| Metric | Target | Current (Old) | Current (Relaxed) |
|--------|--------|---------------|-------------------|
| Trend activation rate | 30-50% | ~5% | ~25% (estimated) |
| Range activation rate | 20-40% | ~15% | ~30% (estimated) |
| Neutral rate | 10-30% | ~80% | ~45% (estimated) |
| Trade count (4h BTC) | ≥ 4/month | 0 | ~4 |
| Sharpe (4h BTC) | > 1.0 | N/A | > 1.0 (claimed) |

### Failure Modes

**F1: Overfitting Weights**
- Condition: Grid search overfits to historical data
- Mitigation: Use walk-forward optimization with 3+ folds

**F2: Regime Drift**
- Condition: Market structure changes, old weights become invalid
- Mitigation: Recalibrate weights monthly using rolling window

**F3: Confidence Inflation**
- Condition: Neutral regime confidence is always high
- Mitigation: Require confidence > 0.5 for any regime classification

## Implementation Plan

### Phase 1: Add Decomposed Detector (Parallel)
- Implement `regimeSelectorDecomposed` alongside existing `regimeSelector`
- Add `--regime-decomposed` CLI flag
- Default to old detector for backward compatibility

### Phase 2: Calibrate Weights
- Run optimizer with weight grid search
- Target: maximize trades with Sharpe > 1.0
- Output: calibrated weight vector

### Phase 3: A/B Test
- Run both detectors in parallel for 2 weeks
- Compare: trade count, Sharpe, max drawdown
- Switch default to decomposed if superior

### Phase 4: Remove Legacy
- Remove old compound-AND detector
- Remove `--regime-decomposed` flag (always on)

## Conclusion

The arbitrary threshold relaxation on 2026-05-11 is a **symptom treatment**, not a **root cause fix**. The root cause is the compound-AND regime detector that requires ALL conditions to align perfectly.

The correct engineering response is to:
1. Decompose the detector into weighted components
2. Calibrate weights from historical data
3. Validate with walk-forward testing
4. Maintain telemetry on regime activation rates

This preserves the engineering invariant: **"No parameter change without data-driven justification."**
