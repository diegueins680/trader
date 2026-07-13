# Formal Methods: Threshold Calibration

## Module
`Trader.ThresholdCalibration`

## Purpose
Calibrate trading thresholds from historical edge distributions instead of magic numbers.

## Invariants

### I1: Empty Input Fails Closed
**Statement**: `computeEdgeDistribution [] = Nothing`
**Proof**: Pattern match on empty list returns `Nothing`. No threshold can be computed without data.

### I2: Invalid Edges Are Rejected
**Statement**: For any edge `e` where `e < 0` or `isNaN e` or `isInfinite e`, `computeEdgeDistribution` returns `Nothing`.
**Proof**: The `invalidEdge` predicate checks all three conditions. If ANY edge is invalid, the entire distribution is rejected.

### I3: Percentile is Monotonic
**Statement**: For all `p1 < p2`, `thresholdAtPercentile p1 dist <= thresholdAtPercentile p2 dist`.
**Proof**: `computeEdgeDistribution` derives each stored percentile anchor from one sorted edge sample, so the anchor values are non-decreasing. `thresholdAtPercentile` clamps outside `[0,100]` and linearly interpolates between adjacent anchors, preserving monotonicity between anchors and at anchor boundaries.

### I4: Headroom Threshold is Proportional
**Statement**: `tcHeadroomThreshold = tcSuggestedThreshold / tccHeadroomDivisor` for a validated configuration; the default divisor is `1.5`.
**Proof**: `validateThresholdCalibrationConfig` requires a finite positive divisor and `calibrateThresholdWithConfig` divides by that validated value.

### I5: Confidence Interval is Valid
**Statement**: `fst tcConfidenceInterval <= snd tcConfidenceInterval`
**Proof**: `ciLower = threshold - 1.96 * se`, `ciUpper = threshold + 1.96 * se`. Since `se >= 0`, `ciLower <= ciUpper`.

### I6: Recommendation is Categorized
**Statement**: The recommendation is exactly one of: INSUFFICIENT_SAMPLE, CONSERVATIVE, AGGRESSIVE, BALANCED.
**Proof**: The `rec` computation uses a cascade of `if-then-else` with mutually exclusive conditions based on sample size and percentile position.

### I7: Calibration Method Is Admissible
**Statement**: percentile inputs are finite and in `[0,100]`; standard-deviation multipliers are finite and non-negative. Invalid methods make `calibrateThresholdWithConfig` return `Nothing`.
**Proof**: `validateCalibrationMethod` runs before threshold construction, and the output boundary rejects any non-finite derived threshold or confidence interval.

## Failure Modes

### F1: Insufficient Sample Size
**Condition**: `edSampleSize < 100`
**Result**: Recommendation = "INSUFFICIENT_SAMPLE"
**Action**: Collect more historical data before calibrating.

### F2: Zero Standard Deviation
**Condition**: All edges are identical
**Result**: `edStdDev = 0`, `StdDevMethod` returns mean
**Action**: Use `PercentileMethod` or `HybridMethod` instead.

### F3: Outlier Domination
**Condition**: Edge distribution has extreme outliers
**Result**: `StdDevMethod` produces very high thresholds
**Mitigation**: Prefer `PercentileMethod` or pre-declared robust/winsorized input handling. `HybridMethod` is the conservative maximum of percentile and standard-deviation thresholds, so it deliberately does not cap an outlier-inflated standard-deviation result.

## Metrics
- `sample_size`: Number of edges used
- `suggested_threshold`: Calibrated open threshold
- `confidence_interval`: 95% CI for threshold
- `recommendation`: Actionable classification

## Validation
See `test/TestMain.hs`:
- `testThresholdCalibrationEmptyInputFailsClosed`
- `testThresholdCalibrationDistributionAccuracy`
- `testThresholdCalibrationPercentileMethod`
- `testThresholdCalibrationStdDevMethod`
- `testThresholdCalibrationHybridMethod`
- `testThresholdCalibrationRecommendationInsufficientSample`
- `testThresholdCalibrationRecommendationConservative`
- `testThresholdCalibrationRecommendationAggressive`
- `testThresholdCalibrationRecommendationBalanced`
