Replace the ROI normalization helpers and verifier block so `roiImplementationScore` and `roiViewFromMetrics` use explicit conservative canonicalizers:

1. Add new constants near the existing ROI helpers:
- `roiMalformedRewardSentinel :: Double = -1.0e6`
- `roiMalformedPenaltySentinel :: Double = 1.0e6`

2. Add new helper functions and use them from both the implementation and spec mirror:
- `normalizeRoiRewardMetric :: Double -> Double`
  - finite -> original value
  - NaN/Infinity -> `roiMalformedRewardSentinel`
- `normalizeRoiPenaltyMetric :: Double -> Double`
  - finite and >= 0 -> original value
  - negative or NaN/Infinity -> `roiMalformedPenaltySentinel`
- `normalizeRoiExposureMetric :: Double -> Double`
  - finite and >= 0 -> original value
  - negative or NaN/Infinity -> `0`
- `normalizeRoiActivityCount :: Int -> Int`
  - `max 0`
- `normalizeRoiPaybackDuration :: Double -> Maybe Double`
  - delegate to `positiveFiniteDuration`

3. Change `roiImplementationScore` to use those helpers instead of the current permissive `sanitizeFinite0` / `max 0` combinations for annualized return, drawdown, tail loss, turnover, expectancy, payback duration, activity counts, and exposure.

4. Change `roiViewFromMetrics`, `activityCountFromMetrics`, and `completedRoundTripsFromMetrics` to use the same canonicalization contract.

5. Extend `FormalVerificationReport` with:
- `fvrMalformedRoiInputsFailClosed :: !Bool`

6. Add malformed domains and a bounded property checker, for example:
- `malformedRewardDomain = nonFiniteDomain`
- `malformedPenaltyDomain = (-0.5) : nonFiniteDomain`
- `malformedExposureDomain = (-0.5) : nonFiniteDomain`
- `malformedActivityDomain = [-2, -1]`
- `malformedRoiInputsFailClosedFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Double -> Bool`
  - build a finite baseline `RoiState`
  - assert that replacing annualized return, drawdown, tail loss, turnover, expectancy, avgHold, roundTrips, tradeCount, or exposure with malformed-domain values never yields a higher score than the baseline

7. Wire the new property into `verifyFormalOptimization`:
- compute `fvrMalformedRoiInputsFailClosed` over the existing bounded ROI domains
- include it in the constructed `FormalVerificationReport`

8. Keep the existing tie-break normalization unchanged; the ROI fail-closed guarantee is now enforced before tie-break selection is even relevant.