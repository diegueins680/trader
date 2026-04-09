Keep the existing module header, exports, imports, tie-break normalization, and unrelated proof machinery unchanged.

Replace the ROI normalization helpers and verifier block so `roiImplementationScore` and `roiViewFromMetrics` use explicit conservative canonicalizers:

roiMalformedRewardSentinel :: Double
roiMalformedRewardSentinel = -1.0e6

roiMalformedPenaltySentinel :: Double
roiMalformedPenaltySentinel = 1.0e6

normalizeRoiRewardMetric :: Double -> Double
normalizeRoiRewardMetric x
    | isNaN x || isInfinite x = roiMalformedRewardSentinel
    | otherwise = x

normalizeRoiPenaltyMetric :: Double -> Double
normalizeRoiPenaltyMetric x
    | isNaN x || isInfinite x = roiMalformedPenaltySentinel
    | x < 0 = roiMalformedPenaltySentinel
    | otherwise = x

normalizeRoiExposureMetric :: Double -> Double
normalizeRoiExposureMetric x
    | isNaN x || isInfinite x = 0
    | x < 0 = 0
    | otherwise = x

normalizeRoiActivityCount :: Int -> Int
normalizeRoiActivityCount = max 0

normalizeRoiPaybackDuration :: Double -> Maybe Double
normalizeRoiPaybackDuration = positiveFiniteDuration

Update `roiImplementationScore` so annualized return and expectancy go through `normalizeRoiRewardMetric`; drawdown, tail loss, turnover, and any penalty-only quantity go through `normalizeRoiPenaltyMetric`; activity counts go through `normalizeRoiActivityCount`; payback duration goes through `normalizeRoiPaybackDuration`; and exposure goes through `normalizeRoiExposureMetric`.

Update `roiViewFromMetrics`, `activityCountFromMetrics`, and `completedRoundTripsFromMetrics` to use the same canonicalization contract so the implementation and spec mirror agree.

Extend `FormalVerificationReport` with:
    , fvrMalformedRoiInputsFailClosed :: !Bool

Add malformed domains:
malformedRewardDomain = nonFiniteDomain
malformedPenaltyDomain = (-0.5) : nonFiniteDomain
malformedExposureDomain = (-0.5) : nonFiniteDomain
malformedActivityDomain = [-2, -1]

Add `malformedRoiInputsFailClosedFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Double -> Bool` that builds a finite baseline `RoiState` and proves replacing annualized return, drawdown, tail loss, turnover, expectancy, avgHold, roundTrips, tradeCount, or exposure with malformed-domain values never yields a higher ROI score than the baseline.

Wire the new property into `verifyFormalOptimization` by computing `fvrMalformedRoiInputsFailClosed` over the existing bounded ROI domains and including it in the constructed `FormalVerificationReport`.

Do not change the existing tie-break normalization logic.