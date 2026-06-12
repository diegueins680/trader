{- | Execution-cost calibration from realized fills.

The live entry gates and equity accounting price every order with the
configured @--slippage@ assumption. Real fills tell the truth: the average
fill price (@cummulativeQuoteQty / executedQty@) versus the decision-bar
close measures what crossing the book actually cost, including spread and
short-horizon drift between decision and execution. This module turns those
measurements into a calibrated per-side slippage estimate that replaces the
static assumption once enough fills accumulate.

The estimator is deliberately conservative:

* the median of a bounded recent window (robust to fat-tailed fills),
* shrunk toward the configured prior with weight @n / (n + k)@ — the same
  shrinkage pattern used for combo live-performance blending,
* floored at a fraction of the configured prior (good fills may relax the
  gates only so far; underestimating costs is the dangerous direction),
* capped absolutely so a run of terrible fills tightens gates rather than
  producing a nonsensical cost model.

The fee rate itself (@--fee@) is contractual and stays configured; only the
uncertain price-impact component is calibrated.
-}
module Trader.CostCalibration (
    calibratedSlippagePerSide,
    costCalibrationFloorFactor,
    costCalibrationMaxPerSide,
    costCalibrationMinObservations,
    costCalibrationOutlierBound,
    costCalibrationShrinkageObs,
    costCalibrationWindow,
    observedSlippageFraction,
) where

import Data.List (sort)

-- | Fills required before the calibrated estimate replaces the configured
-- assumption (8 fills = roughly 4 round trips).
costCalibrationMinObservations :: Int
costCalibrationMinObservations = 8

-- | Pseudo-count at which realized evidence carries the same weight as the
-- configured prior (w = n / (n + k)).
costCalibrationShrinkageObs :: Double
costCalibrationShrinkageObs = 16

-- | Only the most recent fills inform the estimate, so a venue's changing
-- liquidity is tracked rather than averaged away.
costCalibrationWindow :: Int
costCalibrationWindow = 64

-- | The calibrated estimate never drops below this fraction of the
-- configured slippage: price improvement may relax gates, but not gut them.
costCalibrationFloorFactor :: Double
costCalibrationFloorFactor = 0.25

-- | Absolute per-side ceiling (1%). Worse realized costs than this tighten
-- gates at the cap rather than poisoning the cost model.
costCalibrationMaxPerSide :: Double
costCalibrationMaxPerSide = 0.01

-- | Single-fill measurements beyond this magnitude (5%) are treated as data
-- errors (bad decision price, corrupted fill) and discarded.
costCalibrationOutlierBound :: Double
costCalibrationOutlierBound = 0.05

{- | Realized per-side slippage fraction of one fill: positive when the fill
was worse than the decision price (paying up on a BUY, getting hit lower on
a SELL), negative on price improvement. Nothing when the order didn't
actually fill, any input is non-positive or non-finite, the side is
unrecognized, or the measurement exceeds 'costCalibrationOutlierBound'.
-}
observedSlippageFraction :: String -> Double -> Maybe Double -> Maybe Double -> Maybe Double
observedSlippageFraction side decisionPrice mExecutedQty mCumQuote = do
    direction <-
        case side of
            "BUY" -> Just 1
            "SELL" -> Just (-1)
            _ -> Nothing
    decision <- positiveFinite decisionPrice
    qty <- mExecutedQty >>= positiveFinite
    quote <- mCumQuote >>= positiveFinite
    let avgFill = quote / qty
        slip = direction * (avgFill - decision) / decision
    if isNaN slip || isInfinite slip || abs slip > costCalibrationOutlierBound
        then Nothing
        else Just slip

{- | Blend the configured per-side slippage with realized fill evidence
(observations oldest-first). Below 'costCalibrationMinObservations' fills
the configured value passes through untouched, so paper bots and fresh
sessions behave exactly as before.
-}
calibratedSlippagePerSide :: Double -> [Double] -> Double
calibratedSlippagePerSide configured observations =
    let prior =
            if isNaN configured || isInfinite configured || configured < 0
                then 0
                else configured
        recent = takeLastObs costCalibrationWindow (filter finiteObs observations)
        n = length recent
     in if n < costCalibrationMinObservations
            then prior
            else
                let est = median recent
                    w = fromIntegral n / (fromIntegral n + costCalibrationShrinkageObs)
                    blended = w * est + (1 - w) * prior
                    floorVal = costCalibrationFloorFactor * prior
                 in min costCalibrationMaxPerSide (max floorVal blended)
  where
    finiteObs x = not (isNaN x || isInfinite x)

    takeLastObs k xs =
        let len = length xs
         in if len <= k then xs else drop (len - k) xs

    median xs =
        let sorted = sort xs
            len = length sorted
            mid = len `div` 2
         in if len == 0
                then 0
                else
                    if odd len
                        then sorted !! mid
                        else ((sorted !! (mid - 1)) + (sorted !! mid)) / 2

positiveFinite :: Double -> Maybe Double
positiveFinite x
    | isNaN x || isInfinite x || x <= 0 = Nothing
    | otherwise = Just x
