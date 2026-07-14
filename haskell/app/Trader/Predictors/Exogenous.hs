{- | Point-in-time alignment of irregular exogenous series (funding rate, open
interest, basis, taker flow) onto a bar grid, for use as 'FeatureInputs'
derivatives series.

The whole point of this module is to make look-ahead leakage structurally
impossible: 'alignToBars' forward-fills using only observations whose timestamp
is at or before a bar's /close/, so no future observation can ever influence an
earlier bar. This is exactly the kind of join that silently fabricates
in-sample edge when done carelessly, so it is isolated here and unit-tested.
-}
module Trader.Predictors.Exogenous (
    alignToBars,
    neutralFill,
    alignedFeatureSeries,
) where

import Data.Int (Int64)
import Data.List (sortOn)
import Data.Maybe (fromMaybe, isJust)
import qualified Data.Vector as V

{- | Forward-fill an irregular @(timestampMs, value)@ series onto @barOpenTimes@,
point-in-time. The value at bar @i@ is the most recent finite observation whose
timestamp is @<=@ that bar's close time (@openTime_i + intervalMs - 1@). Bars
with no prior admissible observation get 'Nothing'.

Assumes @barOpenTimes@ is ascending (closed klines already are); the series may
be in any order and is sorted internally. Malformed bar grids and non-positive
intervals fail closed to all 'Nothing' values rather than emitting
time-incoherent features. Runs in O(n + m log m).
-}
alignToBars :: V.Vector Int64 -> Int64 -> [(Int64, Double)] -> V.Vector (Maybe Double)
alignToBars barOpenTimes intervalMs series
    | intervalMs <= 0 = neutral
    | not (strictlyAscending barOpenTimes) = neutral
    | otherwise = V.fromList (go (sortOn fst (filter finiteObservation series)) Nothing 0)
  where
    n = V.length barOpenTimes
    neutral = V.replicate n Nothing
    go remaining lastV i
        | i >= n = []
        | otherwise =
            case closeTime (barOpenTimes V.! i) of
                Nothing -> Nothing : go remaining lastV (i + 1)
                Just closeT ->
                    let (usable, rest) = span ((<= closeT) . fst) remaining
                        lastV' = case usable of
                            [] -> lastV
                            _ -> Just (snd (last usable))
                     in lastV' : go rest lastV' (i + 1)
    closeTime openTime =
        let closeT = openTime + intervalMs - 1
         in if closeT < openTime then Nothing else Just closeT
    finiteObservation (_, value) = not (isNaN value || isInfinite value)

strictlyAscending :: V.Vector Int64 -> Bool
strictlyAscending xs =
    V.length xs <= 1 || V.and (V.zipWith (<) xs (V.tail xs))

{- | Replace pre-coverage 'Nothing' bars with a neutral 0. Used when packing an
aligned series into a 'FeatureInputs' field (whose features are deltas/centered
levels, for which 0 reads as "no information").
-}
neutralFill :: V.Vector (Maybe Double) -> V.Vector Double
neutralFill = V.map (fromMaybe 0)

{- | Convenience: align a raw series to bars and pack it for 'FeatureInputs'.
Returns 'Nothing' when the series is empty or has no observation admissible on
the bar grid (so the feature stays fully neutral), otherwise @Just@ a dense,
point-in-time, neutral-filled vector aligned 1:1 with the bar grid.
-}
alignedFeatureSeries :: V.Vector Int64 -> Int64 -> [(Int64, Double)] -> Maybe (V.Vector Double)
alignedFeatureSeries _ _ [] = Nothing
alignedFeatureSeries barOpenTimes intervalMs series =
    let aligned = alignToBars barOpenTimes intervalMs series
     in if V.any isJust aligned
            then Just (neutralFill aligned)
            else Nothing
