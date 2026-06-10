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
import Data.Maybe (fromMaybe)
import qualified Data.Vector as V

{- | Forward-fill an irregular @(timestampMs, value)@ series onto @barOpenTimes@,
point-in-time. The value at bar @i@ is the most recent observation whose
timestamp is @<=@ that bar's close time (@openTime_i + intervalMs - 1@). Bars
with no prior observation get 'Nothing'.

Assumes @barOpenTimes@ is ascending (closed klines already are); the series may
be in any order and is sorted internally. Runs in O(n + m log m).
-}
alignToBars :: V.Vector Int64 -> Int64 -> [(Int64, Double)] -> V.Vector (Maybe Double)
alignToBars barOpenTimes intervalMs series =
    V.fromList (go (sortOn fst series) Nothing 0)
  where
    n = V.length barOpenTimes
    go remaining lastV i
        | i >= n = []
        | otherwise =
            let closeT = (barOpenTimes V.! i) + intervalMs - 1
                (usable, rest) = span ((<= closeT) . fst) remaining
                lastV' = case usable of
                    [] -> lastV
                    _ -> Just (snd (last usable))
             in lastV' : go rest lastV' (i + 1)

{- | Replace pre-coverage 'Nothing' bars with a neutral 0. Used when packing an
aligned series into a 'FeatureInputs' field (whose features are deltas/centered
levels, for which 0 reads as "no information").
-}
neutralFill :: V.Vector (Maybe Double) -> V.Vector Double
neutralFill = V.map (fromMaybe 0)

{- | Convenience: align a raw series to bars and pack it for 'FeatureInputs'.
Returns 'Nothing' when the series is empty (so the feature stays fully neutral),
otherwise @Just@ a dense, point-in-time, neutral-filled vector aligned 1:1 with
the bar grid.
-}
alignedFeatureSeries :: V.Vector Int64 -> Int64 -> [(Int64, Double)] -> Maybe (V.Vector Double)
alignedFeatureSeries _ _ [] = Nothing
alignedFeatureSeries barOpenTimes intervalMs series =
    Just (neutralFill (alignToBars barOpenTimes intervalMs series))
