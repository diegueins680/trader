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
    AlignedFeatureSeriesV2,
    afsV2Values,
    afsV2Available,
    afsV2EventTimesMs,
    afsV2AvailabilityTimesMs,
    alignToBars,
    alignTimedToBars,
    neutralFill,
    alignedFeatureSeries,
    alignedFeatureSeriesV2,
) where

import Data.Int (Int64)
import Data.List (foldl', sortOn)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, isJust)
import qualified Data.Vector as V

import Trader.Predictors.FeatureSchema (TimedFeatureValue (..))

data AlignedFeatureSeriesV2 = AlignedFeatureSeriesV2
    { afsV2Values :: !(V.Vector Double)
    , afsV2Available :: !(V.Vector Bool)
    , afsV2EventTimesMs :: !(V.Vector (Maybe Int64))
    , afsV2AvailabilityTimesMs :: !(V.Vector (Maybe Int64))
    }
    deriving (Eq, Show)

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
alignToBars barOpenTimes intervalMs series =
    V.map (fmap tfvValue) $
        alignTimedToBars
            barOpenTimes
            intervalMs
            [TimedFeatureValue timestamp timestamp value | (timestamp, value) <- series]

{- | Align observations using availability time while retaining event time.
Revisions become visible only from their own availability timestamp. An
observation whose event time follows its claimed availability time is malformed
and is ignored. Malformed grids fail closed to all 'Nothing'.
-}
alignTimedToBars :: V.Vector Int64 -> Int64 -> [TimedFeatureValue] -> V.Vector (Maybe TimedFeatureValue)
alignTimedToBars barOpenTimes intervalMs series
    | intervalMs <= 0 = neutral
    | not (strictlyAscending barOpenTimes) = neutral
    | otherwise = V.fromList (go ordered Map.empty 0)
  where
    n = V.length barOpenTimes
    neutral = V.replicate n Nothing
    ordered = sortOn observationOrder (filter admissibleObservation series)
    go remaining currentByEvent i
        | i >= n = []
        | otherwise =
            case closeTime (barOpenTimes V.! i) of
                Nothing -> Nothing : go remaining currentByEvent (i + 1)
                Just closeT ->
                    let (usable, rest) = span ((<= closeT) . tfvAvailabilityTimeMs) remaining
                        currentByEvent' = foldl' insertRevision currentByEvent usable
                        latestValue = snd <$> Map.lookupMax currentByEvent'
                     in latestValue : go rest currentByEvent' (i + 1)
    closeTime openTime =
        let closeT = openTime + intervalMs - 1
         in if closeT < openTime then Nothing else Just closeT
    observationOrder observation = (tfvAvailabilityTimeMs observation, tfvEventTimeMs observation)
    insertRevision current observation = Map.insert (tfvEventTimeMs observation) observation current
    admissibleObservation observation =
        let eventTime = tfvEventTimeMs observation
            availabilityTime = tfvAvailabilityTimeMs observation
            value = tfvValue observation
         in eventTime >= 0
                && availabilityTime >= eventTime
                && not (isNaN value || isInfinite value)

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
alignedFeatureSeries barOpenTimes intervalMs series =
    afsV2Values
        <$> alignedFeatureSeriesV2
            barOpenTimes
            intervalMs
            [TimedFeatureValue timestamp timestamp value | (timestamp, value) <- series]

{- | Availability-preserving alignment for the versioned feature schema. The
legacy dense values remain available for parity, while the parallel mask and
timestamps keep an observed zero distinct from unavailable evidence.
-}
alignedFeatureSeriesV2 :: V.Vector Int64 -> Int64 -> [TimedFeatureValue] -> Maybe AlignedFeatureSeriesV2
alignedFeatureSeriesV2 _ _ [] = Nothing
alignedFeatureSeriesV2 barOpenTimes intervalMs series =
    let aligned = alignTimedToBars barOpenTimes intervalMs series
     in if V.any isJust aligned
            then
                Just
                    AlignedFeatureSeriesV2
                        { afsV2Values = V.map (maybe 0 tfvValue) aligned
                        , afsV2Available = V.map isJust aligned
                        , afsV2EventTimesMs = V.map (fmap tfvEventTimeMs) aligned
                        , afsV2AvailabilityTimesMs = V.map (fmap tfvAvailabilityTimeMs) aligned
                        }
            else Nothing
