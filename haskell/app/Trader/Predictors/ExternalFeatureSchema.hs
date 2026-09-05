module Trader.Predictors.ExternalFeatureSchema (
    ExternalFeature (..),
    ExternalObservationV2 (..),
    ExternalFeatureInputsV2,
    externalFeatureFamilies,
    externalFeatureColumnName,
    alignedExternalFeatureInputsV2,
    externalFeatureSeriesV2,
) where

import Data.Int (Int64)
import Data.List (foldl')
import qualified Data.Map.Strict as Map
import Data.Maybe (mapMaybe)
import qualified Data.Vector as V

import Trader.Predictors.Exogenous (AlignedFeatureSeriesV2, alignedFeatureSeriesV2)
import Trader.Predictors.FeatureSchema (TimedFeatureValue (..))

data ExternalFeature
    = ExternalMicrostructure
    | ExternalOptionsVol
    | ExternalOnChain
    | ExternalMacro
    | ExternalCot
    | ExternalNews
    | ExternalFilings
    | ExternalPolicy
    | ExternalFundamentals
    | ExternalStablecoin
    | ExternalInstitutionalFlows
    | ExternalNetwork
    | ExternalDeveloper
    | ExternalGovernance
    | ExternalAttention
    | ExternalSocial
    | ExternalPredictionMarket
    | ExternalRealWorld
    | ExternalSecurity
    deriving (Eq, Ord, Show)

data ExternalObservationV2 = ExternalObservationV2
    { eov2Feature :: !ExternalFeature
    , eov2EventTimeMs :: !Int64
    , eov2AvailabilityTimeMs :: !Int64
    , eov2Value :: !Double
    }
    deriving (Eq, Show)

newtype ExternalFeatureInputsV2 = ExternalFeatureInputsV2
    { unExternalFeatureInputsV2 :: Map.Map ExternalFeature AlignedFeatureSeriesV2
    }
    deriving (Eq, Show)

externalFeatureFamilies :: [ExternalFeature]
externalFeatureFamilies =
    [ ExternalMicrostructure
    , ExternalOptionsVol
    , ExternalOnChain
    , ExternalMacro
    , ExternalCot
    , ExternalNews
    , ExternalFilings
    , ExternalPolicy
    , ExternalFundamentals
    , ExternalStablecoin
    , ExternalInstitutionalFlows
    , ExternalNetwork
    , ExternalDeveloper
    , ExternalGovernance
    , ExternalAttention
    , ExternalSocial
    , ExternalPredictionMarket
    , ExternalRealWorld
    , ExternalSecurity
    ]

externalFeatureColumnName :: ExternalFeature -> String
externalFeatureColumnName feature =
    case feature of
        ExternalMicrostructure -> "microstructure"
        ExternalOptionsVol -> "options_vol"
        ExternalOnChain -> "onchain"
        ExternalMacro -> "macro"
        ExternalCot -> "cot"
        ExternalNews -> "news"
        ExternalFilings -> "filings"
        ExternalPolicy -> "policy"
        ExternalFundamentals -> "fundamentals"
        ExternalStablecoin -> "stablecoin"
        ExternalInstitutionalFlows -> "institutional_flows"
        ExternalNetwork -> "network"
        ExternalDeveloper -> "developer"
        ExternalGovernance -> "governance"
        ExternalAttention -> "attention"
        ExternalSocial -> "social"
        ExternalPredictionMarket -> "prediction_market"
        ExternalRealWorld -> "real_world"
        ExternalSecurity -> "security"

{- | Align external observations without collapsing event time, availability
time, or pre-coverage missingness. Exact duplicate releases are averaged to
preserve the legacy external-family aggregation rule. Revisions with a later
availability time remain distinct and become visible only after release.
-}
alignedExternalFeatureInputsV2 ::
    V.Vector Int64 ->
    Int64 ->
    [ExternalObservationV2] ->
    Maybe ExternalFeatureInputsV2
alignedExternalFeatureInputsV2 barOpenTimes intervalMs observations =
    let grouped = foldl' insertObservation Map.empty observations
        aligned =
            Map.fromList
                ( mapMaybe
                    (alignFamily grouped)
                    externalFeatureFamilies
                )
     in if Map.null aligned
            then Nothing
            else Just (ExternalFeatureInputsV2 aligned)
  where
    insertObservation grouped observation
        | not (finite (eov2Value observation)) = grouped
        | otherwise =
            Map.insertWith
                (Map.unionWith mergeBuckets)
                (eov2Feature observation)
                ( Map.singleton
                    (eov2EventTimeMs observation, eov2AvailabilityTimeMs observation)
                    (eov2Value observation, 1 :: Int)
                )
                grouped
    alignFamily grouped feature = do
        releases <- Map.lookup feature grouped
        let series =
                [ TimedFeatureValue eventTime availabilityTime (total / fromIntegral count)
                | ((eventTime, availabilityTime), (total, count)) <- Map.toAscList releases
                , count > 0
                ]
        aligned <- alignedFeatureSeriesV2 barOpenTimes intervalMs series
        pure (feature, aligned)
    mergeBuckets (aSum, aCount) (bSum, bCount) = (aSum + bSum, aCount + bCount)

externalFeatureSeriesV2 :: ExternalFeature -> ExternalFeatureInputsV2 -> Maybe AlignedFeatureSeriesV2
externalFeatureSeriesV2 feature = Map.lookup feature . unExternalFeatureInputsV2

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
