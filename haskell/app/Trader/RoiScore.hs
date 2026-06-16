module Trader.RoiScore (
    RoiScoreConfig (..),
    RoiScoreInputs (..),
    defaultFormalRoiScoreConfig,
    defaultRoiScoreConfig,
    sanitizeRoiScoreConfig,
    roiEvidencePenaltyWithConfig,
    roiScoreFromInputsWithConfig,
    meetsRoiExpectancyFloor,
    meetsRoiExpectancyFloorWithConfig,
    meetsRoiPaybackFloor,
    meetsRoiPaybackFloorWithConfig,
    expectancyRewardFor,
    expectancyRewardForWithConfig,
    paybackRewardFor,
    paybackRewardForWithConfig,
    paybackBonusFor,
    paybackBonusForWithConfig,
    minimumRoiActivityFloor,
    minimumRoiExposureFloor,
    roundTripPenaltyFor,
    roundTripPenaltyForWithConfig,
    activityPenaltyFor,
    activityPenaltyForWithConfig,
    exposurePenaltyFor,
    exposurePenaltyForWithConfig,
) where

import Trader.Duration (positiveFiniteDuration)

data RoiScoreConfig = RoiScoreConfig
    { rscExpectancyRewardWeight :: !Double
    , rscPaybackRewardCap :: !Double
    , rscMinimumActivityFloor :: !Int
    , rscMinimumExposureFloor :: !Double
    , rscZeroRoundTripPenalty :: !Double
    , rscLowRoundTripPenalty :: !Double
    , rscZeroActivityPenalty :: !Double
    , rscLowActivityPenalty :: !Double
    , rscZeroExposurePenalty :: !Double
    , rscLowExposurePenaltyBase :: !Double
    , rscLowExposurePenaltyGapScale :: !Double
    }
    deriving (Eq, Show)

data RoiScoreInputs = RoiScoreInputs
    { rsiAnnualizedReturn :: !Double
    , rsiMaxDrawdown :: !Double
    , rsiTailLoss :: !Double
    , rsiTurnover :: !Double
    , rsiExpectancy :: !Double
    , rsiPaybackDuration :: !(Maybe Double)
    , rsiRoundTrips :: !Int
    , rsiTradeCount :: !Int
    , rsiExposure :: !Double
    }
    deriving (Eq, Show)

defaultRoiScoreConfig :: RoiScoreConfig
defaultRoiScoreConfig =
    RoiScoreConfig
        { rscExpectancyRewardWeight = 0.5
        , rscPaybackRewardCap = 0.05
        , rscMinimumActivityFloor = 3
        , rscMinimumExposureFloor = 0.01
        , rscZeroRoundTripPenalty = 0
        , rscLowRoundTripPenalty = 0
        , rscZeroActivityPenalty = 0.25
        , rscLowActivityPenalty = 0.03
        , rscZeroExposurePenalty = 0.05
        , rscLowExposurePenaltyBase = 0.02
        , rscLowExposurePenaltyGapScale = 0
        }

defaultFormalRoiScoreConfig :: RoiScoreConfig
defaultFormalRoiScoreConfig =
    defaultRoiScoreConfig
        { rscZeroRoundTripPenalty = 0.08
        , rscLowRoundTripPenalty = 0.02
        , rscLowExposurePenaltyGapScale = 0.03
        }

sanitizeRoiScoreConfig :: RoiScoreConfig -> RoiScoreConfig
sanitizeRoiScoreConfig cfg =
    cfg
        { rscExpectancyRewardWeight = finiteNonNegative (rscExpectancyRewardWeight cfg)
        , rscPaybackRewardCap = finiteNonNegative (rscPaybackRewardCap cfg)
        , rscMinimumActivityFloor = max 0 (rscMinimumActivityFloor cfg)
        , rscMinimumExposureFloor = finiteNonNegative (rscMinimumExposureFloor cfg)
        , rscZeroRoundTripPenalty = finiteNonNegative (rscZeroRoundTripPenalty cfg)
        , rscLowRoundTripPenalty = finiteNonNegative (rscLowRoundTripPenalty cfg)
        , rscZeroActivityPenalty = finiteNonNegative (rscZeroActivityPenalty cfg)
        , rscLowActivityPenalty = finiteNonNegative (rscLowActivityPenalty cfg)
        , rscZeroExposurePenalty = finiteNonNegative (rscZeroExposurePenalty cfg)
        , rscLowExposurePenaltyBase = finiteNonNegative (rscLowExposurePenaltyBase cfg)
        , rscLowExposurePenaltyGapScale = finiteNonNegative (rscLowExposurePenaltyGapScale cfg)
        }

roiScoreFromInputsWithConfig :: RoiScoreConfig -> Double -> Double -> RoiScoreInputs -> Double
roiScoreFromInputsWithConfig cfg0 penaltyMaxDd penaltyTurnover inputs =
    let cfg = sanitizeRoiScoreConfig cfg0
        annRet = sanitizeFinite0 (rsiAnnualizedReturn inputs)
        maxDd = max 0 (sanitizeFinite0 (rsiMaxDrawdown inputs))
        tailLoss = max 0 (sanitizeFinite0 (rsiTailLoss inputs))
        turnover = max 0 (sanitizeFinite0 (rsiTurnover inputs))
        expectancy = sanitizeFinite0 (rsiExpectancy inputs)
        completedRoundTrips = max 0 (rsiRoundTrips inputs)
        activityCount = max completedRoundTrips (max 0 (rsiTradeCount inputs))
        exposure = max 0 (sanitizeFinite0 (rsiExposure inputs))
        pDd = max 0 (sanitizeFinite0 penaltyMaxDd)
        pTurn = max 0 (sanitizeFinite0 penaltyTurnover)
        expectancyReward = expectancyRewardForWithConfig cfg completedRoundTrips exposure expectancy
        paybackReward = paybackRewardForWithConfig cfg completedRoundTrips expectancy exposure (rsiPaybackDuration inputs)
        sparseEvidencePenalty = roiEvidencePenaltyWithConfig cfg completedRoundTrips activityCount exposure
     in annRet
            - pDd * (maxDd + tailLoss)
            - pTurn * turnover
            + expectancyReward
            + paybackReward
            - sparseEvidencePenalty

roiEvidencePenaltyWithConfig :: RoiScoreConfig -> Int -> Int -> Double -> Double
roiEvidencePenaltyWithConfig cfg0 completedRoundTrips activityCount exposure =
    let cfg = sanitizeRoiScoreConfig cfg0
     in activityPenaltyForWithConfig cfg activityCount
            + roundTripPenaltyForWithConfig cfg completedRoundTrips
            + exposurePenaltyForWithConfig cfg exposure

minimumRoiActivityFloor :: Int
minimumRoiActivityFloor = rscMinimumActivityFloor defaultFormalRoiScoreConfig

minimumRoiExposureFloor :: Double
minimumRoiExposureFloor = rscMinimumExposureFloor defaultFormalRoiScoreConfig

meetsRoiExpectancyFloor :: Int -> Double -> Bool
meetsRoiExpectancyFloor = meetsRoiExpectancyFloorWithConfig defaultFormalRoiScoreConfig

meetsRoiExpectancyFloorWithConfig :: RoiScoreConfig -> Int -> Double -> Bool
meetsRoiExpectancyFloorWithConfig cfg0 completedRoundTrips exposure =
    let cfg = sanitizeRoiScoreConfig cfg0
     in completedRoundTrips > 0 && exposure >= rscMinimumExposureFloor cfg

expectancyRewardFor :: Int -> Double -> Double -> Double
expectancyRewardFor = expectancyRewardForWithConfig defaultFormalRoiScoreConfig

expectancyRewardForWithConfig :: RoiScoreConfig -> Int -> Double -> Double -> Double
expectancyRewardForWithConfig cfg0 completedRoundTrips exposure expectancy
    | expectancy <= 0 = weight * expectancy
    | not (meetsRoiExpectancyFloorWithConfig cfg completedRoundTrips exposure) = 0
    | otherwise = weight * expectancy
  where
    cfg = sanitizeRoiScoreConfig cfg0
    weight = rscExpectancyRewardWeight cfg

paybackRewardFor :: Int -> Double -> Double -> Maybe Double -> Double
paybackRewardFor = paybackRewardForWithConfig defaultFormalRoiScoreConfig

paybackRewardForWithConfig :: RoiScoreConfig -> Int -> Double -> Double -> Maybe Double -> Double
paybackRewardForWithConfig cfg0 completedRoundTrips expectancy exposure mAvgHold =
    let cfg = sanitizeRoiScoreConfig cfg0
     in if not (meetsRoiPaybackFloorWithConfig cfg completedRoundTrips exposure) || expectancy <= 0
            then 0
            else maybe 0 (paybackBonusForWithConfig cfg) (mAvgHold >>= positiveFiniteDuration)

paybackBonusFor :: Double -> Double
paybackBonusFor = paybackBonusForWithConfig defaultFormalRoiScoreConfig

paybackBonusForWithConfig :: RoiScoreConfig -> Double -> Double
paybackBonusForWithConfig cfg0 avgHold =
    let cfg = sanitizeRoiScoreConfig cfg0
     in min (rscPaybackRewardCap cfg) (1 / (1 + avgHold))

meetsRoiPaybackFloor :: Int -> Double -> Bool
meetsRoiPaybackFloor = meetsRoiPaybackFloorWithConfig defaultFormalRoiScoreConfig

meetsRoiPaybackFloorWithConfig :: RoiScoreConfig -> Int -> Double -> Bool
meetsRoiPaybackFloorWithConfig cfg0 completedRoundTrips exposure =
    let cfg = sanitizeRoiScoreConfig cfg0
     in completedRoundTrips >= rscMinimumActivityFloor cfg && exposure >= rscMinimumExposureFloor cfg

roundTripPenaltyFor :: Int -> Double
roundTripPenaltyFor = roundTripPenaltyForWithConfig defaultFormalRoiScoreConfig

roundTripPenaltyForWithConfig :: RoiScoreConfig -> Int -> Double
roundTripPenaltyForWithConfig cfg0 completedRoundTrips
    | completedRoundTrips <= 0 = rscZeroRoundTripPenalty cfg
    | completedRoundTrips < rscMinimumActivityFloor cfg =
        rscLowRoundTripPenalty cfg * fromIntegral (rscMinimumActivityFloor cfg - completedRoundTrips)
    | otherwise = 0
  where
    cfg = sanitizeRoiScoreConfig cfg0

activityPenaltyFor :: Int -> Double
activityPenaltyFor = activityPenaltyForWithConfig defaultFormalRoiScoreConfig

activityPenaltyForWithConfig :: RoiScoreConfig -> Int -> Double
activityPenaltyForWithConfig cfg0 activityCount
    | activityCount <= 0 = rscZeroActivityPenalty cfg
    | activityCount < rscMinimumActivityFloor cfg =
        fromIntegral (rscMinimumActivityFloor cfg - activityCount) * rscLowActivityPenalty cfg
    | otherwise = 0
  where
    cfg = sanitizeRoiScoreConfig cfg0

exposurePenaltyFor :: Double -> Double
exposurePenaltyFor = exposurePenaltyForWithConfig defaultFormalRoiScoreConfig

exposurePenaltyForWithConfig :: RoiScoreConfig -> Double -> Double
exposurePenaltyForWithConfig cfg0 exposure
    | exposure <= 0 = rscZeroExposurePenalty cfg
    | exposure < rscMinimumExposureFloor cfg =
        let gap =
                if rscMinimumExposureFloor cfg <= 0
                    then 0
                    else 1 - clamp 0 1 (exposure / rscMinimumExposureFloor cfg)
         in rscLowExposurePenaltyBase cfg + rscLowExposurePenaltyGapScale cfg * gap
    | otherwise = 0
  where
    cfg = sanitizeRoiScoreConfig cfg0

sanitizeFinite0 :: Double -> Double
sanitizeFinite0 x =
    if isNaN x || isInfinite x
        then 0
        else x

finiteNonNegative :: Double -> Double
finiteNonNegative = max 0 . sanitizeFinite0

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)
