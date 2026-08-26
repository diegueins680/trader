{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Trader.PortfolioSelection (
    PortfolioDailyReturn (..),
    PortfolioEvidence (..),
    PortfolioCandidate (..),
    PortfolioMember (..),
    PortfolioMetrics (..),
    PortfolioRolloutMode (..),
    PortfolioSelection (..),
    PortfolioSelectorConfig (..),
    PortfolioGraduationConfig (..),
    PortfolioGraduationDecision (..),
    PortfolioGraduationEvidence (..),
    PortfolioGraduationReview (..),
    defaultPortfolioSelectorConfig,
    defaultPortfolioGraduationConfig,
    portfolioGraduationConfigVersion,
    portfolioGraduationDecisionCode,
    portfolioGraduationExecutionReliability,
    portfolioGraduationPerformance,
    portfolioGraduationReview,
    portfolioGraduationReviewApplies,
    portfolioGraduationStatusReliability,
    portfolioSelectorConfigVersion,
    parsePortfolioRolloutMode,
    portfolioRolloutModeCode,
    portfolioAnnualizedReturn,
    portfolioMaxDrawdown,
    portfolioFailureCacheLookup,
    selectPortfolio,
    portfolioMembersRemainAdmitted,
    refreshPortfolioSelection,
    portfolioSelectionShouldRotate,
) where

import Control.Monad (replicateM, unless)
import Data.Aeson (FromJSON (..), ToJSON (..), object, withObject, (.:), (.:?), (.=))
import qualified Data.Aeson as Aeson
import Data.Char (ord, toLower)
import Data.Int (Int64)
import Data.List (foldl', insertBy, nub, sort, sortOn, tails)
import qualified Data.Map.Strict as M
import qualified Data.Ord as Ord
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector as V
import GHC.Generics (Generic)

{- | Reuse an expensive selector failure only while both its TTL and the
leaderboard snapshot that produced it remain unchanged. Fresh evidence must
be reconsidered on the next poll instead of inheriting an hour-old failure.
-}
portfolioFailureCacheLookup :: (Eq snapshot) => Int64 -> Int64 -> snapshot -> Maybe (Int64, snapshot, String) -> Maybe String
portfolioFailureCacheLookup ttlMs now snapshot cached =
    case cached of
        Just (failedAtMs, failedSnapshot, err)
            | failedSnapshot == snapshot
                && now >= failedAtMs
                && now - failedAtMs < max 0 ttlMs ->
                Just err
        _ -> Nothing

data PortfolioDailyReturn = PortfolioDailyReturn
    { pdrDayMs :: !Int64
    , pdrReturn :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioDailyReturn where
    toJSON value = object ["dayMs" .= pdrDayMs value, "return" .= pdrReturn value]

instance FromJSON PortfolioDailyReturn where
    parseJSON = withObject "PortfolioDailyReturn" $ \obj ->
        PortfolioDailyReturn <$> obj .: "dayMs" <*> obj .: "return"

data PortfolioEvidence = PortfolioEvidence
    { peKind :: !Text
    , peWindowStartMs :: !Int64
    , peWindowEndMs :: !Int64
    , peObservationCount :: !Int
    , peCostModel :: !Text
    , peDailyReturns :: ![PortfolioDailyReturn]
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioEvidence where
    toJSON value =
        object
            [ "kind" .= peKind value
            , "windowStartMs" .= peWindowStartMs value
            , "windowEndMs" .= peWindowEndMs value
            , "observationCount" .= peObservationCount value
            , "costModel" .= peCostModel value
            , "dailyReturns" .= peDailyReturns value
            ]

instance FromJSON PortfolioEvidence where
    parseJSON = withObject "PortfolioEvidence" $ \obj ->
        PortfolioEvidence
            <$> obj .: "kind"
            <*> obj .: "windowStartMs"
            <*> obj .: "windowEndMs"
            <*> obj .: "observationCount"
            <*> obj .: "costModel"
            <*> obj .: "dailyReturns"

data PortfolioCandidate = PortfolioCandidate
    { pcUuid :: !Text
    , pcSymbol :: !Text
    , pcMaxWeight :: !Double
    , pcEvidence :: !PortfolioEvidence
    }
    deriving (Eq, Show, Generic)

data PortfolioMember = PortfolioMember
    { pmUuid :: !Text
    , pmSymbol :: !Text
    , pmWeight :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioMember where
    toJSON value =
        object
            [ "uuid" .= pmUuid value
            , "symbol" .= pmSymbol value
            , "weight" .= pmWeight value
            ]

instance FromJSON PortfolioMember where
    parseJSON = withObject "PortfolioMember" $ \obj ->
        PortfolioMember <$> obj .: "uuid" <*> obj .: "symbol" <*> obj .: "weight"

data PortfolioMetrics = PortfolioMetrics
    { pmHistoricalAnnualizedReturn :: !Double
    , pmHistoricalMaxDrawdown :: !Double
    , pmAnnualizedReturnP10 :: !Double
    , pmAnnualizedReturnP50 :: !Double
    , pmAnnualizedReturnP90 :: !Double
    , pmMaxDrawdownP95 :: !Double
    , pmAverageCorrelation :: !Double
    , pmSwitchingCost :: !Double
    , pmPairedOutperformanceProbability :: !Double
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioMetrics where
    toJSON value =
        object
            [ "historicalAnnualizedReturn" .= pmHistoricalAnnualizedReturn value
            , "historicalMaxDrawdown" .= pmHistoricalMaxDrawdown value
            , "annualizedReturnP10" .= pmAnnualizedReturnP10 value
            , "annualizedReturnP50" .= pmAnnualizedReturnP50 value
            , "annualizedReturnP90" .= pmAnnualizedReturnP90 value
            , "maxDrawdownP95" .= pmMaxDrawdownP95 value
            , "averageCorrelation" .= pmAverageCorrelation value
            , "switchingCost" .= pmSwitchingCost value
            , "pairedOutperformanceProbability" .= pmPairedOutperformanceProbability value
            ]

instance FromJSON PortfolioMetrics where
    parseJSON = withObject "PortfolioMetrics" $ \obj ->
        PortfolioMetrics
            <$> obj .: "historicalAnnualizedReturn"
            <*> obj .: "historicalMaxDrawdown"
            <*> obj .: "annualizedReturnP10"
            <*> obj .: "annualizedReturnP50"
            <*> obj .: "annualizedReturnP90"
            <*> obj .: "maxDrawdownP95"
            <*> obj .: "averageCorrelation"
            <*> obj .: "switchingCost"
            <*> obj .:? "pairedOutperformanceProbability" Aeson..!= 1

data PortfolioRolloutMode
    = PortfolioShadow
    | PortfolioCanary
    | PortfolioEnforce
    deriving (Eq, Ord, Show, Generic)

portfolioRolloutModeCode :: PortfolioRolloutMode -> Text
portfolioRolloutModeCode PortfolioShadow = "shadow"
portfolioRolloutModeCode PortfolioCanary = "canary"
portfolioRolloutModeCode PortfolioEnforce = "enforce"

parsePortfolioRolloutMode :: String -> Maybe PortfolioRolloutMode
parsePortfolioRolloutMode raw =
    case map toLower raw of
        "shadow" -> Just PortfolioShadow
        "canary" -> Just PortfolioCanary
        "enforce" -> Just PortfolioEnforce
        _ -> Nothing

instance ToJSON PortfolioRolloutMode where
    toJSON = Aeson.String . portfolioRolloutModeCode

instance FromJSON PortfolioRolloutMode where
    parseJSON = Aeson.withText "PortfolioRolloutMode" $ \raw ->
        case parsePortfolioRolloutMode (T.unpack raw) of
            Nothing -> fail "portfolio rollout mode must be shadow, canary, or enforce"
            Just mode -> pure mode

data PortfolioSelection = PortfolioSelection
    { psGeneratedAtMs :: !Int64
    , psValidUntilMs :: !Int64
    , psEvidenceStartMs :: !Int64
    , psEvidenceEndMs :: !Int64
    , psMode :: !PortfolioRolloutMode
    , psMembers :: ![PortfolioMember]
    , psMetrics :: !PortfolioMetrics
    , psCandidateCount :: !Int
    , psBootstrapSeed :: !Int
    , psConfigVersion :: !Text
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioSelection where
    toJSON value =
        object
            [ "generatedAtMs" .= psGeneratedAtMs value
            , "validUntilMs" .= psValidUntilMs value
            , "evidenceStartMs" .= psEvidenceStartMs value
            , "evidenceEndMs" .= psEvidenceEndMs value
            , "mode" .= psMode value
            , "members" .= psMembers value
            , "metrics" .= psMetrics value
            , "candidateCount" .= psCandidateCount value
            , "bootstrapSeed" .= psBootstrapSeed value
            , "configVersion" .= psConfigVersion value
            ]

instance FromJSON PortfolioSelection where
    parseJSON = withObject "PortfolioSelection" $ \obj ->
        PortfolioSelection
            <$> obj .: "generatedAtMs"
            <*> obj .: "validUntilMs"
            <*> obj .: "evidenceStartMs"
            <*> obj .: "evidenceEndMs"
            <*> obj .:? "mode" Aeson..!= PortfolioShadow
            <*> obj .: "members"
            <*> obj .: "metrics"
            <*> obj .: "candidateCount"
            <*> obj .: "bootstrapSeed"
            <*> obj .:? "configVersion" Aeson..!= "portfolio-v1"

data PortfolioSelectorConfig = PortfolioSelectorConfig
    { pscMaxMembers :: !Int
    , pscMaxMemberWeight :: !Double
    , pscMaxGrossWeight :: !Double
    , pscWeightStep :: !Double
    , pscMaxDrawdown :: !Double
    , pscMinimumObservations :: !Int
    , pscMaximumObservations :: !Int
    , pscCandidateLimit :: !Int
    , pscCandidatesPerSymbol :: !Int
    , pscBootstrapSamples :: !Int
    , pscBootstrapBlockDays :: !Int
    , pscBootstrapPortfolioLimit :: !Int
    , pscSwitchingCostRate :: !Double
    , pscRotationImprovementFloor :: !Double
    , pscRotationProbabilityFloor :: !Double
    , pscValidForMs :: !Int64
    }
    deriving (Eq, Show, Generic)

data PortfolioGraduationConfig = PortfolioGraduationConfig
    { pgcEnabled :: !Bool
    , pgcStartedAtMs :: !Int64
    , pgcMinimumDailyObservations :: !Int
    , pgcMinimumNetReturn :: !Double
    , pgcMaximumDrawdown :: !Double
    , pgcMinimumExecutionAttempts :: !Int
    , pgcMinimumExecutionReliability :: !Double
    , pgcMinimumStatusReliability :: !Double
    }
    deriving (Eq, Show, Generic)

data PortfolioGraduationDecision
    = PortfolioGraduationPending
    | PortfolioGraduated
    deriving (Eq, Ord, Show, Generic)

portfolioGraduationDecisionCode :: PortfolioGraduationDecision -> Text
portfolioGraduationDecisionCode PortfolioGraduationPending = "pending"
portfolioGraduationDecisionCode PortfolioGraduated = "graduated"

instance ToJSON PortfolioGraduationDecision where
    toJSON = Aeson.String . portfolioGraduationDecisionCode

instance FromJSON PortfolioGraduationDecision where
    parseJSON = Aeson.withText "PortfolioGraduationDecision" $ \raw ->
        case T.toLower (T.strip raw) of
            "pending" -> pure PortfolioGraduationPending
            "graduated" -> pure PortfolioGraduated
            _ -> fail "portfolio graduation decision must be pending or graduated"

data PortfolioGraduationEvidence = PortfolioGraduationEvidence
    { pgeDailyObservationCount :: !Int
    , pgeNetReturn :: !Double
    , pgeMaxDrawdown :: !Double
    , pgeExecutionAttempts :: !Int
    , pgeExecutionSuccesses :: !Int
    , pgeStatusSamples :: !Int
    , pgeHealthyStatusSamples :: !Int
    , pgeLatestStatusesHealthy :: !Bool
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioGraduationEvidence where
    toJSON value =
        object
            [ "dailyObservationCount" .= pgeDailyObservationCount value
            , "netReturn" .= pgeNetReturn value
            , "maxDrawdown" .= pgeMaxDrawdown value
            , "executionAttempts" .= pgeExecutionAttempts value
            , "executionSuccesses" .= pgeExecutionSuccesses value
            , "executionReliability" .= portfolioGraduationExecutionReliability value
            , "statusSamples" .= pgeStatusSamples value
            , "healthyStatusSamples" .= pgeHealthyStatusSamples value
            , "statusReliability" .= portfolioGraduationStatusReliability value
            , "latestStatusesHealthy" .= pgeLatestStatusesHealthy value
            ]

instance FromJSON PortfolioGraduationEvidence where
    parseJSON = withObject "PortfolioGraduationEvidence" $ \obj ->
        PortfolioGraduationEvidence
            <$> obj .: "dailyObservationCount"
            <*> obj .: "netReturn"
            <*> obj .: "maxDrawdown"
            <*> obj .: "executionAttempts"
            <*> obj .: "executionSuccesses"
            <*> obj .: "statusSamples"
            <*> obj .: "healthyStatusSamples"
            <*> obj .: "latestStatusesHealthy"

data PortfolioGraduationReview = PortfolioGraduationReview
    { pgrReviewedAtMs :: !Int64
    , pgrReviewedUuids :: ![Text]
    , pgrDecision :: !PortfolioGraduationDecision
    , pgrReasons :: ![Text]
    , pgrEvidence :: !PortfolioGraduationEvidence
    , pgrConfigVersion :: !Text
    }
    deriving (Eq, Show, Generic)

instance ToJSON PortfolioGraduationReview where
    toJSON value =
        object
            [ "reviewedAtMs" .= pgrReviewedAtMs value
            , "reviewedUuids" .= pgrReviewedUuids value
            , "decision" .= pgrDecision value
            , "reasons" .= pgrReasons value
            , "evidence" .= pgrEvidence value
            , "configVersion" .= pgrConfigVersion value
            ]

instance FromJSON PortfolioGraduationReview where
    parseJSON = withObject "PortfolioGraduationReview" $ \obj ->
        PortfolioGraduationReview
            <$> obj .: "reviewedAtMs"
            <*> obj .: "reviewedUuids"
            <*> obj .: "decision"
            <*> obj .: "reasons"
            <*> obj .: "evidence"
            <*> obj .: "configVersion"

defaultPortfolioGraduationConfig :: PortfolioGraduationConfig
defaultPortfolioGraduationConfig =
    PortfolioGraduationConfig
        { pgcEnabled = False
        , pgcStartedAtMs = 0
        , pgcMinimumDailyObservations = 30
        , pgcMinimumNetReturn = 0
        , pgcMaximumDrawdown = 0.10
        , pgcMinimumExecutionAttempts = 10
        , pgcMinimumExecutionReliability = 0.95
        , pgcMinimumStatusReliability = 0.99
        }

portfolioGraduationConfigVersion :: PortfolioGraduationConfig -> Text
portfolioGraduationConfigVersion config =
    T.intercalate
        ":"
        [ "portfolio-graduation-v1"
        , T.pack (show (pgcEnabled config))
        , T.pack (show (pgcStartedAtMs config))
        , T.pack (show (pgcMinimumDailyObservations config))
        , T.pack (show (pgcMinimumNetReturn config))
        , T.pack (show (pgcMaximumDrawdown config))
        , T.pack (show (pgcMinimumExecutionAttempts config))
        , T.pack (show (pgcMinimumExecutionReliability config))
        , T.pack (show (pgcMinimumStatusReliability config))
        ]

portfolioGraduationExecutionReliability :: PortfolioGraduationEvidence -> Double
portfolioGraduationExecutionReliability evidence =
    ratio (pgeExecutionSuccesses evidence) (pgeExecutionAttempts evidence)

portfolioGraduationStatusReliability :: PortfolioGraduationEvidence -> Double
portfolioGraduationStatusReliability evidence =
    ratio (pgeHealthyStatusSamples evidence) (pgeStatusSamples evidence)

ratio :: Int -> Int -> Double
ratio numerator denominator
    | denominator <= 0 = 0
    | otherwise = fromIntegral numerator / fromIntegral denominator

portfolioGraduationPerformance :: [Double] -> Either String (Int, Double, Double)
portfolioGraduationPerformance equities
    | any (\value -> not (isFinite value) || value <= 0) equities = Left "graduation fleet equity must be finite and positive"
    | otherwise =
        case equities of
            [] -> Right (0, 0, 0)
            _ ->
                let returns = zipWith (\previous current -> current / previous - 1) (1 : equities) equities
                 in Right (length equities, last equities - 1, portfolioMaxDrawdown returns)

portfolioGraduationReview :: PortfolioGraduationConfig -> Int64 -> [Text] -> PortfolioGraduationEvidence -> PortfolioGraduationReview
portfolioGraduationReview config now reviewedUuidsRaw evidence =
    PortfolioGraduationReview
        { pgrReviewedAtMs = now
        , pgrReviewedUuids = reviewedUuids
        , pgrDecision = if null reasons then PortfolioGraduated else PortfolioGraduationPending
        , pgrReasons = reasons
        , pgrEvidence = evidence
        , pgrConfigVersion = portfolioGraduationConfigVersion config
        }
  where
    reviewedUuids = nub (sort (map (T.toLower . T.strip) reviewedUuidsRaw))
    finite = isFinite
    reasons =
        [ "automatic-graduation-disabled"
        | not (pgcEnabled config)
        ]
            ++ [ "graduation-start-time-invalid"
               | pgcStartedAtMs config <= 0
               ]
            ++ [ "reviewed-uuid-set-empty"
               | null reviewedUuids
               ]
            ++ [ "daily-observations-below-minimum"
               | pgeDailyObservationCount evidence < pgcMinimumDailyObservations config
               ]
            ++ [ "net-return-below-minimum"
               | not (finite (pgeNetReturn evidence))
                    || pgeNetReturn evidence <= pgcMinimumNetReturn config
               ]
            ++ [ "max-drawdown-above-limit"
               | not (finite (pgeMaxDrawdown evidence))
                    || pgeMaxDrawdown evidence > pgcMaximumDrawdown config
               ]
            ++ [ "execution-attempts-below-minimum"
               | pgeExecutionAttempts evidence < pgcMinimumExecutionAttempts config
               ]
            ++ [ "execution-reliability-below-minimum"
               | portfolioGraduationExecutionReliability evidence < pgcMinimumExecutionReliability config
               ]
            ++ [ "status-reliability-below-minimum"
               | portfolioGraduationStatusReliability evidence < pgcMinimumStatusReliability config
               ]
            ++ [ "latest-status-unhealthy"
               | not (pgeLatestStatusesHealthy evidence)
               ]

portfolioGraduationReviewApplies :: PortfolioGraduationConfig -> [Text] -> PortfolioGraduationReview -> Bool
portfolioGraduationReviewApplies config reviewedUuidsRaw review =
    pgcEnabled config
        && pgrDecision review == PortfolioGraduated
        && pgrConfigVersion review == portfolioGraduationConfigVersion config
        && pgrReviewedUuids review == reviewedUuids
  where
    reviewedUuids = nub (sort (map (T.toLower . T.strip) reviewedUuidsRaw))

defaultPortfolioSelectorConfig :: PortfolioSelectorConfig
defaultPortfolioSelectorConfig =
    PortfolioSelectorConfig
        { pscMaxMembers = 3
        , pscMaxMemberWeight = 0.25
        , pscMaxGrossWeight = 0.75
        , pscWeightStep = 0.05
        , pscMaxDrawdown = 0.10
        , pscMinimumObservations = 180
        , pscMaximumObservations = 365
        , pscCandidateLimit = 30
        , pscCandidatesPerSymbol = 3
        , pscBootstrapSamples = 1000
        , pscBootstrapBlockDays = 7
        , pscBootstrapPortfolioLimit = 256
        , pscSwitchingCostRate = 0
        , pscRotationImprovementFloor = 0.02
        , pscRotationProbabilityFloor = 0.90
        , pscValidForMs = 8 * 86400000
        }

portfolioSelectorConfigVersion :: PortfolioSelectorConfig -> Text
portfolioSelectorConfigVersion config =
    T.intercalate
        ":"
        [ "portfolio-v1"
        , T.pack (show (pscMaxMembers config))
        , T.pack (show (pscMaxMemberWeight config))
        , T.pack (show (pscMaxGrossWeight config))
        , T.pack (show (pscWeightStep config))
        , T.pack (show (pscMaxDrawdown config))
        , T.pack (show (pscMinimumObservations config))
        , T.pack (show (pscMaximumObservations config))
        , T.pack (show (pscCandidateLimit config))
        , T.pack (show (pscCandidatesPerSymbol config))
        , T.pack (show (pscBootstrapSamples config))
        , T.pack (show (pscBootstrapBlockDays config))
        , T.pack (show (pscBootstrapPortfolioLimit config))
        , T.pack (show (pscSwitchingCostRate config))
        , T.pack (show (pscRotationImprovementFloor config))
        , T.pack (show (pscRotationProbabilityFloor config))
        ]

isFinite :: Double -> Bool
isFinite value = not (isNaN value || isInfinite value)

portfolioAnnualizedReturn :: [Double] -> Double
portfolioAnnualizedReturn values
    | null values = -1
    | any (\value -> not (isFinite value) || value <= -1) values = -1
    | otherwise =
        let result = exp (365 * sum (map log1p values) / fromIntegral (length values)) - 1
         in if isFinite result then result else -1
  where
    log1p value = log (1 + value)

portfolioMaxDrawdown :: [Double] -> Double
portfolioMaxDrawdown values = third (foldl' step (1, 1, 0) values)
  where
    third (_, _, value) = value
    step (equity, peak, worst) value
        | not (isFinite value) || value <= -1 = (0, peak, 1)
        | otherwise =
            let equity' = equity * (1 + value)
                peak' = max peak equity'
                drawdown = if peak' <= 0 then 1 else (peak' - equity') / peak'
             in (equity', peak', max worst drawdown)

data PortfolioSpec = PortfolioSpec
    { pspMembers :: ![(PortfolioCandidate, Double)]
    , pspSeries :: ![Double]
    , pspDays :: ![Int64]
    , pspHistoricalAnnualizedReturn :: !Double
    , pspHistoricalMaxDrawdown :: !Double
    , pspAverageCorrelation :: !Double
    , pspSwitchingCost :: !Double
    }

data PreparedMemberSet = PreparedMemberSet
    { pmsCandidates :: ![PortfolioCandidate]
    , pmsDays :: ![Int64]
    , pmsReturns :: ![V.Vector Double]
    , pmsAverageCorrelation :: !Double
    }

selectPortfolio ::
    PortfolioSelectorConfig ->
    Int64 ->
    PortfolioRolloutMode ->
    [PortfolioMember] ->
    [PortfolioCandidate] ->
    Either String PortfolioSelection
selectPortfolio config now mode incumbent candidates0 = do
    validateSelectorConfig config
    let candidates = filter (candidateValid config) candidates0
    if null candidates
        then Left "no portfolio candidates have sufficient aligned return evidence"
        else do
            let seed = stableSeed candidates
                standalone =
                    [ (candidate, candidateStandaloneScore config seed candidate)
                    | candidate <- candidates
                    ]
                rankedCandidates =
                    takePerSymbol
                        (pscCandidatesPerSymbol config)
                        (sortOn (\(candidate, score) -> (Ord.Down score, pcUuid candidate)) standalone)
                incumbentUuids = M.fromList [(pmUuid member, ()) | member <- incumbent]
                incumbentCandidates = [item | item@(candidate, _) <- standalone, M.member (pcUuid candidate) incumbentUuids]
                preselected =
                    take
                        (pscCandidateLimit config)
                        (dedupeCandidatesByUuid (incumbentCandidates ++ rankedCandidates))
                evidenceMaps =
                    M.fromList
                        [ (pcUuid candidate, candidateReturnMap config candidate)
                        | (candidate, _) <- preselected
                        ]
                bootstrapPool =
                    foldl'
                        (retainMemberSet evidenceMaps)
                        []
                        (portfolioMemberSets config (map fst preselected))
                evaluated = map (evaluateSpec config seed) bootstrapPool
                admitted = filter (portfolioAdmitted config . snd) evaluated
            case sortOn selectionKey admitted of
                [] -> Left "no portfolio clears the conservative annualized-return and drawdown gates"
                ((best, metrics0) : _) ->
                    let pairedProbability = incumbentOutperformanceProbability config seed incumbent candidates best
                        metrics = metrics0{pmPairedOutperformanceProbability = pairedProbability}
                        days = pspDays best
                     in case days of
                            [] -> Left "selected portfolio has no aligned evidence days"
                            _ ->
                                Right
                                    PortfolioSelection
                                        { psGeneratedAtMs = now
                                        , psValidUntilMs = now + pscValidForMs config
                                        , psEvidenceStartMs = head days
                                        , psEvidenceEndMs = last days
                                        , psMode = mode
                                        , psMembers = map toMember (pspMembers best)
                                        , psMetrics = metrics
                                        , psCandidateCount = length candidates
                                        , psBootstrapSeed = seed
                                        , psConfigVersion = portfolioSelectorConfigVersion config
                                        }
  where
    retainMemberSet evidenceMaps accepted candidatesForSet =
        case prepareMemberSet config evidenceMaps candidatesForSet of
            Nothing -> accepted
            Just prepared ->
                foldl'
                    (retainHistoricalCandidate prepared)
                    accepted
                    (portfolioWeightVectors config candidatesForSet)
    retainHistoricalCandidate prepared accepted weights =
        let spec = buildPortfolioSpecFromPrepared config incumbent prepared weights
         in if pspHistoricalMaxDrawdown spec <= pscMaxDrawdown config
                then
                    take
                        (pscBootstrapPortfolioLimit config)
                        (insertBy compareHistorical spec accepted)
                else accepted
    compareHistorical left right =
        compare
            (Ord.Down (pspHistoricalAnnualizedReturn left))
            (Ord.Down (pspHistoricalAnnualizedReturn right))
    toMember (candidate, weight) = PortfolioMember (pcUuid candidate) (pcSymbol candidate) weight
    selectionKey (spec, metrics) =
        ( Ord.Down (pmAnnualizedReturnP10 metrics)
        , pmMaxDrawdownP95 metrics
        , pspSwitchingCost spec
        , pspAverageCorrelation spec
        , map (pcUuid . fst) (pspMembers spec)
        )

portfolioSelectionShouldRotate :: PortfolioSelectorConfig -> PortfolioSelection -> PortfolioSelection -> Bool
portfolioSelectionShouldRotate config incumbent challenger =
    pmAnnualizedReturnP10 (psMetrics challenger)
        >= pmAnnualizedReturnP10 (psMetrics incumbent) + pscRotationImprovementFloor config
        && pmPairedOutperformanceProbability (psMetrics challenger) >= pscRotationProbabilityFloor config

portfolioMembersRemainAdmitted :: PortfolioSelectorConfig -> [PortfolioMember] -> [PortfolioCandidate] -> Bool
portfolioMembersRemainAdmitted config members candidates =
    case evaluatePortfolioMembers config members candidates of
        Left _ -> False
        Right (_, metrics, _) -> portfolioAdmitted config metrics

refreshPortfolioSelection ::
    PortfolioSelectorConfig ->
    Int64 ->
    PortfolioRolloutMode ->
    [PortfolioMember] ->
    [PortfolioCandidate] ->
    Either String PortfolioSelection
refreshPortfolioSelection config now mode members candidates = do
    (spec, metrics, seed) <- evaluatePortfolioMembers config members candidates
    if not (portfolioAdmitted config metrics)
        then Left "the incumbent portfolio no longer clears the conservative return and drawdown gates"
        else case pspDays spec of
            [] -> Left "the incumbent portfolio has no aligned evidence days"
            days ->
                Right
                    PortfolioSelection
                        { psGeneratedAtMs = now
                        , psValidUntilMs = now + pscValidForMs config
                        , psEvidenceStartMs = head days
                        , psEvidenceEndMs = last days
                        , psMode = mode
                        , psMembers = members
                        , psMetrics = metrics
                        , psCandidateCount = length candidates
                        , psBootstrapSeed = seed
                        , psConfigVersion = portfolioSelectorConfigVersion config
                        }

evaluatePortfolioMembers ::
    PortfolioSelectorConfig ->
    [PortfolioMember] ->
    [PortfolioCandidate] ->
    Either String (PortfolioSpec, PortfolioMetrics, Int)
evaluatePortfolioMembers config members candidates = do
    validateSelectorConfig config
    unless (portfolioMembersStructurallyValid config members candidates) $
        Left "the incumbent portfolio membership or weights are invalid"
    weightedCandidates <- maybe (Left "one or more incumbent portfolio members are no longer eligible") Right selectedCandidates
    spec <- maybe (Left "the incumbent portfolio no longer has sufficient aligned evidence") Right (buildPortfolioSpec config{pscSwitchingCostRate = 0} [] weightedCandidates)
    let seed = stableSeed (map fst weightedCandidates)
        (_, metrics) = evaluateSpec config seed spec
    pure (spec, metrics, seed)
  where
    candidatesByUuid = M.fromList [(pcUuid candidate, candidate) | candidate <- candidates]
    selectedCandidates =
        traverse
            (\member -> (,pmWeight member) <$> M.lookup (pmUuid member) candidatesByUuid)
            members

portfolioMembersStructurallyValid :: PortfolioSelectorConfig -> [PortfolioMember] -> [PortfolioCandidate] -> Bool
portfolioMembersStructurallyValid config members candidates =
    not (null members)
        && length members <= pscMaxMembers config
        && length uuids == length (nub uuids)
        && length symbols == length (nub symbols)
        && all memberValid members
        && sum (map pmWeight members) <= pscMaxGrossWeight config + 1.0e-12
  where
    uuids = map pmUuid members
    symbols = map (T.toUpper . T.strip . pmSymbol) members
    candidatesByUuid = M.fromList [(pcUuid candidate, candidate) | candidate <- candidates]
    memberValid member =
        case M.lookup (pmUuid member) candidatesByUuid of
            Nothing -> False
            Just candidate ->
                candidateValid config candidate
                    && T.toUpper (T.strip (pmSymbol member)) == T.toUpper (T.strip (pcSymbol candidate))
                    && isFinite (pmWeight member)
                    && pmWeight member > 0
                    && pmWeight member <= min (pscMaxMemberWeight config) (pcMaxWeight candidate) + 1.0e-12

validateSelectorConfig :: PortfolioSelectorConfig -> Either String ()
validateSelectorConfig config
    | pscMaxMembers config <= 0 = Left "portfolio max members must be positive"
    | not (finitePositive (pscMaxMemberWeight config)) || pscMaxMemberWeight config > 1 = Left "portfolio member weight cap must be in (0,1]"
    | not (finitePositive (pscMaxGrossWeight config)) || pscMaxGrossWeight config > 1 = Left "portfolio gross weight cap must be in (0,1]"
    | pscMaxMemberWeight config > pscMaxGrossWeight config = Left "portfolio member weight cap cannot exceed the gross cap"
    | not (finitePositive (pscWeightStep config)) || pscWeightStep config > pscMaxMemberWeight config = Left "portfolio weight step must be positive and no larger than the member cap"
    | not (finitePositive (pscMaxDrawdown config)) || pscMaxDrawdown config > 1 = Left "portfolio drawdown cap must be in (0,1]"
    | pscMinimumObservations config <= 1 = Left "portfolio minimum observations must exceed one"
    | pscMaximumObservations config < pscMinimumObservations config = Left "portfolio maximum observations must cover the minimum"
    | pscCandidateLimit config <= 0 || pscCandidatesPerSymbol config <= 0 = Left "portfolio candidate limits must be positive"
    | pscBootstrapSamples config <= 0 = Left "portfolio bootstrap samples must be positive"
    | pscBootstrapBlockDays config <= 0 = Left "portfolio bootstrap block length must be positive"
    | pscBootstrapPortfolioLimit config <= 0 = Left "portfolio bootstrap portfolio limit must be positive"
    | not (isFinite (pscSwitchingCostRate config)) || pscSwitchingCostRate config < 0 = Left "portfolio switching cost must be finite and non-negative"
    | not (isFinite (pscRotationImprovementFloor config)) || pscRotationImprovementFloor config < 0 = Left "portfolio rotation improvement must be finite and non-negative"
    | not (isFinite (pscRotationProbabilityFloor config)) || pscRotationProbabilityFloor config < 0 || pscRotationProbabilityFloor config > 1 = Left "portfolio rotation probability must be in [0,1]"
    | pscValidForMs config <= 0 = Left "portfolio selection validity must be positive"
    | otherwise = Right ()
  where
    finitePositive value = isFinite value && value > 0

candidateValid :: PortfolioSelectorConfig -> PortfolioCandidate -> Bool
candidateValid config candidate =
    let evidence = pcEvidence candidate
        rawReturnMap = M.fromList [(pdrDayMs value, value) | value <- peDailyReturns evidence]
        rawReturns = M.elems rawReturnMap
        returns = normalizedEvidenceReturns config evidence
     in not (T.null (T.strip (pcUuid candidate)))
            && not (T.null (T.strip (pcSymbol candidate)))
            && isFinite (pcMaxWeight candidate)
            && pcMaxWeight candidate > 0
            && pcMaxWeight candidate <= 1
            && peKind evidence == "backtest_oos"
            && peCostModel evidence == "backtest_net_equity"
            && peObservationCount evidence == length rawReturns
            && maybe False (\(day, _) -> peWindowStartMs evidence <= day) (M.lookupMin rawReturnMap)
            && maybe False (\(day, _) -> peWindowEndMs evidence == day) (M.lookupMax rawReturnMap)
            && peWindowEndMs evidence >= peWindowStartMs evidence
            && length returns >= pscMinimumObservations config
            && all (\value -> pdrDayMs value >= 0 && isFinite (pdrReturn value) && pdrReturn value > -1) returns

normalizedEvidenceReturns :: PortfolioSelectorConfig -> PortfolioEvidence -> [PortfolioDailyReturn]
normalizedEvidenceReturns config evidence =
    takeLast (pscMaximumObservations config) $ M.elems $ M.fromList [(pdrDayMs value, value) | value <- peDailyReturns evidence]

takeLast :: Int -> [a] -> [a]
takeLast count values = drop (max 0 (length values - max 0 count)) values

candidateStandaloneScore :: PortfolioSelectorConfig -> Int -> PortfolioCandidate -> Double
candidateStandaloneScore config seed candidate =
    let values = map pdrReturn (normalizedEvidenceReturns config (pcEvidence candidate))
        annualized = fst (bootstrapDistributions config (seed + textSeed (pcUuid candidate)) values)
     in quantile 0.10 annualized

takePerSymbol :: Int -> [(PortfolioCandidate, Double)] -> [(PortfolioCandidate, Double)]
takePerSymbol maxPerSymbol = reverse . snd . foldl' step (M.empty, [])
  where
    step (counts, accepted) item@(candidate, _) =
        let symbol = T.toUpper (T.strip (pcSymbol candidate))
            count = M.findWithDefault 0 symbol counts
         in if count >= max 1 maxPerSymbol
                then (counts, accepted)
                else (M.insert symbol (count + 1) counts, item : accepted)

dedupeCandidatesByUuid :: [(PortfolioCandidate, Double)] -> [(PortfolioCandidate, Double)]
dedupeCandidatesByUuid = reverse . snd . foldl' step (M.empty, [])
  where
    step (seen, accepted) item@(candidate, _)
        | M.member (pcUuid candidate) seen = (seen, accepted)
        | otherwise = (M.insert (pcUuid candidate) () seen, item : accepted)

portfolioMemberSets :: PortfolioSelectorConfig -> [PortfolioCandidate] -> [[PortfolioCandidate]]
portfolioMemberSets config candidates =
    concatMap
        (filter uniqueSymbols . (`combinations` candidates))
        [1 .. min (pscMaxMembers config) (length candidates)]
  where
    uniqueSymbols members =
        let symbols = map (T.toUpper . T.strip . pcSymbol) members
         in length symbols == length (nub symbols)

portfolioWeightVectors :: PortfolioSelectorConfig -> [PortfolioCandidate] -> [[Double]]
portfolioWeightVectors config members =
    [ memberWeights
    | memberWeights <- replicateM (length members) weights
    , and (zipWith (\candidate weight -> weight <= min (pscMaxMemberWeight config) (pcMaxWeight candidate) + 1.0e-12) members memberWeights)
    , sum memberWeights <= pscMaxGrossWeight config + 1.0e-12
    ]
  where
    weights =
        [ pscWeightStep config * fromIntegral step
        | step <- [1 .. floor (pscMaxMemberWeight config / pscWeightStep config + 1.0e-9)]
        ]

combinations :: Int -> [a] -> [[a]]
combinations 0 _ = [[]]
combinations _ [] = []
combinations count (value : rest) =
    map (value :) (combinations (count - 1) rest) ++ combinations count rest

buildPortfolioSpec :: PortfolioSelectorConfig -> [PortfolioMember] -> [(PortfolioCandidate, Double)] -> Maybe PortfolioSpec
buildPortfolioSpec config incumbent members =
    buildPortfolioSpecWithMaps config evidenceMaps incumbent members
  where
    evidenceMaps = M.fromList [(pcUuid candidate, candidateReturnMap config candidate) | (candidate, _) <- members]

candidateReturnMap :: PortfolioSelectorConfig -> PortfolioCandidate -> M.Map Int64 Double
candidateReturnMap config candidate =
    M.fromList
        [ (pdrDayMs value, pdrReturn value)
        | value <- normalizedEvidenceReturns config (pcEvidence candidate)
        ]

buildPortfolioSpecWithMaps ::
    PortfolioSelectorConfig ->
    M.Map Text (M.Map Int64 Double) ->
    [PortfolioMember] ->
    [(PortfolioCandidate, Double)] ->
    Maybe PortfolioSpec
buildPortfolioSpecWithMaps config evidenceMaps incumbent members = do
    let candidates = map fst members
        weights = map snd members
    prepared <- prepareMemberSet config evidenceMaps candidates
    pure (buildPortfolioSpecFromPrepared config incumbent prepared weights)

prepareMemberSet ::
    PortfolioSelectorConfig ->
    M.Map Text (M.Map Int64 Double) ->
    [PortfolioCandidate] ->
    Maybe PreparedMemberSet
prepareMemberSet config evidenceMaps candidates = do
    memberMaps <- traverse (\candidate -> M.lookup (pcUuid candidate) evidenceMaps) candidates
    let commonDays =
            case memberMaps of
                [] -> []
                (firstMap : restMaps) -> foldl' (\days values -> filter (`M.member` values) days) (M.keys firstMap) restMaps
        cappedDays = takeLast (pscMaximumObservations config) commonDays
    if length cappedDays < pscMinimumObservations config
        then Nothing
        else
            Just
                PreparedMemberSet
                    { pmsCandidates = candidates
                    , pmsDays = cappedDays
                    , pmsReturns = [V.fromList [values M.! day | day <- cappedDays] | values <- memberMaps]
                    , pmsAverageCorrelation = averagePairwiseCorrelation memberMaps cappedDays
                    }

buildPortfolioSpecFromPrepared ::
    PortfolioSelectorConfig ->
    [PortfolioMember] ->
    PreparedMemberSet ->
    [Double] ->
    PortfolioSpec
buildPortfolioSpecFromPrepared config incumbent prepared weights =
    PortfolioSpec
        { pspMembers = members
        , pspSeries = series
        , pspDays = pmsDays prepared
        , pspHistoricalAnnualizedReturn = portfolioAnnualizedReturn series
        , pspHistoricalMaxDrawdown = portfolioMaxDrawdown series
        , pspAverageCorrelation = pmsAverageCorrelation prepared
        , pspSwitchingCost = switchingCost
        }
  where
    members = zip (pmsCandidates prepared) weights
    rawSeries =
        V.toList $
            V.generate
                (length (pmsDays prepared))
                (\index -> sum (zipWith (\weight values -> weight * (values V.! index)) weights (pmsReturns prepared)))
    switchingCost = portfolioSwitchingCost config incumbent members
    series = applyInitialCost switchingCost rawSeries

applyInitialCost :: Double -> [Double] -> [Double]
applyInitialCost _ [] = []
applyInitialCost cost (value : rest) = value - max 0 cost : rest

portfolioSwitchingCost :: PortfolioSelectorConfig -> [PortfolioMember] -> [(PortfolioCandidate, Double)] -> Double
portfolioSwitchingCost config incumbent members = turnover * max 0 (pscSwitchingCostRate config)
  where
    oldWeights = M.fromList [(pmUuid member, pmWeight member) | member <- incumbent]
    newWeights = M.fromList [(pcUuid candidate, weight) | (candidate, weight) <- members]
    uuids = M.keys (M.union oldWeights newWeights)
    memberDelta = sum [abs (M.findWithDefault 0 uuid newWeights - M.findWithDefault 0 uuid oldWeights) | uuid <- uuids]
    oldCash = 1 - sum (M.elems oldWeights)
    newCash = 1 - sum (M.elems newWeights)
    turnover = 0.5 * (memberDelta + abs (newCash - oldCash))

averagePairwiseCorrelation :: [M.Map Int64 Double] -> [Int64] -> Double
averagePairwiseCorrelation maps days =
    case correlations of
        [] -> 0
        _ -> sum correlations / fromIntegral (length correlations)
  where
    series = [[M.findWithDefault 0 day values | day <- days] | values <- maps]
    correlations = [correlation left right | (left : rest) <- tails series, right <- rest]

correlation :: [Double] -> [Double] -> Double
correlation left right
    | length left < 2 || length left /= length right = 0
    | denominator <= 0 = 0
    | otherwise = covariance / denominator
  where
    count = fromIntegral (length left)
    meanLeft = sum left / count
    meanRight = sum right / count
    centered = zipWith (\x y -> (x - meanLeft, y - meanRight)) left right
    covariance = sum [x * y | (x, y) <- centered]
    denominator = sqrt (sum [x * x | (x, _) <- centered] * sum [y * y | (_, y) <- centered])

evaluateSpec :: PortfolioSelectorConfig -> Int -> PortfolioSpec -> (PortfolioSpec, PortfolioMetrics)
evaluateSpec config seed spec =
    let (annualized, drawdowns) = bootstrapDistributions config (seed + specSeed spec) (pspSeries spec)
     in ( spec
        , PortfolioMetrics
            { pmHistoricalAnnualizedReturn = pspHistoricalAnnualizedReturn spec
            , pmHistoricalMaxDrawdown = pspHistoricalMaxDrawdown spec
            , pmAnnualizedReturnP10 = quantile 0.10 annualized
            , pmAnnualizedReturnP50 = quantile 0.50 annualized
            , pmAnnualizedReturnP90 = quantile 0.90 annualized
            , pmMaxDrawdownP95 = quantile 0.95 drawdowns
            , pmAverageCorrelation = pspAverageCorrelation spec
            , pmSwitchingCost = pspSwitchingCost spec
            , pmPairedOutperformanceProbability = 1
            }
        )

portfolioAdmitted :: PortfolioSelectorConfig -> PortfolioMetrics -> Bool
portfolioAdmitted config metrics =
    pmAnnualizedReturnP10 metrics > 0
        && pmMaxDrawdownP95 metrics <= pscMaxDrawdown config

bootstrapDistributions :: PortfolioSelectorConfig -> Int -> [Double] -> ([Double], [Double])
bootstrapDistributions config seed values = unzip (go (pscBootstrapSamples config) seed [])
  where
    go remaining state acc
        | remaining <= 0 = reverse acc
        | otherwise =
            let (sample, state') = movingBlockSample (pscBootstrapBlockDays config) state values
                result = (portfolioAnnualizedReturn sample, portfolioMaxDrawdown sample)
             in go (remaining - 1) state' (result : acc)

movingBlockSample :: Int -> Int -> [Double] -> ([Double], Int)
movingBlockSample blockDays seed values
    | null values = ([], nextSeed seed)
    | otherwise = go seed count []
  where
    count = length values
    valuesVector = V.fromList values
    block = max 1 blockDays
    go state remaining acc
        | remaining <= 0 = (reverse acc, state)
        | otherwise =
            let state' = nextSeed state
                start = state' `mod` count
                sampleCount = min block remaining
                blockValues = [valuesVector V.! ((start + offset) `mod` count) | offset <- [0 .. sampleCount - 1]]
             in go state' (remaining - sampleCount) (foldl' (flip (:)) acc blockValues)

nextSeed :: Int -> Int
nextSeed seed = fromInteger ((1664525 * toInteger (abs seed) + 1013904223) `mod` 4294967296)

quantile :: Double -> [Double] -> Double
quantile _ [] = -1
quantile probability values =
    let ordered = sort values
        probability' = max 0 (min 1 probability)
        index = floor (probability' * fromIntegral (length ordered - 1))
     in ordered !! index

stableSeed :: [PortfolioCandidate] -> Int
stableSeed =
    foldl'
        (\seed candidate -> nextSeed (seed + textSeed (pcUuid candidate)))
        2166136261
        . sortOn pcUuid

textSeed :: Text -> Int
textSeed = T.foldl' (\seed char -> nextSeed (seed + ord char)) 17

specSeed :: PortfolioSpec -> Int
specSeed = foldl' step 23 . pspMembers
  where
    step seed (candidate, weight) = nextSeed (seed + textSeed (pcUuid candidate) + round (weight * 10000))

incumbentOutperformanceProbability ::
    PortfolioSelectorConfig ->
    Int ->
    [PortfolioMember] ->
    [PortfolioCandidate] ->
    PortfolioSpec ->
    Double
incumbentOutperformanceProbability _ _ [] _ _ = 1
incumbentOutperformanceProbability config seed incumbent candidates challenger =
    case incumbentSpec of
        Nothing -> 1
        Just current -> pairedProbability config seed challenger current
  where
    byUuid = M.fromList [(pcUuid candidate, candidate) | candidate <- candidates]
    incumbentMembers =
        traverse
            (\member -> (,pmWeight member) <$> M.lookup (pmUuid member) byUuid)
            incumbent
    incumbentSpec = incumbentMembers >>= buildPortfolioSpec config{pscSwitchingCostRate = 0} []

pairedProbability :: PortfolioSelectorConfig -> Int -> PortfolioSpec -> PortfolioSpec -> Double
pairedProbability config seed challenger incumbent
    | count < pscMinimumObservations config = 0
    | otherwise = fromIntegral wins / fromIntegral samples
  where
    challengerByDay = M.fromList (zip (pspDays challenger) (pspSeries challenger))
    incumbentByDay = M.fromList (zip (pspDays incumbent) (pspSeries incumbent))
    commonDays = takeLast (pscMaximumObservations config) (filter (`M.member` incumbentByDay) (M.keys challengerByDay))
    challenger' = map (challengerByDay M.!) commonDays
    incumbent' = map (incumbentByDay M.!) commonDays
    count = length commonDays
    samples = pscBootstrapSamples config
    wins = go samples seed 0
    go remaining state total
        | remaining <= 0 = total
        | otherwise =
            let (challengerSample, state') = movingBlockSample (pscBootstrapBlockDays config) state challenger'
                (incumbentSample, _) = movingBlockSample (pscBootstrapBlockDays config) state incumbent'
                total' =
                    if portfolioAnnualizedReturn challengerSample > portfolioAnnualizedReturn incumbentSample
                        then total + 1
                        else total
             in go (remaining - 1) state' total'
