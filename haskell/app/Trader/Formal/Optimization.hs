module Trader.Formal.Optimization (
    FormalVerificationReport (..),
    TieBreakCandidate (..),
    preferTieBreakImplementation,
    preferTieBreakSpec,
    roiImplementationScore,
    roiRequirementClauses,
    roiRequirementSummary,
    roiSpecScore,
    tieBreakCandidateFromMetrics,
    verifyFormalOptimization,
) where

import Data.Ord (Down (..))

import Trader.Duration (positiveFiniteDuration)
import Trader.KalmanFusion (Kalman1 (..), initKalman1, updateMulti)
import Trader.Metrics (BacktestMetrics (..))
import Trader.VolConfGate (
    VolConfGateBehavior (..),
    VolConfGateCell (..),
    VolConfGatePreset (..),
    volConfGateCell,
 )

roiRequirementSummary :: String
roiRequirementSummary =
    "Maximize daily ROI without paying for fragility, churn, or idle no-trade states."

roiRequirementClauses :: [String]
roiRequirementClauses =
    [ "Prefer higher annualized return as the repo's daily-ROI proxy."
    , "Penalize drawdown and tail loss."
    , "Penalize turnover."
    , "Use average trade return as a supporting signal only after at least one completed round trip clears the idle-capital floor."
    , "Reward faster payback only for finite strictly positive payback durations after the candidate clears the minimum completed-round-trip and idle-capital floors with positive expectancy; treat zero, negative, and non-finite payback exactly like missing data."
    , "Penalize low activity and idle capital."
    ]

data TieBreakCandidate = TieBreakCandidate
    { tbcFinalEquity :: !Double
    , tbcTurnover :: !Double
    , tbcRoundTrips :: !Int
    , tbcOpenThreshold :: !Double
    , tbcCloseThreshold :: !Double
    }
    deriving (Eq, Show)

data FormalVerificationReport = FormalVerificationReport
    { fvrRoiStateCount :: !Int
    , fvrTieBreakPairCount :: !Int
    , fvrVolConfStateCount :: !Int
    , fvrKalmanFusionStateCount :: !Int
    , fvrRoiSpecMatchesImplementation :: !Bool
    , fvrReturnMonotone :: !Bool
    , fvrDrawdownMonotone :: !Bool
    , fvrTailLossMonotone :: !Bool
    , fvrTurnoverMonotone :: !Bool
    , fvrExpectancyMonotone :: !Bool
    , fvrPaybackMonotone :: !Bool
    , fvrInvalidPaybackMatchesMissing :: !Bool
    , fvrZeroRoundTripRewardInvariant :: !Bool
    , fvrActivityPenaltyOrdered :: !Bool
    , fvrExposurePenaltyOrdered :: !Bool
    , fvrTieBreakTotalOrderAfterNormalization :: !Bool
    , fvrTieBreakHysteresisPreference :: !Bool
    , fvrTieBreakSpecMatchesImplementation :: !Bool
    , fvrVolConfCanonicalizationInvariant :: !Bool
    , fvrVolConfMalformedVolMatchesMissing :: !Bool
    , fvrVolConfMalformedConfidenceMatchesWeak :: !Bool
    , fvrVolConfMalformedInputsStayConservative :: !Bool
    , fvrVolConfOutputBounded :: !Bool
    , fvrKalmanFusionMalformedMeasurementsIgnored :: !Bool
    , fvrKalmanFusionNoValidMeasurementsKeepPrior :: !Bool
    , fvrKalmanFusionPosteriorFinite :: !Bool
    , fvrKalmanFusionValidEvidenceShrinksVariance :: !Bool
    }
    deriving (Eq, Show)

data RoiState = RoiState
    { rsAnnualizedReturn :: !Double
    , rsMaxDrawdown :: !Double
    , rsTailLoss :: !Double
    , rsTurnover :: !Double
    , rsExpectancy :: !Double
    , rsAvgHold :: !Double
    , rsRoundTrips :: !Int
    , rsTradeCount :: !Int
    , rsExposure :: !Double
    }
    deriving (Eq, Show)

data RoiView = RoiView
    { rvAnnualizedReturn :: !Double
    , rvMaxDrawdown :: !Double
    , rvTailLoss :: !Double
    , rvTurnover :: !Double
    , rvExpectancy :: !Double
    , rvAvgHold :: !Double
    , rvRoundTrips :: !Int
    , rvActivityCount :: !Int
    , rvExposure :: !Double
    }
    deriving (Eq, Show)

roiImplementationScore :: Double -> Double -> BacktestMetrics -> Double
roiImplementationScore penaltyMaxDd penaltyTurnover m =
    let annRet = sanitizeFinite0 (bmAnnualizedReturn m)
        maxDd = max 0 (sanitizeFinite0 (bmMaxDrawdown m))
        tailLoss = max 0 (sanitizeFinite0 (bmCVaR95 m))
        turnover = max 0 (sanitizeFinite0 (bmTurnover m))
        expectancy = sanitizeFinite0 (bmAvgTradeReturn m)
        paybackDuration = positiveFiniteDuration (bmAvgHoldingPeriods m)
        exposure = max 0 (sanitizeFinite0 (bmExposure m))
        completedRoundTrips = completedRoundTripsFromMetrics m
        activityCount = activityCountFromMetrics m
        activityPenalty = activityPenaltyFor activityCount
        exposurePenalty = exposurePenaltyFor exposure
        expectancyReward = expectancyRewardFor completedRoundTrips exposure expectancy
        paybackReward = paybackRewardFor completedRoundTrips expectancy exposure paybackDuration
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
     in annRet
            - pDd * (maxDd + tailLoss)
            - pTurn * turnover
            + expectancyReward
            + paybackReward
            - activityPenalty
            - exposurePenalty

roiSpecScore :: Double -> Double -> BacktestMetrics -> Double
roiSpecScore penaltyMaxDd penaltyTurnover m =
    let view = roiViewFromMetrics m
        activityCount = rvActivityCount view
        completedRoundTrips = rvRoundTrips view
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
        returnReward = rvAnnualizedReturn view
        expectancyReward = expectancyRewardFor completedRoundTrips (rvExposure view) (rvExpectancy view)
        paybackReward = paybackRewardFor completedRoundTrips (rvExpectancy view) (rvExposure view) (positiveFiniteDuration (rvAvgHold view))
        riskPenalty = pDd * (rvMaxDrawdown view + rvTailLoss view)
        turnoverPenalty = pTurn * rvTurnover view
        sparseActivityPenalty = activityPenaltyFor activityCount
        idleCapitalPenalty = exposurePenaltyFor (rvExposure view)
     in returnReward + expectancyReward + paybackReward - riskPenalty - turnoverPenalty - sparseActivityPenalty - idleCapitalPenalty

tieBreakCandidateFromMetrics :: BacktestMetrics -> Double -> Double -> TieBreakCandidate
tieBreakCandidateFromMetrics metrics openThr closeThr =
    normalizeTieBreakCandidate
        ( TieBreakCandidate
            { tbcFinalEquity = bmFinalEquity metrics
            , tbcTurnover = bmTurnover metrics
            , tbcRoundTrips = bmRoundTrips metrics
            , tbcOpenThreshold = openThr
            , tbcCloseThreshold = closeThr
            }
        )

preferTieBreakImplementation :: TieBreakCandidate -> TieBreakCandidate -> Bool
preferTieBreakImplementation cand0 best0 =
    let eqEps = comparisonEps
        cand = normalizeTieBreakCandidate cand0
        best = normalizeTieBreakCandidate best0
        eq = tbcFinalEquity cand
        bestEq = tbcFinalEquity best
        turnover = tbcTurnover cand
        bestTurnover = tbcTurnover best
        roundTrips = tbcRoundTrips cand
        bestRoundTrips = tbcRoundTrips best
        openThr = tbcOpenThreshold cand
        closeThr = tbcCloseThreshold cand
        bestOpen = tbcOpenThreshold best
        bestClose = tbcCloseThreshold best
        inverted = isInvertedNormalized cand
        bestInverted = isInvertedNormalized best
     in (eq > bestEq + eqEps)
            || ( abs (eq - bestEq) <= eqEps
                    && ( turnover < bestTurnover - eqEps
                            || ( abs (turnover - bestTurnover) <= eqEps
                                    && ( roundTrips > bestRoundTrips
                                            || ( roundTrips == bestRoundTrips
                                                    && ( (not inverted && bestInverted)
                                                            || (inverted == bestInverted && (openThr, closeThr) > (bestOpen, bestClose))
                                                       )
                                               )
                                       )
                               )
                       )
               )

preferTieBreakSpec :: TieBreakCandidate -> TieBreakCandidate -> Bool
preferTieBreakSpec cand best = tieBreakKey cand > tieBreakKey best

verifyFormalOptimization :: FormalVerificationReport
verifyFormalOptimization =
    let roiInputs = allRoiInputs
        tieBreakPairs = [(cand, best) | cand <- tieBreakDomain, best <- tieBreakDomain]
        volConfInputs = allVolConfInputs
        kalmanFusionInputs = allKalmanFusionInputs
        zeroRoundTripRewardInvariant =
            and
                [ zeroRoundTripRewardInvariantFor
                    penaltyMaxDd
                    penaltyTurnover
                    annualizedReturn
                    maxDrawdown
                    tailLoss
                    turnover
                    tradeCount
                    exposure
                | penaltyMaxDd <- penaltyMaxDrawdownDomain
                , penaltyTurnover <- penaltyTurnoverDomain
                , annualizedReturn <- annualizedReturnDomain
                , maxDrawdown <- maxDrawdownDomain
                , tailLoss <- tailLossDomain
                , turnover <- turnoverDomain
                , tradeCount <- activityDomain
                , exposure <- exposureDomain
                ]
        activityPenaltyOrdered =
            and
                [ activityPenaltyOrderedFor
                    penaltyMaxDd
                    penaltyTurnover
                    annualizedReturn
                    maxDrawdown
                    tailLoss
                    turnover
                    expectancy
                    avgHold
                    tradeCount
                    exposure
                | penaltyMaxDd <- penaltyMaxDrawdownDomain
                , penaltyTurnover <- penaltyTurnoverDomain
                , annualizedReturn <- annualizedReturnDomain
                , maxDrawdown <- maxDrawdownDomain
                , tailLoss <- tailLossDomain
                , turnover <- turnoverDomain
                , expectancy <- expectancyDomain
                , avgHold <- avgHoldDomain
                , tradeCount <- activityDomain
                , exposure <- exposureDomain
                ]
                && zeroRoundTripRewardInvariant
        idleExposureRewardInvariant =
            and
                [ idleExposureRewardInvariantFor
                    penaltyMaxDd
                    penaltyTurnover
                    annualizedReturn
                    maxDrawdown
                    tailLoss
                    turnover
                    expectancy
                    avgHold
                    roundTrips
                    tradeCount
                | penaltyMaxDd <- penaltyMaxDrawdownDomain
                , penaltyTurnover <- penaltyTurnoverDomain
                , annualizedReturn <- annualizedReturnDomain
                , maxDrawdown <- maxDrawdownDomain
                , tailLoss <- tailLossDomain
                , turnover <- turnoverDomain
                , expectancy <- expectancyDomain
                , avgHold <- avgHoldDomain
                , roundTrips <- activityDomain
                , tradeCount <- activityDomain
                ]
        exposurePenaltyOrdered =
            and
                [ nonDecreasing
                    [ scoreWith
                        (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                        penaltyMaxDd
                        penaltyTurnover
                    | exposure <- exposureDomain
                    ]
                | penaltyMaxDd <- penaltyMaxDrawdownDomain
                , penaltyTurnover <- penaltyTurnoverDomain
                , annualizedReturn <- annualizedReturnDomain
                , maxDrawdown <- maxDrawdownDomain
                , tailLoss <- tailLossDomain
                , turnover <- turnoverDomain
                , expectancy <- expectancyDomain
                , avgHold <- avgHoldDomain
                , roundTrips <- activityDomain
                , tradeCount <- activityDomain
                ]
                && idleExposureRewardInvariant
        tieBreakHysteresisPreference =
            and
                [ tieBreakHysteresisPreferenceFor
                    (TieBreakCandidate finalEquity turnover roundTrips openThr openThr)
                    (TieBreakCandidate finalEquity turnover roundTrips openThr (openThr + 0.01))
                | finalEquity <- [1.0, 1.05, 1.1]
                , turnover <- [0.0, 0.1, 0.2]
                , roundTrips <- [0, 2, 4]
                , openThr <- [0.01, 0.02]
                ]
     in FormalVerificationReport
            { fvrRoiStateCount = length roiInputs
            , fvrTieBreakPairCount = length tieBreakPairs
            , fvrVolConfStateCount = length volConfInputs
            , fvrKalmanFusionStateCount = length kalmanFusionInputs
            , fvrRoiSpecMatchesImplementation =
                all roiSpecMatchesImplementationFor roiInputs
            , fvrReturnMonotone =
                and
                    [ nonDecreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | annualizedReturn <- annualizedReturnDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , avgHold <- avgHoldDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrDrawdownMonotone =
                and
                    [ nonIncreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | maxDrawdown <- maxDrawdownDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , avgHold <- avgHoldDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrTailLossMonotone =
                and
                    [ nonIncreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | tailLoss <- tailLossDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , avgHold <- avgHoldDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrTurnoverMonotone =
                and
                    [ nonIncreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | turnover <- turnoverDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , expectancy <- expectancyDomain
                    , avgHold <- avgHoldDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrExpectancyMonotone =
                and
                    [ nonDecreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | expectancy <- expectancyDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , avgHold <- avgHoldDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrPaybackMonotone =
                and
                    [ paybackExpectancyInvariantFor
                        penaltyMaxDd
                        penaltyTurnover
                        annualizedReturn
                        maxDrawdown
                        tailLoss
                        turnover
                        expectancy
                        roundTrips
                        tradeCount
                        exposure
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrInvalidPaybackMatchesMissing =
                and
                    [ invalidPaybackMatchesMissingFor
                        penaltyMaxDd
                        penaltyTurnover
                        annualizedReturn
                        maxDrawdown
                        tailLoss
                        turnover
                        expectancy
                        roundTrips
                        tradeCount
                        exposure
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , roundTrips <- activityDomain
                    , tradeCount <- activityDomain
                    , exposure <- exposureDomain
                    ]
            , fvrZeroRoundTripRewardInvariant = zeroRoundTripRewardInvariant
            , fvrActivityPenaltyOrdered = activityPenaltyOrdered
            , fvrExposurePenaltyOrdered = exposurePenaltyOrdered
            , fvrTieBreakTotalOrderAfterNormalization =
                all tieBreakTotalOrderAfterNormalizationFor tieBreakPairs
            , fvrTieBreakHysteresisPreference = tieBreakHysteresisPreference
            , fvrTieBreakSpecMatchesImplementation =
                all tieBreakMatchesImplementationFor tieBreakPairs
            , fvrVolConfCanonicalizationInvariant =
                all volConfCanonicalizationInvariantFor volConfInputs
            , fvrVolConfMalformedVolMatchesMissing =
                and
                    [ volConfMalformedVolMatchesMissingFor preset mConfidence
                    | preset <- volConfPresetDomain
                    , mConfidence <- volConfConfidenceDomain
                    ]
            , fvrVolConfMalformedConfidenceMatchesWeak =
                and
                    [ volConfMalformedConfidenceMatchesWeakFor preset mVolatility
                    | preset <- volConfPresetDomain
                    , mVolatility <- volConfVolatilityDomain
                    ]
            , fvrVolConfMalformedInputsStayConservative =
                and
                    [ volConfMalformedInputsStayConservativeFor preset volatility confidence
                    | preset <- volConfPresetDomain
                    , volatility <- volConfFiniteVolatilityDomain
                    , confidence <- volConfFiniteConfidenceDomain
                    ]
            , fvrVolConfOutputBounded =
                all volConfOutputBoundedFor volConfInputs
            , fvrKalmanFusionMalformedMeasurementsIgnored =
                all kalmanFusionMalformedMeasurementsIgnoredFor kalmanFusionInputs
            , fvrKalmanFusionNoValidMeasurementsKeepPrior =
                all kalmanFusionNoValidMeasurementsKeepPriorFor kalmanFusionInputs
            , fvrKalmanFusionPosteriorFinite =
                all kalmanFusionPosteriorFiniteFor kalmanFusionInputs
            , fvrKalmanFusionValidEvidenceShrinksVariance =
                all kalmanFusionValidEvidenceShrinksVarianceFor kalmanFusionInputs
            }

roiViewFromMetrics :: BacktestMetrics -> RoiView
roiViewFromMetrics m =
    RoiView
        { rvAnnualizedReturn = sanitizeFinite0 (bmAnnualizedReturn m)
        , rvMaxDrawdown = max 0 (sanitizeFinite0 (bmMaxDrawdown m))
        , rvTailLoss = max 0 (sanitizeFinite0 (bmCVaR95 m))
        , rvTurnover = max 0 (sanitizeFinite0 (bmTurnover m))
        , rvExpectancy = sanitizeFinite0 (bmAvgTradeReturn m)
        , rvAvgHold = bmAvgHoldingPeriods m
        , rvRoundTrips = completedRoundTripsFromMetrics m
        , rvActivityCount = activityCountFromMetrics m
        , rvExposure = max 0 (sanitizeFinite0 (bmExposure m))
        }

activityCountFromMetrics :: BacktestMetrics -> Int
activityCountFromMetrics metrics = max 0 (max (bmRoundTrips metrics) (bmTradeCount metrics))

completedRoundTripsFromMetrics :: BacktestMetrics -> Int
completedRoundTripsFromMetrics metrics = max 0 (bmRoundTrips metrics)

meetsRoiExpectancyFloor :: Int -> Double -> Bool
meetsRoiExpectancyFloor completedRoundTrips exposure =
    completedRoundTrips > 0 && exposure >= minimumRoiExposureFloor

expectancyRewardFor :: Int -> Double -> Double -> Double
expectancyRewardFor completedRoundTrips exposure expectancy
    | expectancy <= 0 = 0.5 * expectancy
    | not (meetsRoiExpectancyFloor completedRoundTrips exposure) = 0
    | otherwise = 0.5 * expectancy

paybackRewardFor :: Int -> Double -> Double -> Maybe Double -> Double
paybackRewardFor completedRoundTrips expectancy exposure mAvgHold =
    if not (meetsRoiPaybackFloor completedRoundTrips exposure) || expectancy <= 0
        then 0
        else maybe 0 paybackBonusFor mAvgHold

paybackBonusFor :: Double -> Double
paybackBonusFor avgHold =
    case positiveFiniteDuration avgHold of
        Just validAvgHold -> min 0.05 (1 / (1 + validAvgHold))
        Nothing -> 0

minimumRoiActivityFloor :: Int
minimumRoiActivityFloor = 3

minimumRoiExposureFloor :: Double
minimumRoiExposureFloor = 0.01

meetsRoiPaybackFloor :: Int -> Double -> Bool
meetsRoiPaybackFloor completedRoundTrips exposure =
    completedRoundTrips >= minimumRoiActivityFloor && exposure >= minimumRoiExposureFloor

activityPenaltyFor :: Int -> Double
activityPenaltyFor activityCount
    | activityCount <= 0 = 0.25
    | activityCount < minimumRoiActivityFloor = fromIntegral (minimumRoiActivityFloor - activityCount) * 0.03
    | otherwise = 0

exposurePenaltyFor :: Double -> Double
exposurePenaltyFor exposure
    | exposure <= 0 = 0.05
    | exposure < minimumRoiExposureFloor = 0.02
    | otherwise = 0

comparisonEps :: Double
comparisonEps = 1e-12

tieBreakNonFiniteHighSentinel :: Double
tieBreakNonFiniteHighSentinel = 1.0e308

tieBreakNonFiniteLowSentinel :: Double
tieBreakNonFiniteLowSentinel = -1.0e308

-- Compare a finite canonical form so malformed NaN/Infinity rows cannot
-- poison best-combo ordering and the proof model stays total.
normalizeTieBreakCandidate :: TieBreakCandidate -> TieBreakCandidate
normalizeTieBreakCandidate candidate =
    let (openThr, closeThr) =
            normalizeTieBreakThresholdPair
                (tbcOpenThreshold candidate)
                (tbcCloseThreshold candidate)
     in TieBreakCandidate
            { tbcFinalEquity =
                sanitizeFiniteWith tieBreakNonFiniteLowSentinel (tbcFinalEquity candidate)
            , tbcTurnover =
                sanitizeFiniteWith tieBreakNonFiniteHighSentinel (tbcTurnover candidate)
            , tbcRoundTrips = tbcRoundTrips candidate
            , tbcOpenThreshold = openThr
            , tbcCloseThreshold = closeThr
            }

normalizeTieBreakThresholdPair :: Double -> Double -> (Double, Double)
normalizeTieBreakThresholdPair openThr closeThr
    | isFinite openThr && isFinite closeThr = (openThr, closeThr)
    | otherwise = (tieBreakNonFiniteLowSentinel, tieBreakNonFiniteLowSentinel)

-- Keep the documented lexicographic threshold contract explicit:
-- after equity, turnover, and round trips tie, prefer
-- closeThreshold <= openThreshold hysteresis before threshold magnitude.
tieBreakKey :: TieBreakCandidate -> (Double, Down Double, Int, Int, Double, Double)
tieBreakKey candidate =
    let normalized = normalizeTieBreakCandidate candidate
     in ( tbcFinalEquity normalized
        , Down (tbcTurnover normalized)
        , tbcRoundTrips normalized
        , tieBreakHysteresisRank normalized
        , tbcOpenThreshold normalized
        , tbcCloseThreshold normalized
        )

tieBreakHysteresisRank :: TieBreakCandidate -> Int
tieBreakHysteresisRank candidate =
    if tbcCloseThreshold candidate <= tbcOpenThreshold candidate + comparisonEps
        then 1
        else 0

isInvertedNormalized :: TieBreakCandidate -> Bool
isInvertedNormalized candidate =
    tbcCloseThreshold candidate > tbcOpenThreshold candidate + comparisonEps

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)

sanitizeFiniteWith :: Double -> Double -> Double
sanitizeFiniteWith fallback x =
    if isFinite x
        then x
        else fallback

sanitizeFinite0 :: Double -> Double
sanitizeFinite0 = sanitizeFiniteWith 0

sanitizeFiniteMaybe :: Maybe Double -> Maybe Double
sanitizeFiniteMaybe mValue =
    case mValue of
        Just x | isFinite x -> Just x
        _ -> Nothing

clamp :: Double -> Double -> Double -> Double
clamp lo hi x = max lo (min hi x)

-- Mirror the production contract: non-finite and impossible negative
-- volatility collapse to missing data, while missing or malformed
-- confidence collapses to a weak finite score.
canonicalizeVolatilityInput :: Maybe Double -> Maybe Double
canonicalizeVolatilityInput mVolatility =
    case sanitizeFiniteMaybe mVolatility of
        Just rawVol
            | rawVol >= 0 -> Just rawVol
        _ -> Nothing

canonicalizeConfidenceInput :: Maybe Double -> Maybe Double
canonicalizeConfidenceInput mConfidence =
    case sanitizeFiniteMaybe mConfidence of
        Nothing -> Just 0.0
        Just rawConfidence -> Just (clamp 0 1 rawConfidence)

approxEq :: Double -> Double -> Bool
approxEq x y = abs (x - y) <= comparisonEps

scoreInput :: (Double -> Double -> BacktestMetrics -> Double) -> (Double, Double, RoiState) -> Double
scoreInput scorer (penaltyMaxDd, penaltyTurnover, state) =
    scorer penaltyMaxDd penaltyTurnover (metricsFromState state)

scoreWith :: RoiState -> Double -> Double -> Double
scoreWith state penaltyMaxDd penaltyTurnover =
    roiImplementationScore penaltyMaxDd penaltyTurnover (metricsFromState state)

roiSpecMatchesImplementationFor :: (Double, Double, RoiState) -> Bool
roiSpecMatchesImplementationFor input =
    approxEq
        (scoreInput roiSpecScore input)
        (scoreInput roiImplementationScore input)

tieBreakMatchesImplementationFor :: (TieBreakCandidate, TieBreakCandidate) -> Bool
tieBreakMatchesImplementationFor (cand, best) =
    preferTieBreakSpec cand best == preferTieBreakImplementation cand best

tieBreakTotalOrderAfterNormalizationFor :: (TieBreakCandidate, TieBreakCandidate) -> Bool
tieBreakTotalOrderAfterNormalizationFor (cand, best) =
    let candKey = tieBreakKey cand
        bestKey = tieBreakKey best
        candPreferred = preferTieBreakImplementation cand best
        bestPreferred = preferTieBreakImplementation best cand
     in candPreferred == (candKey > bestKey)
            && bestPreferred == (bestKey > candKey)
            && (candPreferred || bestPreferred || candKey == bestKey)
            && not (candPreferred && bestPreferred)

tieBreakHysteresisPreferenceFor :: TieBreakCandidate -> TieBreakCandidate -> Bool
tieBreakHysteresisPreferenceFor preferred inverted =
    preferTieBreakSpec preferred inverted
        && preferTieBreakImplementation preferred inverted
        && not (preferTieBreakSpec inverted preferred)
        && not (preferTieBreakImplementation inverted preferred)

volConfCanonicalizationInvariantFor :: (VolConfGatePreset, Maybe Double, Maybe Double) -> Bool
volConfCanonicalizationInvariantFor (preset, mVolatility, mConfidence) =
    volConfGateCell preset mVolatility mConfidence
        == volConfGateCell
            preset
            (canonicalizeVolatilityInput mVolatility)
            (canonicalizeConfidenceInput mConfidence)

volConfMalformedVolMatchesMissingFor :: VolConfGatePreset -> Maybe Double -> Bool
volConfMalformedVolMatchesMissingFor preset mConfidence =
    let canonicalConfidence = canonicalizeConfidenceInput mConfidence
        missingVolCell = volConfGateCell preset Nothing canonicalConfidence
     in and
            [ volConfGateCell preset (Just badVol) canonicalConfidence == missingVolCell
            | badVol <- malformedVolatilityDomain
            ]

volConfMalformedConfidenceMatchesWeakFor :: VolConfGatePreset -> Maybe Double -> Bool
volConfMalformedConfidenceMatchesWeakFor preset mVolatility =
    let weakConfidenceCell = volConfGateCell preset mVolatility Nothing
     in and
            [ volConfGateCell preset mVolatility (Just badConfidence) == weakConfidenceCell
            | badConfidence <- nonFiniteDomain
            ]

volConfMalformedInputsStayConservativeFor :: VolConfGatePreset -> Double -> Double -> Bool
volConfMalformedInputsStayConservativeFor preset volatility confidence =
    let canonicalVolatility = canonicalizeVolatilityInput (Just volatility)
        canonicalConfidence = canonicalizeConfidenceInput (Just confidence)
        missingVolBaseline = volConfGateCell preset Nothing canonicalConfidence
        weakConfidenceBaseline = volConfGateCell preset canonicalVolatility Nothing
        missingVolWeakConfidenceBaseline = volConfGateCell preset Nothing Nothing
     in and
            [ gateCellNoMorePermissiveThan
                (volConfGateCell preset (Just badVol) canonicalConfidence)
                missingVolBaseline
            | badVol <- malformedVolatilityDomain
            ]
            && and
                [ gateCellNoMorePermissiveThan
                    (volConfGateCell preset canonicalVolatility (Just badConfidence))
                    weakConfidenceBaseline
                | badConfidence <- nonFiniteDomain
                ]
            && and
                [ gateCellNoMorePermissiveThan
                    (volConfGateCell preset (Just badVol) (Just badConfidence))
                    missingVolWeakConfidenceBaseline
                | badVol <- malformedVolatilityDomain
                , badConfidence <- nonFiniteDomain
                ]

volConfOutputBoundedFor :: (VolConfGatePreset, Maybe Double, Maybe Double) -> Bool
volConfOutputBoundedFor (preset, mVolatility, mConfidence) =
    let sizeMult = vcgSizeMult (volConfGateCell preset mVolatility mConfidence)
     in isFinite sizeMult && sizeMult >= 0 && sizeMult <= 1

kalmanFusionMalformedMeasurementsIgnoredFor :: (Kalman1, [(Double, Double)]) -> Bool
kalmanFusionMalformedMeasurementsIgnoredFor (prior, measurements) =
    kalmanApproxEq
        (updateMulti measurements prior)
        (updateMulti (kalmanValidMeasurements measurements) prior)

kalmanFusionNoValidMeasurementsKeepPriorFor :: (Kalman1, [(Double, Double)]) -> Bool
kalmanFusionNoValidMeasurementsKeepPriorFor (prior, measurements) =
    let validMeasurements = kalmanValidMeasurements measurements
     in not (null validMeasurements)
            || kalmanApproxEq (updateMulti measurements prior) prior

kalmanFusionPosteriorFiniteFor :: (Kalman1, [(Double, Double)]) -> Bool
kalmanFusionPosteriorFiniteFor (prior, measurements) =
    let post = updateMulti measurements prior
     in isFinite (kMean post)
            && isFinite (kVar post)
            && kVar post > 0
            && isFinite (kProcessVar post)

kalmanFusionValidEvidenceShrinksVarianceFor :: (Kalman1, [(Double, Double)]) -> Bool
kalmanFusionValidEvidenceShrinksVarianceFor (prior, measurements) =
    let validMeasurements = kalmanValidMeasurements measurements
     in null validMeasurements
            || kVar (updateMulti measurements prior) <= kVar prior + comparisonEps

kalmanValidMeasurements :: [(Double, Double)] -> [(Double, Double)]
kalmanValidMeasurements = filter kalmanMeasurementValid

kalmanMeasurementValid :: (Double, Double) -> Bool
kalmanMeasurementValid (y, r) = isFinite y && isFinite r && r > 0

kalmanApproxEq :: Kalman1 -> Kalman1 -> Bool
kalmanApproxEq lhs rhs =
    approxEq (kMean lhs) (kMean rhs)
        && approxEq (kVar lhs) (kVar rhs)
        && approxEq (kProcessVar lhs) (kProcessVar rhs)

gateCellNoMorePermissiveThan :: VolConfGateCell -> VolConfGateCell -> Bool
gateCellNoMorePermissiveThan candidate baseline =
    let candidateRank = gateCellPermissivenessRank (vcgBehavior candidate)
        baselineRank = gateCellPermissivenessRank (vcgBehavior baseline)
        candidateSize = vcgSizeMult candidate
        baselineSize = vcgSizeMult baseline
     in candidateRank < baselineRank
            || (candidateRank == baselineRank && candidateSize <= baselineSize + comparisonEps)

-- `Block` and `AllowExitOnly` are both reduce-only in the production gate,
-- so the conservative order treats them equally.
gateCellPermissivenessRank :: VolConfGateBehavior -> Int
gateCellPermissivenessRank behavior =
    case behavior of
        VolConfGateBlock -> 0
        VolConfGateAllowExitOnly -> 0
        VolConfGateHold -> 1
        VolConfGateAllowEntry -> 2

zeroRoundTripRewardInvariantFor :: Double -> Double -> Double -> Double -> Double -> Double -> Int -> Double -> Bool
zeroRoundTripRewardInvariantFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover tradeCount exposure =
    let scoreFor roundTrips expectancy avgHold =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        baseline = scoreFor 0 0 0
     in and
            [ let zeroScore = scoreFor 0 expectancy avgHold
                  activeScore = scoreFor 1 expectancy avgHold
               in zeroScore <= activeScore + comparisonEps
            | expectancy <- expectancyDomain
            , avgHold <- avgHoldDomain
            ]
            && and
                [ approxEq baseline (scoreFor 0 expectancy avgHold)
                | expectancy <- positiveExpectancyDomain
                , avgHold <- avgHoldDomain
                ]

paybackExpectancyInvariantFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Double -> Bool
paybackExpectancyInvariantFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy roundTrips tradeCount exposure =
    let completedRoundTrips = max 0 roundTrips
        scoreFor avgHold =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        paybackScores = [scoreFor avgHold | avgHold <- positiveAvgHoldDomain]
     in if expectancy <= 0 || not (meetsRoiPaybackFloor completedRoundTrips exposure)
            then allApproxEq paybackScores
            else nonIncreasing paybackScores

-- Missing payback reaches the scorer as `Nothing`; zero is the proof-model
-- stand-in because the shared duration guard rejects it the same way.
invalidPaybackMatchesMissingFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Double -> Bool
invalidPaybackMatchesMissingFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy roundTrips tradeCount exposure =
    let missingScore =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy 0 roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        scoreFor avgHold =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
     in and
            [ approxEq missingScore (scoreFor avgHold)
            | avgHold <- invalidAvgHoldDomain
            ]

activityPenaltyOrderedFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Double -> Bool
activityPenaltyOrderedFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold tradeCount exposure =
    let scoreFor roundTrips =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
     in nonDecreasing [scoreFor roundTrips | roundTrips <- activityDomain]
            && and [scoreFor 0 <= scoreFor roundTrips | roundTrips <- positiveActivityDomain]

idleExposureRewardInvariantFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Bool
idleExposureRewardInvariantFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount =
    let activeExposure = minimumRoiExposureFloor
        scoreFor exposure =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        activeScore = scoreFor activeExposure
     in and
            [ scoreFor exposure <= activeScore + comparisonEps
            | exposure <- idleExposureDomain
            ]

metricsFromState :: RoiState -> BacktestMetrics
metricsFromState state =
    BacktestMetrics
        { bmPeriods = 0
        , bmFinalEquity = 1
        , bmTotalReturn = 0
        , bmAnnualizedReturn = rsAnnualizedReturn state
        , bmAnnualizedVolatility = 0
        , bmSharpe = 0
        , bmSortino = 0
        , bmCalmar = 0
        , bmDownsideVolatility = 0
        , bmVaR95 = 0
        , bmCVaR95 = rsTailLoss state
        , bmMaxDrawdown = rsMaxDrawdown state
        , bmPositionChanges = 0
        , bmTradeCount = rsTradeCount state
        , bmRoundTrips = rsRoundTrips state
        , bmWinRate = 0
        , bmGrossProfit = 0
        , bmGrossLoss = 0
        , bmProfitFactor = Nothing
        , bmAvgTradeReturn = rsExpectancy state
        , bmAvgHoldingPeriods = rsAvgHold state
        , bmExposure = rsExposure state
        , bmAgreementRate = 0
        , bmTurnover = rsTurnover state
        }

allApproxEq :: [Double] -> Bool
allApproxEq [] = True
allApproxEq (x : xs) = all (approxEq x) xs

nonDecreasing :: [Double] -> Bool
nonDecreasing xs = and (zipWith (\x y -> x <= y + comparisonEps) xs (drop 1 xs))

nonIncreasing :: [Double] -> Bool
nonIncreasing xs = and (zipWith (\x y -> x + comparisonEps >= y) xs (drop 1 xs))

allRoiInputs :: [(Double, Double, RoiState)]
allRoiInputs =
    [ (penaltyMaxDd, penaltyTurnover, state)
    | penaltyMaxDd <- penaltyMaxDrawdownDomain
    , penaltyTurnover <- penaltyTurnoverDomain
    , state <- allRoiStates
    ]

allRoiStates :: [RoiState]
allRoiStates =
    [ RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure
    | annualizedReturn <- annualizedReturnDomain
    , maxDrawdown <- maxDrawdownDomain
    , tailLoss <- tailLossDomain
    , turnover <- turnoverDomain
    , expectancy <- expectancyDomain
    , avgHold <- avgHoldDomain
    , roundTrips <- activityDomain
    , tradeCount <- activityDomain
    , exposure <- exposureDomain
    ]

tieBreakDomain :: [TieBreakCandidate]
tieBreakDomain = tieBreakFiniteDomain ++ tieBreakMalformedDomain

tieBreakFiniteDomain :: [TieBreakCandidate]
tieBreakFiniteDomain =
    [ TieBreakCandidate finalEquity turnover roundTrips openThr closeThr
    | finalEquity <- [1.0, 1.05, 1.1]
    , turnover <- [0.0, 0.1, 0.2]
    , roundTrips <- [0, 2, 4]
    , openThr <- [0.01, 0.02]
    , closeThr <- [0.01, 0.03]
    ]

tieBreakMalformedDomain :: [TieBreakCandidate]
tieBreakMalformedDomain =
    [ TieBreakCandidate finalEquity 0.1 2 0.01 0.01
    | finalEquity <- nonFiniteDomain
    ]
        ++ [ TieBreakCandidate 1.05 turnover 2 0.01 0.01
           | turnover <- nonFiniteDomain
           ]
        ++ [ TieBreakCandidate 1.05 0.1 2 openThr 0.01
           | openThr <- nonFiniteDomain
           ]
        ++ [ TieBreakCandidate 1.05 0.1 2 0.01 closeThr
           | closeThr <- nonFiniteDomain
           ]

allVolConfInputs :: [(VolConfGatePreset, Maybe Double, Maybe Double)]
allVolConfInputs =
    [ (preset, mVolatility, mConfidence)
    | preset <- volConfPresetDomain
    , mVolatility <- volConfVolatilityDomain
    , mConfidence <- volConfConfidenceDomain
    ]

volConfPresetDomain :: [VolConfGatePreset]
volConfPresetDomain =
    [ VolConfGateDisabled
    , VolConfGateV1Default
    , VolConfGateV1HighVolTighter
    , VolConfGateV1HighVolLooser
    , VolConfGateV1ConfStricter
    ]

volConfFiniteVolatilityDomain :: [Double]
volConfFiniteVolatilityDomain = [-1.0, 0.0, 0.25, 0.5, 1.0, 1.2, 1.4, 2.0]

negativeFiniteVolatilityDomain :: [Double]
negativeFiniteVolatilityDomain = filter (< 0) volConfFiniteVolatilityDomain

malformedVolatilityDomain :: [Double]
malformedVolatilityDomain = negativeFiniteVolatilityDomain ++ nonFiniteDomain

volConfVolatilityDomain :: [Maybe Double]
volConfVolatilityDomain =
    [Nothing]
        ++ map Just volConfFiniteVolatilityDomain
        ++ map Just nonFiniteDomain

volConfFiniteConfidenceDomain :: [Double]
volConfFiniteConfidenceDomain = [-0.5, 0.0, 0.59, 0.60, 0.64, 0.65, 0.79, 0.80, 1.0, 1.5]

volConfConfidenceDomain :: [Maybe Double]
volConfConfidenceDomain =
    [Nothing]
        ++ map Just volConfFiniteConfidenceDomain
        ++ map Just nonFiniteDomain

nonFiniteDomain :: [Double]
nonFiniteDomain = [nanValue, positiveInfinity, negativeInfinity]

nanValue :: Double
nanValue = 0 / 0

positiveInfinity :: Double
positiveInfinity = 1 / 0

negativeInfinity :: Double
negativeInfinity = negate positiveInfinity

allKalmanFusionInputs :: [(Kalman1, [(Double, Double)])]
allKalmanFusionInputs =
    [ (prior, measurements)
    | prior <- kalmanPriorDomain
    , measurements <- kalmanMeasurementListDomain
    ]

kalmanPriorDomain :: [Kalman1]
kalmanPriorDomain =
    [ initKalman1 mean0 var0 processVar
    | mean0 <- [-0.05, 0.0, 0.05]
    , var0 <- [1.0e-6, 1.0e-3, 1.0]
    , processVar <- [0.0, 1.0e-4]
    ]

kalmanMeasurementListDomain :: [[(Double, Double)]]
kalmanMeasurementListDomain =
    [[]]
        ++ [[m0] | m0 <- kalmanMeasurementDomain]
        ++ [[m0, m1] | m0 <- kalmanMeasurementDomain, m1 <- kalmanMeasurementDomain]

kalmanMeasurementDomain :: [(Double, Double)]
kalmanMeasurementDomain =
    kalmanValidMeasurementDomain ++ kalmanMalformedMeasurementDomain

kalmanValidMeasurementDomain :: [(Double, Double)]
kalmanValidMeasurementDomain =
    [ (y, r)
    | y <- [-0.1, 0.0, 0.1]
    , r <- [1.0e-18, 1.0e-6, 1.0e-3, 0.1]
    ]

kalmanMalformedMeasurementDomain :: [(Double, Double)]
kalmanMalformedMeasurementDomain =
    [ (badY, 0.1)
    | badY <- nonFiniteDomain
    ]
        ++ [ (0.0, badR)
           | badR <- [0.0, -0.1] ++ nonFiniteDomain
           ]

penaltyMaxDrawdownDomain :: [Double]
penaltyMaxDrawdownDomain = [0.0, 1.5]

penaltyTurnoverDomain :: [Double]
penaltyTurnoverDomain = [0.0, 0.2]

annualizedReturnDomain :: [Double]
annualizedReturnDomain = [-0.25, 0.0, 0.5]

maxDrawdownDomain :: [Double]
maxDrawdownDomain = [0.0, 0.1, 0.3]

tailLossDomain :: [Double]
tailLossDomain = [0.0, 0.05, 0.2]

turnoverDomain :: [Double]
turnoverDomain = [0.0, 0.2, 0.6]

expectancyDomain :: [Double]
expectancyDomain = [-0.1, 0.0, 0.1]

positiveExpectancyDomain :: [Double]
positiveExpectancyDomain = filter (> 0) expectancyDomain

avgHoldDomain :: [Double]
avgHoldDomain = [0.0, 1.0, 39.0]

positiveAvgHoldDomain :: [Double]
positiveAvgHoldDomain = [1.0, 39.0]

invalidAvgHoldDomain :: [Double]
invalidAvgHoldDomain = [0.0, -39.0, -1.0] ++ nonFiniteDomain

activityDomain :: [Int]
activityDomain = [0, 1, 2, 3]

positiveActivityDomain :: [Int]
positiveActivityDomain = filter (> 0) activityDomain

exposureDomain :: [Double]
exposureDomain = [0.0, 0.005, 0.01]

idleExposureDomain :: [Double]
idleExposureDomain = filter (< minimumRoiExposureFloor) exposureDomain