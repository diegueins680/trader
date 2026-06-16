module Trader.Formal.Optimization (
    FormalVerificationReport (..),
    TieBreakCandidate (..),
    activityCountFromMetrics,
    preferTieBreakImplementation,
    preferTieBreakSpec,
    roiImplementationScore,
    roiRequirementClauses,
    roiRequirementSummary,
    roiSpecScore,
    roiViewFromMetrics,
    rvActivityCount,
    tieBreakCandidateFromMetrics,
    verifyFormalOptimization,
) where

import Data.Maybe (isNothing)
import Data.Ord (Down (..))
import qualified Data.Vector as V

import Trader.Duration (positiveFiniteDuration)
import Trader.KalmanFusion (Kalman1 (..), initKalman1, updateMulti)
import Trader.Metrics (BacktestMetrics (..))
import Trader.SignalGates (defaultSignalGateConfig)
import Trader.Trading (
    EnsembleConfig (..),
    IntrabarFill (..),
    PositionSide (..),
    Positioning (..),
 )
import Trader.VolConfGate (
    VolConfGateBehavior (..),
    VolConfGateCell (..),
    VolConfGatePreset (..),
    defaultVolConfGateConfig,
    volConfGateCell,
 )

roiRequirementSummary :: String
roiRequirementSummary =
    "Maximize daily ROI without paying for fragility, churn, or near-idle capital artifacts."

roiRequirementClauses :: [String]
roiRequirementClauses =
    [ "Prefer higher annualized return as the repo's daily-ROI proxy."
    , "Penalize drawdown and tail loss."
    , "Penalize turnover."
    , "Use average trade return as a supporting signal only after at least one completed round trip clears the idle-capital floor."
    , "Reward faster payback only for finite strictly positive payback durations after the candidate clears the minimum completed-round-trip and idle-capital floors with positive expectancy; treat zero, negative, and non-finite payback exactly like missing data."
    , "Penalize low completed-round-trip activity and idle capital."
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
    , fvrActivityCountInvariant :: !Bool
    , fvrExposurePenaltyOrdered :: !Bool
    , fvrTieBreakTotalOrderAfterNormalization :: !Bool
    , fvrTieBreakHysteresisPreference :: !Bool
    , fvrTieBreakSpecMatchesImplementation :: !Bool
    , fvrOptimizerPublicSurfaceInvariant :: !Bool
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
        roundTripPenalty = roundTripPenaltyFor completedRoundTrips
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
            - roundTripPenalty
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
        lowRoundTripPenalty = roundTripPenaltyFor completedRoundTrips
        idleCapitalPenalty = exposurePenaltyFor (rvExposure view)
     in returnReward + expectancyReward + paybackReward - riskPenalty - turnoverPenalty - sparseActivityPenalty - lowRoundTripPenalty - idleCapitalPenalty

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
        lowActivityRoundTripOrdered =
            and
                [ lowActivityRoundTripMonotoneFor
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
                && lowActivityRoundTripOrdered
        activityCountInvariant =
            all activityCountInvariantFor allActivityCountMetrics
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
        lowExposurePenaltyOrdered =
            and
                [ idleExposurePenaltyOrderedFor
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
                && lowExposurePenaltyOrdered
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
            , fvrActivityCountInvariant = activityCountInvariant
            , fvrExposurePenaltyOrdered = exposurePenaltyOrdered
            , fvrTieBreakTotalOrderAfterNormalization =
                all tieBreakTotalOrderAfterNormalizationFor tieBreakPairs
            , fvrTieBreakHysteresisPreference = tieBreakHysteresisPreference
            , fvrTieBreakSpecMatchesImplementation =
                all tieBreakMatchesImplementationFor tieBreakPairs
            , fvrOptimizerPublicSurfaceInvariant = optimizerPublicSurfaceInvariant
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

-- Importing these constructors above makes the formal proof module fail at
-- compile time if `Trader.Trading` narrows its CLI-facing surface again.
-- The value-level checks keep the stable constructor contract explicit,
-- including the bundled legacy `SideLong`/`SideShort` import path.
tradingCliSurfaceInvariant :: Bool
tradingCliSurfaceInvariant =
    and
        [ positionSideSurfaceCode SideLong == "long"
        , positionSideSurfaceCode SideShort == "short"
        , intrabarFillSurfaceCode StopFirst == "stop-first"
        , intrabarFillSurfaceCode TakeProfitFirst == "take-profit-first"
        , positioningSurfaceCode LongFlat == "long-flat"
        , positioningSurfaceCode LongShort == "long-short"
        ]

positionSideSurfaceCode :: PositionSide -> String
positionSideSurfaceCode side =
    case side of
        SideLong -> "long"
        SideShort -> "short"

intrabarFillSurfaceCode :: IntrabarFill -> String
intrabarFillSurfaceCode intrabarFill =
    case intrabarFill of
        StopFirst -> "stop-first"
        TakeProfitFirst -> "take-profit-first"

positioningSurfaceCode :: Positioning -> String
positioningSurfaceCode positioning =
    case positioning of
        LongFlat -> "long-flat"
        LongShort -> "long-short"

-- Witness the public `EnsembleConfig` record surface that `Trader.Optimization`
-- relies on for visibility-only cfg/btCfg updates. Missing `ecMetaMask` must
-- stay represented as `Nothing` so fold handling remains fail-closed, and the
-- legacy hold/cooldown/drawdown knobs must remain visible without changing the
-- optimizer's active scoring inputs.
optimizerPublicSurfaceInvariant :: Bool
optimizerPublicSurfaceInvariant =
    let base = optimizerPublicSurfaceBaseConfig
        compatibilityCfg =
            base
                { ecMinHoldBars = 2
                , ecCooldownBars = 1
                , ecMaxHoldBars = Just 9
                , ecMaxDrawdown = Just 0.12
                }
        metaMask0 = Just (V.fromList [True, False, True])
        openTimes0 = Just (V.fromList [10, 11, 12])
        openPrices0 = Just (V.fromList [100.0, 101.5, 103.0])
        flipDisabled =
            base
                { ecLstmExitFlipBars = 0
                , ecLstmExitFlipGraceBars = 0
                }
        thresholdCfg =
            flipDisabled
                { ecOpenThreshold = 0.02
                , ecCloseThreshold = 0.01
                , ecMetaMask = metaMask0
                }
        foldCfg =
            thresholdCfg
                { ecOpenTimes = openTimes0
                , ecOpenPrices = openPrices0
                , ecMetaMask = Nothing
                }
     in -- The lint-driven rewrite from equality-to-`Nothing` to `isNothing`
        -- is extensional for the optimizer surface exercised below:
        -- `thresholdCfg` keeps a concrete mask, while `foldCfg` witnesses the
        -- absent-mask fold case that drives admissibility and search.
        and
            [ tradingCliSurfaceInvariant
            , optimizerCompatibilityFieldsNeutral base
            , optimizerCompatibilityFieldsNeutral flipDisabled
            , optimizerCompatibilityFieldsNeutral thresholdCfg
            , optimizerCompatibilityFieldsNeutral foldCfg
            , optimizerSurfacePreservedFields base compatibilityCfg
            , optimizerSurfacePreservedFields base flipDisabled
            , optimizerSurfacePreservedFields base thresholdCfg
            , optimizerSurfacePreservedFields base foldCfg
            , ecMinHoldBars compatibilityCfg == 2
            , ecCooldownBars compatibilityCfg == 1
            , ecMaxHoldBars compatibilityCfg == Just 9
            , ecMaxDrawdown compatibilityCfg == Just 0.12
            , ecOpenThreshold compatibilityCfg == ecOpenThreshold base
            , ecCloseThreshold compatibilityCfg == ecCloseThreshold base
            , ecLstmExitFlipBars compatibilityCfg == ecLstmExitFlipBars base
            , ecLstmExitFlipGraceBars compatibilityCfg == ecLstmExitFlipGraceBars base
            , ecMetaMask compatibilityCfg == ecMetaMask base
            , ecOpenTimes compatibilityCfg == ecOpenTimes base
            , ecOpenPrices compatibilityCfg == ecOpenPrices base
            , ecOpenThreshold flipDisabled == ecOpenThreshold base
            , ecCloseThreshold flipDisabled == ecCloseThreshold base
            , ecMetaMask flipDisabled == ecMetaMask base
            , ecOpenTimes thresholdCfg == ecOpenTimes base
            , ecOpenPrices thresholdCfg == ecOpenPrices base
            , ecLstmExitFlipBars flipDisabled == 0
            , ecLstmExitFlipGraceBars flipDisabled == 0
            , ecOpenThreshold thresholdCfg == 0.02
            , ecCloseThreshold thresholdCfg == 0.01
            , ecMetaMask thresholdCfg == metaMask0
            , ecOpenTimes foldCfg == openTimes0
            , ecOpenPrices foldCfg == openPrices0
            , isNothing (ecMetaMask foldCfg)
            ]

optimizerCompatibilityFieldsNeutral :: EnsembleConfig -> Bool
optimizerCompatibilityFieldsNeutral cfg =
    ecMinHoldBars cfg == 0
        && ecCooldownBars cfg == 0
        && isNothing (ecMaxHoldBars cfg)
        && isNothing (ecMaxDrawdown cfg)
        && not (ecKellyLiteSizing cfg)

optimizerSurfacePreservedFields :: EnsembleConfig -> EnsembleConfig -> Bool
optimizerSurfacePreservedFields base updated =
    and
        [ ecPeriodsPerYear updated == ecPeriodsPerYear base
        , ecMinEdge updated == ecMinEdge base
        , ecRouterLookback updated == ecRouterLookback base
        , ecRouterMinScore updated == ecRouterMinScore base
        , ecRouterScorePnlWeight updated == ecRouterScorePnlWeight base
        , ecFee updated == ecFee base
        , ecFeeFixed updated == ecFeeFixed base
        , ecFeeMin updated == ecFeeMin base
        , ecSlippage updated == ecSlippage base
        , ecSlippageVolMult updated == ecSlippageVolMult base
        , ecSlippageImpactPower updated == ecSlippageImpactPower base
        , ecSlippageImpact updated == ecSlippageImpact base
        , ecSpread updated == ecSpread base
        , ecSpreadVolMult updated == ecSpreadVolMult base
        , ecStopLoss updated == ecStopLoss base
        , ecTakeProfit updated == ecTakeProfit base
        , ecTrailingStop updated == ecTrailingStop base
        , ecStopLossVolMult updated == ecStopLossVolMult base
        , ecTakeProfitVolMult updated == ecTakeProfitVolMult base
        , ecTrailingStopVolMult updated == ecTrailingStopVolMult base
        , ecMaxPositionSize updated == ecMaxPositionSize base
        , ecSignalGateConfig updated == ecSignalGateConfig base
        , ecEntryEdgeSpikeAuditOnly updated == ecEntryEdgeSpikeAuditOnly base
        , ecEntryEdgeSpikeConsecutive updated == ecEntryEdgeSpikeConsecutive base
        , ecBlendWeight updated == ecBlendWeight base
        , ecBlendSoftmaxScale updated == ecBlendSoftmaxScale base
        , ecBlendNetSoftmaxScale updated == ecBlendNetSoftmaxScale base
        , ecBlendEdgePower updated == ecBlendEdgePower base
        , ecBlendSmoothAlpha updated == ecBlendSmoothAlpha base
        , ecBlendHedgeEta updated == ecBlendHedgeEta base
        , ecBlendHedgeMaxError updated == ecBlendHedgeMaxError base
        , ecBlendDivergenceK updated == ecBlendDivergenceK base
        , ecBlendRegimeHighVolCutoff updated == ecBlendRegimeHighVolCutoff base
        , ecBlendRegimeKalmanZCutoff updated == ecBlendRegimeKalmanZCutoff base
        , ecBlendBanditExploreScale updated == ecBlendBanditExploreScale base
        , ecBlendFractalReturnClamp updated == ecBlendFractalReturnClamp base
        , ecBlendFractalAlignedGain updated == ecBlendFractalAlignedGain base
        , ecBlendFractalConflictGain updated == ecBlendFractalConflictGain base
        , ecBlendCoherenceConflictFloor updated == ecBlendCoherenceConflictFloor base
        , ecBlendCoherenceConflictScale updated == ecBlendCoherenceConflictScale base
        , ecBlendCoherenceBoostThreshold updated == ecBlendCoherenceBoostThreshold base
        , ecBlendCoherenceBoostGain updated == ecBlendCoherenceBoostGain base
        , ecBlendCoherenceBoostSpan updated == ecBlendCoherenceBoostSpan base
        , ecBlendAnchorConflictBase updated == ecBlendAnchorConflictBase base
        , ecBlendAnchorConflictScale updated == ecBlendAnchorConflictScale base
        , ecBlendAnchorAlignedScale updated == ecBlendAnchorAlignedScale base
        , ecBlendTensionConflictShrink updated == ecBlendTensionConflictShrink base
        , ecBlendTensionNeutralShrink updated == ecBlendTensionNeutralShrink base
        , ecBlendEntropyConflictFloor updated == ecBlendEntropyConflictFloor base
        , ecBlendEntropyConflictScale updated == ecBlendEntropyConflictScale base
        , ecBlendEntropyAlignedBase updated == ecBlendEntropyAlignedBase base
        , ecBlendEntropyAlignedEntropyScale updated == ecBlendEntropyAlignedEntropyScale base
        , ecBlendPhaseCancelReturnClamp updated == ecBlendPhaseCancelReturnClamp base
        , ecBlendPhaseCancelConflictFloor updated == ecBlendPhaseCancelConflictFloor base
        , ecBlendPhaseCancelConflictScale updated == ecBlendPhaseCancelConflictScale base
        , ecBlendPhaseCancelAlignmentScale updated == ecBlendPhaseCancelAlignmentScale base
        , ecKalmanZMin updated == ecKalmanZMin base
        , ecKalmanZMax updated == ecKalmanZMax base
        , ecKellyLiteSizing updated == ecKellyLiteSizing base
        , ecKellyLiteFraction updated == ecKellyLiteFraction base
        , ecKellyLiteFloor updated == ecKellyLiteFloor base
        , ecKellyLiteCap updated == ecKellyLiteCap base
        ]

optimizerPublicSurfaceBaseConfig :: EnsembleConfig
optimizerPublicSurfaceBaseConfig =
    EnsembleConfig
        { ecPeriodsPerYear = 252
        , ecOpenThreshold = 0.015
        , ecCloseThreshold = 0.01
        , ecMinEdge = 0.001
        , ecRouterLookback = 8
        , ecRouterMinScore = 0.55
        , ecRouterScorePnlWeight = 0.25
        , ecFee = 0.001
        , ecFeeFixed = 0
        , ecFeeMin = 0
        , ecSlippage = 0.0005
        , ecSlippageVolMult = 0.1
        , ecSlippageImpactPower = 1
        , ecSlippageImpact = 0.01
        , ecSpread = 0.0002
        , ecSpreadVolMult = 0.05
        , ecStopLoss = Nothing
        , ecTakeProfit = Nothing
        , ecTrailingStop = Nothing
        , ecStopLossVolMult = 0
        , ecTakeProfitVolMult = 0
        , ecTrailingStopVolMult = 0
        , ecMinHoldBars = 0
        , ecCooldownBars = 0
        , ecMaxHoldBars = Nothing
        , ecMaxDrawdown = Nothing
        , ecMaxDailyLoss = Nothing
        , ecMaxWeeklyLoss = Nothing
        , ecRiskPerTrade = Nothing
        , ecMaxTradesPerDay = Nothing
        , ecExpectancyLookback = 0
        , ecMinExpectancy = Nothing
        , ecPerfLookback = 0
        , ecPerfMinWinRate = Nothing
        , ecPerfMinProfitFactor = Nothing
        , ecAdaptiveFilters = False
        , ecAdaptiveEdgeBufferMax = 0
        , ecAdaptiveMinSignalToNoiseMax = 0
        , ecAdaptiveKalmanZMinMax = 0
        , ecAdaptiveWinRateSlack = 0.05
        , ecAdaptiveProfitFactorSlack = 0.10
        , ecAdaptiveTrendLookbackMax = 0
        , ecLossStreakMax = 0
        , ecLossStreakCooldownBars = 0
        , ecNoTradeWindows = []
        , ecIntervalSeconds = Nothing
        , ecOpenTimes = Nothing
        , ecOpenPrices = Nothing
        , ecMetaMask = Nothing
        , ecPositioning = LongFlat
        , ecIntrabarFill = StopFirst
        , ecMaxPositionSize = 1
        , ecSignalGateConfig = defaultSignalGateConfig
        , ecEntryEdgeSpikeAuditOnly = False
        , ecEntryEdgeSpikeConsecutive = 0
        , ecMinSignalToNoise = 0
        , ecSnrSizeWeight = 0
        , ecThresholdFactorEnabled = False
        , ecThresholdFactorAlpha = 0
        , ecThresholdFactorMin = 1
        , ecThresholdFactorMax = 1
        , ecThresholdFactorFloor = 0
        , ecThresholdFactorEdgeKalWeight = 0
        , ecThresholdFactorEdgeLstmWeight = 0
        , ecThresholdFactorKalmanZWeight = 0
        , ecThresholdFactorHighVolWeight = 0
        , ecThresholdFactorConformalWeight = 0
        , ecThresholdFactorQuantileWeight = 0
        , ecThresholdFactorLstmConfWeight = 0
        , ecThresholdFactorLstmHealthWeight = 0
        , ecLstmTrainingHealth = Nothing
        , ecTrendLookback = 0
        , ecVolTarget = Nothing
        , ecVolLookback = 20
        , ecVolEwmaAlpha = Nothing
        , ecVolFloor = 0
        , ecVolScaleMax = 1
        , ecMaxVolatility = Nothing
        , ecVolConfGate = VolConfGateDisabled
        , ecVolConfGateConfig = defaultVolConfGateConfig
        , ecRebalanceBars = 0
        , ecRebalanceThreshold = 0
        , ecRebalanceGlobal = False
        , ecRebalanceResetOnSignal = False
        , ecFundingRate = 0
        , ecFundingBySide = False
        , ecFundingOnOpen = False
        , ecBlendWeight = 0.5
        , ecBlendSoftmaxScale = 600
        , ecBlendNetSoftmaxScale = 6000
        , ecBlendEdgePower = 1.0
        , ecBlendSmoothAlpha = 0.2
        , ecBlendHedgeEta = 6
        , ecBlendHedgeMaxError = 0.1
        , ecBlendDivergenceK = 4
        , ecBlendRegimeHighVolCutoff = 0.6
        , ecBlendRegimeKalmanZCutoff = 1
        , ecBlendBanditExploreScale = 0.25
        , ecBlendFractalReturnClamp = 0.75
        , ecBlendFractalAlignedGain = 1.12
        , ecBlendFractalConflictGain = 0.82
        , ecBlendCoherenceConflictFloor = 0.2
        , ecBlendCoherenceConflictScale = 0.5
        , ecBlendCoherenceBoostThreshold = 0.6
        , ecBlendCoherenceBoostGain = 0.35
        , ecBlendCoherenceBoostSpan = 0.4
        , ecBlendAnchorConflictBase = 0.6
        , ecBlendAnchorConflictScale = 0.4
        , ecBlendAnchorAlignedScale = 0.2
        , ecBlendTensionConflictShrink = 0.25
        , ecBlendTensionNeutralShrink = 0.5
        , ecBlendEntropyConflictFloor = 0.35
        , ecBlendEntropyConflictScale = 0.5
        , ecBlendEntropyAlignedBase = 0.95
        , ecBlendEntropyAlignedEntropyScale = 0.25
        , ecBlendPhaseCancelReturnClamp = 0.75
        , ecBlendPhaseCancelConflictFloor = 0.1
        , ecBlendPhaseCancelConflictScale = 0.6
        , ecBlendPhaseCancelAlignmentScale = 0.4
        , ecKalmanDt = 1
        , ecKalmanProcessVar = 1
        , ecKalmanMeasurementVar = 1
        , ecTriLayer = False
        , ecTriLayerFastMult = 1
        , ecTriLayerSlowMult = 1
        , ecTriLayerCloudPadding = 0
        , ecTriLayerCloudSlope = 0
        , ecTriLayerCloudWidth = 0
        , ecTriLayerTouchLookback = 0
        , ecTriLayerRequirePriceAction = False
        , ecTriLayerPriceActionBody = 0
        , ecTriLayerPriceActionWickRatio = 2
        , ecTriLayerPriceActionOppositeWickMax = 0.5
        , ecTriLayerPriceActionBodyTolerance = 0.2
        , ecTriLayerExitOnSlow = False
        , ecKalmanBandLookback = 0
        , ecKalmanBandStdMult = 0
        , ecKalmanZMin = 0.5
        , ecKalmanZMax = 2
        , ecKalmanMinStdFloor = 1e-6
        , ecLstmExitFlipBars = 3
        , ecLstmExitFlipGraceBars = 1
        , ecLstmExitFlipStrong = False
        , ecLstmConfidenceSoft = 0
        , ecLstmConfidenceHard = 0
        , ecMaxHighVolProb = Nothing
        , ecMaxConformalWidth = Nothing
        , ecMaxQuantileWidth = Nothing
        , ecConfirmConformal = False
        , ecConfirmQuantiles = False
        , ecConfidenceSizing = False
        , ecMinPositionSize = 0
        , ecKellyLiteSizing = False
        , ecKellyLiteFraction = 0.5
        , ecKellyLiteFloor = 0
        , ecKellyLiteCap = 1
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

activityCountInvariantFor :: BacktestMetrics -> Bool
activityCountInvariantFor metrics =
    let activityCount = activityCountFromMetrics metrics
        projectedActivityCount = rvActivityCount (roiViewFromMetrics metrics)
        expectedActivityCount = max 0 (max (bmRoundTrips metrics) (bmTradeCount metrics))
     in activityCount == expectedActivityCount
            && projectedActivityCount == activityCount
            && activityCount >= 0
            && activityCount >= max 0 (bmRoundTrips metrics)
            && activityCount >= max 0 (bmTradeCount metrics)

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

-- Keep completed-round-trip evidence monotone even when raw trade count stays high.
roundTripPenaltyFor :: Int -> Double
roundTripPenaltyFor completedRoundTrips
    | completedRoundTrips <= 0 = 0.08
    | completedRoundTrips < minimumRoiActivityFloor =
        0.02 * fromIntegral (minimumRoiActivityFloor - completedRoundTrips)
    | otherwise = 0

activityPenaltyFor :: Int -> Double
activityPenaltyFor activityCount
    | activityCount <= 0 = 0.25
    | activityCount < minimumRoiActivityFloor = fromIntegral (minimumRoiActivityFloor - activityCount) * 0.03
    | otherwise = 0

exposurePenaltyFor :: Double -> Double
exposurePenaltyFor exposure
    | exposure <= 0 = 0.05
    | exposure < minimumRoiExposureFloor =
        let gap = 1 - clamp 0 1 (exposure / minimumRoiExposureFloor)
         in 0.02 + 0.03 * gap
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
tieBreakKey :: TieBreakCandidate -> (Double, Down Double, Int, Double, Double, Double)
tieBreakKey candidate =
    let normalized = normalizeTieBreakCandidate candidate
     in ( tbcFinalEquity normalized
        , Down (tbcTurnover normalized)
        , tbcRoundTrips normalized
        , fromIntegral (tieBreakHysteresisRank normalized)
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

lowActivityRoundTripMonotoneFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Double -> Bool
lowActivityRoundTripMonotoneFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold tradeCount exposure =
    let scoreFor roundTrips =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        floorScore = scoreFor minimumRoiActivityFloor
     in nonDecreasing [scoreFor roundTrips | roundTrips <- lowActivityDomain]
            && and [scoreFor roundTrips <= floorScore + comparisonEps | roundTrips <- lowActivityDomain]

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

idleExposurePenaltyOrderedFor :: Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Int -> Int -> Bool
idleExposurePenaltyOrderedFor penaltyMaxDd penaltyTurnover annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount =
    let scoreFor exposure =
            scoreWith
                (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                penaltyMaxDd
                penaltyTurnover
        floorScore = scoreFor minimumRoiExposureFloor
     in nonDecreasing [scoreFor exposure | exposure <- idleExposureDomain]
            && and [scoreFor exposure <= floorScore + comparisonEps | exposure <- idleExposureDomain]

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

allActivityCountMetrics :: [BacktestMetrics]
allActivityCountMetrics =
    [ metricsFromState (RoiState 0 0 0 0 0 0 roundTrips tradeCount 0)
    | roundTrips <- activityInvariantDomain
    , tradeCount <- activityInvariantDomain
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

lowActivityDomain :: [Int]
lowActivityDomain = filter (< minimumRoiActivityFloor) activityDomain

activityInvariantDomain :: [Int]
activityInvariantDomain = [-2, -1] ++ activityDomain

exposureDomain :: [Double]
exposureDomain = [0.0, 0.005, 0.01]

idleExposureDomain :: [Double]
idleExposureDomain = filter (< minimumRoiExposureFloor) exposureDomain
