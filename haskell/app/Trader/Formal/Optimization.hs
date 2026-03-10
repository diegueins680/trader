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

import Trader.Metrics (BacktestMetrics (..))

roiRequirementSummary :: String
roiRequirementSummary =
    "Maximize daily ROI without paying for fragility, churn, or inactivity."

roiRequirementClauses :: [String]
roiRequirementClauses =
    [ "Prefer higher annualized return as the repo's daily-ROI proxy."
    , "Penalize drawdown and tail loss."
    , "Penalize turnover."
    , "Reward positive expectancy."
    , "Reward faster payback."
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
    , fvrRoiSpecMatchesImplementation :: !Bool
    , fvrReturnMonotone :: !Bool
    , fvrDrawdownMonotone :: !Bool
    , fvrTailLossMonotone :: !Bool
    , fvrTurnoverMonotone :: !Bool
    , fvrExpectancyMonotone :: !Bool
    , fvrPaybackMonotone :: !Bool
    , fvrActivityPenaltyOrdered :: !Bool
    , fvrExposurePenaltyOrdered :: !Bool
    , fvrTieBreakSpecMatchesImplementation :: !Bool
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
    , rvActivityCount :: !Int
    , rvExposure :: !Double
    }
    deriving (Eq, Show)

roiImplementationScore :: Double -> Double -> BacktestMetrics -> Double
roiImplementationScore penaltyMaxDd penaltyTurnover m =
    let bad x = isNaN x || isInfinite x
        clean x =
            if bad x
                then 0
                else x
        annRet = clean (bmAnnualizedReturn m)
        maxDd = max 0 (clean (bmMaxDrawdown m))
        tailLoss = max 0 (clean (bmCVaR95 m))
        turnover = max 0 (clean (bmTurnover m))
        expectancy = clean (bmAvgTradeReturn m)
        avgHold = max 0 (clean (bmAvgHoldingPeriods m))
        roundTrips = max 0 (bmRoundTrips m)
        tradeCount = max 0 (bmTradeCount m)
        exposure = max 0 (clean (bmExposure m))
        activityCount = max roundTrips tradeCount
        activityPenalty
            | activityCount <= 0 = 0.25
            | activityCount < 3 = fromIntegral (3 - activityCount) * 0.03
            | otherwise = 0
        exposurePenalty
            | exposure <= 0 = 0.05
            | exposure < 0.01 = 0.02
            | otherwise = 0
        paybackBonus =
            if avgHold <= 0
                then 0
                else min 0.05 (1 / (1 + avgHold))
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
     in annRet
            - pDd * (maxDd + tailLoss)
            - pTurn * turnover
            + 0.5 * expectancy
            + paybackBonus
            - activityPenalty
            - exposurePenalty

roiSpecScore :: Double -> Double -> BacktestMetrics -> Double
roiSpecScore penaltyMaxDd penaltyTurnover m =
    let view = roiViewFromMetrics m
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
        returnReward = rvAnnualizedReturn view
        expectancyReward = 0.5 * rvExpectancy view
        paybackReward = paybackBonusFor (rvAvgHold view)
        riskPenalty = pDd * (rvMaxDrawdown view + rvTailLoss view)
        turnoverPenalty = pTurn * rvTurnover view
        sparseActivityPenalty = activityPenaltyFor (rvActivityCount view)
        idleCapitalPenalty = exposurePenaltyFor (rvExposure view)
     in returnReward + expectancyReward + paybackReward - riskPenalty - turnoverPenalty - sparseActivityPenalty - idleCapitalPenalty

tieBreakCandidateFromMetrics :: BacktestMetrics -> Double -> Double -> TieBreakCandidate
tieBreakCandidateFromMetrics metrics openThr closeThr =
    TieBreakCandidate
        { tbcFinalEquity = bmFinalEquity metrics
        , tbcTurnover = bmTurnover metrics
        , tbcRoundTrips = bmRoundTrips metrics
        , tbcOpenThreshold = openThr
        , tbcCloseThreshold = closeThr
        }

preferTieBreakImplementation :: TieBreakCandidate -> TieBreakCandidate -> Bool
preferTieBreakImplementation cand best =
    let eqEps = 1e-12
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
        inverted = closeThr > openThr + eqEps
        bestInverted = bestClose > bestOpen + eqEps
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
     in FormalVerificationReport
            { fvrRoiStateCount = length roiInputs
            , fvrTieBreakPairCount = length tieBreakPairs
            , fvrRoiSpecMatchesImplementation =
                all
                    ( \input ->
                        approxEq
                            (scoreInput roiSpecScore input)
                            (scoreInput roiImplementationScore input)
                    )
                    roiInputs
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
                    [ nonIncreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold roundTrips tradeCount exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | avgHold <- positiveAvgHoldDomain
                        ]
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
            , fvrActivityPenaltyOrdered =
                and
                    [ nonDecreasing
                        [ scoreWith
                            (RoiState annualizedReturn maxDrawdown tailLoss turnover expectancy avgHold activity activity exposure)
                            penaltyMaxDd
                            penaltyTurnover
                        | activity <- activityDomain
                        ]
                    | penaltyMaxDd <- penaltyMaxDrawdownDomain
                    , penaltyTurnover <- penaltyTurnoverDomain
                    , annualizedReturn <- annualizedReturnDomain
                    , maxDrawdown <- maxDrawdownDomain
                    , tailLoss <- tailLossDomain
                    , turnover <- turnoverDomain
                    , expectancy <- expectancyDomain
                    , avgHold <- avgHoldDomain
                    , exposure <- exposureDomain
                    ]
            , fvrExposurePenaltyOrdered =
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
            , fvrTieBreakSpecMatchesImplementation =
                all
                    (\(cand, best) -> preferTieBreakSpec cand best == preferTieBreakImplementation cand best)
                    tieBreakPairs
            }

roiViewFromMetrics :: BacktestMetrics -> RoiView
roiViewFromMetrics m =
    RoiView
        { rvAnnualizedReturn = sanitizeFinite0 (bmAnnualizedReturn m)
        , rvMaxDrawdown = max 0 (sanitizeFinite0 (bmMaxDrawdown m))
        , rvTailLoss = max 0 (sanitizeFinite0 (bmCVaR95 m))
        , rvTurnover = max 0 (sanitizeFinite0 (bmTurnover m))
        , rvExpectancy = sanitizeFinite0 (bmAvgTradeReturn m)
        , rvAvgHold = max 0 (sanitizeFinite0 (bmAvgHoldingPeriods m))
        , rvActivityCount = max (max 0 (bmRoundTrips m)) (max 0 (bmTradeCount m))
        , rvExposure = max 0 (sanitizeFinite0 (bmExposure m))
        }

paybackBonusFor :: Double -> Double
paybackBonusFor avgHold =
    if avgHold <= 0
        then 0
        else min 0.05 (1 / (1 + avgHold))

activityPenaltyFor :: Int -> Double
activityPenaltyFor activityCount
    | activityCount <= 0 = 0.25
    | activityCount < 3 = fromIntegral (3 - activityCount) * 0.03
    | otherwise = 0

exposurePenaltyFor :: Double -> Double
exposurePenaltyFor exposure
    | exposure <= 0 = 0.05
    | exposure < 0.01 = 0.02
    | otherwise = 0

tieBreakKey :: TieBreakCandidate -> (Double, Down Double, Int, Int, Double, Double)
tieBreakKey candidate =
    ( tbcFinalEquity candidate
    , Down (tbcTurnover candidate)
    , tbcRoundTrips candidate
    , if isInverted candidate then 0 else 1
    , tbcOpenThreshold candidate
    , tbcCloseThreshold candidate
    )

isInverted :: TieBreakCandidate -> Bool
isInverted candidate = tbcCloseThreshold candidate > tbcOpenThreshold candidate

sanitizeFinite0 :: Double -> Double
sanitizeFinite0 x =
    if isNaN x || isInfinite x
        then 0
        else x

approxEq :: Double -> Double -> Bool
approxEq x y = abs (x - y) <= 1e-12

scoreInput :: (Double -> Double -> BacktestMetrics -> Double) -> (Double, Double, RoiState) -> Double
scoreInput scorer (penaltyMaxDd, penaltyTurnover, state) =
    scorer penaltyMaxDd penaltyTurnover (metricsFromState state)

scoreWith :: RoiState -> Double -> Double -> Double
scoreWith state penaltyMaxDd penaltyTurnover =
    roiImplementationScore penaltyMaxDd penaltyTurnover (metricsFromState state)

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

nonDecreasing :: [Double] -> Bool
nonDecreasing xs = and (zipWith (<=) xs (drop 1 xs))

nonIncreasing :: [Double] -> Bool
nonIncreasing xs = and (zipWith (>=) xs (drop 1 xs))

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
tieBreakDomain =
    [ TieBreakCandidate finalEquity turnover roundTrips openThr closeThr
    | finalEquity <- [1.0, 1.05, 1.1]
    , turnover <- [0.0, 0.1, 0.2]
    , roundTrips <- [0, 2, 4]
    , openThr <- [0.01, 0.02]
    , closeThr <- [0.01, 0.03]
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

avgHoldDomain :: [Double]
avgHoldDomain = [0.0, 1.0, 39.0]

positiveAvgHoldDomain :: [Double]
positiveAvgHoldDomain = [1.0, 39.0]

activityDomain :: [Int]
activityDomain = [0, 1, 2, 3]

exposureDomain :: [Double]
exposureDomain = [0.0, 0.005, 0.01]
