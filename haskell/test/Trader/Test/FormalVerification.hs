module Trader.Test.FormalVerification (
    formalVerificationSuite,
) where

import Control.Monad (forM_, unless)
import Data.Int (Int64)
import Data.List (isPrefixOf)
import qualified Data.Map.Strict as Map
import Data.Maybe (isNothing)
import qualified Data.Text as T
import Trader.ExternalData (externalSymbolMatches)
import qualified Trader.Formal.Execution as Execution
import qualified Trader.Formal.Optimization as Optimization
import qualified Trader.Formal.Risk as Risk
import Trader.GateTelemetry (
    GateName (..),
    GateRejection (..),
    GateTelemetry (..),
    RejectionReason (..),
    emptyTelemetry,
    recordRejection,
    recordRejectionWithContext,
 )
import Trader.MarketDataIntegrity (
    MarketSeriesBar (..),
    marketDataContinuationIssue,
    marketDataFreshness,
    marketDataStaleReason,
    normalizeClosedMarketSeries,
    validateMarketSeriesContinuity,
 )
import Trader.PortfolioSelection (
    PortfolioGraduationConfig (..),
    PortfolioGraduationDecision (..),
    PortfolioGraduationEvidence (..),
    PortfolioGraduationReview (..),
    defaultPortfolioGraduationConfig,
    portfolioGraduationFleetEquities,
    portfolioGraduationPerformance,
    portfolioGraduationReview,
 )
import Trader.Trading (ExitReason (..), HaltInputs (..), specRiskHalt)


formalVerificationSuite :: [(String, IO ())]
formalVerificationSuite =
    [ ("market timestamps fail closed on Int64 overflow", testMarketTimestampOverflow)
    , ("risk metrics and loss-streak limits fail closed", testRiskMalformedInputs)
    , ("gate telemetry bounds history and unknown cardinality", testGateTelemetryBounds)
    , ("external data symbol scoping fails closed without a target symbol", testExternalDataSymbolScoping)
        , ("automatic graduation equity is session bounded", testGraduationEquitySessionBoundary)
    , ("portfolio graduation equity boundaries fail closed", testGraduationPortfolioBoundaryContract)
    , ("every formal execution obligation holds", testFormalExecutionReport)

    , ("every formal risk obligation holds", testFormalRiskReport)
    , ("every formal optimization obligation holds", testFormalOptimizationReport)
    ]

testMarketTimestampOverflow :: IO ()
testMarketTimestampOverflow = do
    let maxTimestamp = maxBound :: Int64
        overflowBar = marketBar maxTimestamp
        nearOverflowBar = marketBar (maxTimestamp - 1000)
        staleOverflow = marketDataStaleReason "1h" maxTimestamp maxTimestamp
        continuationOverflow = marketDataContinuationIssue "1h" maxTimestamp [maxTimestamp]
    assertBool
        "freshness rejects overflowed close/age arithmetic"
        (isNothing (marketDataFreshness "1h" maxTimestamp maxTimestamp))
    assertBool
        "stale reason identifies timestamp overflow instead of reporting fresh data"
        (maybe False ("MARKET_DATA_TIMESTAMP_OVERFLOW" `isPrefixOf`) staleOverflow)
    assertBool
        "continuation rejects overflowed expected timestamps"
        (maybe False ("MARKET_DATA_TIMESTAMP_OVERFLOW" `isPrefixOf`) continuationOverflow)
    assertBool
        "closed-bar normalization rejects an overflowed close timestamp"
        ( case normalizeClosedMarketSeries "test candle" 3600000 0 [(overflowBar, ())] of
            Left err -> "test candle timestamp overflow" `isPrefixOf` err
            Right _ -> False
        )
    assertBool
        "continuity validation rejects overflowed expected timestamps"
        ( case validateMarketSeriesContinuity "test candle" 3600000 [nearOverflowBar, overflowBar] of
            Left err -> "test candle timestamp overflow" `isPrefixOf` err
            Right _ -> False
        )
    assertBool
        "empty continuation evidence remains a valid no-op"
        (isNothing (marketDataContinuationIssue "1h" maxTimestamp []))
  where
    marketBar openTimeMs =
        MarketSeriesBar
            { msbOpenTimeMs = openTimeMs
            , msbOpen = Just 1
            , msbHigh = Just 1
            , msbLow = Just 1
            , msbClose = 1
            , msbVolume = Just 0
            }

testRiskMalformedInputs :: IO ()
testRiskMalformedInputs = do
    let base =
            HaltInputs
                { hiPrevHaltReason = Nothing
                , hiDayChanged = False
                , hiWeekChanged = False
                , hiDailyLoss = 0
                , hiWeeklyLoss = 0
                , hiDrawdown = 0
                , hiExpectancy = Nothing
                , hiMaxDailyLossLim = Nothing
                , hiMaxWeeklyLossLim = Nothing
                , hiMaxDrawdownLim = Nothing
                , hiMinExpectancy = Nothing
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
                }
        malformedMetrics = [-0.01, 0 / 0, 1 / 0, -(1 / 0)]
        metricCases value =
            [ base{hiDailyLoss = value}
            , base{hiWeeklyLoss = value}
            , base{hiDrawdown = value}
            ]
    forM_ malformedMetrics $ \value ->
        forM_ (metricCases value) $ \inputs ->
            assertBool
                "malformed daily/weekly/drawdown evidence emits RISK_METRIC_INVALID"
                (specRiskHalt inputs == Just (ExitOther "RISK_METRIC_INVALID"))
    assertBool
        "negative loss-streak limit emits LOSS_STREAK_LIMIT_INVALID"
        ( specRiskHalt base{hiConsecutiveLosses = 10, hiMaxLossStreakLim = Just (-1)}
            == Just (ExitOther "LOSS_STREAK_LIMIT_INVALID")
        )
    assertBool
        "zero loss-streak limit retains its documented disabled boundary"
        (isNothing (specRiskHalt base{hiConsecutiveLosses = 10, hiMaxLossStreakLim = Just 0}))

testGateTelemetryBounds :: IO ()
testGateTelemetryBounds = do
    let negativeBound = emptyTelemetry (-10)
        negativeRecorded = recordRejection (unknownRejection "negative") negativeBound
        hugeBound = emptyTelemetry maxBound
        hugeRecorded = iterate (recordRejection (unknownRejection "huge")) hugeBound !! 1005
        explicitHuge = recordRejectionWithContext (unknownRejection "explicit") maxBound (emptyTelemetry 1)
        unknownRecorded =
            foldl
                (flip (recordRejection . unknownRejection))
                (emptyTelemetry 10)
                ["first", "second", "third"]
        canonicalUnknown = ReasonUnknown (T.pack "UNKNOWN")
    assertBool
        "negative recent-history bounds normalize to zero"
        (gtMaxRecent negativeBound == 0 && null (gtRecentRejections negativeRecorded))
    assertBool
        "huge recent-history bounds cap at 1000"
        ( gtMaxRecent hugeRecorded == 1000
            && length (gtRecentRejections hugeRecorded) == 1000
            && gtTotalRejections hugeRecorded == 1005
        )
    assertBool
        "explicit huge bound updates also cap at 1000"
        (gtMaxRecent explicitHuge == 1000 && length (gtRecentRejections explicitHuge) == 1)
    assertBool
        "untrusted unknown reasons collapse to one canonical histogram bucket"
        ( Map.size (gtRejectionHistogram unknownRecorded) == 1
            && Map.size (gtPerReasonCounts unknownRecorded) == 1
            && Map.lookup canonicalUnknown (gtPerReasonCounts unknownRecorded) == Just 3
            && all ((== canonicalUnknown) . grReason) (gtRecentRejections unknownRecorded)
        )
  where
    unknownRejection reason =
        GateRejection
            { grGateName = GateUnknown
            , grReason = ReasonUnknown (T.pack reason)
            , grSymbol = Nothing
            , grInterval = Nothing
            , grTimestamp = Nothing
            , grEdgeValue = Nothing
            , grThreshold = Nothing
            , grEfficiency = Nothing
            , grZScore = Nothing
            , grConfidence = Nothing
            }

testExternalDataSymbolScoping :: IO ()
testExternalDataSymbolScoping =
    let target = Just "BTCUSDT"
     in assertChecks
            [ ("unknown target accepts missing global scope", externalSymbolMatches Nothing Nothing)
            , ("unknown target accepts explicit global scope", externalSymbolMatches Nothing (Just ""))
            , ("unknown target rejects full-symbol scope", not (externalSymbolMatches Nothing (Just "ETHUSDT")))
            , ("unknown target rejects base-symbol scope", not (externalSymbolMatches Nothing (Just "ETH")))
            , ("resolved symbol accepts missing global scope", externalSymbolMatches target Nothing)
            , ("resolved symbol accepts explicit global scope", externalSymbolMatches target (Just ""))
            , ("resolved symbol accepts matching full-symbol scope", externalSymbolMatches target (Just "BTCUSDT"))
            , ("resolved symbol accepts matching base-symbol scope", externalSymbolMatches target (Just "BTC"))
            , ("resolved symbol rejects other full-symbol scope", not (externalSymbolMatches target (Just "ETHUSDT")))
            , ("resolved symbol rejects other base-symbol scope", not (externalSymbolMatches target (Just "ETH")))
            ]

testGraduationEquitySessionBoundary :: IO ()
testGraduationEquitySessionBoundary = do
    let stableWorker =
            [ graduationEvidence "stable-worker" "session-stable" 1.00
            , graduationEvidence "stable-worker" "session-stable" 1.05
            ]
        restartedWorker =
            [ graduationEvidence "restarted-worker" "session-before-restart" 1.00
            , graduationEvidence "restarted-worker" "session-before-restart" 0.92
            , graduationEvidence "restarted-worker" "session-after-restart" 1.00
            , graduationEvidence "restarted-worker" "session-after-restart" 1.03
            ]
        reviewedFleet = stableWorker ++ restartedWorker
    assertBool
        "single-session fleet evidence remains admissible"
        (maybe False (\value -> value `approxEq` 0.05) (sessionBoundedFleetReturn stableWorker))
    assertBool
        "mid-window session changes fail closed for graduation"
        (isNothing (sessionBoundedFleetReturn reviewedFleet))
        assertBool
        "naive raw-level stitching would have manufactured positive fleet return"
        (naiveFleetReturn reviewedFleet > 0.07)

testGraduationPortfolioBoundaryContract :: IO ()
testGraduationPortfolioBoundaryContract = do
    let boundaryMs = 2000
        maximumBaselineAgeMs = 100
        alpha = T.pack "alpha"
        beta = T.pack "beta"
        reviewedUuids = [alpha, beta]
        admissibleBaselines =
            [ (alpha, boundaryMs - maximumBaselineAgeMs, 1.0)
            , (beta, boundaryMs, 1.0)
            ]
        admissibleDailyRows =
            [ (boundaryMs, alpha, 1.0)
            , (boundaryMs, beta, 1.0)
            , (boundaryMs + 86400000, alpha, 1.0625)
            , (boundaryMs + 86400000, beta, 1.0625)
            ]
        staleBaselines =
            [ (alpha, boundaryMs - maximumBaselineAgeMs - 1, 1.0)
            , (beta, boundaryMs, 1.0)
            ]
        malformedDailyRows =
            [ (boundaryMs, alpha, 1.0)
            , (boundaryMs, beta, 1.0)
            , (boundaryMs + 86400000, alpha, 0 / 0)
            , (boundaryMs + 86400000, beta, 1.0625)
            ]
        admissibleEquities =
            portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs reviewedUuids admissibleBaselines admissibleDailyRows
        admissiblePerformance = admissibleEquities >>= portfolioGraduationPerformance
    assertBool
        "exact boundary timestamps and exact freshness equality remain admissible"
        (admissibleEquities == Right [1.0, 1.125])
    assertBool
        "boundary-fresh fleet performance uses only relative in-window evidence"
        (admissiblePerformance == Right (2, 0.125, 0))
    assertBool
        "stale baseline evidence fails closed"
        ( case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs reviewedUuids staleBaselines admissibleDailyRows of
            Left err -> err == "graduation baseline is outside the boundary freshness window"
            Right _ -> False
        )
    assertBool
        "missing baseline evidence fails closed"
        ( case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs reviewedUuids [(alpha, boundaryMs, 1.0)] admissibleDailyRows of
            Left err -> err == "graduation baseline is missing for a reviewed UUID"
            Right _ -> False
        )
    assertBool
        "malformed daily equity fails closed"
        ( case portfolioGraduationFleetEquities boundaryMs maximumBaselineAgeMs reviewedUuids admissibleBaselines malformedDailyRows of
            Left err -> err == "graduation daily equity must be finite and positive"
            Right _ -> False
        )
    case admissiblePerformance of
        Left err ->
            ioError (userError ("unexpected admissible graduation performance failure: " ++ err))
        Right (observationCount, netReturn, maxDrawdown) -> do
            let config minimumNetReturn =
                    defaultPortfolioGraduationConfig
                        { pgcEnabled = True
                        , pgcStartedAtMs = 1
                        , pgcMinimumDailyObservations = observationCount
                        , pgcMinimumNetReturn = minimumNetReturn
                        , pgcMaximumDrawdown = maxDrawdown
                        , pgcMinimumExecutionAttempts = 4
                        , pgcMinimumExecutionReliability = 0.75
                        , pgcMinimumStatusReliability = 0.5
                        }
                evidence =
                    PortfolioGraduationEvidence
                        { pgeDailyObservationCount = observationCount
                        , pgeNetReturn = netReturn
                        , pgeMaxDrawdown = maxDrawdown
                        , pgeExecutionAttempts = 4
                        , pgeExecutionSuccesses = 3
                        , pgeStatusSamples = 2
                        , pgeHealthyStatusSamples = 1
                        , pgeLatestStatusesHealthy = True
                        }
                thresholdReview = portfolioGraduationReview (config netReturn) boundaryMs reviewedUuids evidence
                passingReview = portfolioGraduationReview (config 0.1) boundaryMs reviewedUuids evidence
            assertBool
                "exact minimum net-return equality remains pending"
                ( pgrDecision thresholdReview == PortfolioGraduationPending
                    && map T.unpack (pgrReasons thresholdReview) == ["net-return-below-minimum"]
                )
            assertBool
                "drawdown and reliability equality stay admissible once net return exceeds the floor"
                (pgrDecision passingReview == PortfolioGraduated && null (pgrReasons passingReview))

testFormalExecutionReport :: IO ()

testFormalExecutionReport =
    assertChecks
        [ ("implementation matches position-fill spec", Execution.fvrExecImplMatchesSpec report)
        , ("reduce-only implementation matches spec", Execution.fvrExecReduceOnlyImplMatchesSpec report)
        , ("order applied quantity matches spec", Execution.fvrExecOrderAppliedImplMatchesSpec report)
        , ("order applied fraction matches spec", Execution.fvrExecOrderAppliedFractionImplMatchesSpec report)
        , ("order applied fraction is bounded", Execution.fvrExecOrderAppliedFractionBounded report)
        , ("reduce-only never increases", Execution.fvrExecReduceOnlyNeverIncreases report)
        , ("reduce-only never flips", Execution.fvrExecReduceOnlyNeverFlips report)
        , ("fill quantity is conserved", Execution.fvrExecQtyConservation report)
        , ("close quantity is monotone", Execution.fvrExecCloseQtyMonotone report)
        , ("open quantity is monotone", Execution.fvrExecOpenQtyMonotone report)
        ]
  where
    report = Execution.verifyFormalExecution

testFormalRiskReport :: IO ()
testFormalRiskReport =
    assertChecks
        [ ("halt is monotone", Risk.fvrRiskHaltMonotone report)
        , ("daily halt resets", Risk.fvrRiskHaltResetDaily report)
        , ("weekly halt resets", Risk.fvrRiskHaltResetWeekly report)
        , ("other halts persist", Risk.fvrRiskHaltPreservesOther report)
        , ("no false positive", Risk.fvrRiskHaltNoFalsePositive report)
        , ("position-size halt is complete", Risk.fvrRiskHaltPositionSize report)
        , ("loss-streak halt is complete", Risk.fvrRiskHaltLossStreak report)
        , ("max-position bound holds", Risk.fvrMaxPositionSizeBound report)
        , ("risk limits are finite", Risk.fvrRiskLimitFinite report)
        , ("risk metrics are admissible", Risk.fvrRiskMetricSanity report)
        , ("loss-streak limit is admissible", Risk.fvrLossStreakLimitSanity report)
        , ("drawdown config is sane", Risk.fvrDrawdownSanity report)
        , ("position-size evidence is sane", Risk.fvrPositionSizeSanity report)
        , ("expectancy evidence is sane", Risk.fvrExpectancySanity report)
        , ("volatility target is sane", Risk.fvrVolTargetSanity report)
        , ("leverage is sane", Risk.fvrLeverageSanity report)
        , ("cooldown is non-negative", Risk.fvrCooldownNonNegative report)
        ]
  where
    report = Risk.verifyFormalRisk

testFormalOptimizationReport :: IO ()
testFormalOptimizationReport =
    assertChecks
        [ ("ROI domain is non-empty", Optimization.fvrRoiStateCount report > 0)
        , ("tie-break domain is non-empty", Optimization.fvrTieBreakPairCount report > 0)
        , ("vol-conf domain is non-empty", Optimization.fvrVolConfStateCount report > 0)
        , ("Kalman domain is non-empty", Optimization.fvrKalmanFusionStateCount report > 0)
        , ("ROI spec matches implementation", Optimization.fvrRoiSpecMatchesImplementation report)
        , ("return is monotone", Optimization.fvrReturnMonotone report)
        , ("drawdown is monotone", Optimization.fvrDrawdownMonotone report)
        , ("tail loss is monotone", Optimization.fvrTailLossMonotone report)
        , ("turnover is monotone", Optimization.fvrTurnoverMonotone report)
        , ("expectancy is monotone", Optimization.fvrExpectancyMonotone report)
        , ("payback is monotone", Optimization.fvrPaybackMonotone report)
        , ("invalid payback equals missing", Optimization.fvrInvalidPaybackMatchesMissing report)
        , ("zero-round-trip rewards stay disabled", Optimization.fvrZeroRoundTripRewardInvariant report)
        , ("activity penalty is ordered", Optimization.fvrActivityPenaltyOrdered report)
        , ("activity count is valid", Optimization.fvrActivityCountInvariant report)
        , ("exposure penalty is ordered", Optimization.fvrExposurePenaltyOrdered report)
        , ("tie-break is total", Optimization.fvrTieBreakTotalOrderAfterNormalization report)
        , ("tie-break prefers hysteresis", Optimization.fvrTieBreakHysteresisPreference report)
        , ("tie-break spec matches implementation", Optimization.fvrTieBreakSpecMatchesImplementation report)
        , ("optimizer public surface holds", Optimization.fvrOptimizerPublicSurfaceInvariant report)
        , ("vol-conf canonicalization holds", Optimization.fvrVolConfCanonicalizationInvariant report)
        , ("malformed volatility equals missing", Optimization.fvrVolConfMalformedVolMatchesMissing report)
        , ("malformed confidence fails closed", Optimization.fvrVolConfMalformedConfidenceFailsClosed report)
        , ("malformed vol-conf stays conservative", Optimization.fvrVolConfMalformedInputsStayConservative report)
        , ("vol-conf output is bounded", Optimization.fvrVolConfOutputBounded report)
        , ("malformed Kalman measurements are ignored", Optimization.fvrKalmanFusionMalformedMeasurementsIgnored report)
        , ("no valid Kalman measurement keeps prior", Optimization.fvrKalmanFusionNoValidMeasurementsKeepPrior report)
        , ("Kalman posterior is finite", Optimization.fvrKalmanFusionPosteriorFinite report)
        , ("valid Kalman evidence shrinks variance", Optimization.fvrKalmanFusionValidEvidenceShrinksVariance report)
        ]
  where
    report = Optimization.verifyFormalOptimization

data GraduationEvidence = GraduationEvidence
    { geWorker :: String
    , geSession :: String
    , geModelEquity :: Double
    }

graduationEvidence :: String -> String -> Double -> GraduationEvidence
graduationEvidence = GraduationEvidence

sessionBoundedFleetReturn :: [GraduationEvidence] -> Maybe Double
sessionBoundedFleetReturn =
    fmap sum
        . mapM sessionBoundedWorkerReturn
        . Map.elems
        . evidenceByWorker

evidenceByWorker :: [GraduationEvidence] -> Map.Map String [GraduationEvidence]
evidenceByWorker [] = Map.empty
evidenceByWorker (sample : samples) =
    Map.insertWith (++) (geWorker sample) [sample] (evidenceByWorker samples)

sessionBoundedWorkerReturn :: [GraduationEvidence] -> Maybe Double
sessionBoundedWorkerReturn [] = Nothing
sessionBoundedWorkerReturn (sample : samples)
    | not (validEquity (geModelEquity sample)) = Nothing
    | otherwise = go (geSession sample) (geModelEquity sample) (geModelEquity sample) samples
  where
    go _ startEquity currentEquity [] = Just (currentEquity - startEquity)
    go session startEquity _ (next : rest)
        | geSession next /= session = Nothing
        | not (validEquity (geModelEquity next)) = Nothing
        | otherwise =
            go session startEquity (geModelEquity next) rest

validEquity :: Double -> Bool
validEquity value = not (isNaN value) && not (isInfinite value)

naiveFleetReturn :: [GraduationEvidence] -> Double
naiveFleetReturn =
    sum
        . map rawLevelReturn
        . Map.elems
        . evidenceByWorker

rawLevelReturn :: [GraduationEvidence] -> Double
rawLevelReturn [] = 0
rawLevelReturn (sample : samples) = go (geModelEquity sample) (geModelEquity sample) samples
  where
    go startEquity currentEquity [] = currentEquity - startEquity
    go startEquity _ (next : rest) = go startEquity (geModelEquity next) rest

approxEq :: Double -> Double -> Bool
approxEq left right = abs (left - right) <= 1.0e-12

assertChecks :: [(String, Bool)] -> IO ()
assertChecks = mapM_ (uncurry assertBool)

assertBool :: String -> Bool -> IO ()
assertBool message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))
