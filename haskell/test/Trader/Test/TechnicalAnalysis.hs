module Trader.Test.TechnicalAnalysis (runTechnicalAnalysisTests) where

import Control.Monad (unless)
import Data.Maybe (catMaybes, isJust, isNothing)
import qualified Data.Vector as V
import Trader.Method (Method (..), methodCode, methodIsTechnicalAnalysis, parseMethod)
import Trader.SignalGates (defaultSignalGateConfig)
import Trader.TechnicalAnalysis.Indicators
import Trader.TechnicalAnalysis.Strategies
import Trader.VolConfGate (VolConfGatePreset (..), defaultVolConfGateConfig)

runTechnicalAnalysisTests :: IO ()
runTechnicalAnalysisTests = do
    testEmaPrefixInvariance
    testRsiBounds
    testAtrNonNegative
    testAroonPeriodOneFailsClosed
    testRegimeSelectorFindsTrend
    testTrendCandidateFailsClosedOnShortSeries
    testBreakoutCandidateCanTriggerLong
    testCandidateConfidenceUsesAverageBeforeClamp
    testGatedCandidateAdmissionHonorsRiskGates
    testRegimeSwitchSubMethodSelection
    testRegimeSwitchCandidateUsesBreakoutFallback
    testTechnicalAnalysisMethodsParse

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

testRegimeCalibration :: RegimeCalibration
testRegimeCalibration =
    defaultRegimeCalibration
        { rcAdxWeight = 0.40
        , rcTrendThreshold = 0.55
        , rcRangeThreshold = 0.55
        }

testEmaPrefixInvariance :: IO ()
testEmaPrefixInvariance = do
    let closes = V.fromList [100 + fromIntegral i | i <- [0 .. 79]]
        prefix = V.take 40 closes
        fullEma = emaSeries 10 closes
        prefixEma = emaSeries 10 prefix
    assert "emaSeries is prefix-invariant at the same bar" (prefixEma V.! 39 == fullEma V.! 39)

testRsiBounds :: IO ()
testRsiBounds = do
    let closes = V.fromList [100, 101, 102, 101, 103, 104, 102, 105, 104, 106, 107, 105, 108, 110, 109, 111, 112, 110, 113, 114, 116]
        rsis = rsiSeries 14 closes
        present = catMaybes (V.toList rsis)
    assert "rsiSeries emits bounded RSI values" (all (\value -> value >= 0 && value <= 100) present)

testAtrNonNegative :: IO ()
testAtrNonNegative = do
    let closes = V.fromList [100 + fromIntegral i * 0.5 | i <- [0 .. 79]]
        highs = V.map (+ 1.2) closes
        lows = V.map (subtract 1.1) closes
        atrs = atrSeries 14 highs lows closes
        present = catMaybes (V.toList atrs)
    assert "atrSeries is non-negative" (all (>= 0) present)

testAroonPeriodOneFailsClosed :: IO ()
testAroonPeriodOneFailsClosed = do
    let highs = V.fromList [101, 102, 103]
        lows = V.fromList [99, 98, 97]
    assert "aroonSeries period 1 fails closed" (V.all (== Nothing) (aroonSeries 1 highs lows))

testRegimeSelectorFindsTrend :: IO ()
testRegimeSelectorFindsTrend = do
    let closes = V.fromList [100 + fromIntegral i * 1.5 | i <- [0 .. 119]]
        highs = V.map (+ 1.0) closes
        lows = V.map (subtract 1.0) closes
        opens = V.map (subtract 0.4) closes
        volumes = V.fromList [1000 + fromIntegral (i * 10) | i <- [0 .. 119]]
        series = OhlcvSeries opens highs lows closes volumes
    assert "regimeSelector identifies a clean synthetic trend" (regimeSelector testRegimeCalibration series == Just RegimeTrend)

testTrendCandidateFailsClosedOnShortSeries :: IO ()
testTrendCandidateFailsClosedOnShortSeries = do
    let closes = V.fromList [100 + fromIntegral i | i <- [0 .. 39]]
        highs = V.map (+ 1) closes
        lows = V.map (subtract 1) closes
        opens = V.map (subtract 0.5) closes
        volumes = V.replicate 40 1000
        series = OhlcvSeries opens highs lows closes volumes
    assert "trendFollowingCandidate fails closed on short history" (isNothing (trendFollowingCandidate testRegimeCalibration series))

testBreakoutCandidateCanTriggerLong :: IO ()
testBreakoutCandidateCanTriggerLong = do
    let candidate = volumeConfirmedBreakoutCandidate breakoutTestCalibration breakoutSeries
    assert
        "volumeConfirmedBreakoutCandidate can produce a long breakout candidate on synthetic breakout data"
        ( case candidate of
            Just signal -> scBias signal == BiasLong
            Nothing -> False
        )

testCandidateConfidenceUsesAverageBeforeClamp :: IO ()
testCandidateConfidenceUsesAverageBeforeClamp = do
    let candidate = volumeConfirmedBreakoutCandidate breakoutTestCalibration breakoutSeries
    assert
        "breakout confidence is averaged before clamping"
        ( case candidate of
            Just signal -> scConfidence signal > 0.33 && scConfidence signal <= 1
            Nothing -> False
        )

testGatedCandidateAdmissionHonorsRiskGates :: IO ()
testGatedCandidateAdmissionHonorsRiskGates = do
    let candidate =
            StrategyCandidate
                { scFamily = "test"
                , scName = "manual-risk-gated-candidate"
                , scBias = BiasLong
                , scConfidence = 0.8
                , scEntryPrice = Just 100
                , scStopPrice = Just 98
                , scTakeProfitPrice = Just 103
                , scReason = "synthetic candidate"
                }
        inputs =
            TechnicalAnalysisGateInputs
                { tagFeePerSide = 0.001
                , tagMinConfidence = 0.60
                , tagCurrentBias = Nothing
                , tagVolatility = Just 0.4
                , tagVolConfGate = VolConfGateDisabled
                , tagVolConfGateConfig = defaultVolConfGateConfig
                , tagRegimeCalibration = testRegimeCalibration
                , tagStrategyCalibration = defaultTechnicalStrategyCalibration
                , tagSignalGateConfig = defaultSignalGateConfig
                , tagOpenThreshold = 0.0
                , tagCloseThreshold = 0.0
                }
        admitted = admitStrategyCandidate inputs candidate
        highFeeInputs = inputs{tagFeePerSide = 0.02}
        malformedVolInputs = inputs{tagVolatility = Nothing, tagVolConfGate = VolConfGateV1Default}
        weakCandidate = candidate{scConfidence = 0.3}
    assert "valid candidate passes TA admission gates" (isJust admitted)
    assert
        "candidate reward edge is available for fee/headroom gates"
        ( case candidateRewardEdge candidate of
            Just edge -> abs (edge - 0.03) < 1e-12
            Nothing -> False
        )
    assert "high fees block new TA entries" (isNothing (admitStrategyCandidate highFeeInputs candidate))
    assert "malformed volatility blocks non-disabled volume/confidence gates" (isNothing (admitStrategyCandidate malformedVolInputs candidate))
    assert "weak confidence blocks TA entries" (isNothing (admitStrategyCandidate inputs weakCandidate))

testRegimeSwitchSubMethodSelection :: IO ()
testRegimeSwitchSubMethodSelection = do
    let strongAdx = Just AdxPoint{adxValue = 26, adxPlusDi = 30, adxMinusDi = 10}
        weakAdx = Just AdxPoint{adxValue = 20, adxPlusDi = 20, adxMinusDi = 18}
        narrowBand = Just Band{bandLower = 99.95, bandMiddle = 100, bandUpper = 100.05}
        wideBand = Just Band{bandLower = 95, bandMiddle = 100, bandUpper = 105}
    assert
        "ta_regime_switch selects trend when ADX is above 25 and price is above EMA200"
        (regimeSwitchSubMethod 101 strongAdx (Just 100) wideBand == MethodTaTrend)
    assert
        "ta_regime_switch selects reversion on narrow Bollinger bandwidth outside the trend regime"
        (regimeSwitchSubMethod 99 strongAdx (Just 100) narrowBand == MethodTaReversion)
    assert
        "ta_regime_switch selects breakout by default"
        (regimeSwitchSubMethod 100 weakAdx (Just 101) wideBand == MethodTaBreakout)

testRegimeSwitchCandidateUsesBreakoutFallback :: IO ()
testRegimeSwitchCandidateUsesBreakoutFallback = do
    let inds = precomputeIndicators breakoutSeries
        t = V.length (ohlcvClose breakoutSeries) - 1
        candidate = candidateForMethodAt MethodTaRegimeSwitch taTestInputs inds t
    assert
        "ta_regime_switch uses the breakout candidate when trend and squeeze regimes are absent"
        ( case candidate of
            Just signal -> scFamily (gscCandidate signal) == "volume-confirmed-breakout"
            Nothing -> False
        )

testTechnicalAnalysisMethodsParse :: IO ()
testTechnicalAnalysisMethodsParse = do
    let cases =
            [ ("ta_trend", MethodTaTrend)
            , ("technical-reversion", MethodTaReversion)
            , ("volume_breakout", MethodTaBreakout)
            , ("technical_analysis", MethodTaBest)
            , ("ta_regime_switch", MethodTaRegimeSwitch)
            ]
    mapM_
        ( \(raw, expected) ->
            assert
                ("parseMethod recognizes " ++ raw)
                (parseMethod raw == Right expected && methodIsTechnicalAnalysis expected && not (null (methodCode expected)))
        )
        cases

breakoutSeries :: OhlcvSeries
breakoutSeries =
    let base = [100 + fromIntegral i * 0.2 | i <- [0 .. 58]]
        closes = V.fromList (base ++ [120, 123, 126, 130, 135, 140])
        highs = V.map (+ 0.5) closes
        lows = V.map (subtract 1.5) closes
        opens = V.map (subtract 0.3) closes
        volumes = V.fromList ([1000 + fromIntegral i * 5 | i <- [0 .. 58]] ++ [2000, 2200, 2400, 2600, 2800, 3000])
     in OhlcvSeries opens highs lows closes volumes

taTestInputs :: TechnicalAnalysisGateInputs
taTestInputs =
    TechnicalAnalysisGateInputs
        { tagFeePerSide = 0
        , tagMinConfidence = 0
        , tagCurrentBias = Nothing
        , tagVolatility = Just 0.4
        , tagVolConfGate = VolConfGateDisabled
        , tagVolConfGateConfig = defaultVolConfGateConfig
        , tagRegimeCalibration = testRegimeCalibration
        , tagStrategyCalibration = defaultTechnicalStrategyCalibration
        , tagSignalGateConfig = defaultSignalGateConfig
        , tagOpenThreshold = 0.0
        , tagCloseThreshold = 0.0
        }

breakoutTestCalibration :: RegimeCalibration
breakoutTestCalibration = testRegimeCalibration{rcTrendThreshold = 1.01}
