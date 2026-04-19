module Trader.Test.TechnicalAnalysis (runTechnicalAnalysisTests) where

import Control.Monad (unless)
import Data.Maybe (catMaybes, isJust, isNothing)
import qualified Data.Vector as V
import Trader.TechnicalAnalysis.Indicators
import Trader.TechnicalAnalysis.Strategies
import Trader.VolConfGate (VolConfGatePreset (..))

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

assert :: String -> Bool -> IO ()
assert message condition =
    unless condition (ioError (userError ("Assertion failed: " ++ message)))

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
    assert "regimeSelector identifies a clean synthetic trend" (regimeSelector series == Just RegimeTrend)

testTrendCandidateFailsClosedOnShortSeries :: IO ()
testTrendCandidateFailsClosedOnShortSeries = do
    let closes = V.fromList [100 + fromIntegral i | i <- [0 .. 39]]
        highs = V.map (+ 1) closes
        lows = V.map (subtract 1) closes
        opens = V.map (subtract 0.5) closes
        volumes = V.replicate 40 1000
        series = OhlcvSeries opens highs lows closes volumes
    assert "trendFollowingCandidate fails closed on short history" (isNothing (trendFollowingCandidate series))

testBreakoutCandidateCanTriggerLong :: IO ()
testBreakoutCandidateCanTriggerLong = do
    let candidate = volumeConfirmedBreakoutCandidate breakoutSeries
    assert
        "volumeConfirmedBreakoutCandidate can produce a long breakout candidate on synthetic breakout data"
        ( case candidate of
            Just signal -> scBias signal == BiasLong
            Nothing -> False
        )

testCandidateConfidenceUsesAverageBeforeClamp :: IO ()
testCandidateConfidenceUsesAverageBeforeClamp = do
    let candidate = volumeConfirmedBreakoutCandidate breakoutSeries
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

breakoutSeries :: OhlcvSeries
breakoutSeries =
    let base = [100 + fromIntegral i * 0.2 | i <- [0 .. 58]]
        closes = V.fromList (base ++ [120, 123, 126, 130, 135, 140])
        highs = V.map (+ 0.5) closes
        lows = V.map (subtract 1.5) closes
        opens = V.map (subtract 0.3) closes
        volumes = V.fromList ([1000 + fromIntegral i * 5 | i <- [0 .. 58]] ++ [2000, 2200, 2400, 2600, 2800, 3000])
     in OhlcvSeries opens highs lows closes volumes
