module Trader.Test.TechnicalAnalysis (runTechnicalAnalysisTests) where

import Control.Monad (unless)
import Data.Maybe (catMaybes, isNothing)
import qualified Data.Vector as V
import Trader.TechnicalAnalysis.Indicators
import Trader.TechnicalAnalysis.Strategies

runTechnicalAnalysisTests :: IO ()
runTechnicalAnalysisTests = do
    testEmaPrefixInvariance
    testRsiBounds
    testAtrNonNegative
    testAroonPeriodOneFailsClosed
    testRegimeSelectorFindsTrend
    testTrendCandidateFailsClosedOnShortSeries
    testBreakoutCandidateCanTriggerLong

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
    let base = [100 + fromIntegral i * 0.2 | i <- [0 .. 58]]
        closes = V.fromList (base ++ [120, 123, 126, 130, 135, 140])
        highs = V.map (+ 0.5) closes
        lows = V.map (subtract 1.5) closes
        opens = V.map (subtract 0.3) closes
        volumes = V.fromList ([1000 + fromIntegral i * 5 | i <- [0 .. 58]] ++ [2000, 2200, 2400, 2600, 2800, 3000])
        series = OhlcvSeries opens highs lows closes volumes
        candidate = volumeConfirmedBreakoutCandidate series
    assert
        "volumeConfirmedBreakoutCandidate can produce a long breakout candidate on synthetic breakout data"
        ( case candidate of
            Just signal -> scBias signal == BiasLong
            Nothing -> False
        )
