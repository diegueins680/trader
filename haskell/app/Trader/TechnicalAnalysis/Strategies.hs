module Trader.TechnicalAnalysis.Strategies (
    GatedStrategyCandidate (..),
    OhlcvIndicators (..),
    OhlcvSeries (..),
    Regime (..),
    RegimeCalibration (..),
    RegimeScore (..),
    StrategyCandidate (..),
    TechnicalAnalysisGateInputs (..),
    TradeBias (..),
    admitStrategyCandidate,
    admittedStrategyCandidates,
    bestCandidateAt,
    candidateForMethodAt,
    candidateRewardEdge,
    momentumReversionAt,
    momentumReversionCandidate,
    precomputeIndicators,
    regimeSelector,
    regimeSelectorDecomposed,
    strategyCandidates,
    trendFollowingAt,
    trendFollowingCandidate,
    volumeConfirmedBreakoutAt,
    volumeConfirmedBreakoutCandidate,
) where

import Control.Monad (join)
import Data.List (sortOn)
import Data.Maybe (catMaybes, fromMaybe, listToMaybe, mapMaybe)
import Data.Ord (Down (..))
import qualified Data.Vector as V
import Trader.Method (Method (..))
import Trader.SignalGates (
    finiteDouble,
    normalizeSignalEntryEdge,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )
import Trader.TechnicalAnalysis.Indicators
import Trader.VolConfGate (
    VolConfGateBehavior,
    VolConfGatePreset,
    applyVolConfGateBehavior,
    vcgBehavior,
    vcgSizeMult,
    volConfGateCell,
 )

data OhlcvSeries = OhlcvSeries
    { ohlcvOpen :: !(V.Vector Double)
    , ohlcvHigh :: !(V.Vector Double)
    , ohlcvLow :: !(V.Vector Double)
    , ohlcvClose :: !(V.Vector Double)
    , ohlcvVolume :: !(V.Vector Double)
    }
    deriving (Eq, Show)

data Regime = RegimeTrend | RegimeRange | RegimeNeutral
    deriving (Eq, Show)

data RegimeCalibration = RegimeCalibration
    { rcAdxWeight :: !Double
    , rcTrendThreshold :: !Double
    , rcRangeThreshold :: !Double
    }
    deriving (Eq, Show)

data RegimeScore = RegimeScore
    { rsTrend :: !Double
    , rsRange :: !Double
    , rsNeutral :: !Double
    , rsConfidence :: !Double
    }
    deriving (Eq, Show)

data TradeBias = BiasLong | BiasShort | BiasFlat
    deriving (Eq, Show)

data StrategyCandidate = StrategyCandidate
    { scFamily :: !String
    , scName :: !String
    , scBias :: !TradeBias
    , scConfidence :: !Double
    , scEntryPrice :: !(Maybe Double)
    , scStopPrice :: !(Maybe Double)
    , scTakeProfitPrice :: !(Maybe Double)
    , scReason :: !String
    }
    deriving (Eq, Show)

data TechnicalAnalysisGateInputs = TechnicalAnalysisGateInputs
    { tagFeePerSide :: !Double
    , tagMinConfidence :: !Double
    , tagCurrentBias :: !(Maybe TradeBias)
    , tagVolatility :: !(Maybe Double)
    , tagVolConfGate :: !VolConfGatePreset
    , tagRegimeCalibration :: !RegimeCalibration
    }
    deriving (Eq, Show)

data GatedStrategyCandidate = GatedStrategyCandidate
    { gscCandidate :: !StrategyCandidate
    , gscEntryEdge :: !Double
    , gscVolConfBehavior :: !VolConfGateBehavior
    , gscSizeMultiplier :: !Double
    }
    deriving (Eq, Show)

strategyCandidates :: RegimeCalibration -> OhlcvSeries -> [StrategyCandidate]
strategyCandidates cal series =
    catMaybes
        [ trendFollowingCandidate cal series
        , momentumReversionCandidate cal series
        , volumeConfirmedBreakoutCandidate series
        ]

admittedStrategyCandidates :: TechnicalAnalysisGateInputs -> OhlcvSeries -> [GatedStrategyCandidate]
admittedStrategyCandidates inputs series =
    mapMaybe (admitStrategyCandidate inputs) (strategyCandidates (tagRegimeCalibration inputs) series)

admitStrategyCandidate :: TechnicalAnalysisGateInputs -> StrategyCandidate -> Maybe GatedStrategyCandidate
admitStrategyCandidate inputs candidate = do
    bias <- directionalBias (scBias candidate)
    edge <- candidateRewardEdge candidate
    require (confidenceOk (tagMinConfidence inputs) (scConfidence candidate))
    require (riskFrameOk candidate)
    require (entryGatesOk inputs bias edge)
    let currentBias = tagCurrentBias inputs >>= directionalBias
        currentSize = if currentBias == Just bias then 1 else 0
        volConfCell = volConfGateCell (tagVolConfGate inputs) (tagVolatility inputs) (Just (scConfidence candidate))
        (gatedBias, gatedSize) =
            applyVolConfGateBehavior
                (vcgBehavior volConfCell)
                currentBias
                currentSize
                (Just bias)
                1
    require (gatedBias == Just bias && gatedSize > 0)
    pure
        GatedStrategyCandidate
            { gscCandidate = candidate
            , gscEntryEdge = edge
            , gscVolConfBehavior = vcgBehavior volConfCell
            , gscSizeMultiplier = vcgSizeMult volConfCell
            }

candidateRewardEdge :: StrategyCandidate -> Maybe Double
candidateRewardEdge candidate = do
    bias <- directionalBias (scBias candidate)
    entry <- positiveFinite (scEntryPrice candidate)
    target <- positiveFinite (scTakeProfitPrice candidate)
    case bias of
        BiasLong ->
            if target > entry
                then Just ((target - entry) / entry)
                else Nothing
        BiasShort ->
            if target < entry
                then Just ((entry - target) / entry)
                else Nothing
        BiasFlat -> Nothing

regimeSelectorDecomposed :: RegimeCalibration -> OhlcvSeries -> Maybe RegimeScore
regimeSelectorDecomposed cal series = do
    validateSeries series
    closeNow <- lastValue (ohlcvClose series)
    adxNow <- latestJust (adxSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    aroonNow <- latestJust (aroonSeries 25 (ohlcvHigh series) (ohlcvLow series))
    fastNow <- latestJust (emaSeries 20 (ohlcvClose series))
    fastPrev <- laggedJust 5 (emaSeries 20 (ohlcvClose series))
    bbNow <- latestJust (bollingerBandsSeries 20 2 (ohlcvClose series))
    let slope = safeDivide (fastNow - fastPrev) closeNow
        width = safeDivide (bandUpper bbNow - bandLower bbNow) closeNow
        aroonGap = abs (aroonUp aroonNow - aroonDown aroonNow)
        adxTrendScore = clamp01 ((adxValue adxNow - 15) / 20)
        aroonTrendScore = clamp01 ((aroonGap - 20) / 40)
        slopeTrendScore = clamp01 ((abs slope - 0.005) / 0.015)
        wAdx = rcAdxWeight cal
        trendScore = wAdx * adxTrendScore + (1 - wAdx) * (0.35 / 0.60 * aroonTrendScore + 0.25 / 0.60 * slopeTrendScore)
        adxRangeScore = clamp01 ((25 - adxValue adxNow) / 15)
        widthRangeScore = clamp01 ((0.12 - width) / 0.08)
        rangeScore = 0.5 * adxRangeScore + 0.5 * widthRangeScore
        maxScore = max trendScore rangeScore
        neutralScore = max 0 (1 - maxScore)
    pure
        RegimeScore
            { rsTrend = trendScore
            , rsRange = rangeScore
            , rsNeutral = neutralScore
            , rsConfidence = maxScore
            }

regimeSelector :: RegimeCalibration -> OhlcvSeries -> Maybe Regime
regimeSelector cal series = do
    score <- regimeSelectorDecomposed cal series
    if rsTrend score >= rcTrendThreshold cal
        then Just RegimeTrend
        else
            if rsRange score >= rcRangeThreshold cal
                then Just RegimeRange
                else Just RegimeNeutral

trendFollowingCandidate :: RegimeCalibration -> OhlcvSeries -> Maybe StrategyCandidate
trendFollowingCandidate cal series = do
    validateSeries series
    closeNow <- lastValue (ohlcvClose series)
    fastNow <- latestJust (emaSeries 20 (ohlcvClose series))
    slowNow <- latestJust (emaSeries 50 (ohlcvClose series))
    adxNow <- latestJust (adxSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    aroonNow <- latestJust (aroonSeries 25 (ohlcvHigh series) (ohlcvLow series))
    atrNow <- latestJust (atrSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    regimeNow <- regimeSelector cal series
    let longTrend =
            regimeNow == RegimeTrend
                && fastNow > slowNow
                && adxValue adxNow >= 10
                && aroonUp aroonNow > aroonDown aroonNow
        shortTrend =
            regimeNow == RegimeTrend
                && fastNow < slowNow
                && adxValue adxNow >= 10
                && aroonDown aroonNow > aroonUp aroonNow
        maSpread = abs (safeDivide (fastNow - slowNow) closeNow)
        baseConfidence = clamp01 ((((adxValue adxNow - 20) / 25) + (maSpread * 8) + (abs (aroonUp aroonNow - aroonDown aroonNow) / 100)) / 3)
     in if longTrend
            then
                Just
                    StrategyCandidate
                        { scFamily = "trend-following"
                        , scName = "ema-crossover-adx-aroon-atr"
                        , scBias = BiasLong
                        , scConfidence = baseConfidence
                        , scEntryPrice = Just closeNow
                        , scStopPrice = Just (closeNow - (2 * atrNow))
                        , scTakeProfitPrice = Just (closeNow + (3 * atrNow))
                        , scReason = "Fast EMA is above slow EMA with ADX/Aroon trend confirmation and ATR-based risk framing."
                        }
            else
                if shortTrend
                    then
                        Just
                            StrategyCandidate
                                { scFamily = "trend-following"
                                , scName = "ema-crossover-adx-aroon-atr"
                                , scBias = BiasShort
                                , scConfidence = baseConfidence
                                , scEntryPrice = Just closeNow
                                , scStopPrice = Just (closeNow + (2 * atrNow))
                                , scTakeProfitPrice = Just (closeNow - (3 * atrNow))
                                , scReason = "Fast EMA is below slow EMA with ADX/Aroon trend confirmation and ATR-based risk framing."
                                }
                    else Nothing

momentumReversionCandidate :: RegimeCalibration -> OhlcvSeries -> Maybe StrategyCandidate
momentumReversionCandidate cal series = do
    validateSeries series
    regimeNow <- regimeSelector cal series
    closeNow <- lastValue (ohlcvClose series)
    rsiNow <- latestJust (rsiSeries 14 (ohlcvClose series))
    stochasticNow <- latestJust (stochasticKSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    rocNow <- latestJust (rocSeries 10 (ohlcvClose series))
    macdNow <- latestJust (macdSeries 12 26 9 (ohlcvClose series))
    bollingerNow <- latestJust (bollingerBandsSeries 20 2 (ohlcvClose series))
    keltnerNow <- latestJust (keltnerChannelsSeries 20 1.5 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    atrNow <- latestJust (atrSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    let nearLowerEnvelope = closeNow <= max (bandLower bollingerNow) (bandLower keltnerNow) * 1.02
        nearUpperEnvelope = closeNow >= min (bandUpper bollingerNow) (bandUpper keltnerNow) * 0.98
        longSetup =
            regimeNow /= RegimeTrend
                && rsiNow <= 35
                && stochasticNow <= 20
                && rocNow < 0
                && macdValue macdNow >= macdSignal macdNow
                && nearLowerEnvelope
        shortSetup =
            regimeNow /= RegimeTrend
                && rsiNow >= 65
                && stochasticNow >= 80
                && rocNow > 0
                && macdValue macdNow <= macdSignal macdNow
                && nearUpperEnvelope
        baseConfidence = clamp01 (((abs (50 - rsiNow) / 50) + (abs (50 - stochasticNow) / 50) + min 1 (abs rocNow / 5)) / 3)
     in if longSetup
            then
                Just
                    StrategyCandidate
                        { scFamily = "momentum-reversion"
                        , scName = "rsi-stochastic-roc-macd-envelope"
                        , scBias = BiasLong
                        , scConfidence = baseConfidence
                        , scEntryPrice = Just closeNow
                        , scStopPrice = Just (closeNow - (1.5 * atrNow))
                        , scTakeProfitPrice = Just (bandMiddle bollingerNow)
                        , scReason = "Range-style long setup: RSI/Stochastic oversold, negative ROC, MACD confirmation, lower envelope context."
                        }
            else
                if shortSetup
                    then
                        Just
                            StrategyCandidate
                                { scFamily = "momentum-reversion"
                                , scName = "rsi-stochastic-roc-macd-envelope"
                                , scBias = BiasShort
                                , scConfidence = baseConfidence
                                , scEntryPrice = Just closeNow
                                , scStopPrice = Just (closeNow + (1.5 * atrNow))
                                , scTakeProfitPrice = Just (bandMiddle bollingerNow)
                                , scReason = "Range-style short setup: RSI/Stochastic overbought, positive ROC, MACD confirmation, upper envelope context."
                                }
                    else Nothing

volumeConfirmedBreakoutCandidate :: OhlcvSeries -> Maybe StrategyCandidate
volumeConfirmedBreakoutCandidate series = do
    validateSeries series
    closeNow <- lastValue (ohlcvClose series)
    let donchianSeries = donchianChannelsSeries 20 (ohlcvHigh series) (ohlcvLow series)
    donchianNow <- laggedJust 1 donchianSeries
    atrNow <- latestJust (atrSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    obvNow <- latestJust (obvSeries (ohlcvClose series) (ohlcvVolume series))
    obvPrev <- laggedJust 5 (obvSeries (ohlcvClose series) (ohlcvVolume series))
    adNow <- latestJust (accumulationDistributionSeries (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series) (ohlcvVolume series))
    adPrev <- laggedJust 5 (accumulationDistributionSeries (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series) (ohlcvVolume series))
    cmfNow <- latestJust (cmfSeries 20 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series) (ohlcvVolume series))
    mfiNow <- latestJust (mfiSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series) (ohlcvVolume series))
    vptNow <- latestJust (vptSeries (ohlcvClose series) (ohlcvVolume series))
    vptPrev <- laggedJust 5 (vptSeries (ohlcvClose series) (ohlcvVolume series))
    let obvUp = obvNow > obvPrev
        adUp = adNow > adPrev
        vptUp = vptNow > vptPrev
        longBreakout = closeNow > donchianUpper donchianNow && obvUp && adUp && vptUp && cmfNow > 0 && mfiNow >= 50
        shortBreakout = closeNow < donchianLower donchianNow && not obvUp && not adUp && not vptUp && cmfNow < 0 && mfiNow <= 50
        breakoutDistance = max 0 (safeDivide (abs (closeNow - midChannel donchianNow)) closeNow)
        baseConfidence = clamp01 (((if obvUp == adUp then 1 else 0.5) + min 1 (abs cmfNow) + min 1 (breakoutDistance * 10)) / 3)
     in if longBreakout
            then
                Just
                    StrategyCandidate
                        { scFamily = "volume-confirmed-breakout"
                        , scName = "donchian-obv-ad-cmf-mfi-vpt"
                        , scBias = BiasLong
                        , scConfidence = baseConfidence
                        , scEntryPrice = Just closeNow
                        , scStopPrice = Just (closeNow - (2 * atrNow))
                        , scTakeProfitPrice = Just (closeNow + (4 * atrNow))
                        , scReason = "Upper-channel breakout confirmed by OBV/A-D/VPT slope plus CMF/MFI support."
                        }
            else
                if shortBreakout
                    then
                        Just
                            StrategyCandidate
                                { scFamily = "volume-confirmed-breakout"
                                , scName = "donchian-obv-ad-cmf-mfi-vpt"
                                , scBias = BiasShort
                                , scConfidence = baseConfidence
                                , scEntryPrice = Just closeNow
                                , scStopPrice = Just (closeNow + (2 * atrNow))
                                , scTakeProfitPrice = Just (closeNow - (4 * atrNow))
                                , scReason = "Lower-channel breakout confirmed by OBV/A-D/VPT weakness plus CMF/MFI support."
                                }
                    else Nothing

validateSeries :: OhlcvSeries -> Maybe ()
validateSeries series
    | not (sameLength [ohlcvOpen series, ohlcvHigh series, ohlcvLow series, ohlcvClose series, ohlcvVolume series]) = Nothing
    | V.length (ohlcvClose series) < 60 = Nothing
    | otherwise = Just ()

sameLength :: [V.Vector a] -> Bool
sameLength [] = True
sameLength (x : xs) = all ((== V.length x) . V.length) xs

lastValue :: V.Vector a -> Maybe a
lastValue values
    | V.null values = Nothing
    | otherwise = Just (V.last values)

laggedJust :: Int -> V.Vector (Maybe a) -> Maybe a
laggedJust offset values =
    go (V.length values - 1 - offset)
  where
    go idx
        | idx < 0 = Nothing
        | otherwise =
            case values V.! idx of
                Just value -> Just value
                Nothing -> go (idx - 1)

safeDivide :: Double -> Double -> Double
safeDivide _ 0 = 0
safeDivide numerator denominator = numerator / denominator

entryOpenThreshold :: Double
entryOpenThreshold = 0.001

entryGatesOk :: TechnicalAnalysisGateInputs -> TradeBias -> Double -> Bool
entryGatesOk inputs bias edge
    | (tagCurrentBias inputs >>= directionalBias) == Just bias = True
    | otherwise =
        let roundTripFee = 2 * tagFeePerSide inputs
            normalizedEdge = normalizeSignalEntryEdge edge
         in signalEntryEdgeSpikeOk entryOpenThreshold normalizedEdge
                && signalEntryHeadroomOk entryOpenThreshold normalizedEdge
                && signalEntryFeeBufferOk entryOpenThreshold roundTripFee normalizedEdge

riskFrameOk :: StrategyCandidate -> Bool
riskFrameOk candidate =
    ( case scBias candidate of
        BiasLong ->
            positiveFinite (scEntryPrice candidate) >>= \entry ->
                positiveFinite (scStopPrice candidate) >>= \stop ->
                    positiveFinite (scTakeProfitPrice candidate) >>= \target ->
                        Just (stop < entry && target > entry)
        BiasShort ->
            positiveFinite (scEntryPrice candidate) >>= \entry ->
                positiveFinite (scStopPrice candidate) >>= \stop ->
                    positiveFinite (scTakeProfitPrice candidate) >>= \target ->
                        Just (stop > entry && target < entry)
        BiasFlat -> Just False
    )
        == Just True

positiveFinite :: Maybe Double -> Maybe Double
positiveFinite (Just value)
    | finiteDouble value && value > 0 = Just value
positiveFinite _ = Nothing

confidenceOk :: Double -> Double -> Bool
confidenceOk minConfidence confidence =
    finiteDouble minConfidence
        && finiteDouble confidence
        && confidence >= clamp01 minConfidence
        && confidence <= 1

directionalBias :: TradeBias -> Maybe TradeBias
directionalBias BiasLong = Just BiasLong
directionalBias BiasShort = Just BiasShort
directionalBias BiasFlat = Nothing

require :: Bool -> Maybe ()
require True = Just ()
require False = Nothing

clamp01 :: Double -> Double
clamp01 = max 0 . min 1

midChannel :: DonchianChannel -> Double
midChannel channel = (donchianUpper channel + donchianLower channel) / 2

{- | Precomputed indicator vectors for a full OHLCV series, allowing O(1)
per-bar strategy evaluation during backtests instead of O(n) prefix
recomputation.
-}
data OhlcvIndicators = OhlcvIndicators
    { oiClose :: !(V.Vector Double)
    , oiHigh :: !(V.Vector Double)
    , oiLow :: !(V.Vector Double)
    , oiVolume :: !(V.Vector Double)
    , oiEma20 :: !(V.Vector (Maybe Double))
    , oiEma50 :: !(V.Vector (Maybe Double))
    , oiAdx14 :: !(V.Vector (Maybe AdxPoint))
    , oiAroon25 :: !(V.Vector (Maybe AroonPoint))
    , oiAtr14 :: !(V.Vector (Maybe Double))
    , oiRsi14 :: !(V.Vector (Maybe Double))
    , oiStochastic14 :: !(V.Vector (Maybe Double))
    , oiRoc10 :: !(V.Vector (Maybe Double))
    , oiMacd12269 :: !(V.Vector (Maybe MacdPoint))
    , oiBb202 :: !(V.Vector (Maybe Band))
    , oiKeltner2015 :: !(V.Vector (Maybe Band))
    , oiDonchian20 :: !(V.Vector (Maybe DonchianChannel))
    , oiObv :: !(V.Vector (Maybe Double))
    , oiAd :: !(V.Vector (Maybe Double))
    , oiCmf20 :: !(V.Vector (Maybe Double))
    , oiMfi14 :: !(V.Vector (Maybe Double))
    , oiVpt :: !(V.Vector (Maybe Double))
    , oiRegime :: !(V.Vector (Maybe Regime))
    }
    deriving (Eq, Show)

{- | Compute all indicator series once for the full OHLCV data.
This reduces backtest complexity from O(n²) to O(n).
-}
precomputeIndicators :: OhlcvSeries -> OhlcvIndicators
precomputeIndicators series =
    let closes = ohlcvClose series
        highs = ohlcvHigh series
        lows = ohlcvLow series
        volumes = ohlcvVolume series
        ema20 = emaSeries 20 closes
        ema50 = emaSeries 50 closes
        adx14 = adxSeries 14 highs lows closes
        aroon25 = aroonSeries 25 highs lows
        atr14 = atrSeries 14 highs lows closes
        rsi14 = rsiSeries 14 closes
        stochastic14 = stochasticKSeries 14 highs lows closes
        roc10 = rocSeries 10 closes
        macd12269 = macdSeries 12 26 9 closes
        bb202 = bollingerBandsSeries 20 2 closes
        keltner2015 = keltnerChannelsSeries 20 1.5 highs lows closes
        donchian20 = donchianChannelsSeries 20 highs lows
        obv = obvSeries closes volumes
        ad = accumulationDistributionSeries highs lows closes volumes
        cmf20 = cmfSeries 20 highs lows closes volumes
        mfi14 = mfiSeries 14 highs lows closes volumes
        vpt = vptSeries closes volumes
        n = V.length closes
        regimeAt t = do
            closeNow <- safeIndex closes t
            adxNow <- join (safeIndex adx14 t)
            aroonNow <- join (safeIndex aroon25 t)
            fastNow <- join (safeIndex ema20 t)
            fastPrev <- join (safeIndex ema20 (t - 5))
            bbNow <- join (safeIndex bb202 t)
            let slope = safeDivide (fastNow - fastPrev) closeNow
                width = safeDivide (bandUpper bbNow - bandLower bbNow) closeNow
                aroonGap = abs (aroonUp aroonNow - aroonDown aroonNow)
                adxTrendScore = clamp01 ((adxValue adxNow - 15) / 20)
                aroonTrendScore = clamp01 ((aroonGap - 20) / 40)
                slopeTrendScore = clamp01 ((abs slope - 0.005) / 0.015)
                trendScore = 0.40 * adxTrendScore + 0.35 * aroonTrendScore + 0.25 * slopeTrendScore
                adxRangeScore = clamp01 ((25 - adxValue adxNow) / 15)
                widthRangeScore = clamp01 ((0.12 - width) / 0.08)
                rangeScore = 0.5 * adxRangeScore + 0.5 * widthRangeScore
                maxScore = max trendScore rangeScore
                neutralScore = max 0 (1 - maxScore)
            pure $
                if trendScore >= 0.55
                    then RegimeTrend
                    else
                        if rangeScore >= 0.55
                            then RegimeRange
                            else RegimeNeutral
     in OhlcvIndicators
            { oiClose = closes
            , oiHigh = highs
            , oiLow = lows
            , oiVolume = volumes
            , oiEma20 = ema20
            , oiEma50 = ema50
            , oiAdx14 = adx14
            , oiAroon25 = aroon25
            , oiAtr14 = atr14
            , oiRsi14 = rsi14
            , oiStochastic14 = stochastic14
            , oiRoc10 = roc10
            , oiMacd12269 = macd12269
            , oiBb202 = bb202
            , oiKeltner2015 = keltner2015
            , oiDonchian20 = donchian20
            , oiObv = obv
            , oiAd = ad
            , oiCmf20 = cmf20
            , oiMfi14 = mfi14
            , oiVpt = vpt
            , oiRegime = V.generate n regimeAt
            }

safeIndex :: V.Vector a -> Int -> Maybe a
safeIndex vec idx
    | idx >= 0 && idx < V.length vec = Just (vec V.! idx)
    | otherwise = Nothing

-- | Trend-following candidate at a specific bar using precomputed indicators.
trendFollowingAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate
trendFollowingAt inds t = do
    closeNow <- safeIndex (oiClose inds) t
    fastNow <- join (safeIndex (oiEma20 inds) t)
    slowNow <- join (safeIndex (oiEma50 inds) t)
    adxNow <- join (safeIndex (oiAdx14 inds) t)
    aroonNow <- join (safeIndex (oiAroon25 inds) t)
    atrNow <- join (safeIndex (oiAtr14 inds) t)
    regimeNow <- join (safeIndex (oiRegime inds) t)
    let longTrend =
            regimeNow == RegimeTrend
                && fastNow > slowNow
                && adxValue adxNow >= 10
                && aroonUp aroonNow > aroonDown aroonNow
        shortTrend =
            regimeNow == RegimeTrend
                && fastNow < slowNow
                && adxValue adxNow >= 10
                && aroonDown aroonNow > aroonUp aroonNow
        maSpread = abs (safeDivide (fastNow - slowNow) closeNow)
        baseConfidence = clamp01 ((((adxValue adxNow - 20) / 25) + (maSpread * 8) + (abs (aroonUp aroonNow - aroonDown aroonNow) / 100)) / 3)
    if longTrend
        then
            Just
                StrategyCandidate
                    { scFamily = "trend-following"
                    , scName = "ema-crossover-adx-aroon-atr"
                    , scBias = BiasLong
                    , scConfidence = baseConfidence
                    , scEntryPrice = Just closeNow
                    , scStopPrice = Just (closeNow - (2 * atrNow))
                    , scTakeProfitPrice = Just (closeNow + (3 * atrNow))
                    , scReason = "Fast EMA is above slow EMA with ADX/Aroon trend confirmation and ATR-based risk framing."
                    }
        else
            if shortTrend
                then
                    Just
                        StrategyCandidate
                            { scFamily = "trend-following"
                            , scName = "ema-crossover-adx-aroon-atr"
                            , scBias = BiasShort
                            , scConfidence = baseConfidence
                            , scEntryPrice = Just closeNow
                            , scStopPrice = Just (closeNow + (2 * atrNow))
                            , scTakeProfitPrice = Just (closeNow - (3 * atrNow))
                            , scReason = "Fast EMA is below slow EMA with ADX/Aroon trend confirmation and ATR-based risk framing."
                            }
                else Nothing

-- | Momentum-reversion candidate at a specific bar using precomputed indicators.
momentumReversionAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate
momentumReversionAt inds t = do
    regimeNow <- join (safeIndex (oiRegime inds) t)
    closeNow <- safeIndex (oiClose inds) t
    rsiNow <- join (safeIndex (oiRsi14 inds) t)
    stochasticNow <- join (safeIndex (oiStochastic14 inds) t)
    rocNow <- join (safeIndex (oiRoc10 inds) t)
    macdNow <- join (safeIndex (oiMacd12269 inds) t)
    bollingerNow <- join (safeIndex (oiBb202 inds) t)
    keltnerNow <- join (safeIndex (oiKeltner2015 inds) t)
    atrNow <- join (safeIndex (oiAtr14 inds) t)
    let nearLowerEnvelope = closeNow <= max (bandLower bollingerNow) (bandLower keltnerNow) * 1.02
        nearUpperEnvelope = closeNow >= min (bandUpper bollingerNow) (bandUpper keltnerNow) * 0.98
        longSetup =
            regimeNow /= RegimeTrend
                && rsiNow <= 35
                && stochasticNow <= 20
                && rocNow < 0
                && macdValue macdNow >= macdSignal macdNow
                && nearLowerEnvelope
        shortSetup =
            regimeNow /= RegimeTrend
                && rsiNow >= 65
                && stochasticNow >= 80
                && rocNow > 0
                && macdValue macdNow <= macdSignal macdNow
                && nearUpperEnvelope
        baseConfidence = clamp01 (((abs (50 - rsiNow) / 50) + (abs (50 - stochasticNow) / 50) + min 1 (abs rocNow / 5)) / 3)
    if longSetup
        then
            Just
                StrategyCandidate
                    { scFamily = "momentum-reversion"
                    , scName = "rsi-stochastic-roc-macd-envelope"
                    , scBias = BiasLong
                    , scConfidence = baseConfidence
                    , scEntryPrice = Just closeNow
                    , scStopPrice = Just (closeNow - (1.5 * atrNow))
                    , scTakeProfitPrice = Just (bandMiddle bollingerNow)
                    , scReason = "Range-style long setup: RSI/Stochastic oversold, negative ROC, MACD confirmation, lower envelope context."
                    }
        else
            if shortSetup
                then
                    Just
                        StrategyCandidate
                            { scFamily = "momentum-reversion"
                            , scName = "rsi-stochastic-roc-macd-envelope"
                            , scBias = BiasShort
                            , scConfidence = baseConfidence
                            , scEntryPrice = Just closeNow
                            , scStopPrice = Just (closeNow + (1.5 * atrNow))
                            , scTakeProfitPrice = Just (bandMiddle bollingerNow)
                            , scReason = "Range-style short setup: RSI/Stochastic overbought, positive ROC, MACD confirmation, upper envelope context."
                            }
                else Nothing

-- | Volume-confirmed breakout candidate at a specific bar using precomputed indicators.
volumeConfirmedBreakoutAt :: OhlcvIndicators -> Int -> Maybe StrategyCandidate
volumeConfirmedBreakoutAt inds t = do
    closeNow <- safeIndex (oiClose inds) t
    donchianNow <- join (safeIndex (oiDonchian20 inds) (t - 1))
    atrNow <- join (safeIndex (oiAtr14 inds) t)
    obvNow <- join (safeIndex (oiObv inds) t)
    obvPrev <- join (safeIndex (oiObv inds) (t - 5))
    adNow <- join (safeIndex (oiAd inds) t)
    adPrev <- join (safeIndex (oiAd inds) (t - 5))
    cmfNow <- join (safeIndex (oiCmf20 inds) t)
    mfiNow <- join (safeIndex (oiMfi14 inds) t)
    vptNow <- join (safeIndex (oiVpt inds) t)
    vptPrev <- join (safeIndex (oiVpt inds) (t - 5))
    let obvUp = obvNow > obvPrev
        adUp = adNow > adPrev
        vptUp = vptNow > vptPrev
        longBreakout = closeNow > donchianUpper donchianNow && obvUp && adUp && vptUp && cmfNow > 0 && mfiNow >= 50
        shortBreakout = closeNow < donchianLower donchianNow && not obvUp && not adUp && not vptUp && cmfNow < 0 && mfiNow <= 50
        breakoutDistance = max 0 (safeDivide (abs (closeNow - midChannel donchianNow)) closeNow)
        baseConfidence = clamp01 (((if obvUp == adUp then 1 else 0.5) + min 1 (abs cmfNow) + min 1 (breakoutDistance * 10)) / 3)
    if longBreakout
        then
            Just
                StrategyCandidate
                    { scFamily = "volume-confirmed-breakout"
                    , scName = "donchian-obv-ad-cmf-mfi-vpt"
                    , scBias = BiasLong
                    , scConfidence = baseConfidence
                    , scEntryPrice = Just closeNow
                    , scStopPrice = Just (closeNow - (2 * atrNow))
                    , scTakeProfitPrice = Just (closeNow + (3 * atrNow))
                    , scReason = "Volume-confirmed breakout above Donchian upper band with OBV/AD/VPT divergence and CMF/MFI confirmation."
                    }
        else
            if shortBreakout
                then
                    Just
                        StrategyCandidate
                            { scFamily = "volume-confirmed-breakout"
                            , scName = "donchian-obv-ad-cmf-mfi-vpt"
                            , scBias = BiasShort
                            , scConfidence = baseConfidence
                            , scEntryPrice = Just closeNow
                            , scStopPrice = Just (closeNow + (2 * atrNow))
                            , scTakeProfitPrice = Just (closeNow - (3 * atrNow))
                            , scReason = "Volume-confirmed breakout below Donchian lower band with OBV/AD/VPT divergence and CMF/MFI confirmation."
                            }
                else Nothing

-- | Evaluate the best precomputed candidate at a specific bar.
bestCandidateAt :: TechnicalAnalysisGateInputs -> OhlcvIndicators -> Int -> Maybe GatedStrategyCandidate
bestCandidateAt inputs inds t =
    let candidates =
            catMaybes
                [ trendFollowingAt inds t >>= admitStrategyCandidate inputs
                , momentumReversionAt inds t >>= admitStrategyCandidate inputs
                , volumeConfirmedBreakoutAt inds t >>= admitStrategyCandidate inputs
                ]
     in listToMaybe $ sortOn (Down . candidateRank) candidates
  where
    candidateRank candidate =
        let rawConfidence = scConfidence (gscCandidate candidate)
            confidence = if finiteDouble rawConfidence then rawConfidence else 0
            edge = if finiteDouble (gscEntryEdge candidate) then gscEntryEdge candidate else 0
         in (confidence, edge)

-- | Evaluate a specific method at a bar using precomputed indicators.
candidateForMethodAt :: Method -> TechnicalAnalysisGateInputs -> OhlcvIndicators -> Int -> Maybe GatedStrategyCandidate
candidateForMethodAt method inputs inds t =
    case method of
        MethodTaTrend -> trendFollowingAt inds t >>= admitStrategyCandidate inputs
        MethodTaReversion -> momentumReversionAt inds t >>= admitStrategyCandidate inputs
        MethodTaBreakout -> volumeConfirmedBreakoutAt inds t >>= admitStrategyCandidate inputs
        MethodTaBest -> bestCandidateAt inputs inds t
        _ -> Nothing
