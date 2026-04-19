module Trader.TechnicalAnalysis.Strategies (
    OhlcvSeries (..),
    Regime (..),
    StrategyCandidate (..),
    TradeBias (..),
    momentumReversionCandidate,
    regimeSelector,
    strategyCandidates,
    trendFollowingCandidate,
    volumeConfirmedBreakoutCandidate,
) where

import qualified Data.Vector as V
import Trader.TechnicalAnalysis.Indicators

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

strategyCandidates :: OhlcvSeries -> [StrategyCandidate]
strategyCandidates series =
    catMaybes
        [ trendFollowingCandidate series
        , momentumReversionCandidate series
        , volumeConfirmedBreakoutCandidate series
        ]

regimeSelector :: OhlcvSeries -> Maybe Regime
regimeSelector series = do
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
    if adxValue adxNow >= 25 && aroonGap >= 40 && abs slope >= 0.01
        then Just RegimeTrend
        else
            if adxValue adxNow < 20 && width <= 0.08
                then Just RegimeRange
                else Just RegimeNeutral

trendFollowingCandidate :: OhlcvSeries -> Maybe StrategyCandidate
trendFollowingCandidate series = do
    validateSeries series
    closeNow <- lastValue (ohlcvClose series)
    fastNow <- latestJust (emaSeries 20 (ohlcvClose series))
    slowNow <- latestJust (emaSeries 50 (ohlcvClose series))
    adxNow <- latestJust (adxSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    aroonNow <- latestJust (aroonSeries 25 (ohlcvHigh series) (ohlcvLow series))
    atrNow <- latestJust (atrSeries 14 (ohlcvHigh series) (ohlcvLow series) (ohlcvClose series))
    regimeNow <- regimeSelector series
    let longTrend =
            regimeNow == RegimeTrend
                && fastNow > slowNow
                && adxValue adxNow >= 20
                && aroonUp aroonNow > aroonDown aroonNow
        shortTrend =
            regimeNow == RegimeTrend
                && fastNow < slowNow
                && adxValue adxNow >= 20
                && aroonDown aroonNow > aroonUp aroonNow
        maSpread = abs (safeDivide (fastNow - slowNow) closeNow)
        baseConfidence = clamp01 (((adxValue adxNow - 20) / 25) + (maSpread * 8) + (abs (aroonUp aroonNow - aroonDown aroonNow) / 100)) / 3
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

momentumReversionCandidate :: OhlcvSeries -> Maybe StrategyCandidate
momentumReversionCandidate series = do
    validateSeries series
    regimeNow <- regimeSelector series
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
        baseConfidence = clamp01 ((abs (50 - rsiNow) / 50) + (abs (50 - stochasticNow) / 50) + min 1 (abs rocNow / 5)) / 3
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
        baseConfidence = clamp01 ((if obvUp == adUp then 1 else 0.5) + min 1 (abs cmfNow) + min 1 breakoutDistance * 10) / 3
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

clamp01 :: Double -> Double
clamp01 = max 0 . min 1

midChannel :: DonchianChannel -> Double
midChannel channel = (donchianUpper channel + donchianLower channel) / 2

catMaybes :: [Maybe a] -> [a]
catMaybes = foldr step []
  where
    step (Just value) acc = value : acc
    step Nothing acc = acc
