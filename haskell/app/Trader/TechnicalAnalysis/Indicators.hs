module Trader.TechnicalAnalysis.Indicators (
    AdxPoint (..),
    AroonPoint (..),
    Band (..),
    DonchianChannel (..),
    MacdPoint (..),
    accumulationDistributionSeries,
    adxSeries,
    aroonSeries,
    atrSeries,
    bollingerBandsSeries,
    cmfSeries,
    donchianChannelsSeries,
    emaSeries,
    keltnerChannelsSeries,
    latestJust,
    macdSeries,
    mfiSeries,
    obvSeries,
    rocSeries,
    rsiSeries,
    smaSeries,
    stochasticKSeries,
    vptSeries,
) where

import Data.List (foldl')
import Data.Maybe (catMaybes)
import qualified Data.Vector as V

-- | Generic band container used for Bollinger/Keltner channels.
data Band = Band
    { bandLower :: !Double
    , bandMiddle :: !Double
    , bandUpper :: !Double
    }
    deriving (Eq, Show)

-- | Latest MACD values for a bar.
data MacdPoint = MacdPoint
    { macdValue :: !Double
    , macdSignal :: !Double
    , macdHistogram :: !Double
    }
    deriving (Eq, Show)

-- | ADX plus directional movement components.
data AdxPoint = AdxPoint
    { adxValue :: !Double
    , adxPlusDi :: !Double
    , adxMinusDi :: !Double
    }
    deriving (Eq, Show)

-- | Aroon up/down scores over a lookback window.
data AroonPoint = AroonPoint
    { aroonUp :: !Double
    , aroonDown :: !Double
    }
    deriving (Eq, Show)

-- | Donchian upper/lower bands.
data DonchianChannel = DonchianChannel
    { donchianLower :: !Double
    , donchianUpper :: !Double
    }
    deriving (Eq, Show)

latestJust :: V.Vector (Maybe a) -> Maybe a
latestJust values =
    go (V.length values - 1)
  where
    go idx
        | idx < 0 = Nothing
        | otherwise =
            case values V.! idx of
                Just value -> Just value
                Nothing -> go (idx - 1)

smaSeries :: Int -> V.Vector Double -> V.Vector (Maybe Double)
smaSeries period values
    | period <= 0 = V.replicate (V.length values) Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length values
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                window = V.slice start period values
             in Just (V.sum window / fromIntegral period)

emaSeries :: Int -> V.Vector Double -> V.Vector (Maybe Double)
emaSeries period values
    | period <= 0 = V.replicate (V.length values) Nothing
    | V.length values < period = V.replicate (V.length values) Nothing
    | otherwise =
        let seedWindow = V.slice 0 period values
            seed = V.sum seedWindow / fromIntegral period
            multiplier = 2 / (fromIntegral period + 1)
            tailValues = V.toList (V.slice period (V.length values - period) values)
            rest =
                reverse $
                    foldl'
                        ( \acc price ->
                            let prevEma = case acc of
                                    [] -> seed
                                    (e : _) -> e
                                nextEma = ((price - prevEma) * multiplier) + prevEma
                             in nextEma : acc
                        )
                        []
                        tailValues
         in V.fromList $ replicate (period - 1) Nothing ++ (Just seed : map Just rest)

rocSeries :: Int -> V.Vector Double -> V.Vector (Maybe Double)
rocSeries period values
    | period <= 0 = V.replicate (V.length values) Nothing
    | otherwise =
        V.generate (V.length values) step
  where
    step idx
        | idx < period = Nothing
        | otherwise =
            let prev = values V.! (idx - period)
                cur = values V.! idx
             in if prev == 0 then Nothing else Just (((cur - prev) / prev) * 100)

rsiSeries :: Int -> V.Vector Double -> V.Vector (Maybe Double)
rsiSeries period values
    | period <= 0 = V.replicate n Nothing
    | n <= period = V.replicate n Nothing
    | otherwise =
        let deltas = [values V.! i - values V.! (i - 1) | i <- [1 .. n - 1]]
            gains = map (max 0) deltas
            losses = map (max 0 . negate) deltas
            avgGain0 = sum (take period gains) / fromIntegral period
            avgLoss0 = sum (take period losses) / fromIntegral period
            firstRsi = rsiFromAverages avgGain0 avgLoss0
            tailPairs = drop period (zip gains losses)
            (_, _, rsiAcc) =
                foldl'
                    ( \(avgGain, avgLoss, acc) (gainValue, lossValue) ->
                        let nextGain = ((avgGain * fromIntegral (period - 1)) + gainValue) / fromIntegral period
                            nextLoss = ((avgLoss * fromIntegral (period - 1)) + lossValue) / fromIntegral period
                         in (nextGain, nextLoss, rsiFromAverages nextGain nextLoss : acc)
                    )
                    (avgGain0, avgLoss0, [])
                    tailPairs
            rest = reverse rsiAcc
         in V.fromList $ replicate period Nothing ++ (Just firstRsi : map Just rest)
  where
    n = V.length values

stochasticKSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
stochasticKSeries period highs lows closes
    | period <= 0 = V.replicate n Nothing
    | not (sameLength [highs, lows, closes]) = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length closes
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                highWindow = V.slice start period highs
                lowWindow = V.slice start period lows
                highestHigh = V.maximum highWindow
                lowestLow = V.minimum lowWindow
                rangeValue = highestHigh - lowestLow
                closeValue = closes V.! idx
             in if rangeValue == 0 then Nothing else Just (((closeValue - lowestLow) / rangeValue) * 100)

macdSeries :: Int -> Int -> Int -> V.Vector Double -> V.Vector (Maybe MacdPoint)
macdSeries fastPeriod slowPeriod signalPeriod closes
    | fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0 = V.replicate n Nothing
    | fastPeriod >= slowPeriod = V.replicate n Nothing
    | otherwise =
        let fastEma = emaSeries fastPeriod closes
            slowEma = emaSeries slowPeriod closes
            macdLine = V.zipWith combine fastEma slowEma
            signalLine = emaSeriesFromMaybe signalPeriod macdLine
         in V.generate n (step macdLine signalLine)
  where
    n = V.length closes
    combine (Just fastValue) (Just slowValue) = Just (fastValue - slowValue)
    combine _ _ = Nothing
    step macdLine signalLine idx =
        case (macdLine V.! idx, signalLine V.! idx) of
            (Just macdValueNow, Just signalValueNow) ->
                Just
                    MacdPoint
                        { macdValue = macdValueNow
                        , macdSignal = signalValueNow
                        , macdHistogram = macdValueNow - signalValueNow
                        }
            _ -> Nothing

atrSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
atrSeries period highs lows closes
    | period <= 0 = V.replicate n Nothing
    | not (sameLength [highs, lows, closes]) = V.replicate n Nothing
    | n == 0 = V.empty
    | n <= period = V.replicate n Nothing
    | otherwise =
        let trueRanges = V.generate n trAt
            seed = V.sum (V.slice 1 period trueRanges) / fromIntegral period
            tailValues = V.toList (V.slice (period + 1) (n - period - 1) trueRanges)
            rest =
                reverse $
                    foldl'
                        ( \acc trValue ->
                            let prevAtr = case acc of
                                    [] -> seed
                                    (a : _) -> a
                                nextAtr = ((prevAtr * fromIntegral (period - 1)) + trValue) / fromIntegral period
                             in nextAtr : acc
                        )
                        []
                        tailValues
         in V.fromList $ replicate period Nothing ++ (Just seed : map Just rest)
  where
    n = V.length closes
    trAt 0 = highs V.! 0 - lows V.! 0
    trAt idx =
        let highValue = highs V.! idx
            lowValue = lows V.! idx
            prevClose = closes V.! (idx - 1)
         in maximum
                [ highValue - lowValue
                , abs (highValue - prevClose)
                , abs (lowValue - prevClose)
                ]

adxSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe AdxPoint)
adxSeries period highs lows closes
    | period <= 0 = V.replicate n Nothing
    | not (sameLength [highs, lows, closes]) = V.replicate n Nothing
    | n <= (period * 2) = V.replicate n Nothing
    | otherwise =
        let plusDm = V.generate n plusDmAt
            minusDm = V.generate n minusDmAt
            atrVals = atrSeries period highs lows closes
            smoothPlusDm = smoothedSeries period plusDm
            smoothMinusDm = smoothedSeries period minusDm
            dxSeriesValues =
                V.generate n (dxAt atrVals smoothPlusDm smoothMinusDm)
            seedDx = averageMaybes (V.slice period period dxSeriesValues)
            restDx = V.toList (V.slice (period * 2) (n - (period * 2)) dxSeriesValues)
            adxOut =
                case seedDx of
                    Nothing -> replicate n Nothing
                    Just seedValue ->
                        let (_, adxReversed) =
                                foldl'
                                    ( \(prevAdx, acc) nextDx ->
                                        case nextDx of
                                            Nothing -> (prevAdx, Nothing : acc)
                                            Just dxValue ->
                                                let nextAdx = ((prevAdx * fromIntegral (period - 1)) + dxValue) / fromIntegral period
                                                 in (nextAdx, Just nextAdx : acc)
                                    )
                                    (seedValue, [Just seedValue])
                                    restDx
                         in replicate (period * 2 - 1) Nothing ++ reverse adxReversed
         in V.generate n (assemble atrVals smoothPlusDm smoothMinusDm (V.fromList adxOut))
  where
    n = V.length closes
    plusDmAt 0 = 0
    plusDmAt idx =
        let upMove = highs V.! idx - highs V.! (idx - 1)
            downMove = lows V.! (idx - 1) - lows V.! idx
         in if upMove > downMove && upMove > 0 then upMove else 0
    minusDmAt 0 = 0
    minusDmAt idx =
        let upMove = highs V.! idx - highs V.! (idx - 1)
            downMove = lows V.! (idx - 1) - lows V.! idx
         in if downMove > upMove && downMove > 0 then downMove else 0
    dxAt atrVals plusVals minusVals idx =
        case (atrVals V.! idx, plusVals V.! idx, minusVals V.! idx) of
            (Just atrValue, Just plusValue, Just minusValue)
                | atrValue > 0 ->
                    let plusDiValue = (plusValue / atrValue) * 100
                        minusDiValue = (minusValue / atrValue) * 100
                        denom = plusDiValue + minusDiValue
                     in if denom == 0 then Nothing else Just ((abs (plusDiValue - minusDiValue) / denom) * 100)
            _ -> Nothing
    assemble atrVals plusVals minusVals adxVals idx =
        case (atrVals V.! idx, plusVals V.! idx, minusVals V.! idx, adxVals V.! idx) of
            (Just atrValue, Just plusValue, Just minusValue, Just adxValueNow)
                | atrValue > 0 ->
                    let plusDiValue = (plusValue / atrValue) * 100
                        minusDiValue = (minusValue / atrValue) * 100
                     in Just
                            AdxPoint
                                { adxValue = adxValueNow
                                , adxPlusDi = plusDiValue
                                , adxMinusDi = minusDiValue
                                }
            _ -> Nothing

aroonSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe AroonPoint)
aroonSeries period highs lows
    | period <= 1 = V.replicate n Nothing
    | V.length highs /= V.length lows = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length highs
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                highWindow = V.slice start period highs
                lowWindow = V.slice start period lows
                highestIdx = V.maxIndex highWindow
                lowestIdx = V.minIndex lowWindow
                barsSinceHigh = period - 1 - highestIdx
                barsSinceLow = period - 1 - lowestIdx
                scale = 100 / fromIntegral (period - 1)
             in Just
                    AroonPoint
                        { aroonUp = 100 - fromIntegral barsSinceHigh * scale
                        , aroonDown = 100 - fromIntegral barsSinceLow * scale
                        }

bollingerBandsSeries :: Int -> Double -> V.Vector Double -> V.Vector (Maybe Band)
bollingerBandsSeries period stdevMult closes
    | period <= 1 = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length closes
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                window = V.slice start period closes
                meanValue = V.sum window / fromIntegral period
                variance = V.sum (V.map (\x -> let d = x - meanValue in d * d) window) / fromIntegral period
                sigma = sqrt variance
             in Just
                    Band
                        { bandLower = meanValue - (stdevMult * sigma)
                        , bandMiddle = meanValue
                        , bandUpper = meanValue + (stdevMult * sigma)
                        }

keltnerChannelsSeries :: Int -> Double -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Band)
keltnerChannelsSeries period atrMult highs lows closes =
    let middle = emaSeries period closes
        atrVals = atrSeries period highs lows closes
     in V.generate n (step middle atrVals)
  where
    n = V.length closes
    step middle atrVals idx =
        case (middle V.! idx, atrVals V.! idx) of
            (Just middleValue, Just atrValue) ->
                Just
                    Band
                        { bandLower = middleValue - (atrMult * atrValue)
                        , bandMiddle = middleValue
                        , bandUpper = middleValue + (atrMult * atrValue)
                        }
            _ -> Nothing

donchianChannelsSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe DonchianChannel)
donchianChannelsSeries period highs lows
    | period <= 0 = V.replicate n Nothing
    | V.length highs /= V.length lows = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length highs
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                highWindow = V.slice start period highs
                lowWindow = V.slice start period lows
             in Just
                    DonchianChannel
                        { donchianLower = V.minimum lowWindow
                        , donchianUpper = V.maximum highWindow
                        }

obvSeries :: V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
obvSeries closes volumes
    | V.length closes /= V.length volumes = V.replicate (max (V.length closes) (V.length volumes)) Nothing
    | V.null closes = V.empty
    | otherwise = V.fromList (go 1 0 [Just 0])
  where
    n = V.length closes
    go idx prevObv acc
        | idx >= n = reverse acc
        | otherwise =
            let closeNow = closes V.! idx
                closePrev = closes V.! (idx - 1)
                volumeNow = volumes V.! idx
                nextObv
                    | closeNow > closePrev = prevObv + volumeNow
                    | closeNow < closePrev = prevObv - volumeNow
                    | otherwise = prevObv
             in go (idx + 1) nextObv (Just nextObv : acc)

accumulationDistributionSeries :: V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
accumulationDistributionSeries highs lows closes volumes
    | not (sameLength [highs, lows, closes, volumes]) = V.replicate n Nothing
    | n == 0 = V.empty
    | otherwise = V.fromList (reverse (snd (foldl' step (0, []) [0 .. n - 1])))
  where
    n = V.length closes
    step (prevValue, acc) idx =
        let highValue = highs V.! idx
            lowValue = lows V.! idx
            closeValue = closes V.! idx
            volumeValue = volumes V.! idx
            rangeValue = highValue - lowValue
            moneyFlowMultiplier =
                if rangeValue == 0
                    then 0
                    else ((closeValue - lowValue) - (highValue - closeValue)) / rangeValue
            nextValue = prevValue + (moneyFlowMultiplier * volumeValue)
         in (nextValue, Just nextValue : acc)

cmfSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
cmfSeries period highs lows closes volumes
    | period <= 0 = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length closes
    step idx
        | idx + 1 < period = Nothing
        | otherwise =
            let start = idx + 1 - period
                moneyFlowVolumes =
                    V.generate
                        period
                        ( \offset ->
                            let i = start + offset
                                highValue = highs V.! i
                                lowValue = lows V.! i
                                closeValue = closes V.! i
                                volumeValue = volumes V.! i
                                rangeValue = highValue - lowValue
                                multiplier =
                                    if rangeValue == 0
                                        then 0
                                        else ((closeValue - lowValue) - (highValue - closeValue)) / rangeValue
                             in multiplier * volumeValue
                        )
                volumeWindow = V.slice start period volumes
                denom = V.sum volumeWindow
             in if denom == 0 then Nothing else Just (V.sum moneyFlowVolumes / denom)

mfiSeries :: Int -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
mfiSeries period highs lows closes volumes
    | period <= 0 = V.replicate n Nothing
    | not (sameLength [highs, lows, closes, volumes]) = V.replicate n Nothing
    | n <= period = V.replicate n Nothing
    | otherwise =
        V.generate n step
  where
    n = V.length closes
    typicalPrice idx = ((highs V.! idx) + (lows V.! idx) + (closes V.! idx)) / 3
    rawMoneyFlow idx = typicalPrice idx * (volumes V.! idx)
    step idx
        | idx < period = Nothing
        | otherwise =
            let indices = [idx - period + 1 .. idx]
                (positiveFlow, negativeFlow) =
                    foldl'
                        ( \(posAcc, negAcc) i ->
                            if i == 0
                                then (posAcc, negAcc)
                                else
                                    let prevTypical = typicalPrice (i - 1)
                                        curTypical = typicalPrice i
                                        flowValue = rawMoneyFlow i
                                     in if curTypical > prevTypical
                                            then (posAcc + flowValue, negAcc)
                                            else
                                                if curTypical < prevTypical
                                                    then (posAcc, negAcc + flowValue)
                                                    else (posAcc, negAcc)
                        )
                        (0, 0)
                        indices
             in if negativeFlow == 0
                    then Just 100
                    else
                        let moneyRatio = positiveFlow / negativeFlow
                         in Just (100 - (100 / (1 + moneyRatio)))

vptSeries :: V.Vector Double -> V.Vector Double -> V.Vector (Maybe Double)
vptSeries closes volumes
    | V.length closes /= V.length volumes = V.replicate (max (V.length closes) (V.length volumes)) Nothing
    | V.null closes = V.empty
    | otherwise = V.fromList (go 1 0 [Just 0])
  where
    n = V.length closes
    go idx prevVpt acc
        | idx >= n = reverse acc
        | otherwise =
            let closeNow = closes V.! idx
                closePrev = closes V.! (idx - 1)
                volumeNow = volumes V.! idx
                delta = if closePrev == 0 then 0 else ((closeNow - closePrev) / closePrev) * volumeNow
                nextVpt = prevVpt + delta
             in go (idx + 1) nextVpt (Just nextVpt : acc)

sameLength :: [V.Vector a] -> Bool
sameLength [] = True
sameLength (x : xs) = all ((== V.length x) . V.length) xs

rsiFromAverages :: Double -> Double -> Double
rsiFromAverages avgGain avgLoss
    | avgLoss == 0 = 100
    | otherwise =
        let rs = avgGain / avgLoss
         in 100 - (100 / (1 + rs))

averageMaybes :: V.Vector (Maybe Double) -> Maybe Double
averageMaybes values =
    let present = catMaybes (V.toList values)
     in if null present then Nothing else Just (sum present / fromIntegral (length present))

emaSeriesFromMaybe :: Int -> V.Vector (Maybe Double) -> V.Vector (Maybe Double)
emaSeriesFromMaybe period values
    | period <= 0 = V.replicate (V.length values) Nothing
    | otherwise =
        let indexed = [(idx, value) | (idx, Just value) <- zip [0 ..] (V.toList values)]
         in if length indexed < period
                then V.replicate (V.length values) Nothing
                else
                    let seedEntries = take period indexed
                        seed = sum (map snd seedEntries) / fromIntegral period
                        seedIndex = fst (last seedEntries)
                        multiplier = 2 / (fromIntegral period + 1)
                        revAcc =
                            foldl'
                                ( \acc (_, value) ->
                                    let prevEma = case acc of
                                            [] -> seed
                                            (e : _) -> e
                                        nextEma = ((value - prevEma) * multiplier) + prevEma
                                     in nextEma : acc
                                )
                                []
                                (drop period indexed)
                        out = Just seed : map Just (reverse revAcc)
                        prefix = replicate seedIndex Nothing
                        suffix = replicate (V.length values - seedIndex - 1 - length revAcc) Nothing
                     in V.fromList (prefix ++ out ++ suffix)

smoothedSeries :: Int -> V.Vector Double -> V.Vector (Maybe Double)
smoothedSeries period values
    | period <= 0 = V.replicate n Nothing
    | n <= period = V.replicate n Nothing
    | otherwise =
        let seed = V.sum (V.slice 1 period values)
            tailValues = V.toList (V.slice (period + 1) (n - period - 1) values)
            rest =
                reverse $
                    foldl'
                        ( \acc nextValue ->
                            let prevSmooth = case acc of
                                    [] -> seed
                                    (s : _) -> s
                                nextSmooth = prevSmooth - (prevSmooth / fromIntegral period) + nextValue
                             in nextSmooth : acc
                        )
                        []
                        tailValues
         in V.fromList $ replicate period Nothing ++ (Just seed : map Just rest)
  where
    n = V.length values
