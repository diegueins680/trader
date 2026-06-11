module Trader.Predictors.Features (
    FeatureSpec (..),
    FeatureInputs (..),
    mkFeatureSpec,
    mkFeatureInputs,
    withExogenousInputs,
    withCoinbaseInputs,
    featureInputsFromClose,
    featuresAt,
    featuresAtWithMarket,
    featuresAtWithInputs,
    featuresAtWithInputsWithMarket,
    forwardReturnAt,
    buildDatasetWithIndex,
    buildDatasetWithIndexWithMarket,
    buildDatasetWithIndexWithInputs,
    buildDatasetWithIndexWithInputsWithMarket,
    buildDataset,
    buildDatasetWithMarket,
    buildDatasetWithInputs,
    buildDatasetWithInputsWithMarket,
) where

import Data.Maybe (fromMaybe, isJust, isNothing)
import qualified Data.Vector as V

import Trader.MarketContext (MarketModel (..))

data FeatureSpec = FeatureSpec
    { fsLookbackBars :: !Int
    , fsShortBars :: !Int
    , fsMidBars :: !Int
    }
    deriving (Eq, Show)

data FeatureInputs = FeatureInputs
    { fiClose :: !(V.Vector Double)
    , fiOpen :: !(Maybe (V.Vector Double))
    , fiHigh :: !(Maybe (V.Vector Double))
    , fiLow :: !(Maybe (V.Vector Double))
    , fiVolume :: !(Maybe (V.Vector Double))
    , -- Exogenous derivatives series, pre-aligned 1:1 with fiClose and
      -- point-in-time (each value known by that bar's close). Nothing => the
      -- corresponding exogenous features contribute a neutral 0, so existing
      -- call sites and trained models behave exactly as before.
      fiFunding :: !(Maybe (V.Vector Double)) -- perp funding rate (level)
    , fiOpenInterest :: !(Maybe (V.Vector Double)) -- open interest (level)
    , fiBasis :: !(Maybe (V.Vector Double)) -- (mark - index) / index
    , fiTakerFlow :: !(Maybe (V.Vector Double)) -- taker buy/sell volume ratio (CVD proxy)
    , fiCoinbaseClose :: !(Maybe (V.Vector Double))
    -- ^ Same-asset cross-exchange (Coinbase) closes, aligned to the same bar
    -- grid as 'fiClose'. 'Nothing' (the default) leaves the feature vector
    -- byte-identical to the Binance-only behavior; its presence switches on the
    -- cross-exchange feature block.
    }
    deriving (Eq, Show)

data Bar = Bar
    { barOpen :: !Double
    , barHigh :: !Double
    , barLow :: !Double
    , barClose :: !Double
    , barVolume :: !Double
    }
    deriving (Eq, Show)

mkFeatureSpec :: Int -> FeatureSpec
mkFeatureSpec lookbackBars =
    let lb = max 2 lookbackBars
        shortB = max 1 (min 12 (lb - 1))
        midB = max 1 (min 48 (lb - 1))
     in FeatureSpec{fsLookbackBars = lb, fsShortBars = shortB, fsMidBars = midB}

mkFeatureInputs ::
    V.Vector Double ->
    Maybe (V.Vector Double) ->
    Maybe (V.Vector Double) ->
    Maybe (V.Vector Double) ->
    Maybe (V.Vector Double) ->
    FeatureInputs
mkFeatureInputs closes opens highs lows volumes =
    FeatureInputs
        { fiClose = closes
        , fiOpen = opens
        , fiHigh = highs
        , fiLow = lows
        , fiVolume = volumes
        , fiFunding = Nothing
        , fiOpenInterest = Nothing
        , fiBasis = Nothing
        , fiTakerFlow = Nothing
        , fiCoinbaseClose = Nothing
        }

featureInputsFromClose :: V.Vector Double -> FeatureInputs
featureInputsFromClose closes = mkFeatureInputs closes Nothing Nothing Nothing Nothing

{- | Attach exogenous derivatives series to an existing 'FeatureInputs'. Each
vector must already be aligned 1:1 with 'fiClose' and point-in-time (see
'Trader.Predictors.Exogenous.alignToBars'). Pass 'Nothing' to leave a series
unset (its features stay neutral 0).
-}
withExogenousInputs ::
    Maybe (V.Vector Double) -> -- funding
    Maybe (V.Vector Double) -> -- open interest
    Maybe (V.Vector Double) -> -- basis
    Maybe (V.Vector Double) -> -- taker flow
    FeatureInputs ->
    FeatureInputs
withExogenousInputs funding oi basis takerFlow inputs =
    inputs
        { fiFunding = funding
        , fiOpenInterest = oi
        , fiBasis = basis
        , fiTakerFlow = takerFlow
        }

{- | Attach (or clear) the same-asset cross-exchange close series. The vector,
when present, must be aligned to the same bar grid as 'fiClose'.
-}
withCoinbaseInputs :: Maybe (V.Vector Double) -> FeatureInputs -> FeatureInputs
withCoinbaseInputs cbCloses inputs = inputs{fiCoinbaseClose = cbCloses}

-- | Forward return r_t = p_{t+1}/p_t - 1.
forwardReturnAt :: V.Vector Double -> Int -> Maybe Double
forwardReturnAt prices t =
    if t < 0 || t + 1 >= V.length prices
        then Nothing
        else
            let p0 = prices V.! t
                p1 = prices V.! (t + 1)
             in finiteReturn p0 p1

{- | Feature vector at bar t using only prices up to t.
Requires at least fsLookbackBars history (prices window ending at t).
-}
featuresAt :: FeatureSpec -> V.Vector Double -> Int -> Maybe [Double]
featuresAt fs prices = featuresAtWithInputsWithMarket fs Nothing (featureInputsFromClose prices)

{- | Market-aware feature vector at bar t using only data available by bar t.
Cross-symbol features fall back to zero when market context is missing/invalid.
-}
featuresAtWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> Int -> Maybe [Double]
featuresAtWithMarket fs mMarket prices = featuresAtWithInputsWithMarket fs mMarket (featureInputsFromClose prices)

featuresAtWithInputs :: FeatureSpec -> FeatureInputs -> Int -> Maybe [Double]
featuresAtWithInputs fs = featuresAtWithInputsWithMarket fs Nothing

{- | Market-aware feature vector using close/open/high/low/volume context when
available. Missing OHLCV fields fall back to synthetic values so the feature
dimension stays stable across CSV, exchange, and close-only call sites.
-}
featuresAtWithInputsWithMarket :: FeatureSpec -> Maybe MarketModel -> FeatureInputs -> Int -> Maybe [Double]
featuresAtWithInputsWithMarket fs mMarket inputs t = do
    let closes = fiClose inputs
        lb = fsLookbackBars fs
        maxLag = max 1 (lb - 1)
        shortB = min (fsShortBars fs) maxLag
        midB = min (fsMidBars fs) maxLag
        ret3Bars = min 3 maxLag
    if t < lb - 1 || t >= V.length closes
        then Nothing
        else do
            ret1 <- retOver closes t 1
            ret3 <- retOver closes t ret3Bars
            retShort <- retOver closes t shortB
            retMid <- retOver closes t midB
            retLb <- retOver closes t (lb - 1)
            rsShort <- returnsEndingAt closes t shortB
            rsMid <- returnsEndingAt closes t midB
            barNow <- barAt inputs t
            let (muS, sigS) = meanStd rsShort
                (muM, sigM) = meanStd rsMid
                priceT = barClose barNow
                psych = psychologicalFeatures priceT
                marketFeats = marketFeaturesFromModel mMarket t ret1 rsShort
                klineFeats = klineFeatures inputs barNow t ret1 retShort retMid rsShort rsMid shortB midB
                eps = 1e-12
                retSpread = retShort - retMid
                retMeanReversion = ret1 - muS
                volRatio = if abs sigM <= eps then 0 else sigS / sigM
                trendSlope = muS - muM
                exoFeats = exogenousFeatures inputs t
                crossFeats =
                    if isJust (fiCoinbaseClose inputs)
                        then crossExchangeFeatures inputs t ret1 shortB
                        else []
                feats =
                    [ ret1
                    , ret3
                    , retShort
                    , retMid
                    , retLb
                    , muS
                    , sigS
                    , muM
                    , sigM
                    , retSpread
                    , retMeanReversion
                    , volRatio
                    , trendSlope
                    ]
                        ++ klineFeats
                        ++ marketFeats
                        ++ psych
                        ++ exoFeats
                        ++ crossFeats
            if all isFiniteDouble feats
                then pure feats
                else Nothing

{- | Exogenous derivatives features at bar @t@ (point-in-time — reads only
indices <= t). Always returns a fixed-width vector so the feature dimension is
stable whether or not the series are supplied; an absent, out-of-range, or
non-finite reading contributes a neutral 0. Inputs are stationarized (deltas /
centering) rather than passed as raw levels, matching the price-feature
convention above.

Returns @[]@ when no exogenous series is attached at all, so call sites and
models that do not opt in get a byte-identical feature vector (and unchanged
input dimension) — the presence of any series is itself the opt-in.
-}
exogenousFeatures :: FeatureInputs -> Int -> [Double]
exogenousFeatures inputs t
    | all isNothing [fiFunding inputs, fiOpenInterest inputs, fiBasis inputs, fiTakerFlow inputs] = []
    | otherwise =
        [ levelAt (fiFunding inputs) -- funding rate level (already small/stationary)
        , deltaAt (fiFunding inputs) -- funding momentum (Δ)
        , relDeltaAt (fiOpenInterest inputs) -- ΔOI / |OI| (position buildup)
        , levelAt (fiBasis inputs) -- basis (mean-reverting premium)
        , centeredAt (fiTakerFlow inputs) 1.0 -- taker buy/sell imbalance, centered at 0
        ]
  where
    valAt mv i = case mv of
        Just v
            | i >= 0 && i < V.length v ->
                let x = v V.! i in if isFiniteDouble x then Just x else Nothing
        _ -> Nothing
    levelAt mv = fromMaybe 0 (valAt mv t)
    centeredAt mv c = maybe 0 (subtract c) (valAt mv t)
    deltaAt mv = fromMaybe 0 $ do
        a <- valAt mv (t - 1)
        b <- valAt mv t
        pure (b - a)
    relDeltaAt mv = fromMaybe 0 $ do
        a <- valAt mv (t - 1)
        b <- valAt mv t
        pure (if abs a <= 1e-12 then 0 else (b - a) / abs a)

{- | Build a supervised dataset (features at t, target forward return at t) with bar indices.
Uses t in [lookbackBars-1 .. n-2].
-}
buildDatasetWithIndex :: FeatureSpec -> V.Vector Double -> [(Int, [Double], Double)]
buildDatasetWithIndex fs = buildDatasetWithIndexWithInputsWithMarket fs Nothing . featureInputsFromClose

{- | Market-aware dataset builder; cross-symbol features fall back to zero when
market context is missing/invalid.
-}
buildDatasetWithIndexWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> [(Int, [Double], Double)]
buildDatasetWithIndexWithMarket fs mMarket = buildDatasetWithIndexWithInputsWithMarket fs mMarket . featureInputsFromClose

buildDatasetWithIndexWithInputs :: FeatureSpec -> FeatureInputs -> [(Int, [Double], Double)]
buildDatasetWithIndexWithInputs fs = buildDatasetWithIndexWithInputsWithMarket fs Nothing

buildDatasetWithIndexWithInputsWithMarket :: FeatureSpec -> Maybe MarketModel -> FeatureInputs -> [(Int, [Double], Double)]
buildDatasetWithIndexWithInputsWithMarket fs mMarket inputs =
    let closes = fiClose inputs
        n = V.length closes
        startT = fsLookbackBars fs - 1
        endT = n - 2
     in if startT > endT
            then []
            else
                [ (t, feats, target)
                | t <- [startT .. endT]
                , Just feats <- [featuresAtWithInputsWithMarket fs mMarket inputs t]
                , Just target <- [forwardReturnAt closes t]
                , isFiniteDouble target
                ]

{- | Build a supervised dataset (features at t, target forward return at t).
Uses t in [lookbackBars-1 .. n-2].
-}
buildDataset :: FeatureSpec -> V.Vector Double -> [([Double], Double)]
buildDataset fs = buildDatasetWithInputsWithMarket fs Nothing . featureInputsFromClose

buildDatasetWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> [([Double], Double)]
buildDatasetWithMarket fs mMarket = buildDatasetWithInputsWithMarket fs mMarket . featureInputsFromClose

buildDatasetWithInputs :: FeatureSpec -> FeatureInputs -> [([Double], Double)]
buildDatasetWithInputs fs = buildDatasetWithInputsWithMarket fs Nothing

buildDatasetWithInputsWithMarket :: FeatureSpec -> Maybe MarketModel -> FeatureInputs -> [([Double], Double)]
buildDatasetWithInputsWithMarket fs mMarket inputs =
    [(feats, target) | (_, feats, target) <- buildDatasetWithIndexWithInputsWithMarket fs mMarket inputs]

retOver :: V.Vector Double -> Int -> Int -> Maybe Double
retOver prices t bars =
    if bars <= 0 || t - bars < 0
        then Nothing
        else
            let p0 = prices V.! (t - bars)
                p1 = prices V.! t
             in finiteReturn p0 p1

returnsEndingAt :: V.Vector Double -> Int -> Int -> Maybe [Double]
returnsEndingAt prices t k =
    if k <= 0 || t - k < 0
        then Nothing
        else
            let rs =
                    [ let p0 = prices V.! i
                          p1 = prices V.! (i + 1)
                       in finiteReturn p0 p1
                    | i <- [t - k .. t - 1]
                    ]
             in sequence rs

barAt :: FeatureInputs -> Int -> Maybe Bar
barAt inputs t
    | t < 0 = Nothing
    | t >= V.length closes = Nothing
    | otherwise =
        let closePx = closes V.! t
         in if not (isFiniteDouble closePx)
                then Nothing
                else
                    let prevClose =
                            if t <= 0
                                then closePx
                                else closeOr closePx (t - 1)
                        openPx = fromMaybe prevClose (validVectorValue (fiOpen inputs) t)
                        highBase = max openPx closePx
                        lowBase = min openPx closePx
                        highPx =
                            case validVectorValue (fiHigh inputs) t of
                                Just v -> max highBase v
                                Nothing -> highBase
                        lowPx =
                            case validVectorValue (fiLow inputs) t of
                                Just v -> min lowBase v
                                Nothing -> lowBase
                        volumePx =
                            case validVectorValue (fiVolume inputs) t of
                                Just v | v >= 0 -> v
                                _ -> 1
                        bar =
                            Bar
                                { barOpen = openPx
                                , barHigh = highPx
                                , barLow = lowPx
                                , barClose = closePx
                                , barVolume = volumePx
                                }
                     in if all isFiniteDouble [openPx, highPx, lowPx, volumePx] && highPx >= lowPx
                            then Just bar
                            else Nothing
  where
    closes = fiClose inputs
    closeOr fallback ix =
        if ix < 0 || ix >= V.length closes
            then fallback
            else
                let v = closes V.! ix
                 in if isFiniteDouble v then v else fallback

validVectorValue :: Maybe (V.Vector Double) -> Int -> Maybe Double
validVectorValue mv ix = do
    vec <- mv
    if ix < 0 || ix >= V.length vec
        then Nothing
        else
            let v = vec V.! ix
             in if isFiniteDouble v then Just v else Nothing

barsEndingAt :: FeatureInputs -> Int -> Int -> Maybe [Bar]
barsEndingAt inputs endIx k
    | k <= 0 = Nothing
    | endIx < 0 = Nothing
    | otherwise =
        let startIx = endIx - k + 1
         in if startIx < 0
                then Nothing
                else sequence [barAt inputs ix | ix <- [startIx .. endIx]]

trueRangeAt :: FeatureInputs -> Int -> Maybe Double
trueRangeAt inputs t = do
    bar <- barAt inputs t
    let closes = fiClose inputs
        prevCloseRaw =
            if t <= 0
                then barClose bar
                else closes V.! (t - 1)
        prevClose =
            if isFiniteDouble prevCloseRaw
                then prevCloseRaw
                else barClose bar
        tr =
            maximum
                [ barHigh bar - barLow bar
                , abs (barHigh bar - prevClose)
                , abs (barLow bar - prevClose)
                ]
    if isFiniteDouble tr
        then Just tr
        else Nothing

trueRangesEndingAt :: FeatureInputs -> Int -> Int -> Maybe [Double]
trueRangesEndingAt inputs endIx k
    | k <= 0 = Nothing
    | endIx < 0 = Nothing
    | otherwise =
        let startIx = endIx - k + 1
         in if startIx < 0
                then Nothing
                else sequence [trueRangeAt inputs ix | ix <- [startIx .. endIx]]

klineFeatures :: FeatureInputs -> Bar -> Int -> Double -> Double -> Double -> [Double] -> [Double] -> Int -> Int -> [Double]
klineFeatures inputs barNow t ret1 retShort retMid rsShort rsMid shortB midB =
    let eps = 1e-12
        currentClose = max eps (abs (barClose barNow))
        currentRange = max 0 (barHigh barNow - barLow barNow)
        upperWick = max 0 (barHigh barNow - max (barOpen barNow) (barClose barNow))
        lowerWick = max 0 (min (barOpen barNow) (barClose barNow) - barLow barNow)
        closeInRange =
            if currentRange <= eps
                then 0
                else clamp (-1) 1 (2 * ((barClose barNow - barLow barNow) / currentRange) - 1)
        bodyToRange =
            if currentRange <= eps
                then 0
                else abs (barClose barNow - barOpen barNow) / currentRange
        prevClose =
            if t <= 0
                then barClose barNow
                else fiClose inputs V.! (t - 1)
        gapRet = fromMaybe 0 (finiteReturn prevClose (barOpen barNow))
        barsShort = fromMaybeList [barNow] (barsEndingAt inputs t shortB)
        barsMid = fromMaybeList barsShort (barsEndingAt inputs t midB)
        barsShortPrev = fromMaybeList [] (barsEndingAt inputs (t - 1) shortB)
        barsMidPrev = fromMaybeList barsShortPrev (barsEndingAt inputs (t - 1) midB)
        trueShort = fromMaybeList [currentRange] (trueRangesEndingAt inputs t shortB)
        trueMid = fromMaybeList trueShort (trueRangesEndingAt inputs t midB)
        atrShort = meanList trueShort / currentClose
        atrMid = meanList trueMid / currentClose
        rangeRet = currentRange / currentClose
        rangeToAtr =
            if atrShort <= eps
                then 0
                else rangeRet / atrShort
        recentHigh =
            case map barHigh barsMidPrev of
                [] -> barClose barNow
                hs -> maximum hs
        recentLow =
            case map barLow barsMidPrev of
                [] -> barClose barNow
                ls -> minimum ls
        recentMaxClose = max currentClose (maximum (map barClose barsMid))
        recentMinClose = min (barClose barNow) (minimum (map barClose barsMid))
        breakoutHigh =
            if recentHigh <= eps
                then 0
                else barClose barNow / recentHigh - 1
        breakoutLow =
            if recentLow <= eps
                then 0
                else barClose barNow / recentLow - 1
        closeDrawdown =
            if recentMaxClose <= eps
                then 0
                else barClose barNow / recentMaxClose - 1
        closeRebound =
            if recentMinClose <= eps
                then 0
                else barClose barNow / recentMinClose - 1
        volsShort = map barVolume barsShort
        volsMid = map barVolume barsMid
        (volMuS, volSigS) = meanStd volsShort
        (volMuM, volSigM) = meanStd volsMid
        volZShort =
            if volSigS <= eps
                then 0
                else (barVolume barNow - volMuS) / volSigS
        volZMid =
            if volSigM <= eps
                then 0
                else (barVolume barNow - volMuM) / volSigM
        prevVolume =
            case barAt inputs (t - 1) of
                Just prevBar -> barVolume prevBar
                Nothing -> barVolume barNow
        volMomentum =
            if prevVolume <= eps
                then 0
                else barVolume barNow / prevVolume - 1
        priceVolumePressure = ret1 * volZShort
        efficiencyShort = efficiencyRatio retShort rsShort
        efficiencyMid = efficiencyRatio retMid rsMid
        feats =
            [ safeSignedRatio (barClose barNow - barOpen barNow) (abs (barOpen barNow))
            , rangeRet
            , upperWick / currentClose
            , lowerWick / currentClose
            , closeInRange
            , bodyToRange
            , gapRet
            , atrShort
            , atrMid
            , rangeToAtr
            , breakoutHigh
            , breakoutLow
            , closeDrawdown
            , closeRebound
            , volZShort
            , volZMid
            , volMomentum
            , priceVolumePressure
            , efficiencyShort
            , efficiencyMid
            ]
     in map sanitizeFinite feats

efficiencyRatio :: Double -> [Double] -> Double
efficiencyRatio netRet rs =
    let denom = sum (map abs rs)
     in if denom <= 1e-12
            then 0
            else abs netRet / denom

fromMaybeList :: [a] -> Maybe [a] -> [a]
fromMaybeList = fromMaybe

meanList :: [Double] -> Double
meanList xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

meanStd :: [Double] -> (Double, Double)
meanStd xs =
    case xs of
        [] -> (0, 0)
        _ ->
            let n = length xs
                mu = sum xs / fromIntegral n
                var =
                    if n < 2
                        then 0
                        else
                            let denom = fromIntegral (n - 1)
                             in sum (map (\v -> (v - mu) * (v - mu)) xs) / denom
             in (mu, sqrt (var + 1e-12))

safeSignedRatio :: Double -> Double -> Double
safeSignedRatio numer denom =
    if denom <= 1e-12
        then 0
        else sanitizeFinite (numer / denom)

sanitizeFinite :: Double -> Double
sanitizeFinite x =
    if isFiniteDouble x
        then x
        else 0

{- | Number of same-asset cross-exchange (Coinbase) features appended when a
Coinbase close series is attached. Constant so the feature dimension is stable
across every bar of a run.
-}
crossExchangeFeatureCount :: Int
crossExchangeFeatureCount = 5

zeroCrossExchangeFeatures :: [Double]
zeroCrossExchangeFeatures = replicate crossExchangeFeatureCount 0

{- | Same-asset cross-exchange features derived from the Coinbase close series
aligned to the Binance bar grid. Captures the Binance/Coinbase basis (premium),
its short-window dynamics, and the per-bar lead-lag return divergence. Every
component uses only data available by bar @t@ (no lookahead) and is sanitized to
a finite value; missing/degenerate inputs collapse to neutral zeros.
-}
crossExchangeFeatures :: FeatureInputs -> Int -> Double -> Int -> [Double]
crossExchangeFeatures inputs t ret1 shortB =
    case fiCoinbaseClose inputs of
        Nothing -> zeroCrossExchangeFeatures
        Just cb ->
            let bn = fiClose inputs
                eps = 1e-12
                basisAt i =
                    if i < 0 || i >= V.length cb || i >= V.length bn
                        then Nothing
                        else
                            let b = bn V.! i
                                c = cb V.! i
                             in if abs b <= eps || not (isFiniteDouble b) || not (isFiniteDouble c)
                                    then Nothing
                                    else Just ((c - b) / b)
                mBasisNow = basisAt t
                basisNow = fromMaybe 0 mBasisNow
                basisChange =
                    case (mBasisNow, basisAt (t - 1)) of
                        (Just a, Just b) -> a - b
                        _ -> 0
                window = [b | i <- [t - shortB + 1 .. t], Just b <- [basisAt i]]
                (muB, sigB) = meanStd window
                basisZ = if sigB <= eps then 0 else (basisNow - muB) / sigB
                cbRet1 =
                    if t >= 1 && t < V.length cb
                        then fromMaybe 0 (finiteReturn (cb V.! (t - 1)) (cb V.! t))
                        else 0
                leadLag = cbRet1 - ret1
                feats = [basisNow, basisChange, basisZ, cbRet1, leadLag]
             in map sanitizeFinite feats

marketFeatureCount :: Int
marketFeatureCount = 8

zeroMarketFeatures :: [Double]
zeroMarketFeatures = replicate marketFeatureCount 0

marketFeaturesFromModel :: Maybe MarketModel -> Int -> Double -> [Double] -> [Double]
marketFeaturesFromModel Nothing _ _ _ = zeroMarketFeatures
marketFeaturesFromModel (Just mm) t ret1 symShort =
    case vectorWindowEndingAt (mmLag mm) t (length symShort) of
        Nothing -> zeroMarketFeatures
        Just marketShort -> marketFeaturesFromWindow (mmIntercept mm) (mmBeta mm) ret1 symShort marketShort

vectorWindowEndingAt :: V.Vector Double -> Int -> Int -> Maybe [Double]
vectorWindowEndingAt vec endIx k
    | k <= 0 = Nothing
    | endIx < 0 = Nothing
    | otherwise =
        let startIx = endIx - k + 1
         in if startIx < 0 || endIx >= V.length vec
                then Nothing
                else
                    let xs = [vec V.! i | i <- [startIx .. endIx]]
                     in if all isFiniteDouble xs
                            then Just xs
                            else Nothing

marketFeaturesFromWindow :: Double -> Double -> Double -> [Double] -> [Double] -> [Double]
marketFeaturesFromWindow intercept beta ret1 symShort marketShort
    | length symShort /= length marketShort = zeroMarketFeatures
    | null symShort = zeroMarketFeatures
    | otherwise =
        let marketRet1 = last marketShort
            marketMu = intercept + beta * marketRet1
            residuals = zipWith (\sym marketRet -> sym - (intercept + beta * marketRet)) symShort marketShort
            (muS, sigS) = meanStd symShort
            (muM, sigM) = meanStd marketShort
            (muRes, sigRes) = meanStd residuals
            resid1 = ret1 - marketMu
            residZ =
                case residuals of
                    [] -> 0
                    _ ->
                        let eNow = last residuals
                         in if sigRes <= 1e-12
                                then 0
                                else (eNow - muRes) / sigRes
            (corrShort, betaShort) = correlationAndBeta symShort marketShort
            relMomentum = muS - muM
            relVolRatio =
                if sigM <= 1e-12
                    then 0
                    else sigS / sigM
            feats =
                [ marketRet1
                , marketMu
                , resid1
                , residZ
                , corrShort
                , betaShort
                , relMomentum
                , relVolRatio
                ]
         in if all isFiniteDouble feats
                then feats
                else zeroMarketFeatures

correlationAndBeta :: [Double] -> [Double] -> (Double, Double)
correlationAndBeta xs ys
    | null xs = (0, 0)
    | length xs /= length ys = (0, 0)
    | otherwise =
        let pairs = [(x, y) | (x, y) <- zip xs ys, isFiniteDouble x && isFiniteDouble y]
            n = length pairs
         in if n < 2
                then (0, 0)
                else
                    let meanX = sum (map fst pairs) / fromIntegral n
                        meanY = sum (map snd pairs) / fromIntegral n
                        (sxx, syy, sxy) =
                            foldl
                                ( \(ax, ay, axy) (x, y) ->
                                    let dx = x - meanX
                                        dy = y - meanY
                                     in (ax + dx * dx, ay + dy * dy, axy + dx * dy)
                                )
                                (0, 0, 0)
                                pairs
                        eps = 1e-12
                        corrRaw =
                            if sxx <= eps || syy <= eps
                                then 0
                                else sxy / sqrt (sxx * syy)
                        corr = clamp (-1) 1 corrRaw
                        betaShort =
                            if syy <= eps
                                then 0
                                else sxy / syy
                     in if all isFiniteDouble [corr, betaShort]
                            then (corr, betaShort)
                            else (0, 0)

psychologicalFeatures :: Double -> [Double]
psychologicalFeatures price
    | not (isFiniteDouble price) = replicate 12 0
    | price <= 0 = replicate 12 0
    | otherwise =
        let base = 10 ** fromIntegral (floor (logBase 10 price) :: Int)
            steps = [base, base / 2, base / 4, base / 10, base / 20, base / 100]
         in concatMap (roundLevelFeatures price) steps

roundLevelFeatures :: Double -> Double -> [Double]
roundLevelFeatures price step =
    let offset = roundOffset price step
        roundness = clamp 0 1 (1 - 2 * abs offset)
     in [offset, roundness]

roundOffset :: Double -> Double -> Double
roundOffset price step
    | step <= 0 = 0
    | otherwise =
        let level = step * fromIntegral (floor (price / step + 0.5) :: Int)
            off = (price - level) / step
         in clamp (-0.5) 0.5 off

clamp :: Double -> Double -> Double -> Double
clamp lo hi x =
    max lo (min hi x)

finiteReturn :: Double -> Double -> Maybe Double
finiteReturn p0 p1
    | not (isFiniteDouble p0 && isFiniteDouble p1) = Nothing
    | p0 == 0 = Nothing
    | otherwise =
        let r = p1 / p0 - 1
         in if isFiniteDouble r then Just r else Nothing

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)
