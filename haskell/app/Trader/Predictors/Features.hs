module Trader.Predictors.Features (
    FeatureSpec (..),
    mkFeatureSpec,
    featuresAt,
    featuresAtWithMarket,
    forwardReturnAt,
    buildDatasetWithIndex,
    buildDatasetWithIndexWithMarket,
    buildDataset,
    buildDatasetWithMarket,
) where

import qualified Data.Vector as V

import Trader.MarketContext (MarketModel (..))

data FeatureSpec = FeatureSpec
    { fsLookbackBars :: !Int
    , fsShortBars :: !Int
    , fsMidBars :: !Int
    }
    deriving (Eq, Show)

mkFeatureSpec :: Int -> FeatureSpec
mkFeatureSpec lookbackBars =
    let lb = max 2 lookbackBars
        shortB = max 1 (min 12 (lb - 1))
        midB = max 1 (min 48 (lb - 1))
     in FeatureSpec{fsLookbackBars = lb, fsShortBars = shortB, fsMidBars = midB}

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
featuresAt fs = featuresAtWithMarket fs Nothing

{- | Market-aware feature vector at bar t using only data available by bar t.
Cross-symbol features fall back to zero when market context is missing/invalid.
-}
featuresAtWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> Int -> Maybe [Double]
featuresAtWithMarket fs mMarket prices t = do
    let lb = fsLookbackBars fs
        maxLag = max 1 (lb - 1)
        shortB = min (fsShortBars fs) maxLag
        midB = min (fsMidBars fs) maxLag
        ret3Bars = min 3 maxLag
    if t < lb - 1 || t >= V.length prices
        then Nothing
        else do
            ret1 <- retOver prices t 1
            ret3 <- retOver prices t ret3Bars
            retShort <- retOver prices t shortB
            retMid <- retOver prices t midB
            retLb <- retOver prices t (lb - 1)
            rsShort <- returnsEndingAt prices t shortB
            rsMid <- returnsEndingAt prices t midB
            let (muS, sigS) = meanStd rsShort
                (muM, sigM) = meanStd rsMid
                priceT = prices V.! t
                psych = psychologicalFeatures priceT
                marketFeats = marketFeaturesFromModel mMarket t ret1 rsShort
                eps = 1e-12
                retSpread = retShort - retMid
                retMeanReversion = ret1 - muS
                volRatio = if abs sigM <= eps then 0 else sigS / sigM
                trendSlope = muS - muM
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
                        ++ marketFeats
                        ++ psych
            if all isFiniteDouble feats
                then pure feats
                else Nothing

{- | Build a supervised dataset (features at t, target forward return at t) with bar indices.
Uses t in [lookbackBars-1 .. n-2].
-}
buildDatasetWithIndex :: FeatureSpec -> V.Vector Double -> [(Int, [Double], Double)]
buildDatasetWithIndex fs = buildDatasetWithIndexWithMarket fs Nothing

{- | Market-aware dataset builder; cross-symbol features fall back to zero when
market context is missing/invalid.
-}
buildDatasetWithIndexWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> [(Int, [Double], Double)]
buildDatasetWithIndexWithMarket fs mMarket prices =
    let n = V.length prices
        startT = fsLookbackBars fs - 1
        endT = n - 2
        maxLag = max 1 (fsLookbackBars fs - 1)
        shortB = min (fsShortBars fs) maxLag
        midB = min (fsMidBars fs) maxLag
        ret3Bars = min 3 maxLag
        retLen = max 0 (n - 1)
        returns =
            V.generate retLen $ \i ->
                let p0 = prices V.! i
                    p1 = prices V.! (i + 1)
                 in finiteReturn p0 p1
        retRows = V.map retRow returns
        retRow mRet =
            case mRet of
                Just r
                    | isFiniteDouble r ->
                        let r2 = r * r
                         in if isFiniteDouble r2
                                then (r, r2, 0 :: Int)
                                else (0, 0, 1)
                _ -> (0, 0, 1)
        retVals = V.map (\(r, _, _) -> r) retRows
        retSqVals = V.map (\(_, r2, _) -> r2) retRows
        retInvalid = V.map (\(_, _, bad) -> bad) retRows
        prefixSum = V.scanl' (+) 0 retVals
        prefixSumSq = V.scanl' (+) 0 retSqVals
        prefixInvalid = V.scanl' (+) 0 retInvalid

        windowStats t k =
            if k <= 0 || t - k < 0 || t - 1 >= retLen
                then Nothing
                else
                    let i0 = t - k
                        i1 = t - 1
                        invalid = prefixInvalid V.! (i1 + 1) - prefixInvalid V.! i0
                     in if invalid > 0
                            then Nothing
                            else
                                let s = prefixSum V.! (i1 + 1) - prefixSum V.! i0
                                    ss = prefixSumSq V.! (i1 + 1) - prefixSumSq V.! i0
                                    k' = fromIntegral k
                                    mu = s / k'
                                    varRaw =
                                        if k < 2
                                            then 0
                                            else (ss - k' * mu * mu) / fromIntegral (k - 1)
                                    var = max 0 varRaw
                                 in if all isFiniteDouble [s, ss, mu, var]
                                        then Just (mu, sqrt (var + 1e-12))
                                        else Nothing

        windowReturns t k =
            if k <= 0 || t - k < 0 || t - 1 >= retLen
                then Nothing
                else
                    let i0 = t - k
                        i1 = t - 1
                        invalid = prefixInvalid V.! (i1 + 1) - prefixInvalid V.! i0
                     in if invalid > 0
                            then Nothing
                            else Just [retVals V.! i | i <- [i0 .. i1]]

        retOverFast t bars =
            if bars <= 0 || t - bars < 0
                then Nothing
                else
                    let p0 = prices V.! (t - bars)
                        p1 = prices V.! t
                     in finiteReturn p0 p1

        featuresAtFast t = do
            if t < fsLookbackBars fs - 1 || t >= n
                then Nothing
                else do
                    ret1 <- retOverFast t 1
                    ret3 <- retOverFast t ret3Bars
                    retShort <- retOverFast t shortB
                    retMid <- retOverFast t midB
                    retLb <- retOverFast t (fsLookbackBars fs - 1)
                    (muS, sigS) <- windowStats t shortB
                    (muM, sigM) <- windowStats t midB
                    rsShort <- windowReturns t shortB
                    let priceT = prices V.! t
                        psych = psychologicalFeatures priceT
                        marketFeats = marketFeaturesFromModel mMarket t ret1 rsShort
                        eps = 1e-12
                        retSpread = retShort - retMid
                        retMeanReversion = ret1 - muS
                        volRatio = if abs sigM <= eps then 0 else sigS / sigM
                        trendSlope = muS - muM
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
                                ++ marketFeats
                                ++ psych
                    if all isFiniteDouble feats
                        then pure feats
                        else Nothing

        forwardReturnFast t =
            if t < 0 || t >= retLen
                then Nothing
                else returns V.! t
     in if startT > endT
            then []
            else
                [ (t, f, y)
                | t <- [startT .. endT]
                , Just f <- [featuresAtFast t]
                , all isFiniteDouble f
                , Just y <- [forwardReturnFast t]
                , isFiniteDouble y
                ]

{- | Build a supervised dataset (features at t, target forward return at t).
Uses t in [lookbackBars-1 .. n-2].
-}
buildDataset :: FeatureSpec -> V.Vector Double -> [([Double], Double)]
buildDataset fs = buildDatasetWithMarket fs Nothing

buildDatasetWithMarket :: FeatureSpec -> Maybe MarketModel -> V.Vector Double -> [([Double], Double)]
buildDatasetWithMarket fs mMarket prices =
    [(f, y) | (_, f, y) <- buildDatasetWithIndexWithMarket fs mMarket prices]

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
