module Trader.Predictors.HMM (
    HMM3 (..),
    HMMFilter (..),
    defaultHmmIterations,
    fitHMM3,
    filterPosterior,
    predictNextFromPosterior,
    updatePosterior,
) where

import Data.List (foldl')
import qualified Data.Vector as V

import Trader.Predictors.Types (RegimeProbs (..))

data HMM3 = HMM3
    { hmmPi :: [Double] -- length 3
    , hmmA :: [[Double]] -- 3x3 row-stochastic
    , hmmMu :: [Double] -- length 3
    , hmmVar :: [Double] -- length 3 (variance)
    , hmmTrendIx :: !Int
    , hmmMrIx :: !Int
    , hmmHighVolIx :: !Int
    }
    deriving (Eq, Show)

newtype HMMFilter = HMMFilter
    { hfPosterior :: [Double] -- posterior over last observed state
    }
    deriving (Eq, Show)

defaultHmmIterations :: Int
defaultHmmIterations = 10

fitHMM3 :: Int -> [Double] -> HMM3
fitHMM3 iters obs
    | null obs' = defaultHMM
    | otherwise =
        let iters' = max 0 iters
            mu0 = mean obs'
            s0 = std obs'
            pi0 = replicate 3 (1 / 3)
            a0 =
                [ [0.90, 0.05, 0.05]
                , [0.05, 0.90, 0.05]
                , [0.05, 0.05, 0.90]
                ]
            mus0 = [mu0 - s0, mu0, mu0 + s0]
            vars0 = [s0 * s0, s0 * s0, max 1e-8 (4 * s0 * s0)]
            base0 = HMM3{hmmPi = pi0, hmmA = a0, hmmMu = mus0, hmmVar = vars0, hmmTrendIx = 0, hmmMrIx = 1, hmmHighVolIx = 2}
            fitted = applyN iters' (emStep obs') base0
         in normalizeHMM3 fitted
  where
    obs' = filter isFiniteDouble obs

defaultHMM :: HMM3
defaultHMM =
    let pi0 = replicate 3 (1 / 3)
        a0 = [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]
        mus0 = [0, 0, 0]
        vars0 = [1e-4, 1e-4, 1e-3]
     in HMM3{hmmPi = pi0, hmmA = a0, hmmMu = mus0, hmmVar = vars0, hmmTrendIx = 0, hmmMrIx = 1, hmmHighVolIx = 2}

normalizeHMM3 :: HMM3 -> HMM3
normalizeHMM3 hmm =
    let pad3 def xs = take 3 (xs ++ repeat def)
        pi0 = normalize (pad3 (1 / 3) (hmmPi hmm))
        mu0 = map (finiteOr 0) (pad3 0 (hmmMu hmm))
        var0 = map finiteVariance (pad3 1e-4 (hmmVar hmm))
        rowDef = [0.90, 0.05, 0.05]
        rows =
            take
                3
                (map (normalize . pad3 (1 / 3)) (hmmA hmm) ++ repeat rowDef)
        hmm' = hmm{hmmPi = pi0, hmmA = rows, hmmMu = mu0, hmmVar = var0}
     in remapRegimes hmm'

remapRegimes :: HMM3 -> HMM3
remapRegimes hmm =
    let vars = hmmVar hmm
        mus = hmmMu hmm
        highVol = argmax vars
        remaining = filter (/= highVol) [0, 1, 2]
        trend =
            case remaining of
                [i, j] -> if abs (atDef 0 mus i) >= abs (atDef 0 mus j) then i else j
                _ -> 0
        mr =
            case filter (\k -> k /= highVol && k /= trend) [0, 1, 2] of
                (k : _) -> k
                [] -> highVol
     in hmm{hmmTrendIx = trend, hmmMrIx = mr, hmmHighVolIx = highVol}

-- | Posterior after filtering through a sequence of observations.
filterPosterior :: HMM3 -> [Double] -> HMMFilter
filterPosterior hmm obs =
    let hmm' = normalizeHMM3 hmm
        obs' = filter isFiniteDouble obs
     in case obs' of
            [] -> HMMFilter{hfPosterior = hmmPi hmm'}
            (o0 : os) ->
                let alpha0Un = zipWith (*) (hmmPi hmm') (emissions hmm' o0)
                    alpha0 = normalizeOr (hmmPi hmm') alpha0Un
                    go alphaPrev [] = alphaPrev
                    go alphaPrev (o : rest) =
                        let alphaPred = vecMat alphaPrev (hmmA hmm')
                            alphaUn = zipWith (*) alphaPred (emissions hmm' o)
                            alpha = normalizeOr alphaPred alphaUn
                         in go alpha rest
                 in HMMFilter{hfPosterior = go alpha0 os}

{- | Predict regime probabilities and return distribution for the next step given
posterior over the last observed state.
-}
predictNextFromPosterior :: HMM3 -> HMMFilter -> (RegimeProbs, Double, Double, [Double])
predictNextFromPosterior hmm filt =
    let hmm' = normalizeHMM3 hmm
        post = normalizePosterior (hfPosterior filt)
        predState = normalize (vecMat post (hmmA hmm'))
        pTrend = atDef 0 predState (hmmTrendIx hmm')
        pMr = atDef 0 predState (hmmMrIx hmm')
        pHv = atDef 0 predState (hmmHighVolIx hmm')
        muRaw = sum (zipWith (*) predState (hmmMu hmm'))
        mu = finiteOr 0 muRaw
        varRaw = sum (zipWith3 (\w m v -> w * (v + m * m)) predState (hmmMu hmm') (hmmVar hmm')) - mu * mu
        sigmaRaw = sqrt (max 1e-12 varRaw)
        sigma = finiteOr 1e-6 sigmaRaw
     in (RegimeProbs pTrend pMr pHv, mu, sigma, predState)

{- | Update posterior for the current state given predicted state distribution and
an observed return.
-}
updatePosterior :: HMM3 -> [Double] -> Double -> HMMFilter
updatePosterior hmm predState obs =
    let hmm' = normalizeHMM3 hmm
        predState' = normalizePosterior predState
     in if not (isFiniteDouble obs)
            then HMMFilter{hfPosterior = predState'}
            else
                let like = emissions hmm' obs
                    un = zipWith (*) predState' like
                    post = normalizeOr predState' un
                 in HMMFilter{hfPosterior = post}

-- EM training (Baum-Welch) with scaling

emStep :: [Double] -> HMM3 -> HMM3
emStep obs hmm =
    let (alphas, cs) = forwardScaled hmm obs
        betas = backwardScaled hmm obs cs
        gammas = zipWith (\a b -> normalize (zipWith (*) a b)) alphas betas
        xis = xiList hmm obs alphas betas cs

        pi' =
            case gammas of
                g0 : _ -> g0
                [] -> hmmPi hmm
        a' = updateA gammas xis
        (mus', vars') = updateEmissions obs gammas
     in hmm{hmmPi = pi', hmmA = a', hmmMu = mus', hmmVar = vars'}

forwardScaled :: HMM3 -> [Double] -> ([[Double]], [Double])
forwardScaled hmm obs =
    case obs of
        [] -> ([], [])
        (o0 : os) ->
            let alpha0Un = map finiteNonNegative (zipWith (*) (hmmPi hmm) (emissions hmm o0))
                c0 = scaleDenom (sum alpha0Un)
                alpha0 = normalize alpha0Un
                go accA accC _ [] = (reverse accA, reverse accC)
                go accA accC aPrev (o : rest) =
                    let aPred = vecMat aPrev (hmmA hmm)
                        aUn = map finiteNonNegative (zipWith (*) aPred (emissions hmm o))
                        ct = scaleDenom (sum aUn)
                        aNow = normalize aUn
                     in go (aNow : accA) (ct : accC) aNow rest
             in go [alpha0] [c0] alpha0 os

backwardScaled :: HMM3 -> [Double] -> [Double] -> [[Double]]
backwardScaled hmm obs cs =
    case obs of
        [] -> []
        _ ->
            let obsV = V.fromList obs
                csV = V.fromList cs
                tMax = V.length obsV
                betaT = replicate 3 1
                go t acc betaNext
                    | t < 0 = acc
                    | otherwise =
                        let oNext = obsV V.! (t + 1)
                            likeNext = emissions hmm oNext
                            denom = scaleDenom (csV V.! (t + 1))
                            betaUn =
                                [ sum [atDef2 0 (hmmA hmm) i j * atDef 0 likeNext j * atDef 0 betaNext j | j <- [0 .. 2]]
                                | i <- [0 .. 2]
                                ]
                            beta = map (finiteOr 0 . (/ denom)) betaUn
                         in go (t - 1) (beta : acc) beta
             in go (tMax - 2) [betaT] betaT

xiList :: HMM3 -> [Double] -> [[Double]] -> [[Double]] -> [Double] -> [[[Double]]]
xiList hmm obs alphas betas cs =
    let obsV = V.fromList obs
        alphasV = V.fromList alphas
        betasV = V.fromList betas
        csV = V.fromList cs
        tMax = V.length obsV
        xiAt t =
            let aT = alphasV V.! t
                bNext = betasV V.! (t + 1)
                oNext = obsV V.! (t + 1)
                likeNext = emissions hmm oNext
                denom = scaleDenom (csV V.! (t + 1))
                unRaw =
                    [ [ atDef 0 aT i * atDef2 0 (hmmA hmm) i j * atDef 0 likeNext j * atDef 0 bNext j / denom
                      | j <- [0 .. 2]
                      ]
                    | i <- [0 .. 2]
                    ]
                un = map (map finiteNonNegative) unRaw
                z = sum (map sum un)
             in if isFiniteDouble z && z > 0
                    then map (map (/ z)) un
                    else replicate 3 (replicate 3 (1 / 9))
     in [xiAt t | t <- [0 .. tMax - 2]]

updateA :: [[Double]] -> [[[Double]]] -> [[Double]]
updateA gammas xis =
    let gammasV = V.fromList gammas
        xisV = V.fromList xis
        tMax = V.length gammasV
        denom i = sum [atDef 0 (gammasV V.! t) i | t <- [0 .. tMax - 2]] + 1e-12
        num i j = sum [atDef2 0 (xisV V.! t) i j | t <- [0 .. V.length xisV - 1]]
        row i =
            let r = [num i j / denom i | j <- [0 .. 2]]
             in normalize r
     in [row i | i <- [0 .. 2]]

updateEmissions :: [Double] -> [[Double]] -> ([Double], [Double])
updateEmissions obs gammas =
    let obsV = V.fromList obs
        gammasV = V.fromList gammas
        tMax = V.length gammasV
        denom k = sum [atDef 0 (gammasV V.! t) k | t <- [0 .. tMax - 1]] + 1e-12
        mu k = finiteOr 0 (sum [atDef 0 (gammasV V.! t) k * (obsV V.! t) | t <- [0 .. tMax - 1]] / denom k)
        mus = [mu k | k <- [0 .. 2]]
        var k =
            let mk = atDef 0 mus k
                raw = sum [atDef 0 (gammasV V.! t) k * ((obsV V.! t) - mk) ^ (2 :: Int) | t <- [0 .. tMax - 1]] / denom k + 1e-8
             in finiteVariance raw
        vars = [var k | k <- [0 .. 2]]
     in (mus, vars)

emissions :: HMM3 -> Double -> [Double]
emissions hmm x =
    [normalPdf x (atDef 0 (hmmMu hmm) k) (atDef 1e-8 (hmmVar hmm) k) | k <- [0 .. 2]]

normalPdf :: Double -> Double -> Double -> Double
normalPdf x mu var =
    if not (all isFiniteDouble [x, mu, var])
        then 0
        else
            let v = max 1e-12 var
                c = 1 / sqrt (2 * pi * v)
                z = x - mu
                y = c * exp (-((z * z) / (2 * v)))
             in if isFiniteDouble y then y else 0

vecMat :: [Double] -> [[Double]] -> [Double]
vecMat v m =
    [sum (zipWith (*) v (col j m)) | j <- [0 .. length v - 1]]

col :: Int -> [[Double]] -> [Double]
col j m = [atDef 0 row j | row <- m]

normalize :: [Double] -> [Double]
normalize xs =
    let ys = map finiteNonNegative xs
        n = length ys
        s = sum ys
     in if n <= 0
            then []
            else
                if not (isFiniteDouble s) || s <= 0
                    then replicate n (1 / fromIntegral n)
                    else map (/ s) ys

normalizeOr :: [Double] -> [Double] -> [Double]
normalizeOr fallback xs =
    let ys = map finiteNonNegative xs
        s = sum ys
     in if isFiniteDouble s && s > 0
            then normalize ys
            else normalize fallback

normalizePosterior :: [Double] -> [Double]
normalizePosterior xs =
    let xs' = take 3 (xs ++ repeat 0)
     in normalize xs'

mean :: [Double] -> Double
mean xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

std :: [Double] -> Double
std xs =
    case xs of
        [] -> 0
        _ ->
            let mu = mean xs
                var = sum (map (\v -> (v - mu) * (v - mu)) xs) / fromIntegral (length xs)
             in sqrt (var + 1e-12)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

finiteOr :: Double -> Double -> Double
finiteOr fallback x
    | isFiniteDouble x = x
    | otherwise = fallback

finiteNonNegative :: Double -> Double
finiteNonNegative x
    | isFiniteDouble x && x >= 0 = x
    | otherwise = 0

finiteVariance :: Double -> Double
finiteVariance x
    | isFiniteDouble x && x > 0 = max 1e-8 x
    | otherwise = 1e-4

scaleDenom :: Double -> Double
scaleDenom x
    | isFiniteDouble x && x > 1e-300 = x
    | otherwise = 1e-300

argmax :: [Double] -> Int
argmax xs =
    case xs of
        [] -> 0
        (x0 : rest) ->
            fst $
                foldl'
                    (\(bi, bv) (i, v) -> if v > bv then (i, v) else (bi, bv))
                    (0, x0)
                    (zip [1 ..] rest)

applyN :: Int -> (a -> a) -> a -> a
applyN n f = go n
  where
    go k x
        | k <= 0 = x
        | otherwise =
            let x' = f x
             in x' `seq` go (k - 1) x'

atDef :: a -> [a] -> Int -> a
atDef fallback xs i
    | i < 0 = fallback
    | otherwise =
        case drop i xs of
            y : _ -> y
            [] -> fallback

atDef2 :: a -> [[a]] -> Int -> Int -> a
atDef2 fallback rows i = atDef fallback (atDef [] rows i)
