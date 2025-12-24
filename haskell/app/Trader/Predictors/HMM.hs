module Trader.Predictors.HMM
  ( HMM3(..)
  , HMMFilter(..)
  , fitHMM3
  , filterPosterior
  , predictNextFromPosterior
  , updatePosterior
  ) where

import Data.List (foldl')
import qualified Data.Vector as V

import Trader.Predictors.Types (RegimeProbs(..))

data HMM3 = HMM3
  { hmmPi :: [Double]      -- length 3
  , hmmA :: [[Double]]     -- 3x3 row-stochastic
  , hmmMu :: [Double]      -- length 3
  , hmmVar :: [Double]     -- length 3 (variance)
  , hmmTrendIx :: !Int
  , hmmMrIx :: !Int
  , hmmHighVolIx :: !Int
  } deriving (Eq, Show)

data HMMFilter = HMMFilter
  { hfPosterior :: [Double] -- posterior over last observed state
  } deriving (Eq, Show)

fitHMM3 :: Int -> [Double] -> HMM3
fitHMM3 iters obs
  | iters < 0 = error "iters must be >= 0"
  | null obs = defaultHMM
  | otherwise =
      let mu0 = mean obs
          s0 = std obs
          pi0 = replicate 3 (1 / 3)
          a0 =
            [ [0.90, 0.05, 0.05]
            , [0.05, 0.90, 0.05]
            , [0.05, 0.05, 0.90]
            ]
          mus0 = [mu0 - s0, mu0, mu0 + s0]
          vars0 = [s0 * s0, s0 * s0, max 1e-8 (4 * s0 * s0)]
          base0 = HMM3 { hmmPi = pi0, hmmA = a0, hmmMu = mus0, hmmVar = vars0, hmmTrendIx = 0, hmmMrIx = 1, hmmHighVolIx = 2 }
          fitted = applyN iters (emStep obs) base0
       in remapRegimes fitted

defaultHMM :: HMM3
defaultHMM =
  let pi0 = replicate 3 (1 / 3)
      a0 = [ [0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90] ]
      mus0 = [0, 0, 0]
      vars0 = [1e-4, 1e-4, 1e-3]
   in HMM3 { hmmPi = pi0, hmmA = a0, hmmMu = mus0, hmmVar = vars0, hmmTrendIx = 0, hmmMrIx = 1, hmmHighVolIx = 2 }

remapRegimes :: HMM3 -> HMM3
remapRegimes hmm =
  let vars = hmmVar hmm
      mus = hmmMu hmm
      highVol = argmax vars
      remaining = filter (/= highVol) [0, 1, 2]
      trend =
        case remaining of
          [i, j] -> if abs (mus !! i) >= abs (mus !! j) then i else j
          _ -> 0
      mr = head (filter (\k -> k /= highVol && k /= trend) [0, 1, 2])
   in hmm { hmmTrendIx = trend, hmmMrIx = mr, hmmHighVolIx = highVol }

-- | Posterior after filtering through a sequence of observations.
filterPosterior :: HMM3 -> [Double] -> HMMFilter
filterPosterior hmm obs =
  case obs of
    [] -> HMMFilter { hfPosterior = hmmPi hmm }
    (o0:os) ->
      let alpha0Un = zipWith (*) (hmmPi hmm) (emissions hmm o0)
          c0 = sum alpha0Un
          alpha0 = if c0 == 0 then hmmPi hmm else map (/ c0) alpha0Un
          go alphaPrev [] = alphaPrev
          go alphaPrev (o:rest) =
            let alphaPred = vecMat alphaPrev (hmmA hmm)
                alphaUn = zipWith (*) alphaPred (emissions hmm o)
                ct = sum alphaUn
                alpha = if ct == 0 then alphaPred else map (/ ct) alphaUn
             in go alpha rest
       in HMMFilter { hfPosterior = go alpha0 os }

-- | Predict regime probabilities and return distribution for the next step given
-- posterior over the last observed state.
predictNextFromPosterior :: HMM3 -> HMMFilter -> (RegimeProbs, Double, Double, [Double])
predictNextFromPosterior hmm filt =
  let predState = vecMat (hfPosterior filt) (hmmA hmm)
      pTrend = predState !! hmmTrendIx hmm
      pMr = predState !! hmmMrIx hmm
      pHv = predState !! hmmHighVolIx hmm
      mu = sum (zipWith (*) predState (hmmMu hmm))
      var = sum (zipWith3 (\w m v -> w * (v + m * m)) predState (hmmMu hmm) (hmmVar hmm)) - mu * mu
      sigma = sqrt (max 1e-12 var)
   in (RegimeProbs pTrend pMr pHv, mu, sigma, predState)

-- | Update posterior for the current state given predicted state distribution and
-- an observed return.
updatePosterior :: HMM3 -> [Double] -> Double -> HMMFilter
updatePosterior hmm predState obs =
  let like = emissions hmm obs
      un = zipWith (*) predState like
      z = sum un
      post = if z == 0 then predState else map (/ z) un
   in HMMFilter { hfPosterior = post }

-- EM training (Baum-Welch) with scaling

emStep :: [Double] -> HMM3 -> HMM3
emStep obs hmm =
  let (alphas, cs) = forwardScaled hmm obs
      betas = backwardScaled hmm obs cs
      gammas = zipWith (\a b -> normalize (zipWith (*) a b)) alphas betas
      xis = xiList hmm obs alphas betas cs

      pi' = head gammas
      a' = updateA gammas xis
      (mus', vars') = updateEmissions obs gammas
   in hmm { hmmPi = pi', hmmA = a', hmmMu = mus', hmmVar = vars' }

forwardScaled :: HMM3 -> [Double] -> ([[Double]], [Double])
forwardScaled hmm obs =
  case obs of
    [] -> ([], [])
    (o0:os) ->
      let alpha0Un = zipWith (*) (hmmPi hmm) (emissions hmm o0)
          c0 = max 1e-300 (sum alpha0Un)
          alpha0 = map (/ c0) alpha0Un
          go accA accC _ [] = (reverse accA, reverse accC)
          go accA accC aPrev (o:rest) =
            let aPred = vecMat aPrev (hmmA hmm)
                aUn = zipWith (*) aPred (emissions hmm o)
                ct = max 1e-300 (sum aUn)
                aNow = map (/ ct) aUn
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
                    betaUn =
                      [ sum [ (hmmA hmm !! i !! j) * (likeNext !! j) * (betaNext !! j) | j <- [0..2] ]
                      | i <- [0..2]
                      ]
                    beta = map (/ (csV V.! (t + 1))) betaUn
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
            denom = max 1e-300 (csV V.! (t + 1))
            un =
              [ [ aT !! i * (hmmA hmm !! i !! j) * (likeNext !! j) * (bNext !! j) / denom
                | j <- [0..2]
                ]
              | i <- [0..2]
              ]
            z = sum (map sum un)
         in if z == 0 then un else map (map (/ z)) un
   in [ xiAt t | t <- [0 .. tMax - 2] ]

updateA :: [[Double]] -> [[[Double]]] -> [[Double]]
updateA gammas xis =
  let gammasV = V.fromList gammas
      xisV = V.fromList xis
      tMax = V.length gammasV
      denom i = sum [ (gammasV V.! t) !! i | t <- [0 .. tMax - 2] ] + 1e-12
      num i j = sum [ (xisV V.! t) !! i !! j | t <- [0 .. V.length xisV - 1] ]
      row i =
        let r = [ num i j / denom i | j <- [0..2] ]
         in normalize r
   in [ row i | i <- [0..2] ]

updateEmissions :: [Double] -> [[Double]] -> ([Double], [Double])
updateEmissions obs gammas =
  let obsV = V.fromList obs
      gammasV = V.fromList gammas
      tMax = V.length gammasV
      denom k = sum [ (gammasV V.! t) !! k | t <- [0 .. tMax - 1] ] + 1e-12
      mu k = sum [ (gammasV V.! t) !! k * (obsV V.! t) | t <- [0 .. tMax - 1] ] / denom k
      mus = [ mu k | k <- [0..2] ]
      var k =
        let mk = mus !! k
         in sum [ (gammasV V.! t) !! k * ((obsV V.! t) - mk) ^ (2 :: Int) | t <- [0 .. tMax - 1] ] / denom k + 1e-8
      vars = [ var k | k <- [0..2] ]
   in (mus, vars)

emissions :: HMM3 -> Double -> [Double]
emissions hmm x =
  [ normalPdf x (hmmMu hmm !! k) (hmmVar hmm !! k) | k <- [0..2] ]

normalPdf :: Double -> Double -> Double -> Double
normalPdf x mu var =
  let v = max 1e-12 var
      c = 1 / sqrt (2 * pi * v)
      z = (x - mu)
   in c * exp (-(z * z) / (2 * v))

vecMat :: [Double] -> [[Double]] -> [Double]
vecMat v m =
  [ sum (zipWith (*) v (col j m)) | j <- [0 .. length v - 1] ]

col :: Int -> [[a]] -> [a]
col j m = [ row !! j | row <- m ]

normalize :: [Double] -> [Double]
normalize xs =
  let s = sum xs
   in if s == 0 then replicate (length xs) (1 / fromIntegral (length xs)) else map (/ s) xs

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

std :: [Double] -> Double
std xs =
  let mu = mean xs
      var = sum (map (\v -> (v - mu) * (v - mu)) xs) / fromIntegral (length xs)
   in sqrt (var + 1e-12)

argmax :: [Double] -> Int
argmax xs =
  case xs of
    [] -> 0
    _ ->
      fst $
        foldl'
          (\(bi, bv) (i, v) -> if v > bv then (i, v) else (bi, bv))
          (0, head xs)
          (zip [0..] xs)

applyN :: Int -> (a -> a) -> a -> a
applyN n f x0 = go n x0
  where
    go k x
      | k <= 0 = x
      | otherwise =
          let x' = f x
           in x' `seq` go (k - 1) x'
