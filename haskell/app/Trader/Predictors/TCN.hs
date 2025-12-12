module Trader.Predictors.TCN
  ( TCNModel(..)
  , trainTCN
  , tcnFeaturesAt
  , predictTCN
  ) where

import Data.List (foldl')
import qualified Data.Vector as V

data TCNModel = TCNModel
  { tmDilations :: [Int]
  , tmKernelSize :: !Int
  , tmWeights :: [Double]      -- includes bias as last element
  , tmSigma :: !(Maybe Double)
  } deriving (Eq, Show)

tcnFeaturesAt :: [Int] -> Int -> V.Vector Double -> Int -> Maybe [Double]
tcnFeaturesAt dilations kernelSize prices t = do
  let maxLag = maximum (map (\d -> 1 + d * (kernelSize - 1)) dilations)
  if t < maxLag || t >= V.length prices
    then Nothing
    else do
      let retLag lag =
            let p0 = prices V.! (t - lag)
                p1 = prices V.! (t - lag + 1)
             in if p0 == 0 then 0 else (p1 / p0 - 1)
          feats =
            [ retLag (1 + d * k)
            | d <- dilations
            , k <- [0 .. kernelSize - 1]
            ]
      pure feats

predictTCN :: TCNModel -> V.Vector Double -> Int -> Maybe (Double, Maybe Double)
predictTCN m prices t = do
  feats <- tcnFeaturesAt (tmDilations m) (tmKernelSize m) prices t
  let x = feats ++ [1.0] -- bias
      y = dot (tmWeights m) x
  pure (y, tmSigma m)

trainTCN :: Int -> V.Vector Double -> [(Int, Double)] -> TCNModel
trainTCN lookbackBars prices trainTargets =
  let kernelSize = 3
      maxD = max 1 ((lookbackBars - 1) `div` (kernelSize - 1))
      dilations = takeWhile (<= maxD) (iterate (* 2) 1)
      lambda = 1e-3
      xsYs =
        [ (x ++ [1.0], y)
        | (t, y) <- trainTargets
        , Just x <- [tcnFeaturesAt dilations kernelSize prices t]
        ]
   in if null xsYs
        then TCNModel { tmDilations = dilations, tmKernelSize = kernelSize, tmWeights = replicate (kernelSize * length dilations + 1) 0, tmSigma = Nothing }
        else
          let xs = map fst xsYs
              ys = map snd xsYs
              w = ridgeFit lambda xs ys
              preds = map (dot w) xs
              residuals = zipWith (-) ys preds
              sigma = sqrt (mean (map (\e -> e * e) residuals) + 1e-12)
           in TCNModel { tmDilations = dilations, tmKernelSize = kernelSize, tmWeights = w, tmSigma = Just sigma }

ridgeFit :: Double -> [[Double]] -> [Double] -> [Double]
ridgeFit lambda xs ys =
  let d = length (head xs)
      xtx = foldl' (matAdd) (zeroMat d d) (map (\x -> outer x x) xs)
      xty = foldl' (zipWith (+)) (replicate d 0) (zipWith (\x y -> map (* y) x) xs ys)
      xtxReg = addDiag lambda xtx
   in solveLinear xtxReg xty

-- Linear algebra (small, dense)

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

zeroMat :: Int -> Int -> [[Double]]
zeroMat r c = replicate r (replicate c 0)

matAdd :: [[Double]] -> [[Double]] -> [[Double]]
matAdd = zipWith (zipWith (+))

outer :: [Double] -> [Double] -> [[Double]]
outer x y = [ map (* xi) y | xi <- x ]

addDiag :: Double -> [[Double]] -> [[Double]]
addDiag lambda m =
  [ [ if i == j then m !! i !! j + lambda else m !! i !! j | j <- [0 .. n - 1] ]
  | i <- [0 .. n - 1]
  ]
  where
    n = length m

solveLinear :: [[Double]] -> [Double] -> [Double]
solveLinear a b =
  let n = length a
      aug0 = zipWith (\row bi -> row ++ [bi]) a b
      aug = forwardElimination n aug0
   in backSubstitution n aug

forwardElimination :: Int -> [[Double]] -> [[Double]]
forwardElimination n = go 0
  where
    eps = 1e-12

    go k m
      | k >= n = m
      | otherwise =
          let pivotRow = argMaxAbs (\row -> abs (row !! k)) [k .. n - 1] m
              m1 = swapRows k pivotRow m
              pivot = (m1 !! k) !! k
           in if abs pivot < eps
                then error "Singular matrix in solveLinear"
                else
                  let m2 =
                        [ if i <= k
                            then m1 !! i
                            else
                              let row = m1 !! i
                                  factor = (row !! k) / pivot
                                  rowK = m1 !! k
                               in [ if j < k then row !! j else row !! j - factor * rowK !! j | j <- [0 .. n] ]
                        | i <- [0 .. n - 1]
                        ]
                   in go (k + 1) m2

backSubstitution :: Int -> [[Double]] -> [Double]
backSubstitution n m =
  let rhs i = (m !! i) !! n
      coeff i j = (m !! i) !! j
      go i acc =
        if i < 0
          then acc
          else
            let s = sum [ coeff i j * (acc !! (j - i - 1)) | j <- [i + 1 .. n - 1] ]
                x = (rhs i - s) / coeff i i
             in go (i - 1) (x : acc)
   in go (n - 1) []

argMaxAbs :: (a -> Double) -> [Int] -> [a] -> Int
argMaxAbs f is xs =
  case is of
    [] -> error "argMaxAbs: empty"
    (i0:rest) ->
      let v0 = f (xs !! i0)
       in fst $ foldl' (\(ib, vb) i -> let v = f (xs !! i) in if v > vb then (i, v) else (ib, vb)) (i0, v0) rest

swapRows :: Int -> Int -> [[a]] -> [[a]]
swapRows i j rows
  | i == j = rows
  | otherwise =
      [ rowAt k | k <- [0 .. length rows - 1] ]
  where
    rowAt k
      | k == i = rows !! j
      | k == j = rows !! i
      | otherwise = rows !! k
