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
tcnFeaturesAt dilations kernelSize prices t =
  if null dilations || kernelSize <= 0
    then Nothing
    else
      let maxLag = maximum (map (\d -> 1 + d * (kernelSize - 1)) dilations)
       in if t < maxLag || t >= V.length prices
            then Nothing
            else
              let retLag lag =
                    let p0 = prices V.! (t - lag)
                        p1 = prices V.! (t - lag + 1)
                     in if p0 == 0 then 0 else (p1 / p0 - 1)
                  feats =
                    [ retLag (1 + d * k)
                    | d <- dilations
                    , k <- [0 .. kernelSize - 1]
                    ]
               in Just feats

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
  [ [ if i == j then v + lambda else v | (j, v) <- zip [0..] row ]
  | (i, row) <- zip [0..] m
  ]

solveLinear :: [[Double]] -> [Double] -> [Double]
solveLinear a b =
  let n = length a
      aug0 = toMatrix (zipWith (\row bi -> row ++ [bi]) a b)
      aug = forwardElimination n aug0
   in V.toList (backSubstitution n aug)

type Matrix = V.Vector (V.Vector Double)

toMatrix :: [[Double]] -> Matrix
toMatrix = V.fromList . map V.fromList

forwardElimination :: Int -> Matrix -> Matrix
forwardElimination n = go 0
  where
    eps = 1e-12

    go k m
      | k >= n = m
      | otherwise =
          let pivotRow = argMaxAbs (\row -> abs (row V.! k)) [k .. n - 1] m
              m1 = swapRows k pivotRow m
              rowK = m1 V.! k
              pivot = rowK V.! k
           in if abs pivot < eps
                then error "Singular matrix in solveLinear"
                else
                  let m2 =
                        V.imap
                          (\i row ->
                              if i <= k
                                then row
                                else
                                  let factor = (row V.! k) / pivot
                                   in V.imap
                                        (\j v -> if j < k then v else v - factor * (rowK V.! j))
                                        row
                          )
                          m1
                   in go (k + 1) m2

backSubstitution :: Int -> Matrix -> V.Vector Double
backSubstitution n m =
  let go i acc =
        if i < 0
          then V.fromList acc
          else
            let row = m V.! i
                rhs = row V.! n
                coeffs = V.toList (V.slice (i + 1) (n - i - 1) row)
                s = sum (zipWith (*) coeffs acc)
                x = (rhs - s) / (row V.! i)
             in go (i - 1) (x : acc)
   in go (n - 1) []

argMaxAbs :: (V.Vector Double -> Double) -> [Int] -> Matrix -> Int
argMaxAbs f is xs =
  case is of
    [] -> error "argMaxAbs: empty"
    (i0:rest) ->
      let v0 = f (xs V.! i0)
       in fst $ foldl' (\(ib, vb) i -> let v = f (xs V.! i) in if v > vb then (i, v) else (ib, vb)) (i0, v0) rest

swapRows :: Int -> Int -> Matrix -> Matrix
swapRows i j rows
  | i == j = rows
  | otherwise = rows V.// [(i, rows V.! j), (j, rows V.! i)]
