module Trader.Predictors.PatchTST (
    PatchTSTModel (..),
    defaultPatchTstRidgeLambda,
    patchTstFeaturesAt,
    predictPatchTST,
    trainPatchTST,
    trainPatchTSTWithLambda,
) where

import Data.List (foldl', sort)
import qualified Data.Vector as V

data PatchTSTModel = PatchTSTModel
    { pmPatchLengths :: ![Int]
    , pmWeights :: ![Double] -- includes bias as last element
    , pmSigma :: !(Maybe Double)
    }
    deriving (Eq, Show)

defaultPatchTstRidgeLambda :: Double
defaultPatchTstRidgeLambda = 1e-3

patchTstFeaturesAt :: [Int] -> V.Vector Double -> Int -> Maybe [Double]
patchTstFeaturesAt patchLengths prices t =
    let lengths = normalizePatchLengths patchLengths
     in if null lengths || t < maximum lengths || t >= V.length prices
            then Nothing
            else concat <$> traverse (patchSummary prices t) lengths

predictPatchTST :: PatchTSTModel -> V.Vector Double -> Int -> Maybe (Double, Maybe Double)
predictPatchTST model prices t = do
    feats <- patchTstFeaturesAt (pmPatchLengths model) prices t
    let x = feats ++ [1.0]
        w = pmWeights model
    if length w /= length x || not (all isFiniteDouble w) || not (all isFiniteDouble x)
        then Nothing
        else
            let y = dot w x
             in if isFiniteDouble y
                    then Just (y, pmSigma model >>= finiteMaybe)
                    else Nothing

trainPatchTST :: Int -> V.Vector Double -> [(Int, Double)] -> PatchTSTModel
trainPatchTST = trainPatchTSTWithLambda defaultPatchTstRidgeLambda

trainPatchTSTWithLambda :: Double -> Int -> V.Vector Double -> [(Int, Double)] -> PatchTSTModel
trainPatchTSTWithLambda ridgeLambda lookbackBars prices trainTargets
    | lookbackBars <= 1 = emptyPatchTSTModel
    | otherwise =
        let patchLengths = patchLengthsForLookback lookbackBars
            featureDim = length patchLengths * 5 + 1
            lambda = normalizeRidgeLambda ridgeLambda
            xsYs =
                [ (x ++ [1.0], y)
                | (t, y) <- trainTargets
                , isFiniteDouble y
                , Just x <- [patchTstFeaturesAt patchLengths prices t]
                , all isFiniteDouble x
                ]
            configuredEmpty =
                PatchTSTModel
                    { pmPatchLengths = patchLengths
                    , pmWeights = replicate featureDim 0
                    , pmSigma = Nothing
                    }
         in if null xsYs
                then configuredEmpty
                else
                    let xs = map fst xsYs
                        ys = map snd xsYs
                        w = ridgeFit lambda xs ys
                     in if length w /= featureDim || not (all isFiniteDouble w)
                            then configuredEmpty
                            else
                                let preds = map (dot w) xs
                                    residuals = zipWith (-) ys preds
                                    sigma = sqrt (mean (map (\e -> e * e) residuals) + 1e-12)
                                 in PatchTSTModel
                                        { pmPatchLengths = patchLengths
                                        , pmWeights = w
                                        , pmSigma = finiteMaybe sigma
                                        }

patchLengthsForLookback :: Int -> [Int]
patchLengthsForLookback lookbackBars =
    normalizePatchLengths
        [ 4
        , max 4 (lookbackBars `div` 4)
        , max 6 (lookbackBars `div` 2)
        , lookbackBars
        ]

normalizePatchLengths :: [Int] -> [Int]
normalizePatchLengths =
    dedupeSorted . sort . filter (> 1)
  where
    dedupeSorted xs =
        case xs of
            [] -> []
            y : ys -> y : go y ys
    go _ [] = []
    go prev (y : ys)
        | y == prev = go prev ys
        | otherwise = y : go y ys

patchSummary :: V.Vector Double -> Int -> Int -> Maybe [Double]
patchSummary prices t len = do
    p0 <- validAt prices (t - len)
    p1 <- validAt prices t
    totalRet <- safeReturn p0 p1
    rets <-
        traverse
            (\i -> validAt prices (i - 1) >>= \a -> validAt prices i >>= safeReturn a)
            [t - len + 1 .. t]
    let mu = mean rets
        sd = stddev rets
        lastRet =
            case reverse rets of
                x : _ -> x
                [] -> 0
        direction = signum totalRet
        agreement =
            if direction == 0 || null rets
                then 0
                else
                    let aligned = length [r | r <- rets, signum r == direction]
                     in fromIntegral aligned / fromIntegral (length rets)
    if all isFiniteDouble [totalRet, mu, sd, lastRet, agreement]
        then Just [totalRet, mu, sd, lastRet, agreement]
        else Nothing

safeReturn :: Double -> Double -> Maybe Double
safeReturn p0 p1 =
    if p0 <= 0 || not (isFiniteDouble p0) || not (isFiniteDouble p1)
        then Nothing
        else
            let r = p1 / p0 - 1
             in if isFiniteDouble r then Just r else Nothing

validAt :: V.Vector Double -> Int -> Maybe Double
validAt xs i =
    if i < 0 || i >= V.length xs
        then Nothing
        else
            let x = xs V.! i
             in if isFiniteDouble x then Just x else Nothing

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

normalizeRidgeLambda :: Double -> Double
normalizeRidgeLambda x
    | not (isFiniteDouble x) = defaultPatchTstRidgeLambda
    | otherwise = max 0 x

finiteMaybe :: Double -> Maybe Double
finiteMaybe x
    | isFiniteDouble x = Just x
    | otherwise = Nothing

dot :: [Double] -> [Double] -> Double
dot a b = sum (zipWith (*) a b)

mean :: [Double] -> Double
mean xs =
    case xs of
        [] -> 0
        _ -> sum xs / fromIntegral (length xs)

stddev :: [Double] -> Double
stddev xs =
    let clean = filter isFiniteDouble xs
        n = length clean
     in if n < 2
            then 0
            else
                let m = mean clean
                    var = sum (map (\x -> (x - m) * (x - m)) clean) / fromIntegral (n - 1)
                 in sqrt (max 0 var)

ridgeFit :: Double -> [[Double]] -> [Double] -> [Double]
ridgeFit lambda xs ys =
    case xs of
        [] -> []
        (x0 : _) ->
            let d = length x0
             in if d <= 0
                    then []
                    else
                        let xtx = foldl' matAdd (zeroMat d d) (map (\x -> outer x x) xs)
                            xty = foldl' (zipWith (+)) (replicate d 0) (zipWith (\x y -> map (* y) x) xs ys)
                            xtxReg = addDiag lambda xtx
                         in solveLinear xtxReg xty

zeroMat :: Int -> Int -> [[Double]]
zeroMat r c = replicate r (replicate c 0)

matAdd :: [[Double]] -> [[Double]] -> [[Double]]
matAdd = zipWith (zipWith (+))

outer :: [Double] -> [Double] -> [[Double]]
outer x y = [map (* xi) y | xi <- x]

addDiag :: Double -> [[Double]] -> [[Double]]
addDiag lambda m =
    [ [if i == j then v + lambda else v | (j, v) <- zip [0 ..] row]
    | (i, row) <- zip [0 ..] m
    ]

solveLinear :: [[Double]] -> [Double] -> [Double]
solveLinear a b =
    let n = length a
     in if n <= 0
            then []
            else
                let aug0 = V.fromList (map V.fromList (zipWith (\row bi -> row ++ [bi]) a b))
                 in case forwardElimination n aug0 of
                        Nothing -> replicate n 0
                        Just aug -> V.toList (backSubstitution n aug)

type Matrix = V.Vector (V.Vector Double)

forwardElimination :: Int -> Matrix -> Maybe Matrix
forwardElimination n = go 0
  where
    eps = 1e-12
    go k m
        | k >= n = Just m
        | otherwise =
            let pivotRow = argMaxAbs (\row -> abs (row V.! k)) [k .. n - 1] m
                m1 = swapRows k pivotRow m
                rowK = m1 V.! k
                pivot = rowK V.! k
             in if abs pivot < eps
                    then Nothing
                    else
                        let m2 =
                                V.imap
                                    ( \i row ->
                                        if i <= k
                                            then row
                                            else
                                                let factor = (row V.! k) / pivot
                                                 in V.imap (\j v -> if j < k then v else v - factor * (rowK V.! j)) row
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
        [] -> 0
        i0 : rest ->
            let v0 = f (xs V.! i0)
             in fst $ foldl' (\(ib, vb) i -> let v = f (xs V.! i) in if v > vb then (i, v) else (ib, vb)) (i0, v0) rest

swapRows :: Int -> Int -> Matrix -> Matrix
swapRows i j rows
    | i == j = rows
    | otherwise = rows V.// [(i, rows V.! j), (j, rows V.! i)]

emptyPatchTSTModel :: PatchTSTModel
emptyPatchTSTModel =
    PatchTSTModel
        { pmPatchLengths = []
        , pmWeights = []
        , pmSigma = Nothing
        }
