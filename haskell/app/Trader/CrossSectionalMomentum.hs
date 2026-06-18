module Trader.CrossSectionalMomentum (
    CrossSectionalMomentumSignal (..),
    crossSectionalMomentumPredictionAt,
    crossSectionalMomentumPredictionsRange,
) where

import qualified Data.Vector as V

import Trader.MarketContext (MarketModel (..))
import Trader.Predictors.Features (FeatureInputs (..))

data CrossSectionalMomentumSignal = CrossSectionalMomentumSignal
    { csmPrediction :: !Double
    , csmExpectedReturn :: !Double
    , csmConfidence :: !Double
    , csmResidualMomentum :: !Double
    }
    deriving (Eq, Show)

crossSectionalMomentumPredictionsRange ::
    Int ->
    Maybe MarketModel ->
    FeatureInputs ->
    Int ->
    Int ->
    V.Vector Double
crossSectionalMomentumPredictionsRange lookback mMarket inputs start count =
    V.generate (max 0 count) $ \i ->
        let t = start + i
            prices = fiClose inputs
            fallback =
                if t >= 0 && t < V.length prices
                    then prices V.! t
                    else 0
         in maybe fallback csmPrediction (crossSectionalMomentumPredictionAt lookback mMarket inputs t)

crossSectionalMomentumPredictionAt ::
    Int ->
    Maybe MarketModel ->
    FeatureInputs ->
    Int ->
    Maybe CrossSectionalMomentumSignal
crossSectionalMomentumPredictionAt lookback mMarket inputs t = do
    let prices = fiClose inputs
        n = V.length prices
        lookback' = max 12 lookback
        shortBars = clampInt 4 36 (lookback' `div` 3)
        midBars = max (shortBars + 1) (clampInt 12 96 lookback')
        volBars = max 12 (min 144 lookback')
    if t < midBars || t < volBars || t < 0 || t >= n
        then Nothing
        else do
            currentPrice <- validAt prices t
            if currentPrice <= 0
                then Nothing
                else do
                    retShort <- retOver prices t shortBars
                    retMid <- retOver prices t midBars
                    rs <- returnsEndingAt prices t volBars
                    let vol = stddev rs
                    if vol <= 1.0e-12
                        then Nothing
                        else do
                            let marketShort = marketReturnOver mMarket t shortBars
                                marketMid = marketReturnOver mMarket t midBars
                                beta = maybe 1 mmBeta mMarket
                                residualShort = retShort - beta * marketShort
                                residualMid = retMid - beta * marketMid
                                signScale =
                                    if sameSign residualShort residualMid
                                        then 1
                                        else 0.35
                                residualMomentum = signScale * (0.65 * residualShort + 0.35 * residualMid)
                                z = residualMomentum / max 1.0e-12 vol
                                rawEdge =
                                    if abs z < 0.65
                                        then 0
                                        else clamp (0.50 * residualMomentum) (negate (edgeCap vol)) (edgeCap vol)
                                direction = signumInt rawEdge
                                contextScale =
                                    case mMarket of
                                        Just _ -> 1
                                        Nothing -> 0.65
                                priorScale =
                                    if direction == 0
                                        then 1
                                        else
                                            fundingScale direction inputs t
                                                * basisScale direction inputs t
                                                * flowScale direction inputs t
                                                * openInterestScale inputs t
                                scaledEdge = rawEdge * clamp (contextScale * priorScale) 0 1.25
                                prediction = currentPrice * (1 + scaledEdge)
                                confidence = clamp ((abs z - 0.50) / 2.50) 0 1 * clamp priorScale 0 1.15
                            if not (all isFiniteDouble [scaledEdge, prediction, confidence, residualMomentum])
                                then Nothing
                                else
                                    Just
                                        CrossSectionalMomentumSignal
                                            { csmPrediction = prediction
                                            , csmExpectedReturn = scaledEdge
                                            , csmConfidence = clamp confidence 0 1
                                            , csmResidualMomentum = residualMomentum
                                            }

edgeCap :: Double -> Double
edgeCap vol =
    min 0.035 (max 0.0025 (2.0 * vol))

marketReturnOver :: Maybe MarketModel -> Int -> Int -> Double
marketReturnOver Nothing _ _ = 0
marketReturnOver (Just mm) t bars =
    let lag = mmLag mm
        start = max 0 (t - bars + 1)
        end = min t (V.length lag - 1)
     in if bars <= 0 || start > end
            then 0
            else
                V.foldl'
                    (\acc v -> if isFiniteDouble v then acc + v else acc)
                    0
                    (V.slice start (end - start + 1) lag)

fundingScale :: Int -> FeatureInputs -> Int -> Double
fundingScale dir inputs t =
    case fiFunding inputs >>= (`validAt` t) of
        Nothing -> 1
        Just funding ->
            let cap = 0.001
                carryAdvantage = negate (fromIntegral dir) * funding
                pressure = min 1 (abs carryAdvantage / cap)
             in if carryAdvantage >= 0
                    then 1 + 0.15 * pressure
                    else max 0.40 (1 - 0.60 * pressure)

basisScale :: Int -> FeatureInputs -> Int -> Double
basisScale dir inputs t =
    case fiBasis inputs >>= (`validAt` t) of
        Nothing -> 1
        Just basis ->
            let cap = 0.004
                crowding = fromIntegral dir * basis
                pressure = min 1 (abs crowding / cap)
             in if crowding <= 0
                    then 1 + 0.05 * pressure
                    else max 0.70 (1 - 0.30 * pressure)

flowScale :: Int -> FeatureInputs -> Int -> Double
flowScale dir inputs t =
    case fiTakerFlow inputs >>= (`validAt` t) of
        Nothing -> 1
        Just flow ->
            let imbalance = flow - 1
                aligned = fromIntegral dir * imbalance
             in if aligned >= 0.05
                    then 1.10
                    else
                        if aligned <= (-0.05)
                            then 0.75
                            else 1

openInterestScale :: FeatureInputs -> Int -> Double
openInterestScale inputs t =
    case relDeltaAt (fiOpenInterest inputs) t of
        Nothing -> 1
        Just delta
            | delta >= 0.005 -> 1.08
            | delta <= (-0.005) -> 0.85
            | otherwise -> 1

retOver :: V.Vector Double -> Int -> Int -> Maybe Double
retOver prices t bars =
    if bars <= 0 || t - bars < 0
        then Nothing
        else do
            p0 <- validAt prices (t - bars)
            p1 <- validAt prices t
            finiteReturn p0 p1

returnsEndingAt :: V.Vector Double -> Int -> Int -> Maybe [Double]
returnsEndingAt prices t bars =
    if bars <= 0 || t - bars < 0
        then Nothing
        else sequence [validAt prices i >>= \p0 -> validAt prices (i + 1) >>= finiteReturn p0 | i <- [t - bars .. t - 1]]

finiteReturn :: Double -> Double -> Maybe Double
finiteReturn p0 p1 =
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

relDeltaAt :: Maybe (V.Vector Double) -> Int -> Maybe Double
relDeltaAt mv t = do
    xs <- mv
    prev <- validAt xs (t - 1)
    cur <- validAt xs t
    let denom = max 1.0e-12 (abs prev)
        v = (cur - prev) / denom
    if isFiniteDouble v then Just v else Nothing

stddev :: [Double] -> Double
stddev xs =
    let clean = filter isFiniteDouble xs
        n = length clean
     in if n < 2
            then 0
            else
                let m = sum clean / fromIntegral n
                    var = sum (map (\x -> (x - m) * (x - m)) clean) / fromIntegral (n - 1)
                 in sqrt (max 0 var)

sameSign :: Double -> Double -> Bool
sameSign a b =
    let sa = signumInt a
        sb = signumInt b
     in sa /= 0 && sa == sb

signumInt :: Double -> Int
signumInt x
    | x > 1.0e-12 = 1
    | x < (-1.0e-12) = -1
    | otherwise = 0

clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)

clampInt :: Int -> Int -> Int -> Int
clampInt lo hi x = max lo (min hi x)

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)
