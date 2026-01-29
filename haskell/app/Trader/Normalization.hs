module Trader.Normalization (
    NormType (..),
    NormState (..),
    parseNormType,
    fitNorm,
    forwardNorm,
    inverseNorm,
    forwardSeries,
    inverseSeries,
) where

import Data.Char (isSpace, toLower)

data NormType = NormNone | NormMinMax | NormStandard | NormLog
    deriving (Eq, Show)

data NormState
    = NSNone
    | NSMinMax !Double !Double
    | NSStandard !Double !Double
    | NSLog
    deriving (Eq, Show)

parseNormType :: String -> Maybe NormType
parseNormType s =
    case map toLower (trim s) of
        "none" -> Just NormNone
        "minmax" -> Just NormMinMax
        "standard" -> Just NormStandard
        "log" -> Just NormLog
        _ -> Nothing

fitNorm :: NormType -> [Double] -> NormState
fitNorm nt xs =
    let xsFinite = filter isFinite xs
     in case nt of
            NormNone -> NSNone
            NormMinMax ->
                case xsFinite of
                    [] -> NSNone
                    _ ->
                        let mn = minimum xsFinite
                            mx = maximum xsFinite
                         in NSMinMax mn mx
            NormStandard ->
                case xsFinite of
                    [] -> NSNone
                    _ ->
                        let n = fromIntegral (length xsFinite)
                            mu = sum xsFinite / n
                            var = sum (map (\v -> (v - mu) ** 2) xsFinite) / n
                            sigma = sqrt (var + 1e-8)
                         in NSStandard mu sigma
            NormLog ->
                case xsFinite of
                    [] -> NSNone
                    _ ->
                        if any (\v -> v <= 0) xsFinite
                            then NSNone
                            else NSLog

forwardNorm :: NormState -> Double -> Double
forwardNorm st x =
    case st of
        NSNone -> x
        NSMinMax mn mx -> (x - mn) / (mx - mn + 1e-8)
        NSStandard mu sigma -> (x - mu) / sigma
        NSLog -> log x

inverseNorm :: NormState -> Double -> Double
inverseNorm st x =
    case st of
        NSNone -> x
        NSMinMax mn mx -> x * (mx - mn + 1e-8) + mn
        NSStandard mu sigma -> x * sigma + mu
        NSLog -> exp x

forwardSeries :: NormState -> [Double] -> [Double]
forwardSeries st = map (forwardNorm st)

inverseSeries :: NormState -> [Double] -> [Double]
inverseSeries st = map (inverseNorm st)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)
