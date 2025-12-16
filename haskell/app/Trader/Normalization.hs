module Trader.Normalization
  ( NormType(..)
  , NormState(..)
  , parseNormType
  , fitNorm
  , forwardNorm
  , inverseNorm
  , forwardSeries
  , inverseSeries
  ) where

import Data.Char (toLower, isSpace)

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
  case nt of
    NormNone -> NSNone
    NormMinMax ->
      let mn = minimum xs
          mx = maximum xs
       in NSMinMax mn mx
    NormStandard ->
      let n = fromIntegral (length xs)
          mu = sum xs / n
          var = sum (map (\v -> (v - mu) ** 2) xs) / n
          sigma = sqrt (var + 1e-8)
       in NSStandard mu sigma
    NormLog ->
      if any (<= 0) xs
        then error "log normalization requires all prices > 0"
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
