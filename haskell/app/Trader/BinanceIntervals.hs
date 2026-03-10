module Trader.BinanceIntervals (
    binanceIntervals,
    binanceIntervalsCsv,
    isBinanceInterval,
) where

import Data.Char (isDigit, isSpace, toLower)
import Data.List (intercalate)

binanceIntervals :: [String]
binanceIntervals =
    [ "1m"
    , "3m"
    , "5m"
    , "15m"
    , "30m"
    , "1h"
    , "2h"
    , "4h"
    , "6h"
    , "8h"
    , "12h"
    , "1d"
    , "3d"
    , "1w"
    , "1M"
    ]

binanceIntervalsCsv :: String
binanceIntervalsCsv = intercalate "," binanceIntervals

isBinanceInterval :: String -> Bool
isBinanceInterval v =
    let normalized = normalizeIntervalCode v
     in any ((== normalized) . normalizeIntervalCode) binanceIntervals

normalizeIntervalCode :: String -> String
normalizeIntervalCode raw =
    let s = trim raw
     in case span isDigit s of
            (digits, [u])
                | not (null digits) ->
                    digits ++ [if u == 'M' then 'M' else toLower u]
            _ -> s

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse
