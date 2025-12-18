module Trader.BinanceIntervals
  ( binanceIntervals
  , isBinanceInterval
  ) where

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

isBinanceInterval :: String -> Bool
isBinanceInterval s = s `elem` binanceIntervals
