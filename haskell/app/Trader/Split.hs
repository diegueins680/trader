module Trader.Split
  ( Split(..)
  , splitTrainBacktest
  ) where

import Text.Printf (printf)

data Split a = Split
  { splitTrainEndRaw :: !Int
  , splitTrainEnd :: !Int
  , splitTrain :: [a]
  , splitBacktest :: [a]
  } deriving (Eq, Show)

splitTrainBacktest :: Int -> Double -> [a] -> Either String (Split a)
splitTrainBacktest lookback backtestRatio xs =
  let n = length xs
      trainEndRaw = floor (fromIntegral n * (1 - backtestRatio) + 1e-9)
      minTrainEnd = lookback + 1
      maxTrainEnd = n - 2
      trainEnd = min maxTrainEnd (max minTrainEnd trainEndRaw)
   in if backtestRatio <= 0 || backtestRatio >= 1
        then Left "--backtest-ratio must be between 0 and 1"
        else
          if n < lookback + 3
            then
              Left
                ( printf
                    "Not enough data for train/backtest split with lookback=%d (need >= %d prices, got %d). Reduce --lookback-bars/--lookback-window or increase --bars."
                    lookback
                    (lookback + 3)
                    n
                )
            else
              Right
                Split
                  { splitTrainEndRaw = trainEndRaw
                  , splitTrainEnd = trainEnd
                  , splitTrain = take trainEnd xs
                  , splitBacktest = drop trainEnd xs
                  }
