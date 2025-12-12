module Trader.Predictors.Types
  ( SensorId(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  , SensorOutput(..)
  ) where

data SensorId
  = SensorGBT
  | SensorTCN
  | SensorTransformer
  | SensorHMM
  | SensorQuantile
  | SensorConformal
  deriving (Eq, Ord, Show, Enum, Bounded)

data RegimeProbs = RegimeProbs
  { rpTrend :: !Double
  , rpMR :: !Double
  , rpHighVol :: !Double
  } deriving (Eq, Show)

data Quantiles = Quantiles
  { q10 :: !Double
  , q50 :: !Double
  , q90 :: !Double
  } deriving (Eq, Show)

data Interval = Interval
  { iLo :: !Double
  , iHi :: !Double
  } deriving (Eq, Show)

data SensorOutput = SensorOutput
  { soMu :: !Double
  , soSigma :: !(Maybe Double)
  , soRegimes :: !(Maybe RegimeProbs)
  , soQuantiles :: !(Maybe Quantiles)
  , soInterval :: !(Maybe Interval)
  } deriving (Eq, Show)

