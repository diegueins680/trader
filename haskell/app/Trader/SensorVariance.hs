module Trader.SensorVariance
  ( SensorVar(..)
  , emptySensorVar
  , updateResidual
  , varianceFor
  ) where

import Trader.OnlineStats (Welford, emptyWelford, updateWelford, varianceWelford)
import Trader.Predictors.Types (SensorId(..))

data SensorVar = SensorVar
  { svGBT :: !Welford
  , svTCN :: !Welford
  , svTransformer :: !Welford
  , svHMM :: !Welford
  , svQuantile :: !Welford
  , svConformal :: !Welford
  } deriving (Eq, Show)

emptySensorVar :: SensorVar
emptySensorVar =
  SensorVar
    { svGBT = emptyWelford
    , svTCN = emptyWelford
    , svTransformer = emptyWelford
    , svHMM = emptyWelford
    , svQuantile = emptyWelford
    , svConformal = emptyWelford
    }

updateResidual :: SensorId -> Double -> SensorVar -> SensorVar
updateResidual sid resid sv =
  case sid of
    SensorGBT -> sv { svGBT = updateWelford resid (svGBT sv) }
    SensorTCN -> sv { svTCN = updateWelford resid (svTCN sv) }
    SensorTransformer -> sv { svTransformer = updateWelford resid (svTransformer sv) }
    SensorHMM -> sv { svHMM = updateWelford resid (svHMM sv) }
    SensorQuantile -> sv { svQuantile = updateWelford resid (svQuantile sv) }
    SensorConformal -> sv { svConformal = updateWelford resid (svConformal sv) }

varianceFor :: SensorId -> SensorVar -> Maybe Double
varianceFor sid sv =
  case sid of
    SensorGBT -> varianceWelford (svGBT sv)
    SensorTCN -> varianceWelford (svTCN sv)
    SensorTransformer -> varianceWelford (svTransformer sv)
    SensorHMM -> varianceWelford (svHMM sv)
    SensorQuantile -> varianceWelford (svQuantile sv)
    SensorConformal -> varianceWelford (svConformal sv)

