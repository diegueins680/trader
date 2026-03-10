module Trader.SensorVariance (
    SensorVar (..),
    emptySensorVar,
    updateResidual,
    varianceFor,
) where

import Trader.OnlineStats (Welford, emptyWelford, updateWelford, varianceWelford)
import Trader.Predictors.Types (SensorId (..))

data EwmaVar = EwmaVar
    { evVar :: !Double
    , evReady :: !Bool
    }
    deriving (Eq, Show)

ewmaAlpha :: Double
ewmaAlpha = 0.05

emptyEwmaVar :: EwmaVar
emptyEwmaVar = EwmaVar{evVar = 0, evReady = False}

updateEwmaVar :: Double -> EwmaVar -> EwmaVar
updateEwmaVar resid ev =
    if not (isFinite resid)
        then ev
        else
            let x2 = resid * resid
             in if evReady ev
                    then
                        let a = ewmaAlpha
                         in ev{evVar = (1 - a) * evVar ev + a * x2}
                    else EwmaVar{evVar = x2, evReady = True}

varianceEwma :: EwmaVar -> Maybe Double
varianceEwma ev =
    if evReady ev then finiteNonNegative (evVar ev) else Nothing

data SensorVar = SensorVar
    { svGBT :: !Welford
    , svGBTEwma :: !EwmaVar
    , svTCN :: !Welford
    , svTCNEwma :: !EwmaVar
    , svTransformer :: !Welford
    , svTransformerEwma :: !EwmaVar
    , svHMM :: !Welford
    , svHMMEwma :: !EwmaVar
    , svQuantile :: !Welford
    , svQuantileEwma :: !EwmaVar
    , svConformal :: !Welford
    , svConformalEwma :: !EwmaVar
    }
    deriving (Eq, Show)

emptySensorVar :: SensorVar
emptySensorVar =
    SensorVar
        { svGBT = emptyWelford
        , svGBTEwma = emptyEwmaVar
        , svTCN = emptyWelford
        , svTCNEwma = emptyEwmaVar
        , svTransformer = emptyWelford
        , svTransformerEwma = emptyEwmaVar
        , svHMM = emptyWelford
        , svHMMEwma = emptyEwmaVar
        , svQuantile = emptyWelford
        , svQuantileEwma = emptyEwmaVar
        , svConformal = emptyWelford
        , svConformalEwma = emptyEwmaVar
        }

updateResidual :: SensorId -> Double -> SensorVar -> SensorVar
updateResidual sid resid sv =
    if not (isFinite resid)
        then sv
        else case sid of
            SensorGBT ->
                sv
                    { svGBT = updateWelford resid (svGBT sv)
                    , svGBTEwma = updateEwmaVar resid (svGBTEwma sv)
                    }
            SensorTCN ->
                sv
                    { svTCN = updateWelford resid (svTCN sv)
                    , svTCNEwma = updateEwmaVar resid (svTCNEwma sv)
                    }
            SensorTransformer ->
                sv
                    { svTransformer = updateWelford resid (svTransformer sv)
                    , svTransformerEwma = updateEwmaVar resid (svTransformerEwma sv)
                    }
            SensorHMM ->
                sv
                    { svHMM = updateWelford resid (svHMM sv)
                    , svHMMEwma = updateEwmaVar resid (svHMMEwma sv)
                    }
            SensorQuantile ->
                sv
                    { svQuantile = updateWelford resid (svQuantile sv)
                    , svQuantileEwma = updateEwmaVar resid (svQuantileEwma sv)
                    }
            SensorConformal ->
                sv
                    { svConformal = updateWelford resid (svConformal sv)
                    , svConformalEwma = updateEwmaVar resid (svConformalEwma sv)
                    }

varianceFor :: SensorId -> SensorVar -> Maybe Double
varianceFor sid sv =
    let preferEwma ewma welford =
            case varianceEwma ewma of
                Just v -> Just v
                Nothing -> varianceWelford welford >>= finiteNonNegative
     in case sid of
            SensorGBT -> preferEwma (svGBTEwma sv) (svGBT sv)
            SensorTCN -> preferEwma (svTCNEwma sv) (svTCN sv)
            SensorTransformer -> preferEwma (svTransformerEwma sv) (svTransformer sv)
            SensorHMM -> preferEwma (svHMMEwma sv) (svHMM sv)
            SensorQuantile -> preferEwma (svQuantileEwma sv) (svQuantile sv)
            SensorConformal -> preferEwma (svConformalEwma sv) (svConformal sv)

finiteNonNegative :: Double -> Maybe Double
finiteNonNegative x =
    if isFinite x && x >= 0
        then Just x
        else Nothing

isFinite :: Double -> Bool
isFinite x = not (isNaN x || isInfinite x)
