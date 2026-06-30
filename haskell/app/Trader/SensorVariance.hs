module Trader.SensorVariance (
    SensorVarianceConfig (..),
    SensorVar (..),
    defaultSensorVarianceConfig,
    defaultSensorVarianceEwmaAlpha,
    emptySensorVar,
    updateResidual,
    updateResidualWith,
    validateSensorVarianceConfig,
    varianceFor,
) where

import Data.Either (fromRight)

import Trader.OnlineStats (Welford, emptyWelford, updateWelford, varianceWelford)
import Trader.Predictors.Types (SensorId (..))

data EwmaVar = EwmaVar
    { evVar :: !Double
    , evReady :: !Bool
    }
    deriving (Eq, Show)

newtype SensorVarianceConfig = SensorVarianceConfig
    { svcEwmaAlpha :: Double
    }
    deriving (Eq, Show)

defaultSensorVarianceEwmaAlpha :: Double
defaultSensorVarianceEwmaAlpha = 0.05

defaultSensorVarianceConfig :: SensorVarianceConfig
defaultSensorVarianceConfig =
    SensorVarianceConfig
        { svcEwmaAlpha = defaultSensorVarianceEwmaAlpha
        }

validateSensorVarianceConfig :: SensorVarianceConfig -> Either String SensorVarianceConfig
validateSensorVarianceConfig config = do
    let alpha = svcEwmaAlpha config
    if isFinite alpha && alpha >= 0 && alpha <= 1
        then Right config
        else Left "sensor variance EWMA alpha must be finite and between 0 and 1"

sanitizeSensorVarianceConfig :: SensorVarianceConfig -> SensorVarianceConfig
sanitizeSensorVarianceConfig config =
    fromRight defaultSensorVarianceConfig (validateSensorVarianceConfig config)

emptyEwmaVar :: EwmaVar
emptyEwmaVar = EwmaVar{evVar = 0, evReady = False}

updateEwmaVarWith :: SensorVarianceConfig -> Double -> EwmaVar -> EwmaVar
updateEwmaVarWith config resid ev =
    if not (isFinite resid)
        then ev
        else
            let x2 = resid * resid
             in if evReady ev
                    then
                        let a = svcEwmaAlpha (sanitizeSensorVarianceConfig config)
                         in ev{evVar = (1 - a) * evVar ev + a * x2}
                    else EwmaVar{evVar = x2, evReady = True}

varianceEwma :: EwmaVar -> Maybe Double
varianceEwma ev =
    if evReady ev then finiteNonNegative (evVar ev) else Nothing

data SensorVar = SensorVar
    { svGBT :: !Welford
    , svGBTEwma :: !EwmaVar
    , svKNN :: !Welford
    , svKNNEwma :: !EwmaVar
    , svDecisionTree :: !Welford
    , svDecisionTreeEwma :: !EwmaVar
    , svTCN :: !Welford
    , svTCNEwma :: !EwmaVar
    , svPatchTST :: !Welford
    , svPatchTSTEwma :: !EwmaVar
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
        , svKNN = emptyWelford
        , svKNNEwma = emptyEwmaVar
        , svDecisionTree = emptyWelford
        , svDecisionTreeEwma = emptyEwmaVar
        , svTCN = emptyWelford
        , svTCNEwma = emptyEwmaVar
        , svPatchTST = emptyWelford
        , svPatchTSTEwma = emptyEwmaVar
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
updateResidual = updateResidualWith defaultSensorVarianceConfig

updateResidualWith :: SensorVarianceConfig -> SensorId -> Double -> SensorVar -> SensorVar
updateResidualWith config sid resid sv =
    if not (isFinite resid)
        then sv
        else case sid of
            SensorGBT ->
                sv
                    { svGBT = updateWelford resid (svGBT sv)
                    , svGBTEwma = updateEwmaVarWith config resid (svGBTEwma sv)
                    }
            SensorKNN ->
                sv
                    { svKNN = updateWelford resid (svKNN sv)
                    , svKNNEwma = updateEwmaVarWith config resid (svKNNEwma sv)
                    }
            SensorDecisionTree ->
                sv
                    { svDecisionTree = updateWelford resid (svDecisionTree sv)
                    , svDecisionTreeEwma = updateEwmaVarWith config resid (svDecisionTreeEwma sv)
                    }
            SensorTCN ->
                sv
                    { svTCN = updateWelford resid (svTCN sv)
                    , svTCNEwma = updateEwmaVarWith config resid (svTCNEwma sv)
                    }
            SensorPatchTST ->
                sv
                    { svPatchTST = updateWelford resid (svPatchTST sv)
                    , svPatchTSTEwma = updateEwmaVarWith config resid (svPatchTSTEwma sv)
                    }
            SensorTransformer ->
                sv
                    { svTransformer = updateWelford resid (svTransformer sv)
                    , svTransformerEwma = updateEwmaVarWith config resid (svTransformerEwma sv)
                    }
            SensorHMM ->
                sv
                    { svHMM = updateWelford resid (svHMM sv)
                    , svHMMEwma = updateEwmaVarWith config resid (svHMMEwma sv)
                    }
            SensorQuantile ->
                sv
                    { svQuantile = updateWelford resid (svQuantile sv)
                    , svQuantileEwma = updateEwmaVarWith config resid (svQuantileEwma sv)
                    }
            SensorConformal ->
                sv
                    { svConformal = updateWelford resid (svConformal sv)
                    , svConformalEwma = updateEwmaVarWith config resid (svConformalEwma sv)
                    }

varianceFor :: SensorId -> SensorVar -> Maybe Double
varianceFor sid sv =
    let preferEwma ewma welford =
            case varianceEwma ewma of
                Just v -> Just v
                Nothing -> varianceWelford welford >>= finiteNonNegative
     in case sid of
            SensorGBT -> preferEwma (svGBTEwma sv) (svGBT sv)
            SensorKNN -> preferEwma (svKNNEwma sv) (svKNN sv)
            SensorDecisionTree -> preferEwma (svDecisionTreeEwma sv) (svDecisionTree sv)
            SensorTCN -> preferEwma (svTCNEwma sv) (svTCN sv)
            SensorPatchTST -> preferEwma (svPatchTSTEwma sv) (svPatchTST sv)
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
