module Trader.Predictors
  ( SensorId(..)
  , SensorOutput(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  , PredictorBundle(..)
  , HMMFilter(..)
  , HMM3(..)
  , trainPredictors
  , initHMMFilter
  , predictSensors
  , updateHMM
  ) where

import qualified Data.Vector as V

import Trader.Predictors.Types
  ( SensorId(..)
  , SensorOutput(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  )
import Trader.Predictors.Features (FeatureSpec, mkFeatureSpec, featuresAt, forwardReturnAt, buildDataset)
import Trader.Predictors.GBDT (GBDTModel(..), trainGBDT, predictGBDT)
import Trader.Predictors.TCN (TCNModel(..), trainTCN, predictTCN)
import Trader.Predictors.Transformer (TransformerModel(..), trainTransformer, predictTransformer)
import Trader.Predictors.Quantile (QuantileModel(..), trainQuantileModel, predictQuantiles)
import Trader.Predictors.Conformal (ConformalModel(..), fitConformal, predictInterval)
import Trader.Predictors.HMM (HMM3(..), HMMFilter(..), fitHMM3, filterPosterior, predictNextFromPosterior, updatePosterior)

data PredictorBundle = PredictorBundle
  { pbFeatureSpec :: !FeatureSpec
  , pbGBDT :: !GBDTModel
  , pbTCN :: !TCNModel
  , pbTransformer :: !TransformerModel
  , pbQuantile :: !QuantileModel
  , pbConformal :: !ConformalModel
  , pbHMM :: !HMM3
  } deriving (Eq, Show)

trainPredictors :: Int -> V.Vector Double -> PredictorBundle
trainPredictors lookbackBars trainPrices =
  let fs = mkFeatureSpec lookbackBars
      dataset = buildDataset fs trainPrices
      gbdt = trainGBDT 60 0.1 dataset
      transformer = trainTransformer 5.0 512 dataset
      quant = trainQuantileModel 20 5e-2 1e-3 dataset
      hmmObs =
        [ y
        | t <- [0 .. V.length trainPrices - 2]
        , Just y <- [forwardReturnAt trainPrices t]
        ]
      hmm = fitHMM3 10 hmmObs
      -- Conformal: calibrate on the last 20% of the training dataset.
      (_, calib) = splitCalib dataset
      absRes =
        [ abs (y - mu)
        | (x, y) <- calib
        , let (mu, _) = predictGBDT gbdt x
        ]
      conformal = fitConformal 0.2 absRes
      tcnTargets =
        [ (t, y)
        | t <- [0 .. V.length trainPrices - 2]
        , Just y <- [forwardReturnAt trainPrices t]
        ]
      tcn = trainTCN lookbackBars trainPrices tcnTargets
   in PredictorBundle
        { pbFeatureSpec = fs
        , pbGBDT = gbdt
        , pbTCN = tcn
        , pbTransformer = transformer
        , pbQuantile = quant
        , pbConformal = conformal
        , pbHMM = hmm
        }

initHMMFilter :: PredictorBundle -> [Double] -> HMMFilter
initHMMFilter pb obs = filterPosterior (pbHMM pb) obs

splitCalib :: [a] -> ([a], [a])
splitCalib xs =
  let n = length xs
      k = max 1 (floor (0.2 * fromIntegral n))
   in splitAt (n - k) xs

-- | Sensor predictions at bar t (end of bar t) for forward return r_t.
-- Returns (sensor outputs, HMM predicted state distribution) where the latter
-- must be passed back to 'updateHMM' once r_t is observed.
predictSensors
  :: PredictorBundle
  -> V.Vector Double
  -> HMMFilter
  -> Int
  -> ([(SensorId, SensorOutput)], [Double])
predictSensors pb prices hmmFilt t =
  let fs = pbFeatureSpec pb
      feat = featuresAt fs prices t

      gbdtOut =
        case feat of
          Nothing -> []
          Just x ->
            let (mu, sig) = predictGBDT (pbGBDT pb) x
             in [(SensorGBT, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]

      tcnOut =
        case predictTCN (pbTCN pb) prices t of
          Nothing -> []
          Just (mu, sig) ->
            [(SensorTCN, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]

      transformerOut =
        case feat of
          Nothing -> []
          Just x ->
            let (mu, sig) = predictTransformer (pbTransformer pb) x
             in [(SensorTransformer, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]

      quantOut =
        case feat of
          Nothing -> []
          Just x ->
            let (q10', q50', q90', mu, sig) = predictQuantiles (pbQuantile pb) x
             in
              [ ( SensorQuantile
                , SensorOutput
                    { soMu = mu
                    , soSigma = sig
                    , soRegimes = Nothing
                    , soQuantiles = Just (Quantiles q10' q50' q90')
                    , soInterval = Nothing
                    }
                )
              ]

      conformalOut =
        case feat of
          Nothing -> []
          Just x ->
            let (mu, _) = predictGBDT (pbGBDT pb) x
                (lo, hi, sig) = predictInterval (pbConformal pb) mu
             in
              [ ( SensorConformal
                , SensorOutput
                    { soMu = mu
                    , soSigma = Just sig
                    , soRegimes = Nothing
                    , soQuantiles = Nothing
                    , soInterval = Just (Interval lo hi)
                    }
                )
              ]

      (reg, hmmMu, hmmSigma, predState) = predictNextFromPosterior (pbHMM pb) hmmFilt
      hmmOut =
        [ ( SensorHMM
          , SensorOutput
              { soMu = hmmMu
              , soSigma = Just hmmSigma
              , soRegimes = Just reg
              , soQuantiles = Nothing
              , soInterval = Nothing
              }
          )
        ]

   in (gbdtOut ++ tcnOut ++ transformerOut ++ hmmOut ++ quantOut ++ conformalOut, predState)

updateHMM :: PredictorBundle -> [Double] -> Double -> HMMFilter
updateHMM pb predState realizedReturn =
  updatePosterior (pbHMM pb) predState realizedReturn
