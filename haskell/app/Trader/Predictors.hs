module Trader.Predictors
  ( SensorId(..)
  , PredictorSet
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
  , PredictorSet
  , predictorEnabled
  , SensorOutput(..)
  , RegimeProbs(..)
  , Quantiles(..)
  , Interval(..)
  )
import Trader.Predictors.Features (FeatureSpec, mkFeatureSpec, featuresAt, forwardReturnAt, buildDataset)
import Trader.Predictors.GBDT (GBDTModel(..), trainGBDT, predictGBDT)
import Trader.Predictors.TCN (TCNModel(..), trainTCN, predictTCN)
import Trader.Predictors.Transformer (TransformerModel(..), trainTransformer, predictTransformer)
import Trader.Predictors.Quantile (LinModel(..), QuantileModel(..), trainQuantileModel, predictQuantiles)
import Trader.Predictors.Conformal (ConformalModel(..), fitConformal, predictInterval)
import Trader.Predictors.HMM (HMM3(..), HMMFilter(..), fitHMM3, filterPosterior, predictNextFromPosterior, updatePosterior)

data PredictorBundle = PredictorBundle
  { pbEnabled :: !PredictorSet
  , pbFeatureSpec :: !FeatureSpec
  , pbGBDT :: !GBDTModel
  , pbTCN :: !TCNModel
  , pbTransformer :: !TransformerModel
  , pbQuantile :: !QuantileModel
  , pbConformal :: !ConformalModel
  , pbHMM :: !HMM3
  } deriving (Eq, Show)

trainPredictors :: PredictorSet -> Int -> V.Vector Double -> PredictorBundle
trainPredictors enabled lookbackBars trainPrices =
  let fs = mkFeatureSpec lookbackBars
      useGbdt = predictorEnabled enabled SensorGBT
      useTcn = predictorEnabled enabled SensorTCN
      useTransformer = predictorEnabled enabled SensorTransformer
      useHmm = predictorEnabled enabled SensorHMM
      useQuantile = predictorEnabled enabled SensorQuantile
      useConformal = predictorEnabled enabled SensorConformal
      needFeatures = useGbdt || useTransformer || useQuantile || useConformal
      dataset = if needFeatures then buildDataset fs trainPrices else []
      (trainSet, calib) =
        if needFeatures
          then splitCalib dataset
          else ([], [])
      trainSet' = if null trainSet then dataset else trainSet
      trainLen = length trainSet'
      trainPriceLen =
        if needFeatures
          then min (V.length trainPrices) (lookbackBars + trainLen)
          else V.length trainPrices
      trainPrices' = V.slice 0 trainPriceLen trainPrices

      emptyGbdt =
        GBDTModel
          { gmBase = 0
          , gmLearningRate = 0
          , gmFeatureDim = 0
          , gmStumps = []
          , gmSigma = Nothing
          }
      emptyLin = LinModel { lmW = [], lmB = 0 }
      emptyQuant = QuantileModel emptyLin emptyLin emptyLin
      emptyTransformer = TransformerModel { trKeys = [], trTargets = [], trTemperature = 0, trFeatureDim = 0 }
      emptyTcn =
        TCNModel
          { tmDilations = []
          , tmKernelSize = 0
          , tmWeights = []
          , tmSigma = Nothing
          }

      gbdtTrained = useGbdt || useConformal
      gbdt =
        if not gbdtTrained
          then emptyGbdt
          else if null trainSet'
            then emptyGbdt
            else trainGBDT 60 0.1 trainSet'
      quant =
        if useQuantile
          then
            if null trainSet'
              then emptyQuant
              else trainQuantileModel 20 5e-2 1e-3 trainSet'
          else emptyQuant
      transformer =
        if useTransformer
          then trainTransformer 5.0 512 trainSet'
          else emptyTransformer
      hmmObs =
        if useHmm
          then
            [ y
            | t <- [0 .. V.length trainPrices' - 2]
            , Just y <- [forwardReturnAt trainPrices' t]
            ]
          else []
      hmm =
        if useHmm
          then fitHMM3 10 hmmObs
          else fitHMM3 0 []
      -- Conformal: calibrate on a holdout split (last 20%).
      absRes =
        if useConformal
          then
            [ abs (y - mu)
            | (x, y) <- calib
            , let (mu, _) = predictGBDT gbdt x
            ]
          else []
      conformal =
        if useConformal
          then fitConformal 0.2 absRes
          else fitConformal 0.2 []
      tcnTargets =
        if useTcn
          then
            [ (t, y)
            | t <- [0 .. V.length trainPrices' - 2]
            , Just y <- [forwardReturnAt trainPrices' t]
            ]
          else []
      tcn =
        if useTcn
          then trainTCN lookbackBars trainPrices' tcnTargets
          else emptyTcn
   in PredictorBundle
        { pbEnabled = enabled
        , pbFeatureSpec = fs
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
      k0 = floor (0.2 * fromIntegral n)
      k =
        if n < 3
          then 0
          else min (n - 2) (max 1 k0)
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
      enabled = pbEnabled pb
      useGbdt = predictorEnabled enabled SensorGBT
      useTcn = predictorEnabled enabled SensorTCN
      useTransformer = predictorEnabled enabled SensorTransformer
      useHmm = predictorEnabled enabled SensorHMM
      useQuantile = predictorEnabled enabled SensorQuantile
      useConformal = predictorEnabled enabled SensorConformal
      needFeatures = useGbdt || useTransformer || useQuantile || useConformal
      feat = if needFeatures then featuresAt fs prices t else Nothing
      gbdtReady = gmFeatureDim (pbGBDT pb) > 0
      quantReady = not (null (lmW (qm50 (pbQuantile pb))))
      gbdtPred =
        case feat of
          Nothing -> Nothing
          Just x ->
            if gbdtReady && (useGbdt || useConformal)
              then Just (predictGBDT (pbGBDT pb) x)
              else Nothing

      gbdtOut =
        case gbdtPred of
          Just (mu, sig) | useGbdt ->
            [(SensorGBT, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]
          _ -> []

      tcnOut =
        if not useTcn
          then []
          else
            case predictTCN (pbTCN pb) prices t of
              Nothing -> []
              Just (mu, sig) ->
                [(SensorTCN, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]

      transformerOut =
        case feat of
          Nothing -> []
          Just x ->
            if not useTransformer
              then []
              else
                let (mu, sig) = predictTransformer (pbTransformer pb) x
                 in [(SensorTransformer, SensorOutput { soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing })]

      quantOut =
        case feat of
          Nothing -> []
          Just x ->
            if not quantReady || not useQuantile
              then []
              else
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
        case gbdtPred of
          Just (mu, _) | useConformal ->
            let (lo, hi, sig) = predictInterval (pbConformal pb) mu
             in
              [ ( SensorConformal
                , SensorOutput
                    { soMu = mu
                    , soSigma = sig
                    , soRegimes = Nothing
                    , soQuantiles = Nothing
                    , soInterval = Just (Interval lo hi)
                    }
                )
              ]
          _ -> []

      (hmmOut, predState) =
        if not useHmm
          then ([], hfPosterior hmmFilt)
          else
            let (reg, hmmMu, hmmSigma, predState') = predictNextFromPosterior (pbHMM pb) hmmFilt
             in ( [ ( SensorHMM
                    , SensorOutput
                        { soMu = hmmMu
                        , soSigma = Just hmmSigma
                        , soRegimes = Just reg
                        , soQuantiles = Nothing
                        , soInterval = Nothing
                        }
                    )
                  ]
                , predState'
                )

   in (gbdtOut ++ tcnOut ++ transformerOut ++ hmmOut ++ quantOut ++ conformalOut, predState)

updateHMM :: PredictorBundle -> [Double] -> Double -> HMMFilter
updateHMM pb predState realizedReturn =
  if predictorEnabled (pbEnabled pb) SensorHMM
    then updatePosterior (pbHMM pb) predState realizedReturn
    else HMMFilter { hfPosterior = predState }
