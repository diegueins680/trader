module Trader.Predictors (
    SensorId (..),
    PredictorSet,
    SensorOutput (..),
    RegimeProbs (..),
    Quantiles (..),
    Interval (..),
    PredictorTrainingConfig (..),
    PredictorBundle (..),
    HMMFilter (..),
    HMM3 (..),
    defaultPredictorTrainingConfig,
    trainPredictors,
    trainPredictorsWithConfig,
    trainPredictorsWithMarket,
    trainPredictorsWithMarketConfig,
    trainPredictorsWithInputs,
    trainPredictorsWithInputsConfig,
    trainPredictorsWithInputsWithMarket,
    trainPredictorsWithInputsWithMarketConfig,
    initHMMFilter,
    predictSensors,
    predictSensorsWithInputs,
    updateHMM,
) where

import qualified Data.Vector as V

import Trader.MarketContext (MarketModel)
import Trader.Predictors.Conformal (ConformalModel (..), fitConformal, predictInterval)
import Trader.Predictors.DecisionTree (DecisionTreeModel (..), predictDecisionTree, trainDecisionTree)
import Trader.Predictors.Features (
    FeatureInputs (..),
    FeatureSpec,
    buildDatasetWithIndexWithInputsWithMarket,
    featureInputsFromClose,
    featuresAtWithInputsWithMarket,
    forwardReturnAt,
    mkFeatureSpec,
 )
import Trader.Predictors.GBDT (GBDTModel (..), predictGBDT, trainGBDT)
import Trader.Predictors.HMM (HMM3 (..), HMMFilter (..), filterPosterior, fitHMM3, predictNextFromPosterior, updatePosterior)
import Trader.Predictors.KNN (KNNModel (..), predictKNN, trainKNN)
import Trader.Predictors.Quantile (LinModel (..), QuantileModel (..), predictQuantiles, trainQuantileModel)
import Trader.Predictors.TCN (TCNModel (..), defaultTcnRidgeLambda, predictTCN, trainTCNWithLambda)
import Trader.Predictors.Transformer (TransformerModel (..), predictTransformer, trainTransformer)
import Trader.Predictors.Types (
    Interval (..),
    PredictorSet,
    Quantiles (..),
    RegimeProbs (..),
    SensorId (..),
    SensorOutput (..),
    predictorEnabled,
 )

data PredictorBundle = PredictorBundle
    { pbEnabled :: !PredictorSet
    , pbFeatureSpec :: !FeatureSpec
    , pbMarketModel :: !(Maybe MarketModel)
    , pbGBDT :: !GBDTModel
    , pbKNN :: !KNNModel
    , pbDecisionTree :: !DecisionTreeModel
    , pbTCN :: !TCNModel
    , pbTransformer :: !TransformerModel
    , pbQuantile :: !QuantileModel
    , pbConformal :: !ConformalModel
    , pbHMM :: !HMM3
    }
    deriving (Eq, Show)

data PredictorTrainingConfig = PredictorTrainingConfig
    { ptcGbdtTrees :: !Int
    , ptcGbdtLearningRate :: !Double
    , ptcCalibrationRatio :: !Double
    , ptcConformalAlpha :: !Double
    , ptcTcnRidgeLambda :: !Double
    }
    deriving (Eq, Show)

defaultPredictorTrainingConfig :: PredictorTrainingConfig
defaultPredictorTrainingConfig =
    PredictorTrainingConfig
        { ptcGbdtTrees = 60
        , ptcGbdtLearningRate = 0.1
        , ptcCalibrationRatio = 0.2
        , ptcConformalAlpha = 0.2
        , ptcTcnRidgeLambda = defaultTcnRidgeLambda
        }

sanitizePredictorTrainingConfig :: PredictorTrainingConfig -> PredictorTrainingConfig
sanitizePredictorTrainingConfig cfg =
    cfg
        { ptcGbdtTrees = max 1 (ptcGbdtTrees cfg)
        , ptcGbdtLearningRate = max 1e-12 (ptcGbdtLearningRate cfg)
        , ptcCalibrationRatio = clamp 0 0.95 (ptcCalibrationRatio cfg)
        , ptcConformalAlpha = clamp 1e-6 0.999999 (ptcConformalAlpha cfg)
        , ptcTcnRidgeLambda =
            let lambda = ptcTcnRidgeLambda cfg
             in if isNaN lambda || isInfinite lambda
                    then defaultTcnRidgeLambda
                    else max 0 lambda
        }

trainPredictors :: PredictorSet -> Int -> V.Vector Double -> PredictorBundle
trainPredictors =
    trainPredictorsWithConfig defaultPredictorTrainingConfig

trainPredictorsWithConfig :: PredictorTrainingConfig -> PredictorSet -> Int -> V.Vector Double -> PredictorBundle
trainPredictorsWithConfig cfg enabled lookbackBars =
    trainPredictorsWithMarketConfig cfg enabled lookbackBars Nothing

trainPredictorsWithMarket :: PredictorSet -> Int -> Maybe MarketModel -> V.Vector Double -> PredictorBundle
trainPredictorsWithMarket =
    trainPredictorsWithMarketConfig defaultPredictorTrainingConfig

trainPredictorsWithMarketConfig :: PredictorTrainingConfig -> PredictorSet -> Int -> Maybe MarketModel -> V.Vector Double -> PredictorBundle
trainPredictorsWithMarketConfig cfg enabled lookbackBars mMarketModel trainPrices =
    trainPredictorsWithInputsWithMarketConfig cfg enabled lookbackBars mMarketModel (featureInputsFromClose trainPrices)

trainPredictorsWithInputs :: PredictorSet -> Int -> FeatureInputs -> PredictorBundle
trainPredictorsWithInputs =
    trainPredictorsWithInputsConfig defaultPredictorTrainingConfig

trainPredictorsWithInputsConfig :: PredictorTrainingConfig -> PredictorSet -> Int -> FeatureInputs -> PredictorBundle
trainPredictorsWithInputsConfig cfg enabled lookbackBars =
    trainPredictorsWithInputsWithMarketConfig cfg enabled lookbackBars Nothing

trainPredictorsWithInputsWithMarket :: PredictorSet -> Int -> Maybe MarketModel -> FeatureInputs -> PredictorBundle
trainPredictorsWithInputsWithMarket =
    trainPredictorsWithInputsWithMarketConfig defaultPredictorTrainingConfig

trainPredictorsWithInputsWithMarketConfig :: PredictorTrainingConfig -> PredictorSet -> Int -> Maybe MarketModel -> FeatureInputs -> PredictorBundle
trainPredictorsWithInputsWithMarketConfig cfg0 enabled lookbackBars mMarketModel trainInputs =
    let cfg = sanitizePredictorTrainingConfig cfg0
        fs = mkFeatureSpec lookbackBars
        trainPrices = fiClose trainInputs
        useGbdt = predictorEnabled enabled SensorGBT
        useKnn = predictorEnabled enabled SensorKNN
        useDecisionTree = predictorEnabled enabled SensorDecisionTree
        useTcn = predictorEnabled enabled SensorTCN
        useTransformer = predictorEnabled enabled SensorTransformer
        useHmm = predictorEnabled enabled SensorHMM
        useQuantile = predictorEnabled enabled SensorQuantile
        useConformal = predictorEnabled enabled SensorConformal
        needFeatures = useGbdt || useKnn || useDecisionTree || useTransformer || useQuantile || useConformal
        datasetWithIndex =
            if needFeatures
                then buildDatasetWithIndexWithInputsWithMarket fs mMarketModel trainInputs
                else []
        (trainSetIdx, calibIdx) =
            if needFeatures
                then splitCalib (ptcCalibrationRatio cfg) datasetWithIndex
                else ([], [])
        trainSetIdx' = if null trainSetIdx then datasetWithIndex else trainSetIdx
        trainSet = [(x, y) | (_, x, y) <- trainSetIdx']
        calib = [(x, y) | (_, x, y) <- calibIdx]
        lastTrainIndex =
            case trainSetIdx' of
                [] -> Nothing
                (x : xs) ->
                    let step _ curr = curr
                        (t, _, _) = foldl step x xs
                     in Just t
        trainPriceLen =
            if needFeatures
                then case lastTrainIndex of
                    Nothing -> V.length trainPrices
                    Just tLast -> min (V.length trainPrices) (tLast + 2)
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
        emptyKnn =
            KNNModel
                { kmK = 0
                , kmFeatureDim = 0
                , kmMeans = []
                , kmScales = []
                , kmExamples = []
                , kmSigmaBase = Nothing
                }
        emptyDecisionTree =
            DecisionTreeModel
                { dmFeatureDim = 0
                , dmMaxDepth = 0
                , dmMinLeafSize = 0
                , dmRoot = Nothing
                , dmSigmaBase = Nothing
                }
        emptyLin = LinModel{lmW = [], lmB = 0}
        emptyQuant = QuantileModel emptyLin emptyLin emptyLin
        emptyTransformer = TransformerModel{trKeys = [], trTargets = [], trTemperature = 0, trFeatureDim = 0}
        emptyTcn =
            TCNModel
                { tmDilations = []
                , tmKernelSize = 0
                , tmWeights = []
                , tmSigma = Nothing
                }

        gbdtTrained = useGbdt || useConformal
        gbdt
            | not gbdtTrained = emptyGbdt
            | null trainSet = emptyGbdt
            | otherwise = trainGBDT (ptcGbdtTrees cfg) (ptcGbdtLearningRate cfg) trainSet
        knn =
            if useKnn
                then
                    if null trainSet
                        then emptyKnn
                        else trainKNN 1024 15 trainSet
                else emptyKnn
        decisionTree =
            if useDecisionTree
                then
                    if null trainSet
                        then emptyDecisionTree
                        else trainDecisionTree 6 12 trainSet
                else emptyDecisionTree
        quant =
            if useQuantile
                then
                    if null trainSet
                        then emptyQuant
                        else trainQuantileModel 20 5e-2 1e-3 trainSet
                else emptyQuant
        transformer =
            if useTransformer
                then trainTransformer 5.0 512 trainSet
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
                then fitConformal (ptcConformalAlpha cfg) absRes
                else fitConformal (ptcConformalAlpha cfg) []
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
                then trainTCNWithLambda (ptcTcnRidgeLambda cfg) lookbackBars trainPrices' tcnTargets
                else emptyTcn
     in PredictorBundle
            { pbEnabled = enabled
            , pbFeatureSpec = fs
            , pbMarketModel = mMarketModel
            , pbGBDT = gbdt
            , pbKNN = knn
            , pbDecisionTree = decisionTree
            , pbTCN = tcn
            , pbTransformer = transformer
            , pbQuantile = quant
            , pbConformal = conformal
            , pbHMM = hmm
            }

initHMMFilter :: PredictorBundle -> [Double] -> HMMFilter
initHMMFilter pb = filterPosterior (pbHMM pb)

splitCalib :: Double -> [a] -> ([a], [a])
splitCalib ratio xs =
    let n = length xs
        ratio' = clamp 0 0.95 ratio
        k0 = floor (ratio' * fromIntegral n)
        k =
            if n < 3
                then 0
                else min (n - 2) (max 1 k0)
     in splitAt (n - k) xs

clamp :: Double -> Double -> Double -> Double
clamp lo hi x =
    let x' = if isNaN x || isInfinite x then lo else x
     in max lo (min hi x')

{- | Sensor predictions at bar t (end of bar t) for forward return r_t.
Returns (sensor outputs, HMM predicted state distribution) where the latter
must be passed back to 'updateHMM' once r_t is observed.
-}
predictSensors ::
    PredictorBundle ->
    V.Vector Double ->
    HMMFilter ->
    Int ->
    ([(SensorId, SensorOutput)], [Double])
predictSensors pb prices =
    predictSensorsWithInputs pb (featureInputsFromClose prices)

predictSensorsWithInputs ::
    PredictorBundle ->
    FeatureInputs ->
    HMMFilter ->
    Int ->
    ([(SensorId, SensorOutput)], [Double])
predictSensorsWithInputs pb inputs hmmFilt t =
    let fs = pbFeatureSpec pb
        enabled = pbEnabled pb
        prices = fiClose inputs
        useGbdt = predictorEnabled enabled SensorGBT
        useKnn = predictorEnabled enabled SensorKNN
        useDecisionTree = predictorEnabled enabled SensorDecisionTree
        useTcn = predictorEnabled enabled SensorTCN
        useTransformer = predictorEnabled enabled SensorTransformer
        useHmm = predictorEnabled enabled SensorHMM
        useQuantile = predictorEnabled enabled SensorQuantile
        useConformal = predictorEnabled enabled SensorConformal
        needFeatures = useGbdt || useKnn || useDecisionTree || useTransformer || useQuantile || useConformal
        feat =
            if needFeatures
                then featuresAtWithInputsWithMarket fs (pbMarketModel pb) inputs t
                else Nothing
        gbdtReady = gmFeatureDim (pbGBDT pb) > 0
        knnReady = kmFeatureDim (pbKNN pb) > 0
        decisionTreeReady = dmFeatureDim (pbDecisionTree pb) > 0
        quantReady = not (null (lmW (qm50 (pbQuantile pb))))
        transformerReady = trFeatureDim (pbTransformer pb) > 0
        gbdtPred =
            case feat of
                Nothing -> Nothing
                Just x ->
                    if gbdtReady && (useGbdt || useConformal)
                        then Just (predictGBDT (pbGBDT pb) x)
                        else Nothing

        gbdtOut =
            case gbdtPred of
                Just (mu, sig)
                    | useGbdt ->
                        [(SensorGBT, SensorOutput{soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing})]
                _ -> []

        knnOut =
            case feat of
                Nothing -> []
                Just x ->
                    if not useKnn || not knnReady || length x /= kmFeatureDim (pbKNN pb)
                        then []
                        else
                            let (mu, sig) = predictKNN (pbKNN pb) x
                             in [(SensorKNN, SensorOutput{soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing})]

        decisionTreeOut =
            case feat of
                Nothing -> []
                Just x ->
                    if not useDecisionTree || not decisionTreeReady || length x /= dmFeatureDim (pbDecisionTree pb)
                        then []
                        else
                            let (mu, sig) = predictDecisionTree (pbDecisionTree pb) x
                             in [(SensorDecisionTree, SensorOutput{soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing})]

        tcnOut =
            if not useTcn
                then []
                else case predictTCN (pbTCN pb) prices t of
                    Nothing -> []
                    Just (mu, sig) ->
                        [(SensorTCN, SensorOutput{soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing})]

        transformerOut =
            case feat of
                Nothing -> []
                Just x ->
                    if not useTransformer || not transformerReady || length x /= trFeatureDim (pbTransformer pb)
                        then []
                        else
                            let (mu, sig) = predictTransformer (pbTransformer pb) x
                             in [(SensorTransformer, SensorOutput{soMu = mu, soSigma = sig, soRegimes = Nothing, soQuantiles = Nothing, soInterval = Nothing})]

        quantOut =
            case feat of
                Nothing -> []
                Just x ->
                    if not quantReady || not useQuantile
                        then []
                        else case predictQuantiles (pbQuantile pb) x of
                            Nothing -> []
                            Just (q10', q50', q90', _muRaw, sig) ->
                                [
                                    ( SensorQuantile
                                    , SensorOutput
                                        { soMu = q50'
                                        , soSigma = sig
                                        , soRegimes = Nothing
                                        , soQuantiles = Just (Quantiles q10' q50' q90')
                                        , soInterval = Nothing
                                        }
                                    )
                                ]

        conformalOut =
            case gbdtPred of
                Just (mu, _)
                    | useConformal ->
                        let (lo, hi, sig) = predictInterval (pbConformal pb) mu
                         in [
                                ( SensorConformal
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
                     in (
                            [
                                ( SensorHMM
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
     in (gbdtOut ++ knnOut ++ decisionTreeOut ++ tcnOut ++ transformerOut ++ hmmOut ++ quantOut ++ conformalOut, predState)

updateHMM :: PredictorBundle -> [Double] -> Double -> HMMFilter
updateHMM pb predState realizedReturn =
    if predictorEnabled (pbEnabled pb) SensorHMM
        then updatePosterior (pbHMM pb) predState realizedReturn
        else HMMFilter{hfPosterior = predState}
