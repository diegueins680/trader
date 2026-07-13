{-# LANGUAGE OverloadedStrings #-}

module Trader.NeuralGovernor (
    NeuralGovernorConfig (..),
    NeuralGovernorDecision (..),
    NeuralGovernorFeatures (..),
    NeuralGovernorMode (..),
    NeuralGovernorPendingEntry (..),
    NeuralGovernorState (..),
    defaultNeuralGovernorConfig,
    initNeuralGovernorState,
    neuralGovernorDecide,
    neuralGovernorEnsureState,
    neuralGovernorFeatureCount,
    neuralGovernorHoldReason,
    neuralGovernorObserveTrade,
    neuralGovernorOpenBlockReason,
    neuralGovernorReward,
    neuralGovernorSizingMultiplier,
    neuralGovernorTextFeature,
) where

import Data.Aeson (
    FromJSON (..),
    ToJSON (..),
    object,
    withObject,
    (.!=),
    (.:),
    (.:?),
    (.=),
 )
import Data.Char (ord, toLower)
import Data.List (foldl')
import System.Random (mkStdGen, randomRs)

data NeuralGovernorMode
    = NeuralGovernorSizing
    deriving (Eq, Show)

data NeuralGovernorConfig = NeuralGovernorConfig
    { ngcEnabled :: !Bool
    , ngcMode :: !NeuralGovernorMode
    , ngcHiddenSize :: !Int
    , ngcLearningRate :: !Double
    , ngcL2 :: !Double
    , ngcGradientClip :: !Double
    , ngcRewardClip :: !Double
    , ngcLossPenaltyScale :: !Double
    , ngcMinTrades :: !Int
    , ngcOpenScoreFloor :: !Double
    , ngcHoldScoreFloor :: !Double
    , ngcMinMultiplier :: !Double
    , ngcMaxMultiplier :: !Double
    , ngcInfluence :: !Double
    , ngcSeed :: !Int
    }
    deriving (Eq, Show)

data NeuralGovernorFeatures = NeuralGovernorFeatures
    { ngfVolatility :: !(Maybe Double)
    , ngfConfidence :: !(Maybe Double)
    , ngfTrendProbability :: !(Maybe Double)
    , ngfMeanReversionProbability :: !(Maybe Double)
    , ngfHighVolProbability :: !(Maybe Double)
    , ngfDrawdown :: !Double
    , ngfLossStreak :: !Int
    , ngfRollingLoss :: !(Maybe Double)
    , ngfDirection :: !Int
    , ngfBasePositionSize :: !Double
    , ngfMarketGovernorMultiplier :: !Double
    , ngfMarketGovernorBlocked :: !Bool
    , ngfSymbolFeature :: !Double
    , ngfMethodFeature :: !Double
    , ngfIntervalFeature :: !Double
    }
    deriving (Eq, Show)

data NeuralGovernorModel = NeuralGovernorModel
    { ngmInputDim :: !Int
    , ngmHiddenSize :: !Int
    , ngmExamples :: !Int
    , ngmW1 :: [[Double]]
    , ngmB1 :: [Double]
    , ngmW2 :: [Double]
    , ngmB2 :: !Double
    }
    deriving (Eq, Show)

data NeuralGovernorState = NeuralGovernorState
    { ngsModel :: !NeuralGovernorModel
    , ngsLastReward :: !(Maybe Double)
    }
    deriving (Eq, Show)

data NeuralGovernorDecision = NeuralGovernorDecision
    { ngdEnabled :: !Bool
    , ngdMode :: !NeuralGovernorMode
    , ngdReady :: !Bool
    , ngdExamples :: !Int
    , ngdScore :: !Double
    , ngdMultiplier :: !Double
    , ngdOpenBlockReason :: !(Maybe String)
    , ngdHoldReason :: !(Maybe String)
    , ngdReason :: !String
    , ngdWarnings :: ![String]
    }
    deriving (Eq, Show)

data NeuralGovernorPendingEntry = NeuralGovernorPendingEntry
    { ngpeFeatures :: !NeuralGovernorFeatures
    , ngpeDecision :: !NeuralGovernorDecision
    }
    deriving (Eq, Show)

defaultNeuralGovernorConfig :: NeuralGovernorConfig
defaultNeuralGovernorConfig =
    NeuralGovernorConfig
        { ngcEnabled = True
        , ngcMode = NeuralGovernorSizing
        , ngcHiddenSize = 12
        , ngcLearningRate = 0.02
        , ngcL2 = 1e-4
        , ngcGradientClip = 5
        , ngcRewardClip = 0.08
        , ngcLossPenaltyScale = 2.5
        , ngcMinTrades = 8
        , ngcOpenScoreFloor = 0
        , ngcHoldScoreFloor = 0.005
        , ngcMinMultiplier = 0.25
        , ngcMaxMultiplier = 1.15
        , ngcInfluence = 5
        , ngcSeed = 1337
        }

initNeuralGovernorState :: NeuralGovernorConfig -> NeuralGovernorState
initNeuralGovernorState cfg0 =
    let cfg = sanitizeConfig cfg0
     in NeuralGovernorState
            { ngsModel = initModel cfg neuralGovernorFeatureCount
            , ngsLastReward = Nothing
            }

neuralGovernorEnsureState :: NeuralGovernorConfig -> NeuralGovernorState -> NeuralGovernorState
neuralGovernorEnsureState cfg0 state =
    let cfg = sanitizeConfig cfg0
        model = ngsModel state
     in if ngmInputDim model == neuralGovernorFeatureCount
            && ngmHiddenSize model == ngcHiddenSize cfg
            && modelFinite model
            then state
            else initNeuralGovernorState cfg

neuralGovernorDecide ::
    NeuralGovernorConfig ->
    NeuralGovernorState ->
    NeuralGovernorFeatures ->
    NeuralGovernorDecision
neuralGovernorDecide cfg0 state0 features =
    let cfg = sanitizeConfig cfg0
        state = neuralGovernorEnsureState cfg state0
        model = ngsModel state
        examples = ngmExamples model
        score = clamp (-1) 1 (predictScore model (featureVector features))
        ready = ngcEnabled cfg && examples >= ngcMinTrades cfg
        multiplierRaw = 1 + ngcInfluence cfg * score
        multiplier = clamp (ngcMinMultiplier cfg) (ngcMaxMultiplier cfg) multiplierRaw
        openBlockReason =
            if ready && score <= ngcOpenScoreFloor cfg
                then Just neuralGovernorAvoidOpenReason
                else Nothing
        holdReason =
            if ready && score >= ngcHoldScoreFloor cfg
                then Just neuralGovernorPreferHoldReason
                else Nothing
     in if not (ngcEnabled cfg)
            then mkDecision cfg False examples score 1 Nothing Nothing "NEURAL_GOVERNOR_DISABLED" []
            else
                if ngfMarketGovernorBlocked features
                    then mkDecision cfg False examples score 1 Nothing Nothing "NEURAL_GOVERNOR_HARD_GATE" []
                    else
                        if not ready
                            then mkDecision cfg False examples score 1 Nothing Nothing "NEURAL_GOVERNOR_WARMUP" []
                            else mkDecision cfg True examples score multiplier openBlockReason holdReason "NEURAL_GOVERNOR_SIZING" []

neuralGovernorObserveTrade ::
    NeuralGovernorConfig ->
    NeuralGovernorState ->
    NeuralGovernorPendingEntry ->
    Double ->
    NeuralGovernorState
neuralGovernorObserveTrade cfg0 state0 pending realizedReturn =
    let cfg = sanitizeConfig cfg0
        state = neuralGovernorEnsureState cfg state0
     in case neuralGovernorReward cfg realizedReturn of
            Nothing -> state
            Just reward ->
                state
                    { ngsModel = trainOne cfg (ngsModel state) (featureVector (ngpeFeatures pending)) reward
                    , ngsLastReward = Just reward
                    }

neuralGovernorReward :: NeuralGovernorConfig -> Double -> Maybe Double
neuralGovernorReward cfg0 realizedReturn
    | not (isFiniteDouble realizedReturn) = Nothing
    | realizedReturn == 0 = Just 0
    | otherwise =
        let cfg = sanitizeConfig cfg0
            raw =
                if realizedReturn < 0
                    then realizedReturn * ngcLossPenaltyScale cfg
                    else realizedReturn
         in Just (clamp (negate (ngcRewardClip cfg)) (ngcRewardClip cfg) raw)

neuralGovernorSizingMultiplier :: NeuralGovernorDecision -> Double
neuralGovernorSizingMultiplier decision =
    if ngdEnabled decision && ngdReady decision
        then clamp 0 10 (ngdMultiplier decision)
        else 1

neuralGovernorOpenBlockReason :: NeuralGovernorDecision -> Maybe String
neuralGovernorOpenBlockReason decision =
    if ngdEnabled decision && ngdReady decision
        then ngdOpenBlockReason decision
        else Nothing

neuralGovernorHoldReason :: NeuralGovernorDecision -> Maybe String
neuralGovernorHoldReason decision =
    if ngdEnabled decision && ngdReady decision
        then ngdHoldReason decision
        else Nothing

neuralGovernorFeatureCount :: Int
neuralGovernorFeatureCount =
    length
        ( featureVector
            NeuralGovernorFeatures
                { ngfVolatility = Nothing
                , ngfConfidence = Nothing
                , ngfTrendProbability = Nothing
                , ngfMeanReversionProbability = Nothing
                , ngfHighVolProbability = Nothing
                , ngfDrawdown = 0
                , ngfLossStreak = 0
                , ngfRollingLoss = Nothing
                , ngfDirection = 0
                , ngfBasePositionSize = 0
                , ngfMarketGovernorMultiplier = 1
                , ngfMarketGovernorBlocked = False
                , ngfSymbolFeature = 0
                , ngfMethodFeature = 0
                , ngfIntervalFeature = 0
                }
        )

neuralGovernorTextFeature :: String -> Double
neuralGovernorTextFeature raw =
    let h =
            foldl'
                (\acc c -> (acc * 16777619 + toInteger (ord c)) `mod` 1000003)
                2166136261
                (map toLower raw)
        bucket = fromInteger (h `mod` 2001) :: Double
     in bucket / 1000 - 1

featureVector :: NeuralGovernorFeatures -> [Double]
featureVector f =
    [ maybeScaled 0.5 (1 / 2) (ngfVolatility f)
    , maybe01 (ngfConfidence f)
    , maybe01 (ngfTrendProbability f)
    , maybe01 (ngfMeanReversionProbability f)
    , maybe01 (ngfHighVolProbability f)
    , clamp 0 1 (finiteOr 0 (ngfDrawdown f) * 5)
    , clamp 0 1 (fromIntegral (max 0 (ngfLossStreak f)) / 5)
    , maybeScaled 0 (10 :: Double) (ngfRollingLoss f)
    , clamp (-1) 1 (fromIntegral (signum (ngfDirection f)))
    , clamp 0 2 (finiteOr 0 (ngfBasePositionSize f))
    , clamp 0 1 (finiteOr 1 (ngfMarketGovernorMultiplier f))
    , if ngfMarketGovernorBlocked f then 1 else 0
    , clamp (-1) 1 (finiteOr 0 (ngfSymbolFeature f))
    , clamp (-1) 1 (finiteOr 0 (ngfMethodFeature f))
    , clamp (-1) 1 (finiteOr 0 (ngfIntervalFeature f))
    ]

maybe01 :: Maybe Double -> Double
maybe01 = maybe 0.5 (clamp 0 1)

maybeScaled :: Double -> Double -> Maybe Double -> Double
maybeScaled fallback scaleV =
    maybe fallback (clamp 0 1 . (* scaleV))

neuralGovernorAvoidOpenReason :: String
neuralGovernorAvoidOpenReason = "NEURAL_GOVERNOR_AVOID_OPEN"

neuralGovernorPreferHoldReason :: String
neuralGovernorPreferHoldReason = "NEURAL_GOVERNOR_PREFER_HOLD"

mkDecision ::
    NeuralGovernorConfig ->
    Bool ->
    Int ->
    Double ->
    Double ->
    Maybe String ->
    Maybe String ->
    String ->
    [String] ->
    NeuralGovernorDecision
mkDecision cfg ready examples score multiplier openBlockReason holdReason reason warnings =
    NeuralGovernorDecision
        { ngdEnabled = ngcEnabled cfg
        , ngdMode = ngcMode cfg
        , ngdReady = ready
        , ngdExamples = max 0 examples
        , ngdScore = finiteOr 0 score
        , ngdMultiplier = clamp 0 10 multiplier
        , ngdOpenBlockReason = openBlockReason
        , ngdHoldReason = holdReason
        , ngdReason = reason
        , ngdWarnings = warnings
        }

sanitizeConfig :: NeuralGovernorConfig -> NeuralGovernorConfig
sanitizeConfig cfg =
    let minMult = finitePositive 0.25 (ngcMinMultiplier cfg)
        maxMult = max minMult (finitePositive 1.15 (ngcMaxMultiplier cfg))
        openScoreFloor = clamp (-1) 1 (finiteOr 0 (ngcOpenScoreFloor cfg))
        holdScoreFloor = clamp openScoreFloor 1 (finiteOr 0.005 (ngcHoldScoreFloor cfg))
     in cfg
            { ngcMode = NeuralGovernorSizing
            , ngcHiddenSize = max 2 (ngcHiddenSize cfg)
            , ngcLearningRate = finitePositive 0.02 (ngcLearningRate cfg)
            , ngcL2 = finiteNonNegative 1e-4 (ngcL2 cfg)
            , ngcGradientClip = finitePositive 5 (ngcGradientClip cfg)
            , ngcRewardClip = finitePositive 0.08 (ngcRewardClip cfg)
            , ngcLossPenaltyScale = finiteNonNegative 2.5 (ngcLossPenaltyScale cfg)
            , ngcMinTrades = max 0 (ngcMinTrades cfg)
            , ngcOpenScoreFloor = openScoreFloor
            , ngcHoldScoreFloor = holdScoreFloor
            , ngcMinMultiplier = minMult
            , ngcMaxMultiplier = maxMult
            , ngcInfluence = finiteNonNegative 5 (ngcInfluence cfg)
            }

initModel :: NeuralGovernorConfig -> Int -> NeuralGovernorModel
initModel cfg inputDim0 =
    let inputDim = max 1 inputDim0
        hidden = max 2 (ngcHiddenSize cfg)
        gen = mkStdGen (ngcSeed cfg + inputDim * 101 + hidden * 17)
        vals = randomRs (-1.0, 1.0) gen
        scale fanIn fanOut = sqrt (6 / fromIntegral (max 1 (fanIn + fanOut)))
        (w1Vals, rest) = splitAt (hidden * inputDim) vals
        (w2Vals, _) = splitAt hidden rest
     in NeuralGovernorModel
            { ngmInputDim = inputDim
            , ngmHiddenSize = hidden
            , ngmExamples = 0
            , ngmW1 = scaleMatrix (scale inputDim hidden) (chunksOf inputDim w1Vals)
            , ngmB1 = replicate hidden 0
            , ngmW2 = map (* scale hidden 1) w2Vals
            , ngmB2 = 0
            }

predictScore :: NeuralGovernorModel -> [Double] -> Double
predictScore model x0 =
    let x = sanitizeFeatures (ngmInputDim model) x0
        (_, yRaw) = forwardRaw model x
     in finiteOr 0 yRaw

forwardRaw :: NeuralGovernorModel -> [Double] -> ([Double], Double)
forwardRaw model x =
    let a1 = zipWith (\row b -> tanh (dot row x + b)) (ngmW1 model) (ngmB1 model)
        yRaw = dot (ngmW2 model) a1 + ngmB2 model
     in (a1, finiteOr 0 yRaw)

trainOne :: NeuralGovernorConfig -> NeuralGovernorModel -> [Double] -> Double -> NeuralGovernorModel
trainOne cfg model feats target0 =
    let lr = ngcLearningRate cfg
        l2 = ngcL2 cfg
        x = sanitizeFeatures (ngmInputDim model) feats
        (a1, yRaw) = forwardRaw model x
        target = clamp (negate (ngcRewardClip cfg)) (ngcRewardClip cfg) target0
        err = clipGrad cfg (yRaw - target)
        delta1 =
            [ clipGrad cfg (err * w2 * tanhDeriv a)
            | (w2, a) <- zip (ngmW2 model) a1
            ]
        w2' =
            [ finiteOr old (old - lr * clipGrad cfg (err * a + l2 * old))
            | (old, a) <- zip (ngmW2 model) a1
            ]
        b2' = finiteOr (ngmB2 model) (ngmB2 model - lr * clipGrad cfg err)
        w1' =
            [ [ finiteOr old (old - lr * clipGrad cfg (d1 * xi + l2 * old))
              | (old, xi) <- zip row x
              ]
            | (row, d1) <- zip (ngmW1 model) delta1
            ]
        b1' =
            [ finiteOr old (old - lr * clipGrad cfg d1)
            | (old, d1) <- zip (ngmB1 model) delta1
            ]
     in model
            { ngmExamples = ngmExamples model + 1
            , ngmW1 = w1'
            , ngmB1 = b1'
            , ngmW2 = w2'
            , ngmB2 = b2'
            }

modelFinite :: NeuralGovernorModel -> Bool
modelFinite model =
    ngmInputDim model > 0
        && ngmHiddenSize model > 0
        && length (ngmW1 model) == ngmHiddenSize model
        && all ((== ngmInputDim model) . length) (ngmW1 model)
        && length (ngmB1 model) == ngmHiddenSize model
        && length (ngmW2 model) == ngmHiddenSize model
        && all isFiniteDouble (concat (ngmW1 model) ++ ngmB1 model ++ ngmW2 model ++ [ngmB2 model])

sanitizeFeatures :: Int -> [Double] -> [Double]
sanitizeFeatures inputDim feats =
    take inputDim (map (clamp (-8) 8 . finiteOr 0) feats ++ repeat 0)

clipGrad :: NeuralGovernorConfig -> Double -> Double
clipGrad cfg = clamp (negate (ngcGradientClip cfg)) (ngcGradientClip cfg)

tanhDeriv :: Double -> Double
tanhDeriv a = max 0 (1 - a * a)

dot :: [Double] -> [Double] -> Double
dot xs ys = sum (zipWith (*) xs ys)

chunksOf :: Int -> [a] -> [[a]]
chunksOf n xs
    | n <= 0 = []
    | otherwise =
        case splitAt n xs of
            ([], _) -> []
            (chunk, rest) -> chunk : chunksOf n rest

scaleMatrix :: Double -> [[Double]] -> [[Double]]
scaleMatrix scaleV = map (map (* scaleV))

finitePositive :: Double -> Double -> Double
finitePositive fallback x
    | not (isFiniteDouble x) || x <= 0 = fallback
    | otherwise = x

finiteNonNegative :: Double -> Double -> Double
finiteNonNegative fallback x
    | not (isFiniteDouble x) || x < 0 = fallback
    | otherwise = x

finiteOr :: Double -> Double -> Double
finiteOr fallback x =
    if isFiniteDouble x then x else fallback

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

clamp :: Double -> Double -> Double -> Double
clamp lo hi x
    | not (isFiniteDouble x) = lo
    | x < lo = lo
    | x > hi = hi
    | otherwise = x

modeCode :: NeuralGovernorMode -> String
modeCode mode =
    case mode of
        NeuralGovernorSizing -> "sizing"

parseMode :: String -> NeuralGovernorMode
parseMode _ = NeuralGovernorSizing

instance ToJSON NeuralGovernorMode where
    toJSON = toJSON . modeCode

instance FromJSON NeuralGovernorMode where
    parseJSON value = parseMode <$> parseJSON value

instance ToJSON NeuralGovernorConfig where
    toJSON cfg =
        object
            [ "enabled" .= ngcEnabled cfg
            , "mode" .= ngcMode cfg
            , "hiddenSize" .= ngcHiddenSize cfg
            , "learningRate" .= ngcLearningRate cfg
            , "l2" .= ngcL2 cfg
            , "gradientClip" .= ngcGradientClip cfg
            , "rewardClip" .= ngcRewardClip cfg
            , "lossPenaltyScale" .= ngcLossPenaltyScale cfg
            , "minTrades" .= ngcMinTrades cfg
            , "openScoreFloor" .= ngcOpenScoreFloor cfg
            , "holdScoreFloor" .= ngcHoldScoreFloor cfg
            , "minMultiplier" .= ngcMinMultiplier cfg
            , "maxMultiplier" .= ngcMaxMultiplier cfg
            , "influence" .= ngcInfluence cfg
            , "seed" .= ngcSeed cfg
            ]

instance FromJSON NeuralGovernorConfig where
    parseJSON =
        withObject "NeuralGovernorConfig" $ \o -> do
            let defaults = defaultNeuralGovernorConfig
            NeuralGovernorConfig
                <$> o .:? "enabled" .!= ngcEnabled defaults
                <*> o .:? "mode" .!= ngcMode defaults
                <*> o .:? "hiddenSize" .!= ngcHiddenSize defaults
                <*> o .:? "learningRate" .!= ngcLearningRate defaults
                <*> o .:? "l2" .!= ngcL2 defaults
                <*> o .:? "gradientClip" .!= ngcGradientClip defaults
                <*> o .:? "rewardClip" .!= ngcRewardClip defaults
                <*> o .:? "lossPenaltyScale" .!= ngcLossPenaltyScale defaults
                <*> o .:? "minTrades" .!= ngcMinTrades defaults
                <*> o .:? "openScoreFloor" .!= ngcOpenScoreFloor defaults
                <*> o .:? "holdScoreFloor" .!= ngcHoldScoreFloor defaults
                <*> o .:? "minMultiplier" .!= ngcMinMultiplier defaults
                <*> o .:? "maxMultiplier" .!= ngcMaxMultiplier defaults
                <*> o .:? "influence" .!= ngcInfluence defaults
                <*> o .:? "seed" .!= ngcSeed defaults

instance ToJSON NeuralGovernorFeatures where
    toJSON f =
        object
            [ "volatility" .= ngfVolatility f
            , "confidence" .= ngfConfidence f
            , "trendProbability" .= ngfTrendProbability f
            , "meanReversionProbability" .= ngfMeanReversionProbability f
            , "highVolProbability" .= ngfHighVolProbability f
            , "drawdown" .= ngfDrawdown f
            , "lossStreak" .= ngfLossStreak f
            , "rollingLoss" .= ngfRollingLoss f
            , "direction" .= ngfDirection f
            , "basePositionSize" .= ngfBasePositionSize f
            , "marketGovernorMultiplier" .= ngfMarketGovernorMultiplier f
            , "marketGovernorBlocked" .= ngfMarketGovernorBlocked f
            , "symbolFeature" .= ngfSymbolFeature f
            , "methodFeature" .= ngfMethodFeature f
            , "intervalFeature" .= ngfIntervalFeature f
            ]

instance FromJSON NeuralGovernorFeatures where
    parseJSON =
        withObject "NeuralGovernorFeatures" $ \o ->
            NeuralGovernorFeatures
                <$> o .:? "volatility"
                <*> o .:? "confidence"
                <*> o .:? "trendProbability"
                <*> o .:? "meanReversionProbability"
                <*> o .:? "highVolProbability"
                <*> o .: "drawdown"
                <*> o .: "lossStreak"
                <*> o .:? "rollingLoss"
                <*> o .: "direction"
                <*> o .: "basePositionSize"
                <*> o .: "marketGovernorMultiplier"
                <*> o .: "marketGovernorBlocked"
                <*> o .: "symbolFeature"
                <*> o .: "methodFeature"
                <*> o .: "intervalFeature"

instance ToJSON NeuralGovernorModel where
    toJSON model =
        object
            [ "inputDim" .= ngmInputDim model
            , "hiddenSize" .= ngmHiddenSize model
            , "examples" .= ngmExamples model
            , "w1" .= ngmW1 model
            , "b1" .= ngmB1 model
            , "w2" .= ngmW2 model
            , "b2" .= ngmB2 model
            ]

instance FromJSON NeuralGovernorModel where
    parseJSON =
        withObject "NeuralGovernorModel" $ \o ->
            NeuralGovernorModel
                <$> o .: "inputDim"
                <*> o .: "hiddenSize"
                <*> o .: "examples"
                <*> o .: "w1"
                <*> o .: "b1"
                <*> o .: "w2"
                <*> o .: "b2"

instance ToJSON NeuralGovernorState where
    toJSON state =
        object
            [ "model" .= ngsModel state
            , "lastReward" .= ngsLastReward state
            ]

instance FromJSON NeuralGovernorState where
    parseJSON =
        withObject "NeuralGovernorState" $ \o ->
            NeuralGovernorState
                <$> o .: "model"
                <*> o .:? "lastReward"

instance ToJSON NeuralGovernorDecision where
    toJSON d =
        object
            [ "enabled" .= ngdEnabled d
            , "mode" .= ngdMode d
            , "ready" .= ngdReady d
            , "examples" .= ngdExamples d
            , "score" .= ngdScore d
            , "multiplier" .= ngdMultiplier d
            , "openBlockReason" .= ngdOpenBlockReason d
            , "holdReason" .= ngdHoldReason d
            , "reason" .= ngdReason d
            , "warnings" .= ngdWarnings d
            ]

instance FromJSON NeuralGovernorDecision where
    parseJSON =
        withObject "NeuralGovernorDecision" $ \o ->
            NeuralGovernorDecision
                <$> o .: "enabled"
                <*> o .: "mode"
                <*> o .: "ready"
                <*> o .: "examples"
                <*> o .: "score"
                <*> o .: "multiplier"
                <*> o .:? "openBlockReason"
                <*> o .:? "holdReason"
                <*> o .: "reason"
                <*> o .:? "warnings" .!= []

instance ToJSON NeuralGovernorPendingEntry where
    toJSON pending =
        object
            [ "features" .= ngpeFeatures pending
            , "decision" .= ngpeDecision pending
            ]

instance FromJSON NeuralGovernorPendingEntry where
    parseJSON =
        withObject "NeuralGovernorPendingEntry" $ \o ->
            NeuralGovernorPendingEntry
                <$> o .: "features"
                <*> o .: "decision"
