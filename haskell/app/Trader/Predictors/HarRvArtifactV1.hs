{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.HarRvArtifactV1 (
    HarRvFitRequestV1 (..),
    HarRvArtifactV1,
    HarRvEstimateV1 (..),
    harRvArtifactSchemaIdV1,
    harRvArtifactSchemaVersionV1,
    harRvCompatibilityVersionV1,
    harRvSemanticModelIdV1,
    harRvFeatureNamesV1,
    harRvRegisteredSymbolsV1,
    harRvRidgeLambdaV1,
    harRvTrainingEvidenceSha256V1,
    fitHarRvArtifactV1,
    predictHarRvV1,
    harRvRiskScaleV1,
    scaleChampionExposureHarRvV1,
    encodeHarRvArtifactV1,
    decodeHarRvArtifactV1,
    hra1Request,
    hra1TrainingRowCount,
    hra1ObservedTrainingRowCount,
    hra1FeatureMeans,
    hra1FeatureScales,
    hra1Intercept,
    hra1Coefficients,
    hra1ResidualLogVariance,
    hra1TrainingMse,
    hra1TrainingMedianForecastVolatility,
    hra1PayloadSha256,
) where

import Control.Monad (guard, unless, when)
import Crypto.Hash (Digest, SHA256, hash)
import Data.Aeson (Value, object, withObject, (.:), (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.Aeson.Types as AesonTypes
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum, isAscii, isControl, isDigit)
import Data.Int (Int64)
import Data.List (elemIndex, foldl', nub, sort, sortOn)
import qualified Data.Vector as V

import Trader.Predictors.FeatureSchema (
    FeatureRowV2,
    featureAvailabilitySchemaIdV2,
    frv2AvailabilityTimesMs,
    frv2Available,
    frv2DecisionTimeMs,
    frv2EventTimesMs,
    frv2Names,
    frv2Required,
    frv2SchemaId,
    frv2Values,
 )
import Trader.Predictors.OhlcvInputsV2 (
    CompleteOhlcvInputsV2,
    completeOhlcvFieldNamesV2,
    completeOhlcvGridV2,
    completeOhlcvRowsV2,
    completeOhlcvSchemaIdV2,
    completeOhlcvSchemaSignatureV2,
    completeOhlcvSchemaVersionV2,
    completeOhlcvScopeV2,
 )

-- | Immutable inputs that bind one fold-local fit to its research evidence.
data HarRvFitRequestV1 = HarRvFitRequestV1
    { hrr1RegistrationId :: !String
    , hrr1CodeCommit :: !String
    , hrr1TrainingDataSha256 :: !String
    , hrr1TrainingEvidenceSha256 :: !String
    , hrr1SourceManifestSha256 :: !String
    , hrr1SplitManifestSha256 :: !String
    , hrr1AcademicOrigins :: ![String]
    , hrr1Symbol :: !String
    , hrr1IntervalMs :: !Int64
    , hrr1HorizonBars :: !Int
    , hrr1TrainingStartEventTimeMs :: !Int64
    , hrr1TrainingEndEventTimeMs :: !Int64
    , hrr1ValidationStartEventTimeMs :: !Int64
    , hrr1ValidationEndEventTimeMs :: !Int64
    , hrr1FinalHoldoutStartEventTimeMs :: !Int64
    , hrr1PurgeBars :: !Int
    , hrr1EmbargoBars :: !Int
    , hrr1FitAvailableAtMs :: !Int64
    , hrr1CreatedAtMs :: !Int64
    , hrr1RandomSeed :: !Int
    , hrr1RuntimeVersions :: ![String]
    , hrr1CostModelId :: !String
    }
    deriving (Eq, Show)

data HarRvArtifactV1 = HarRvArtifactV1
    { hra1Request :: !HarRvFitRequestV1
    , hra1TrainingRowCount :: !Int
    , hra1ObservedTrainingRowCount :: !Int
    , hra1FeatureMeans :: ![Double]
    , hra1FeatureScales :: ![Double]
    , hra1Intercept :: !Double
    , hra1Coefficients :: ![Double]
    , hra1ResidualLogVariance :: !Double
    , hra1TrainingMse :: !Double
    , hra1TrainingMedianForecastVolatility :: !Double
    , hra1PayloadSha256 :: !String
    }
    deriving (Eq, Show)

-- | Non-directional distribution summary. It cannot express a trade side.
data HarRvEstimateV1 = HarRvEstimateV1
    { hre1EventTimeMs :: !Int64
    , hre1DecisionTimeMs :: !Int64
    , hre1ExpectedLogRealizedVariance :: !Double
    , hre1ForecastVolatility :: !Double
    , hre1Lower95Volatility :: !Double
    , hre1Upper95Volatility :: !Double
    , hre1RiskScale :: !Double
    }
    deriving (Eq, Show)

data HarRvObservationV1 = HarRvObservationV1
    { hro1EventTimeMs :: !Int64
    , hro1DecisionTimeMs :: !Int64
    , hro1TargetEventTimeMs :: !Int64
    , hro1TargetAvailabilityTimeMs :: !Int64
    , hro1Features :: !(Maybe [Double])
    , hro1TargetLogRealizedVariance :: !(Maybe Double)
    }
    deriving (Eq, Show)

harRvArtifactSchemaIdV1 :: String
harRvArtifactSchemaIdV1 = "har_rv_risk_gate_artifact_v1"

harRvArtifactSchemaVersionV1 :: Int
harRvArtifactSchemaVersionV1 = 1

harRvCompatibilityVersionV1 :: Int
harRvCompatibilityVersionV1 = 1

harRvSemanticModelIdV1 :: String
harRvSemanticModelIdV1 = "bar_har_rv_ridge_risk_gate_v1"

harRvModelFamilyV1 :: String
harRvModelFamilyV1 = "har_log_realized_variance_ridge"

harRvTargetIdV1 :: String
harRvTargetIdV1 = "log_forward_sum_squared_bar_log_returns"

harRvFitMethodV1 :: String
harRvFitMethodV1 = "per_symbol_expanding_window_standardized_ridge_v1"

harRvPromotionStateV1 :: String
harRvPromotionStateV1 = "offline_research_only"

harRvValidationMetricsStateV1 :: String
harRvValidationMetricsStateV1 = "not_evaluated"

harRvFinalHoldoutStateV1 :: String
harRvFinalHoldoutStateV1 = "untouched"

harRvFeatureNamesV1 :: [String]
harRvFeatureNamesV1 =
    [ "log_rv.trailing_1_bar"
    , "log_rv.trailing_6_bars"
    , "log_rv.trailing_24_bars"
    ]

harRvFeatureSignatureV1 :: String
harRvFeatureSignatureV1 = harRvSemanticModelIdV1 ++ "|" ++ concatWithComma harRvFeatureNamesV1

harRvRidgeLambdaV1 :: Double
harRvRidgeLambdaV1 = 1.0e-6

harRvMinimumObservationsV1 :: Int
harRvMinimumObservationsV1 = 5

harRvResidualVarianceFloorV1 :: Double
harRvResidualVarianceFloorV1 = 1.0e-12

registeredDatasetStartMsV1 :: Int64
registeredDatasetStartMsV1 = 1800489600000

registeredDevelopmentEndMsV1 :: Int64
registeredDevelopmentEndMsV1 = 1821481200000

registeredFinalHoldoutStartMsV1 :: Int64
registeredFinalHoldoutStartMsV1 = 1821484800000

registeredAcademicOriginsV1 :: [String]
registeredAcademicOriginsV1 =
    [ "https://doi.org/10.1111/1468-0262.00418"
    , "https://doi.org/10.1093/jjfinec/nbp001"
    , "https://doi.org/10.1016/0304-4076(86)90063-1"
    ]

harRvRegisteredSymbolsV1 :: [String]
harRvRegisteredSymbolsV1 =
    [ "BTCUSDT"
    , "ETHUSDT"
    , "SOLUSDT"
    , "BNBUSDT"
    , "XRPUSDT"
    , "AVAXUSDT"
    , "UNIUSDT"
    , "SUIUSDT"
    , "ETCUSDT"
    , "ADAUSDT"
    ]

{- | Hash the exact ordered complete-OHLCV slice that can affect this fit.
The slice includes the 24-return lookback and every forward target bar.
-}
harRvTrainingEvidenceSha256V1 :: HarRvFitRequestV1 -> CompleteOhlcvInputsV2 -> Maybe String
harRvTrainingEvidenceSha256V1 request inputs = do
    either (const Nothing) Just (validateRequest request)
    (startIndex, endIndex) <- validateInputCoverage request inputs
    let evidenceStartIndex = startIndex - 24
        evidenceEndIndex = endIndex + hrr1HorizonBars request
        rows = completeOhlcvRowsV2 inputs
        (openTimes, _, _, intervalMs) = completeOhlcvGridV2 inputs
        selectedRows = take (evidenceEndIndex - evidenceStartIndex + 1) (drop evidenceStartIndex rows)
        selectedOpenTimes = V.toList (V.slice evidenceStartIndex (evidenceEndIndex - evidenceStartIndex + 1) openTimes)
    guard (length selectedRows == length selectedOpenTimes)
    pure
        ( digestValue
            ( object
                [ "schemaId" .= ("har_rv_training_evidence_v1" :: String)
                , "schemaVersion" .= (1 :: Int)
                , "scope" .= completeOhlcvScopeV2 inputs
                , "intervalMs" .= intervalMs
                , "openTimesMs" .= selectedOpenTimes
                , "rows" .= map featureRowValue selectedRows
                ]
            )
        )

fitHarRvArtifactV1 :: HarRvFitRequestV1 -> CompleteOhlcvInputsV2 -> Either String HarRvArtifactV1
fitHarRvArtifactV1 request inputs = do
    validateRequest request
    evidenceDigest <- maybe (Left "HAR-RV training input does not cover the registered causal fold") Right (harRvTrainingEvidenceSha256V1 request inputs)
    unless (evidenceDigest == hrr1TrainingEvidenceSha256 request) (Left "HAR-RV training evidence does not match its provenance digest")
    observations <- maybe (Left "HAR-RV training observations are unavailable or malformed") Right (trainingObservations request inputs)
    let usable = [(features, target) | observation <- observations, Just features <- [hro1Features observation], Just target <- [hro1TargetLogRealizedVariance observation]]
    (means, scales, intercept, coefficients, residualVariance, trainingMse, medianForecastVolatility) <-
        maybe (Left "HAR-RV ridge fit is degenerate or non-finite") Right (fitRidge usable)
    let artifactWithoutDigest =
            HarRvArtifactV1
                { hra1Request = request
                , hra1TrainingRowCount = length observations
                , hra1ObservedTrainingRowCount = length usable
                , hra1FeatureMeans = means
                , hra1FeatureScales = scales
                , hra1Intercept = intercept
                , hra1Coefficients = coefficients
                , hra1ResidualLogVariance = residualVariance
                , hra1TrainingMse = trainingMse
                , hra1TrainingMedianForecastVolatility = medianForecastVolatility
                , hra1PayloadSha256 = ""
                }
        artifact = artifactWithoutDigest{hra1PayloadSha256 = payloadDigest artifactWithoutDigest}
    validateArtifact artifact
    pure artifact

-- | Forecast only inside the request's development-validation window.
predictHarRvV1 :: HarRvArtifactV1 -> CompleteOhlcvInputsV2 -> Int64 -> Maybe HarRvEstimateV1
predictHarRvV1 artifact inputs eventTime = do
    either (const Nothing) Just (validateArtifact artifact)
    let request = hra1Request artifact
        intervalMs = hrr1IntervalMs request
        horizonBars = hrr1HorizonBars request
    guard (eventTime >= hrr1ValidationStartEventTimeMs request)
    guard (eventTime <= hrr1ValidationEndEventTimeMs request)
    guard (eventTime < hrr1FinalHoldoutStartEventTimeMs request)
    guard (gridAligned (hrr1ValidationStartEventTimeMs request) intervalMs eventTime)
    guard (toInteger eventTime + toInteger horizonBars * toInteger intervalMs < toInteger (hrr1FinalHoldoutStartEventTimeMs request))
    validateInputScope request inputs
    index <- eventIndex inputs eventTime
    (decisionTime, features) <- featureAt inputs index
    let expectedLogVariance = predictStandardized artifact features
        residualStd = sqrt (hra1ResidualLogVariance artifact)
        forecastVolatility = exp (0.5 * expectedLogVariance)
        lowerVolatility = exp (0.5 * (expectedLogVariance - 1.96 * residualStd))
        upperVolatility = exp (0.5 * (expectedLogVariance + 1.96 * residualStd))
        scale = min 1 (hra1TrainingMedianForecastVolatility artifact / forecastVolatility)
    guard (all finite [expectedLogVariance, residualStd, forecastVolatility, lowerVolatility, upperVolatility, scale])
    guard (forecastVolatility > 0 && lowerVolatility > 0 && upperVolatility >= lowerVolatility)
    guard (scale >= 0 && scale <= 1)
    pure
        HarRvEstimateV1
            { hre1EventTimeMs = eventTime
            , hre1DecisionTimeMs = decisionTime
            , hre1ExpectedLogRealizedVariance = expectedLogVariance
            , hre1ForecastVolatility = forecastVolatility
            , hre1Lower95Volatility = lowerVolatility
            , hre1Upper95Volatility = upperVolatility
            , hre1RiskScale = scale
            }

-- | Fail-closed scalar used by an offline evaluator. It never exceeds one.
harRvRiskScaleV1 :: HarRvArtifactV1 -> CompleteOhlcvInputsV2 -> Int64 -> Double
harRvRiskScaleV1 artifact inputs eventTime = maybe 0 hre1RiskScale (predictHarRvV1 artifact inputs eventTime)

{- | Preserve the champion's direction while only reducing absolute exposure.
Invalid champion exposure or invalid model evidence produces neutral exposure.
-}
scaleChampionExposureHarRvV1 :: HarRvArtifactV1 -> CompleteOhlcvInputsV2 -> Int64 -> Double -> Double
scaleChampionExposureHarRvV1 artifact inputs eventTime championExposure
    | not (finite championExposure) = 0
    | otherwise = championExposure * harRvRiskScaleV1 artifact inputs eventTime

encodeHarRvArtifactV1 :: HarRvArtifactV1 -> BL.ByteString
encodeHarRvArtifactV1 = Aeson.encode . artifactValue

decodeHarRvArtifactV1 :: BL.ByteString -> Either String HarRvArtifactV1
decodeHarRvArtifactV1 bytes = do
    value <- Aeson.eitherDecode bytes
    AesonTypes.parseEither parseArtifact value

trainingObservations :: HarRvFitRequestV1 -> CompleteOhlcvInputsV2 -> Maybe [HarRvObservationV1]
trainingObservations request inputs = do
    (startIndex, endIndex) <- validateInputCoverage request inputs
    traverse build [startIndex .. endIndex]
  where
    build index = do
        currentRow <- atMay (completeOhlcvRowsV2 inputs) index
        targetRow <- atMay (completeOhlcvRowsV2 inputs) (index + hrr1HorizonBars request)
        (eventTime, _, _) <- closeWitness currentRow
        (targetEventTime, targetAvailabilityTime, _) <- closeWitness targetRow
        guard (targetAvailabilityTime <= hrr1FitAvailableAtMs request)
        let expectedTargetEvent = toInteger eventTime + toInteger (hrr1HorizonBars request) * toInteger (hrr1IntervalMs request)
        guard (toInteger targetEventTime == expectedTargetEvent)
        pure
            HarRvObservationV1
                { hro1EventTimeMs = eventTime
                , hro1DecisionTimeMs = frv2DecisionTimeMs currentRow
                , hro1TargetEventTimeMs = targetEventTime
                , hro1TargetAvailabilityTimeMs = targetAvailabilityTime
                , hro1Features = snd <$> featureAt inputs index
                , hro1TargetLogRealizedVariance = targetAt inputs index (hrr1HorizonBars request)
                }

validateInputCoverage :: HarRvFitRequestV1 -> CompleteOhlcvInputsV2 -> Maybe (Int, Int)
validateInputCoverage request inputs = do
    validateInputScope request inputs
    startIndex <- eventIndex inputs (hrr1TrainingStartEventTimeMs request)
    endIndex <- eventIndex inputs (hrr1TrainingEndEventTimeMs request)
    guard (startIndex >= 24)
    guard (endIndex >= startIndex)
    guard (endIndex + hrr1HorizonBars request < length (completeOhlcvRowsV2 inputs))
    guard (endIndex - startIndex + 1 == expectedTrainingRowCount request)
    pure (startIndex, endIndex)

validateInputScope :: HarRvFitRequestV1 -> CompleteOhlcvInputsV2 -> Maybe ()
validateInputScope request inputs = do
    guard (completeOhlcvScopeV2 inputs == hrr1Symbol request)
    let (openTimes, decisionTimes, availabilityTimes, intervalMs) = completeOhlcvGridV2 inputs
        rows = completeOhlcvRowsV2 inputs
    guard (intervalMs == hrr1IntervalMs request)
    guard (V.length openTimes == length rows)
    guard (V.length decisionTimes == length rows)
    guard (V.length availabilityTimes == length rows)
    guard (all validCompleteOhlcvRow rows)

eventIndex :: CompleteOhlcvInputsV2 -> Int64 -> Maybe Int
eventIndex inputs wantedEvent = do
    let (_, _, _, intervalMs) = completeOhlcvGridV2 inputs
        openTimeInteger = toInteger wantedEvent - toInteger intervalMs
    guard (intervalMs > 0)
    guard (openTimeInteger >= 0 && openTimeInteger <= toInteger (maxBound :: Int64))
    let (openTimes, _, _, _) = completeOhlcvGridV2 inputs
    V.elemIndex (fromInteger openTimeInteger) openTimes

featureAt :: CompleteOhlcvInputsV2 -> Int -> Maybe (Int64, [Double])
featureAt inputs index = do
    currentRow <- atMay (completeOhlcvRowsV2 inputs) index
    (eventTime, _, _) <- closeWitness currentRow
    values <- traverse (logRealizedVarianceEndingAt inputs index) [1, 6, 24]
    guard (all finite values)
    guard (frv2DecisionTimeMs currentRow >= eventTime)
    pure (frv2DecisionTimeMs currentRow, values)

targetAt :: CompleteOhlcvInputsV2 -> Int -> Int -> Maybe Double
targetAt inputs index horizonBars = do
    guard (horizonBars > 0)
    squaredReturns <- traverse (fmap square . logReturnAt inputs) [index + 1 .. index + horizonBars]
    let realizedVariance = sum squaredReturns
        result = log realizedVariance
    guard (realizedVariance > 0 && finite result)
    pure result

logRealizedVarianceEndingAt :: CompleteOhlcvInputsV2 -> Int -> Int -> Maybe Double
logRealizedVarianceEndingAt inputs endIndex windowBars = do
    guard (windowBars > 0 && endIndex >= windowBars)
    squaredReturns <- traverse (fmap square . logReturnAt inputs) [endIndex - windowBars + 1 .. endIndex]
    let realizedVariance = sum squaredReturns
        result = log realizedVariance
    guard (realizedVariance > 0 && finite result)
    pure result

logReturnAt :: CompleteOhlcvInputsV2 -> Int -> Maybe Double
logReturnAt inputs index = do
    guard (index > 0)
    previousRow <- atMay (completeOhlcvRowsV2 inputs) (index - 1)
    currentRow <- atMay (completeOhlcvRowsV2 inputs) index
    (_, _, previousClose) <- closeWitness previousRow
    (_, _, currentClose) <- closeWitness currentRow
    let result = log (currentClose / previousClose)
    guard (finite result)
    pure result

fitRidge :: [([Double], Double)] -> Maybe ([Double], [Double], Double, [Double], Double, Double, Double)
fitRidge rows = do
    guard (length rows >= harRvMinimumObservationsV1)
    guard (all ((== length harRvFeatureNamesV1) . length . fst) rows)
    guard (all (all finite . fst) rows && all (finite . snd) rows)
    let columns = transposeRect (map fst rows)
        means = map average columns
        scales = map populationScale columns
    guard (length means == length harRvFeatureNamesV1)
    guard (all finite means && all (\value -> finite value && value > 1.0e-12) scales)
    standardized <- traverse (standardize means scales . fst) rows
    let targets = map snd rows
        intercept = average targets
        centeredTargets = map (subtract intercept) targets
        dimension = length harRvFeatureNamesV1
        xtx = foldl' matrixAdd (zeroMatrix dimension) (map outerProduct standardized)
        xty = foldl' (zipWith (+)) (replicate dimension 0) (zipWith (\features target -> map (* target) features) standardized centeredTargets)
        regularized = addDiagonal harRvRidgeLambdaV1 xtx
    coefficients <- solveLinear regularized xty
    guard (length coefficients == dimension && all finite coefficients && finite intercept)
    let predictions = map (\features -> intercept + dot coefficients features) standardized
        residuals = zipWith (-) targets predictions
        sumSquaredErrors = sum (map square residuals)
        trainingMse = sumSquaredErrors / fromIntegral (length rows)
        residualDegrees = length rows - dimension - 1
        residualVariance = max harRvResidualVarianceFloorV1 (sumSquaredErrors / fromIntegral residualDegrees)
        forecastVolatilities = map (exp . (* 0.5)) predictions
    medianForecastVolatility <- median forecastVolatilities
    guard (residualDegrees > 0)
    guard (all finite [trainingMse, residualVariance, medianForecastVolatility])
    guard (trainingMse >= 0 && residualVariance > 0 && medianForecastVolatility > 0)
    pure (means, scales, intercept, coefficients, residualVariance, trainingMse, medianForecastVolatility)

predictStandardized :: HarRvArtifactV1 -> [Double] -> Double
predictStandardized artifact features =
    case standardize (hra1FeatureMeans artifact) (hra1FeatureScales artifact) features of
        Nothing -> 0 / 0
        Just standardized -> hra1Intercept artifact + dot (hra1Coefficients artifact) standardized

standardize :: [Double] -> [Double] -> [Double] -> Maybe [Double]
standardize means scales values = do
    guard (length means == length scales && length means == length values)
    guard (all (\scale -> finite scale && scale > 0) scales)
    let standardized = zipWith3 (\value meanValue scale -> (value - meanValue) / scale) values means scales
    guard (all finite standardized)
    pure standardized

populationScale :: [Double] -> Double
populationScale values =
    let meanValue = average values
     in sqrt (average (map (square . subtract meanValue) values))

transposeRect :: [[a]] -> [[a]]
transposeRect rows =
    case rows of
        [] -> []
        firstRow : _ -> [[row !! index | row <- rows] | index <- [0 .. length firstRow - 1]]

zeroMatrix :: Int -> [[Double]]
zeroMatrix dimension = replicate dimension (replicate dimension 0)

matrixAdd :: [[Double]] -> [[Double]] -> [[Double]]
matrixAdd = zipWith (zipWith (+))

outerProduct :: [Double] -> [[Double]]
outerProduct values = [map (* value) values | value <- values]

addDiagonal :: Double -> [[Double]] -> [[Double]]
addDiagonal lambda matrix =
    [ [if rowIndex == columnIndex then value + lambda else value | (columnIndex, value) <- zip [0 ..] row]
    | (rowIndex, row) <- zip [0 ..] matrix
    ]

type Matrix = V.Vector (V.Vector Double)

solveLinear :: [[Double]] -> [Double] -> Maybe [Double]
solveLinear matrix rhs = do
    let dimension = length matrix
    guard (dimension > 0)
    guard (length rhs == dimension)
    guard (all ((== dimension) . length) matrix)
    guard (all (all finite) matrix && all finite rhs)
    triangular <- forwardElimination dimension (V.fromList (zipWith (\row value -> V.fromList (row ++ [value])) matrix rhs))
    let solution = V.toList (backSubstitution dimension triangular)
    guard (all finite solution)
    pure solution

forwardElimination :: Int -> Matrix -> Maybe Matrix
forwardElimination dimension = go 0
  where
    go column matrix
        | column >= dimension = Just matrix
        | otherwise = do
            pivotRow <- maximumAbsRow column matrix
            let swapped = swapRows column pivotRow matrix
                pivotValues = swapped V.! column
                pivot = pivotValues V.! column
            guard (finite pivot && abs pivot > 1.0e-12)
            let eliminated =
                    V.imap
                        ( \rowIndex row ->
                            if rowIndex <= column
                                then row
                                else
                                    let factor = (row V.! column) / pivot
                                     in V.imap (\valueIndex value -> if valueIndex < column then value else value - factor * (pivotValues V.! valueIndex)) row
                        )
                        swapped
            guard (all (all finite . V.toList) (V.toList eliminated))
            go (column + 1) eliminated

maximumAbsRow :: Int -> Matrix -> Maybe Int
maximumAbsRow column matrix = do
    let candidates = [(rowIndex, abs ((matrix V.! rowIndex) V.! column)) | rowIndex <- [column .. V.length matrix - 1]]
    guard (not (null candidates))
    pure (fst (last (sortOn snd candidates)))

swapRows :: Int -> Int -> Matrix -> Matrix
swapRows left right rows
    | left == right = rows
    | otherwise = rows V.// [(left, rows V.! right), (right, rows V.! left)]

backSubstitution :: Int -> Matrix -> V.Vector Double
backSubstitution dimension matrix = V.fromList (go (dimension - 1) [])
  where
    go rowIndex solved
        | rowIndex < 0 = solved
        | otherwise =
            let row = matrix V.! rowIndex
                rhs = row V.! dimension
                coefficients = V.toList (V.slice (rowIndex + 1) (dimension - rowIndex - 1) row)
                value = (rhs - sum (zipWith (*) coefficients solved)) / (row V.! rowIndex)
             in go (rowIndex - 1) (value : solved)

parseArtifact :: Value -> AesonTypes.Parser HarRvArtifactV1
parseArtifact = withObject "HarRvArtifactV1" $ \obj -> do
    exactKeys "HAR-RV artifact" ["schemaId", "schemaVersion", "payload", "payloadSha256"] obj
    schemaId <- obj .: "schemaId"
    schemaVersion <- obj .: "schemaVersion"
    payload <- obj .: "payload"
    digest <- obj .: "payloadSha256"
    unless (schemaId == harRvArtifactSchemaIdV1) (fail "unsupported HAR-RV artifact schemaId")
    unless (schemaVersion == harRvArtifactSchemaVersionV1) (fail "unsupported HAR-RV artifact schemaVersion")
    unless (digest == digestValue payload) (fail "HAR-RV artifact payload digest mismatch")
    artifact <- parsePayload digest payload
    either fail pure (validateArtifact artifact)
    pure artifact

parsePayload :: String -> Value -> AesonTypes.Parser HarRvArtifactV1
parsePayload digest = withObject "HarRvArtifactPayloadV1" $ \obj -> do
    exactKeys
        "HAR-RV artifact payload"
        [ "compatibilityVersion"
        , "semanticModelId"
        , "modelFamily"
        , "sourceFeatureSchemaId"
        , "sourceFeatureSchemaVersion"
        , "sourceFeatureSchemaSignature"
        , "derivedFeatureSignature"
        , "targetId"
        , "fitMethod"
        , "ridgeLambda"
        , "request"
        , "fit"
        , "safety"
        ]
        obj
    compatibilityVersion <- obj .: "compatibilityVersion"
    semanticModelId <- obj .: "semanticModelId"
    modelFamily <- obj .: "modelFamily"
    sourceSchemaId <- obj .: "sourceFeatureSchemaId"
    sourceSchemaVersion <- obj .: "sourceFeatureSchemaVersion"
    sourceSchemaSignature <- obj .: "sourceFeatureSchemaSignature"
    derivedFeatureSignature <- obj .: "derivedFeatureSignature"
    targetId <- obj .: "targetId"
    fitMethod <- obj .: "fitMethod"
    ridgeLambda <- obj .: "ridgeLambda"
    unless (compatibilityVersion == harRvCompatibilityVersionV1) (fail "unsupported HAR-RV compatibilityVersion")
    unless (semanticModelId == harRvSemanticModelIdV1) (fail "unsupported HAR-RV semanticModelId")
    unless (modelFamily == harRvModelFamilyV1) (fail "unsupported HAR-RV modelFamily")
    unless (sourceSchemaId == completeOhlcvSchemaIdV2) (fail "unsupported HAR-RV source feature schemaId")
    unless (sourceSchemaVersion == completeOhlcvSchemaVersionV2) (fail "unsupported HAR-RV source feature schemaVersion")
    unless (sourceSchemaSignature == completeOhlcvSchemaSignatureV2) (fail "unsupported HAR-RV source feature signature")
    unless (derivedFeatureSignature == harRvFeatureSignatureV1) (fail "unsupported HAR-RV derived feature signature")
    unless (targetId == harRvTargetIdV1) (fail "unsupported HAR-RV targetId")
    unless (fitMethod == harRvFitMethodV1) (fail "unsupported HAR-RV fitMethod")
    unless (ridgeLambda == harRvRidgeLambdaV1) (fail "unsupported HAR-RV ridge lambda")
    requestValueObject <- obj .: "request"
    fitValueObject <- obj .: "fit"
    safetyValueObject <- obj .: "safety"
    request <- parseRequest requestValueObject
    (rowCount, observedCount, means, scales, intercept, coefficients, residualVariance, trainingMse, medianForecastVolatility) <- parseFit fitValueObject
    parseSafety safetyValueObject
    pure
        HarRvArtifactV1
            { hra1Request = request
            , hra1TrainingRowCount = rowCount
            , hra1ObservedTrainingRowCount = observedCount
            , hra1FeatureMeans = means
            , hra1FeatureScales = scales
            , hra1Intercept = intercept
            , hra1Coefficients = coefficients
            , hra1ResidualLogVariance = residualVariance
            , hra1TrainingMse = trainingMse
            , hra1TrainingMedianForecastVolatility = medianForecastVolatility
            , hra1PayloadSha256 = digest
            }

parseRequest :: Value -> AesonTypes.Parser HarRvFitRequestV1
parseRequest = withObject "HarRvFitRequestV1" $ \obj -> do
    exactKeys "HAR-RV fit request" requestKeys obj
    request <-
        HarRvFitRequestV1
            <$> obj .: "registrationId"
            <*> obj .: "codeCommit"
            <*> obj .: "trainingDataSha256"
            <*> obj .: "trainingEvidenceSha256"
            <*> obj .: "sourceManifestSha256"
            <*> obj .: "splitManifestSha256"
            <*> obj .: "academicOrigins"
            <*> obj .: "symbol"
            <*> obj .: "intervalMs"
            <*> obj .: "horizonBars"
            <*> obj .: "trainingStartEventTimeMs"
            <*> obj .: "trainingEndEventTimeMs"
            <*> obj .: "validationStartEventTimeMs"
            <*> obj .: "validationEndEventTimeMs"
            <*> obj .: "finalHoldoutStartEventTimeMs"
            <*> obj .: "purgeBars"
            <*> obj .: "embargoBars"
            <*> obj .: "fitAvailableAtMs"
            <*> obj .: "createdAtMs"
            <*> obj .: "randomSeed"
            <*> obj .: "runtimeVersions"
            <*> obj .: "costModelId"
    either fail pure (validateRequest request)
    pure request

parseFit :: Value -> AesonTypes.Parser (Int, Int, [Double], [Double], Double, [Double], Double, Double, Double)
parseFit = withObject "HarRvFitV1" $ \obj -> do
    exactKeys
        "HAR-RV fit"
        [ "trainingRowCount"
        , "observedTrainingRowCount"
        , "featureMeans"
        , "featureScales"
        , "intercept"
        , "coefficients"
        , "residualLogVariance"
        , "trainingMse"
        , "trainingMedianForecastVolatility"
        ]
        obj
    (,,,,,,,,)
        <$> obj .: "trainingRowCount"
        <*> obj .: "observedTrainingRowCount"
        <*> obj .: "featureMeans"
        <*> obj .: "featureScales"
        <*> obj .: "intercept"
        <*> obj .: "coefficients"
        <*> obj .: "residualLogVariance"
        <*> obj .: "trainingMse"
        <*> obj .: "trainingMedianForecastVolatility"

parseSafety :: Value -> AesonTypes.Parser ()
parseSafety = withObject "HarRvSafetyV1" $ \obj -> do
    exactKeys "HAR-RV safety" safetyKeys obj
    validationState <- obj .: "validationMetricsState"
    holdoutState <- obj .: "finalHoldoutState"
    promotionState <- obj .: "promotionState"
    authorities <-
        sequence
            [ obj .: "researchAdmissionAuthorized"
            , obj .: "experimentAuthorized"
            , obj .: "holdoutAuthorized"
            , obj .: "modelAuthorized"
            , obj .: "promotionAuthorized"
            , obj .: "deploymentAuthorized"
            , obj .: "orderAuthorized"
            , obj .: "liveTradingAuthorized"
            ]
    unless (validationState == harRvValidationMetricsStateV1) (fail "HAR-RV validation metrics state must be not_evaluated")
    unless (holdoutState == harRvFinalHoldoutStateV1) (fail "HAR-RV final holdout state must be untouched")
    unless (promotionState == harRvPromotionStateV1) (fail "HAR-RV promotion state must be offline_research_only")
    when (or authorities) (fail "HAR-RV component artifact cannot carry downstream authority")

artifactValue :: HarRvArtifactV1 -> Value
artifactValue artifact =
    object
        [ "schemaId" .= harRvArtifactSchemaIdV1
        , "schemaVersion" .= harRvArtifactSchemaVersionV1
        , "payload" .= payloadValue artifact
        , "payloadSha256" .= hra1PayloadSha256 artifact
        ]

payloadValue :: HarRvArtifactV1 -> Value
payloadValue artifact =
    object
        [ "compatibilityVersion" .= harRvCompatibilityVersionV1
        , "semanticModelId" .= harRvSemanticModelIdV1
        , "modelFamily" .= harRvModelFamilyV1
        , "sourceFeatureSchemaId" .= completeOhlcvSchemaIdV2
        , "sourceFeatureSchemaVersion" .= completeOhlcvSchemaVersionV2
        , "sourceFeatureSchemaSignature" .= completeOhlcvSchemaSignatureV2
        , "derivedFeatureSignature" .= harRvFeatureSignatureV1
        , "targetId" .= harRvTargetIdV1
        , "fitMethod" .= harRvFitMethodV1
        , "ridgeLambda" .= harRvRidgeLambdaV1
        , "request" .= requestValue (hra1Request artifact)
        , "fit"
            .= object
                [ "trainingRowCount" .= hra1TrainingRowCount artifact
                , "observedTrainingRowCount" .= hra1ObservedTrainingRowCount artifact
                , "featureMeans" .= hra1FeatureMeans artifact
                , "featureScales" .= hra1FeatureScales artifact
                , "intercept" .= hra1Intercept artifact
                , "coefficients" .= hra1Coefficients artifact
                , "residualLogVariance" .= hra1ResidualLogVariance artifact
                , "trainingMse" .= hra1TrainingMse artifact
                , "trainingMedianForecastVolatility" .= hra1TrainingMedianForecastVolatility artifact
                ]
        , "safety"
            .= object
                [ "validationMetricsState" .= harRvValidationMetricsStateV1
                , "finalHoldoutState" .= harRvFinalHoldoutStateV1
                , "promotionState" .= harRvPromotionStateV1
                , "researchAdmissionAuthorized" .= False
                , "experimentAuthorized" .= False
                , "holdoutAuthorized" .= False
                , "modelAuthorized" .= False
                , "promotionAuthorized" .= False
                , "deploymentAuthorized" .= False
                , "orderAuthorized" .= False
                , "liveTradingAuthorized" .= False
                ]
        ]

requestValue :: HarRvFitRequestV1 -> Value
requestValue request =
    object
        [ "registrationId" .= hrr1RegistrationId request
        , "codeCommit" .= hrr1CodeCommit request
        , "trainingDataSha256" .= hrr1TrainingDataSha256 request
        , "trainingEvidenceSha256" .= hrr1TrainingEvidenceSha256 request
        , "sourceManifestSha256" .= hrr1SourceManifestSha256 request
        , "splitManifestSha256" .= hrr1SplitManifestSha256 request
        , "academicOrigins" .= hrr1AcademicOrigins request
        , "symbol" .= hrr1Symbol request
        , "intervalMs" .= hrr1IntervalMs request
        , "horizonBars" .= hrr1HorizonBars request
        , "trainingStartEventTimeMs" .= hrr1TrainingStartEventTimeMs request
        , "trainingEndEventTimeMs" .= hrr1TrainingEndEventTimeMs request
        , "validationStartEventTimeMs" .= hrr1ValidationStartEventTimeMs request
        , "validationEndEventTimeMs" .= hrr1ValidationEndEventTimeMs request
        , "finalHoldoutStartEventTimeMs" .= hrr1FinalHoldoutStartEventTimeMs request
        , "purgeBars" .= hrr1PurgeBars request
        , "embargoBars" .= hrr1EmbargoBars request
        , "fitAvailableAtMs" .= hrr1FitAvailableAtMs request
        , "createdAtMs" .= hrr1CreatedAtMs request
        , "randomSeed" .= hrr1RandomSeed request
        , "runtimeVersions" .= hrr1RuntimeVersions request
        , "costModelId" .= hrr1CostModelId request
        ]

validateRequest :: HarRvFitRequestV1 -> Either String ()
validateRequest request
    | hrr1RegistrationId request /= "har_rv_risk_gate_v1" = Left "HAR-RV registrationId is unsupported"
    | not (validCommit (hrr1CodeCommit request)) = Left "HAR-RV codeCommit is invalid"
    | not (all validSha256 requestDigests) = Left "HAR-RV provenance digest is invalid"
    | hrr1AcademicOrigins request /= registeredAcademicOriginsV1 = Left "HAR-RV academic origins do not match the registration"
    | hrr1Symbol request `notElem` harRvRegisteredSymbolsV1 = Left "HAR-RV symbol is outside the registered universe"
    | hrr1IntervalMs request `notElem` [3600000, 14400000, 28800000] = Left "HAR-RV interval is outside the registered set"
    | hrr1HorizonBars request `notElem` [1, 3, 6] = Left "HAR-RV horizon is outside the registered set"
    | hrr1PurgeBars request /= 6 || hrr1EmbargoBars request /= 6 = Left "HAR-RV purge and embargo must match the registration"
    | trainingStart < registeredDatasetStartMsV1 = Left "HAR-RV training starts before the prospective dataset"
    | not (gridAligned registeredDatasetStartMsV1 interval trainingStart) = Left "HAR-RV training start is off the registered bar phase"
    | toInteger trainingStart - toInteger registeredDatasetStartMsV1 < 24 * toInteger interval = Left "HAR-RV training start lacks an in-dataset 24-return lookback"
    | trainingEnd < trainingStart = Left "HAR-RV training event range is invalid"
    | not (gridAligned trainingStart interval trainingEnd) = Left "HAR-RV training event range is off grid"
    | validationStart <= trainingEnd = Left "HAR-RV validation must follow training"
    | not (gridAligned trainingStart interval validationStart) = Left "HAR-RV validation start is off grid"
    | validationEnd < validationStart || validationEnd > registeredDevelopmentEndMsV1 = Left "HAR-RV validation event range is invalid"
    | not (gridAligned validationStart interval validationEnd) = Left "HAR-RV validation end is off grid"
    | holdoutStart /= registeredFinalHoldoutStartMsV1 = Left "HAR-RV final holdout boundary does not match the registration"
    | validationLastTargetEvent >= toInteger holdoutStart = Left "HAR-RV validation target horizon reaches the final holdout"
    | toInteger validationStart - toInteger trainingEnd < gapMs = Left "HAR-RV purge and embargo gap is insufficient"
    | toInteger fitAvailableAt < lastTrainingTargetEvent || fitAvailableAt > validationStart = Left "HAR-RV fit availability crosses its causal fold boundary"
    | createdAt < fitAvailableAt = Left "HAR-RV artifact creation predates fit availability"
    | hrr1RandomSeed request /= 20270904 = Left "HAR-RV random seed does not match the registration"
    | null runtimes || not (all validText runtimes) || length runtimes /= length (nub runtimes) = Left "HAR-RV runtimeVersions are invalid"
    | not (validLabel (hrr1CostModelId request)) = Left "HAR-RV costModelId is invalid"
    | expectedTrainingRowCount request < harRvMinimumObservationsV1 = Left "HAR-RV training grid is too short"
    | otherwise = Right ()
  where
    requestDigests =
        [ hrr1TrainingDataSha256 request
        , hrr1TrainingEvidenceSha256 request
        , hrr1SourceManifestSha256 request
        , hrr1SplitManifestSha256 request
        ]
    interval = hrr1IntervalMs request
    horizon = hrr1HorizonBars request
    trainingStart = hrr1TrainingStartEventTimeMs request
    trainingEnd = hrr1TrainingEndEventTimeMs request
    validationStart = hrr1ValidationStartEventTimeMs request
    validationEnd = hrr1ValidationEndEventTimeMs request
    holdoutStart = hrr1FinalHoldoutStartEventTimeMs request
    fitAvailableAt = hrr1FitAvailableAtMs request
    createdAt = hrr1CreatedAtMs request
    runtimes = hrr1RuntimeVersions request
    gapMs = toInteger (hrr1PurgeBars request + hrr1EmbargoBars request) * toInteger interval
    lastTrainingTargetEvent = toInteger trainingEnd + toInteger horizon * toInteger interval
    validationLastTargetEvent = toInteger validationEnd + toInteger horizon * toInteger interval

validateArtifact :: HarRvArtifactV1 -> Either String ()
validateArtifact artifact = do
    validateRequest (hra1Request artifact)
    unless (hra1TrainingRowCount artifact == expectedTrainingRowCount (hra1Request artifact)) (Left "HAR-RV artifact training row count is invalid")
    unless (hra1ObservedTrainingRowCount artifact >= harRvMinimumObservationsV1 && hra1ObservedTrainingRowCount artifact <= hra1TrainingRowCount artifact) (Left "HAR-RV artifact observed row count is invalid")
    unless (length (hra1FeatureMeans artifact) == length harRvFeatureNamesV1) (Left "HAR-RV artifact feature means have the wrong shape")
    unless (length (hra1FeatureScales artifact) == length harRvFeatureNamesV1) (Left "HAR-RV artifact feature scales have the wrong shape")
    unless (length (hra1Coefficients artifact) == length harRvFeatureNamesV1) (Left "HAR-RV artifact coefficients have the wrong shape")
    unless (all finite fitValues) (Left "HAR-RV artifact contains non-finite fit values")
    unless (all (> 0) (hra1FeatureScales artifact)) (Left "HAR-RV artifact feature scale is not positive")
    unless (hra1ResidualLogVariance artifact >= harRvResidualVarianceFloorV1) (Left "HAR-RV artifact residual variance is below its floor")
    unless (hra1TrainingMse artifact >= 0) (Left "HAR-RV artifact training MSE is negative")
    unless (hra1TrainingMedianForecastVolatility artifact > 0) (Left "HAR-RV artifact training median forecast volatility is invalid")
    unless (validSha256 (hra1PayloadSha256 artifact) && hra1PayloadSha256 artifact == payloadDigest artifact) (Left "HAR-RV artifact payload digest is invalid")
  where
    fitValues =
        hra1FeatureMeans artifact
            ++ hra1FeatureScales artifact
            ++ [hra1Intercept artifact]
            ++ hra1Coefficients artifact
            ++ [hra1ResidualLogVariance artifact, hra1TrainingMse artifact, hra1TrainingMedianForecastVolatility artifact]

validCompleteOhlcvRow :: FeatureRowV2 -> Bool
validCompleteOhlcvRow row =
    frv2SchemaId row == featureAvailabilitySchemaIdV2
        && frv2Names row == completeOhlcvFieldNamesV2
        && frv2Required row == replicate (length completeOhlcvFieldNamesV2) True
        && frv2Available row == replicate (length completeOhlcvFieldNamesV2) True
        && length (frv2Values row) == length completeOhlcvFieldNamesV2
        && all finite (frv2Values row)
        && all isPresent (frv2EventTimesMs row)
        && all isPresent (frv2AvailabilityTimesMs row)
  where
    isPresent (Just _) = True
    isPresent Nothing = False

closeWitness :: FeatureRowV2 -> Maybe (Int64, Int64, Double)
closeWitness row = do
    guard (validCompleteOhlcvRow row)
    closeIndex <- elemIndex "market.close" (frv2Names row)
    value <- atMay (frv2Values row) closeIndex
    Just eventTime <- atMay (frv2EventTimesMs row) closeIndex
    Just availabilityTime <- atMay (frv2AvailabilityTimesMs row) closeIndex
    guard (value > 0 && finite value)
    guard (eventTime >= 0 && availabilityTime >= eventTime && availabilityTime <= frv2DecisionTimeMs row)
    pure (eventTime, availabilityTime, value)

expectedTrainingRowCount :: HarRvFitRequestV1 -> Int
expectedTrainingRowCount request =
    fromInteger
        ( (toInteger (hrr1TrainingEndEventTimeMs request) - toInteger (hrr1TrainingStartEventTimeMs request))
            `div` toInteger (hrr1IntervalMs request)
            + 1
        )

payloadDigest :: HarRvArtifactV1 -> String
payloadDigest = digestValue . payloadValue

featureRowValue :: FeatureRowV2 -> Value
featureRowValue row =
    object
        [ "schemaId" .= frv2SchemaId row
        , "decisionTimeMs" .= frv2DecisionTimeMs row
        , "names" .= frv2Names row
        , "values" .= frv2Values row
        , "available" .= frv2Available row
        , "required" .= frv2Required row
        , "eventTimesMs" .= frv2EventTimesMs row
        , "availabilityTimesMs" .= frv2AvailabilityTimesMs row
        ]

digestValue :: Value -> String
digestValue value = show (hash (BL.toStrict (Aeson.encode value)) :: Digest SHA256)

exactKeys :: String -> [String] -> KeyMap.KeyMap Value -> AesonTypes.Parser ()
exactKeys label expected obj =
    unless (sort expected == sort (map Key.toString (KeyMap.keys obj))) (fail (label ++ " has missing or unknown fields"))

requestKeys :: [String]
requestKeys =
    [ "registrationId"
    , "codeCommit"
    , "trainingDataSha256"
    , "trainingEvidenceSha256"
    , "sourceManifestSha256"
    , "splitManifestSha256"
    , "academicOrigins"
    , "symbol"
    , "intervalMs"
    , "horizonBars"
    , "trainingStartEventTimeMs"
    , "trainingEndEventTimeMs"
    , "validationStartEventTimeMs"
    , "validationEndEventTimeMs"
    , "finalHoldoutStartEventTimeMs"
    , "purgeBars"
    , "embargoBars"
    , "fitAvailableAtMs"
    , "createdAtMs"
    , "randomSeed"
    , "runtimeVersions"
    , "costModelId"
    ]

safetyKeys :: [String]
safetyKeys =
    [ "validationMetricsState"
    , "finalHoldoutState"
    , "promotionState"
    , "researchAdmissionAuthorized"
    , "experimentAuthorized"
    , "holdoutAuthorized"
    , "modelAuthorized"
    , "promotionAuthorized"
    , "deploymentAuthorized"
    , "orderAuthorized"
    , "liveTradingAuthorized"
    ]

gridAligned :: Int64 -> Int64 -> Int64 -> Bool
gridAligned anchor interval value =
    interval > 0
        && value >= anchor
        && (toInteger value - toInteger anchor) `mod` toInteger interval == 0

validCommit :: String -> Bool
validCommit value = length value == 40 && all isLowerHex value

validSha256 :: String -> Bool
validSha256 value = length value == 64 && all isLowerHex value

isLowerHex :: Char -> Bool
isLowerHex character = isAscii character && (isDigit character || character >= 'a' && character <= 'f')

validLabel :: String -> Bool
validLabel value =
    not (null value)
        && all (\character -> isAscii character && (isAlphaNum character || character `elem` ("_.:-" :: String))) value

validText :: String -> Bool
validText value = not (null value) && all (\character -> isAscii character && not (isControl character)) value

atMay :: [a] -> Int -> Maybe a
atMay values index
    | index < 0 = Nothing
    | otherwise = case drop index values of
        value : _ -> Just value
        [] -> Nothing

average :: [Double] -> Double
average values = sum values / fromIntegral (length values)

median :: [Double] -> Maybe Double
median values = do
    guard (not (null values) && all finite values)
    let sortedValues = sort values
        count = length sortedValues
        middle = count `div` 2
    if odd count
        then atMay sortedValues middle
        else do
            left <- atMay sortedValues (middle - 1)
            right <- atMay sortedValues middle
            pure ((left + right) / 2)

dot :: [Double] -> [Double] -> Double
dot left right = sum (zipWith (*) left right)

square :: Double -> Double
square value = value * value

concatWithComma :: [String] -> String
concatWithComma [] = ""
concatWithComma (value : values) = value ++ concatMap (',' :) values

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
