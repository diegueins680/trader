{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.MissingnessAwarePreprocessorV1 (
    MissingnessAwarePreprocessorFitRequestV1 (..),
    MissingnessAwarePreprocessorArtifactV1,
    MissingnessAwarePreparedRowV1 (..),
    missingnessAwarePreprocessorSchemaIdV1,
    missingnessAwarePreprocessorSchemaVersionV1,
    missingnessAwarePreprocessorCompatibilityVersionV1,
    missingnessAwarePreprocessorInputNamesV1,
    missingnessAwarePreprocessorInputSignatureV1,
    missingnessAwareTrainingPanelSha256V1,
    fitMissingnessAwarePreprocessorV1,
    transformMissingnessAwarePanelV1,
    encodeMissingnessAwarePreprocessorV1,
    decodeMissingnessAwarePreprocessorV1,
    mappa1Request,
    mappa1TrainingRowCount,
    mappa1ObservationCounts,
    mappa1FeatureMeans,
    mappa1FeatureScales,
    mappa1PayloadSha256,
) where

import Control.Monad (guard, unless, when)
import Crypto.Hash (Digest, SHA256, hash)
import Data.Aeson (Value, object, withObject, (.:), (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KeyMap
import qualified Data.Aeson.Types as AesonTypes
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum, isAscii, isControl, isDigit, isUpper)
import Data.Int (Int64)
import Data.List (intercalate, nub, sort, transpose)
import Data.Maybe (catMaybes, isNothing, mapMaybe)
import qualified Data.Vector as V

import Trader.Predictors.CrossExchangeFeaturesV2 (crossExchangeModelFeatureNamesV2)
import Trader.Predictors.DerivativesFeaturesV2 (derivativesModelFeatureNamesV2)
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
import Trader.Predictors.MissingnessAwareFeaturesV1 (
    MissingnessAwareFeaturePanelV1 (..),
    missingnessAwareFeatureNamesV1,
    missingnessAwareFeaturePanelSchemaIdV1,
    missingnessAwareFeaturePanelSchemaVersionV1,
    missingnessAwareFeatureSignatureV1,
    missingnessAwareLookbackBarsV1,
    missingnessAwarePriceFeatureNamesV1,
    missingnessAwareRegisteredSymbolsV1,
 )

data MissingnessAwarePreprocessorFitRequestV1 = MissingnessAwarePreprocessorFitRequestV1
    { mapfr1RegistrationId :: !String
    , mapfr1CodeCommit :: !String
    , mapfr1TrainingDataSha256 :: !String
    , mapfr1TrainingPanelSha256 :: !String
    , mapfr1SourceManifestSha256 :: !String
    , mapfr1SplitManifestSha256 :: !String
    , mapfr1AcademicOrigins :: ![String]
    , mapfr1Symbol :: !String
    , mapfr1IntervalMs :: !Int64
    , mapfr1HorizonBars :: !Int
    , mapfr1TrainingStartOpenTimeMs :: !Int64
    , mapfr1TrainingEndOpenTimeMs :: !Int64
    , mapfr1ValidationStartOpenTimeMs :: !Int64
    , mapfr1ValidationEndOpenTimeMs :: !Int64
    , mapfr1FinalHoldoutStartOpenTimeMs :: !Int64
    , mapfr1PurgeBars :: !Int
    , mapfr1EmbargoBars :: !Int
    , mapfr1FitAvailableAtMs :: !Int64
    , mapfr1CreatedAtMs :: !Int64
    , mapfr1RandomSeed :: !Int
    , mapfr1RuntimeVersions :: ![String]
    , mapfr1CostModelId :: !String
    }
    deriving (Eq, Show)

data MissingnessAwarePreprocessorArtifactV1 = MissingnessAwarePreprocessorArtifactV1
    { mappa1Request :: !MissingnessAwarePreprocessorFitRequestV1
    , mappa1TrainingRowCount :: !Int
    , mappa1ObservationCounts :: ![Int]
    , mappa1FeatureMeans :: ![Double]
    , mappa1FeatureScales :: ![Double]
    , mappa1PayloadSha256 :: !String
    }
    deriving (Eq, Show)

data MissingnessAwarePreparedRowV1 = MissingnessAwarePreparedRowV1
    { mapr1OpenTimeMs :: !Int64
    , mapr1DecisionTimeMs :: !Int64
    , mapr1Inputs :: ![Double]
    }
    deriving (Eq, Show)

missingnessAwarePreprocessorSchemaIdV1 :: String
missingnessAwarePreprocessorSchemaIdV1 = "missingness_aware_preprocessor_artifact_v1"

missingnessAwarePreprocessorSchemaVersionV1 :: Int
missingnessAwarePreprocessorSchemaVersionV1 = 1

missingnessAwarePreprocessorCompatibilityVersionV1 :: Int
missingnessAwarePreprocessorCompatibilityVersionV1 = 1

missingnessAwarePreprocessorInputNamesV1 :: [String]
missingnessAwarePreprocessorInputNamesV1 =
    map ("standardized." ++) missingnessAwareFeatureNamesV1
        ++ map ("available." ++) optionalFeatureNamesV1

missingnessAwarePreprocessorInputSignatureV1 :: String
missingnessAwarePreprocessorInputSignatureV1 =
    missingnessAwarePreprocessorSchemaIdV1
        ++ "|"
        ++ intercalate "," missingnessAwarePreprocessorInputNamesV1

optionalFeatureNamesV1 :: [String]
optionalFeatureNamesV1 = drop (length missingnessAwarePriceFeatureNamesV1) missingnessAwareFeatureNamesV1

registeredAcademicOriginsV1 :: [String]
registeredAcademicOriginsV1 =
    [ "https://doi.org/10.1093/rfs/hhaa009"
    , "https://doi.org/10.1214/aos/1013203451"
    , "https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html"
    ]

registeredDatasetStartMsV1, registeredDevelopmentEndMsV1, registeredFinalHoldoutStartMsV1 :: Int64
registeredDatasetStartMsV1 = 1800489600000
registeredDevelopmentEndMsV1 = 1821481200000
registeredFinalHoldoutStartMsV1 = 1821484800000

missingnessAwareTrainingPanelSha256V1 :: MissingnessAwareFeaturePanelV1 -> String
missingnessAwareTrainingPanelSha256V1 = digestValue . panelValue

fitMissingnessAwarePreprocessorV1 ::
    MissingnessAwarePreprocessorFitRequestV1 ->
    MissingnessAwareFeaturePanelV1 ->
    Either String MissingnessAwarePreprocessorArtifactV1
fitMissingnessAwarePreprocessorV1 request panel = do
    validateRequest request
    validateTrainingPanel request panel
    unless
        (mapfr1TrainingPanelSha256 request == missingnessAwareTrainingPanelSha256V1 panel)
        (Left "missingness-aware training panel does not match its provenance digest")
    let rows = mafp1Rows panel
        columns = transpose (map rowCells rows)
        observedColumns = map (mapMaybe observedCell) columns
        observationCounts = map length observedColumns
    unless (length columns == length missingnessAwareFeatureNamesV1) (Left "missingness-aware training panel has the wrong feature shape")
    unless (all (> 0) observationCounts) (Left "missingness-aware training feature has no observed training value")
    let means = map mean observedColumns
        scales = zipWith scaleAround means observedColumns
        artifactWithoutDigest =
            MissingnessAwarePreprocessorArtifactV1
                { mappa1Request = request
                , mappa1TrainingRowCount = length rows
                , mappa1ObservationCounts = observationCounts
                , mappa1FeatureMeans = means
                , mappa1FeatureScales = scales
                , mappa1PayloadSha256 = ""
                }
        artifact = artifactWithoutDigest{mappa1PayloadSha256 = payloadDigest artifactWithoutDigest}
    validateArtifact artifact
    pure artifact

transformMissingnessAwarePanelV1 ::
    MissingnessAwarePreprocessorArtifactV1 ->
    MissingnessAwareFeaturePanelV1 ->
    Maybe [MissingnessAwarePreparedRowV1]
transformMissingnessAwarePanelV1 artifact panel = do
    either (const Nothing) Just (validateArtifact artifact)
    let request = mappa1Request artifact
        opens = V.toList (mafp1OpenTimesMs panel)
        rows = mafp1Rows panel
        trainingGrid = grid (mapfr1TrainingStartOpenTimeMs request) (mapfr1TrainingEndOpenTimeMs request) (mapfr1IntervalMs request)
    guard (mafp1Scope panel == mapfr1Symbol request)
    guard (mafp1IntervalMs panel == mapfr1IntervalMs request)
    guard (length opens == length rows && not (null rows))
    guard (validTransformGrid request opens)
    guard (opens /= trainingGrid || missingnessAwareTrainingPanelSha256V1 panel == mapfr1TrainingPanelSha256 request)
    guard (and (zipWith (validPanelRow (mafp1IntervalMs panel)) opens rows))
    traverse (prepareRow artifact) (zip opens rows)

encodeMissingnessAwarePreprocessorV1 :: MissingnessAwarePreprocessorArtifactV1 -> BL.ByteString
encodeMissingnessAwarePreprocessorV1 = Aeson.encode . artifactValue

decodeMissingnessAwarePreprocessorV1 :: BL.ByteString -> Either String MissingnessAwarePreprocessorArtifactV1
decodeMissingnessAwarePreprocessorV1 bytes = do
    value <- Aeson.eitherDecode bytes
    AesonTypes.parseEither parseArtifact value

prepareRow :: MissingnessAwarePreprocessorArtifactV1 -> (Int64, FeatureRowV2) -> Maybe MissingnessAwarePreparedRowV1
prepareRow artifact (openTime, row) = do
    guard (validPanelRow (mapfr1IntervalMs (mappa1Request artifact)) openTime row)
    standardized <- sequence (zipWith3 standardizedCell (mappa1FeatureMeans artifact) (mappa1FeatureScales artifact) (rowCells row))
    let optionalMasks = map boolValue (drop (length missingnessAwarePriceFeatureNamesV1) (frv2Available row))
        inputs = standardized ++ optionalMasks
    guard (length inputs == length missingnessAwarePreprocessorInputNamesV1 && all finite inputs)
    pure
        MissingnessAwarePreparedRowV1
            { mapr1OpenTimeMs = openTime
            , mapr1DecisionTimeMs = frv2DecisionTimeMs row
            , mapr1Inputs = inputs
            }

standardizedCell :: Double -> Double -> (Double, Bool) -> Maybe Double
standardizedCell location scale (value, available) = do
    guard (finite location && finite scale && scale > 0)
    let imputed = if available then value else location
        result = (imputed - location) / scale
    guard (finite imputed && finite result)
    pure result

rowCells :: FeatureRowV2 -> [(Double, Bool)]
rowCells row = zip (frv2Values row) (frv2Available row)

observedCell :: (Double, Bool) -> Maybe Double
observedCell (value, available)
    | available && finite value = Just value
    | otherwise = Nothing

validateTrainingPanel :: MissingnessAwarePreprocessorFitRequestV1 -> MissingnessAwareFeaturePanelV1 -> Either String ()
validateTrainingPanel request panel = do
    unless (mafp1Scope panel == mapfr1Symbol request) (Left "missingness-aware training panel symbol mismatch")
    unless (mafp1IntervalMs panel == mapfr1IntervalMs request) (Left "missingness-aware training panel interval mismatch")
    let opens = V.toList (mafp1OpenTimesMs panel)
        rows = mafp1Rows panel
        expectedOpens = grid (mapfr1TrainingStartOpenTimeMs request) (mapfr1TrainingEndOpenTimeMs request) (mapfr1IntervalMs request)
    unless (opens == expectedOpens && length rows == length opens) (Left "missingness-aware training panel does not match the registered training grid")
    unless (and (zipWith (validPanelRow (mafp1IntervalMs panel)) opens rows)) (Left "missingness-aware training panel row is malformed")
    let availabilityTimes = concatMap (catMaybes . frv2AvailabilityTimesMs) rows
    unless (not (null availabilityTimes) && maximum availabilityTimes <= mapfr1FitAvailableAtMs request) (Left "missingness-aware fit predates its training evidence")
    unless (all ((<= mapfr1FitAvailableAtMs request) . frv2DecisionTimeMs) rows) (Left "missingness-aware fit predates a training decision")

validPanelRow :: Int64 -> Int64 -> FeatureRowV2 -> Bool
validPanelRow intervalMs openTime row =
    frv2SchemaId row == featureAvailabilitySchemaIdV2
        && frv2Names row == missingnessAwareFeatureNamesV1
        && frv2Required row == replicate requiredCount True ++ replicate optionalCount False
        && all (== featureCount) lengths
        && and (zipWith5 validCell (frv2Values row) (frv2Available row) (frv2Required row) (frv2EventTimesMs row) (frv2AvailabilityTimesMs row))
        && validBarTiming
  where
    featureCount = length missingnessAwareFeatureNamesV1
    requiredCount = length missingnessAwarePriceFeatureNamesV1
    optionalCount = featureCount - requiredCount
    barEnd = toInteger openTime + toInteger intervalMs
    decisionTime = toInteger (frv2DecisionTimeMs row)
    validBarTiming =
        intervalMs > 0
            && barEnd <= toInteger (maxBound :: Int64)
            && decisionTime >= barEnd
            && decisionTime < barEnd + toInteger intervalMs
            && all (== Just (fromInteger barEnd)) (take requiredCount (frv2EventTimesMs row))
            && and
                ( zipWith3
                    validDerivativeTiming
                    derivativesModelFeatureNamesV2
                    (take derivativeCount (drop requiredCount (frv2Available row)))
                    ( take derivativeCount (drop requiredCount (zip (frv2EventTimesMs row) (frv2AvailabilityTimesMs row)))
                    )
                )
            && and
                ( zipWith
                    (\available eventTime -> not available || eventTime == Just (fromInteger barEnd))
                    (drop crossExchangeOffset (frv2Available row))
                    (drop crossExchangeOffset (frv2EventTimesMs row))
                )
    crossExchangeOffset = featureCount - length crossExchangeModelFeatureNamesV2
    derivativeCount = length derivativesModelFeatureNamesV2
    derivativeDecision = barEnd - 1
    validDerivativeTiming name available (eventTime, availabilityTime)
        | not available = True
        | otherwise =
            case (eventTime, availabilityTime) of
                (Just event, Just observedAt) ->
                    toInteger observedAt <= derivativeDecision
                        && derivativeDecision - toInteger event <= freshnessLimit name
                _ -> False
    freshnessLimit name
        | name `elem` take 2 derivativesModelFeatureNamesV2 = 9 * 60 * 60 * 1000
        | otherwise = 2 * toInteger intervalMs
    lengths =
        [ length (frv2Values row)
        , length (frv2Available row)
        , length (frv2Required row)
        , length (frv2EventTimesMs row)
        , length (frv2AvailabilityTimesMs row)
        ]
    validCell value available required eventTime availabilityTime
        | available =
            finite value
                && case (eventTime, availabilityTime) of
                    (Just event, Just observedAt) -> event >= 0 && event <= observedAt && observedAt <= frv2DecisionTimeMs row
                    _ -> False
        | otherwise = not required && value == 0 && isNothing eventTime && isNothing availabilityTime

validTransformGrid :: MissingnessAwarePreprocessorFitRequestV1 -> [Int64] -> Bool
validTransformGrid request opens =
    case opens of
        [] -> False
        firstOpen : _ ->
            firstOpen >= registeredDatasetStartMsV1
                && last opens <= registeredDevelopmentEndMsV1
                && all (gridAligned registeredDatasetStartMsV1 interval) opens
                && and (zipWith adjacent opens (drop 1 opens))
                && (opens == trainingGrid || (firstOpen >= mapfr1ValidationStartOpenTimeMs request && last opens <= mapfr1ValidationEndOpenTimeMs request))
  where
    interval = mapfr1IntervalMs request
    trainingGrid = grid (mapfr1TrainingStartOpenTimeMs request) (mapfr1TrainingEndOpenTimeMs request) interval
    adjacent earlier later = toInteger later - toInteger earlier == toInteger interval

validateRequest :: MissingnessAwarePreprocessorFitRequestV1 -> Either String ()
validateRequest request
    | mapfr1RegistrationId request /= "missingness_aware_calibrated_shallow_v1" = Left "missingness-aware registrationId is unsupported"
    | not (validCommit (mapfr1CodeCommit request)) = Left "missingness-aware codeCommit is invalid"
    | not (all validSha256 digests) = Left "missingness-aware provenance digest is invalid"
    | mapfr1AcademicOrigins request /= registeredAcademicOriginsV1 = Left "missingness-aware academic origins do not match the registration"
    | mapfr1Symbol request `notElem` missingnessAwareRegisteredSymbolsV1 = Left "missingness-aware symbol is outside the registered universe"
    | interval `notElem` [3600000, 14400000, 28800000] = Left "missingness-aware interval is outside the registered set"
    | horizon `notElem` [1, 3, 6] = Left "missingness-aware horizon is outside the registered set"
    | mapfr1PurgeBars request /= horizon || mapfr1EmbargoBars request /= 6 = Left "missingness-aware purge or embargo does not match the registration"
    | toInteger trainingStart < minimumTrainingStart || trainingEnd < trainingStart = Left "missingness-aware training range does not retain the registered lookback"
    | not (gridAligned registeredDatasetStartMsV1 interval trainingStart && gridAligned trainingStart interval trainingEnd) = Left "missingness-aware training range is off grid"
    | validationStart <= trainingEnd || validationEnd < validationStart || validationEnd > registeredDevelopmentEndMsV1 = Left "missingness-aware validation range is invalid"
    | not (gridAligned trainingStart interval validationStart && gridAligned validationStart interval validationEnd) = Left "missingness-aware validation range is off grid"
    | mapfr1FinalHoldoutStartOpenTimeMs request /= registeredFinalHoldoutStartMsV1 = Left "missingness-aware holdout boundary does not match the registration"
    | toInteger validationEnd + toInteger horizon * toInteger interval >= toInteger registeredFinalHoldoutStartMsV1 = Left "missingness-aware validation label reaches the final holdout"
    | toInteger validationStart - toInteger trainingEnd < toInteger (horizon + 6) * toInteger interval = Left "missingness-aware purge and embargo gap is insufficient"
    | mapfr1FitAvailableAtMs request > validationStart = Left "missingness-aware fit availability crosses validation"
    | mapfr1CreatedAtMs request < mapfr1FitAvailableAtMs request || mapfr1CreatedAtMs request > validationStart = Left "missingness-aware artifact creation crosses validation"
    | mapfr1RandomSeed request /= 20270904 = Left "missingness-aware random seed does not match the registration"
    | null runtimes || length runtimes /= length (nub runtimes) || not (all validText runtimes) = Left "missingness-aware runtimeVersions are invalid"
    | not (validLabel (mapfr1CostModelId request)) = Left "missingness-aware costModelId is invalid"
    | otherwise = Right ()
  where
    digests = [mapfr1TrainingDataSha256 request, mapfr1TrainingPanelSha256 request, mapfr1SourceManifestSha256 request, mapfr1SplitManifestSha256 request]
    interval = mapfr1IntervalMs request
    horizon = mapfr1HorizonBars request
    trainingStart = mapfr1TrainingStartOpenTimeMs request
    trainingEnd = mapfr1TrainingEndOpenTimeMs request
    validationStart = mapfr1ValidationStartOpenTimeMs request
    validationEnd = mapfr1ValidationEndOpenTimeMs request
    runtimes = mapfr1RuntimeVersions request
    minimumTrainingStart = toInteger registeredDatasetStartMsV1 + toInteger missingnessAwareLookbackBarsV1 * toInteger interval

validateArtifact :: MissingnessAwarePreprocessorArtifactV1 -> Either String ()
validateArtifact artifact = do
    validateRequest (mappa1Request artifact)
    let featureCount = length missingnessAwareFeatureNamesV1
        rowCount = mappa1TrainingRowCount artifact
    unless (rowCount == length (grid (mapfr1TrainingStartOpenTimeMs request) (mapfr1TrainingEndOpenTimeMs request) (mapfr1IntervalMs request))) (Left "missingness-aware artifact row count is invalid")
    unless
        ( length (mappa1ObservationCounts artifact) == featureCount
            && length (mappa1FeatureMeans artifact) == featureCount
            && length (mappa1FeatureScales artifact) == featureCount
        )
        (Left "missingness-aware artifact fit shape is invalid")
    unless (all (\count -> count > 0 && count <= rowCount) (mappa1ObservationCounts artifact)) (Left "missingness-aware artifact observation count is invalid")
    unless (all finite (mappa1FeatureMeans artifact ++ mappa1FeatureScales artifact) && all (> 0) (mappa1FeatureScales artifact)) (Left "missingness-aware artifact fit value is invalid")
    unless (validSha256 (mappa1PayloadSha256 artifact) && mappa1PayloadSha256 artifact == payloadDigest artifact) (Left "missingness-aware artifact payload digest is invalid")
  where
    request = mappa1Request artifact

parseArtifact :: Value -> AesonTypes.Parser MissingnessAwarePreprocessorArtifactV1
parseArtifact = withObject "MissingnessAwarePreprocessorArtifactV1" $ \obj -> do
    exactKeys "missingness-aware artifact" ["schemaId", "schemaVersion", "payload", "payloadSha256"] obj
    schemaId <- obj .: "schemaId"
    schemaVersion <- obj .: "schemaVersion"
    payload <- obj .: "payload"
    digest <- obj .: "payloadSha256"
    unless (schemaId == missingnessAwarePreprocessorSchemaIdV1) (fail "unsupported missingness-aware schemaId")
    unless (schemaVersion == missingnessAwarePreprocessorSchemaVersionV1) (fail "unsupported missingness-aware schemaVersion")
    unless (digest == digestValue payload) (fail "missingness-aware artifact payload digest mismatch")
    artifact <- parsePayload digest payload
    either fail pure (validateArtifact artifact)
    pure artifact

parsePayload :: String -> Value -> AesonTypes.Parser MissingnessAwarePreprocessorArtifactV1
parsePayload digest = withObject "MissingnessAwarePreprocessorPayloadV1" $ \obj -> do
    exactKeys "missingness-aware payload" payloadKeys obj
    compatibility <- obj .: "compatibilityVersion"
    sourceSchemaId <- obj .: "sourceFeatureSchemaId"
    sourceSchemaVersion <- obj .: "sourceFeatureSchemaVersion"
    sourceSignature <- obj .: "sourceFeatureSignature"
    outputSignature <- obj .: "outputSignature"
    request <- obj .: "request" >>= parseRequest
    fit <- obj .: "fit"
    safety <- obj .: "safety"
    unless (compatibility == missingnessAwarePreprocessorCompatibilityVersionV1) (fail "unsupported missingness-aware compatibilityVersion")
    unless (sourceSchemaId == missingnessAwareFeaturePanelSchemaIdV1 && sourceSchemaVersion == missingnessAwareFeaturePanelSchemaVersionV1 && sourceSignature == missingnessAwareFeatureSignatureV1) (fail "missingness-aware source feature schema mismatch")
    unless (outputSignature == missingnessAwarePreprocessorInputSignatureV1) (fail "missingness-aware output signature mismatch")
    (rowCount, counts, means, scales) <- parseFit fit
    parseSafety safety
    pure (MissingnessAwarePreprocessorArtifactV1 request rowCount counts means scales digest)

parseRequest :: Value -> AesonTypes.Parser MissingnessAwarePreprocessorFitRequestV1
parseRequest = withObject "MissingnessAwarePreprocessorFitRequestV1" $ \obj -> do
    exactKeys "missingness-aware request" requestKeys obj
    MissingnessAwarePreprocessorFitRequestV1
        <$> obj .: "registrationId"
        <*> obj .: "codeCommit"
        <*> obj .: "trainingDataSha256"
        <*> obj .: "trainingPanelSha256"
        <*> obj .: "sourceManifestSha256"
        <*> obj .: "splitManifestSha256"
        <*> obj .: "academicOrigins"
        <*> obj .: "symbol"
        <*> obj .: "intervalMs"
        <*> obj .: "horizonBars"
        <*> obj .: "trainingStartOpenTimeMs"
        <*> obj .: "trainingEndOpenTimeMs"
        <*> obj .: "validationStartOpenTimeMs"
        <*> obj .: "validationEndOpenTimeMs"
        <*> obj .: "finalHoldoutStartOpenTimeMs"
        <*> obj .: "purgeBars"
        <*> obj .: "embargoBars"
        <*> obj .: "fitAvailableAtMs"
        <*> obj .: "createdAtMs"
        <*> obj .: "randomSeed"
        <*> obj .: "runtimeVersions"
        <*> obj .: "costModelId"

parseFit :: Value -> AesonTypes.Parser (Int, [Int], [Double], [Double])
parseFit = withObject "MissingnessAwarePreprocessorFitV1" $ \obj -> do
    exactKeys "missingness-aware fit" ["trainingRowCount", "observationCounts", "featureMeans", "featureScales"] obj
    (,,,) <$> obj .: "trainingRowCount" <*> obj .: "observationCounts" <*> obj .: "featureMeans" <*> obj .: "featureScales"

parseSafety :: Value -> AesonTypes.Parser ()
parseSafety = withObject "MissingnessAwarePreprocessorSafetyV1" $ \obj -> do
    exactKeys "missingness-aware safety" safetyKeys obj
    states <- sequence [obj .: Key.fromString key | key <- safetyKeys]
    when (or states) (fail "missingness-aware preprocessor cannot carry downstream authority")

artifactValue :: MissingnessAwarePreprocessorArtifactV1 -> Value
artifactValue artifact =
    object
        [ "schemaId" .= missingnessAwarePreprocessorSchemaIdV1
        , "schemaVersion" .= missingnessAwarePreprocessorSchemaVersionV1
        , "payload" .= payloadValue artifact
        , "payloadSha256" .= mappa1PayloadSha256 artifact
        ]

payloadValue :: MissingnessAwarePreprocessorArtifactV1 -> Value
payloadValue artifact =
    object
        [ "compatibilityVersion" .= missingnessAwarePreprocessorCompatibilityVersionV1
        , "sourceFeatureSchemaId" .= missingnessAwareFeaturePanelSchemaIdV1
        , "sourceFeatureSchemaVersion" .= missingnessAwareFeaturePanelSchemaVersionV1
        , "sourceFeatureSignature" .= missingnessAwareFeatureSignatureV1
        , "outputSignature" .= missingnessAwarePreprocessorInputSignatureV1
        , "request" .= requestValue (mappa1Request artifact)
        , "fit" .= object ["trainingRowCount" .= mappa1TrainingRowCount artifact, "observationCounts" .= mappa1ObservationCounts artifact, "featureMeans" .= mappa1FeatureMeans artifact, "featureScales" .= mappa1FeatureScales artifact]
        , "safety" .= object [Key.fromString key .= False | key <- safetyKeys]
        ]

requestValue :: MissingnessAwarePreprocessorFitRequestV1 -> Value
requestValue request =
    object
        [ "registrationId" .= mapfr1RegistrationId request
        , "codeCommit" .= mapfr1CodeCommit request
        , "trainingDataSha256" .= mapfr1TrainingDataSha256 request
        , "trainingPanelSha256" .= mapfr1TrainingPanelSha256 request
        , "sourceManifestSha256" .= mapfr1SourceManifestSha256 request
        , "splitManifestSha256" .= mapfr1SplitManifestSha256 request
        , "academicOrigins" .= mapfr1AcademicOrigins request
        , "symbol" .= mapfr1Symbol request
        , "intervalMs" .= mapfr1IntervalMs request
        , "horizonBars" .= mapfr1HorizonBars request
        , "trainingStartOpenTimeMs" .= mapfr1TrainingStartOpenTimeMs request
        , "trainingEndOpenTimeMs" .= mapfr1TrainingEndOpenTimeMs request
        , "validationStartOpenTimeMs" .= mapfr1ValidationStartOpenTimeMs request
        , "validationEndOpenTimeMs" .= mapfr1ValidationEndOpenTimeMs request
        , "finalHoldoutStartOpenTimeMs" .= mapfr1FinalHoldoutStartOpenTimeMs request
        , "purgeBars" .= mapfr1PurgeBars request
        , "embargoBars" .= mapfr1EmbargoBars request
        , "fitAvailableAtMs" .= mapfr1FitAvailableAtMs request
        , "createdAtMs" .= mapfr1CreatedAtMs request
        , "randomSeed" .= mapfr1RandomSeed request
        , "runtimeVersions" .= mapfr1RuntimeVersions request
        , "costModelId" .= mapfr1CostModelId request
        ]

panelValue :: MissingnessAwareFeaturePanelV1 -> Value
panelValue panel = object ["schemaId" .= missingnessAwareFeaturePanelSchemaIdV1, "scope" .= mafp1Scope panel, "intervalMs" .= mafp1IntervalMs panel, "openTimesMs" .= V.toList (mafp1OpenTimesMs panel), "rows" .= map rowValue (mafp1Rows panel)]

rowValue :: FeatureRowV2 -> Value
rowValue row = object ["schemaId" .= frv2SchemaId row, "decisionTimeMs" .= frv2DecisionTimeMs row, "names" .= frv2Names row, "values" .= frv2Values row, "available" .= frv2Available row, "required" .= frv2Required row, "eventTimesMs" .= frv2EventTimesMs row, "availabilityTimesMs" .= frv2AvailabilityTimesMs row]

payloadDigest :: MissingnessAwarePreprocessorArtifactV1 -> String
payloadDigest = digestValue . payloadValue

digestValue :: Value -> String
digestValue value = show (hash (BL.toStrict (Aeson.encode value)) :: Digest SHA256)

mean :: [Double] -> Double
mean values = sum values / fromIntegral (length values)

scaleAround :: Double -> [Double] -> Double
scaleAround location values =
    let variance = sum [(value - location) * (value - location) | value <- values] / fromIntegral (length values)
        scale = sqrt (max 0 variance)
     in if scale <= 1.0e-12 then 1 else scale

grid :: Int64 -> Int64 -> Int64 -> [Int64]
grid start end intervalMs
    | intervalMs <= 0 || start > end = []
    | otherwise = map fromInteger [toInteger start, toInteger start + toInteger intervalMs .. toInteger end]

gridAligned :: Int64 -> Int64 -> Int64 -> Bool
gridAligned anchor intervalMs value = intervalMs > 0 && value >= anchor && (toInteger value - toInteger anchor) `mod` toInteger intervalMs == 0

boolValue :: Bool -> Double
boolValue True = 1
boolValue False = 0

validCommit :: String -> Bool
validCommit value = length value == 40 && all isHex value
  where
    isHex character = isDigit character || character `elem` ['a' .. 'f']

validSha256 :: String -> Bool
validSha256 value = length value == 64 && all isHex value
  where
    isHex character = isDigit character || character `elem` ['a' .. 'f']

validLabel :: String -> Bool
validLabel value =
    not (null value)
        && length value <= 128
        && all
            (\character -> isAscii character && (isAlphaNum character || character `elem` ("._:-" :: String)))
            value

validText :: String -> Bool
validText value = not (null value) && length value <= 256 && not (any isControl value)

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

exactKeys :: String -> [String] -> KeyMap.KeyMap Value -> AesonTypes.Parser ()
exactKeys label expected obj = unless (sort expected == sort (map Key.toString (KeyMap.keys obj))) (fail (label ++ " has missing or unknown fields"))

payloadKeys, requestKeys, safetyKeys :: [String]
payloadKeys = ["compatibilityVersion", "sourceFeatureSchemaId", "sourceFeatureSchemaVersion", "sourceFeatureSignature", "outputSignature", "request", "fit", "safety"]
requestKeys = ["registrationId", "codeCommit", "trainingDataSha256", "trainingPanelSha256", "sourceManifestSha256", "splitManifestSha256", "academicOrigins", "symbol", "intervalMs", "horizonBars", "trainingStartOpenTimeMs", "trainingEndOpenTimeMs", "validationStartOpenTimeMs", "validationEndOpenTimeMs", "finalHoldoutStartOpenTimeMs", "purgeBars", "embargoBars", "fitAvailableAtMs", "createdAtMs", "randomSeed", "runtimeVersions", "costModelId"]
safetyKeys = ["researchAdmissionAuthorized", "experimentAuthorized", "holdoutAuthorized", "modelAuthorized", "promotionAuthorized", "deploymentAuthorized", "orderAuthorized", "liveTradingAuthorized"]

zipWith5 :: (a -> b -> c -> d -> e -> f) -> [a] -> [b] -> [c] -> [d] -> [e] -> [f]
zipWith5 function (a : as) (b : bs) (c : cs) (d : ds) (e : es) = function a b c d e : zipWith5 function as bs cs ds es
zipWith5 _ _ _ _ _ _ = []
