{-# LANGUAGE OverloadedStrings #-}

module Trader.Predictors.MarketContextLinearArtifactV1 (
    MarketContextLinearFitRequestV1 (..),
    MarketContextTargetReturnV1,
    MarketContextTrainingObservationV1,
    MarketContextLinearArtifactV1,
    MarketContextLinearEstimateV1 (..),
    marketContextLinearArtifactSchemaIdV1,
    marketContextLinearArtifactSchemaVersionV1,
    marketContextLinearCompatibilityVersionV1,
    marketContextLinearSemanticModelIdV1,
    marketContextTargetReturnV1,
    mkMarketContextTrainingObservationV1,
    fitMarketContextLinearArtifactV1,
    predictMarketContextLinearV1,
    encodeMarketContextLinearArtifactV1,
    decodeMarketContextLinearArtifactV1,
    mcla1Request,
    mcla1TrainingRowCount,
    mcla1ObservedTrainingRowCount,
    mcla1Intercept,
    mcla1Beta,
    mcla1ResidualVariance,
    mcla1TrainingMse,
    mcla1PayloadSha256,
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
import Data.List (elemIndex, foldl', isPrefixOf, nub, sort)
import qualified Data.Vector as V

import Trader.PointInTimeUniverseV2 (
    pointInTimeUniverseSchemaIdV2,
    pointInTimeUniverseSchemaVersionV2,
 )
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
import Trader.Predictors.MarketContextFeaturesV2 (
    MarketContextFactorV2,
    marketContextFactorFeatureNamesV2,
    marketContextFactorSchemaSignatureV2,
    marketContextFactorSchemaVersionV2,
    mcf2BarOpenTimeMs,
    mcf2EventTimeMs,
    mcf2FeatureRow,
    mcf2IntervalMs,
    mcf2PeerSymbols,
    mcf2Quote,
    mcf2RequiredPeerCount,
    mcf2TargetSymbol,
 )
import Trader.Predictors.OhlcvInputsV2 (
    CompleteOhlcvInputsV2,
    completeOhlcvGridV2,
    completeOhlcvRowsV2,
    completeOhlcvScopeV2,
 )

-- | Immutable inputs that bind one fold-local fit to its research evidence.
data MarketContextLinearFitRequestV1 = MarketContextLinearFitRequestV1
    { mclfr1RegistrationId :: !String
    , mclfr1CodeCommit :: !String
    , mclfr1TrainingDataSha256 :: !String
    , mclfr1SourceManifestSha256 :: !String
    , mclfr1SplitManifestSha256 :: !String
    , mclfr1AcademicOrigins :: ![String]
    , mclfr1Symbol :: !String
    , mclfr1Quote :: !String
    , mclfr1RequiredPeerCount :: !Int
    , mclfr1IntervalMs :: !Int64
    , mclfr1HorizonBars :: !Int
    , mclfr1TrainingStartEventTimeMs :: !Int64
    , mclfr1TrainingEndEventTimeMs :: !Int64
    , mclfr1ValidationStartEventTimeMs :: !Int64
    , mclfr1PurgeBars :: !Int
    , mclfr1EmbargoBars :: !Int
    , mclfr1FitAvailableAtMs :: !Int64
    , mclfr1CreatedAtMs :: !Int64
    , mclfr1MinimumResidualVariance :: !Double
    , mclfr1RuntimeVersions :: ![String]
    , mclfr1CostModelId :: !String
    }
    deriving (Eq, Show)

-- | A scope-bound realized return computed from canonical complete target bars.
data MarketContextTargetReturnV1 = MarketContextTargetReturnV1
    { marketContextTargetSymbolV1 :: !String
    , marketContextTargetStartEventTimeMsV1 :: !Int64
    , marketContextTargetEventTimeMsV1 :: !Int64
    , marketContextTargetAvailabilityTimeMsV1 :: !Int64
    , marketContextTargetIntervalMsV1 :: !Int64
    , marketContextTargetHorizonBarsV1 :: !Int
    , marketContextTargetForwardReturnV1 :: !Double
    }
    deriving (Eq, Show)

-- | One grid row. Factor and label scope are both immutable abstract values.
data MarketContextTrainingObservationV1 = MarketContextTrainingObservationV1
    { mcto1Factor :: !MarketContextFactorV2
    , mcto1Target :: !MarketContextTargetReturnV1
    }
    deriving (Eq, Show)

data MarketContextLinearArtifactV1 = MarketContextLinearArtifactV1
    { mcla1Request :: !MarketContextLinearFitRequestV1
    , mcla1TrainingRowCount :: !Int
    , mcla1ObservedTrainingRowCount :: !Int
    , mcla1Intercept :: !Double
    , mcla1Beta :: !Double
    , mcla1ResidualVariance :: !Double
    , mcla1TrainingMse :: !Double
    , mcla1PayloadSha256 :: !String
    }
    deriving (Eq, Show)

data MarketContextLinearEstimateV1 = MarketContextLinearEstimateV1
    { mcle1ExpectedForwardReturn :: !Double
    , mcle1ResidualVariance :: !Double
    }
    deriving (Eq, Show)

marketContextLinearArtifactSchemaIdV1 :: String
marketContextLinearArtifactSchemaIdV1 = "point_in_time_market_context_linear_artifact_v1"

marketContextLinearArtifactSchemaVersionV1 :: Int
marketContextLinearArtifactSchemaVersionV1 = 1

marketContextLinearCompatibilityVersionV1 :: Int
marketContextLinearCompatibilityVersionV1 = 1

marketContextLinearSemanticModelIdV1 :: String
marketContextLinearSemanticModelIdV1 = "point_in_time_market_context_linear_ols_v1"

marketContextLinearModelFamilyV1 :: String
marketContextLinearModelFamilyV1 = "ordinary_least_squares_with_intercept"

marketContextLinearTargetIdV1 :: String
marketContextLinearTargetIdV1 = "simple_forward_return"

marketContextLinearFitMethodV1 :: String
marketContextLinearFitMethodV1 = "complete_grid_observed_factor_rows_only_v1"

marketContextLinearRandomSeedContractV1 :: String
marketContextLinearRandomSeedContractV1 = "none_deterministic_closed_form"

marketContextLinearPromotionStateV1 :: String
marketContextLinearPromotionStateV1 = "offline_research_only"

marketContextLinearValidationMetricsStateV1 :: String
marketContextLinearValidationMetricsStateV1 = "not_evaluated"

marketContextLinearFinalHoldoutStateV1 :: String
marketContextLinearFinalHoldoutStateV1 = "untouched"

{- | Compute a scope-bound realized return from two canonical complete target
bars. The start bar must be the factor's exact target bar and must be available
by the factor decision; the target bar is selected by the declared horizon.
-}
marketContextTargetReturnV1 ::
    MarketContextFactorV2 ->
    Int ->
    CompleteOhlcvInputsV2 ->
    Maybe MarketContextTargetReturnV1
marketContextTargetReturnV1 factor horizonBars targetInputs = do
    guard (horizonBars > 0)
    guard (completeOhlcvScopeV2 targetInputs == mcf2TargetSymbol factor)
    let (openTimes, _, _, intervalMs) = completeOhlcvGridV2 targetInputs
        rows = completeOhlcvRowsV2 targetInputs
    guard (intervalMs == mcf2IntervalMs factor)
    guard (V.length openTimes == length rows)
    startIndex <- V.elemIndex (mcf2BarOpenTimeMs factor) openTimes
    let targetIndexInteger = toInteger startIndex + toInteger horizonBars
    guard (targetIndexInteger <= toInteger (maxBound :: Int))
    startRow <- atMay rows startIndex
    targetRow <- atMay rows (fromInteger targetIndexInteger)
    (startEventTime, startAvailabilityTime, startClose) <- marketCloseWitness startRow
    (targetEventTime, targetAvailabilityTime, targetClose) <- marketCloseWitness targetRow
    guard (startEventTime == mcf2EventTimeMs factor)
    guard (startAvailabilityTime <= frv2DecisionTimeMs (mcf2FeatureRow factor))
    let expectedTargetEvent =
            toInteger startEventTime
                + toInteger horizonBars * toInteger intervalMs
    guard (toInteger targetEventTime == expectedTargetEvent)
    let forwardReturn = targetClose / startClose - 1
    guard (forwardReturn > -1 && finite forwardReturn)
    pure
        MarketContextTargetReturnV1
            { marketContextTargetSymbolV1 = completeOhlcvScopeV2 targetInputs
            , marketContextTargetStartEventTimeMsV1 = startEventTime
            , marketContextTargetEventTimeMsV1 = targetEventTime
            , marketContextTargetAvailabilityTimeMsV1 = targetAvailabilityTime
            , marketContextTargetIntervalMsV1 = intervalMs
            , marketContextTargetHorizonBarsV1 = horizonBars
            , marketContextTargetForwardReturnV1 = forwardReturn
            }

-- | Bind a validated factor to a scope-compatible realized target.
mkMarketContextTrainingObservationV1 ::
    MarketContextFactorV2 ->
    MarketContextTargetReturnV1 ->
    Maybe MarketContextTrainingObservationV1
mkMarketContextTrainingObservationV1 factor target = do
    guard (marketContextTargetSymbolV1 target == mcf2TargetSymbol factor)
    guard (marketContextTargetStartEventTimeMsV1 target == mcf2EventTimeMs factor)
    guard (marketContextTargetIntervalMsV1 target == mcf2IntervalMs factor)
    pure
        MarketContextTrainingObservationV1
            { mcto1Factor = factor
            , mcto1Target = target
            }

{- | Fit only after validating the complete chronological fold and every label.
Unavailable factor rows remain in the row count but are excluded from OLS.
-}
fitMarketContextLinearArtifactV1 ::
    MarketContextLinearFitRequestV1 ->
    [MarketContextTrainingObservationV1] ->
    Either String MarketContextLinearArtifactV1
fitMarketContextLinearArtifactV1 request observations = do
    validateRequest request
    validateObservationGrid request observations
    let observedPairs = foldr observedPair [] observations
    (intercept, beta, residualVariance, trainingMse) <-
        maybe
            (Left "market-context linear fit is degenerate or non-finite")
            Right
            (fitLinearObservedPairs (mclfr1MinimumResidualVariance request) observedPairs)
    let artifactWithoutDigest =
            MarketContextLinearArtifactV1
                { mcla1Request = request
                , mcla1TrainingRowCount = length observations
                , mcla1ObservedTrainingRowCount = length observedPairs
                , mcla1Intercept = intercept
                , mcla1Beta = beta
                , mcla1ResidualVariance = residualVariance
                , mcla1TrainingMse = trainingMse
                , mcla1PayloadSha256 = ""
                }
        digest = payloadDigest artifactWithoutDigest
        artifact = artifactWithoutDigest{mcla1PayloadSha256 = digest}
    validateArtifact artifact
    pure artifact

{- | Produce a non-actionable distribution summary only for the exact fitted scope.
Missing, incompatible, pre-validation, or non-finite evidence yields 'Nothing'.
-}
predictMarketContextLinearV1 ::
    MarketContextLinearArtifactV1 ->
    MarketContextFactorV2 ->
    Maybe MarketContextLinearEstimateV1
predictMarketContextLinearV1 artifact factor = do
    either (const Nothing) Just (validateArtifact artifact)
    let request = mcla1Request artifact
    factorValue <- observedInferenceFactor request factor
    let expected = mcla1Intercept artifact + mcla1Beta artifact * factorValue
        residualVariance = mcla1ResidualVariance artifact
    guard (expected > -1 && finite expected && finite residualVariance && residualVariance > 0)
    pure
        MarketContextLinearEstimateV1
            { mcle1ExpectedForwardReturn = expected
            , mcle1ResidualVariance = residualVariance
            }

encodeMarketContextLinearArtifactV1 :: MarketContextLinearArtifactV1 -> BL.ByteString
encodeMarketContextLinearArtifactV1 = Aeson.encode . artifactValue

decodeMarketContextLinearArtifactV1 :: BL.ByteString -> Either String MarketContextLinearArtifactV1
decodeMarketContextLinearArtifactV1 bytes = do
    value <- Aeson.eitherDecode bytes
    AesonTypes.parseEither parseArtifact value

parseArtifact :: Value -> AesonTypes.Parser MarketContextLinearArtifactV1
parseArtifact = withObject "MarketContextLinearArtifactV1" $ \obj -> do
    exactKeys
        "market-context linear artifact"
        ["schemaId", "schemaVersion", "payload", "payloadSha256"]
        obj
    schemaId <- obj .: "schemaId"
    schemaVersion <- obj .: "schemaVersion"
    payload <- obj .: "payload"
    digest <- obj .: "payloadSha256"
    unless (schemaId == marketContextLinearArtifactSchemaIdV1) (fail "unsupported market-context artifact schemaId")
    unless (schemaVersion == marketContextLinearArtifactSchemaVersionV1) (fail "unsupported market-context artifact schemaVersion")
    unless (digest == digestValue payload) (fail "market-context artifact payload digest mismatch")
    artifact <- parsePayload digest payload
    either fail pure (validateArtifact artifact)
    pure artifact

parsePayload :: String -> Value -> AesonTypes.Parser MarketContextLinearArtifactV1
parsePayload digest = withObject "MarketContextLinearArtifactPayloadV1" $ \obj -> do
    exactKeys
        "market-context linear artifact payload"
        [ "compatibilityVersion"
        , "semanticModelId"
        , "modelFamily"
        , "featureSchemaId"
        , "featureSchemaVersion"
        , "featureSchemaSignature"
        , "universeSchemaId"
        , "universeSchemaVersion"
        , "targetId"
        , "fitMethod"
        , "randomSeedContract"
        , "request"
        , "fit"
        , "safety"
        ]
        obj
    compatibilityVersion <- obj .: "compatibilityVersion"
    semanticModelId <- obj .: "semanticModelId"
    modelFamily <- obj .: "modelFamily"
    featureSchemaId <- obj .: "featureSchemaId"
    featureSchemaVersion <- obj .: "featureSchemaVersion"
    featureSchemaSignature <- obj .: "featureSchemaSignature"
    universeSchemaId <- obj .: "universeSchemaId"
    universeSchemaVersion <- obj .: "universeSchemaVersion"
    targetId <- obj .: "targetId"
    fitMethod <- obj .: "fitMethod"
    randomSeedContract <- obj .: "randomSeedContract"
    unless (compatibilityVersion == marketContextLinearCompatibilityVersionV1) (fail "unsupported market-context compatibilityVersion")
    unless (semanticModelId == marketContextLinearSemanticModelIdV1) (fail "unsupported market-context semanticModelId")
    unless (modelFamily == marketContextLinearModelFamilyV1) (fail "unsupported market-context modelFamily")
    unless (featureSchemaId == featureAvailabilitySchemaIdV2) (fail "unsupported market-context featureSchemaId")
    unless (featureSchemaVersion == marketContextFactorSchemaVersionV2) (fail "unsupported market-context featureSchemaVersion")
    unless (featureSchemaSignature == marketContextFactorSchemaSignatureV2) (fail "unsupported market-context featureSchemaSignature")
    unless (universeSchemaId == pointInTimeUniverseSchemaIdV2) (fail "unsupported market-context universeSchemaId")
    unless (universeSchemaVersion == pointInTimeUniverseSchemaVersionV2) (fail "unsupported market-context universeSchemaVersion")
    unless (targetId == marketContextLinearTargetIdV1) (fail "unsupported market-context targetId")
    unless (fitMethod == marketContextLinearFitMethodV1) (fail "unsupported market-context fitMethod")
    unless (randomSeedContract == marketContextLinearRandomSeedContractV1) (fail "unsupported market-context randomSeedContract")
    requestValue <- obj .: "request"
    fitValue <- obj .: "fit"
    safetyValue <- obj .: "safety"
    request <- parseRequest requestValue
    (rowCount, observedCount, intercept, beta, residualVariance, trainingMse) <- parseFit fitValue
    parseSafety safetyValue
    pure
        MarketContextLinearArtifactV1
            { mcla1Request = request
            , mcla1TrainingRowCount = rowCount
            , mcla1ObservedTrainingRowCount = observedCount
            , mcla1Intercept = intercept
            , mcla1Beta = beta
            , mcla1ResidualVariance = residualVariance
            , mcla1TrainingMse = trainingMse
            , mcla1PayloadSha256 = digest
            }

parseRequest :: Value -> AesonTypes.Parser MarketContextLinearFitRequestV1
parseRequest = withObject "MarketContextLinearFitRequestV1" $ \obj -> do
    exactKeys "market-context linear fit request" requestKeys obj
    request <-
        MarketContextLinearFitRequestV1
            <$> obj .: "registrationId"
            <*> obj .: "codeCommit"
            <*> obj .: "trainingDataSha256"
            <*> obj .: "sourceManifestSha256"
            <*> obj .: "splitManifestSha256"
            <*> obj .: "academicOrigins"
            <*> obj .: "symbol"
            <*> obj .: "quote"
            <*> obj .: "requiredPeerCount"
            <*> obj .: "intervalMs"
            <*> obj .: "horizonBars"
            <*> obj .: "trainingStartEventTimeMs"
            <*> obj .: "trainingEndEventTimeMs"
            <*> obj .: "validationStartEventTimeMs"
            <*> obj .: "purgeBars"
            <*> obj .: "embargoBars"
            <*> obj .: "fitAvailableAtMs"
            <*> obj .: "createdAtMs"
            <*> obj .: "minimumResidualVariance"
            <*> obj .: "runtimeVersions"
            <*> obj .: "costModelId"
    either fail pure (validateRequest request)
    pure request

parseFit :: Value -> AesonTypes.Parser (Int, Int, Double, Double, Double, Double)
parseFit = withObject "MarketContextLinearFitV1" $ \obj -> do
    exactKeys
        "market-context linear fit"
        ["trainingRowCount", "observedTrainingRowCount", "intercept", "beta", "residualVariance", "trainingMse"]
        obj
    (,,,,,)
        <$> obj .: "trainingRowCount"
        <*> obj .: "observedTrainingRowCount"
        <*> obj .: "intercept"
        <*> obj .: "beta"
        <*> obj .: "residualVariance"
        <*> obj .: "trainingMse"

parseSafety :: Value -> AesonTypes.Parser ()
parseSafety = withObject "MarketContextLinearSafetyV1" $ \obj -> do
    exactKeys "market-context linear safety" safetyKeys obj
    validationMetricsState <- obj .: "validationMetricsState"
    finalHoldoutState <- obj .: "finalHoldoutState"
    promotionState <- obj .: "promotionState"
    researchAdmissionAuthorized <- obj .: "researchAdmissionAuthorized"
    experimentAuthorized <- obj .: "experimentAuthorized"
    holdoutAuthorized <- obj .: "holdoutAuthorized"
    modelAuthorized <- obj .: "modelAuthorized"
    promotionAuthorized <- obj .: "promotionAuthorized"
    deploymentAuthorized <- obj .: "deploymentAuthorized"
    orderAuthorized <- obj .: "orderAuthorized"
    liveTradingAuthorized <- obj .: "liveTradingAuthorized"
    unless (validationMetricsState == marketContextLinearValidationMetricsStateV1) (fail "market-context validation metrics state must be not_evaluated")
    unless (finalHoldoutState == marketContextLinearFinalHoldoutStateV1) (fail "market-context final holdout state must be untouched")
    unless (promotionState == marketContextLinearPromotionStateV1) (fail "market-context promotion state must be offline_research_only")
    when
        ( or
            [ researchAdmissionAuthorized
            , experimentAuthorized
            , holdoutAuthorized
            , modelAuthorized
            , promotionAuthorized
            , deploymentAuthorized
            , orderAuthorized
            , liveTradingAuthorized
            ]
        )
        (fail "market-context component artifact cannot carry downstream authority")

artifactValue :: MarketContextLinearArtifactV1 -> Value
artifactValue artifact =
    object
        [ "schemaId" .= marketContextLinearArtifactSchemaIdV1
        , "schemaVersion" .= marketContextLinearArtifactSchemaVersionV1
        , "payload" .= payloadValue artifact
        , "payloadSha256" .= mcla1PayloadSha256 artifact
        ]

payloadValue :: MarketContextLinearArtifactV1 -> Value
payloadValue artifact =
    object
        [ "compatibilityVersion" .= marketContextLinearCompatibilityVersionV1
        , "semanticModelId" .= marketContextLinearSemanticModelIdV1
        , "modelFamily" .= marketContextLinearModelFamilyV1
        , "featureSchemaId" .= featureAvailabilitySchemaIdV2
        , "featureSchemaVersion" .= marketContextFactorSchemaVersionV2
        , "featureSchemaSignature" .= marketContextFactorSchemaSignatureV2
        , "universeSchemaId" .= pointInTimeUniverseSchemaIdV2
        , "universeSchemaVersion" .= pointInTimeUniverseSchemaVersionV2
        , "targetId" .= marketContextLinearTargetIdV1
        , "fitMethod" .= marketContextLinearFitMethodV1
        , "randomSeedContract" .= marketContextLinearRandomSeedContractV1
        , "request" .= requestValue (mcla1Request artifact)
        , "fit"
            .= object
                [ "trainingRowCount" .= mcla1TrainingRowCount artifact
                , "observedTrainingRowCount" .= mcla1ObservedTrainingRowCount artifact
                , "intercept" .= mcla1Intercept artifact
                , "beta" .= mcla1Beta artifact
                , "residualVariance" .= mcla1ResidualVariance artifact
                , "trainingMse" .= mcla1TrainingMse artifact
                ]
        , "safety"
            .= object
                [ "validationMetricsState" .= marketContextLinearValidationMetricsStateV1
                , "finalHoldoutState" .= marketContextLinearFinalHoldoutStateV1
                , "promotionState" .= marketContextLinearPromotionStateV1
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

requestValue :: MarketContextLinearFitRequestV1 -> Value
requestValue request =
    object
        [ "registrationId" .= mclfr1RegistrationId request
        , "codeCommit" .= mclfr1CodeCommit request
        , "trainingDataSha256" .= mclfr1TrainingDataSha256 request
        , "sourceManifestSha256" .= mclfr1SourceManifestSha256 request
        , "splitManifestSha256" .= mclfr1SplitManifestSha256 request
        , "academicOrigins" .= mclfr1AcademicOrigins request
        , "symbol" .= mclfr1Symbol request
        , "quote" .= mclfr1Quote request
        , "requiredPeerCount" .= mclfr1RequiredPeerCount request
        , "intervalMs" .= mclfr1IntervalMs request
        , "horizonBars" .= mclfr1HorizonBars request
        , "trainingStartEventTimeMs" .= mclfr1TrainingStartEventTimeMs request
        , "trainingEndEventTimeMs" .= mclfr1TrainingEndEventTimeMs request
        , "validationStartEventTimeMs" .= mclfr1ValidationStartEventTimeMs request
        , "purgeBars" .= mclfr1PurgeBars request
        , "embargoBars" .= mclfr1EmbargoBars request
        , "fitAvailableAtMs" .= mclfr1FitAvailableAtMs request
        , "createdAtMs" .= mclfr1CreatedAtMs request
        , "minimumResidualVariance" .= mclfr1MinimumResidualVariance request
        , "runtimeVersions" .= mclfr1RuntimeVersions request
        , "costModelId" .= mclfr1CostModelId request
        ]

validateRequest :: MarketContextLinearFitRequestV1 -> Either String ()
validateRequest request
    | not (validLabel (mclfr1RegistrationId request)) = Left "market-context registrationId is invalid"
    | not (validCommit (mclfr1CodeCommit request)) = Left "market-context codeCommit is invalid"
    | not (all validSha256 requestDigests) = Left "market-context provenance digest is invalid"
    | null origins || not (all validAcademicOrigin origins) || length origins /= length (nub origins) = Left "market-context academicOrigins are invalid"
    | not (validSymbol (mclfr1Symbol request)) = Left "market-context symbol is invalid"
    | not (validSymbol (mclfr1Quote request)) = Left "market-context quote is invalid"
    | mclfr1RequiredPeerCount request <= 0 = Left "market-context requiredPeerCount must be positive"
    | interval <= 0 = Left "market-context intervalMs must be positive"
    | horizon <= 0 = Left "market-context horizonBars must be positive"
    | purge < horizon = Left "market-context purgeBars must cover the target horizon"
    | embargo < 0 = Left "market-context embargoBars cannot be negative"
    | startTime < 0 || endTime < startTime = Left "market-context training event range is invalid"
    | (toInteger endTime - toInteger startTime) `mod` toInteger interval /= 0 = Left "market-context training event range is not an exact grid"
    | rowCountInteger > toInteger (maxBound :: Int) = Left "market-context training event grid is too large"
    | validationStart <= endTime = Left "market-context validation must follow training"
    | (toInteger validationStart - toInteger startTime) `mod` toInteger interval /= 0 = Left "market-context validation start is off grid"
    | toInteger validationStart - toInteger endTime < gapMs = Left "market-context purge and embargo gap is insufficient"
    | toInteger fitAvailableAt < lastTargetEvent || fitAvailableAt > validationStart = Left "market-context fit availability crosses its causal fold boundary"
    | createdAt < fitAvailableAt = Left "market-context artifact creation predates fit availability"
    | not (finite minimumVariance && minimumVariance > 0) = Left "market-context residual variance floor is invalid"
    | null runtimes || not (all validText runtimes) || length runtimes /= length (nub runtimes) = Left "market-context runtimeVersions are invalid"
    | not (validLabel (mclfr1CostModelId request)) = Left "market-context costModelId is invalid"
    | otherwise = Right ()
  where
    requestDigests =
        [ mclfr1TrainingDataSha256 request
        , mclfr1SourceManifestSha256 request
        , mclfr1SplitManifestSha256 request
        ]
    origins = mclfr1AcademicOrigins request
    interval = mclfr1IntervalMs request
    horizon = mclfr1HorizonBars request
    purge = mclfr1PurgeBars request
    embargo = mclfr1EmbargoBars request
    startTime = mclfr1TrainingStartEventTimeMs request
    endTime = mclfr1TrainingEndEventTimeMs request
    validationStart = mclfr1ValidationStartEventTimeMs request
    fitAvailableAt = mclfr1FitAvailableAtMs request
    createdAt = mclfr1CreatedAtMs request
    minimumVariance = mclfr1MinimumResidualVariance request
    runtimes = mclfr1RuntimeVersions request
    rowCountInteger = (toInteger endTime - toInteger startTime) `div` toInteger interval + 1
    gapMs = (toInteger purge + toInteger embargo) * toInteger interval
    lastTargetEvent = toInteger endTime + toInteger horizon * toInteger interval

validateObservationGrid :: MarketContextLinearFitRequestV1 -> [MarketContextTrainingObservationV1] -> Either String ()
validateObservationGrid request observations
    | length observations /= expectedRowCount request = Left "market-context training row count does not cover the full fold grid"
    | eventTimes /= expectedEventTimes request = Left "market-context training rows are not the exact ordered fold grid"
    | otherwise = mapM_ (validateObservation request) observations
  where
    eventTimes = map (mcf2EventTimeMs . mcto1Factor) observations

validateObservation :: MarketContextLinearFitRequestV1 -> MarketContextTrainingObservationV1 -> Either String ()
validateObservation request observation
    | mcf2TargetSymbol factor /= mclfr1Symbol request = Left "market-context training symbol does not match artifact scope"
    | mcf2Quote factor /= mclfr1Quote request = Left "market-context training quote does not match artifact scope"
    | mcf2RequiredPeerCount factor /= mclfr1RequiredPeerCount request = Left "market-context training peer count does not match artifact scope"
    | length (mcf2PeerSymbols factor) /= mcf2RequiredPeerCount factor = Left "market-context training peer identity count is invalid"
    | mcf2IntervalMs factor /= interval = Left "market-context training interval does not match artifact scope"
    | not (validFactorRowForEvent interval decisionEvent featureRow) = Left "market-context factor row is malformed or causally incompatible"
    | marketContextTargetSymbolV1 target /= mcf2TargetSymbol factor = Left "market-context target symbol does not match factor scope"
    | marketContextTargetStartEventTimeMsV1 target /= decisionEvent = Left "market-context target start does not match factor event"
    | marketContextTargetIntervalMsV1 target /= interval = Left "market-context target interval does not match artifact scope"
    | marketContextTargetHorizonBarsV1 target /= mclfr1HorizonBars request = Left "market-context target horizon does not match artifact scope"
    | toInteger targetEvent /= expectedTargetEvent = Left "market-context target event does not match the registered horizon"
    | targetAvailability < targetEvent || targetAvailability > mclfr1FitAvailableAtMs request = Left "market-context target was not available by the fit cutoff"
    | targetReturn <= -1 || not (finite targetReturn) = Left "market-context target return is invalid"
    | otherwise = Right ()
  where
    factor = mcto1Factor observation
    target = mcto1Target observation
    featureRow = mcf2FeatureRow factor
    decisionEvent = mcf2EventTimeMs factor
    targetEvent = marketContextTargetEventTimeMsV1 target
    targetAvailability = marketContextTargetAvailabilityTimeMsV1 target
    targetReturn = marketContextTargetForwardReturnV1 target
    interval = mclfr1IntervalMs request
    expectedTargetEvent = toInteger decisionEvent + toInteger (mclfr1HorizonBars request) * toInteger interval

validateArtifact :: MarketContextLinearArtifactV1 -> Either String ()
validateArtifact artifact = do
    validateRequest (mcla1Request artifact)
    unless (mcla1TrainingRowCount artifact == expectedRowCount (mcla1Request artifact)) (Left "market-context artifact training row count is invalid")
    unless (mcla1ObservedTrainingRowCount artifact >= 3 && mcla1ObservedTrainingRowCount artifact <= mcla1TrainingRowCount artifact) (Left "market-context artifact observed row count is invalid")
    unless (all finite [mcla1Intercept artifact, mcla1Beta artifact, mcla1ResidualVariance artifact, mcla1TrainingMse artifact]) (Left "market-context artifact contains non-finite fit values")
    unless (mcla1ResidualVariance artifact >= mclfr1MinimumResidualVariance (mcla1Request artifact)) (Left "market-context artifact residual variance is below its floor")
    unless (mcla1TrainingMse artifact >= 0) (Left "market-context artifact training MSE is negative")
    unless (validSha256 (mcla1PayloadSha256 artifact) && mcla1PayloadSha256 artifact == payloadDigest artifact) (Left "market-context artifact payload digest is invalid")

observedPair :: MarketContextTrainingObservationV1 -> [(Double, Double)] -> [(Double, Double)]
observedPair observation pairs =
    case (frv2Available row, frv2Values row) of
        ([True], [value]) -> (value, marketContextTargetForwardReturnV1 (mcto1Target observation)) : pairs
        _ -> pairs
  where
    row = mcf2FeatureRow (mcto1Factor observation)

observedInferenceFactor :: MarketContextLinearFitRequestV1 -> MarketContextFactorV2 -> Maybe Double
observedInferenceFactor request factor = do
    guard (mcf2TargetSymbol factor == mclfr1Symbol request)
    guard (mcf2Quote factor == mclfr1Quote request)
    guard (mcf2RequiredPeerCount factor == mclfr1RequiredPeerCount request)
    guard (length (mcf2PeerSymbols factor) == mcf2RequiredPeerCount factor)
    guard (mcf2IntervalMs factor == mclfr1IntervalMs request)
    let eventTime = mcf2EventTimeMs factor
        row = mcf2FeatureRow factor
    guard (eventTime >= mclfr1ValidationStartEventTimeMs request)
    guard
        ( (toInteger eventTime - toInteger (mclfr1ValidationStartEventTimeMs request))
            `mod` toInteger (mclfr1IntervalMs request)
            == 0
        )
    guard (validFactorRowForEvent (mclfr1IntervalMs request) eventTime row)
    [True] <- pure (frv2Available row)
    [value] <- pure (frv2Values row)
    guard (finite value)
    pure value

validFactorRowForEvent :: Int64 -> Int64 -> FeatureRowV2 -> Bool
validFactorRowForEvent interval eventTime row =
    interval > 0
        && eventTime >= 0
        && frv2SchemaId row == featureAvailabilitySchemaIdV2
        && frv2Names row == marketContextFactorFeatureNamesV2
        && frv2Required row == [False]
        && length (frv2Values row) == 1
        && length (frv2Available row) == 1
        && validDecisionTime
        && case (frv2Available row, frv2Values row, frv2EventTimesMs row, frv2AvailabilityTimesMs row) of
            ([True], [value], [Just featureEvent], [Just availabilityTime]) ->
                featureEvent >= eventTime
                    && availabilityTime >= featureEvent
                    && availabilityTime <= frv2DecisionTimeMs row
                    && value > -1
                    && finite value
            ([False], [value], [Nothing], [Nothing]) -> value == 0
            _ -> False
  where
    decisionTime = frv2DecisionTimeMs row
    validDecisionTime =
        decisionTime >= eventTime
            && toInteger decisionTime < toInteger eventTime + toInteger interval

marketCloseWitness :: FeatureRowV2 -> Maybe (Int64, Int64, Double)
marketCloseWitness row = do
    closeIndex <- elemIndex "market.close" (frv2Names row)
    True <- atMay (frv2Available row) closeIndex
    value <- atMay (frv2Values row) closeIndex
    Just eventTime <- atMay (frv2EventTimesMs row) closeIndex
    Just availabilityTime <- atMay (frv2AvailabilityTimesMs row) closeIndex
    guard (eventTime >= 0)
    guard (availabilityTime >= eventTime && availabilityTime <= frv2DecisionTimeMs row)
    guard (value > 0 && finite value)
    pure (eventTime, availabilityTime, value)

atMay :: [a] -> Int -> Maybe a
atMay values index
    | index < 0 = Nothing
    | otherwise = case drop index values of
        value : _ -> Just value
        [] -> Nothing

fitLinearObservedPairs :: Double -> [(Double, Double)] -> Maybe (Double, Double, Double, Double)
fitLinearObservedPairs minimumVariance pairs = do
    guard (length pairs >= 3)
    (count, meanX, meanY, sumSquaresX, sumCross) <- foldl' updateMoments (Just (0, 0, 0, 0, 0)) pairs
    guard (count == length pairs)
    guard (sumSquaresX > 0 && finite sumSquaresX && finite sumCross)
    let beta = sumCross / sumSquaresX
        intercept = meanY - beta * meanX
    guard (finite beta && finite intercept)
    sumSquaredErrors <- foldl' (accumulateSquaredError intercept beta) (Just 0) pairs
    let countD = fromIntegral count
        residualDegrees = fromIntegral (count - 2)
        trainingMse = sumSquaredErrors / countD
        residualVariance = max minimumVariance (sumSquaredErrors / residualDegrees)
    guard (finite trainingMse && trainingMse >= 0)
    guard (finite residualVariance && residualVariance > 0)
    pure (intercept, beta, residualVariance, trainingMse)

updateMoments :: Maybe (Int, Double, Double, Double, Double) -> (Double, Double) -> Maybe (Int, Double, Double, Double, Double)
updateMoments Nothing _ = Nothing
updateMoments (Just (count, meanX, meanY, sumSquaresX, sumCross)) (xValue, yValue) = do
    guard (finite xValue && finite yValue)
    let nextCount = count + 1
        nextCountD = fromIntegral nextCount
        deltaX = xValue - meanX
        nextMeanX = meanX + deltaX / nextCountD
        deltaY = yValue - meanY
        nextMeanY = meanY + deltaY / nextCountD
        nextSumSquaresX = sumSquaresX + deltaX * (xValue - nextMeanX)
        nextSumCross = sumCross + deltaX * (yValue - nextMeanY)
    guard (all finite [nextMeanX, nextMeanY, nextSumSquaresX, nextSumCross])
    pure (nextCount, nextMeanX, nextMeanY, nextSumSquaresX, nextSumCross)

accumulateSquaredError :: Double -> Double -> Maybe Double -> (Double, Double) -> Maybe Double
accumulateSquaredError _ _ Nothing _ = Nothing
accumulateSquaredError intercept beta (Just total) (xValue, yValue) = do
    let residual = yValue - (intercept + beta * xValue)
        nextTotal = total + residual * residual
    guard (finite residual && finite nextTotal && nextTotal >= 0)
    pure nextTotal

expectedRowCount :: MarketContextLinearFitRequestV1 -> Int
expectedRowCount request =
    fromInteger
        ( ( toInteger (mclfr1TrainingEndEventTimeMs request)
                - toInteger (mclfr1TrainingStartEventTimeMs request)
          )
            `div` toInteger (mclfr1IntervalMs request)
            + 1
        )

expectedEventTimes :: MarketContextLinearFitRequestV1 -> [Int64]
expectedEventTimes request =
    [ fromInteger
        ( toInteger (mclfr1TrainingStartEventTimeMs request)
            + toInteger index * toInteger (mclfr1IntervalMs request)
        )
    | index <- [0 .. expectedRowCount request - 1]
    ]

payloadDigest :: MarketContextLinearArtifactV1 -> String
payloadDigest = digestValue . payloadValue

digestValue :: Value -> String
digestValue value = show (hash (BL.toStrict (Aeson.encode value)) :: Digest SHA256)

exactKeys :: String -> [String] -> KeyMap.KeyMap Value -> AesonTypes.Parser ()
exactKeys label expected obj =
    unless
        (sort expected == sort (map Key.toString (KeyMap.keys obj)))
        (fail (label ++ " has missing or unknown fields"))

requestKeys :: [String]
requestKeys =
    [ "registrationId"
    , "codeCommit"
    , "trainingDataSha256"
    , "sourceManifestSha256"
    , "splitManifestSha256"
    , "academicOrigins"
    , "symbol"
    , "quote"
    , "requiredPeerCount"
    , "intervalMs"
    , "horizonBars"
    , "trainingStartEventTimeMs"
    , "trainingEndEventTimeMs"
    , "validationStartEventTimeMs"
    , "purgeBars"
    , "embargoBars"
    , "fitAvailableAtMs"
    , "createdAtMs"
    , "minimumResidualVariance"
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

validCommit :: String -> Bool
validCommit value = length value == 40 && all isLowerHex value

validSha256 :: String -> Bool
validSha256 value = length value == 64 && all isLowerHex value

isLowerHex :: Char -> Bool
isLowerHex character = isAscii character && (isDigit character || character >= 'a' && character <= 'f')

validSymbol :: String -> Bool
validSymbol value = not (null value) && all (\character -> isAscii character && (isUpper character || isDigit character)) value

validLabel :: String -> Bool
validLabel value =
    not (null value)
        && all
            (\character -> isAscii character && (isAlphaNum character || character `elem` ("_.:-" :: String)))
            value

validText :: String -> Bool
validText value = not (null value) && all (\character -> isAscii character && not (isControl character)) value

validAcademicOrigin :: String -> Bool
validAcademicOrigin value =
    validText value
        && any
            (`isPrefixOf` value)
            [ "https://"
            , "doi:"
            , "arxiv:"
            , "ssrn:"
            ]

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)
