{-# LANGUAGE OverloadedStrings #-}

module Trader.TopCombosStore (
    ComboBacktestUpdate (..),
    TopCombosMergeStats (..),
    TopCombosStore (..),
    applyComboUpdates,
    compactTopCombosPayloadForSync,
    comboFinalEquityValue,
    comboIdentityKey,
    comboMetricDouble,
    comboMetricsDouble,
    comboPerformanceKey,
    isBinancePlatformKey,
    isCoinbasePlatformKey,
    isPoloniexPlatformKey,
    isTopCombosPayload,
    mergeTopCombosPayloads,
    mergeTopCombosPayloadsWithStats,
    newTopCombosStore,
    normalizeTopCombosPayload,
    recalculateComboPerformanceFromOperation,
    resolveComboSymbol,
    normalizeComboPlatform,
    readTopCombosValueLocal,
    sanitizeComboSymbolForPlatform,
    sanitizeTopCombosValue,
    topCombosPayloadEquivalent,
    topCombosGeneratedAtMs,
    withTopCombosLock,
    writeTopCombosValue,
) where

import Control.Applicative ((<|>))
import Control.Concurrent (ThreadId, forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (SomeException, bracket, throwIO, try)
import Data.Aeson (object, toJSON, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import Data.Bool (bool)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (foldl', isPrefixOf, sortBy)
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, isJust, listToMaybe)
import qualified Data.Maybe
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)
import qualified Data.Vector as V
import System.Directory (createDirectory, createDirectoryIfMissing, doesDirectoryExist, doesFileExist, getModificationTime, removeDirectory, removeFile, renameFile, setModificationTime)
import System.FilePath (takeDirectory)
import System.IO (Handle, hClose, openTempFile)
import System.IO.Error (isAlreadyExistsError)
import Text.Read (readMaybe)

import Trader.Duration (inferPeriodsPerYear)
import Trader.Optimizer.Json (encodePretty)
import qualified Trader.Symbol as Symbol
import Trader.Text (normalizeKey, trim)

data TopCombosStore = TopCombosStore
    { tcsPath :: !FilePath
    , tcsHistoryDir :: !(Maybe FilePath)
    , tcsLock :: !(MVar ())
    }

data TopCombosMergeStats = TopCombosMergeStats
    { tcmsRawCount :: !Int
    , tcmsDroppedCount :: !Int
    , tcmsDedupedCount :: !Int
    }
    deriving (Eq, Show)

newTopCombosStore :: FilePath -> Maybe FilePath -> IO TopCombosStore
newTopCombosStore path historyDir = do
    lock <- newMVar ()
    pure TopCombosStore{tcsPath = path, tcsHistoryDir = historyDir, tcsLock = lock}

withTopCombosLock :: TopCombosStore -> IO a -> IO a
withTopCombosLock store action =
    withTopCombosProcessLock (tcsPath store) $
        withMVar (tcsLock store) (const action)

withTopCombosProcessLock :: FilePath -> IO a -> IO a
withTopCombosProcessLock path action =
    bracket (acquireProcessLock path) releaseProcessLock $ \lockPath ->
        bracket (startLockHeartbeat lockPath) stopLockHeartbeat (const action)
  where
    acquireProcessLock :: FilePath -> IO FilePath
    acquireProcessLock basePath = do
        let lockPath = basePath ++ ".lock"
        dirResult <- try (createDirectoryIfMissing True (takeDirectory lockPath)) :: IO (Either IOError ())
        case dirResult of
            Left err -> throwIO err
            Right _ -> pure ()
        let startDelay = 50000
            maxDelay = 1000000
            loop delay = do
                result <- try (createDirectory lockPath) :: IO (Either IOError ())
                case result of
                    Right _ -> pure lockPath
                    Left err ->
                        if isAlreadyExistsError err
                            then do
                                stale <- isLockStale lockPath
                                if stale
                                    then do
                                        _ <- try (removeDirectory lockPath) :: IO (Either IOError ())
                                        loop delay
                                    else do
                                        threadDelay delay
                                        loop (min maxDelay (delay * 2))
                            else throwIO err
        loop startDelay

    releaseProcessLock :: FilePath -> IO ()
    releaseProcessLock lockPath = do
        _ <- try (removeDirectory lockPath) :: IO (Either IOError ())
        pure ()

    startLockHeartbeat :: FilePath -> IO ThreadId
    startLockHeartbeat lockPath = forkIO (heartbeatLoop lockPath)

    stopLockHeartbeat :: ThreadId -> IO ()
    stopLockHeartbeat = killThread

    heartbeatLoop :: FilePath -> IO ()
    heartbeatLoop lockPath = do
        now <- getCurrentTime
        _ <- try (setModificationTime lockPath now) :: IO (Either SomeException ())
        threadDelay lockHeartbeatDelayMicros
        heartbeatLoop lockPath

    isLockStale :: FilePath -> IO Bool
    isLockStale lockPath = do
        exists <- doesDirectoryExist lockPath
        if not exists
            then pure False
            else do
                now <- getCurrentTime
                modifiedAtResult <- try (getModificationTime lockPath) :: IO (Either SomeException UTCTime)
                case modifiedAtResult of
                    Left _ -> pure True
                    Right modifiedAt -> pure (diffUTCTime now modifiedAt > processLockStaleAfter)

processLockStaleAfter :: NominalDiffTime
processLockStaleAfter = 45

lockHeartbeatDelayMicros :: Int
lockHeartbeatDelayMicros = 10 * 1000000

readTopCombosValueLocal :: FilePath -> IO (Either String Aeson.Value)
readTopCombosValueLocal path = do
    exists <- doesFileExist path
    if not exists
        then pure (Left "Top combos JSON not found.")
        else do
            contentsOrErr <- (try (BL.readFile path) :: IO (Either SomeException BL.ByteString))
            case contentsOrErr of
                Left e -> pure (Left ("Failed to read top combos JSON: " ++ show e))
                Right contents ->
                    case Aeson.eitherDecode' contents of
                        Left err -> pure (Left ("Failed to parse top combos JSON: " ++ err))
                        Right val -> pure (Right val)

writeTopCombosValue :: FilePath -> Aeson.Value -> IO (Either String ())
writeTopCombosValue path val = do
    let (filteredVal, _) = sanitizeTopCombosValue val
    let dir = takeDirectory path
    dirResult <- try (createDirectoryIfMissing True dir) :: IO (Either SomeException ())
    case dirResult of
        Left e -> pure (Left ("Failed to create top combos directory: " ++ show e))
        Right _ -> do
            tempResult <- try (openTempFile dir "top-combos-backtest.json") :: IO (Either SomeException (FilePath, Handle))
            case tempResult of
                Left e -> pure (Left ("Failed to create temp top combos file: " ++ show e))
                Right (tmpPath, handle) -> do
                    writeResult <- try (BL.hPut handle (encodePretty filteredVal <> "\n")) :: IO (Either SomeException ())
                    closeResult <- try (hClose handle) :: IO (Either SomeException ())
                    case writeResult of
                        Left e -> do
                            _ <- try (removeFile tmpPath) :: IO (Either SomeException ())
                            pure (Left ("Failed to write top combos JSON: " ++ show e))
                        Right _ ->
                            case closeResult of
                                Left e -> do
                                    _ <- try (removeFile tmpPath) :: IO (Either SomeException ())
                                    pure (Left ("Failed to finalize top combos JSON: " ++ show e))
                                Right _ -> do
                                    renameResult <- try (renameFile tmpPath path) :: IO (Either SomeException ())
                                    case renameResult of
                                        Left e -> do
                                            _ <- try (removeFile tmpPath) :: IO (Either SomeException ())
                                            pure (Left ("Failed to write top combos JSON: " ++ show e))
                                        Right _ -> pure (Right ())

compactTopCombosPayloadForSync :: Aeson.Value -> Aeson.Value
compactTopCombosPayloadForSync payload =
    case payload of
        Aeson.Object o ->
            case KM.lookup (AK.fromString "combos") o of
                Just (Aeson.Array combos) ->
                    let compacted = Aeson.Array (V.map compactCombo combos)
                     in Aeson.Object (KM.insert (AK.fromString "combos") compacted o)
                _ -> payload
        _ -> payload
  where
    compactCombo combo =
        case combo of
            Aeson.Object o -> Aeson.Object (KM.delete (AK.fromString "operations") o)
            _ -> combo

sanitizeTopCombosValue :: Aeson.Value -> (Aeson.Value, Int)
sanitizeTopCombosValue val =
    case val of
        Aeson.Object o ->
            case KM.lookup (AK.fromString "combos") o of
                Just (Aeson.Array combos) ->
                    let combosList = V.toList combos
                        (kept, changed) = foldl' apply ([], 0) combosList
                        apply (acc, count) comboVal =
                            if not (comboEquityAboveOne comboVal)
                                then (acc, count + 1)
                                else
                                    let (comboVal', updated) = sanitizeComboValue comboVal
                                     in (comboVal' : acc, count + if updated then 1 else 0)
                        combosOut = Aeson.Array (V.fromList (reverse kept))
                        o' = KM.insert (AK.fromString "combos") combosOut o
                     in (Aeson.Object o', changed)
                _ -> (val, 0)
        _ -> (val, 0)

normalizeTopCombosPayload :: Aeson.Value -> Aeson.Value
normalizeTopCombosPayload payload =
    let (sanitized, _) = sanitizeTopCombosValue payload
     in case sanitized of
            Aeson.Object o ->
                case KM.lookup (AK.fromString "combos") o of
                    Just (Aeson.Array arr) ->
                        let combos = V.toList arr
                            combosRanked = zipWith addRank [1 ..] (sortBy compareCombos combos)
                            o' = KM.insert (AK.fromString "combos") (Aeson.Array (V.fromList combosRanked)) o
                         in Aeson.Object o'
                    _ -> sanitized
            _ -> sanitized
  where
    addRank :: Int -> Aeson.Value -> Aeson.Value
    addRank rank val =
        case val of
            Aeson.Object o -> Aeson.Object (KM.insert (AK.fromString "rank") (toJSON rank) o)
            other -> other

    compareCombos :: Aeson.Value -> Aeson.Value -> Ordering
    compareCombos a b = compare (comboPerformanceKey a) (comboPerformanceKey b)

topCombosPayloadEquivalent :: Aeson.Value -> Aeson.Value -> Bool
topCombosPayloadEquivalent a b =
    stripEphemeralRootFields (normalizeTopCombosPayload a) == stripEphemeralRootFields (normalizeTopCombosPayload b)
  where
    stripEphemeralRootFields val =
        case val of
            Aeson.Object o ->
                Aeson.Object
                    ( KM.delete
                        (AK.fromString "generatedAtMs")
                        (KM.delete (AK.fromString "source") o)
                    )
            _ -> val

topCombosGeneratedAtMs :: Aeson.Value -> Maybe Int64
topCombosGeneratedAtMs val =
    case val of
        Aeson.Object o -> KM.lookup (AK.fromString "generatedAtMs") o >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

isTopCombosPayload :: Aeson.Value -> Bool
isTopCombosPayload val =
    case val of
        Aeson.Object o ->
            case KM.lookup (AK.fromString "combos") o of
                Just (Aeson.Array _) -> True
                _ -> False
        _ -> False

comboMetricValue :: String -> Aeson.Value -> Maybe Aeson.Value
comboMetricValue key val =
    case val of
        Aeson.Object o -> KM.lookup (AK.fromString key) o
        _ -> Nothing

coerceDoubleValue :: Aeson.Value -> Maybe Double
coerceDoubleValue value =
    case AT.parseMaybe Aeson.parseJSON value of
        Just v
            | isNaN v || isInfinite v -> Nothing
            | otherwise -> Just v
        Nothing ->
            case value of
                Aeson.String s ->
                    let trimmed = trim (T.unpack s)
                     in case readMaybe trimmed of
                            Just v | not (isNaN v || isInfinite v) -> Just v
                            _ -> Nothing
                _ -> Nothing

coerceIntValue :: Aeson.Value -> Maybe Int
coerceIntValue value =
    case AT.parseMaybe Aeson.parseJSON value of
        Just v -> Just v
        Nothing ->
            case value of
                Aeson.String s ->
                    let trimmed = trim (T.unpack s)
                     in readMaybe trimmed
                _ -> Nothing

comboMetricDouble :: String -> Aeson.Value -> Maybe Double
comboMetricDouble key val =
    comboMetricValue key val >>= coerceDoubleValue

comboMetricsDouble :: String -> Aeson.Value -> Maybe Double
comboMetricsDouble key val = do
    metrics <- comboMetricValue "metrics" val
    comboMetricDouble key metrics

comboFinalEquityValue :: Aeson.Value -> Maybe Double
comboFinalEquityValue val =
    comboMetricDouble "finalEquity" val <|> comboMetricsDouble "finalEquity" val

comboScoreValue :: Aeson.Value -> Maybe Double
comboScoreValue val =
    comboMetricDouble "score" val <|> comboMetricsDouble "score" val

recalculateComboPerformanceFromOperation ::
    Maybe String ->
    Maybe Double ->
    Maybe Double ->
    Aeson.Object ->
    Maybe Double ->
    Double ->
    (Double, Double, Aeson.Object)
recalculateComboPerformanceFromOperation mInterval mStoredFinalEq mStoredAnnualized metricsObj mPrevOrderEq currentOrderEq =
    let prevEq = fromMaybe 1 (positiveFiniteMaybe mPrevOrderEq)
        currentEq = fromMaybe prevEq (nonNegativeFinite currentOrderEq)
        ratioRaw =
            if prevEq > 0
                then currentEq / prevEq
                else 1
        ratio = fromMaybe 1 (nonNegativeFinite ratioRaw)
        baselineEqRaw =
            fromMaybe 1 (finiteMaybe mStoredFinalEq <|> comboMetricFromObject "finalEquity" metricsObj)
        baselineEq = max 0 baselineEqRaw
        nextFinalEqRaw = baselineEq * ratio
        nextFinalEq =
            if isFiniteDouble nextFinalEqRaw && nextFinalEqRaw >= 0
                then nextFinalEqRaw
                else baselineEq
        mExponent = comboAnnualizationExponent mInterval metricsObj baselineEq mStoredAnnualized
        nextAnnualized = comboAnnualizedReturnFromExponent nextFinalEq mExponent
        metricsObj' =
            KM.insert
                (AK.fromString "annualizedReturn")
                (toJSON nextAnnualized)
                (KM.insert (AK.fromString "finalEquity") (toJSON nextFinalEq) metricsObj)
     in (nextFinalEq, nextAnnualized, metricsObj')
  where
    comboMetricFromObject key obj =
        KM.lookup (AK.fromString key) obj >>= coerceDoubleValue

    isFiniteDouble x = not (isNaN x || isInfinite x)

    finiteMaybe mX = do
        x <- mX
        bool Nothing (Just x) (isFiniteDouble x)

    positiveFiniteMaybe mX = do
        x <- finiteMaybe mX
        bool Nothing (Just x) (x > 0)

    positiveFinite = positiveFiniteMaybe . Just

    nonNegativeFiniteMaybe mX = do
        x <- finiteMaybe mX
        bool Nothing (Just x) (x >= 0)

    nonNegativeFinite = nonNegativeFiniteMaybe . Just

    comboAnnualizationExponent interval metrics baselineEq mAnnualized =
        fromMetrics <|> fromExisting
      where
        fromMetrics = do
            periods <- comboMetricFromObject "periods" metrics >>= positiveFinite
            let ppy =
                    fromMaybe
                        (maybe 365 inferPeriodsPerYear interval)
                        (comboMetricFromObject "periodsPerYear" metrics >>= positiveFinite)
            positiveFinite (ppy / periods)

        fromExisting =
            case (positiveFinite baselineEq, finiteMaybe mAnnualized) of
                (Just base, Just ann)
                    | ann > (-1) && abs (base - 1) > 1e-9 ->
                        let denom = log base
                            numer = log (1 + ann)
                         in if abs denom <= 1e-12
                                then Nothing
                                else positiveFinite (numer / denom)
                _ -> Nothing

    comboAnnualizedReturnFromExponent finalEq mExponent =
        let fallbackRaw =
                if isFiniteDouble finalEq
                    then finalEq - 1
                    else 0
            fallback =
                if isFiniteDouble fallbackRaw
                    then max (-1) fallbackRaw
                    else 0
            fromExponent =
                case mExponent of
                    Just expo
                        | expo > 0 ->
                            if finalEq <= 0
                                then -1
                                else finalEq ** expo - 1
                    _ -> fallback
            chosen =
                if isFiniteDouble fromExponent
                    then fromExponent
                    else fallback
         in max (-1) chosen

comboEquityAboveOne :: Aeson.Value -> Bool
comboEquityAboveOne val =
    case comboFinalEquityValue val of
        Just eq -> eq > 1 && not (isInfinite eq)
        Nothing -> False

valueStringMaybe :: Aeson.Value -> Maybe String
valueStringMaybe = AT.parseMaybe Aeson.parseJSON

nonEmptyString :: String -> Maybe String
nonEmptyString s =
    case s of
        "" -> Nothing
        _ -> Just s

normalizeComboPlatform :: Maybe String -> Maybe String
normalizeComboPlatform raw =
    raw >>= nonEmptyString . normalizeKey

isBinancePlatformKey :: String -> Bool
isBinancePlatformKey key = key == "binance" || "binance" `isPrefixOf` key

isCoinbasePlatformKey :: String -> Bool
isCoinbasePlatformKey key = key == "coinbase" || "coinbase" `isPrefixOf` key

isPoloniexPlatformKey :: String -> Bool
isPoloniexPlatformKey key = key == "poloniex" || "poloniex" `isPrefixOf` key

sanitizeComboSymbolForPlatform :: Maybe String -> String -> Maybe String
sanitizeComboSymbolForPlatform platform =
    Symbol.sanitizeComboSymbolForPlatform (canonicalComboPlatform platform)

resolveComboSymbol :: Maybe String -> Maybe String -> Maybe String -> Maybe String
resolveComboSymbol platform source symbol =
    let platformHint = preferredComboPlatform platform source
     in symbol >>= sanitizeComboSymbolForPlatform platformHint

canonicalComboPlatform :: Maybe String -> Maybe String
canonicalComboPlatform platform =
    case normalizeComboPlatform platform of
        Just key | isCoinbasePlatformKey key -> Just "coinbase"
        Just key | isPoloniexPlatformKey key -> Just "poloniex"
        Just key | isBinancePlatformKey key -> Just "binance"
        other -> other

preferredComboPlatform :: Maybe String -> Maybe String -> Maybe String
preferredComboPlatform platform source =
    canonicalComboPlatform platform <|> canonicalComboPlatform source

sanitizeComboSymbolValue :: Aeson.Value -> (Aeson.Value, Bool)
sanitizeComboSymbolValue val =
    case val of
        Aeson.Object comboObj ->
            case KM.lookup (AK.fromString "params") comboObj of
                Just (Aeson.Object params) ->
                    let platformRaw =
                            (KM.lookup (AK.fromString "platform") params >>= valueStringMaybe)
                        sourceRaw = KM.lookup (AK.fromString "source") comboObj >>= valueStringMaybe
                        platform = preferredComboPlatform platformRaw sourceRaw
                        symbolRaw =
                            (KM.lookup (AK.fromString "binanceSymbol") params >>= valueStringMaybe)
                                <|> (KM.lookup (AK.fromString "symbol") params >>= valueStringMaybe)
                        hadBinance = KM.member (AK.fromString "binanceSymbol") params
                        hadSymbol = KM.member (AK.fromString "symbol") params
                        hasSymbolField = hadBinance || hadSymbol
                        sanitized = symbolRaw >>= sanitizeComboSymbolForPlatform platform
                        params' =
                            case sanitized of
                                Just sym ->
                                    let params1 =
                                            if hadBinance
                                                then KM.insert (AK.fromString "binanceSymbol") (Aeson.String (T.pack sym)) params
                                                else params
                                        params2 =
                                            if hadSymbol
                                                then KM.insert (AK.fromString "symbol") (Aeson.String (T.pack sym)) params1
                                                else params1
                                     in params2
                                Nothing ->
                                    if hasSymbolField
                                        then KM.delete (AK.fromString "symbol") (KM.delete (AK.fromString "binanceSymbol") params)
                                        else params
                        changed = params' /= params
                        comboObj' =
                            if changed
                                then KM.insert (AK.fromString "params") (Aeson.Object params') comboObj
                                else comboObj
                     in (Aeson.Object comboObj', changed)
                _ -> (val, False)
        _ -> (val, False)

sanitizeComboValue :: Aeson.Value -> (Aeson.Value, Bool)
sanitizeComboValue comboVal =
    let (comboVal1, symbolUpdated) = sanitizeComboSymbolValue comboVal
        (comboVal2, closeTimingUpdated) = applyCloseTimingRecommendationValue comboVal1
     in (comboVal2, symbolUpdated || closeTimingUpdated)

applyCloseTimingRecommendationValue :: Aeson.Value -> (Aeson.Value, Bool)
applyCloseTimingRecommendationValue val =
    case val of
        Aeson.Object comboObj ->
            case (KM.lookup (AK.fromString "params") comboObj, comboCloseTimingReport comboObj) of
                (Just (Aeson.Object params), Just report) ->
                    case validatedCloseTimingRecommendation report of
                        Just maxHoldBars ->
                            let params' = KM.insert (AK.fromString "maxHoldBars") (toJSON maxHoldBars) params
                                changed = params' /= params
                                comboObj' =
                                    if changed
                                        then KM.insert (AK.fromString "params") (Aeson.Object params') comboObj
                                        else comboObj
                             in (Aeson.Object comboObj', changed)
                        Nothing -> (val, False)
                _ -> (val, False)
        _ -> (val, False)

comboCloseTimingReport :: Aeson.Object -> Maybe Aeson.Object
comboCloseTimingReport comboObj =
    listToMaybe
        ( Data.Maybe.mapMaybe
            (\key -> KM.lookup key comboObj >>= closeTimingObject)
            [ AK.fromString "closeTimingReport"
            , AK.fromString "closeTiming"
            ]
        )
  where
    closeTimingObject (Aeson.Object report) = Just report
    closeTimingObject _ = Nothing

closeTimingRecommendationEvidenceFloor :: Int
closeTimingRecommendationEvidenceFloor = 3

closeTimingMinimumSampleCount :: Int
closeTimingMinimumSampleCount = 5

closeTimingEvidenceSampleCount :: Aeson.Object -> Maybe Int
closeTimingEvidenceSampleCount report =
    (KM.lookup (AK.fromString "positiveLiftSampleCount") report >>= coerceIntValue)
        <|> (KM.lookup (AK.fromString "profitableSupportSampleCount") report >>= coerceIntValue)
        <|> (KM.lookup (AK.fromString "positiveLiftSupportCount") report >>= coerceIntValue)

closeTimingPositiveLiftFloorMaybe :: Aeson.Object -> Maybe Int
closeTimingPositiveLiftFloorMaybe report =
    (KM.lookup (AK.fromString "minimumPositiveLiftSampleCount") report >>= coerceIntValue)
        <|> (KM.lookup (AK.fromString "minimumProfitableSupportSampleCount") report >>= coerceIntValue)
        <|> (KM.lookup (AK.fromString "positiveLiftSupportFloor") report >>= coerceIntValue)

closeTimingMinimumSampleCountMaybe :: Aeson.Object -> Maybe Int
closeTimingMinimumSampleCountMaybe report =
    (KM.lookup (AK.fromString "minimumSampleCount") report >>= coerceIntValue)
        <|> (KM.lookup (AK.fromString "minimumEvidenceSampleCount") report >>= coerceIntValue)

hasSufficientCloseTimingPositiveLiftSupport :: Int -> Int -> Int -> Bool
hasSufficientCloseTimingPositiveLiftSupport sampleCount positiveLiftSampleCount positiveLiftFloor =
    sampleCount >= positiveLiftSampleCount
        && positiveLiftSampleCount >= positiveLiftFloor

validatedCloseTimingRecommendation :: Aeson.Object -> Maybe Int
validatedCloseTimingRecommendation report = do
    recommended <- KM.lookup (AK.fromString "recommendedMaxHoldBars") report >>= coerceIntValue
    supportBound <- KM.lookup (AK.fromString "q75OptimalDuration") report >>= coerceIntValue
    sampleCount <- KM.lookup (AK.fromString "sampleCount") report >>= coerceIntValue
    medianLift <- KM.lookup (AK.fromString "medianLift") report >>= coerceDoubleValue
    let positiveLiftFloor =
            fromMaybe closeTimingRecommendationEvidenceFloor (closeTimingPositiveLiftFloorMaybe report)
        minimumSampleCount =
            fromMaybe closeTimingMinimumSampleCount (closeTimingMinimumSampleCountMaybe report)
        supportSamples =
            fromMaybe positiveLiftFloor (closeTimingEvidenceSampleCount report)
        evidenceBacked =
            hasSufficientCloseTimingPositiveLiftSupport sampleCount supportSamples positiveLiftFloor
                && recommended == supportBound
    if recommended > 0
        && supportBound > 0
        && sampleCount >= minimumSampleCount
        && medianLift > 0
        && evidenceBacked
        then Just recommended
        else Nothing

comboIdentityKey :: Aeson.Value -> Maybe BS.ByteString
comboIdentityKey val = do
    params <- comboMetricValue "params" val
    let openThr = comboMetricValue "openThreshold" val
        closeThr = comboMetricValue "closeThreshold" val
        objective = comboMetricValue "objective" val
        baseIdentity =
            object
                [ "params" .= params
                , "openThreshold" .= openThr
                , "closeThreshold" .= closeThr
                , "objective" .= objective
                ]
        identity =
            case comboMetricValue "source" val >>= comboIdentitySourceValue of
                Just source -> addField "source" source baseIdentity
                Nothing -> baseIdentity
    pure (BL.toStrict (encodePretty identity))

comboIdentitySourceValue :: Aeson.Value -> Maybe Aeson.Value
comboIdentitySourceValue raw =
    case raw of
        Aeson.String txt ->
            let source = trim (T.unpack txt)
             in if null source
                    then Nothing
                    else Just (Aeson.String (T.pack source))
        _ -> Nothing

addField :: String -> Aeson.Value -> Aeson.Value -> Aeson.Value
addField key value val =
    case val of
        Aeson.Object obj -> Aeson.Object (KM.insert (AK.fromString key) value obj)
        _ -> val

comboMergeKey :: Aeson.Value -> Maybe BS.ByteString
comboMergeKey val = do
    params <- comboMetricValue "params" val
    let openThr = comboMetricValue "openThreshold" val
        closeThr = comboMetricValue "closeThreshold" val
        objective = comboMetricValue "objective" val
        source = comboMetricValue "source" val
        identity =
            object
                [ "source" .= source
                , "params" .= params
                , "openThreshold" .= openThr
                , "closeThreshold" .= closeThr
                , "objective" .= objective
                ]
    pure (BL.toStrict (encodePretty identity))

comboPerformanceKey :: Aeson.Value -> (Double, Double, Double, Int)
comboPerformanceKey val =
    let ann =
            fromMaybe
                (negate (1 / 0))
                (comboMetricsDouble "annualizedReturn" val <|> comboMetricDouble "annualizedReturn" val)
        eq = fromMaybe 0 (comboMetricDouble "finalEquity" val <|> comboMetricsDouble "finalEquity" val)
        score = fromMaybe (negate (1 / 0)) (comboScoreValue val)
        rank =
            case val of
                Aeson.Object o -> fromMaybe maxBound (KM.lookup (AK.fromString "rank") o >>= AT.parseMaybe Aeson.parseJSON)
                _ -> maxBound
        ann' = if isNaN ann || isInfinite ann then negate (1 / 0) else ann
        eq' = if isNaN eq || isInfinite eq then 0 else eq
        score' = if isNaN score || isInfinite score then negate (1 / 0) else score
     in (negate ann', negate score', negate eq', rank)

extractPayloadSource :: Aeson.Value -> Maybe String
extractPayloadSource val =
    case val of
        Aeson.Object o -> KM.lookup (AK.fromString "source") o >>= AT.parseMaybe Aeson.parseJSON >>= cleanPayloadSource
        _ -> Nothing

cleanPayloadSource :: String -> Maybe String
cleanPayloadSource = nonEmptyString . trim

extractCombos :: Aeson.Value -> [Aeson.Value]
extractCombos val =
    case val of
        Aeson.Object o ->
            let generatedAtMs = KM.lookup (AK.fromString "generatedAtMs") o >>= AT.parseMaybe Aeson.parseJSON
                payloadSource = KM.lookup (AK.fromString "source") o >>= AT.parseMaybe Aeson.parseJSON >>= cleanPayloadSource
                applyPayloadMetadata = applyComboCreatedAt generatedAtMs . applyComboSource payloadSource
             in case KM.lookup (AK.fromString "combos") o of
                    Just (Aeson.Array arr) -> map applyPayloadMetadata (V.toList arr)
                    _ -> []
        _ -> []

applyComboSource :: Maybe String -> Aeson.Value -> Aeson.Value
applyComboSource source val =
    case (source, val) of
        (Just src, Aeson.Object o) ->
            case KM.lookup (AK.fromString "source") o of
                Just Aeson.Null -> Aeson.Object (KM.insert (AK.fromString "source") (toJSON src) o)
                Just _ -> val
                Nothing -> Aeson.Object (KM.insert (AK.fromString "source") (toJSON src) o)
        _ -> val

applyComboCreatedAt :: Maybe Int64 -> Aeson.Value -> Aeson.Value
applyComboCreatedAt createdAtMs val =
    case (createdAtMs, val) of
        (Just ts, Aeson.Object o) ->
            case KM.lookup (AK.fromString "createdAtMs") o of
                Just Aeson.Null -> Aeson.Object (KM.insert (AK.fromString "createdAtMs") (toJSON ts) o)
                Just _ -> val
                Nothing -> Aeson.Object (KM.insert (AK.fromString "createdAtMs") (toJSON ts) o)
        _ -> val

mergeTopCombosPayloads :: Int -> Int64 -> [Aeson.Value] -> Aeson.Value
mergeTopCombosPayloads maxItems now payloads = fst (mergeTopCombosPayloadsWithStats maxItems now payloads)

mergeTopCombosPayloadsWithStats :: Int -> Int64 -> [Aeson.Value] -> (Aeson.Value, TopCombosMergeStats)
mergeTopCombosPayloadsWithStats maxItems now payloads =
    let rawCount = sum (map payloadComboCount payloads)
        sanitized = map (fst . sanitizeTopCombosValue) payloads
        combos = concatMap extractCombos sanitized
        payloadSource = listToMaybe (Data.Maybe.mapMaybe extractPayloadSource sanitized)
        payloadMetadata = mergePayloadMetadata sanitized
        mergedMap = foldl' mergeCombo M.empty combos
        mergedUniqueCount = M.size mergedMap
        merged = take (max 0 maxItems) (sortBy compareCombos (M.elems mergedMap))
        ranked = zipWith addRank [1 ..] merged
        sourceVal = fromMaybe "top-combos-store" payloadSource
        sanitizedCount = length combos
        stats =
            TopCombosMergeStats
                { tcmsRawCount = rawCount
                , tcmsDroppedCount = max 0 (rawCount - sanitizedCount)
                , tcmsDedupedCount = max 0 (sanitizedCount - mergedUniqueCount)
                }
        mergedObj =
            KM.insert
                (AK.fromString "generatedAtMs")
                (toJSON now)
                ( KM.insert
                    (AK.fromString "source")
                    (toJSON sourceVal)
                    (KM.insert (AK.fromString "combos") (toJSON ranked) payloadMetadata)
                )
     in ( Aeson.Object mergedObj
        , stats
        )
  where
    payloadComboCount :: Aeson.Value -> Int
    payloadComboCount = length . extractCombos

    mergePayloadMetadata :: [Aeson.Value] -> KM.KeyMap Aeson.Value
    mergePayloadMetadata vals =
        let byFreshness = sortBy comparePayloadFreshness vals
         in foldl' mergeOne KM.empty byFreshness

    mergeOne acc val =
        case val of
            Aeson.Object o ->
                foldl'
                    ( \obj (k, v) ->
                        if isControlKey k || KM.member k obj
                            then obj
                            else KM.insert k v obj
                    )
                    acc
                    (KM.toList o)
            _ -> acc

    comparePayloadFreshness a b =
        compareMaybeDesc (topCombosGeneratedAtMs a) (topCombosGeneratedAtMs b)

    compareMaybeDesc :: (Ord a) => Maybe a -> Maybe a -> Ordering
    compareMaybeDesc lhs rhs =
        case compare lhs rhs of
            LT -> GT
            GT -> LT
            EQ -> EQ

    isControlKey key =
        key == AK.fromString "generatedAtMs"
            || key == AK.fromString "source"
            || key == AK.fromString "combos"

    mergeCombo acc comboVal =
        case comboMergeKey comboVal of
            Nothing -> acc
            Just key ->
                case M.lookup key acc of
                    Nothing -> M.insert key comboVal acc
                    Just prev -> M.insert key (pickBestCombo comboVal prev) acc

    addRank :: Int -> Aeson.Value -> Aeson.Value
    addRank rank val =
        case val of
            Aeson.Object o -> Aeson.Object (KM.insert (AK.fromString "rank") (toJSON rank) o)
            other -> other

    compareCombos :: Aeson.Value -> Aeson.Value -> Ordering
    compareCombos a b =
        let annA = comboAnnualizedReturn a
            annB = comboAnnualizedReturn b
            scoreA = sanitizeScore (fromMaybe (negate (1 / 0)) (comboScoreValue a))
            scoreB = sanitizeScore (fromMaybe (negate (1 / 0)) (comboScoreValue b))
            eqA = sanitizeEq (fromMaybe 0 (comboFinalEquityValue a))
            eqB = sanitizeEq (fromMaybe 0 (comboFinalEquityValue b))
         in case compareDesc annA annB of
                EQ ->
                    case compareDesc scoreA scoreB of
                        EQ -> compareDesc eqA eqB
                        ord -> ord
                ord -> ord

    pickBestCombo newer prev =
        let objNew = comboMetricString "objective" newer
            objPrev = comboMetricString "objective" prev
            scoreNew = comboScoreValue newer
            scorePrev = comboScoreValue prev
            scoreVal = fromMaybe (negate (1 / 0))
            finalEqNew = fromMaybe 0 (comboFinalEquityValue newer)
            finalEqPrev = fromMaybe 0 (comboFinalEquityValue prev)
         in if objNew == objPrev && (isJust scoreNew || isJust scorePrev)
                then if scoreVal scoreNew > scoreVal scorePrev then newer else prev
                else if finalEqNew > finalEqPrev then newer else prev

    comboMetricString key val =
        comboMetricValue key val >>= valueStringMaybe

    comboAnnualizedReturn val =
        let ann = fromMaybe (negate (1 / 0)) (comboMetricsDouble "annualizedReturn" val <|> comboMetricDouble "annualizedReturn" val)
         in if isNaN ann || isInfinite ann then negate (1 / 0) else ann

    sanitizeScore score
        | isNaN score || isInfinite score = negate (1 / 0)
        | otherwise = score

    sanitizeEq eq
        | isNaN eq || isInfinite eq = 0
        | otherwise = eq

    compareDesc a b
        | a > b = LT
        | a < b = GT
        | otherwise = EQ

data ComboBacktestUpdate = ComboBacktestUpdate
    { cbuMetrics :: !Aeson.Value
    , cbuFinalEquity :: !(Maybe Double)
    , cbuScore :: !(Maybe Double)
    , cbuOperations :: !(Maybe Aeson.Value)
    }

updateComboWithBacktest :: ComboBacktestUpdate -> Aeson.Value -> Aeson.Value
updateComboWithBacktest update comboVal =
    case comboVal of
        Aeson.Object o ->
            let o1 = KM.insert (AK.fromString "metrics") (cbuMetrics update) o
                o2 =
                    case cbuFinalEquity update of
                        Nothing -> o1
                        Just eq -> KM.insert (AK.fromString "finalEquity") (toJSON eq) o1
                o3 =
                    case cbuScore update of
                        Nothing -> o2
                        Just score -> KM.insert (AK.fromString "score") (toJSON score) o2
                o4 =
                    case cbuOperations update of
                        Nothing -> o3
                        Just ops -> KM.insert (AK.fromString "operations") ops o3
             in Aeson.Object o4
        _ -> comboVal

applyComboUpdates :: Int64 -> HM.HashMap BS.ByteString ComboBacktestUpdate -> Aeson.Value -> Either String (Aeson.Value, Int)
applyComboUpdates now updates val =
    case val of
        Aeson.Object o ->
            case KM.lookup (AK.fromString "combos") o of
                Just (Aeson.Array combos) -> do
                    let combosList = V.toList combos
                        (updatedCombos, updatedCount) = foldl' applyOne ([], 0 :: Int) combosList
                        applyOne (acc, count) comboVal =
                            case comboIdentityKey comboVal >>= (`HM.lookup` updates) of
                                Nothing -> (comboVal : acc, count)
                                Just upd -> (updateComboWithBacktest upd comboVal : acc, count + 1)
                        combosOut = Aeson.Array (V.fromList (reverse updatedCombos))
                        o' = KM.insert (AK.fromString "combos") combosOut (KM.insert (AK.fromString "generatedAtMs") (toJSON now) o)
                    Right (Aeson.Object o', updatedCount)
                _ -> Left "Top combos JSON missing combos array."
        _ -> Left "Top combos JSON root must be an object."
