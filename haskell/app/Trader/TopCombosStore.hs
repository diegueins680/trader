{-# LANGUAGE OverloadedStrings #-}

module Trader.TopCombosStore (
    ComboBacktestUpdate (..),
    TopCombosStore (..),
    applyComboUpdates,
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
    newTopCombosStore,
    normalizeComboPlatform,
    readTopCombosValueLocal,
    sanitizeComboSymbolForPlatform,
    sanitizeTopCombosValue,
    topCombosGeneratedAtMs,
    withTopCombosLock,
    writeTopCombosValue,
) where

import Control.Applicative ((<|>))
import Control.Concurrent (ThreadId, forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (MVar, newMVar, withMVar)
import Control.Exception (SomeException, bracket, throwIO, try)
import Data.Aeson (Value (..), object, toJSON, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAsciiUpper, isDigit, toUpper)
import Data.Int (Int64)
import Data.List (foldl', isPrefixOf, isSuffixOf, sortBy)
import qualified Data.HashMap.Strict as HM
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, isJust, listToMaybe, maybeToList)
import qualified Data.Maybe
import qualified Data.Text as T
import Data.Time.Clock (NominalDiffTime, diffUTCTime, getCurrentTime)
import qualified Data.Vector as V
import System.Directory (createDirectory, createDirectoryIfMissing, doesDirectoryExist, doesFileExist, getModificationTime, removeDirectory, renameFile, setModificationTime)
import System.FilePath (takeDirectory)
import System.IO (Handle, hClose, openTempFile)
import System.IO.Error (isAlreadyExistsError)
import Text.Read (readMaybe)

import Trader.Optimizer.Json (encodePretty)
import Trader.Symbol (commonQuotes, sanitizeSymbolForPlatform)
import Trader.Text (normalizeKey, trim)


data TopCombosStore = TopCombosStore
    { tcsPath :: !FilePath
    , tcsHistoryDir :: !(Maybe FilePath)
    , tcsLock :: !(MVar ())
    }

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
                modifiedAt <- getModificationTime lockPath
                pure (diffUTCTime now modifiedAt > processLockStaleAfter)

processLockStaleAfter :: NominalDiffTime
processLockStaleAfter = 900

lockHeartbeatDelayMicros :: Int
lockHeartbeatDelayMicros = 60 * 1000000

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
                    _ <- try (BL.hPut handle (encodePretty filteredVal <> "\n")) :: IO (Either SomeException ())
                    hClose handle
                    renameResult <- try (renameFile tmpPath path) :: IO (Either SomeException ())
                    case renameResult of
                        Left e -> pure (Left ("Failed to write top combos JSON: " ++ show e))
                        Right _ -> pure (Right ())

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
                                    let (comboVal', updated) = sanitizeComboSymbolValue comboVal
                                     in (comboVal' : acc, count + if updated then 1 else 0)
                        combosOut = Aeson.Array (V.fromList (reverse kept))
                        o' = KM.insert (AK.fromString "combos") combosOut o
                     in (Aeson.Object o', changed)
                _ -> (val, 0)
        _ -> (val, 0)

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
                Aeson.Bool v -> Just (if v then 1 else 0)
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

comboEquityAboveOne :: Aeson.Value -> Bool
comboEquityAboveOne val =
    case comboFinalEquityValue val of
        Just eq -> eq > 1 && not (isInfinite eq)
        Nothing -> False

valueStringMaybe :: Aeson.Value -> Maybe String
valueStringMaybe = AT.parseMaybe Aeson.parseJSON

normalizeComboPlatform :: Maybe String -> Maybe String
normalizeComboPlatform raw =
    case raw of
        Nothing -> Nothing
        Just v ->
            let key = normalizeKey v
             in if null key then Nothing else Just key

isBinancePlatformKey :: String -> Bool
isBinancePlatformKey key = key == "binance" || "binance" `isPrefixOf` key

isCoinbasePlatformKey :: String -> Bool
isCoinbasePlatformKey key = key == "coinbase" || "coinbase" `isPrefixOf` key

isPoloniexPlatformKey :: String -> Bool
isPoloniexPlatformKey key = key == "poloniex" || "poloniex" `isPrefixOf` key

sanitizeComboSymbolForPlatform :: Maybe String -> String -> Maybe String
sanitizeComboSymbolForPlatform platform raw =
    case normalizeComboPlatform platform of
        Just key | isCoinbasePlatformKey key -> sanitizeSymbolForPlatform (Just "coinbase") raw
        Just key | isPoloniexPlatformKey key -> sanitizeSymbolForPlatform (Just "poloniex") raw
        Just key
            | isBinancePlatformKey key ->
                sanitizeBinanceComboSymbol raw <|> sanitizeSymbolForPlatform (Just "binance") raw
        _ -> sanitizeBinanceComboSymbol raw <|> sanitizeSymbolForPlatform platform raw

sanitizeBinanceComboSymbol :: String -> Maybe String
sanitizeBinanceComboSymbol raw =
    let s = normalizeSymbol raw
        tokens = splitAlphaNumTokens s
        isValid sym =
            let n = length sym
             in n >= 3 && n <= 30 && sym `notElem` commonQuotes && all isAsciiAlphaNum sym
        pickTokenCandidate =
            case tokens of
                [] -> Nothing
                [a] -> if isValid a then Just a else Nothing
                a : b : _rest ->
                    let joined = a ++ b
                     in if isValid a && endsWithQuote a
                            then Just a
                            else
                                if b `elem` commonQuotes && isValid joined
                                    then Just joined
                                    else
                                        if isValid a && isSuffixToken b
                                            then Just a
                                            else Nothing
        pickQuoteSuffix = trimBinanceComboSuffix s
        isSuffixToken = any isDigit
     in pickQuoteSuffix <|> pickTokenCandidate <|> if isValidBinanceSymbol s then Just s else Nothing

splitAlphaNumTokens :: String -> [String]
splitAlphaNumTokens =
    filter (not . null) . foldr step [""]
  where
    step c acc@(w : ws)
        | isAsciiAlphaNum c = (c : w) : ws
        | otherwise = "" : acc
    step _ [] = []

endsWithQuote :: String -> Bool
endsWithQuote token = any (`isSuffixOf` token) commonQuotes

trimBinanceComboSuffix :: String -> Maybe String
trimBinanceComboSuffix raw =
    let compact = filter isAsciiAlphaNum (normalizeSymbol raw)
        best = foldl' pickLongest Nothing (concatMap (trimQuoteCandidates compact) commonQuotes)
     in best
  where
    pickLongest acc candidate =
        case acc of
            Nothing -> Just candidate
            Just prev -> if length candidate > length prev then Just candidate else acc

trimQuoteCandidates :: String -> String -> [String]
trimQuoteCandidates compact quote =
    let positions = findSubstrPositions quote compact
        total = length compact
        quoteLen = length quote
     in [ candidate
        | idx <- positions
        , let end = idx + quoteLen
        , end < total
        , let suffix = drop end compact
        , any isDigit suffix
        , let candidate = take end compact
        , isValidBinanceSymbol candidate
        , candidate `notElem` commonQuotes
        ]

findSubstrPositions :: String -> String -> [Int]
findSubstrPositions needle hay =
    let go _ [] = []
        go i xs@(_ : rest) =
            if needle `isPrefixOf` xs
                then i : go (i + 1) rest
                else go (i + 1) rest
     in if null needle then [] else go 0 hay

isValidBinanceSymbol :: String -> Bool
isValidBinanceSymbol s =
    let n = length s
     in n >= 3 && n <= 30 && all isAsciiAlphaNum s

isAsciiAlphaNum :: Char -> Bool
isAsciiAlphaNum c =
    isAsciiUpper c || isDigit c

sanitizeComboSymbolValue :: Aeson.Value -> (Aeson.Value, Bool)
sanitizeComboSymbolValue val =
    case val of
        Aeson.Object comboObj ->
            case KM.lookup (AK.fromString "params") comboObj of
                Just (Aeson.Object params) ->
                    let platform =
                            (KM.lookup (AK.fromString "platform") params >>= valueStringMaybe)
                                <|> (KM.lookup (AK.fromString "source") comboObj >>= valueStringMaybe)
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

comboIdentityKey :: Aeson.Value -> Maybe BS.ByteString
comboIdentityKey val = do
    params <- comboMetricValue "params" val
    let openThr = comboMetricValue "openThreshold" val
        closeThr = comboMetricValue "closeThreshold" val
        objective = comboMetricValue "objective" val
        identity =
            object
                [ "params" .= params
                , "openThreshold" .= openThr
                , "closeThreshold" .= closeThr
                , "objective" .= objective
                ]
    pure (BL.toStrict (encodePretty identity))

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
        score = fromMaybe (negate (1 / 0)) (comboMetricDouble "score" val)
        rank =
            case val of
                Aeson.Object o -> fromMaybe maxBound (KM.lookup (AK.fromString "rank") o >>= AT.parseMaybe Aeson.parseJSON)
                _ -> maxBound
        ann' = if isNaN ann || isInfinite ann then negate (1 / 0) else ann
        eq' = if isNaN eq || isInfinite eq then 0 else eq
        score' = if isNaN score || isInfinite score then negate (1 / 0) else score
     in (negate ann', negate eq', negate score', rank)

extractPayloadSource :: Aeson.Value -> Maybe String
extractPayloadSource val =
    case val of
        Aeson.Object o -> KM.lookup (AK.fromString "source") o >>= AT.parseMaybe Aeson.parseJSON >>= cleanPayloadSource
        _ -> Nothing
  where
    cleanPayloadSource raw =
        let s = trim raw
         in if null s then Nothing else Just s

extractCombos :: Aeson.Value -> [Aeson.Value]
extractCombos val =
    case val of
        Aeson.Object o ->
            let generatedAtMs = KM.lookup (AK.fromString "generatedAtMs") o >>= AT.parseMaybe Aeson.parseJSON
                applyCreatedAt = applyComboCreatedAt generatedAtMs
             in case KM.lookup (AK.fromString "combos") o of
                    Just (Aeson.Array arr) -> map applyCreatedAt (V.toList arr)
                    _ -> []
        _ -> []

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
mergeTopCombosPayloads maxItems now payloads =
    let sanitized = map (fst . sanitizeTopCombosValue) payloads
        combos = concatMap extractCombos sanitized
        payloadSource = listToMaybe (Data.Maybe.mapMaybe extractPayloadSource sanitized)
        mergedMap = foldl' mergeCombo M.empty combos
        merged = take (max 0 maxItems) (sortBy compareCombos (M.elems mergedMap))
        ranked = zipWith addRank [1 ..] merged
        sourceVal = fromMaybe "top-combos-store" payloadSource
     in object
            [ "generatedAtMs" .= now
            , "source" .= sourceVal
            , "combos" .= ranked
            ]
  where
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
            scoreA = sanitizeScore (fromMaybe (negate (1 / 0)) (comboMetricDouble "score" a))
            scoreB = sanitizeScore (fromMaybe (negate (1 / 0)) (comboMetricDouble "score" b))
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
            scoreNew = comboMetricDouble "score" newer
            scorePrev = comboMetricDouble "score" prev
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

normalizeSymbol :: String -> String
normalizeSymbol = map toUpper . trim



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
