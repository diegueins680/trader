{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Trader.ExternalData (
    ExternalDataConfig (..),
    ExternalFeature (..),
    ExternalJsonSpec (..),
    alignedExternalFeatureInputs,
    externalDataConfigFromEnv,
    fetchExternalFeatureInputs,
    parseExternalJsonSpec,
) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, try)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isDigit, toLower, toUpper)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (isSuffixOf)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe, listToMaybe, mapMaybe)
import Data.Scientific (toRealFloat)
import qualified Data.Text as T
import Data.Time.Clock.POSIX (posixSecondsToUTCTime)
import Data.Time.Format (defaultTimeLocale, formatTime)
import qualified Data.Vector as V
import Network.HTTP.Client (Request, method, parseRequest, queryString, requestHeaders, responseBody)
import Network.HTTP.Types (RequestHeaders, renderSimpleQuery)
import System.Directory (doesFileExist)
import System.Environment (lookupEnv)
import System.IO (hPutStrLn, stderr)
import Text.Read (readMaybe)

import Trader.App.Args (parseTimestampMs)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)
import Trader.Predictors.Exogenous (alignToBars, neutralFill)
import Trader.Predictors.Features (ExternalFeatureInputs (..))
import Trader.Text (normalizeKey, trim)

data ExternalFeature
    = ExternalMicrostructure
    | ExternalOptionsVol
    | ExternalOnChain
    | ExternalMacro
    | ExternalCot
    | ExternalNews
    | ExternalFilings
    deriving (Eq, Ord, Show)

data ExternalJsonSpec = ExternalJsonSpec
    { ejsFeature :: !ExternalFeature
    , ejsUrl :: !String
    , ejsTimeKey :: !String
    , ejsValueKey :: !String
    }
    deriving (Eq, Show)

data ExternalDataConfig = ExternalDataConfig
    { edcEnabled :: !Bool
    , edcCsvPaths :: ![FilePath]
    , edcJsonSpecs :: ![ExternalJsonSpec]
    , edcFredApiKey :: !(Maybe String)
    , edcFredSeries :: ![String]
    , edcFredPitLagDays :: !Int
    , edcDeribitCurrencies :: ![String]
    , edcGlassnodeApiKey :: !(Maybe String)
    , edcGlassnodeAsset :: !(Maybe String)
    , edcGlassnodeMetrics :: ![String]
    , edcGlassnodeInterval :: !String
    , edcGdeltQueries :: ![String]
    , edcSecCiks :: ![String]
    , edcSecUserAgent :: !(Maybe String)
    }
    deriving (Eq, Show)

type ExternalObservation = (ExternalFeature, Int64, Double)

externalDataConfigFromEnv :: Bool -> [FilePath] -> [String] -> IO ExternalDataConfig
externalDataConfigFromEnv enabledFlag cliCsvPaths cliJsonSpecs = do
    enabledEnv <- readEnvBool "TRADER_EXTERNAL_DATA" False
    csvEnv <- envList "TRADER_EXTERNAL_DATA_CSVS"
    jsonEnv <- envListSemi "TRADER_EXTERNAL_JSON_SOURCES"
    fredApiKey <- nonEmptyEnv "TRADER_FRED_API_KEY"
    fredSeries <- envList "TRADER_FRED_SERIES"
    fredLag <- envInt "TRADER_FRED_PIT_LAG_DAYS" 1
    deribitCurrencies <- envList "TRADER_DERIBIT_CURRENCIES"
    glassnodeApiKey <- nonEmptyEnv "TRADER_GLASSNODE_API_KEY"
    glassnodeAsset <- nonEmptyEnv "TRADER_GLASSNODE_ASSET"
    glassnodeMetrics <- envList "TRADER_GLASSNODE_METRICS"
    glassnodeInterval <- fromMaybe "24h" <$> nonEmptyEnv "TRADER_GLASSNODE_INTERVAL"
    gdeltQueries <- envListSemi "TRADER_GDELT_QUERIES"
    secCiks <- envList "TRADER_SEC_FILINGS_CIKS"
    secUserAgent <- nonEmptyEnv "TRADER_SEC_USER_AGENT"
    let jsonSpecs = mapMaybe parseExternalJsonSpec (cliJsonSpecs ++ jsonEnv)
        enabled =
            enabledFlag
                || enabledEnv
                || not (null cliCsvPaths)
                || not (null csvEnv)
                || not (null jsonSpecs)
                || not (null fredSeries)
                || not (null deribitCurrencies)
                || not (null glassnodeMetrics)
                || not (null gdeltQueries)
                || not (null secCiks)
    pure
        ExternalDataConfig
            { edcEnabled = enabled
            , edcCsvPaths = dedupe (cliCsvPaths ++ csvEnv)
            , edcJsonSpecs = jsonSpecs
            , edcFredApiKey = fredApiKey
            , edcFredSeries = dedupe fredSeries
            , edcFredPitLagDays = max 0 fredLag
            , edcDeribitCurrencies = map (map toUpper) (dedupe deribitCurrencies)
            , edcGlassnodeApiKey = glassnodeApiKey
            , edcGlassnodeAsset = glassnodeAsset
            , edcGlassnodeMetrics = dedupe glassnodeMetrics
            , edcGlassnodeInterval = glassnodeInterval
            , edcGdeltQueries = dedupe gdeltQueries
            , edcSecCiks = dedupe secCiks
            , edcSecUserAgent = secUserAgent
            }

parseExternalJsonSpec :: String -> Maybe ExternalJsonSpec
parseExternalJsonSpec raw =
    case splitOn '|' raw of
        [featureRaw, urlRaw, timeKeyRaw, valueKeyRaw] -> do
            feature <- parseExternalFeature featureRaw
            let url = trim urlRaw
                timeKey = trim timeKeyRaw
                valueKey = trim valueKeyRaw
            if null url || null timeKey || null valueKey
                then Nothing
                else Just (ExternalJsonSpec feature url timeKey valueKey)
        _ -> Nothing

fetchExternalFeatureInputs ::
    ExternalDataConfig ->
    Maybe String ->
    V.Vector Int64 ->
    Int64 ->
    IO (Maybe ExternalFeatureInputs)
fetchExternalFeatureInputs cfg mSymbol barOpenTimes intervalMs
    | not (edcEnabled cfg) = pure Nothing
    | V.null barOpenTimes = pure Nothing
    | otherwise = do
        observations <- fetchExternalObservations cfg mSymbol barOpenTimes
        pure (alignedExternalFeatureInputs barOpenTimes intervalMs observations)

fetchExternalObservations :: ExternalDataConfig -> Maybe String -> V.Vector Int64 -> IO [ExternalObservation]
fetchExternalObservations cfg mSymbol barOpenTimes = do
    csvObs <- concat <$> mapM fetchCsvObservations (edcCsvPaths cfg)
    jsonObs <- concat <$> mapM fetchJsonSpecObservations (edcJsonSpecs cfg)
    fredObs <- fetchFredObservations cfg barOpenTimes
    deribitObs <- fetchDeribitObservations cfg barOpenTimes
    glassnodeObs <- fetchGlassnodeObservations cfg mSymbol
    gdeltObs <- fetchGdeltObservations cfg
    secObs <- fetchSecObservations cfg
    pure (csvObs ++ jsonObs ++ fredObs ++ deribitObs ++ glassnodeObs ++ gdeltObs ++ secObs)

alignedExternalFeatureInputs ::
    V.Vector Int64 ->
    Int64 ->
    [ExternalObservation] ->
    Maybe ExternalFeatureInputs
alignedExternalFeatureInputs barOpenTimes intervalMs observations =
    let grouped =
            foldl
                ( \acc (feature, ts, value) ->
                    if finite value
                        then
                            Map.insertWith
                                (Map.unionWith mergeBuckets)
                                feature
                                (Map.singleton ts (value, 1 :: Int))
                                acc
                        else acc
                )
                Map.empty
                observations
        seriesFor feature =
            case Map.lookup feature grouped of
                Nothing -> Nothing
                Just byTs ->
                    let series =
                            [ (ts, total / fromIntegral count)
                            | (ts, (total, count)) <- Map.toAscList byTs
                            , count > 0
                            ]
                     in if null series
                            then Nothing
                            else Just (neutralFill (alignToBars barOpenTimes intervalMs series))
        bundle =
            ExternalFeatureInputs
                { efiMicrostructure = seriesFor ExternalMicrostructure
                , efiOptionsVol = seriesFor ExternalOptionsVol
                , efiOnChain = seriesFor ExternalOnChain
                , efiMacro = seriesFor ExternalMacro
                , efiCot = seriesFor ExternalCot
                , efiNews = seriesFor ExternalNews
                , efiFilings = seriesFor ExternalFilings
                }
     in if Map.null grouped then Nothing else Just bundle
  where
    mergeBuckets (aSum, aCount) (bSum, bCount) = (aSum + bSum, aCount + bCount)

fetchCsvObservations :: FilePath -> IO [ExternalObservation]
fetchCsvObservations path = do
    exists <- doesFileExist path
    if not exists
        then do
            hPutStrLn stderr ("WARN: external data CSV not found: " ++ path)
            pure []
        else do
            bs <- BL.readFile path
            case Csv.decodeByName bs of
                Left err -> do
                    hPutStrLn stderr ("WARN: external data CSV decode failed (" ++ path ++ "): " ++ err)
                    pure []
                Right (hdr, rows) -> do
                    let hdrList = V.toList hdr
                        mTimeKey = firstJust [findHeaderKey hdrList c | c <- timeColumns]
                    case mTimeKey of
                        Nothing -> do
                            hPutStrLn stderr ("WARN: external data CSV has no timestamp column: " ++ path)
                            pure []
                        Just timeKey -> pure (concatMap (rowObservations hdrList timeKey) (V.toList rows))

rowObservations :: [BS.ByteString] -> BS.ByteString -> Csv.NamedRecord -> [ExternalObservation]
rowObservations hdrList timeKey row =
    case HM.lookup timeKey row >>= parseCellTime of
        Nothing -> []
        Just ts ->
            genericSourceObservations ts row
                ++ concatMap (featureColumnObservations ts row) featureColumns
  where
    featureColumnObservations ts row0 (feature, columns) =
        let values =
                [ value
                | col <- columns
                , Just key <- [findHeaderKey hdrList col]
                , Just raw <- [HM.lookup key row0]
                , Just value <- [parseCellDouble raw]
                , finite value
                ]
         in case values of
                [] -> []
                _ -> [(feature, ts, sum values / fromIntegral (length values))]

genericSourceObservations :: Int64 -> Csv.NamedRecord -> [ExternalObservation]
genericSourceObservations ts row =
    case (lookupAny sourceColumns row, lookupAny valueColumns row) of
        (Just sourceRaw, Just valueRaw) ->
            case (parseExternalFeature (BS.unpack sourceRaw), parseCellDouble valueRaw) of
                (Just feature, Just value) | finite value -> [(feature, ts, value)]
                _ -> []
        _ -> []

fetchJsonSpecObservations :: ExternalJsonSpec -> IO [ExternalObservation]
fetchJsonSpecObservations spec =
    fetchJsonValue (ejsUrl spec) [] >>= \case
        Nothing -> pure []
        Just value ->
            pure
                [ (ejsFeature spec, ts, metric)
                | obj <- collectObjects value
                , Just ts <- [jsonLookupByKey (ejsTimeKey spec) obj >>= jsonTime]
                , Just metric <- [jsonLookupByKey (ejsValueKey spec) obj >>= jsonMetric]
                , finite metric
                ]

fetchFredObservations :: ExternalDataConfig -> V.Vector Int64 -> IO [ExternalObservation]
fetchFredObservations cfg barOpenTimes =
    case (edcFredApiKey cfg, edcFredSeries cfg) of
        (Just key, seriesIds) | not (null seriesIds) -> do
            rows <- concat <$> mapM (fetchOne key) seriesIds
            pure [(ExternalMacro, ts, value) | (ts, value) <- rows]
        _ -> pure []
  where
    fetchOne key seriesId = do
        let url = "https://api.stlouisfed.org/fred/series/observations"
            params =
                [ ("series_id", BS.pack seriesId)
                , ("api_key", BS.pack key)
                , ("file_type", "json")
                , ("observation_start", BS.pack (windowDate V.head))
                , ("observation_end", BS.pack (windowDate V.last))
                ]
        fetchJsonValue url params >>= \case
            Nothing -> pure []
            Just value -> pure (parseFredRows (edcFredPitLagDays cfg) value)
    windowDate pick =
        let ts = pick barOpenTimes
         in take 10 (timestampDateText ts)

parseFredRows :: Int -> Aeson.Value -> [(Int64, Double)]
parseFredRows lagDays value =
    case value of
        Aeson.Object o ->
            case KM.lookup "observations" o of
                Just (Aeson.Array xs) -> mapMaybe parseRow (V.toList xs)
                _ -> []
        _ -> []
  where
    lagMs = fromIntegral (max 0 lagDays) * 86400000
    parseRow (Aeson.Object o) = do
        dateTxt <- jsonLookupText "date" o
        raw <- jsonLookupText "value" o
        ts <- parseTimestampMs (T.unpack dateTxt)
        valueD <- readMaybe (T.unpack raw)
        if finite valueD then Just (ts + lagMs, valueD) else Nothing
    parseRow _ = Nothing

fetchDeribitObservations :: ExternalDataConfig -> V.Vector Int64 -> IO [ExternalObservation]
fetchDeribitObservations cfg barOpenTimes =
    concat <$> mapM fetchCurrency (edcDeribitCurrencies cfg)
  where
    startTs = V.head barOpenTimes
    endTs = V.last barOpenTimes
    fetchCurrency currency = do
        let url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
            params =
                [ ("currency", BS.pack currency)
                , ("start_timestamp", BS.pack (show startTs))
                , ("end_timestamp", BS.pack (show endTs))
                , ("resolution", "60")
                ]
        fetchJsonValue url params >>= \case
            Nothing -> pure []
            Just value -> pure [(ExternalOptionsVol, ts, v) | (ts, v) <- parseFlexibleRows value]

fetchGlassnodeObservations :: ExternalDataConfig -> Maybe String -> IO [ExternalObservation]
fetchGlassnodeObservations cfg mSymbol =
    case (edcGlassnodeApiKey cfg, resolveGlassnodeAsset cfg mSymbol, edcGlassnodeMetrics cfg) of
        (Just key, Just asset, metrics) | not (null metrics) -> do
            rows <- concat <$> mapM (fetchMetric key asset) metrics
            pure [(ExternalOnChain, ts, v) | (ts, v) <- rows]
        _ -> pure []
  where
    fetchMetric key asset metric = do
        let url = "https://api.glassnode.com/v1/metrics/" ++ dropWhile (== '/') metric
            params =
                [ ("a", BS.pack asset)
                , ("i", BS.pack (edcGlassnodeInterval cfg))
                , ("api_key", BS.pack key)
                ]
        fetchJsonValue url params >>= \case
            Nothing -> pure []
            Just value -> pure (parseFlexibleRows value)

resolveGlassnodeAsset :: ExternalDataConfig -> Maybe String -> Maybe String
resolveGlassnodeAsset cfg mSymbol =
    edcGlassnodeAsset cfg <|> (symbolBase <$> mSymbol)

fetchGdeltObservations :: ExternalDataConfig -> IO [ExternalObservation]
fetchGdeltObservations cfg =
    concat <$> mapM fetchQuery (edcGdeltQueries cfg)
  where
    fetchQuery queryText = do
        let url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params =
                [ ("query", BS.pack queryText)
                , ("mode", "timelinevol")
                , ("format", "json")
                , ("timespan", "30d")
                ]
        fetchJsonValue url params >>= \case
            Nothing -> pure []
            Just value -> pure [(ExternalNews, ts, v) | (ts, v) <- parseFlexibleRows value]

fetchSecObservations :: ExternalDataConfig -> IO [ExternalObservation]
fetchSecObservations cfg =
    case edcSecUserAgent cfg of
        Nothing -> pure []
        Just userAgent ->
            concat <$> mapM (fetchCik userAgent) (edcSecCiks cfg)
  where
    fetchCik userAgent cikRaw = do
        let cik = padCik cikRaw
            url = "https://data.sec.gov/submissions/CIK" ++ cik ++ ".json"
        fetchJsonValue url [("__header_user_agent", BS.pack userAgent)] >>= \case
            Nothing -> pure []
            Just value -> pure (parseSecRows value)

parseSecRows :: Aeson.Value -> [ExternalObservation]
parseSecRows value =
    case value of
        Aeson.Object root -> do
            filings <- maybeToList (KM.lookup "filings" root)
            recent <- case filings of
                Aeson.Object o -> maybeToList (KM.lookup "recent" o)
                _ -> []
            datesValue <- case recent of
                Aeson.Object o -> maybeToList (KM.lookup "filingDate" o)
                _ -> []
            dates <- case datesValue of
                Aeson.Array xs -> V.toList xs
                _ -> []
            dateTxt <- maybeToList (jsonText dates)
            ts <- maybeToList (parseTimestampMs (T.unpack dateTxt))
            pure (ExternalFilings, ts, 1)
        _ -> []

fetchJsonValue :: String -> [(BS.ByteString, BS.ByteString)] -> IO (Maybe Aeson.Value)
fetchJsonValue url params0 = do
    let (headerParams, queryParams) = spanHeaderParams params0
    result <- try $ do
        req0 <- parseRequest url
        let req =
                (req0 :: Request)
                    { method = "GET"
                    , requestHeaders = headerParams ++ requestHeaders req0
                    }
            reqWithQuery =
                if null queryParams
                    then req
                    else req{queryString = renderSimpleQuery True queryParams}
        mgr <- getSharedManager
        resp <- httpLbsWithRetry defaultRetryConfig (Just "external-data") mgr reqWithQuery
        case Aeson.eitherDecode (responseBody resp) of
            Left err -> do
                hPutStrLn stderr ("WARN: external data JSON decode failed (" ++ url ++ "): " ++ err)
                pure Nothing
            Right value -> pure (Just value)
    case result of
        Left (ex :: SomeException) -> do
            hPutStrLn stderr ("WARN: external data fetch failed (" ++ url ++ "): " ++ show ex)
            pure Nothing
        Right value -> pure value

spanHeaderParams :: [(BS.ByteString, BS.ByteString)] -> (RequestHeaders, [(BS.ByteString, BS.ByteString)])
spanHeaderParams =
    foldr step ([], [])
  where
    step (k, v) (headers, params)
        | k == "__header_user_agent" = (("User-Agent", v) : headers, params)
        | otherwise = (headers, (k, v) : params)

parseFlexibleRows :: Aeson.Value -> [(Int64, Double)]
parseFlexibleRows value =
    [ (ts, metric)
    | obj <- collectObjects value
    , Just ts <- [firstJust (map (`jsonLookupByKey` obj) flexibleTimeKeys) >>= jsonTime]
    , Just metric <- [(firstJust (map (`jsonLookupByKey` obj) flexibleValueKeys) >>= jsonMetric) <|> averageObjectNumbers obj]
    , finite metric
    ]
        ++ parseArrayRows value

parseArrayRows :: Aeson.Value -> [(Int64, Double)]
parseArrayRows value =
    case value of
        Aeson.Array xs -> concatMap parseArrayRows (V.toList xs) ++ mapMaybe parsePointArray (V.toList xs)
        Aeson.Object o -> concatMap parseArrayRows (KM.elems o)
        _ -> []
  where
    parsePointArray (Aeson.Array xs) =
        case V.toList xs of
            [] -> Nothing
            first : rest -> do
                ts <- jsonTime first
                metric <- lastMaybe (mapMaybe jsonMetric rest)
                pure (ts, metric)
    parsePointArray _ = Nothing

collectObjects :: Aeson.Value -> [KM.KeyMap Aeson.Value]
collectObjects value =
    case value of
        Aeson.Object o ->
            o : concatMap collectObjects (KM.elems o)
        Aeson.Array xs ->
            concatMap collectObjects (V.toList xs)
        _ -> []

jsonLookupByKey :: String -> KM.KeyMap Aeson.Value -> Maybe Aeson.Value
jsonLookupByKey wanted obj =
    let wantedNorm = normalizeKey wanted
     in listToMaybe
            [ v
            | (k, v) <- KM.toList obj
            , normalizeKey (T.unpack (AK.toText k)) == wantedNorm
            ]

jsonLookupText :: String -> KM.KeyMap Aeson.Value -> Maybe T.Text
jsonLookupText key obj = jsonLookupByKey key obj >>= jsonText

jsonText :: Aeson.Value -> Maybe T.Text
jsonText value =
    case value of
        Aeson.String t -> Just t
        Aeson.Number n -> Just (T.pack (show n))
        _ -> Nothing

jsonTime :: Aeson.Value -> Maybe Int64
jsonTime value =
    case value of
        Aeson.String t -> parseTimestampMs (T.unpack t)
        Aeson.Number n ->
            let x = floor (toRealFloat n :: Double) :: Int64
             in Just (normalizeEpochMsLocal x)
        _ -> Nothing

jsonMetric :: Aeson.Value -> Maybe Double
jsonMetric value =
    case value of
        Aeson.Number n ->
            let x = toRealFloat n :: Double
             in if finite x then Just x else Nothing
        Aeson.String t ->
            case readMaybe (T.unpack t) of
                Just x | finite x -> Just x
                _ -> Nothing
        Aeson.Object o -> averageObjectNumbers o
        _ -> Nothing

averageObjectNumbers :: KM.KeyMap Aeson.Value -> Maybe Double
averageObjectNumbers o =
    case mapMaybe jsonMetric (KM.elems o) of
        [] -> Nothing
        xs -> Just (sum xs / fromIntegral (length xs))

parseCellTime :: BS.ByteString -> Maybe Int64
parseCellTime = parseTimestampMs . trim . BS.unpack

parseCellDouble :: BS.ByteString -> Maybe Double
parseCellDouble raw =
    case readMaybe (trim (BS.unpack raw)) of
        Just value | finite value -> Just value
        _ -> Nothing

lookupAny :: [String] -> Csv.NamedRecord -> Maybe BS.ByteString
lookupAny candidates row =
    firstJust
        [ HM.lookup key row
        | key <- HM.keys row
        , normalizeKey (BS.unpack key) `elem` map normalizeKey candidates
        ]

findHeaderKey :: [BS.ByteString] -> String -> Maybe BS.ByteString
findHeaderKey hdrList wanted =
    let wantedNorm = normalizeKey wanted
     in listToMaybe [h | h <- hdrList, normalizeKey (BS.unpack h) == wantedNorm]

parseExternalFeature :: String -> Maybe ExternalFeature
parseExternalFeature raw =
    case normalizeKey raw of
        "microstructure" -> Just ExternalMicrostructure
        "derivativesmicrostructure" -> Just ExternalMicrostructure
        "orderbook" -> Just ExternalMicrostructure
        "l2" -> Just ExternalMicrostructure
        "liquidation" -> Just ExternalMicrostructure
        "liquidations" -> Just ExternalMicrostructure
        "options" -> Just ExternalOptionsVol
        "option" -> Just ExternalOptionsVol
        "optionsvol" -> Just ExternalOptionsVol
        "dvol" -> Just ExternalOptionsVol
        "iv" -> Just ExternalOptionsVol
        "onchain" -> Just ExternalOnChain
        "glassnode" -> Just ExternalOnChain
        "cryptoquant" -> Just ExternalOnChain
        "macro" -> Just ExternalMacro
        "fred" -> Just ExternalMacro
        "alfred" -> Just ExternalMacro
        "liquidity" -> Just ExternalMacro
        "cot" -> Just ExternalCot
        "cftc" -> Just ExternalCot
        "positioning" -> Just ExternalCot
        "news" -> Just ExternalNews
        "sentiment" -> Just ExternalNews
        "gdelt" -> Just ExternalNews
        "filings" -> Just ExternalFilings
        "filing" -> Just ExternalFilings
        "sec" -> Just ExternalFilings
        "edgar" -> Just ExternalFilings
        "etf" -> Just ExternalFilings
        _ -> Nothing

featureColumns :: [(ExternalFeature, [String])]
featureColumns =
    [
        ( ExternalMicrostructure
        ,
            [ "bookImbalance"
            , "orderBookImbalance"
            , "l2Imbalance"
            , "depthImbalance"
            , "tradeImbalance"
            , "orderFlowImbalance"
            , "liquidations"
            , "liquidationUsd"
            , "openInterestDelta"
            ]
        )
    ,
        ( ExternalOptionsVol
        ,
            [ "dvol"
            , "volatilityIndex"
            , "iv"
            , "atmIv"
            , "impliedVol"
            , "riskReversal"
            , "putCallSkew"
            , "skew"
            , "optionsOpenInterest"
            ]
        )
    ,
        ( ExternalOnChain
        ,
            [ "exchangeNetFlow"
            , "netflow"
            , "exchangeBalance"
            , "stablecoinSupply"
            , "stablecoinFlows"
            , "sopr"
            , "mvrv"
            , "activeAddresses"
            , "minerFlow"
            , "whaleFlow"
            ]
        )
    ,
        ( ExternalMacro
        ,
            [ "dxy"
            , "vix"
            , "us02y"
            , "us2y"
            , "us10y"
            , "realYield"
            , "fedLiquidity"
            , "rrp"
            , "tga"
            , "creditSpread"
            , "macro"
            ]
        )
    ,
        ( ExternalCot
        ,
            [ "cot"
            , "dealerNet"
            , "assetManagerNet"
            , "leveragedFundsNet"
            , "noncommercialNet"
            , "commercialNet"
            ]
        )
    ,
        ( ExternalNews
        ,
            [ "news"
            , "sentiment"
            , "tone"
            , "newsVolume"
            , "eventScore"
            , "gdelt"
            ]
        )
    ,
        ( ExternalFilings
        ,
            [ "filings"
            , "filingScore"
            , "secScore"
            , "filingsCount"
            , "etfFlow"
            , "flows"
            , "edgar"
            ]
        )
    ]

timeColumns :: [String]
timeColumns =
    [ "openTimeMs"
    , "open_time_ms"
    , "timestampMs"
    , "timestamp"
    , "time"
    , "date"
    , "datetime"
    ]

sourceColumns :: [String]
sourceColumns = ["source", "family", "feature", "dataset", "class", "type"]

valueColumns :: [String]
valueColumns = ["value", "metric", "signal", "score", "reading"]

flexibleTimeKeys :: [String]
flexibleTimeKeys = ["timestamp", "timestampMs", "time", "t", "date", "datetime", "filingDate"]

flexibleValueKeys :: [String]
flexibleValueKeys = ["value", "v", "close", "price", "metric", "score", "tone", "volume"]

readEnvBool :: String -> Bool -> IO Bool
readEnvBool key fallback = do
    mRaw <- lookupEnv key
    pure $
        case fmap (map toLower . trim) mRaw of
            Just "1" -> True
            Just "true" -> True
            Just "yes" -> True
            Just "on" -> True
            Just "0" -> False
            Just "false" -> False
            Just "no" -> False
            Just "off" -> False
            _ -> fallback

envList :: String -> IO [String]
envList key = maybe [] splitComma <$> nonEmptyEnv key

envListSemi :: String -> IO [String]
envListSemi key = maybe [] (filter (not . null) . map trim . splitOn ';') <$> nonEmptyEnv key

envInt :: String -> Int -> IO Int
envInt key fallback = do
    mRaw <- nonEmptyEnv key
    pure (fromMaybe fallback (mRaw >>= readMaybe))

nonEmptyEnv :: String -> IO (Maybe String)
nonEmptyEnv key = do
    mRaw <- lookupEnv key
    pure $ do
        raw <- mRaw
        let cleaned = trim raw
        if null cleaned then Nothing else Just cleaned

splitComma :: String -> [String]
splitComma = filter (not . null) . map trim . splitOn ','

splitOn :: Char -> String -> [String]
splitOn sep raw =
    case break (== sep) raw of
        (a, []) -> [a]
        (a, _ : rest) -> a : splitOn sep rest

dedupe :: (Ord a) => [a] -> [a]
dedupe = go Map.empty
  where
    go _ [] = []
    go seen (x : xs)
        | Map.member x seen = go seen xs
        | otherwise = x : go (Map.insert x () seen) xs

firstJust :: [Maybe a] -> Maybe a
firstJust xs =
    case xs of
        [] -> Nothing
        y : ys ->
            case y of
                Just _ -> y
                Nothing -> firstJust ys

lastMaybe :: [a] -> Maybe a
lastMaybe xs =
    case xs of
        [] -> Nothing
        _ -> Just (last xs)

maybeToList :: Maybe a -> [a]
maybeToList Nothing = []
maybeToList (Just x) = [x]

finite :: Double -> Bool
finite x = not (isNaN x || isInfinite x)

normalizeEpochMsLocal :: Int64 -> Int64
normalizeEpochMsLocal n =
    if abs n < 10000000000
        then n * 1000
        else n

timestampDateText :: Int64 -> String
timestampDateText tsMs =
    let seconds = fromIntegral tsMs / 1000
     in formatTime defaultTimeLocale "%Y-%m-%d" (posixSecondsToUTCTime (realToFrac seconds))

symbolBase :: String -> String
symbolBase raw =
    let upper = map toUpper raw
        quotes = ["USDT", "USDC", "USD", "BUSD", "BTC", "ETH"]
        stripQuote s qs =
            case qs of
                [] -> s
                q : rest ->
                    if q `isSuffixOf` s
                        then take (length s - length q) s
                        else stripQuote s rest
     in stripQuote upper quotes

padCik :: String -> String
padCik raw =
    let digits = filter isDigit raw
        padded = replicate (max 0 (10 - length digits)) '0' ++ digits
     in take 10 padded
