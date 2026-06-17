{-# LANGUAGE OverloadedStrings #-}

module Trader.PredictionMarkets (
    PredictionMarketFetchConfig (..),
    PredictionMarketEvent (..),
    PredictionMarketMarket (..),
    PredictionMarketOutcome (..),
    PredictionMarketSignal (..),
    defaultPredictionMarketFetchConfig,
    fetchPolymarketHerdSignal,
    nearestPredictionMarketInterval,
    predictionMarketProbabilityForDir,
    selectPredictionMarketSignal,
    selectPredictionMarketSignalWithConfig,
) where

import Control.Applicative ((<|>))
import Control.Exception (throwIO)
import Data.Aeson (FromJSON (..), ToJSON (..), Value (..), eitherDecode, object, withObject, (.!=), (.:?), (.=))
import qualified Data.Aeson.Key as AK
import Data.Aeson.Types (Object, Parser)
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum)
import Data.List (maximumBy)
import Data.Maybe (fromMaybe, isJust, listToMaybe, mapMaybe)
import Data.Ord (comparing)
import Data.Scientific (toRealFloat)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)
import Data.Time.Format (defaultTimeLocale, parseTimeM)
import qualified Data.Vector as V
import Network.HTTP.Client (Request, Response, parseRequest, requestHeaders, responseBody, responseStatus, setQueryString)
import Network.HTTP.Types.Status (statusCode)
import System.IO.Unsafe (unsafePerformIO)
import Text.Read (readMaybe)

import Trader.Cache (TtlCache, fetchWithCache, newTtlCacheWithMaxEntries)
import Trader.Duration (parseIntervalSeconds)
import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)
import Trader.Symbol (splitSymbol)

data PredictionMarketFetchConfig = PredictionMarketFetchConfig
    { pmfcGammaBaseUrl :: !String
    , pmfcLimit :: !Int
    , pmfcMinVolume :: !Double
    , pmfcFreshTtl :: !NominalDiffTime
    , pmfcStaleTtl :: !NominalDiffTime
    , pmfcScoreBase :: !Double
    , pmfcIntervalMatchBonus :: !Double
    , pmfcTimeDecayBonus :: !Double
    , pmfcPastEndPenalty :: !Double
    , pmfcVolumeScoreWeight :: !Double
    }
    deriving (Eq, Show)

defaultPredictionMarketFetchConfig :: PredictionMarketFetchConfig
defaultPredictionMarketFetchConfig =
    PredictionMarketFetchConfig
        { pmfcGammaBaseUrl = "https://gamma-api.polymarket.com"
        , pmfcLimit = 50
        , pmfcMinVolume = 0
        , pmfcFreshTtl = 60
        , pmfcStaleTtl = 300
        , pmfcScoreBase = 100
        , pmfcIntervalMatchBonus = 30
        , pmfcTimeDecayBonus = 20
        , pmfcPastEndPenalty = -10
        , pmfcVolumeScoreWeight = 1
        }

data PredictionMarketEvent = PredictionMarketEvent
    { pmeSlug :: !(Maybe Text)
    , pmeTitle :: !Text
    , pmeEndDate :: !(Maybe Text)
    , pmeVolume :: !(Maybe Double)
    , pmeVolume24hr :: !(Maybe Double)
    , pmeMarkets :: ![PredictionMarketMarket]
    }
    deriving (Eq, Show)

data PredictionMarketMarket = PredictionMarketMarket
    { pmmId :: !(Maybe Text)
    , pmmSlug :: !(Maybe Text)
    , pmmQuestion :: !Text
    , pmmEndDate :: !(Maybe Text)
    , pmmActive :: !Bool
    , pmmClosed :: !Bool
    , pmmVolume :: !(Maybe Double)
    , pmmVolume24hr :: !(Maybe Double)
    , pmmOutcomes :: ![Text]
    , pmmOutcomePrices :: ![Double]
    }
    deriving (Eq, Show)

data PredictionMarketOutcome = PredictionMarketOutcome
    { pmoOutcome :: !Text
    , pmoProbability :: !Double
    }
    deriving (Eq, Show)

instance ToJSON PredictionMarketOutcome where
    toJSON o =
        object
            [ "outcome" .= pmoOutcome o
            , "probability" .= pmoProbability o
            ]

data PredictionMarketSignal = PredictionMarketSignal
    { pmsSource :: !Text
    , pmsSymbol :: !Text
    , pmsBaseAsset :: !Text
    , pmsInterval :: !Text
    , pmsIntervalSeconds :: !Int
    , pmsEventSlug :: !(Maybe Text)
    , pmsEventTitle :: !Text
    , pmsMarketId :: !(Maybe Text)
    , pmsMarketSlug :: !(Maybe Text)
    , pmsMarketQuestion :: !Text
    , pmsUrl :: !(Maybe Text)
    , pmsEndDate :: !(Maybe Text)
    , pmsVolume :: !(Maybe Double)
    , pmsVolume24hr :: !(Maybe Double)
    , pmsOutcomes :: ![PredictionMarketOutcome]
    , pmsUpProbability :: !(Maybe Double)
    , pmsDownProbability :: !(Maybe Double)
    , pmsScore :: !Double
    }
    deriving (Eq, Show)

instance ToJSON PredictionMarketSignal where
    toJSON s =
        object
            [ "source" .= pmsSource s
            , "symbol" .= pmsSymbol s
            , "baseAsset" .= pmsBaseAsset s
            , "interval" .= pmsInterval s
            , "intervalSeconds" .= pmsIntervalSeconds s
            , "eventSlug" .= pmsEventSlug s
            , "eventTitle" .= pmsEventTitle s
            , "marketId" .= pmsMarketId s
            , "marketSlug" .= pmsMarketSlug s
            , "marketQuestion" .= pmsMarketQuestion s
            , "url" .= pmsUrl s
            , "endDate" .= pmsEndDate s
            , "volume" .= pmsVolume s
            , "volume24hr" .= pmsVolume24hr s
            , "outcomes" .= pmsOutcomes s
            , "upProbability" .= pmsUpProbability s
            , "downProbability" .= pmsDownProbability s
            , "score" .= pmsScore s
            ]

instance FromJSON PredictionMarketEvent where
    parseJSON =
        withObject "PredictionMarketEvent" $ \o -> do
            slug <- o .:? "slug"
            title <- o .:? "title" .!= ""
            endDate <- firstJustFields o ["endDateIso", "endDate", "endDateTime"]
            volumeRaw <- firstJustFields o ["volumeNum", "volume"]
            volume24hrRaw <- firstJustFields o ["volume24hr", "volume_24hr"]
            markets <- o .:? "markets" .!= []
            pure
                PredictionMarketEvent
                    { pmeSlug = slug
                    , pmeTitle = title
                    , pmeEndDate = endDate >>= textFromValue
                    , pmeVolume = volumeRaw >>= doubleFromValue
                    , pmeVolume24hr = volume24hrRaw >>= doubleFromValue
                    , pmeMarkets = markets
                    }

instance FromJSON PredictionMarketMarket where
    parseJSON =
        withObject "PredictionMarketMarket" $ \o -> do
            marketId <- firstJustTextFields o ["id", "conditionId"]
            slug <- o .:? "slug"
            question <- o .:? "question" .!= ""
            endDate <- firstJustFields o ["endDateIso", "endDate", "endDateTime"]
            active <- o .:? "active" .!= True
            closed <- o .:? "closed" .!= False
            volumeRaw <- firstJustFields o ["volumeNum", "volume"]
            volume24hrRaw <- firstJustFields o ["volume24hr", "volume_24hr"]
            outcomesRaw <- o .:? "outcomes"
            pricesRaw <- o .:? "outcomePrices"
            pure
                PredictionMarketMarket
                    { pmmId = marketId
                    , pmmSlug = slug
                    , pmmQuestion = question
                    , pmmEndDate = endDate >>= textFromValue
                    , pmmActive = active
                    , pmmClosed = closed
                    , pmmVolume = volumeRaw >>= doubleFromValue
                    , pmmVolume24hr = volume24hrRaw >>= doubleFromValue
                    , pmmOutcomes = maybe [] textListFromValue outcomesRaw
                    , pmmOutcomePrices = maybe [] doubleListFromValue pricesRaw
                    }

newtype SearchResponse = SearchResponse
    { srEvents :: [PredictionMarketEvent]
    }
    deriving (Eq, Show)

instance FromJSON SearchResponse where
    parseJSON =
        withObject "SearchResponse" $ \o ->
            SearchResponse <$> o .:? "events" .!= []

{-# NOINLINE polymarketSignalCache #-}
polymarketSignalCache :: TtlCache String (Maybe PredictionMarketSignal)
polymarketSignalCache = unsafePerformIO (newTtlCacheWithMaxEntries 256)

fetchPolymarketHerdSignal :: PredictionMarketFetchConfig -> String -> String -> IO (Maybe PredictionMarketSignal)
fetchPolymarketHerdSignal cfg symbol interval =
    fetchWithCache polymarketSignalCache (pmfcFreshTtl cfg) (pmfcStaleTtl cfg) cacheKey $ do
        now <- getCurrentTime
        events <- fetchSearchEvents cfg searchQuery
        pure (selectPredictionMarketSignalAt now cfg symbol interval events)
  where
    (label, seconds) = nearestPredictionMarketInterval interval
    cacheKey = symbol ++ ":" ++ interval ++ ":" ++ label ++ ":" ++ show (pmfcLimit cfg) ++ ":" ++ show (pmfcMinVolume cfg)
    searchQuery = predictionMarketSearchQuery symbol label seconds

fetchSearchEvents :: PredictionMarketFetchConfig -> String -> IO [PredictionMarketEvent]
fetchSearchEvents cfg query = do
    req0 <- parseRequest (trimTrailingSlash (pmfcGammaBaseUrl cfg) ++ "/public-search")
    let req =
            setQueryString
                [ ("q", Just (BS.pack query))
                , ("events_status", Just "active")
                , ("keep_closed_markets", Just "0")
                , ("limit_per_type", Just (BS.pack (show (max 1 (pmfcLimit cfg)))))
                , ("search_profiles", Just "false")
                , ("sort", Just "volume_24hr")
                , ("ascending", Just "false")
                ]
                req0
                    { requestHeaders = ("User-Agent", "trader-hs/0.1") : requestHeaders req0
                    }
    resp <- httpGet req
    ensureOk "Polymarket public-search" resp
    case eitherDecode (responseBody resp) of
        Right sr -> pure (srEvents sr)
        Left err -> throwIO (userError ("Polymarket public-search decode failed: " ++ err))

httpGet :: Request -> IO (Response BL.ByteString)
httpGet req = do
    manager <- getSharedManager
    httpLbsWithRetry defaultRetryConfig (Just "polymarket.public-search") manager req

ensureOk :: String -> Response BL.ByteString -> IO ()
ensureOk label resp = do
    let code = statusCode (responseStatus resp)
    if code >= 200 && code < 300
        then pure ()
        else throwIO (userError (label ++ " HTTP " ++ show code))

selectPredictionMarketSignal :: String -> String -> [PredictionMarketEvent] -> Maybe PredictionMarketSignal
selectPredictionMarketSignal =
    selectPredictionMarketSignalAt fixedNow defaultPredictionMarketFetchConfig
  where
    fixedNow = read "2026-01-01 00:00:00 UTC" :: UTCTime

selectPredictionMarketSignalWithConfig :: PredictionMarketFetchConfig -> String -> String -> [PredictionMarketEvent] -> Maybe PredictionMarketSignal
selectPredictionMarketSignalWithConfig =
    selectPredictionMarketSignalAt fixedNow
  where
    fixedNow = read "2026-01-01 00:00:00 UTC" :: UTCTime

selectPredictionMarketSignalAt :: UTCTime -> PredictionMarketFetchConfig -> String -> String -> [PredictionMarketEvent] -> Maybe PredictionMarketSignal
selectPredictionMarketSignalAt now cfg symbol interval events =
    let candidates = concatMap (eventCandidates now cfg symbol interval) events
     in case candidates of
            [] -> Nothing
            xs -> Just (maximumBy (comparing pmsScore) xs)

eventCandidates :: UTCTime -> PredictionMarketFetchConfig -> String -> String -> PredictionMarketEvent -> [PredictionMarketSignal]
eventCandidates now cfg symbol interval event =
    mapMaybe (marketSignal now cfg symbol interval event) (pmeMarkets event)

marketSignal :: UTCTime -> PredictionMarketFetchConfig -> String -> String -> PredictionMarketEvent -> PredictionMarketMarket -> Maybe PredictionMarketSignal
marketSignal now cfg symbol interval event market = do
    let volume = pmmVolume market <|> pmeVolume event
        volume24hr = pmmVolume24hr market <|> pmeVolume24hr event
        volumeForFloor = fromMaybe 0 (volume <|> volume24hr)
    if not (pmmActive market) || pmmClosed market || volumeForFloor < pmfcMinVolume cfg
        then Nothing
        else do
            let (baseRaw, _) = splitSymbol symbol
                base = T.pack baseRaw
                aliases = assetAliases baseRaw
                (intervalLabel, intervalSeconds) = nearestPredictionMarketInterval interval
                haystack = normalizedText (T.unwords [pmeTitle event, pmmQuestion market, fromMaybe "" (pmeSlug event), fromMaybe "" (pmmSlug market)])
                matchesAsset = any (`containsNormalizedTerm` haystack) aliases
                matchesInterval = intervalMatches intervalLabel haystack
                outcomePairs = zip (pmmOutcomes market) (pmmOutcomePrices market)
                outcomes =
                    [ PredictionMarketOutcome outcome prob
                    | (outcome, prob) <- outcomePairs
                    , finitePositiveProbability prob
                    ]
                upProbability = outcomeProbabilityFor 1 (pmmQuestion market) outcomePairs
                downProbability = outcomeProbabilityFor (-1) (pmmQuestion market) outcomePairs
                directional = isJust upProbability && isJust downProbability
            if not matchesAsset || not directional
                then Nothing
                else
                    let endText = pmmEndDate market <|> pmeEndDate event
                        timeScore =
                            case endText >>= parsePolymarketTime of
                                Nothing -> 0
                                Just endTime ->
                                    let diffHours = realToFrac (diffUTCTime endTime now) / (3600 :: Double)
                                     in if diffHours < 0 then pmfcPastEndPenalty cfg else pmfcTimeDecayBonus cfg / (1 + diffHours)
                        intervalScore = if matchesInterval then pmfcIntervalMatchBonus cfg else 0
                        volumeScore = pmfcVolumeScoreWeight cfg * log (1 + max 0 volumeForFloor)
                        score = pmfcScoreBase cfg + intervalScore + timeScore + volumeScore
                        url = ("https://polymarket.com/event/" <>) <$> pmeSlug event
                     in Just
                            PredictionMarketSignal
                                { pmsSource = "polymarket"
                                , pmsSymbol = T.pack symbol
                                , pmsBaseAsset = base
                                , pmsInterval = T.pack intervalLabel
                                , pmsIntervalSeconds = intervalSeconds
                                , pmsEventSlug = pmeSlug event
                                , pmsEventTitle = pmeTitle event
                                , pmsMarketId = pmmId market
                                , pmsMarketSlug = pmmSlug market
                                , pmsMarketQuestion = pmmQuestion market
                                , pmsUrl = url
                                , pmsEndDate = endText
                                , pmsVolume = volume
                                , pmsVolume24hr = volume24hr
                                , pmsOutcomes = outcomes
                                , pmsUpProbability = upProbability
                                , pmsDownProbability = downProbability
                                , pmsScore = score
                                }

predictionMarketProbabilityForDir :: Int -> PredictionMarketSignal -> Maybe Double
predictionMarketProbabilityForDir dir signal
    | dir > 0 = pmsUpProbability signal
    | dir < 0 = pmsDownProbability signal
    | otherwise = Nothing

outcomeProbabilityFor :: Int -> Text -> [(Text, Double)] -> Maybe Double
outcomeProbabilityFor dir question pairs =
    listToMaybe
        [ prob
        | (outcome, prob) <- pairs
        , finitePositiveProbability prob
        , outcomeDirection question outcome == Just dir
        ]

outcomeDirection :: Text -> Text -> Maybe Int
outcomeDirection question outcome
    | any (`T.isInfixOf` out) ["up", "higher", "above", "over"] = Just 1
    | any (`T.isInfixOf` out) ["down", "lower", "below", "under"] = Just (-1)
    | out `elem` ["yes", "y"] = yesNoQuestionDirection question
    | out `elem` ["no", "n"] = negate <$> yesNoQuestionDirection question
    | otherwise = Nothing
  where
    out = normalizedText outcome

yesNoQuestionDirection :: Text -> Maybe Int
yesNoQuestionDirection question
    | any (`T.isInfixOf` q) [" up ", " higher ", " above ", " over "] = Just 1
    | any (`T.isInfixOf` q) [" down ", " lower ", " below ", " under "] = Just (-1)
    | otherwise = Nothing
  where
    q = " " <> normalizedText question <> " "

nearestPredictionMarketInterval :: String -> (String, Int)
nearestPredictionMarketInterval interval =
    let seconds = fromMaybe 900 (parseIntervalSeconds interval)
        supported = [("15m", 900), ("1h", 3600), ("4h", 14400), ("1d", 86400)]
     in firstAtLeast seconds supported

firstAtLeast :: Int -> [(String, Int)] -> (String, Int)
firstAtLeast _ [] = ("15m", 900)
firstAtLeast seconds supported =
    case filter ((>= seconds) . snd) supported of
        match : _ -> match
        [] -> last supported

predictionMarketSearchQuery :: String -> String -> Int -> String
predictionMarketSearchQuery symbol intervalLabel intervalSeconds =
    let (base, _) = splitSymbol symbol
        aliases = map T.unpack (assetAliases base)
        timeWords =
            case intervalSeconds of
                900 -> ["15m", "15 minute"]
                3600 -> ["1h", "hour"]
                14400 -> ["4h", "4 hour"]
                86400 -> ["daily", "day"]
                _ -> [intervalLabel]
     in unwords (aliases ++ ["up", "down"] ++ timeWords)

assetAliases :: String -> [Text]
assetAliases raw =
    case normalizedText (T.pack raw) of
        "btc" -> ["btc", "bitcoin"]
        "eth" -> ["eth", "ethereum"]
        "sol" -> ["sol", "solana"]
        "xrp" -> ["xrp", "ripple"]
        "doge" -> ["doge", "dogecoin"]
        "bnb" -> ["bnb", "binance"]
        "ada" -> ["ada", "cardano"]
        "avax" -> ["avax", "avalanche"]
        "link" -> ["link", "chainlink"]
        "ltc" -> ["ltc", "litecoin"]
        "bch" -> ["bch", "bitcoin cash"]
        other -> [other]

intervalMatches :: String -> Text -> Bool
intervalMatches label haystack =
    any ((`T.isInfixOf` haystack) . normalizedText) (intervalVariants label)

containsNormalizedTerm :: Text -> Text -> Bool
containsNormalizedTerm rawTerm haystack =
    let term = normalizedText rawTerm
     in (" " <> term <> " ") `T.isInfixOf` (" " <> haystack <> " ")

intervalVariants :: String -> [Text]
intervalVariants label =
    case label of
        "15m" -> ["15m", "15 m", "15min", "15 min", "15 minute", "15 minutes", "15-minute"]
        "1h" -> ["1h", "1 h", "1hr", "1 hr", "1 hour", "hourly"]
        "4h" -> ["4h", "4 h", "4hr", "4 hr", "4 hour", "4 hours"]
        "1d" -> ["1d", "1 d", "daily", "1 day", "day"]
        _ -> [T.pack label]

firstJustFields :: Object -> [Text] -> Parser (Maybe Value)
firstJustFields o = go
  where
    go [] = pure Nothing
    go (key : rest) = do
        mValue <- o .:? AK.fromText key
        case mValue of
            Just v -> pure (Just v)
            Nothing -> go rest

firstJustTextFields :: Object -> [Text] -> Parser (Maybe Text)
firstJustTextFields o keys = do
    mValue <- firstJustFields o keys
    pure (mValue >>= textFromValue)

textFromValue :: Value -> Maybe Text
textFromValue value =
    case value of
        String txt -> Just txt
        Number n -> Just (T.pack (show (toRealFloat n :: Double)))
        Bool b -> Just (if b then "true" else "false")
        _ -> Nothing

doubleFromValue :: Value -> Maybe Double
doubleFromValue value =
    case value of
        Number n -> finiteMaybe (toRealFloat n)
        String txt -> readMaybe (T.unpack txt) >>= finiteMaybe
        _ -> Nothing

textListFromValue :: Value -> [Text]
textListFromValue value =
    case value of
        Array arr -> mapMaybe textFromValue (V.toList arr)
        String txt ->
            fromMaybe [] (decodeJsonText txt <|> Just [txt | not (T.null txt)])
        _ -> []

doubleListFromValue :: Value -> [Double]
doubleListFromValue value =
    case value of
        Array arr -> mapMaybe doubleFromValue (V.toList arr)
        String txt ->
            fromMaybe [] (decodeJsonDoubles txt <|> fmap (: []) (readMaybe (T.unpack txt) >>= finiteMaybe))
        _ -> []

decodeJsonText :: Text -> Maybe [Text]
decodeJsonText txt =
    case eitherDecode (BL.fromStrict (TE.encodeUtf8 txt)) of
        Right xs -> Just xs
        Left _ -> Nothing

decodeJsonDoubles :: Text -> Maybe [Double]
decodeJsonDoubles txt =
    case eitherDecode (BL.fromStrict (TE.encodeUtf8 txt)) of
        Right values -> Just (mapMaybe doubleFromValue (values :: [Value]))
        Left _ -> Nothing

finitePositiveProbability :: Double -> Bool
finitePositiveProbability p = not (isNaN p) && not (isInfinite p) && p >= 0 && p <= 1

finiteMaybe :: Double -> Maybe Double
finiteMaybe value
    | isNaN value || isInfinite value = Nothing
    | otherwise = Just value

normalizedText :: Text -> Text
normalizedText =
    T.unwords
        . T.words
        . T.map normalizeChar
        . T.toLower
  where
    normalizeChar c
        | isAlphaNum c = c
        | otherwise = ' '

parsePolymarketTime :: Text -> Maybe UTCTime
parsePolymarketTime txt =
    firstParse
        [ "%Y-%m-%dT%H:%M:%S%QZ"
        , "%Y-%m-%dT%H:%M:%SZ"
        , "%Y-%m-%d %H:%M:%S%Q UTC"
        ]
  where
    raw = T.unpack txt
    firstParse [] = Nothing
    firstParse (fmt : rest) =
        parseTimeM True defaultTimeLocale fmt raw <|> firstParse rest

trimTrailingSlash :: String -> String
trimTrailingSlash raw =
    case reverse raw of
        '/' : rest -> reverse rest
        _ -> raw
