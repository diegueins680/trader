{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Merge (
    MergeArgs (..),
    runMerge,
) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, try)
import Control.Monad (when)
import Crypto.Hash (Digest, hash)
import Crypto.Hash.Algorithms (SHA256)
import Data.Aeson (Value (..), object, toJSON, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import Data.ByteArray (convert)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as BL
import Data.Char (isSpace, toLower)
import Data.List (foldl', intercalate, isPrefixOf, isSuffixOf, sort, sortBy)
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, isJust, isNothing, mapMaybe)
import Data.Scientific (FPFormat (..), Scientific, formatScientific, fromFloatDigits, toBoundedInteger, toRealFloat)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.Encoding.Error as TEE
import Data.Time.Clock.POSIX (POSIXTime, getPOSIXTime)
import qualified Data.Vector as V
import System.Directory (
    copyFile,
    createDirectoryIfMissing,
    doesDirectoryExist,
    doesFileExist,
    getHomeDirectory,
    listDirectory,
    renameFile,
 )
import System.FilePath (takeDirectory, (</>))
import System.IO (hClose, hFlush, hPutStrLn, openBinaryTempFile, stderr)
import Text.Read (readMaybe)

import Trader.BotStartSemantics (adoptionMinTradeCount, comboTradeCountMeetsAdoptionFloor, comboWalkForwardSharpeMeetsAdoptionFloor)
import Trader.Duration (inferPeriodsPerYear)
import Trader.Optimizer.Json (encodePretty)
import Trader.SignalGates (signalEntryOpenThresholdFeasible)
import Trader.Symbol (sanitizeComboSymbolForPlatform)
import Trader.TopComboScoring (
    TopComboScoringConfig (..),
    clampLiveAnnualizedReturnWithConfig,
    defaultTopComboScoringConfig,
    tcscLiveAnnualizedReturnCeiling,
    tcscLiveAnnualizedReturnFloor,
    tcscLiveBlendShrinkageOps,
    tcscLiveQuarantineMaxFinalEquity,
    tcscLiveQuarantineMinOperations,
    topComboDrawdownMultiplier,
    topComboEquityTerm,
    topComboLiveBlendWeight,
    topComboLiveQuarantinedByConfig,
    topComboValidatedAnnualizedReturn,
    topComboWalkForwardMultiplier,
    validateTopComboScoringConfig,
 )

data MergeArgs = MergeArgs
    { maTopJson :: !FilePath
    , maFromJsonl :: ![FilePath]
    , maFromTopJson :: ![FilePath]
    , maOut :: !FilePath
    , maMax :: !Int
    , maHistoryDir :: !(Maybe FilePath)
    , maScoringConfig :: !TopComboScoringConfig
    , maCopyToDist :: !Bool
    }
    deriving (Eq, Show)

data Combo = Combo
    { comboFinalEquity :: !Double
    , comboObjective :: !(Maybe String)
    , comboScore :: !(Maybe Double)
    , comboOpenThreshold :: !(Maybe Double)
    , comboCloseThreshold :: !(Maybe Double)
    , comboCreatedAtMs :: !(Maybe Int)
    , comboSource :: !(Maybe String)
    , comboMetrics :: !(Maybe (KM.KeyMap Value))
    , comboOperations :: !(Maybe [Value])
    , comboParams :: !(KM.KeyMap Value)
    }
    deriving (Eq, Show)

data ComboLiveStats = ComboLiveStats
    { clsFinalEquity :: !Double
    , clsAnnualizedReturn :: !(Maybe Double)
    , clsOperationCount :: !Int
    }
    deriving (Eq, Show)

runMerge :: MergeArgs -> IO Int
runMerge args = do
    topJsonPath <- expandUser (maTopJson args)
    outPath <-
        if null (trim (maOut args))
            then pure topJsonPath
            else expandUser (maOut args)
    jsonlPaths <-
        if null (maFromJsonl args)
            then discoverDefaultJsonlPaths
            else traverse expandUser (maFromJsonl args)
    otherTopJsonPaths <- traverse expandUser (maFromTopJson args)
    historyDir <- traverse expandUser (maHistoryDir args)
    sourcesResult <- loadSources topJsonPath otherTopJsonPaths jsonlPaths
    case sourcesResult of
        Left err -> do
            hPutStrLn stderr err
            pure 1
        Right sources -> do
            let scoringConfig = maScoringConfig args
                merged = mergeCombos sources
                maxItems = max 0 (maMax args)
                sourceCount = sum (map length sources)
            writeTopJson scoringConfig outPath merged maxItems
            archiveTopJson historyDir outPath
            when (maCopyToDist args) (copyToDist outPath)
            putStrLn ("Merged " ++ show sourceCount ++ " candidates into " ++ show (length merged) ++ " unique combos.")
            putStrLn ("Wrote: " ++ outPath)
            pure 0

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

expandUser :: FilePath -> IO FilePath
expandUser path =
    case path of
        ('~' : '/' : rest) -> do
            home <- getHomeDirectory
            pure (home </> rest)
        "~" -> getHomeDirectory
        _ -> pure path

loadSources ::
    FilePath ->
    [FilePath] ->
    [FilePath] ->
    IO (Either String [[Value]])
loadSources topJsonPath otherTopJsonPaths jsonlPaths = do
    topResult <- loadTopCombos topJsonPath
    case topResult of
        Left err -> pure (Left err)
        Right topCombos -> do
            otherResults <- traverse loadTopCombos otherTopJsonPaths
            case sequence otherResults of
                Left err -> pure (Left err)
                Right otherCombos -> do
                    jsonlCombos <- traverse loadCombosFromJsonl jsonlPaths
                    pure (Right (topCombos : otherCombos ++ jsonlCombos))

loadTopCombos :: FilePath -> IO (Either String [Value])
loadTopCombos path = do
    exists <- doesFileExist path
    if not exists
        then pure (Right [])
        else do
            raw <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
            case raw of
                Left e -> pure (Left ("Failed to read " ++ path ++ ": " ++ show e))
                Right contents ->
                    case Aeson.eitherDecode contents of
                        Left err -> pure (Left ("Failed to parse " ++ path ++ ": " ++ err))
                        Right (Object obj) ->
                            let generatedAtMs = KM.lookup (Key.fromString "generatedAtMs") obj >>= coerceIntValue
                                payloadSource = KM.lookup (Key.fromString "source") obj >>= trimmedNonEmptyStringValue
                                applyMetadata = applyCreatedAtMs generatedAtMs . applySourceValue payloadSource
                             in case KM.lookup (Key.fromString "combos") obj of
                                    Just (Array combos) ->
                                        pure (Right ([applyMetadata (Object c) | Object c <- V.toList combos]))
                                    _ -> pure (Right [])
                        Right _ -> pure (Right [])

loadCombosFromJsonl :: FilePath -> IO [Value]
loadCombosFromJsonl path = do
    exists <- doesFileExist path
    if not exists
        then pure []
        else do
            raw <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
            case raw of
                Left _ -> pure []
                Right contents ->
                    let lines' = filter (not . T.null . T.strip) (T.lines (TE.decodeUtf8With TEE.lenientDecode (BL.toStrict contents)))
                     in pure (mapMaybe parseJsonlLine lines')
  where
    parseJsonlLine line =
        case Aeson.decode (BL.fromStrict (TE.encodeUtf8 line)) of
            Just (Object rec) ->
                let metrics = nestedMetricsObject rec
                 in case KM.lookup (Key.fromString "ok") rec of
                        Just okVal | valueTruthyFlag okVal ->
                            case comboFloatField "finalEquity" rec metrics of
                                Nothing -> Nothing
                                Just finalEq ->
                                    let source =
                                            case KM.lookup (Key.fromString "source") rec of
                                                Just (String s) -> Just (T.unpack s)
                                                _ -> Nothing
                                        params =
                                            case KM.lookup (Key.fromString "params") rec of
                                                Just (Object p) -> p
                                                _ -> KM.empty
                                        objective = KM.lookup (Key.fromString "objective") rec >>= normalizeObjectiveValue
                                        score = comboFloatField "score" rec metrics
                                        openThr = KM.lookup (Key.fromString "openThreshold") rec >>= coerceFloatValue
                                        closeThr = KM.lookup (Key.fromString "closeThreshold") rec >>= coerceFloatValue
                                        operations = KM.lookup (Key.fromString "operations") rec >>= coerceOperations
                                        createdAtMs = KM.lookup (Key.fromString "createdAtMs") rec >>= coerceIntValue
                                        combo =
                                            object
                                                [ "finalEquity" .= finalEq
                                                , "objective" .= objective
                                                , "score" .= score
                                                , "openThreshold" .= openThr
                                                , "closeThreshold" .= closeThr
                                                , "createdAtMs" .= createdAtMs
                                                , "source" .= source
                                                , "metrics" .= metrics
                                                , "operations" .= operations
                                                , "params" .= Object params
                                                ]
                                     in Just combo
                        _ -> Nothing
            _ -> Nothing

discoverDefaultJsonlPaths :: IO [FilePath]
discoverDefaultJsonlPaths = do
    tmp <- discoverInDir ".tmp"
    hsTmp <- discoverInDir ("haskell" </> ".tmp")
    pure (tmp ++ hsTmp)
  where
    discoverInDir dir = do
        exists <- doesDirectoryExist dir
        if not exists
            then pure []
            else do
                entries <- listDirectory dir
                let matches =
                        sort
                            [ dir </> entry
                            | entry <- entries
                            , "optimize_equity" `isPrefixOf` entry
                            , ".jsonl" `isSuffixOf` entry
                            ]
                pure matches

mergeCombos :: [[Value]] -> [Combo]
mergeCombos sources =
    let merged = foldl' mergeSource M.empty sources
     in M.elems merged
  where
    mergeCombo acc comboVal =
        case normalizeCombo comboVal of
            Nothing -> acc
            Just combo ->
                let key = signatureKey combo
                 in case M.lookup key acc of
                        Nothing -> M.insert key combo acc
                        Just prev -> M.insert key (pickBest combo prev) acc
    pickBest newer prev =
        let objNew = comboObjective newer
            objPrev = comboObjective prev
            scoreNew = comboScore newer
            scorePrev = comboScore prev
            scoreVal = fromMaybe (-(1 / 0))
         in if objNew == objPrev && (isJust scoreNew || isJust scorePrev)
                then if scoreVal scoreNew > scoreVal scorePrev then newer else prev
                else if comboFinalEquity newer > comboFinalEquity prev then newer else prev
    mergeSource = foldl' mergeCombo

signatureKey :: Combo -> BS.ByteString
signatureKey combo =
    BL.toStrict (encodePretty (comboProcessingIdentityValue combo))

comboProcessingIdentityValue :: Combo -> Value
comboProcessingIdentityValue combo =
    object
        [ "params" .= normalizeComboIdentityValue (Object (comboParams combo))
        , "openThreshold" .= comboOpenThreshold combo
        , "closeThreshold" .= comboCloseThreshold combo
        , "objective" .= comboObjective combo
        ]

normalizeComboIdentityValue :: Value -> Value
normalizeComboIdentityValue raw =
    case raw of
        Object obj ->
            Object $
                KM.fromList
                    [ (key, normalized)
                    | (key, val) <- KM.toList obj
                    , let normalized = normalizeComboIdentityField key val
                    , normalized /= Null
                    ]
        Array arr ->
            Array (V.map normalizeComboIdentityValue arr)
        other -> other

normalizeComboIdentityField :: Key.Key -> Value -> Value
normalizeComboIdentityField key val
    | key == Key.fromString "platform" =
        case normalizeComboIdentityValue val of
            String "binance" -> Null
            normalized -> normalized
    | otherwise = normalizeComboIdentityValue val

writeTopJson :: TopComboScoringConfig -> FilePath -> [Combo] -> Int -> IO ()
writeTopJson scoringConfig path combos maxItems = do
    let eligible = filter (\combo -> sanitizeEq (comboFinalEquity combo) > 1 && comboOpenThresholdDeployable combo) combos
        sorted = take maxItems (sortBy (compareCombos scoringConfig) eligible)
    nowMs <- fmap (floor . (* 1000) :: POSIXTime -> Int) getPOSIXTime
    let comboValues = zipWith (comboToValue scoringConfig) [1 ..] sorted
        exportVal =
            object
                [ "generatedAtMs" .= nowMs
                , "source" .= ("merge_top_combos.py" :: String)
                , "combos" .= comboValues
                ]
    let dir = takeDirectory path
    createDirectoryIfMissing True dir
    -- Atomic write: write to temp in same dir then rename.
    (tmpPath, h) <- openBinaryTempFile dir "top-combos.tmp"
    BL.hPutStr h (encodePretty exportVal <> "\n")
    hFlush h
    hClose h
    renameFile tmpPath path

compareCombos :: TopComboScoringConfig -> Combo -> Combo -> Ordering
compareCombos config a b =
    compare (comboPerformanceKey config a) (comboPerformanceKey config b)

comboPerformanceKey :: TopComboScoringConfig -> Combo -> (Int, Double, Double, Double, Double)
comboPerformanceKey config combo =
    let ann = comboAnnualizedReturn combo
        ann'
            | comboLiveQuarantined config combo = negate (1 / 0)
            | comboProcessingTierRank config combo > comboProcessingTierRankForCandidate = negate (1 / 0)
            | otherwise = blendedAnnualizedReturn config ann (comboLiveStats combo)
        score = sanitizeScore (fromMaybe (-(1 / 0)) (comboScore combo))
        eq = sanitizeEq (comboFinalEquity combo)
     in (comboProcessingTierRank config combo, negate (comboValidatedScore config combo), negate ann', negate score, negate eq)

comboMetricDouble :: String -> Combo -> Maybe Double
comboMetricDouble key combo = do
    metrics <- comboMetrics combo
    KM.lookup (Key.fromString key) metrics >>= AT.parseMaybe Aeson.parseJSON

comboMetricInt :: String -> Combo -> Maybe Int
comboMetricInt key combo = do
    metrics <- comboMetrics combo
    KM.lookup (Key.fromString key) metrics >>= coerceIntValue

comboAnnualizedReturnMaybe :: Combo -> Maybe Double
comboAnnualizedReturnMaybe = comboMetricDouble "annualizedReturn"

comboAnnualizedReturn :: Combo -> Double
comboAnnualizedReturn combo =
    let ann = fromMaybe (-(1 / 0)) (comboAnnualizedReturnMaybe combo)
     in if isNaN ann || isInfinite ann then -(1 / 0) else ann

comboWalkForwardSharpeMean :: Combo -> Maybe Double
comboWalkForwardSharpeMean combo = do
    metrics <- comboMetrics combo
    wf <- KM.lookup (Key.fromString "walkForwardSummary") metrics
    case wf of
        Object obj -> KM.lookup (Key.fromString "sharpeMean") obj >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboProcessingTier :: TopComboScoringConfig -> Combo -> String
comboProcessingTier config combo
    | comboLiveQuarantined config combo = "quarantined"
    | not (comboTradeCountMeetsAdoptionFloor (comboMetricInt "tradeCount" combo)) = "raw"
    | isNothing (comboAnnualizedReturnMaybe combo) = "raw"
    | comboWalkForwardSharpeMeetsAdoptionFloor (comboWalkForwardSharpeMean combo) = "deployable"
    | otherwise = "candidate"

comboProcessingTierRank :: TopComboScoringConfig -> Combo -> Int
comboProcessingTierRank config combo =
    case comboProcessingTier config combo of
        "deployable" -> 0
        "candidate" -> 1
        "raw" -> 2
        _ -> 3

comboProcessingTierRankForCandidate :: Int
comboProcessingTierRankForCandidate = 1

comboProcessingReasons :: TopComboScoringConfig -> Combo -> [String]
comboProcessingReasons config combo =
    concat
        [ ["live-quarantined" | comboLiveQuarantined config combo]
        , case comboMetricInt "tradeCount" combo of
            Nothing -> ["trade-count-missing"]
            Just trades
                | not (comboTradeCountMeetsAdoptionFloor (Just trades)) -> ["trade-count-below-floor"]
                | otherwise -> []
        , ["annualized-return-missing" | isNothing (comboAnnualizedReturnMaybe combo)]
        , case comboWalkForwardSharpeMean combo of
            Nothing -> ["walk-forward-missing"]
            Just sharpe
                | not (comboWalkForwardSharpeMeetsAdoptionFloor (Just sharpe)) -> ["walk-forward-below-floor"]
                | otherwise -> []
        ]

comboValidatedScore :: TopComboScoringConfig -> Combo -> Double
comboValidatedScore config combo =
    let ann = comboAnnualizedReturn combo
        annLive =
            if comboLiveQuarantined config combo || isNaN ann || isInfinite ann
                then 0
                else topComboValidatedAnnualizedReturn config (blendedAnnualizedReturn config ann (comboLiveStats combo))
        trades = max 0 (fromMaybe 0 (comboMetricInt "tradeCount" combo))
        tradeShrinkage =
            let n = fromIntegral trades
                floorN = fromIntegral adoptionMinTradeCount
             in if n <= 0 then 0 else n / (n + floorN)
        wfMultiplier =
            topComboWalkForwardMultiplier
                config
                (comboWalkForwardSharpeMeetsAdoptionFloor . Just <$> comboWalkForwardSharpeMean combo)
        drawdown = max 0 (fromMaybe 0 (comboMetricDouble "maxDrawdown" combo))
        drawdownMultiplier = topComboDrawdownMultiplier config drawdown
        equityTerm = topComboEquityTerm config (sanitizeEq (comboFinalEquity combo))
     in annLive * tradeShrinkage * wfMultiplier * drawdownMultiplier + equityTerm

comboProcessingValue :: TopComboScoringConfig -> Combo -> Value
comboProcessingValue config combo =
    object
        [ "tier" .= comboProcessingTier config combo
        , "tierRank" .= comboProcessingTierRank config combo
        , "validatedScore" .= comboValidatedScore config combo
        , "reasons" .= comboProcessingReasons config combo
        ]

liveBlendShrinkageOps :: Double
liveBlendShrinkageOps = tcscLiveBlendShrinkageOps defaultTopComboScoringConfig

liveQuarantineMinOperations :: Int
liveQuarantineMinOperations = tcscLiveQuarantineMinOperations defaultTopComboScoringConfig

liveQuarantineMaxFinalEquity :: Double
liveQuarantineMaxFinalEquity = tcscLiveQuarantineMaxFinalEquity defaultTopComboScoringConfig

liveAnnualizedReturnFloor :: Double
liveAnnualizedReturnFloor = tcscLiveAnnualizedReturnFloor defaultTopComboScoringConfig

liveAnnualizedReturnCeiling :: Double
liveAnnualizedReturnCeiling = tcscLiveAnnualizedReturnCeiling defaultTopComboScoringConfig

clampLiveAnnualizedReturn :: Double -> Double
clampLiveAnnualizedReturn =
    clampLiveAnnualizedReturnWithConfig defaultTopComboScoringConfig

comboLiveStats :: Combo -> Maybe ComboLiveStats
comboLiveStats combo = do
    metrics <- comboMetrics combo
    liveVal <- KM.lookup (Key.fromString "live") metrics
    liveObj <-
        case liveVal of
            Object obj -> Just obj
            _ -> Nothing
    eq <- KM.lookup (Key.fromString "finalEquity") liveObj >>= coerceFloatValue
    count <- KM.lookup (Key.fromString "operationCount") liveObj >>= coerceIntValue
    if eq < 0 || count < 0
        then Nothing
        else
            pure
                ComboLiveStats
                    { clsFinalEquity = eq
                    , clsAnnualizedReturn = KM.lookup (Key.fromString "annualizedReturn") liveObj >>= coerceFloatValue
                    , clsOperationCount = count
                    }

comboLiveQuarantined :: TopComboScoringConfig -> Combo -> Bool
comboLiveQuarantined config combo =
    maybe False (liveStatsQuarantined config) (comboLiveStats combo)

liveStatsQuarantined :: TopComboScoringConfig -> ComboLiveStats -> Bool
liveStatsQuarantined config stats =
    topComboLiveQuarantinedByConfig
        config
        (clsOperationCount stats)
        (clsFinalEquity stats)

blendedAnnualizedReturn :: TopComboScoringConfig -> Double -> Maybe ComboLiveStats -> Double
blendedAnnualizedReturn config backtestAnn mLive =
    case mLive >>= liveAnnWithCount of
        Nothing -> backtestAnn
        Just (liveAnn, n) ->
            let w = topComboLiveBlendWeight config n
             in w * clampLiveAnnualizedReturnWithConfig config liveAnn + (1 - w) * backtestAnn
  where
    liveAnnWithCount stats = do
        ann <- clsAnnualizedReturn stats
        if isNaN ann || isInfinite ann || clsOperationCount stats <= 0
            then Nothing
            else Just (ann, clsOperationCount stats)

comboOpenThresholdDeployable :: Combo -> Bool
comboOpenThresholdDeployable combo =
    maybe True signalEntryOpenThresholdFeasible (comboOpenThreshold combo)

sanitizeScore :: Double -> Double
sanitizeScore score
    | isNaN score || isInfinite score = -(1 / 0)
    | otherwise = score

sanitizeEq :: Double -> Double
sanitizeEq eq
    | isNaN eq || isInfinite eq = 0
    | otherwise = eq

calcAnnualizedReturn :: Double -> Double -> Int -> Double
calcAnnualizedReturn finalEq periodsPerYear periods
    | periodsPerYear <= 0 = 0
    | periods <= 0 = 0
    | otherwise = finalEq ** (periodsPerYear / fromIntegral periods) - 1

metricsHasAnnualizedReturn :: KM.KeyMap Value -> Bool
metricsHasAnnualizedReturn metrics =
    case KM.lookup (Key.fromString "annualizedReturn") metrics >>= coerceFloatValue of
        Just v -> not (isNaN v || isInfinite v)
        Nothing -> False

ensureAnnualizedReturnMetrics :: Maybe (KM.KeyMap Value) -> Double -> Double -> Maybe Int -> Maybe (KM.KeyMap Value)
ensureAnnualizedReturnMetrics metrics finalEq periodsPerYear mPeriods =
    case metrics of
        Just m
            | metricsHasAnnualizedReturn m -> Just m
        _ ->
            case positivePeriodCount mPeriods of
                Just periods ->
                    let eq = max 0 (sanitizeEq finalEq)
                        annRet = calcAnnualizedReturn eq periodsPerYear periods
                        annVal = toJSON annRet
                        addMetric = KM.insert (Key.fromString "annualizedReturn") annVal
                     in case metrics of
                            Just m -> Just (addMetric m)
                            Nothing -> Just (KM.fromList [(Key.fromString "annualizedReturn", annVal)])
                Nothing -> metrics

compareDesc :: (Ord a) => a -> a -> Ordering
compareDesc a b
    | a > b = LT
    | a < b = GT
    | otherwise = EQ

comboToValue :: TopComboScoringConfig -> Int -> Combo -> Value
comboToValue scoringConfig rank combo =
    let metricsVal =
            maybe Null Object (comboMetrics combo)
        uuidVal = comboUuid combo
        base =
            object
                [ "uuid" .= uuidVal
                , "rank" .= rank
                , "createdAtMs" .= comboCreatedAtMs combo
                , "finalEquity" .= comboFinalEquity combo
                , "objective" .= comboObjective combo
                , "score" .= comboScore combo
                , "openThreshold" .= comboOpenThreshold combo
                , "closeThreshold" .= comboCloseThreshold combo
                , "source" .= comboSource combo
                , "metrics" .= metricsVal
                , "processing" .= comboProcessingValue scoringConfig combo
                , "params" .= Object (comboParams combo)
                ]
     in case comboOperations combo of
            Just ops -> addField "operations" (Array (V.fromList ops)) base
            Nothing -> base

comboIdentityValue :: Combo -> Value
comboIdentityValue = comboProcessingIdentityValue

comboUuid :: Combo -> String
comboUuid combo =
    formatUuidFromHex (hashBytesHex (encodePretty (comboIdentityValue combo)))

formatUuidFromHex :: String -> String
formatUuidFromHex hex =
    let h = take 32 hex
        seg1 = take 8 h
        seg2 = take 4 (drop 8 h)
        seg3 = take 4 (drop 12 h)
        seg4 = take 4 (drop 16 h)
        seg5 = take 12 (drop 20 h)
     in intercalate "-" [seg1, seg2, seg3, seg4, seg5]

hashBytesHex :: BL.ByteString -> String
hashBytesHex bs =
    let digest :: Digest SHA256
        digest = hash (BL.toStrict bs)
     in BS8.unpack (B16.encode (convert digest))

archiveTopJson :: Maybe FilePath -> FilePath -> IO ()
archiveTopJson historyDir outPath =
    case historyDir of
        Nothing -> pure ()
        Just dir -> do
            dirResult <- try (createDirectoryIfMissing True dir) :: IO (Either SomeException ())
            case dirResult of
                Left _ -> pure ()
                Right _ -> do
                    nowMs <- fmap (floor . (* 1000)) getPOSIXTime
                    let archivePath = dir </> ("top-combos-" ++ show (nowMs :: Integer) ++ ".json")
                    _ <- try (copyFile outPath archivePath) :: IO (Either SomeException ())
                    pure ()

copyToDist :: FilePath -> IO ()
copyToDist outPath = do
    let distDir = "haskell" </> "web" </> "dist"
    exists <- doesDirectoryExist distDir
    when exists $ do
        let target = distDir </> "top-combos.json"
        _ <- try (copyFile outPath target) :: IO (Either SomeException ())
        pure ()

normalizeCombo :: Value -> Maybe Combo
normalizeCombo value =
    case value of
        Object obj -> do
            let metricsRaw = nestedMetricsObject obj
            finalEqRaw <- comboFloatField "finalEquity" obj metricsRaw
            let finalEq = sanitizeEq finalEqRaw
            if finalEq <= 1
                then Nothing
                else do
                    let source = normalizeSource (KM.lookup (Key.fromString "source") obj)
                        paramsRaw =
                            case KM.lookup (Key.fromString "params") obj of
                                Just (Object p) -> p
                                _ -> KM.empty
                        platform = normalizePlatform paramsRaw source
                        objective = KM.lookup (Key.fromString "objective") obj >>= normalizeObjectiveValue
                        score = comboFloatField "score" obj metricsRaw
                        operations = KM.lookup (Key.fromString "operations") obj >>= coerceOperations
                        createdAtMs = KM.lookup (Key.fromString "createdAtMs") obj >>= coerceIntValue
                        interval = valueToStringMaybe (KM.lookup (Key.fromString "interval") paramsRaw)
                        bars = fromMaybe 0 (KM.lookup (Key.fromString "bars") paramsRaw >>= coerceIntValue)
                        periods =
                            positivePeriodCount
                                ( positivePeriodCount (Just bars)
                                    <|> comboIntField "periods" obj metricsRaw
                                )
                        periodsPerYear =
                            fromMaybe
                                (inferPeriodsPerYear interval)
                                (KM.lookup (Key.fromString "periodsPerYear") paramsRaw >>= coerceFloatValue)
                        metrics = ensureAnnualizedReturnMetrics metricsRaw finalEq periodsPerYear periods
                        method = valueToStringMaybe (KM.lookup (Key.fromString "method") paramsRaw)
                        normalization = valueToStringMaybe (KM.lookup (Key.fromString "normalization") paramsRaw)
                        epochs = fromMaybe 0 (KM.lookup (Key.fromString "epochs") paramsRaw >>= coerceIntValue)
                        hiddenSize = KM.lookup (Key.fromString "hiddenSize") paramsRaw >>= coerceIntValue
                        learningRate = KM.lookup (Key.fromString "learningRate") paramsRaw >>= coerceFloatValue
                        valRatio = KM.lookup (Key.fromString "valRatio") paramsRaw >>= coerceFloatValue
                        patience = KM.lookup (Key.fromString "patience") paramsRaw >>= coerceIntValue
                        gradClip = KM.lookup (Key.fromString "gradClip") paramsRaw >>= coerceFloatValue
                        positioning = valueToStringMaybe (KM.lookup (Key.fromString "positioning") paramsRaw)
                        slippage = KM.lookup (Key.fromString "slippage") paramsRaw >>= coerceFloatValue
                        spread = KM.lookup (Key.fromString "spread") paramsRaw >>= coerceFloatValue
                        -- Missing costAwareEdge must fail SAFE (cost-aware on): defaulting it
                        -- off silently stripped the break-even edge floor from every combo
                        -- that passed through a merge.
                        costAwareEdge = fromMaybe True (KM.lookup (Key.fromString "costAwareEdge") paramsRaw >>= coerceBoolValue)
                        triLayer = fromMaybe False (KM.lookup (Key.fromString "triLayer") paramsRaw >>= coerceBoolValue)
                        confirmConformal = fromMaybe False (KM.lookup (Key.fromString "confirmConformal") paramsRaw >>= coerceBoolValue)
                        confirmQuantiles = fromMaybe False (KM.lookup (Key.fromString "confirmQuantiles") paramsRaw >>= coerceBoolValue)
                        confidenceSizing = fromMaybe False (KM.lookup (Key.fromString "confidenceSizing") paramsRaw >>= coerceBoolValue)
                        symbolRaw =
                            fromMaybe
                                Null
                                ( KM.lookup (Key.fromString "binanceSymbol") paramsRaw
                                    <|> KM.lookup (Key.fromString "symbol") paramsRaw
                                )
                        symbol = normalizeSymbolValue platform symbolRaw
                        normalizedParams =
                            KM.union
                                ( KM.fromList
                                    [ (Key.fromString "platform", maybe Null (String . T.pack) platform)
                                    , (Key.fromString "interval", String (T.pack interval))
                                    , (Key.fromString "bars", Number (fromIntegral bars))
                                    , (Key.fromString "method", String (T.pack method))
                                    , (Key.fromString "normalization", String (T.pack normalization))
                                    , (Key.fromString "positioning", String (T.pack positioning))
                                    , (Key.fromString "baseOpenThreshold", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "baseOpenThreshold") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "baseCloseThreshold", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "baseCloseThreshold") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "blendWeight", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "blendWeight") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "minHoldBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "minHoldBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "cooldownBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "cooldownBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "maxHoldBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "maxHoldBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "minEdge", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "minEdge") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "edgeBuffer", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "edgeBuffer") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "costAwareEdge", Bool costAwareEdge)
                                    , (Key.fromString "trendLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "trendLookback") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "maxPositionSize", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxPositionSize") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "volTarget", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volTarget") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "volLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "volLookback") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "volEwmaAlpha", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volEwmaAlpha") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "volFloor", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volFloor") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "volScaleMax", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "volScaleMax") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxVolatility", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxVolatility") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "periodsPerYear", Number (fromFloatDigits periodsPerYear))
                                    , (Key.fromString "kalmanMarketTopN", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "kalmanMarketTopN") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "fee", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "fee") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "epochs", Number (fromIntegral epochs))
                                    , (Key.fromString "hiddenSize", maybe Null (Number . fromIntegral) hiddenSize)
                                    , (Key.fromString "learningRate", maybe Null (Number . fromFloatDigits) learningRate)
                                    , (Key.fromString "valRatio", maybe Null (Number . fromFloatDigits) valRatio)
                                    , (Key.fromString "patience", maybe Null (Number . fromIntegral) patience)
                                    , (Key.fromString "walkForwardFolds", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "walkForwardFolds") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "tuneStressVolMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressVolMult") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "tuneStressShock", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressShock") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "tuneStressWeight", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "tuneStressWeight") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "gradClip", maybe Null (Number . fromFloatDigits) gradClip)
                                    , (Key.fromString "slippage", maybe Null (Number . fromFloatDigits) slippage)
                                    , (Key.fromString "spread", maybe Null (Number . fromFloatDigits) spread)
                                    , (Key.fromString "intrabarFill", maybe Null (String . T.pack) (valueToStringMaybeMaybe (KM.lookup (Key.fromString "intrabarFill") paramsRaw)))
                                    , (Key.fromString "triLayer", Bool triLayer)
                                    , (Key.fromString "triLayerFastMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerFastMult") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerSlowMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerSlowMult") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerCloudPadding", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudPadding") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerCloudSlope", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudSlope") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerCloudWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerCloudWidth") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerTouchLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "triLayerTouchLookback") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "triLayerPriceAction", maybe Null Bool (KM.lookup (Key.fromString "triLayerPriceAction") paramsRaw >>= coerceBoolValue))
                                    , (Key.fromString "triLayerPriceActionBody", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerPriceActionBody") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerPriceActionWickRatio", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerPriceActionWickRatio") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerPriceActionOppositeWickMax", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerPriceActionOppositeWickMax") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerPriceActionBodyTolerance", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "triLayerPriceActionBodyTolerance") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "triLayerExitOnSlow", maybe Null Bool (KM.lookup (Key.fromString "triLayerExitOnSlow") paramsRaw >>= coerceBoolValue))
                                    , (Key.fromString "kalmanBandLookback", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "kalmanBandLookback") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "kalmanBandStdMult", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanBandStdMult") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "lstmExitFlipBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "lstmExitFlipBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "lstmExitFlipGraceBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "lstmExitFlipGraceBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "lstmExitFlipStrong", maybe Null Bool (KM.lookup (Key.fromString "lstmExitFlipStrong") paramsRaw >>= coerceBoolValue))
                                    , (Key.fromString "lstmConfidenceSoft", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "lstmConfidenceSoft") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "lstmConfidenceHard", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "lstmConfidenceHard") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "stopLoss", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "stopLoss") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "takeProfit", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "takeProfit") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "trailingStop", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "trailingStop") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "riskPerTrade", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "riskPerTrade") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxDrawdown", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxDrawdown") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxDailyLoss", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxDailyLoss") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxOrderErrors", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "maxOrderErrors") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "kalmanDt", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanDt") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanProcessVar", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanProcessVar") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanMeasurementVar", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanMeasurementVar") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsBars", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "kalmanPhysicsBars") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "kalmanPhysicsBacktestRatio", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsBacktestRatio") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsVolumeEwmaAlpha", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsVolumeEwmaAlpha") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsVolumeSignalClamp", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsVolumeSignalClamp") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsCloseBiasScale", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsCloseBiasScale") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsCandidateValidationRatio", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsCandidateValidationRatio") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsSmallSampleValidationRatio", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsSmallSampleValidationRatio") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanPhysicsCandidateTrees", maybe Null (Number . fromIntegral) (KM.lookup (Key.fromString "kalmanPhysicsCandidateTrees") paramsRaw >>= coerceIntValue))
                                    , (Key.fromString "kalmanPhysicsCandidateLearningRate", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanPhysicsCandidateLearningRate") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanZMin", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanZMin") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "kalmanZMax", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "kalmanZMax") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxHighVolProb", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxHighVolProb") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxConformalWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxConformalWidth") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "maxQuantileWidth", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "maxQuantileWidth") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "confirmConformal", Bool confirmConformal)
                                    , (Key.fromString "confirmQuantiles", Bool confirmQuantiles)
                                    , (Key.fromString "confidenceSizing", Bool confidenceSizing)
                                    , (Key.fromString "protectionMinConfidence", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "protectionMinConfidence") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "minPositionSize", maybe Null (Number . fromFloatDigits) (KM.lookup (Key.fromString "minPositionSize") paramsRaw >>= coerceFloatValue))
                                    , (Key.fromString "binanceSymbol", maybe Null (String . T.pack) symbol)
                                    ]
                                )
                                paramsRaw
                        openThreshold = KM.lookup (Key.fromString "openThreshold") obj >>= coerceFloatValue
                        closeThreshold = KM.lookup (Key.fromString "closeThreshold") obj >>= coerceFloatValue
                    pure
                        Combo
                            { comboFinalEquity = finalEq
                            , comboObjective = objective
                            , comboScore = score
                            , comboOpenThreshold = openThreshold
                            , comboCloseThreshold = closeThreshold
                            , comboCreatedAtMs = createdAtMs
                            , comboSource = source
                            , comboMetrics = metrics
                            , comboOperations = operations
                            , comboParams = normalizedParams
                            }
        _ -> Nothing

normalizeSource :: Maybe Value -> Maybe String
normalizeSource value =
    value >>= trimmedNonEmptyStringValue

trimmedNonEmptyStringValue :: Value -> Maybe String
trimmedNonEmptyStringValue value =
    case value of
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed then Nothing else Just trimmed
        _ -> Nothing

applySourceValue :: Maybe String -> Value -> Value
applySourceValue source val =
    case (source, val) of
        (Just src, Object obj) ->
            case KM.lookup (Key.fromString "source") obj of
                Just Null -> Object (KM.insert (Key.fromString "source") (toJSON src) obj)
                Just _ -> val
                Nothing -> Object (KM.insert (Key.fromString "source") (toJSON src) obj)
        _ -> val

normalizePlatform :: KM.KeyMap Value -> Maybe String -> Maybe String
normalizePlatform params source =
    let fromParams =
            case KM.lookup (Key.fromString "platform") params of
                Just (String s) -> normalizeKnownPlatform (T.unpack s)
                _ -> Nothing
        fromBinanceSymbol =
            case KM.lookup (Key.fromString "binanceSymbol") params of
                Just Null -> Nothing
                Just _ -> Just "binance"
                Nothing -> Nothing
        fromSource = source >>= normalizeKnownPlatform
     in fromParams <|> fromBinanceSymbol <|> fromSource

normalizeKnownPlatform :: String -> Maybe String
normalizeKnownPlatform raw =
    let normalized = canonicalPlatformKey raw
     in if normalized `elem` knownPlatformKeys
            then Just normalized
            else Nothing

knownPlatformKeys :: [String]
knownPlatformKeys =
    [ "binance"
    , "coinbase"
    , "kraken"
    , "poloniex"
    , "uniswap"
    , "curve"
    , "sushiswap"
    , "balancer"
    , "pancakeswap"
    , "1inch"
    ]

canonicalPlatformKey :: String -> String
canonicalPlatformKey raw =
    let key = map toLower (trim raw)
     in if "coinbase" `isPrefixOf` key
            then "coinbase"
            else
                if "poloniex" `isPrefixOf` key
                    then "poloniex"
                    else
                        if "binance" `isPrefixOf` key
                            then "binance"
                            else
                                if "uniswap" `isPrefixOf` key
                                    then "uniswap"
                                    else
                                        if "curve" `isPrefixOf` key
                                            then "curve"
                                            else
                                                if "sushiswap" `isPrefixOf` key
                                                    then "sushiswap"
                                                    else
                                                        if "balancer" `isPrefixOf` key
                                                            then "balancer"
                                                            else
                                                                if "pancakeswap" `isPrefixOf` key
                                                                    then "pancakeswap"
                                                                    else
                                                                        if "1inch" `isPrefixOf` key || "oneinch" `isPrefixOf` key
                                                                            then "1inch"
                                                                            else key

normalizeObjectiveValue :: Value -> Maybe String
normalizeObjectiveValue value =
    let s = map toLower (trim (valueToString value))
     in if null s then Nothing else Just s

normalizeSymbolValue :: Maybe String -> Value -> Maybe String
normalizeSymbolValue platform value =
    sanitizeComboSymbolForPlatform platform (valueToString value)

valueToStringMaybe :: Maybe Value -> String
valueToStringMaybe = maybe "" valueToString

valueToStringMaybeMaybe :: Maybe Value -> Maybe String
valueToStringMaybeMaybe value =
    case value of
        Nothing -> Nothing
        Just Null -> Nothing
        Just v -> Just (valueToString v)

valueToString :: Value -> String
valueToString value =
    case value of
        String s -> T.unpack s
        Number n -> formatScientific Generic Nothing n
        Bool True -> "True"
        Bool False -> "False"
        Null -> ""
        _ -> T.unpack (TE.decodeUtf8 (BL.toStrict (Aeson.encode value)))

coerceFloatValue :: Value -> Maybe Double
coerceFloatValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToDouble n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case reads trimmed of
                        [(v, "")] -> if isInfinite v || isNaN v then Nothing else Just v
                        _ -> Nothing
        _ -> Nothing

coerceIntValue :: Value -> Maybe Int
coerceIntValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToBoundedInt n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case readMaybe trimmed :: Maybe Scientific of
                        Just n -> scientificToBoundedInt n
                        _ -> Nothing
        _ -> Nothing

coerceBoolValue :: Value -> Maybe Bool
coerceBoolValue value =
    case value of
        Null -> Nothing
        Bool v -> Just v
        Number n ->
            case scientificToDouble n of
                Just d -> Just (d /= 0)
                Nothing -> Nothing
        String s ->
            let trimmed = map toLower (trim (T.unpack s))
             in if trimmed `elem` ["1", "true", "t", "yes", "y", "on"]
                    then Just True
                    else
                        if trimmed `elem` ["0", "false", "f", "no", "n", "off"]
                            then Just False
                            else Nothing
        _ -> Nothing

coerceOperations :: Value -> Maybe [Value]
coerceOperations value =
    case value of
        Array arr ->
            let ops = [Object o | Object o <- V.toList arr]
             in if null ops then Nothing else Just ops
        _ -> Nothing

nestedMetricsObject :: KM.KeyMap Value -> Maybe (KM.KeyMap Value)
nestedMetricsObject obj =
    case KM.lookup (Key.fromString "metrics") obj of
        Just (Object metrics) -> Just metrics
        _ -> Nothing

comboFloatField :: String -> KM.KeyMap Value -> Maybe (KM.KeyMap Value) -> Maybe Double
comboFloatField key obj metrics =
    (KM.lookup (Key.fromString key) obj >>= coerceFloatValue)
        <|> (metrics >>= KM.lookup (Key.fromString key) >>= coerceFloatValue)

comboIntField :: String -> KM.KeyMap Value -> Maybe (KM.KeyMap Value) -> Maybe Int
comboIntField key obj metrics =
    (KM.lookup (Key.fromString key) obj >>= coerceIntValue)
        <|> (metrics >>= KM.lookup (Key.fromString key) >>= coerceIntValue)

positivePeriodCount :: Maybe Int -> Maybe Int
positivePeriodCount mPeriods =
    case mPeriods of
        Just periods | periods > 0 -> Just periods
        _ -> Nothing

valueTruthyFlag :: Value -> Bool
valueTruthyFlag value =
    fromMaybe False (coerceBoolValue value)

scientificToDouble :: Scientific -> Maybe Double
scientificToDouble n =
    let d = toRealFloat n
     in if isInfinite d || isNaN d then Nothing else Just d

scientificToBoundedInt :: Scientific -> Maybe Int
scientificToBoundedInt = toBoundedInteger

addField :: String -> Value -> Value -> Value
addField key value val =
    case val of
        Object obj -> Object (KM.insert (Key.fromString key) value obj)
        _ -> val

applyCreatedAtMs :: Maybe Int -> Value -> Value
applyCreatedAtMs createdAtMs val =
    case (createdAtMs, val) of
        (Just ts, Object obj) ->
            case KM.lookup (Key.fromString "createdAtMs") obj of
                Just Null -> Object (KM.insert (Key.fromString "createdAtMs") (toJSON ts) obj)
                Just _ -> val
                Nothing -> Object (KM.insert (Key.fromString "createdAtMs") (toJSON ts) obj)
        _ -> val
