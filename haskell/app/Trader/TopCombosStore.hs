{-# LANGUAGE OverloadedStrings #-}

module Trader.TopCombosStore (
    ComboBacktestApplyStats (..),
    ComboBacktestUpdate (..),
    ComboLiveStats (..),
    TopCombosMergeStats (..),
    TopCombosStore (..),
    applyComboUpdates,
    applyComboUpdatesWithStats,
    applyComboUpdatesKeepAllWithStats,
    blendedAnnualizedReturn,
    comboBacktestDueForRefresh,
    comboBacktestFreshnessMs,
    comboBacktestStaleAfterMs,
    compactTopCombosPayloadForSync,
    comboFinalEquityValue,
    comboIdentityKey,
    comboLiveQuarantined,
    comboLiveStats,
    comboLiveStatsFromObject,
    comboLiveStatsValue,
    comboMetricDouble,
    comboMetricInt,
    comboMetricsDouble,
    comboPerformanceKey,
    liveBlendShrinkageOps,
    liveQuarantineMinOperations,
    liveQuarantineMaxFinalEquity,
    liveStatsQuarantined,
    liveStatsFamilyQuarantined,
    aggregateLiveStats,
    setComboLiveStats,
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
    selectCombosForBacktestRefresh,
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
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Time.Clock (NominalDiffTime, UTCTime, diffUTCTime, getCurrentTime)
import qualified Data.Vector as V
import System.Directory (createDirectory, createDirectoryIfMissing, doesDirectoryExist, doesFileExist, getModificationTime, removeDirectory, removeFile, renameFile, setModificationTime)
import System.FilePath (takeDirectory)
import System.IO (Handle, hClose, openTempFile)
import System.IO.Error (isAlreadyExistsError)
import Text.Read (readMaybe)

import Trader.BotStartSemantics (adoptionMinTradeCount, comboTradeCountMeetsAdoptionFloor, comboWalkForwardSharpeMeetsAdoptionFloor)
import Trader.Duration (inferPeriodsPerYear)
import Trader.Optimizer.Json (encodePretty)
import Trader.SignalGates (signalEntryOpenThresholdFeasible)
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
                    let dropTombstones = comboDropTombstoneMap val
                        combosList = V.toList combos
                        (kept, changed) = foldl' apply ([], 0) combosList
                        apply (acc, count) comboVal =
                            -- A sub-1.0 combo carrying a refresh stamp is an honest
                            -- re-backtest reading, not junk: it must survive into the
                            -- merge so it can beat the stale, inflated copies peer
                            -- instances still publish (see 'pickBestCombo') and then
                            -- sink off the capped board by rank. Dropping it here
                            -- would resurrect the stale copy on every union merge.
                            if comboDroppedByTombstones dropTombstones comboVal
                                || (not (comboEquityAboveOne comboVal) && not (comboCarriesRefreshStamp comboVal))
                                || not (comboOpenThresholdDeployable comboVal)
                                then (acc, count + 1)
                                else
                                    let (comboVal', updated) = sanitizeComboValue comboVal
                                     in (comboVal' : acc, count + if updated then 1 else 0)
                        combosOut = Aeson.Array (V.fromList (reverse kept))
                        o' = insertComboDropTombstones dropTombstones (KM.insert (AK.fromString "combos") combosOut o)
                     in (Aeson.Object o', changed)
                _ -> (val, 0)
        _ -> (val, 0)

comboOpenThresholdDeployable :: Aeson.Value -> Bool
comboOpenThresholdDeployable val =
    maybe True signalEntryOpenThresholdFeasible (comboMetricDouble "openThreshold" val)

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
        case annotateComboProcessing val of
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

{- | Read a metric value as an integer, supporting both numeric and stringly
encoded fields. Returns 'Nothing' for non-finite or non-integral inputs.
Used by the bot-start guard to distinguish a zero-trade smoke window
(@tradeCount == 0@) from an actual loss reading.
-}
comboMetricInt :: String -> Aeson.Value -> Maybe Int
comboMetricInt key val = do
    d <- comboMetricDouble key val
    if isNaN d || isInfinite d
        then Nothing
        else
            let r = round d :: Integer
             in if fromIntegral r == d
                    then Just (fromInteger r)
                    else Nothing

comboMetricsDouble :: String -> Aeson.Value -> Maybe Double
comboMetricsDouble key val = do
    metrics <- comboMetricValue "metrics" val
    comboMetricDouble key metrics

comboMetricsInt :: String -> Aeson.Value -> Maybe Int
comboMetricsInt key val = do
    metrics <- comboMetricValue "metrics" val
    comboMetricInt key metrics

comboMetricObject :: String -> Aeson.Value -> Maybe Aeson.Object
comboMetricObject key val =
    case comboMetricValue key val of
        Just (Aeson.Object obj) -> Just obj
        _ -> Nothing

comboFinalEquityValue :: Aeson.Value -> Maybe Double
comboFinalEquityValue val =
    comboMetricDouble "finalEquity" val <|> comboMetricsDouble "finalEquity" val

comboScoreValue :: Aeson.Value -> Maybe Double
comboScoreValue val =
    comboMetricDouble "score" val <|> comboMetricsDouble "score" val

{- | Realized live-trading performance of a combo, accumulated from completed
@bot.order@ operations. Stored as a @live@ object inside the combo's metrics
so it survives the JSON round-trips between the DB, the top-combos file, and
the S3 bus, and stays clearly separated from backtest metrics.
-}
data ComboLiveStats = ComboLiveStats
    { clsFinalEquity :: !Double
    -- ^ Compounded product of per-order equity ratios; 1.0 = break-even.
    , clsAnnualizedReturn :: !(Maybe Double)
    -- ^ Annualized from the observed live span; Nothing until the span is
    -- long enough to annualize without exploding.
    , clsOperationCount :: !Int
    , clsFirstAtMs :: !(Maybe Int64)
    , clsLastAtMs :: !(Maybe Int64)
    }
    deriving (Eq, Show)

{- | Pseudo-count of live operations at which live evidence carries the same
weight as the backtest prior in the blended ranking (w = n / (n + k)).
-}
liveBlendShrinkageOps :: Double
liveBlendShrinkageOps = 25

-- | Minimum completed live orders before a combo can be quarantined.
liveQuarantineMinOperations :: Int
liveQuarantineMinOperations = 30

{- | A combo with at least 'liveQuarantineMinOperations' live orders whose
compounded live equity sits at or below this ceiling is quarantined: a real
net loss, not break-even noise.
-}
liveQuarantineMaxFinalEquity :: Double
liveQuarantineMaxFinalEquity = 0.99

-- | Minimum observed live span before annualizing live equity (1 day).
liveAnnualizationMinSpanMs :: Int64
liveAnnualizationMinSpanMs = 86400000

liveAnnualizedReturnFloor :: Double
liveAnnualizedReturnFloor = -0.9999

liveAnnualizedReturnCeiling :: Double
liveAnnualizedReturnCeiling = 10

clampLiveAnnualizedReturn :: Double -> Double
clampLiveAnnualizedReturn ann =
    max liveAnnualizedReturnFloor (min liveAnnualizedReturnCeiling ann)

liveAnnualizedReturnFromSpan :: Double -> Int64 -> Maybe Double
liveAnnualizedReturnFromSpan liveEq spanMs
    | spanMs < liveAnnualizationMinSpanMs = Nothing
    | liveEq <= 0 = Just liveAnnualizedReturnFloor
    | otherwise =
        let msPerYear = 365 * 86400000 :: Double
            exponent' = msPerYear / fromIntegral spanMs
            ann = (liveEq ** exponent') - 1
         in if isNaN ann || isInfinite ann
                then Nothing
                else Just (clampLiveAnnualizedReturn ann)

comboLiveStatsFromObject :: Aeson.Object -> Maybe ComboLiveStats
comboLiveStatsFromObject obj = do
    liveVal <- KM.lookup (AK.fromString "live") obj
    liveObj <-
        case liveVal of
            Aeson.Object o -> Just o
            _ -> Nothing
    eq <- KM.lookup (AK.fromString "finalEquity") liveObj >>= coerceDoubleValue
    count <- KM.lookup (AK.fromString "operationCount") liveObj >>= coerceIntValue
    if eq < 0 || count < 0
        then Nothing
        else
            pure
                ComboLiveStats
                    { clsFinalEquity = eq
                    , clsAnnualizedReturn =
                        KM.lookup (AK.fromString "annualizedReturn") liveObj
                            >>= coerceDoubleValue
                    , clsOperationCount = count
                    , clsFirstAtMs = KM.lookup (AK.fromString "firstAtMs") liveObj >>= coerceInt64Value
                    , clsLastAtMs = KM.lookup (AK.fromString "lastAtMs") liveObj >>= coerceInt64Value
                    }

{- | Read live stats from a combo JSON value (under @metrics.live@, falling
back to a root-level @live@ object).
-}
comboLiveStats :: Aeson.Value -> Maybe ComboLiveStats
comboLiveStats val =
    (metricsObjectMaybe val >>= comboLiveStatsFromObject)
        <|> (rootObjectMaybe val >>= comboLiveStatsFromObject)
  where
    metricsObjectMaybe v =
        case comboMetricValue "metrics" v of
            Just (Aeson.Object o) -> Just o
            _ -> Nothing
    rootObjectMaybe v =
        case v of
            Aeson.Object o -> Just o
            _ -> Nothing

comboLiveStatsValue :: ComboLiveStats -> Aeson.Value
comboLiveStatsValue stats =
    object
        ( [ "finalEquity" .= clsFinalEquity stats
          , "operationCount" .= clsOperationCount stats
          ]
            ++ maybe [] (\v -> ["annualizedReturn" .= v]) (clsAnnualizedReturn stats)
            ++ maybe [] (\v -> ["firstAtMs" .= v]) (clsFirstAtMs stats)
            ++ maybe [] (\v -> ["lastAtMs" .= v]) (clsLastAtMs stats)
        )

{- | Blend the backtest annualized return with realized live performance.

Shrinkage weighting: with n live orders, live evidence gets weight
@n / (n + liveBlendShrinkageOps)@. A combo with no (or too-short) live
history ranks purely on its backtest prior; live evidence takes over
gradually as orders accumulate, so a handful of noisy fills cannot
overturn the backtest, but a sustained live record can.
-}
blendedAnnualizedReturn :: Double -> Maybe ComboLiveStats -> Double
blendedAnnualizedReturn backtestAnn mLive =
    case mLive >>= liveAnnWithCount of
        Nothing -> backtestAnn
        Just (liveAnn, n) ->
            let w = fromIntegral n / (fromIntegral n + liveBlendShrinkageOps)
             in w * clampLiveAnnualizedReturn liveAnn + (1 - w) * backtestAnn
  where
    liveAnnWithCount stats = do
        ann <- clsAnnualizedReturn stats
        if isNaN ann || isInfinite ann || clsOperationCount stats <= 0
            then Nothing
            else Just (ann, clsOperationCount stats)

liveStatsQuarantined :: ComboLiveStats -> Bool
liveStatsQuarantined stats =
    clsOperationCount stats >= liveQuarantineMinOperations
        && clsFinalEquity stats <= liveQuarantineMaxFinalEquity

{- | Aggregate the live evidence of a strategy family (every combo that shares
the same stable trading identity — symbol/interval/method) into a single
@(totalOrders, equity)@ verdict. Orders sum; equity is the order-count-weighted
geometric mean of each member's compounded equity, i.e. the average per-order
log-return of the family. Returns 'Nothing' when no member carries usable live
evidence (positive, finite equity and at least one order).

The family view exists because a combo's UUID is derived from its backtest
result (objective + tuned thresholds), so re-discovering the same strategy
mints a fresh UUID and its accumulated live record — keyed by the old UUID —
is orphaned. A losing symbol can therefore churn through a sequence of UUIDs
that each individually stay under 'liveQuarantineMinOperations' and never
quarantine. Summing orders across the family closes that gap.
-}
aggregateLiveStats :: [ComboLiveStats] -> Maybe (Int, Double)
aggregateLiveStats statsList =
    let usable =
            [ s
            | s <- statsList
            , clsOperationCount s > 0
            , clsFinalEquity s > 0
            , not (isNaN (clsFinalEquity s) || isInfinite (clsFinalEquity s))
            ]
        totalOps = sum (map clsOperationCount usable)
     in if null usable || totalOps <= 0
            then Nothing
            else
                let weightedLogSum =
                        sum [fromIntegral (clsOperationCount s) * log (clsFinalEquity s) | s <- usable]
                    aggEq = exp (weightedLogSum / fromIntegral totalOps)
                 in Just (totalOps, aggEq)

{- | A strategy family is quarantined when its pooled live record clears the
same order floor and net-loss ceiling that quarantine a single combo. A
family of one reduces exactly to 'liveStatsQuarantined', so this is a strict
superset of the per-combo check.
-}
liveStatsFamilyQuarantined :: [ComboLiveStats] -> Bool
liveStatsFamilyQuarantined statsList =
    case aggregateLiveStats statsList of
        Nothing -> False
        Just (totalOps, aggEq) ->
            totalOps >= liveQuarantineMinOperations
                && aggEq <= liveQuarantineMaxFinalEquity

{- | A quarantined combo has enough live evidence to conclude it is losing
real money regardless of what its backtest claims. Quarantined combos sink
to the bottom of every ranking and are skipped when starting bots; they are
kept in the store (flagged by their live stats) so the verdict is visible
and survives merges instead of being silently re-added fresh.
-}
comboLiveQuarantined :: Aeson.Value -> Bool
comboLiveQuarantined val = maybe False liveStatsQuarantined (comboLiveStats val)

{- | Write live stats into a combo's @metrics.live@, creating the metrics
object if needed.
-}
setComboLiveStats :: ComboLiveStats -> Aeson.Value -> Aeson.Value
setComboLiveStats stats val =
    case val of
        Aeson.Object o ->
            let metricsObj =
                    case KM.lookup (AK.fromString "metrics") o of
                        Just (Aeson.Object m) -> m
                        _ -> KM.empty
                metricsObj' = KM.insert (AK.fromString "live") (comboLiveStatsValue stats) metricsObj
             in Aeson.Object (KM.insert (AK.fromString "metrics") (Aeson.Object metricsObj') o)
        _ -> val

{- | Ensure the chosen combo carries the richest live record (most live
operations) seen across all candidate duplicates, so merges and backtest
refreshes can never silently erase accumulated live evidence.
-}
preserveRichestLiveStats :: [Aeson.Value] -> Aeson.Value -> Aeson.Value
preserveRichestLiveStats candidates chosen =
    let richest =
            listToMaybe
                ( sortBy
                    (\a b -> compare (negate (clsOperationCount a)) (negate (clsOperationCount b)))
                    (Data.Maybe.mapMaybe comboLiveStats candidates)
                )
     in case richest of
            Just stats
                | maybe True ((< clsOperationCount stats) . clsOperationCount) (comboLiveStats chosen) ->
                    setComboLiveStats stats chosen
            _ -> chosen

coerceInt64Value :: Aeson.Value -> Maybe Int64
coerceInt64Value value =
    case AT.parseMaybe Aeson.parseJSON value of
        Just v -> Just v
        Nothing ->
            case value of
                Aeson.String s -> readMaybe (trim (T.unpack s))
                _ -> Nothing

comboTopLevelInt64 :: String -> Aeson.Value -> Maybe Int64
comboTopLevelInt64 key val =
    case val of
        Aeson.Object o -> KM.lookup (AK.fromString key) o >>= coerceInt64Value
        _ -> Nothing

-- | When this combo's backtest metrics were last refreshed in place.
comboBacktestRefreshedAtMs :: Aeson.Value -> Maybe Int64
comboBacktestRefreshedAtMs = comboTopLevelInt64 "backtestRefreshedAtMs"

comboCarriesRefreshStamp :: Aeson.Value -> Bool
comboCarriesRefreshStamp = isJust . comboBacktestRefreshedAtMs

{- | How fresh this combo's backtest reading is: the in-place refresh stamp
when present, else the discovery time. A freshly discovered duplicate is
itself a fresh backtest of the same identity, so 'createdAtMs' is the right
fallback when comparing against a refreshed record.
-}
comboBacktestFreshnessMs :: Aeson.Value -> Maybe Int64
comboBacktestFreshnessMs val =
    comboBacktestRefreshedAtMs val <|> comboTopLevelInt64 "createdAtMs" val

comboBacktestStaleAfterMs :: Int64
comboBacktestStaleAfterMs = 3 * 86400000

comboBacktestDueForRefresh :: Int64 -> Aeson.Value -> Bool
comboBacktestDueForRefresh now val =
    case comboBacktestFreshnessMs val of
        Nothing -> True
        Just refreshedAt -> now - refreshedAt > comboBacktestStaleAfterMs

selectCombosForBacktestRefresh :: Int -> Int64 -> [Aeson.Value] -> [Aeson.Value]
selectCombosForBacktestRefresh topNRaw now combos =
    let topN = max 1 topNRaw
        indexed = zip [0 :: Int ..] combos
        ranked = take topN (sortBy compareRank indexed)
        stale = filter (comboBacktestDueForRefresh now . snd) indexed
     in map snd (dedupeSelected (ranked ++ stale))
  where
    compareRank (_, a) (_, b) = compare (comboPerformanceKey a) (comboPerformanceKey b)

    dedupeSelected = reverse . fst . foldl' keep ([], Set.empty)

    keep (acc, seen) item@(_, comboVal) =
        case comboProcessingIdentityKey comboVal <|> comboIdentityKey comboVal of
            Just key
                | key `Set.member` seen -> (acc, seen)
                | otherwise -> (item : acc, Set.insert key seen)
            Nothing -> (item : acc, seen)

recalculateComboPerformanceFromOperation ::
    Int64 ->
    Maybe String ->
    Maybe Double ->
    Maybe Double ->
    Aeson.Object ->
    Maybe Double ->
    Double ->
    (Double, Double, Aeson.Object)
recalculateComboPerformanceFromOperation now mInterval mStoredFinalEq mStoredAnnualized metricsObj mPrevOrderEq currentOrderEq =
    -- No prior order equity from the SAME bot session means no measurable
    -- equity change: the session's model equity restarts near 1.0 on every
    -- bot start, so treating "no previous order" as prevEq=1 would book the
    -- session's absolute equity level (warmup backtest included) as a live
    -- ratio, and a restart would cancel whatever the previous session lost.
    let mPrevEq = positiveFiniteMaybe mPrevOrderEq
        currentEq = fromMaybe (fromMaybe 1 mPrevEq) (nonNegativeFinite currentOrderEq)
        ratioRaw =
            case mPrevEq of
                Just prevEq | prevEq > 0 -> currentEq / prevEq
                _ -> 1
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
        liveStats =
            fromMaybe
                ComboLiveStats
                    { clsFinalEquity = 1
                    , clsAnnualizedReturn = Nothing
                    , clsOperationCount = 0
                    , clsFirstAtMs = Nothing
                    , clsLastAtMs = Nothing
                    }
                (comboLiveStatsFromObject metricsObj)
        liveFirstAtMs = fromMaybe now (clsFirstAtMs liveStats)
        liveEqRaw = clsFinalEquity liveStats * ratio
        liveEq =
            if isFiniteDouble liveEqRaw && liveEqRaw >= 0
                then liveEqRaw
                else clsFinalEquity liveStats
        liveStats' =
            ComboLiveStats
                { clsFinalEquity = liveEq
                , clsAnnualizedReturn = liveAnnualizedReturnFromSpan liveEq (max 0 (now - liveFirstAtMs))
                , clsOperationCount = clsOperationCount liveStats + 1
                , clsFirstAtMs = Just liveFirstAtMs
                , clsLastAtMs = Just now
                }
        metricsObj' =
            KM.insert
                (AK.fromString "live")
                (comboLiveStatsValue liveStats')
                ( KM.insert
                    (AK.fromString "annualizedReturn")
                    (toJSON nextAnnualized)
                    (KM.insert (AK.fromString "finalEquity") (toJSON nextFinalEq) metricsObj)
                )
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

closeTimingRatioValue :: String -> Aeson.Object -> Maybe Double
closeTimingRatioValue key report =
    KM.lookup (AK.fromString key) report >>= coerceDoubleValue

closeTimingRatiosWithinAnalyzedBounds :: Aeson.Object -> Bool
closeTimingRatiosWithinAnalyzedBounds report =
    fromMaybe False $ do
        q25 <- closeTimingRatioValue "q25Ratio" report
        median <- closeTimingRatioValue "medianRatio" report
        q75 <- closeTimingRatioValue "q75Ratio" report
        pure
            ( q25 >= 0
                && q25 <= median
                && median <= q75
                && q75 <= 2
            )

closeTimingDurationSummaryConsistent :: Aeson.Object -> Int -> Bool
closeTimingDurationSummaryConsistent report supportBound =
    fromMaybe False $ do
        medianObservedDuration <- KM.lookup (AK.fromString "medianObservedDuration") report >>= coerceIntValue
        medianOptimalDuration <- KM.lookup (AK.fromString "medianOptimalDuration") report >>= coerceIntValue
        pure
            ( medianObservedDuration > 0
                && medianOptimalDuration > 0
                && medianOptimalDuration <= supportBound
            )

closeTimingRecommendationContractHolds :: Aeson.Object -> Int -> Int -> Int -> Int -> Double -> Int -> Int -> Bool
closeTimingRecommendationContractHolds report recommended supportBound sampleCount positiveLiftSampleCount medianLift positiveLiftFloor minimumSampleCount =
    recommended > 0
        && supportBound > 0
        && recommended == supportBound
        && sampleCount >= minimumSampleCount
        && medianLift > 0
        && hasSufficientCloseTimingPositiveLiftSupport sampleCount positiveLiftSampleCount positiveLiftFloor
        && closeTimingRatiosWithinAnalyzedBounds report
        && closeTimingDurationSummaryConsistent report supportBound

validatedCloseTimingRecommendation :: Aeson.Object -> Maybe Int
validatedCloseTimingRecommendation report = do
    recommended <- KM.lookup (AK.fromString "recommendedMaxHoldBars") report >>= coerceIntValue
    supportBound <- KM.lookup (AK.fromString "q75OptimalDuration") report >>= coerceIntValue
    sampleCount <- KM.lookup (AK.fromString "sampleCount") report >>= coerceIntValue
    positiveLiftSampleCount <- closeTimingEvidenceSampleCount report
    medianLift <- KM.lookup (AK.fromString "medianLift") report >>= coerceDoubleValue
    let positiveLiftFloor =
            max
                closeTimingRecommendationEvidenceFloor
                (fromMaybe closeTimingRecommendationEvidenceFloor (closeTimingPositiveLiftFloorMaybe report))
        minimumSampleCount =
            max
                closeTimingMinimumSampleCount
                (fromMaybe closeTimingMinimumSampleCount (closeTimingMinimumSampleCountMaybe report))
    if closeTimingRecommendationContractHolds
        report
        recommended
        supportBound
        sampleCount
        positiveLiftSampleCount
        medianLift
        positiveLiftFloor
        minimumSampleCount
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
comboMergeKey = comboProcessingIdentityKey

comboProcessingIdentityKey :: Aeson.Value -> Maybe BS.ByteString
comboProcessingIdentityKey val = do
    params <- comboMetricValue "params" val
    let openThr = comboMetricValue "openThreshold" val
        closeThr = comboMetricValue "closeThreshold" val
        objective = comboMetricValue "objective" val
        identity =
            object
                [ "params" .= normalizeComboIdentityValue params
                , "openThreshold" .= normalizeComboIdentityMaybe openThr
                , "closeThreshold" .= normalizeComboIdentityMaybe closeThr
                , "objective" .= normalizeComboIdentityMaybe objective
                ]
    pure (BL.toStrict (encodePretty identity))

normalizeComboIdentityMaybe :: Maybe Aeson.Value -> Maybe Aeson.Value
normalizeComboIdentityMaybe = fmap normalizeComboIdentityValue

normalizeComboIdentityValue :: Aeson.Value -> Aeson.Value
normalizeComboIdentityValue raw =
    case raw of
        Aeson.Object obj ->
            Aeson.Object $
                KM.fromList
                    [ (key, normalized)
                    | (key, val) <- KM.toList obj
                    , let normalized = normalizeComboIdentityValue val
                    , normalized /= Aeson.Null
                    ]
        Aeson.Array arr ->
            Aeson.Array (V.map normalizeComboIdentityValue arr)
        other -> other

{- | Ranking key: primarily the backtest annualized return blended with
realized live performance ('blendedAnnualizedReturn').

Quarantined combos and combos whose stored backtest evidence does not meet the
same trade-count floor required for live adoption sink to the bottom. This
keeps legacy one- or two-trade annualized-return outliers from occupying the
top leaderboard slots and blocking freshly discovered deployable candidates.
-}
comboPerformanceKey :: Aeson.Value -> (Int, Double, Double, Double, Double, Int)
comboPerformanceKey val =
    let ann = fromMaybe (negate (1 / 0)) (comboAnnualizedReturnValue val)
        eq = fromMaybe 0 (comboMetricDouble "finalEquity" val <|> comboMetricsDouble "finalEquity" val)
        score = fromMaybe (negate (1 / 0)) (comboScoreValue val)
        rank =
            case val of
                Aeson.Object o -> fromMaybe maxBound (KM.lookup (AK.fromString "rank") o >>= AT.parseMaybe Aeson.parseJSON)
                _ -> maxBound
        ann'
            | comboLiveQuarantined val = negate (1 / 0)
            | comboProcessingTierRank val > comboProcessingTierRankForCandidate = negate (1 / 0)
            | isNaN ann || isInfinite ann = negate (1 / 0)
            | otherwise = blendedAnnualizedReturn ann (comboLiveStats val)
        eq' = if isNaN eq || isInfinite eq then 0 else eq
        score' = if isNaN score || isInfinite score then negate (1 / 0) else score
     in (comboProcessingTierRank val, negate (comboValidatedScore val), negate ann', negate score', negate eq', rank)

comboProcessingTierRankForCandidate :: Int
comboProcessingTierRankForCandidate = 1

comboAnnualizedReturnValue :: Aeson.Value -> Maybe Double
comboAnnualizedReturnValue val =
    comboMetricsDouble "annualizedReturn" val <|> comboMetricDouble "annualizedReturn" val

comboTradeCountValue :: Aeson.Value -> Maybe Int
comboTradeCountValue val =
    comboMetricsInt "tradeCount" val <|> comboMetricInt "tradeCount" val

comboWalkForwardSharpeMeanValue :: Aeson.Value -> Maybe Double
comboWalkForwardSharpeMeanValue val = do
    metrics <- comboMetricObject "metrics" val
    wf <- KM.lookup (AK.fromString "walkForwardSummary") metrics
    case wf of
        Aeson.Object obj -> KM.lookup (AK.fromString "sharpeMean") obj >>= AT.parseMaybe Aeson.parseJSON
        _ -> Nothing

comboProcessingTier :: Aeson.Value -> String
comboProcessingTier val
    | comboLiveQuarantined val = "quarantined"
    | not tradeCountMeetsFloor = "raw"
    | isNothingFinite (comboAnnualizedReturnValue val) = "raw"
    | comboWalkForwardSharpeMeetsAdoptionFloor (comboWalkForwardSharpeMeanValue val) = "deployable"
    | otherwise = "candidate"
  where
    tradeCountMeetsFloor =
        comboTradeCountMeetsAdoptionFloor (comboTradeCountValue val)

    isNothingFinite Nothing = True
    isNothingFinite (Just x) = isNaN x || isInfinite x

comboProcessingTierRank :: Aeson.Value -> Int
comboProcessingTierRank val =
    case comboProcessingTier val of
        "deployable" -> 0
        "candidate" -> 1
        "raw" -> 2
        _ -> 3

comboProcessingReasons :: Aeson.Value -> [String]
comboProcessingReasons val =
    concat
        [ ["live-quarantined" | comboLiveQuarantined val]
        , case comboTradeCountValue val of
            Nothing -> ["trade-count-missing"]
            Just trades
                | not (comboTradeCountMeetsAdoptionFloor (Just trades)) -> ["trade-count-below-floor"]
                | otherwise -> []
        , case comboAnnualizedReturnValue val of
            Nothing -> ["annualized-return-missing"]
            Just ann
                | isNaN ann || isInfinite ann -> ["annualized-return-invalid"]
                | otherwise -> []
        , case comboWalkForwardSharpeMeanValue val of
            Nothing -> ["walk-forward-missing"]
            Just sharpe
                | not (comboWalkForwardSharpeMeetsAdoptionFloor (Just sharpe)) -> ["walk-forward-below-floor"]
                | otherwise -> []
        ]

comboValidatedScore :: Aeson.Value -> Double
comboValidatedScore val =
    let ann = fromMaybe 0 (comboAnnualizedReturnValue val)
        annLive =
            if comboLiveQuarantined val || isNaN ann || isInfinite ann
                then 0
                else min 20 (max 0 (blendedAnnualizedReturn ann (comboLiveStats val)))
        trades = max 0 (fromMaybe 0 (comboTradeCountValue val))
        tradeShrinkage =
            let n = fromIntegral trades
                floorN = fromIntegral adoptionMinTradeCount
             in if n <= 0 then 0 else n / (n + floorN)
        wfMultiplier =
            case comboWalkForwardSharpeMeanValue val of
                Just sharpe | comboWalkForwardSharpeMeetsAdoptionFloor (Just sharpe) -> 1.0
                Just _ -> 0.35
                Nothing -> 0.60
        drawdown = max 0 (fromMaybe 0 (comboMetricsDouble "maxDrawdown" val <|> comboMetricDouble "maxDrawdown" val))
        drawdownMultiplier = 1 / (1 + 10 * drawdown)
        eq = max 1.0e-9 (fromMaybe 1 (comboFinalEquityValue val))
        equityTerm = max (-1) (log eq)
     in annLive * tradeShrinkage * wfMultiplier * drawdownMultiplier + equityTerm

comboProcessingValue :: Aeson.Value -> Aeson.Value
comboProcessingValue val =
    object
        [ "tier" .= comboProcessingTier val
        , "tierRank" .= comboProcessingTierRank val
        , "validatedScore" .= comboValidatedScore val
        , "reasons" .= comboProcessingReasons val
        ]

annotateComboProcessing :: Aeson.Value -> Aeson.Value
annotateComboProcessing val =
    case val of
        Aeson.Object o -> Aeson.Object (KM.insert (AK.fromString "processing") (comboProcessingValue val) o)
        _ -> val

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
        dropTombstones = mergeComboDropTombstones sanitized
        combos =
            concatMap
                (filter (not . comboDroppedByTombstones dropTombstones) . extractCombos)
                sanitized
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
            insertComboDropTombstones dropTombstones $
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
            || key == comboDropTombstonesKey

    mergeCombo acc comboVal =
        case comboMergeKey comboVal of
            Nothing -> acc
            Just key ->
                case M.lookup key acc of
                    Nothing -> M.insert key comboVal acc
                    Just prev -> M.insert key (pickBestCombo comboVal prev) acc

    addRank :: Int -> Aeson.Value -> Aeson.Value
    addRank rank val =
        case annotateComboProcessing val of
            Aeson.Object o -> Aeson.Object (KM.insert (AK.fromString "rank") (toJSON rank) o)
            other -> other

    compareCombos :: Aeson.Value -> Aeson.Value -> Ordering
    compareCombos a b = compare (comboPerformanceKey a) (comboPerformanceKey b)

    pickBestCombo newer prev =
        let objNew = comboMetricString "objective" newer
            objPrev = comboMetricString "objective" prev
            scoreNew = comboScoreValue newer
            scorePrev = comboScoreValue prev
            scoreVal = fromMaybe (negate (1 / 0))
            finalEqNew = fromMaybe 0 (comboFinalEquityValue newer)
            finalEqPrev = fromMaybe 0 (comboFinalEquityValue prev)
            -- Once either duplicate carries an in-place refresh stamp, the
            -- most recent backtest reading is authoritative — best-ever-score
            -- semantics would let a stale replica's inflated record undo an
            -- honest (deflating) refresh on every union merge. Unstamped
            -- duplicates keep the historical best-ever behavior.
            anyStamped = isJust (comboBacktestRefreshedAtMs newer) || isJust (comboBacktestRefreshedAtMs prev)
            freshNew = comboBacktestFreshnessMs newer
            freshPrev = comboBacktestFreshnessMs prev
            best
                | anyStamped
                , fn /= fp =
                    if fn > fp then newer else prev
                | objNew == objPrev && (isJust scoreNew || isJust scorePrev) =
                    if scoreVal scoreNew > scoreVal scorePrev then newer else prev
                | finalEqNew > finalEqPrev = newer
                | otherwise = prev
              where
                fn = freshNew
                fp = freshPrev
         in preserveRichestLiveStats [newer, prev] best

    comboMetricString key val =
        comboMetricValue key val >>= valueStringMaybe

data ComboBacktestUpdate = ComboBacktestUpdate
    { cbuMetrics :: !Aeson.Value
    , cbuFinalEquity :: !(Maybe Double)
    , cbuScore :: !(Maybe Double)
    , cbuOperations :: !(Maybe Aeson.Value)
    }

comboDropTombstonesKey :: AK.Key
comboDropTombstonesKey = AK.fromString "droppedComboIdentities"

comboDropTombstoneIdentityKey :: AK.Key
comboDropTombstoneIdentityKey = AK.fromString "identityKey"

comboDropTombstoneDroppedAtKey :: AK.Key
comboDropTombstoneDroppedAtKey = AK.fromString "droppedAtMs"

comboDropTombstoneMap :: Aeson.Value -> M.Map BS.ByteString Int64
comboDropTombstoneMap val =
    case val of
        Aeson.Object o ->
            case KM.lookup comboDropTombstonesKey o of
                Just (Aeson.Array arr) -> V.foldl' add M.empty arr
                _ -> M.empty
        _ -> M.empty
  where
    add acc tombstone =
        case tombstone of
            Aeson.Object o -> fromMaybe acc $ do
                keyVal <- KM.lookup comboDropTombstoneIdentityKey o
                key <- comboIdentityKeyFromJson keyVal
                droppedAt <- KM.lookup comboDropTombstoneDroppedAtKey o >>= coerceInt64Value
                pure (M.insertWith max key droppedAt acc)
            _ -> acc

comboIdentityKeyFromJson :: Aeson.Value -> Maybe BS.ByteString
comboIdentityKeyFromJson raw =
    case raw of
        Aeson.String txt
            | not (T.null txt) -> Just (TE.encodeUtf8 txt)
        _ -> Nothing

comboIdentityKeyToJson :: BS.ByteString -> Aeson.Value
comboIdentityKeyToJson = Aeson.String . TE.decodeUtf8

comboDropTombstonesValue :: M.Map BS.ByteString Int64 -> Aeson.Value
comboDropTombstonesValue tombstones =
    Aeson.Array $
        V.fromList
            [ object
                [ "identityKey" .= comboIdentityKeyToJson key
                , "droppedAtMs" .= droppedAt
                ]
            | (key, droppedAt) <- M.toList tombstones
            ]

insertComboDropTombstones :: M.Map BS.ByteString Int64 -> Aeson.Object -> Aeson.Object
insertComboDropTombstones tombstones obj =
    if M.null tombstones
        then KM.delete comboDropTombstonesKey obj
        else KM.insert comboDropTombstonesKey (comboDropTombstonesValue tombstones) obj

mergeComboDropTombstones :: [Aeson.Value] -> M.Map BS.ByteString Int64
mergeComboDropTombstones =
    foldl' (M.unionWith max) M.empty . map comboDropTombstoneMap

comboDroppedByTombstones :: M.Map BS.ByteString Int64 -> Aeson.Value -> Bool
comboDroppedByTombstones tombstones comboVal =
    any tombstoneApplies (comboDropIdentityKeys comboVal)
  where
    tombstoneApplies key =
        case M.lookup key tombstones of
            Nothing -> False
            Just droppedAt ->
                case comboBacktestFreshnessMs comboVal of
                    Nothing -> True
                    Just freshness -> droppedAt >= freshness

comboDropIdentityKey :: Aeson.Value -> Maybe BS.ByteString
comboDropIdentityKey = comboProcessingIdentityKey . comboWithoutSourceField

legacyComboDropIdentityKey :: Aeson.Value -> Maybe BS.ByteString
legacyComboDropIdentityKey = comboIdentityKey . comboWithoutSourceField

comboDropIdentityKeys :: Aeson.Value -> [BS.ByteString]
comboDropIdentityKeys comboVal =
    Set.toList $
        Set.fromList $
            Data.Maybe.mapMaybe
                ($ comboVal)
                [comboDropIdentityKey, legacyComboDropIdentityKey]

comboWithoutSourceField :: Aeson.Value -> Aeson.Value
comboWithoutSourceField val =
    case val of
        Aeson.Object o -> Aeson.Object (KM.delete (AK.fromString "source") o)
        _ -> val

updateComboWithBacktest :: Int64 -> ComboBacktestUpdate -> Aeson.Value -> Aeson.Value
updateComboWithBacktest now update comboVal =
    case comboVal of
        Aeson.Object o ->
            -- Backtests know nothing about live trading: carry the prior
            -- metrics' live record over so a refresh cannot erase it.
            let metricsWithLive =
                    case (comboLiveStats comboVal, cbuMetrics update) of
                        (Just stats, Aeson.Object m) ->
                            Aeson.Object (KM.insert (AK.fromString "live") (comboLiveStatsValue stats) m)
                        (_, m) -> m
                o1 = KM.insert (AK.fromString "metrics") metricsWithLive o
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
                -- Stamp the refresh time so identity merges can prefer the most
                -- recent backtest reading (see 'pickBestCombo'); without the
                -- stamp a deflating refresh would lose every union merge to a
                -- stale replica still carrying the old, higher score.
                o5 = KM.insert (AK.fromString "backtestRefreshedAtMs") (toJSON now) o4
             in Aeson.Object o5
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
                                Just upd -> (updateComboWithBacktest now upd comboVal : acc, count + 1)
                        combosOut = Aeson.Array (V.fromList (reverse updatedCombos))
                        o' = KM.insert (AK.fromString "combos") combosOut (KM.insert (AK.fromString "generatedAtMs") (toJSON now) o)
                    Right (Aeson.Object o', updatedCount)
                _ -> Left "Top combos JSON missing combos array."
        _ -> Left "Top combos JSON root must be an object."

data ComboBacktestApplyStats = ComboBacktestApplyStats
    { cbasUpdatedCount :: !Int
    , cbasPrunedCount :: !Int
    , cbasPrunedKeys :: ![BS.ByteString]
    }
    deriving (Eq, Show)

{- | Apply backtest updates to the top-combos JSON.

Invariant (2026-06-10): a combo is /never/ pruned when the incoming backtest
update represents a zero-trade smoke window. A zero-trade backtest is not a
verdict on the combo's profitability; it is a verdict on the smoke window.
Pruning on @finalEquity == 1.0@ produced by @tradeCount == 0@ silently
deleted 124 healthy combos in the 2026-06-10 launchd log. We detect the
zero-trade case by reading the inbound update's metrics directly.
-}
applyComboUpdatesWithStats :: Int64 -> HM.HashMap BS.ByteString ComboBacktestUpdate -> Aeson.Value -> Either String (Aeson.Value, ComboBacktestApplyStats)
applyComboUpdatesWithStats = applyComboUpdatesWithStatsPrune True

{- | Like 'applyComboUpdatesWithStats' but never prunes: an unprofitable
refresh keeps the combo with its deflated metrics (and refresh stamp).
Startup guards use this so an abort blocks the start without deleting the
selected combo. Scheduled stale refreshes should use
'applyComboUpdatesWithStats' so a sub-1.0 refreshed combo is pruned and
tombstoned against stale replica resurrection.
-}
applyComboUpdatesKeepAllWithStats :: Int64 -> HM.HashMap BS.ByteString ComboBacktestUpdate -> Aeson.Value -> Either String (Aeson.Value, ComboBacktestApplyStats)
applyComboUpdatesKeepAllWithStats = applyComboUpdatesWithStatsPrune False

applyComboUpdatesWithStatsPrune :: Bool -> Int64 -> HM.HashMap BS.ByteString ComboBacktestUpdate -> Aeson.Value -> Either String (Aeson.Value, ComboBacktestApplyStats)
applyComboUpdatesWithStatsPrune pruneUnprofitable now updates val =
    case val of
        Aeson.Object o ->
            case KM.lookup (AK.fromString "combos") o of
                Just (Aeson.Array combos) -> do
                    let priorDropTombstones = comboDropTombstoneMap val
                        combosList = V.toList combos
                        (updatedCombos, updatedCount, prunedCount, prunedKeys, tombstoneKeys) = foldl' applyOne ([], 0, 0, [], []) combosList
                        applyOne (acc, updCount, pruneCount, pKeys, tKeys) comboVal =
                            case comboIdentityKey comboVal >>= (`HM.lookup` updates) of
                                Nothing -> (comboVal : acc, updCount, pruneCount, pKeys, tKeys)
                                Just upd ->
                                    let updated = updateComboWithBacktest now upd comboVal
                                        mEquity = comboFinalEquityValue updated
                                        mTrades = comboMetricInt "tradeCount" (cbuMetrics upd)
                                        -- A zero-trade update is not a verdict; keep the combo as-is.
                                        zeroTradeUpdate = mTrades == Just 0
                                        keep = not pruneUnprofitable || maybe True (> 1.0) mEquity || zeroTradeUpdate
                                     in if keep
                                            then (updated : acc, updCount + 1, pruneCount, pKeys, tKeys)
                                            else case comboIdentityKey comboVal of
                                                Nothing -> (acc, updCount + 1, pruneCount + 1, pKeys, tKeys)
                                                Just k ->
                                                    let tKeys' = comboDropIdentityKeys comboVal ++ tKeys
                                                     in (acc, updCount + 1, pruneCount + 1, k : pKeys, tKeys')
                        combosOut = Aeson.Array (V.fromList (reverse updatedCombos))
                        dropTombstones' = foldl' (\acc key -> M.insertWith max key now acc) priorDropTombstones tombstoneKeys
                        o' =
                            insertComboDropTombstones dropTombstones' $
                                KM.insert (AK.fromString "combos") combosOut (KM.insert (AK.fromString "generatedAtMs") (toJSON now) o)
                        stats = ComboBacktestApplyStats updatedCount prunedCount (reverse prunedKeys)
                    Right (Aeson.Object o', stats)
                _ -> Left "Top combos JSON missing combos array."
        _ -> Left "Top combos JSON root must be an object."
