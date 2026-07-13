module Trader.MarketDataIntegrity (
    MarketDataFreshness (..),
    MarketSeriesBar (..),
    marketDataFreshness,
    marketDataStaleReason,
    marketDataContinuationIssue,
    normalizeClosedMarketSeries,
    validateMarketSeriesBars,
    validateMarketSeriesContinuity,
    isTransientMarketDataError,
) where

import Data.Foldable (traverse_)
import Data.Int (Int64)
import Data.List (isPrefixOf, sortOn)
import Data.Maybe (catMaybes)
import Trader.Duration (parseIntervalSeconds)

data MarketDataFreshness = MarketDataFreshness
    { mdfLastOpenTimeMs :: !Int64
    , mdfLastCloseTimeMs :: !Int64
    , mdfAgeMs :: !Int64
    , mdfFreshnessBudgetMs :: !Int64
    , mdfStale :: !Bool
    }
    deriving (Eq, Show)

data MarketSeriesBar = MarketSeriesBar
    { msbOpenTimeMs :: !Int64
    , msbOpen :: !(Maybe Double)
    , msbHigh :: !(Maybe Double)
    , msbLow :: !(Maybe Double)
    , msbClose :: !Double
    , msbVolume :: !(Maybe Double)
    }
    deriving (Eq, Show)

marketDataFreshness :: String -> Int64 -> Int64 -> Maybe MarketDataFreshness
marketDataFreshness interval nowMs lastOpenTimeMs = do
    intervalMs <- intervalMsFrom interval
    lastCloseTimeMs <- checkedAddInt64 lastOpenTimeMs intervalMs
    ageMs <- checkedSubtractInt64 nowMs lastCloseTimeMs
    pure
        MarketDataFreshness
            { mdfLastOpenTimeMs = lastOpenTimeMs
            , mdfLastCloseTimeMs = lastCloseTimeMs
            , mdfAgeMs = ageMs
            , mdfFreshnessBudgetMs = intervalMs
            , mdfStale = ageMs > intervalMs
            }

marketDataStaleReason :: String -> Int64 -> Int64 -> Maybe String
marketDataStaleReason interval nowMs lastOpenTimeMs =
    case intervalMsFrom interval of
        Nothing -> Just (invalidIntervalReason interval)
        Just intervalMs ->
            case marketDataFreshness interval nowMs lastOpenTimeMs of
                Nothing -> Just (timestampOverflowReason nowMs lastOpenTimeMs intervalMs)
                Just freshness
                    | mdfStale freshness ->
                        Just
                            ( "STALE_MARKET_DATA ageMs="
                                ++ show (mdfAgeMs freshness)
                                ++ " budgetMs="
                                ++ show (mdfFreshnessBudgetMs freshness)
                                ++ " lastCloseTimeMs="
                                ++ show (mdfLastCloseTimeMs freshness)
                            )
                    | otherwise -> Nothing

marketDataContinuationIssue :: String -> Int64 -> [Int64] -> Maybe String
marketDataContinuationIssue interval lastOpenTimeMs openTimeMs =
    case intervalMsFrom interval of
        Nothing -> Just (invalidIntervalReason interval)
        Just _ | null openTimeMs -> Nothing
        Just intervalMs ->
            case checkedAddInt64 lastOpenTimeMs intervalMs of
                Nothing -> Just (continuationOverflowReason lastOpenTimeMs intervalMs)
                Just expectedOpenTimeMs -> firstContinuationIssue intervalMs expectedOpenTimeMs openTimeMs

normalizeClosedMarketSeries :: String -> Int64 -> Int64 -> [(MarketSeriesBar, a)] -> Either String [(MarketSeriesBar, a)]
normalizeClosedMarketSeries label intervalMs nowMs barsWithPayload
    | intervalMs <= 0 = Left (label ++ " invalid intervalMs=" ++ show intervalMs)
    | otherwise = do
        let sorted = sortOn (msbOpenTimeMs . fst) barsWithPayload
            bars = map fst sorted
        validateMarketSeriesBars label bars
        closes <- traverse closeWitness sorted
        let closed = [barWithPayload | (barWithPayload, closeTimeMs) <- zip sorted closes, closeTimeMs <= nowMs]
        validateMarketSeriesContinuity label intervalMs (map fst closed)
        pure closed
  where
    closeWitness (bar, _) =
        case checkedAddInt64 (msbOpenTimeMs bar) intervalMs of
            Nothing -> Left (label ++ " timestamp overflow openTimeMs=" ++ show (msbOpenTimeMs bar) ++ " intervalMs=" ++ show intervalMs)
            Just closeTimeMs -> Right closeTimeMs

validateMarketSeriesBars :: String -> [MarketSeriesBar] -> Either String ()
validateMarketSeriesBars label bars = do
    traverse_ (validateMarketSeriesBar label) bars
    validateStrictMarketSeriesTimes label bars

validateMarketSeriesBar :: String -> MarketSeriesBar -> Either String ()
validateMarketSeriesBar label bar
    | not (all finiteDouble numericValues) =
        Left (label ++ " invalid numeric payload at openTimeMs=" ++ show (msbOpenTimeMs bar))
    | maybe False (< 0) (msbVolume bar) =
        Left (label ++ " negative volume at openTimeMs=" ++ show (msbOpenTimeMs bar))
    | not (marketSeriesOhlcOk bar) =
        Left (label ++ " invalid OHLC relationship at openTimeMs=" ++ show (msbOpenTimeMs bar))
    | otherwise = Right ()
  where
    numericValues =
        catMaybes
            [ msbOpen bar
            , msbHigh bar
            , msbLow bar
            , Just (msbClose bar)
            , msbVolume bar
            ]

validateStrictMarketSeriesTimes :: String -> [MarketSeriesBar] -> Either String ()
validateStrictMarketSeriesTimes label bars =
    case [msbOpenTimeMs b | (a, b) <- zip bars (drop 1 bars), msbOpenTimeMs a >= msbOpenTimeMs b] of
        bad : _ -> Left (label ++ " duplicate/non-increasing openTimeMs=" ++ show bad)
        [] -> Right ()

validateMarketSeriesContinuity :: String -> Int64 -> [MarketSeriesBar] -> Either String ()
validateMarketSeriesContinuity label intervalMs bars
    | intervalMs <= 0 = Left (label ++ " invalid intervalMs=" ++ show intervalMs)
    | otherwise = traverse_ validatePair (zip bars (drop 1 bars))
  where
    validatePair (a, b) =
        case checkedAddInt64 (msbOpenTimeMs a) intervalMs of
            Nothing ->
                Left
                    ( label
                        ++ " timestamp overflow openTimeMs="
                        ++ show (msbOpenTimeMs a)
                        ++ " intervalMs="
                        ++ show intervalMs
                    )
            Just expected
                | msbOpenTimeMs b == expected -> Right ()
                | otherwise ->
                    Left
                        ( label
                            ++ " gap expectedOpenTimeMs="
                            ++ show expected
                            ++ " actualOpenTimeMs="
                            ++ show (msbOpenTimeMs b)
                            ++ " intervalMs="
                            ++ show intervalMs
                        )

marketSeriesOhlcOk :: MarketSeriesBar -> Bool
marketSeriesOhlcOk bar =
    case (msbHigh bar, msbLow bar) of
        (Just high, Just low) ->
            high >= low
                && high >= msbClose bar
                && low <= msbClose bar
                && maybe True (\open -> high >= open && low <= open) (msbOpen bar)
        _ -> True

finiteDouble :: Double -> Bool
finiteDouble x = not (isNaN x || isInfinite x)

firstContinuationIssue :: Int64 -> Int64 -> [Int64] -> Maybe String
firstContinuationIssue _ _ [] = Nothing
firstContinuationIssue intervalMs expectedOpenTimeMs (actualOpenTimeMs : rest)
    | actualOpenTimeMs == expectedOpenTimeMs =
        case rest of
            [] -> Nothing
            _ ->
                case checkedAddInt64 expectedOpenTimeMs intervalMs of
                    Nothing -> Just (continuationOverflowReason expectedOpenTimeMs intervalMs)
                    Just nextExpectedOpenTimeMs -> firstContinuationIssue intervalMs nextExpectedOpenTimeMs rest
    | otherwise =
        Just
            ( "MARKET_DATA_GAP expectedOpenTimeMs="
                ++ show expectedOpenTimeMs
                ++ " actualOpenTimeMs="
                ++ show actualOpenTimeMs
                ++ " intervalMs="
                ++ show intervalMs
            )

intervalMsFrom :: String -> Maybe Int64
intervalMsFrom interval =
    parseIntervalSeconds interval >>= \seconds ->
        if seconds <= 0
            then Nothing
            else integerToInt64 (toInteger seconds * 1000)

checkedAddInt64 :: Int64 -> Int64 -> Maybe Int64
checkedAddInt64 a b = integerToInt64 (toInteger a + toInteger b)

checkedSubtractInt64 :: Int64 -> Int64 -> Maybe Int64
checkedSubtractInt64 a b = integerToInt64 (toInteger a - toInteger b)

integerToInt64 :: Integer -> Maybe Int64
integerToInt64 value
    | value < toInteger (minBound :: Int64) = Nothing
    | value > toInteger (maxBound :: Int64) = Nothing
    | otherwise = Just (fromInteger value)

invalidIntervalReason :: String -> String
invalidIntervalReason interval =
    "MARKET_DATA_INTERVAL_INVALID interval=" ++ show interval

timestampOverflowReason :: Int64 -> Int64 -> Int64 -> String
timestampOverflowReason nowMs lastOpenTimeMs intervalMs =
    "MARKET_DATA_TIMESTAMP_OVERFLOW nowMs="
        ++ show nowMs
        ++ " lastOpenTimeMs="
        ++ show lastOpenTimeMs
        ++ " intervalMs="
        ++ show intervalMs

continuationOverflowReason :: Int64 -> Int64 -> String
continuationOverflowReason openTimeMs intervalMs =
    "MARKET_DATA_TIMESTAMP_OVERFLOW openTimeMs="
        ++ show openTimeMs
        ++ " intervalMs="
        ++ show intervalMs

{- | Returns True for transient market-data issues that should not block
queued bot starts (the bot is in a safe HOLD state and the condition
is self-healing).
-}
isTransientMarketDataError :: String -> Bool
isTransientMarketDataError err =
    any (`isPrefixOf` err) ["MARKET_DATA_GAP", "STALE_MARKET_DATA"]
