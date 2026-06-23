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
    let lastCloseTimeMs = lastOpenTimeMs + intervalMs
        ageMs = nowMs - lastCloseTimeMs
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
    case marketDataFreshness interval nowMs lastOpenTimeMs of
        Nothing -> Just (invalidIntervalReason interval)
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
        Just intervalMs -> firstContinuationIssue intervalMs (lastOpenTimeMs + intervalMs) openTimeMs

normalizeClosedMarketSeries :: String -> Int64 -> Int64 -> [(MarketSeriesBar, a)] -> Either String [(MarketSeriesBar, a)]
normalizeClosedMarketSeries label intervalMs nowMs barsWithPayload = do
    let sorted = sortOn (msbOpenTimeMs . fst) barsWithPayload
        bars = map fst sorted
    validateMarketSeriesBars label bars
    let closed = filter (\(bar, _) -> msbOpenTimeMs bar + intervalMs <= nowMs) sorted
    validateMarketSeriesContinuity label intervalMs (map fst closed)
    pure closed

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
    | otherwise =
        case [ (msbOpenTimeMs a + intervalMs, msbOpenTimeMs b)
             | (a, b) <- zip bars (drop 1 bars)
             , msbOpenTimeMs b /= msbOpenTimeMs a + intervalMs
             ] of
            (expected, actual) : _ ->
                Left
                    ( label
                        ++ " gap expectedOpenTimeMs="
                        ++ show expected
                        ++ " actualOpenTimeMs="
                        ++ show actual
                        ++ " intervalMs="
                        ++ show intervalMs
                    )
            [] -> Right ()

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
        firstContinuationIssue intervalMs (expectedOpenTimeMs + intervalMs) rest
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
    (* 1000) . fromIntegral <$> parseIntervalSeconds interval

invalidIntervalReason :: String -> String
invalidIntervalReason interval =
    "MARKET_DATA_INTERVAL_INVALID interval=" ++ show interval

{- | Returns True for transient market-data issues that should not block
queued bot starts (the bot is in a safe HOLD state and the condition
is self-healing).
-}
isTransientMarketDataError :: String -> Bool
isTransientMarketDataError err =
    any (`isPrefixOf` err) ["MARKET_DATA_GAP", "STALE_MARKET_DATA"]
