module Trader.MarketDataIntegrity (
    MarketDataFreshness (..),
    marketDataFreshness,
    marketDataStaleReason,
    marketDataContinuationIssue,
) where

import Trader.Duration (parseIntervalSeconds)

data MarketDataFreshness = MarketDataFreshness
    { mdfLastCloseTimeMs :: !Int
    , mdfAgeMs :: !Int
    , mdfFreshnessBudgetMs :: !Int
    , mdfStale :: !Bool
    }
    deriving (Eq, Show)

marketDataFreshness :: String -> Int -> Int -> Maybe MarketDataFreshness
marketDataFreshness interval nowMs lastOpenTimeMs = do
    intervalMs <- intervalMsFrom interval
    let lastCloseTimeMs = lastOpenTimeMs + intervalMs
        ageMs = nowMs - lastCloseTimeMs
    pure
        MarketDataFreshness
            { mdfLastCloseTimeMs = lastCloseTimeMs
            , mdfAgeMs = ageMs
            , mdfFreshnessBudgetMs = intervalMs
            , mdfStale = ageMs > intervalMs
            }

marketDataStaleReason :: String -> Int -> Int -> Maybe String
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

marketDataContinuationIssue :: String -> Int -> [Int] -> Maybe String
marketDataContinuationIssue interval lastOpenTimeMs openTimeMs =
    case intervalMsFrom interval of
        Nothing -> Just (invalidIntervalReason interval)
        Just intervalMs -> firstContinuationIssue intervalMs (lastOpenTimeMs + intervalMs) openTimeMs

firstContinuationIssue :: Int -> Int -> [Int] -> Maybe String
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

intervalMsFrom :: String -> Maybe Int
intervalMsFrom interval =
    (* 1000) <$> parseIntervalSeconds interval

invalidIntervalReason :: String -> String
invalidIntervalReason interval =
    "MARKET_DATA_INTERVAL_INVALID interval=" ++ show interval
