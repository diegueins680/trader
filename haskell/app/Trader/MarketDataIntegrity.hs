module Trader.MarketDataIntegrity (
    MarketDataFreshness (..),
    intervalMsFromCode,
    marketDataFreshness,
    marketDataStaleReason,
    marketDataContinuationIssue,
) where

import Data.Int (Int64)
import Text.Printf (printf)
import Trader.Duration (parseIntervalSeconds)

data MarketDataFreshness = MarketDataFreshness
    { mdfLastOpenTimeMs :: !Int64
    , mdfLastCloseTimeMs :: !Int64
    , mdfAgeMs :: !Int64
    , mdfFreshnessBudgetMs :: !Int64
    , mdfStale :: !Bool
    }
    deriving (Eq, Show)

intervalMsFromCode :: String -> Maybe Int64
intervalMsFromCode interval =
    case parseIntervalSeconds interval of
        Just sec | sec > 0 -> Just (fromIntegral sec * 1000)
        _ -> Nothing

marketDataFreshness :: String -> Int64 -> Int64 -> Maybe MarketDataFreshness
marketDataFreshness interval nowMs lastOpenTimeMs = do
    intervalMs <- intervalMsFromCode interval
    let lastCloseTimeMs = lastOpenTimeMs + intervalMs
        ageMs = max 0 (nowMs - lastCloseTimeMs)
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
        Nothing -> Just ("MARKET_DATA_INTERVAL_INVALID interval=" ++ show interval)
        Just freshness ->
            if mdfStale freshness
                then
                    Just
                        ( printf
                            "STALE_MARKET_DATA ageMs=%d budgetMs=%d lastCloseTimeMs=%d"
                            (mdfAgeMs freshness)
                            (mdfFreshnessBudgetMs freshness)
                            (mdfLastCloseTimeMs freshness)
                        )
                else Nothing

marketDataContinuationIssue :: String -> Int64 -> [Int64] -> Maybe String
marketDataContinuationIssue interval lastSeenOpenTimeMs newOpenTimes =
    case newOpenTimes of
        [] -> Nothing
        _ ->
            case intervalMsFromCode interval of
                Nothing -> Just ("MARKET_DATA_INTERVAL_INVALID interval=" ++ show interval)
                Just intervalMs ->
                    let expectedFirst = lastSeenOpenTimeMs + intervalMs
                        firstOpen = head newOpenTimes
                        pairs = zip newOpenTimes (drop 1 newOpenTimes)
                        mBadStep =
                            case [actual - prev | (prev, actual) <- pairs, actual - prev /= intervalMs] of
                                bad : _ -> Just bad
                                [] -> Nothing
                     in if firstOpen /= expectedFirst
                            then
                                Just
                                    ( printf
                                        "MARKET_DATA_GAP expectedOpenTimeMs=%d actualOpenTimeMs=%d intervalMs=%d"
                                        expectedFirst
                                        firstOpen
                                        intervalMs
                                    )
                            else case mBadStep of
                                Just actualStep ->
                                    Just
                                        ( printf
                                            "MARKET_DATA_GAP actualStepMs=%d intervalMs=%d"
                                            actualStep
                                            intervalMs
                                        )
                                Nothing -> Nothing
