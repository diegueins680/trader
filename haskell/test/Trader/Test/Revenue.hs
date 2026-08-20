{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.Revenue (
    revenueSuite,
) where

import Data.Aeson (eitherDecode)
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Int (Int64)
import Data.Maybe (fromMaybe)
import Trader.Binance (BinanceTrade (..), FuturesIncome (..), FuturesPositionRisk (..))
import Trader.Revenue (RevenueBreakdown (..), RevenueExecution (..), RevenueLedger (..), RevenueSymbol (..), buildRevenueLedger)

revenueSuite :: [(String, IO ())]
revenueSuite =
    [ ("exchange income is authoritative and excludes account transfers", testExchangeIncomeAuthority)
    , ("daily and symbol revenue remain attributable", testRevenueAttribution)
    , ("unclassified income fails closed outside net revenue", testUnclassifiedIncomeFailsClosed)
    , ("Binance futures income decodes string amounts", testFuturesIncomeDecode)
    ]

testExchangeIncomeAuthority :: IO ()
testExchangeIncomeAuthority = do
    let startAt = 1704067200000
        endAt = startAt + 86400000 - 1
        incomes =
            [ income "BTCUSDT" "REALIZED_PNL" 100 startAt
            , income "BTCUSDT" "COMMISSION" (-4) (startAt + 1)
            , income "BTCUSDT" "FUNDING_FEE" (-2) (startAt + 2)
            , income "BTCUSDT" "COMMISSION_REBATE" 1 (startAt + 3)
            , income "" "TRANSFER" 500 (startAt + 4)
            ]
        -- Trade-level realizedPnl is deliberately huge. The ledger must not
        -- add it again because futures income is the accounting authority.
        trades = [trade "BTCUSDT" startAt True 1000 9999]
        positions = [position "BTCUSDT" 1 5]
        ledger = buildRevenueLedger "USDT" startAt endAt False False 3 incomes positions trades
        breakdown = rlBreakdown ledger
    expectNear "realized PnL" 100 (rbRealizedPnl breakdown)
    expectNear "commission" (-4) (rbCommission breakdown)
    expectNear "funding" (-2) (rbFunding breakdown)
    expectNear "rebates" 1 (rbRebates breakdown)
    expectNear "exchange net" 95 (rbExchangeNet breakdown)
    expectNear "excluded transfer" 500 (rbExcludedNonOperating breakdown)
    expectNear "unrealized PnL" 5 (rlUnrealizedPnl ledger)
    expectNear "net revenue" 97 (rlNetRevenue ledger)
    expectNear "maker rate" 1 (fromMaybe (-1) (reMakerRate (rlExecution ledger)))

testRevenueAttribution :: IO ()
testRevenueAttribution = do
    let day = 86400000
        startAt = 1704067200000
        endAt = startAt + 2 * day - 1
        incomes =
            [ income "BTCUSDT" "REALIZED_PNL" 10 startAt
            , income "ETHUSDT" "FUNDING_FEE" 2 (startAt + day)
            , FuturesIncome "BTCUSDT" "REALIZED_PNL" 900 "USDC" startAt
            ]
        trades =
            [ trade "BTCUSDT" startAt True 100 10
            , trade "ETHUSDT" (startAt + day) False 50 2
            ]
        ledger = buildRevenueLedger "USDT" startAt endAt False False 0 incomes [] trades
        btc = findSymbol "BTCUSDT" (rlSymbols ledger)
        eth = findSymbol "ETHUSDT" (rlSymbols ledger)
    expectEq "two UTC daily buckets" 2 (length (rlDaily ledger))
    expectNear "USDC income excluded from USDT ledger" 12 (rbExchangeNet (rlBreakdown ledger))
    expectNear "BTC symbol revenue" 10 (rbExchangeNet (rsBreakdown btc))
    expectNear "ETH symbol revenue" 2 (rbExchangeNet (rsBreakdown eth))
    expectEq "one maker fill" 1 (reMakerTrades (rlExecution ledger))
    expectEq "one taker fill" 1 (reTakerTrades (rlExecution ledger))

testUnclassifiedIncomeFailsClosed :: IO ()
testUnclassifiedIncomeFailsClosed = do
    let startAt = 1704067200000
        ledger =
            buildRevenueLedger
                "USDT"
                startAt
                (startAt + 1000)
                True
                False
                (-5)
                [income "BTCUSDT" "NEW_UNKNOWN_REWARD" 50 startAt]
                []
                []
    expectNear "unknown income excluded from exchange net" 0 (rbExchangeNet (rlBreakdown ledger))
    expectNear "unknown income disclosed" 50 (rbUnclassified (rlBreakdown ledger))
    expectEq "unknown type listed" ["NEW_UNKNOWN_REWARD"] (rlUnclassifiedIncomeTypes ledger)
    expectNear "negative infrastructure cost cannot increase revenue" 0 (rlInfrastructureCost ledger)
    expectEq "income limit warning" True (rlIncomeMayBeTruncated ledger)

testFuturesIncomeDecode :: IO ()
testFuturesIncomeDecode = do
    let payload = "{\"symbol\":\"btcusdt\",\"incomeType\":\"funding_fee\",\"income\":\"-0.375\",\"asset\":\"usdt\",\"time\":1570608000000}"
    case eitherDecode (BL8.pack payload) of
        Left err -> fail ("income decode failed: " ++ err)
        Right row -> do
            expectEq "symbol normalized" "BTCUSDT" (fiSymbol row)
            expectEq "income type normalized" "FUNDING_FEE" (fiIncomeType row)
            expectEq "asset normalized" "USDT" (fiAsset row)
            expectNear "income amount" (-0.375) (fiIncome row)

income :: String -> String -> Double -> Int64 -> FuturesIncome
income symbol incomeType amount =
    FuturesIncome symbol incomeType amount "USDT"

trade :: String -> Int64 -> Bool -> Double -> Double -> BinanceTrade
trade symbol time maker quoteQty realizedPnl =
    BinanceTrade
        { btSymbol = symbol
        , btTradeId = 1
        , btOrderId = Nothing
        , btPrice = 1
        , btQty = quoteQty
        , btQuoteQty = quoteQty
        , btCommission = Nothing
        , btCommissionAsset = Just "USDT"
        , btTime = time
        , btIsBuyer = Just True
        , btIsMaker = Just maker
        , btSide = Just "BUY"
        , btPositionSide = Just "BOTH"
        , btRealizedPnl = Just realizedPnl
        , btOriginIp = Nothing
        , btExecutorIp = Nothing
        , btOriginInstance = Nothing
        , btEntryIp = Nothing
        , btExitIp = Nothing
        , btEntryInstance = Nothing
        , btExitInstance = Nothing
        , btEntryTime = Nothing
        , btExitTime = Nothing
        , btMaxPnl = Nothing
        , btMaxPnlCloseTime = Nothing
        , btMethod = Nothing
        , btStrategy = Nothing
        , btDecisionSummary = Nothing
        , btDecisionReason = Nothing
        }

position :: String -> Double -> Double -> FuturesPositionRisk
position symbol amount unrealized =
    FuturesPositionRisk
        { fprSymbol = symbol
        , fprPositionAmt = amount
        , fprEntryPrice = 1
        , fprMarkPrice = 1
        , fprUnrealizedProfit = unrealized
        , fprLiquidationPrice = Nothing
        , fprBreakEvenPrice = Nothing
        , fprLeverage = 1
        , fprMarginType = Just "cross"
        , fprPositionSide = Just "BOTH"
        }

findSymbol :: String -> [RevenueSymbol] -> RevenueSymbol
findSymbol symbol rows =
    case filter ((== symbol) . rsSymbol) rows of
        row : _ -> row
        [] -> error ("missing revenue symbol " ++ symbol)

expectNear :: String -> Double -> Double -> IO ()
expectNear label expected actual =
    if abs (expected - actual) < 1.0e-9
        then pure ()
        else fail (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)

expectEq :: (Eq value, Show value) => String -> value -> value -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else fail (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)
