{-# LANGUAGE OverloadedStrings #-}

module Trader.Revenue (
    RevenueBreakdown (..),
    RevenueBucket (..),
    RevenueExecution (..),
    RevenueLedger (..),
    RevenueSymbol (..),
    buildRevenueLedger,
) where

import Data.Aeson (ToJSON (..), object, (.=))
import Data.Int (Int64)
import Data.List (foldl')
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import Trader.Binance (BinanceTrade (..), FuturesIncome (..), FuturesPositionRisk (..))

data RevenueBreakdown = RevenueBreakdown
    { rbRealizedPnl :: !Double
    , rbFunding :: !Double
    , rbCommission :: !Double
    , rbRebates :: !Double
    , rbOtherOperating :: !Double
    , rbExchangeNet :: !Double
    , rbExcludedNonOperating :: !Double
    , rbUnclassified :: !Double
    }
    deriving (Eq, Show)

instance ToJSON RevenueBreakdown where
    toJSON value =
        object
            [ "realizedPnl" .= rbRealizedPnl value
            , "funding" .= rbFunding value
            , "commission" .= rbCommission value
            , "rebates" .= rbRebates value
            , "otherOperating" .= rbOtherOperating value
            , "exchangeNet" .= rbExchangeNet value
            , "excludedNonOperating" .= rbExcludedNonOperating value
            , "unclassified" .= rbUnclassified value
            ]

data RevenueExecution = RevenueExecution
    { reTrades :: !Int
    , reMakerTrades :: !Int
    , reTakerTrades :: !Int
    , reUnknownLiquidityTrades :: !Int
    , reMakerRate :: !(Maybe Double)
    , reQuoteNotional :: !Double
    }
    deriving (Eq, Show)

instance ToJSON RevenueExecution where
    toJSON value =
        object
            [ "trades" .= reTrades value
            , "makerTrades" .= reMakerTrades value
            , "takerTrades" .= reTakerTrades value
            , "unknownLiquidityTrades" .= reUnknownLiquidityTrades value
            , "makerRate" .= reMakerRate value
            , "quoteNotional" .= reQuoteNotional value
            ]

data RevenueBucket = RevenueBucket
    { rbuStartAtMs :: !Int64
    , rbuBreakdown :: !RevenueBreakdown
    }
    deriving (Eq, Show)

instance ToJSON RevenueBucket where
    toJSON value =
        object
            [ "startAtMs" .= rbuStartAtMs value
            , "breakdown" .= rbuBreakdown value
            ]

data RevenueSymbol = RevenueSymbol
    { rsSymbol :: !String
    , rsBreakdown :: !RevenueBreakdown
    , rsExecution :: !RevenueExecution
    }
    deriving (Eq, Show)

instance ToJSON RevenueSymbol where
    toJSON value =
        object
            [ "symbol" .= rsSymbol value
            , "breakdown" .= rsBreakdown value
            , "execution" .= rsExecution value
            ]

data RevenueLedger = RevenueLedger
    { rlAsset :: !String
    , rlStartAtMs :: !Int64
    , rlEndAtMs :: !Int64
    , rlIncomeRecords :: !Int
    , rlTradeRecords :: !Int
    , rlIncomeMayBeTruncated :: !Bool
    , rlTradesMayBeTruncated :: !Bool
    , rlBreakdown :: !RevenueBreakdown
    , rlUnrealizedPnl :: !Double
    , rlInfrastructureCost :: !Double
    , rlNetRevenue :: !Double
    , rlExecution :: !RevenueExecution
    , rlDaily :: ![RevenueBucket]
    , rlSymbols :: ![RevenueSymbol]
    , rlUnclassifiedIncomeTypes :: ![String]
    }
    deriving (Eq, Show)

instance ToJSON RevenueLedger where
    toJSON value =
        object
            [ "asset" .= rlAsset value
            , "startAtMs" .= rlStartAtMs value
            , "endAtMs" .= rlEndAtMs value
            , "incomeRecords" .= rlIncomeRecords value
            , "tradeRecords" .= rlTradeRecords value
            , "incomeMayBeTruncated" .= rlIncomeMayBeTruncated value
            , "tradesMayBeTruncated" .= rlTradesMayBeTruncated value
            , "breakdown" .= rlBreakdown value
            , "unrealizedPnl" .= rlUnrealizedPnl value
            , "infrastructureCost" .= rlInfrastructureCost value
            , "netRevenue" .= rlNetRevenue value
            , "execution" .= rlExecution value
            , "daily" .= rlDaily value
            , "symbols" .= rlSymbols value
            , "unclassifiedIncomeTypes" .= rlUnclassifiedIncomeTypes value
            ]

data IncomeClass
    = IncomeRealized
    | IncomeFunding
    | IncomeCommission
    | IncomeRebate
    | IncomeOtherOperating
    | IncomeNonOperating
    | IncomeUnclassified
    deriving (Eq, Show)

data RevenueAccumulator = RevenueAccumulator
    { raRealizedPnl :: !Double
    , raFunding :: !Double
    , raCommission :: !Double
    , raRebates :: !Double
    , raOtherOperating :: !Double
    , raExcludedNonOperating :: !Double
    , raUnclassified :: !Double
    }

emptyAccumulator :: RevenueAccumulator
emptyAccumulator = RevenueAccumulator 0 0 0 0 0 0 0

classifyIncome :: String -> IncomeClass
classifyIncome incomeType
    | incomeType == "REALIZED_PNL" = IncomeRealized
    | incomeType == "FUNDING_FEE" = IncomeFunding
    | incomeType == "COMMISSION" = IncomeCommission
    | incomeType `S.member` rebateIncomeTypes = IncomeRebate
    | incomeType `S.member` otherOperatingIncomeTypes = IncomeOtherOperating
    | incomeType `S.member` nonOperatingIncomeTypes = IncomeNonOperating
    | otherwise = IncomeUnclassified

rebateIncomeTypes :: S.Set String
rebateIncomeTypes =
    S.fromList
        [ "REFERRAL_KICKBACK"
        , "COMMISSION_REBATE"
        , "API_REBATE"
        , "FEE_RETURN"
        ]

otherOperatingIncomeTypes :: S.Set String
otherOperatingIncomeTypes =
    S.fromList
        [ "INSURANCE_CLEAR"
        , "OPTIONS_PREMIUM_FEE"
        , "OPTIONS_SETTLE_PROFIT"
        , "DELIVERED_SETTLEMENT"
        , "DELIVERED_SETTELMENT"
        , "POSITION_LIMIT_INCREASE_FEE"
        ]

nonOperatingIncomeTypes :: S.Set String
nonOperatingIncomeTypes =
    S.fromList
        [ "TRANSFER"
        , "WELCOME_BONUS"
        , "CONTEST_REWARD"
        , "CROSS_COLLATERAL_TRANSFER"
        , "INTERNAL_TRANSFER"
        , "AUTO_EXCHANGE"
        , "COIN_SWAP_DEPOSIT"
        , "COIN_SWAP_WITHDRAW"
        ]

accumulateIncome :: RevenueAccumulator -> FuturesIncome -> RevenueAccumulator
accumulateIncome acc row =
    let amount = finiteOrZero (fiIncome row)
     in case classifyIncome (fiIncomeType row) of
            IncomeRealized -> acc{raRealizedPnl = raRealizedPnl acc + amount}
            IncomeFunding -> acc{raFunding = raFunding acc + amount}
            IncomeCommission -> acc{raCommission = raCommission acc + amount}
            IncomeRebate -> acc{raRebates = raRebates acc + amount}
            IncomeOtherOperating -> acc{raOtherOperating = raOtherOperating acc + amount}
            IncomeNonOperating -> acc{raExcludedNonOperating = raExcludedNonOperating acc + amount}
            IncomeUnclassified -> acc{raUnclassified = raUnclassified acc + amount}

accumulatorBreakdown :: RevenueAccumulator -> RevenueBreakdown
accumulatorBreakdown acc =
    let exchangeNet =
            raRealizedPnl acc
                + raFunding acc
                + raCommission acc
                + raRebates acc
                + raOtherOperating acc
     in RevenueBreakdown
            { rbRealizedPnl = raRealizedPnl acc
            , rbFunding = raFunding acc
            , rbCommission = raCommission acc
            , rbRebates = raRebates acc
            , rbOtherOperating = raOtherOperating acc
            , rbExchangeNet = exchangeNet
            , rbExcludedNonOperating = raExcludedNonOperating acc
            , rbUnclassified = raUnclassified acc
            }

summarizeIncome :: [FuturesIncome] -> RevenueBreakdown
summarizeIncome = accumulatorBreakdown . foldl' accumulateIncome emptyAccumulator

summarizeExecution :: [BinanceTrade] -> RevenueExecution
summarizeExecution trades =
    let makerTrades = length [() | row <- trades, btIsMaker row == Just True]
        takerTrades = length [() | row <- trades, btIsMaker row == Just False]
        knownTrades = makerTrades + takerTrades
        makerRate =
            if knownTrades <= 0
                then Nothing
                else Just (fromIntegral makerTrades / fromIntegral knownTrades)
        quoteNotional = sum [max 0 (finiteOrZero (btQuoteQty row)) | row <- trades]
     in RevenueExecution
            { reTrades = length trades
            , reMakerTrades = makerTrades
            , reTakerTrades = takerTrades
            , reUnknownLiquidityTrades = length trades - knownTrades
            , reMakerRate = makerRate
            , reQuoteNotional = quoteNotional
            }

buildRevenueLedger ::
    String ->
    Int64 ->
    Int64 ->
    Bool ->
    Bool ->
    Double ->
    [FuturesIncome] ->
    [FuturesPositionRisk] ->
    [BinanceTrade] ->
    RevenueLedger
buildRevenueLedger asset startAtMs endAtMs incomeMayBeTruncated tradesMayBeTruncated infrastructureCost incomes positions trades =
    let relevantIncomes =
            [ row
            | row <- incomes
            , fiAsset row == asset
            , fiTime row >= startAtMs
            , fiTime row <= endAtMs
            ]
        relevantTrades =
            [ row
            | row <- trades
            , btTime row >= startAtMs
            , btTime row <= endAtMs
            ]
        breakdown = summarizeIncome relevantIncomes
        unrealizedPnl =
            sum
                [ finiteOrZero (fprUnrealizedProfit row)
                | row <- positions
                , asset `suffixOf` fprSymbol row
                , abs (finiteOrZero (fprPositionAmt row)) > 1.0e-12
                ]
        infrastructureCostSafe = max 0 (finiteOrZero infrastructureCost)
        netRevenue = rbExchangeNet breakdown + unrealizedPnl - infrastructureCostSafe
        daily =
            [ RevenueBucket dayStart (summarizeIncome rows)
            | (dayStart, rows) <- M.toAscList (groupOn incomeDayStart relevantIncomes)
            ]
        symbols =
            [ RevenueSymbol symbol (summarizeIncome incomeRows) (summarizeExecution tradeRows)
            | symbol <- S.toAscList (S.fromList (map fiSymbol relevantIncomes ++ map btSymbol relevantTrades))
            , let incomeRows = filter ((== symbol) . fiSymbol) relevantIncomes
            , let tradeRows = filter ((== symbol) . btSymbol) relevantTrades
            , not (null incomeRows && null tradeRows)
            ]
        unclassifiedTypes =
            S.toAscList
                ( S.fromList
                    [ fiIncomeType row
                    | row <- relevantIncomes
                    , classifyIncome (fiIncomeType row) == IncomeUnclassified
                    ]
                )
     in RevenueLedger
            { rlAsset = asset
            , rlStartAtMs = startAtMs
            , rlEndAtMs = endAtMs
            , rlIncomeRecords = length relevantIncomes
            , rlTradeRecords = length relevantTrades
            , rlIncomeMayBeTruncated = incomeMayBeTruncated
            , rlTradesMayBeTruncated = tradesMayBeTruncated
            , rlBreakdown = breakdown
            , rlUnrealizedPnl = unrealizedPnl
            , rlInfrastructureCost = infrastructureCostSafe
            , rlNetRevenue = netRevenue
            , rlExecution = summarizeExecution relevantTrades
            , rlDaily = daily
            , rlSymbols = symbols
            , rlUnclassifiedIncomeTypes = unclassifiedTypes
            }

incomeDayStart :: FuturesIncome -> Int64
incomeDayStart row =
    let dayMs = 86400000
     in fiTime row - fiTime row `mod` dayMs

groupOn :: (Ord key) => (value -> key) -> [value] -> M.Map key [value]
groupOn key =
    foldl'
        (\acc row -> M.insertWith (flip (++)) (key row) [row] acc)
        M.empty

finiteOrZero :: Double -> Double
finiteOrZero value
    | isNaN value || isInfinite value = 0
    | otherwise = value

suffixOf :: String -> String -> Bool
suffixOf suffix value =
    let suffixLength = length suffix
        valueLength = length value
     in suffixLength <= valueLength && drop (valueLength - suffixLength) value == suffix
