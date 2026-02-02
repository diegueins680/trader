module Trader.Platform (
    Platform (..),
    platformCode,
    platformLabel,
    parsePlatform,
    platformIntervals,
    platformIntervalsCsv,
    isPlatformInterval,
    platformDefaultBars,
    platformSupportsTrading,
    platformSupportsLiveBot,
    platformSupportsFutures,
    platformSupportsMargin,
    platformSupportsTestnet,
    platformSupportsMarketContext,
    coinbaseIntervalSeconds,
    krakenIntervalMinutes,
    poloniexIntervalLabel,
    poloniexIntervalSeconds,
) where

import Data.Char (toLower)
import Data.List (intercalate)

import Trader.BinanceIntervals (binanceIntervals)

data Platform
    = PlatformBinance
    | PlatformCoinbase
    | PlatformKraken
    | PlatformPoloniex
    | PlatformUniswap
    | PlatformCurve
    | PlatformSushiswap
    | PlatformBalancer
    | PlatformPancakeswap
    | PlatformOneInch
    deriving (Eq, Show)

platformCode :: Platform -> String
platformCode p =
    case p of
        PlatformBinance -> "binance"
        PlatformCoinbase -> "coinbase"
        PlatformKraken -> "kraken"
        PlatformPoloniex -> "poloniex"
        PlatformUniswap -> "uniswap"
        PlatformCurve -> "curve"
        PlatformSushiswap -> "sushiswap"
        PlatformBalancer -> "balancer"
        PlatformPancakeswap -> "pancakeswap"
        PlatformOneInch -> "1inch"

platformLabel :: Platform -> String
platformLabel p =
    case p of
        PlatformBinance -> "Binance"
        PlatformCoinbase -> "Coinbase"
        PlatformKraken -> "Kraken"
        PlatformPoloniex -> "Poloniex"
        PlatformUniswap -> "Uniswap"
        PlatformCurve -> "Curve"
        PlatformSushiswap -> "SushiSwap"
        PlatformBalancer -> "Balancer"
        PlatformPancakeswap -> "PancakeSwap"
        PlatformOneInch -> "1inch"

parsePlatform :: String -> Either String Platform
parsePlatform raw =
    case map toLower raw of
        "binance" -> Right PlatformBinance
        "coinbase" -> Right PlatformCoinbase
        "kraken" -> Right PlatformKraken
        "poloniex" -> Right PlatformPoloniex
        "uniswap" -> Right PlatformUniswap
        "curve" -> Right PlatformCurve
        "sushiswap" -> Right PlatformSushiswap
        "balancer" -> Right PlatformBalancer
        "pancakeswap" -> Right PlatformPancakeswap
        "1inch" -> Right PlatformOneInch
        "oneinch" -> Right PlatformOneInch
        other -> Left ("Invalid platform: " ++ show other ++ " (expected binance|coinbase|kraken|poloniex|uniswap|curve|sushiswap|balancer|pancakeswap|1inch)")

platformIntervals :: Platform -> [String]
platformIntervals p =
    case p of
        PlatformBinance -> binanceIntervals
        PlatformCoinbase -> ["1m", "5m", "15m", "1h", "6h", "1d"]
        PlatformKraken -> ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        PlatformPoloniex -> ["5m", "15m", "30m", "2h", "4h", "1d"]
        PlatformUniswap -> binanceIntervals
        PlatformCurve -> binanceIntervals
        PlatformSushiswap -> binanceIntervals
        PlatformBalancer -> binanceIntervals
        PlatformPancakeswap -> binanceIntervals
        PlatformOneInch -> binanceIntervals

platformIntervalsCsv :: Platform -> String
platformIntervalsCsv = intercalate "," . platformIntervals

isPlatformInterval :: Platform -> String -> Bool
isPlatformInterval p v = v `elem` platformIntervals p

platformDefaultBars :: Platform -> Int
platformDefaultBars p =
    case p of
        PlatformCoinbase -> 300
        _ -> 500

platformSupportsTrading :: Platform -> Bool
platformSupportsTrading p =
    p == PlatformBinance
        || p == PlatformCoinbase
        || p == PlatformUniswap
        || p == PlatformCurve
        || p == PlatformSushiswap
        || p == PlatformBalancer
        || p == PlatformPancakeswap
        || p == PlatformOneInch

platformSupportsLiveBot :: Platform -> Bool
platformSupportsLiveBot p = p == PlatformBinance

platformSupportsFutures :: Platform -> Bool
platformSupportsFutures p = p == PlatformBinance

platformSupportsMargin :: Platform -> Bool
platformSupportsMargin p = p == PlatformBinance

platformSupportsTestnet :: Platform -> Bool
platformSupportsTestnet p = p == PlatformBinance

platformSupportsMarketContext :: Platform -> Bool
platformSupportsMarketContext p = p == PlatformBinance

coinbaseIntervalSeconds :: String -> Maybe Int
coinbaseIntervalSeconds interval =
    case interval of
        "1m" -> Just 60
        "5m" -> Just 300
        "15m" -> Just 900
        "1h" -> Just 3600
        "6h" -> Just 21600
        "1d" -> Just 86400
        _ -> Nothing

krakenIntervalMinutes :: String -> Maybe Int
krakenIntervalMinutes interval =
    case interval of
        "1m" -> Just 1
        "5m" -> Just 5
        "15m" -> Just 15
        "30m" -> Just 30
        "1h" -> Just 60
        "4h" -> Just 240
        "1d" -> Just 1440
        "1w" -> Just 10080
        _ -> Nothing

poloniexIntervalSeconds :: String -> Maybe Int
poloniexIntervalSeconds interval =
    case interval of
        "5m" -> Just 300
        "15m" -> Just 900
        "30m" -> Just 1800
        "2h" -> Just 7200
        "4h" -> Just 14400
        "1d" -> Just 86400
        _ -> Nothing

poloniexIntervalLabel :: String -> Maybe String
poloniexIntervalLabel interval =
    case interval of
        "5m" -> Just "MINUTE_5"
        "15m" -> Just "MINUTE_15"
        "30m" -> Just "MINUTE_30"
        "2h" -> Just "HOUR_2"
        "4h" -> Just "HOUR_4"
        "1d" -> Just "DAY_1"
        _ -> Nothing
