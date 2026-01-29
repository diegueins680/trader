module Trader.LstmPersistence (
    lstmModelKey,
) where

import Control.Exception (SomeException, try)
import Data.List (intercalate)
import System.Directory (canonicalizePath)

import Trader.App.Args (Args (..), argBinanceMarket)
import Trader.Binance (BinanceMarket (..))
import Trader.Platform (Platform (..), platformCode)

safeCanonicalizePath :: FilePath -> IO FilePath
safeCanonicalizePath path = do
    r <- try (canonicalizePath path) :: IO (Either SomeException FilePath)
    pure (either (const path) id r)

binanceMarketKey :: BinanceMarket -> String
binanceMarketKey m =
    case m of
        MarketSpot -> "spot"
        MarketMargin -> "margin"
        MarketFutures -> "futures"

exchangeSourceKey :: Platform -> BinanceMarket -> String -> String
exchangeSourceKey platform market sym =
    case platform of
        PlatformBinance -> platformCode platform ++ ":" ++ binanceMarketKey market ++ ":" ++ sym
        _ -> platformCode platform ++ ":" ++ sym

lstmModelKey :: Args -> Int -> IO String
lstmModelKey args lookback = do
    src <-
        case (argBinanceSymbol args, argData args) of
            (Just sym, _) -> pure (exchangeSourceKey (argPlatform args) (argBinanceMarket args) sym)
            (Nothing, Just path0) -> do
                path <- safeCanonicalizePath path0
                pure ("csv:" ++ path ++ ":" ++ argPriceCol args)
            _ -> pure "unknown"
    pure
        ( intercalate
            "|"
            [ "v1"
            , src
            , "interval=" ++ argInterval args
            , "norm=" ++ show (argNormalization args)
            , "hidden=" ++ show (argHiddenSize args)
            , "lookback=" ++ show lookback
            ]
        )
