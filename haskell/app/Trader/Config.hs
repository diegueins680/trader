{-# LANGUAGE RecordWildCards #-}

module Trader.Config (
    validateRuntimeConfig,
    shouldRequireUserTradeKeys,
) where

import Data.Maybe (isJust)
import System.Environment (lookupEnv)

import Trader.App.Args (Args (..))
import Trader.Dex (resolveDexEnv)
import Trader.Platform (Platform (..))
import Trader.Text (trim)

shouldRequireUserTradeKeys :: Platform -> Maybe tenant -> Bool -> Bool -> Bool
shouldRequireUserTradeKeys platform mReqTenant useServerKeys dryRun
    | dryRun = False
    | otherwise =
        case (platform, mReqTenant) of
            (PlatformBinance, Just _) -> not useServerKeys
            (PlatformCoinbase, Just _) -> not useServerKeys
            _ -> False

validateRuntimeConfig :: Args -> IO (Either String ())
validateRuntimeConfig args
    | not (argBinanceTrade args) = pure (Right ())
    | argDryRun args = pure (Right ())
    | otherwise =
        case argPlatform args of
            PlatformBinance -> validateBinanceCredentials args
            PlatformCoinbase -> validateCoinbaseCredentials args
            PlatformUniswap -> validateDexTrading args
            PlatformCurve -> validateDexTrading args
            PlatformSushiswap -> validateDexTrading args
            PlatformBalancer -> validateDexTrading args
            PlatformPancakeswap -> validateDexTrading args
            PlatformOneInch -> validateDexTrading args
            _ -> pure (Right ())

validateBinanceCredentials :: Args -> IO (Either String ())
validateBinanceCredentials Args{..} = do
    mKey <- resolveCredential "BINANCE_API_KEY" argBinanceApiKey
    mSecret <- resolveCredential "BINANCE_API_SECRET" argBinanceApiSecret
    pure $
        if present mKey && present mSecret
            then Right ()
            else Left "Trading requires BINANCE_API_KEY and BINANCE_API_SECRET (or --binance-api-key/--binance-api-secret)."

validateCoinbaseCredentials :: Args -> IO (Either String ())
validateCoinbaseCredentials Args{..}
    | not argBinanceLive =
        pure (Left "Coinbase trading requires --binance-live (Coinbase does not support test orders).")
    | otherwise = do
        mKey <- resolveCredential "COINBASE_API_KEY" argCoinbaseApiKey
        mSecret <- resolveCredential "COINBASE_API_SECRET" argCoinbaseApiSecret
        mPassphrase <- resolveCredential "COINBASE_API_PASSPHRASE" argCoinbaseApiPassphrase
        pure $
            if present mKey && present mSecret && present mPassphrase
                then Right ()
                else Left "Trading requires COINBASE_API_KEY, COINBASE_API_SECRET, and COINBASE_API_PASSPHRASE (or matching CLI flags)."

validateDexTrading :: Args -> IO (Either String ())
validateDexTrading Args{..} = do
    resolved <- resolveDexEnv argDexChainId (Just argDexAutoApprove)
    pure $
        case resolved of
            Left err -> Left ("DEX trading config invalid: " ++ err)
            Right _ -> Right ()

resolveCredential :: String -> Maybe String -> IO (Maybe String)
resolveCredential envName override =
    case override >>= normalize of
        Just value -> pure (Just value)
        Nothing -> do
            raw <- lookupEnv envName
            pure (raw >>= normalize)

nonEmptyString :: String -> Maybe String
nonEmptyString s =
    case s of
        "" -> Nothing
        _ -> Just s

normalize :: String -> Maybe String
normalize = nonEmptyString . trim

present :: Maybe String -> Bool
present = isJust
