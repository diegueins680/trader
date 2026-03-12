module Trader.App.Runtime (
    TenantKey,
    hashKeyHex,
    marketCode,
    normalizeBinanceCredential,
    normalizeSymbol,
    normalizeTenantKey,
    readEnvBool,
    resolveTenantKeyFromParams,
    resolveTenantKeyFromPlatformParams,
    splitEnvList,
    tenantKeyFromBinanceKeys,
    tenantKeyFromCoinbaseKeys,
) where

import Crypto.Hash (Digest, hash)
import Crypto.Hash.Algorithms (SHA256)
import Data.ByteArray (convert)
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS
import Data.Char (toLower, toUpper)
import Data.List (nub)
import Data.Maybe (catMaybes)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE

import Trader.Binance (BinanceMarket (..))
import Trader.Platform (Platform (..))
import Trader.Text (trim)

marketCode :: BinanceMarket -> String
marketCode m =
    case m of
        MarketSpot -> "spot"
        MarketMargin -> "margin"
        MarketFutures -> "futures"

normalizeBinanceCredential :: Maybe String -> Maybe String
normalizeBinanceCredential raw =
    case raw of
        Nothing -> Nothing
        Just v ->
            let trimmed = trim v
             in if null trimmed then Nothing else Just trimmed

type TenantKey = Text

tenantKeyPrefixBinance :: String
tenantKeyPrefixBinance = "binance"

tenantKeyPrefixCoinbase :: String
tenantKeyPrefixCoinbase = "coinbase"

normalizeTenantKey :: Maybe String -> Maybe TenantKey
normalizeTenantKey raw =
    case raw of
        Nothing -> Nothing
        Just v ->
            let trimmed = trim v
             in if null trimmed then Nothing else Just (T.pack trimmed)

tenantKeyFromBinanceKeys :: Maybe String -> Maybe String -> Maybe TenantKey
tenantKeyFromBinanceKeys mKey mSecret = do
    key <- normalizeBinanceCredential mKey
    secret <- normalizeBinanceCredential mSecret
    let payload = key ++ ":" ++ secret
    pure (T.pack (tenantKeyPrefixBinance ++ ":" ++ hashKeyHex payload))

tenantKeyFromCoinbaseKeys :: Maybe String -> Maybe String -> Maybe String -> Maybe TenantKey
tenantKeyFromCoinbaseKeys mKey mSecret mPass = do
    key <- normalizeBinanceCredential mKey
    secret <- normalizeBinanceCredential mSecret
    passphrase <- normalizeBinanceCredential mPass
    let payload = key ++ ":" ++ secret ++ ":" ++ passphrase
    pure (T.pack (tenantKeyPrefixCoinbase ++ ":" ++ hashKeyHex payload))

tenantKeysFromParams ::
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    [TenantKey]
tenantKeysFromParams bKey bSecret cKey cSecret cPass =
    nub
        ( catMaybes
            [ tenantKeyFromBinanceKeys bKey bSecret
            , tenantKeyFromCoinbaseKeys cKey cSecret cPass
            ]
        )

resolveTenantKeyMatch :: Maybe TenantKey -> [TenantKey] -> Either String (Maybe TenantKey)
resolveTenantKeyMatch explicit computedKeys =
    case explicit of
        Just tenant
            | null computedKeys || tenant `elem` computedKeys -> Right (Just tenant)
            | otherwise -> Left "tenantKey does not match provided API keys."
        Nothing ->
            case computedKeys of
                [] -> Right Nothing
                [tenant] -> Right (Just tenant)
                _ -> Left "Multiple API key sets provided; specify platform or tenantKey."

resolveTenantKeyFromPlatformParams ::
    Platform ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Either String (Maybe TenantKey)
resolveTenantKeyFromPlatformParams platform mTenantRaw bKey bSecret cKey cSecret cPass =
    let explicit = normalizeTenantKey mTenantRaw
        computed =
            case platform of
                PlatformBinance -> tenantKeyFromBinanceKeys bKey bSecret
                PlatformCoinbase -> tenantKeyFromCoinbaseKeys cKey cSecret cPass
                _ -> Nothing
     in resolveTenantKeyMatch explicit (maybe [] pure computed)

resolveTenantKeyFromParams ::
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Maybe String ->
    Either String (Maybe TenantKey)
resolveTenantKeyFromParams mTenantRaw bKey bSecret cKey cSecret cPass =
    let explicit = normalizeTenantKey mTenantRaw
        computedKeys = tenantKeysFromParams bKey bSecret cKey cSecret cPass
     in resolveTenantKeyMatch explicit computedKeys

splitEnvList :: String -> [String]
splitEnvList raw =
    let cleaned = map (\c -> if c == ',' then ' ' else c) raw
     in filter (not . null) (map trim (words cleaned))

readEnvBool :: Maybe String -> Bool -> Bool
readEnvBool raw def =
    case fmap (map toLower . trim) raw of
        Just "1" -> True
        Just "true" -> True
        Just "yes" -> True
        Just "on" -> True
        Just "0" -> False
        Just "false" -> False
        Just "no" -> False
        Just "off" -> False
        _ -> def

normalizeSymbol :: String -> String
normalizeSymbol = map toUpper . trim

hashKeyHex :: String -> String
hashKeyHex s =
    let digest :: Digest SHA256
        digest = hash (TE.encodeUtf8 (T.pack s))
     in BS.unpack (B16.encode (convert digest))
