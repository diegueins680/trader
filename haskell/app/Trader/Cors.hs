{-# LANGUAGE OverloadedStrings #-}

module Trader.Cors (
    CorsConfig (..),
    corsHeadersFor,
    resolveCorsConfig,
    withCors,
) where

import Control.Applicative ((<|>))
import qualified Data.ByteString.Char8 as BS
import qualified Data.CaseInsensitive as CI
import Data.List (find)
import Data.Maybe (isJust)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Network.HTTP.Types (RequestHeaders, ResponseHeaders)
import qualified Network.Wai as Wai
import System.Environment (lookupEnv)

import Trader.App.Runtime (TenantKey, normalizeTenantKey, splitEnvList)
import Trader.Text (normalizeKey)

data CorsConfig = CorsConfig
    { ccAllowedOrigins :: ![BS.ByteString]
    , ccAllowAnyOrigin :: !Bool
    , ccAllowAuthOrigin :: !Bool
    }

resolveCorsConfig :: Maybe BS.ByteString -> IO CorsConfig
resolveCorsConfig _mToken = do
    mOriginEnv <- lookupEnv "TRADER_CORS_ORIGIN"
    let rawOrigins = maybe [] splitEnvList mOriginEnv
        trimmed = map (T.strip . T.pack) rawOrigins
        normalized = filter (not . T.null) trimmed
        tokens = map (normalizeKey . T.unpack) normalized
        allowAny = "*" `elem` normalized
        origins =
            [ TE.encodeUtf8 (T.dropWhileEnd (== '/') t)
            | (t, tok) <- zip normalized tokens
            , t /= "*"
            , tok /= "*"
            ]
        allowAuth = null origins && not allowAny
    pure (CorsConfig origins allowAny allowAuth)

lookupHeaderNormalized :: String -> RequestHeaders -> Maybe BS.ByteString
lookupHeaderNormalized wanted hs =
    let wantedNorm = normalizeKey wanted
     in snd <$> find (\(h, _) -> normalizeKey (BS.unpack (CI.original h)) == wantedNorm) hs

tenantKeyFromRequest :: Wai.Request -> Maybe TenantKey
tenantKeyFromRequest req =
    let headers = Wai.requestHeaders req
        headerRaw = lookupHeaderNormalized "X-Tenant-Key" headers
        queryRaw =
            case lookup "tenantKey" (Wai.queryString req) of
                Just (Just raw) -> Just raw
                _ -> Nothing
        headerStr = fmap BS.unpack headerRaw
        queryStr = fmap BS.unpack queryRaw
     in normalizeTenantKey headerStr <|> normalizeTenantKey queryStr

requestPathNoApiPrefix :: Wai.Request -> [Text]
requestPathNoApiPrefix req =
    case Wai.pathInfo req of
        ("api" : rest) -> rest
        path -> path

corsRequestHasAuthHeaders :: Wai.Request -> Bool
corsRequestHasAuthHeaders req =
    let hs = Wai.requestHeaders req
        hasHeader key = isJust (lookupHeaderNormalized key hs)
        requested =
            case lookupHeaderNormalized "Access-Control-Request-Headers" hs of
                Nothing -> []
                Just raw -> map normalizeKey (splitEnvList (BS.unpack raw))
        requestedHasAuth =
            let authHeaderKeys = map normalizeKey ["authorization", "x-api-key", "x-tenant-key"]
             in any (`elem` authHeaderKeys) requested
     in case Wai.requestMethod req of
            "OPTIONS" -> requestedHasAuth
            _ -> hasHeader "Authorization" || hasHeader "X-API-Key" || hasHeader "X-Tenant-Key"

corsRequestHasImplicitReadOrigin :: Wai.Request -> Bool
corsRequestHasImplicitReadOrigin req =
    Wai.requestMethod req == "GET"
        && ( path `elem` publicReadPaths
                || (path `elem` tenantScopedReadPaths && isJust (tenantKeyFromRequest req))
           )
  where
    path = requestPathNoApiPrefix req
    publicReadPaths =
        [ ["health"]
        , ["version"]
        , ["optimizer", "combos"]
        ]
    tenantScopedReadPaths =
        [ ["ops"]
        , ["ops", "performance"]
        , ["outbox"]
        , ["bot", "status"]
        , ["state", "sync"]
        , ["binance", "listenKey", "stream"]
        ]

corsHeadersFor :: CorsConfig -> Wai.Request -> ResponseHeaders
corsHeadersFor cors req =
    let base =
            [ ("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            , ("Access-Control-Allow-Headers", "Authorization,Content-Type,X-API-Key,X-Tenant-Key,X-Trader-Request-Id")
            , ("Access-Control-Max-Age", "86400")
            ]
        mAllowed
            | ccAllowAnyOrigin cors = Just "*"
            | otherwise =
                case lookupHeaderNormalized "Origin" (Wai.requestHeaders req) of
                    Just origin | origin `elem` ccAllowedOrigins cors -> Just origin
                    Just origin | ccAllowAuthOrigin cors && (corsRequestHasAuthHeaders req || corsRequestHasImplicitReadOrigin req) -> Just origin
                    _ -> Nothing
        extra =
            case mAllowed of
                Nothing -> []
                Just origin ->
                    if origin == "*"
                        then [("Access-Control-Allow-Origin", origin)]
                        else [("Access-Control-Allow-Origin", origin), ("Vary", "Origin")]
     in extra ++ base

withCors :: CorsConfig -> Wai.Request -> Wai.Response -> Wai.Response
withCors cors req = Wai.mapResponseHeaders (\hs -> corsHeadersFor cors req ++ hs)
