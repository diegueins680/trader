{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.Cors (
    corsSuite,
) where

import qualified Data.ByteString.Char8 as BS
import qualified Data.Text as T
import Network.HTTP.Types (HeaderName, RequestHeaders, ResponseHeaders)
import qualified Network.Wai as Wai

import Trader.Cors (CorsConfig (..), corsHeadersFor)

corsSuite :: [(String, IO ())]
corsSuite =
    [ ("implicit cors allows health reads", testHealthReadAllowed)
    , ("implicit cors allows optimizer combos reads", testOptimizerCombosReadAllowed)
    , ("implicit cors allows tenant-scoped ops reads", testTenantScopedOpsReadAllowed)
    , ("implicit cors rejects tenant-scoped reads without tenant key", testTenantScopedReadRequiresTenantKey)
    , ("implicit cors rejects non-read tenant routes without auth headers", testTenantScopedWriteStillBlocked)
    , ("auth-like headers still allow origin echo", testAuthHeaderAllowsOriginEcho)
    , ("preflight allows request-progress header on tenant writes", testTenantWritePreflightAllowsRequestProgressHeader)
    ]

implicitCorsConfig :: CorsConfig
implicitCorsConfig = CorsConfig [] False True

originHeader :: (HeaderName, BS.ByteString)
originHeader = ("Origin", "https://trader-web-hs.fly.dev")

mkRequest :: BS.ByteString -> [T.Text] -> [(BS.ByteString, Maybe BS.ByteString)] -> RequestHeaders -> Wai.Request
mkRequest method path query extraHeaders =
    Wai.defaultRequest
        { Wai.requestMethod = method
        , Wai.pathInfo = path
        , Wai.queryString = query
        , Wai.requestHeaders = originHeader : extraHeaders
        }

lookupResponseHeader :: HeaderName -> ResponseHeaders -> Maybe BS.ByteString
lookupResponseHeader = lookup

expectOriginAllowed :: String -> Wai.Request -> IO ()
expectOriginAllowed label req =
    expectEq label (Just "https://trader-web-hs.fly.dev") (lookupResponseHeader "Access-Control-Allow-Origin" (corsHeadersFor implicitCorsConfig req))

expectOriginBlocked :: String -> Wai.Request -> IO ()
expectOriginBlocked label req =
    expectEq label Nothing (lookupResponseHeader "Access-Control-Allow-Origin" (corsHeadersFor implicitCorsConfig req))

testHealthReadAllowed :: IO ()
testHealthReadAllowed =
    expectOriginAllowed "GET /health echoes origin" (mkRequest "GET" ["health"] [] [])

testOptimizerCombosReadAllowed :: IO ()
testOptimizerCombosReadAllowed =
    expectOriginAllowed "GET /api/optimizer/combos echoes origin" (mkRequest "GET" ["api", "optimizer", "combos"] [] [])

testTenantScopedOpsReadAllowed :: IO ()
testTenantScopedOpsReadAllowed =
    expectOriginAllowed
        "GET /ops with tenantKey echoes origin"
        (mkRequest "GET" ["ops"] [("tenantKey", Just "binance:abc123")] [])

testTenantScopedReadRequiresTenantKey :: IO ()
testTenantScopedReadRequiresTenantKey =
    expectOriginBlocked "GET /ops without tenantKey does not echo origin" (mkRequest "GET" ["ops"] [] [])

testTenantScopedWriteStillBlocked :: IO ()
testTenantScopedWriteStillBlocked =
    expectOriginBlocked
        "POST /state/sync with tenantKey still does not echo origin"
        (mkRequest "POST" ["state", "sync"] [("tenantKey", Just "binance:abc123")] [])

testAuthHeaderAllowsOriginEcho :: IO ()
testAuthHeaderAllowsOriginEcho =
    expectOriginAllowed
        "auth-like headers still allow origin echo"
        (mkRequest "GET" ["metrics"] [] [("X-API-Key", "demo-token")])

testTenantWritePreflightAllowsRequestProgressHeader :: IO ()
testTenantWritePreflightAllowsRequestProgressHeader = do
    let req =
            mkRequest
                "OPTIONS"
                ["binance", "positions"]
                []
                [ ("Access-Control-Request-Method", "POST")
                , ("Access-Control-Request-Headers", "content-type,x-tenant-key,x-trader-request-id")
                ]
        headers = corsHeadersFor implicitCorsConfig req
    expectEq
        "OPTIONS /binance/positions with tenant/request-progress headers echoes origin"
        (Just "https://trader-web-hs.fly.dev")
        (lookupResponseHeader "Access-Control-Allow-Origin" headers)
    expectEq
        "OPTIONS /binance/positions allows request-progress header"
        (Just "Authorization,Content-Type,X-API-Key,X-Tenant-Key,X-Trader-Request-Id")
        (lookupResponseHeader "Access-Control-Allow-Headers" headers)

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)
