{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.ApiRoutes (
    apiRouteSuite,
) where

import Data.Aeson (Value (..))
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import Data.List (sort)
import qualified Data.Text as T

import Trader.Api.Routes (apiEndpointDocs, apiRouteLabel)

apiRouteSuite :: [(String, IO ())]
apiRouteSuite =
    [ ("api route labels cover special-case labels", testRouteLabels)
    , ("api endpoint docs match the published catalog", testEndpointDocsCatalog)
    , ("api endpoint docs round-trip async placeholder labels", testAsyncEndpointDocLabels)
    , ("api endpoint docs have unique method/path pairs", testEndpointDocsUnique)
    ]

testRouteLabels :: IO ()
testRouteLabels =
    mapM_ (uncurry expectRouteLabel) expectedRouteLabels

testEndpointDocsCatalog :: IO ()
testEndpointDocsCatalog = do
    let endpoints = sort (endpointPairs apiEndpointDocs)
    expectEq
        "api endpoint docs match expected catalog"
        (sort expectedEndpointPairs)
        endpoints

testAsyncEndpointDocLabels :: IO ()
testAsyncEndpointDocLabels =
    mapM_ expectAsyncDocLabel expectedAsyncEndpointPairs

testEndpointDocsUnique :: IO ()
testEndpointDocsUnique = do
    let endpoints = endpointPairs apiEndpointDocs
        uniq = unique endpoints
    expectEq "endpoint list has unique method/path pairs" (length uniq) (length endpoints)

expectedRouteLabels :: [([T.Text], String)]
expectedRouteLabels =
    [ ([], "root")
    , (["signal", "async", "job-1"], "signal/async/:jobId")
    , (["signal", "async", "job-1", "cancel"], "signal/async/:jobId/cancel")
    , (["trade", "async", "job-1"], "trade/async/:jobId")
    , (["trade", "async", "job-1", "cancel"], "trade/async/:jobId/cancel")
    , (["backtest", "async", "job-1"], "backtest/async/:jobId")
    , (["backtest", "async", "job-1", "cancel"], "backtest/async/:jobId/cancel")
    , (["request-progress", "request-1"], "request-progress/:requestId")
    , (["optimizer", "run"], "optimizer/run")
    , (["optimizer", "combos"], "optimizer/combos")
    , (["state", "sync"], "state/sync")
    ]

expectedEndpointPairs :: [(String, String)]
expectedEndpointPairs =
    [ ("GET", "/")
    , ("GET", "/health")
    , ("GET", "/version")
    , ("GET", "/metrics")
    , ("GET", "/ops")
    , ("GET", "/ops/performance")
    , ("GET", "/outbox")
    , ("GET", "/cache")
    , ("POST", "/cache/clear")
    , ("POST", "/signal")
    , ("POST", "/signal/async")
    , ("GET", "/signal/async/:jobId")
    , ("POST", "/signal/async/:jobId")
    , ("POST", "/signal/async/:jobId/cancel")
    , ("POST", "/trade")
    , ("POST", "/trade/async")
    , ("GET", "/trade/async/:jobId")
    , ("POST", "/trade/async/:jobId")
    , ("POST", "/trade/async/:jobId/cancel")
    , ("POST", "/backtest")
    , ("POST", "/backtest/async")
    , ("GET", "/backtest/async/:jobId")
    , ("POST", "/backtest/async/:jobId")
    , ("POST", "/backtest/async/:jobId/cancel")
    , ("GET", "/request-progress/:requestId")
    , ("POST", "/binance/keys")
    , ("POST", "/binance/trades")
    , ("GET", "/binance/positions")
    , ("POST", "/binance/positions")
    , ("POST", "/binance/positions/close")
    , ("GET", "/binance/proxy/health")
    , ("POST", "/coinbase/keys")
    , ("POST", "/binance/listenKey")
    , ("GET", "/binance/listenKey/stream")
    , ("POST", "/binance/listenKey/keepAlive")
    , ("POST", "/binance/listenKey/close")
    , ("POST", "/bot/start")
    , ("POST", "/bot/stop")
    , ("GET", "/bot/status")
    , ("POST", "/optimizer/run")
    , ("GET", "/optimizer/combos")
    , ("GET", "/state/sync")
    , ("POST", "/state/sync")
    ]

expectedAsyncEndpointPairs :: [(String, String)]
expectedAsyncEndpointPairs =
    filter
        (T.isInfixOf ":jobId" . T.pack . snd)
        expectedEndpointPairs

expectRouteLabel :: [T.Text] -> String -> IO ()
expectRouteLabel path expected =
    expectEq ("route label for " ++ show path) expected (apiRouteLabel path)

expectAsyncDocLabel :: (String, String) -> IO ()
expectAsyncDocLabel (_method, path) =
    expectEq
        ("doc placeholder label for " ++ path)
        (docPathLabel path)
        (apiRouteLabel (segmentsFromDocPath path))

docPathLabel :: String -> String
docPathLabel path =
    case path of
        "/" -> "root"
        '/' : rest -> rest
        _ -> path

segmentsFromDocPath :: String -> [T.Text]
segmentsFromDocPath path =
    map normalizeSegment (filter (not . T.null) (T.splitOn "/" (T.pack path)))
  where
    normalizeSegment segment
        | segment == ":jobId" = "job-1"
        | otherwise = segment

endpointPairs :: [Value] -> [(String, String)]
endpointPairs =
    foldr gather []
  where
    gather value acc =
        case value of
            Object o ->
                let mMethod = KM.lookup (AK.fromString "method") o >>= parseString
                    mPath = KM.lookup (AK.fromString "path") o >>= parseString
                 in case (mMethod, mPath) of
                        (Just method, Just path) -> (method, path) : acc
                        _ -> acc
            _ -> acc

    parseString :: Value -> Maybe String
    parseString v =
        case v of
            String txt -> Just (T.unpack txt)
            _ -> Nothing

unique :: (Eq a) => [a] -> [a]
unique =
    foldr keep []
  where
    keep x acc
        | x `elem` acc = acc
        | otherwise = x : acc

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)
