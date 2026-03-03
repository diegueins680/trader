{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.ApiRoutes (
    apiRouteSuite,
) where

import Data.Aeson (Value (..))
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Text as T

import Trader.Api.Routes (apiEndpointDocs, apiRouteLabel)

apiRouteSuite :: [(String, IO ())]
apiRouteSuite =
    [ ("api route labels normalize async placeholders", testRouteLabels)
    , ("api endpoint docs include core endpoints", testEndpointDocsCore)
    , ("api endpoint docs have unique method/path pairs", testEndpointDocsUnique)
    ]

testRouteLabels :: IO ()
testRouteLabels = do
    expectEq "signal async cancel label" "signal/async/:jobId/cancel" (apiRouteLabel ["signal", "async", "job-1", "cancel"])
    expectEq "trade async poll label" "trade/async/:jobId" (apiRouteLabel ["trade", "async", "job-1"])
    expectEq "state sync label" "state/sync" (apiRouteLabel ["state", "sync"])
    expectEq "root label" "root" (apiRouteLabel [])

testEndpointDocsCore :: IO ()
testEndpointDocsCore = do
    let endpoints = endpointPairs apiEndpointDocs
    expectTrue "contains GET /health" (("GET", "/health") `elem` endpoints)
    expectTrue "contains GET /version" (("GET", "/version") `elem` endpoints)
    expectTrue "contains POST /trade" (("POST", "/trade") `elem` endpoints)
    expectTrue "contains POST /bot/start" (("POST", "/bot/start") `elem` endpoints)

testEndpointDocsUnique :: IO ()
testEndpointDocsUnique = do
    let endpoints = endpointPairs apiEndpointDocs
        uniq = unique endpoints
    expectEq "endpoint list has unique method/path pairs" (length uniq) (length endpoints)

endpointPairs :: [Value] -> [(String, String)]
endpointPairs vals =
    foldr gather [] vals
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

    parseString v =
        case v of
            String txt -> Just (T.unpack txt)
            _ -> Nothing

unique :: (Ord a) => [a] -> [a]
unique = foldr keep []
  where
    keep x acc =
        if x `elem` acc
            then acc
            else x : acc

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)

expectTrue :: String -> Bool -> IO ()
expectTrue label cond =
    if cond
        then pure ()
        else error (label ++ ": condition failed")
