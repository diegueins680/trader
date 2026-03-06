module Trader.Test.BinanceProbe (
    binanceProbeSuite,
) where

import Trader.App.BinanceProbe (BinanceErrorInfo (..), binanceTradeTestConfirmsAuth, parseBinanceError)

binanceProbeSuite :: [(String, IO ())]
binanceProbeSuite =
    [ ("binance probe parser keeps wrapped auth failures as failures", testWrappedAuthFailure)
    , ("binance trade-test confirmation recognizes order validation rejects", testOrderValidationReject)
    , ("binance trade-test confirmation ignores transient upstream failures", testTransientFailure)
    ]

testWrappedAuthFailure :: IO ()
testWrappedAuthFailure = do
    let err = parseBinanceError "order/test HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action."
    expectEq "http code" (Just 401) (beiHttpCode err)
    expectEq "binance code" (Just (-2015)) (beiCode err)
    expectEq "summary" "Invalid API-key, IP, or permissions for action." (beiSummary err)
    expectFalse "auth/IP failures must stay failed" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

testOrderValidationReject :: IO ()
testOrderValidationReject = do
    let err = parseBinanceError "order/test HTTP 400: Binance code -1013: Filter failure: LOT_SIZE"
    expectEq "order reject code" (Just (-1013)) (beiCode err)
    expectEq "order reject summary" "Filter failure: LOT_SIZE" (beiSummary err)
    expectTrue "validation reject should confirm auth" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

testTransientFailure :: IO ()
testTransientFailure = do
    let err = parseBinanceError "order/test HTTP 503: Service Unavailable"
    expectEq "transient http code" (Just 503) (beiCode err)
    expectFalse "transient errors should not confirm auth" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

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

expectFalse :: String -> Bool -> IO ()
expectFalse label cond = expectTrue label (not cond)
