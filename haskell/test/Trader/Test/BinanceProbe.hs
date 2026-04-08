module Trader.Test.BinanceProbe (
    binanceProbeSuite,
) where

import Trader.App.BinanceProbe (BinanceErrorInfo (..), binanceTradeTestConfirmsAuth, parseBinanceError)
import qualified Trader.App.BinanceProbe as BinanceProbe

binanceProbeSuite :: [(String, IO ())]
binanceProbeSuite =
    [ ("binance probe parser keeps wrapped auth failures as failures", testWrappedAuthFailure)
    , ("binance auth helper classifies wrapped order failures", testAuthFailureHelperWrappedOrderFailure)
    , ("binance auth helper ignores validation rejects", testAuthFailureHelperIgnoresValidationReject)
    , ("binance trade-test confirmation recognizes order validation rejects", testOrderValidationReject)
    , ("binance trade-test confirmation ignores transient upstream failures", testTransientFailure)
    , ("binance probe parser handles nested json bodies", testNestedJsonBody)
    , ("binance probe parser keeps brace characters inside messages", testBraceInMessage)
    , ("binance probe parser recognizes HTTP status-line prefixes", testHttpStatusLine)
    , ("binance probe parser recognizes wrapped HTTP/2 status lines", testWrappedHttp2StatusLine)
    , ("binance probe parser keeps long json messages intact", testLongJsonBody)
    ]

testWrappedAuthFailure :: IO ()
testWrappedAuthFailure = do
    let err = parseBinanceError "order/test HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action."
    expectEq "http code" (Just 401) (beiHttpCode err)
    expectEq "binance code" (Just (-2015)) (beiCode err)
    expectEq "summary" "Invalid API-key, IP, or permissions for action." (beiSummary err)
    expectFalse "auth/IP failures must stay failed" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

testAuthFailureHelperWrappedOrderFailure :: IO ()
testAuthFailureHelperWrappedOrderFailure =
    case BinanceProbe.binanceAuthFailureFromMessage "Order failed: user error (futures/positionRisk HTTP 401: Binance code -2015: Invalid API-key, IP, or permissions for action.)" of
        Nothing -> error "expected auth failure classification"
        Just err -> do
            expectEq "helper http code" (Just 401) (beiHttpCode err)
            expectEq "helper binance code" (Just (-2015)) (beiCode err)
            expectEq "helper summary" "Invalid API-key, IP, or permissions for action." (beiSummary err)

testAuthFailureHelperIgnoresValidationReject :: IO ()
testAuthFailureHelperIgnoresValidationReject =
    case BinanceProbe.binanceAuthFailureFromMessage "Order failed: Binance code -1013: Filter failure: LOT_SIZE" of
        Nothing -> pure ()
        Just err -> error ("unexpected auth failure classification: " ++ show err)

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

testNestedJsonBody :: IO ()
testNestedJsonBody = do
    let err = parseBinanceError "order/test HTTP 400: {\"code\":-2010,\"msg\":\"New order rejected\",\"data\":{\"reason\":\"LOT_SIZE\"}}"
    expectEq "nested json code" (Just (-2010)) (beiCode err)
    expectEq "nested json summary" "New order rejected" (beiSummary err)

testBraceInMessage :: IO ()
testBraceInMessage = do
    let err = parseBinanceError "order/test HTTP 400: {\"code\":-1013,\"msg\":\"Filter failure: LOT_SIZE {qty}\"}"
    expectEq "brace message code" (Just (-1013)) (beiCode err)
    expectEq "brace message summary" "Filter failure: LOT_SIZE {qty}" (beiSummary err)

testHttpStatusLine :: IO ()
testHttpStatusLine = do
    let err = parseBinanceError "POST /api/v3/order/test failed: HTTP/1.1 429 Too Many Requests"
    expectEq "status-line http code" (Just 429) (beiHttpCode err)
    expectEq "status-line output code" (Just 429) (beiCode err)
    expectFalse "status-line transient errors should not confirm auth" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

testWrappedHttp2StatusLine :: IO ()
testWrappedHttp2StatusLine = do
    let err =
            parseBinanceError
                "trade test wrapper: upstream status line: HTTP/2 429 Too Many Requests"
    expectEq "wrapped HTTP/2 http code" (Just 429) (beiHttpCode err)
    expectEq "wrapped HTTP/2 output code" (Just 429) (beiCode err)
    expectFalse
        "wrapped HTTP/2 transient errors should not confirm auth"
        (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

testLongJsonBody :: IO ()
testLongJsonBody = do
    let longMsg = replicate 260 'x' ++ " Filter failure: LOT_SIZE"
        err = parseBinanceError ("order/test HTTP/1.1 400: {\"code\":-1013,\"msg\":\"" ++ longMsg ++ "\"}")
    expectEq "long json code" (Just (-1013)) (beiCode err)
    expectEq "long json summary" longMsg (beiSummary err)
    expectTrue "long json validation reject should confirm auth" (binanceTradeTestConfirmsAuth (beiCode err) (beiSummary err))

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
