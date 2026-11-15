module Trader.Test.BacktestGate (backtestGateSuite) where

import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (AsyncException (ThreadKilled), MaskingState (MaskedInterruptible, Unmasked), SomeException, finally, fromException, getMaskingState, mask_, try)
import Control.Monad (forM, forM_, void)
import System.Timeout (timeout)
import Trader.App.BacktestGate

backtestGateSuite :: [(String, IO ())]
backtestGateSuite =
    [ ("backtest gate success and synchronous errors release", testResults)
    , ("backtest own expiry is a timeout", testOwnTimeout)
    , ("backtest cancellation propagates and releases", testCancellation)
    , ("backtest outer timeout propagates and releases", testOuterTimeout)
    , ("backtest saturation and waiting cancellation preserve ownership", testWaiting)
    , ("backtest gate preserves caller masking", testMasking)
    , ("backtest generated concurrency remains bounded", testConcurrent)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

await :: IO a -> IO a
await action = timeout 5000000 action >>= maybe (ioError (userError "backtest gate fixture timed out")) pure

code :: Either BacktestFailure a -> String
code result = case result of
    Right _ -> "success"
    Left BacktestBusy -> "busy"
    Left BacktestDraining -> "draining"
    Left BacktestTimedOut -> "timeout"
    Left (BacktestException _) -> "error"

testResults :: IO ()
testResults = do
    gate <- newBacktestGate 0 0
    expect "minimum timeout" 1 (btTimeoutSec gate)
    expect "successful callback" "success" . code =<< runBacktestWithGate gate (pure ())
    expect "synchronous failure" "error" . code =<< runBacktestWithGate gate (ioError (userError "fixture"))
    expect "released" 0 =<< backtestRunningCount gate

testOwnTimeout :: IO ()
testOwnTimeout = do
    gate <- newBacktestGate 1 1
    expect "own timer classification" "timeout" . code =<< await (runBacktestWithGate gate (threadDelay 10000000))
    expect "timeout releases" 0 =<< backtestRunningCount gate

testCancellation :: IO ()
testCancellation = forM_ [runBacktestWithGate, runBacktestWithGateWait] $ \run -> do
    gate <- newBacktestGate 1 10
    entered <- newEmptyMVar
    hold <- newEmptyMVar
    done <- newEmptyMVar
    tid <- forkIO $ do
        result <- try (run gate (putMVar entered () >> takeMVar hold)) :: IO (Either SomeException (Either BacktestFailure ()))
        putMVar done result
    await (takeMVar entered)
    killThread tid
    result <- await (takeMVar done)
    case result of
        Left ex -> expect "original cancellation" (Just ThreadKilled) (fromException ex)
        Right value -> ioError (userError ("cancellation swallowed: " ++ code value))
    expect "cancel releases" 0 =<< backtestRunningCount gate

testOuterTimeout :: IO ()
testOuterTimeout = do
    gate <- newBacktestGate 1 10
    result <- timeout 20000 (runBacktestWithGate gate (threadDelay 10000000))
    expect "outer timer owns exception" Nothing (fmap code result)
    expect "outer timer releases" 0 =<< backtestRunningCount gate

testWaiting :: IO ()
testWaiting = do
    gate <- newBacktestGate 1 10
    entered <- newEmptyMVar
    finish <- newEmptyMVar
    ownerDone <- newEmptyMVar
    _ <- forkIO (void (runBacktestWithGate gate (putMVar entered () >> takeMVar finish)) `finally` putMVar ownerDone ())
    await (takeMVar entered)
    expect "full gate rejects" "busy" . code =<< runBacktestWithGate gate (ioError (userError "busy callback ran"))
    cancelled <- timeout 20000 (runBacktestWithGateWait gate (pure ()))
    expect "waiting external timeout" Nothing (fmap code cancelled)
    expect "waiter owns no slot" 1 =<< backtestRunningCount gate
    queued <- newEmptyMVar
    queuedResult <- newEmptyMVar
    _ <- forkIO $ do
        value <- runBacktestWithGateWait gate (putMVar queued ())
        putMVar queuedResult value
    expect "queued callback cannot bypass owner" Nothing =<< timeout 20000 (takeMVar queued)
    putMVar finish ()
    await (takeMVar ownerDone)
    expect "waiting mode resumes" "success" . code =<< await (takeMVar queuedResult)
    await (takeMVar queued)
    expect "owners released" 0 =<< backtestRunningCount gate

testMasking :: IO ()
testMasking = do
    gate <- newBacktestGate 1 10
    let check wanted action = do
            result <- action
            case result of
                Right state -> expect "caller masking" wanted state
                Left err -> ioError (userError (show err))
    check Unmasked (runBacktestWithGate gate getMaskingState)
    check MaskedInterruptible (mask_ (runBacktestWithGate gate getMaskingState))

testConcurrent :: IO ()
testConcurrent = forM_ seeds $ \seed -> do
    gate <- newBacktestGate 2 10
    completions <- forM [1 .. 4 :: Integer] $ \i -> do
        done <- newEmptyMVar
        _ <- forkIO $ do
            threadDelay (fromInteger ((seed `div` i) `mod` 3) * 1000)
            result <- runBacktestWithGate gate $ do
                count <- backtestRunningCount gate
                expect "in-flight bound" True (count >= 1 && count <= 2)
                threadDelay 3000
            putMVar done result
        pure done
    results <- await (mapM takeMVar completions)
    forM_ results $ \result -> expect "only success or contention" True (code result `elem` ["success", "busy"])
    expect "final count" 0 =<< backtestRunningCount gate
  where
    seeds = take 32 (tail (iterate (\n -> (1664525 * n + 1013904223) `mod` 4294967296) (20261004 :: Integer)))
