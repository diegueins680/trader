module Trader.Test.AdmissionProgress (admissionProgressSuite) where

import Control.Concurrent (ThreadId, forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (MVar, newEmptyMVar, putMVar, takeMVar)
import Control.Exception (SomeException, finally, try)
import Control.Monad (forM_, replicateM_, void)
import System.Timeout (timeout)
import Trader.App.BacktestGate
import Trader.App.GracefulShutdown (beginDrain, newDrainController)

admissionProgressSuite :: [(String, IO ())]
admissionProgressSuite =
    [ ("backtest FIFO blocks barging and preserves order", testOrder)
    , ("backtest cancels head and middle waiters", testCancel)
    , ("backtest drain wakes registered waiters", testDrain)
    , ("backtest queues are independent", testIsolation)
    , ("backtest repeated failures cannot retain capacity", testRepeated)
    ]

expect :: Bool -> IO ()
expect ok = if ok then pure () else fail "admission progress assertion"

await :: IO a -> IO a
await action = timeout 5000000 action >>= maybe (fail "admission progress fixture timeout") pure

untilCount :: BacktestGate -> Int -> IO ()
untilCount gate wanted = await loop
  where
    loop = do
        count <- backtestWaitingCount gate
        if count == wanted then pure () else threadDelay 100 >> loop

held :: BacktestGate -> IO (MVar (), MVar ())
held gate = do
    entered <- newEmptyMVar
    finish <- newEmptyMVar
    done <- newEmptyMVar
    _ <- forkIO (void (runBacktestWithGate gate (putMVar entered () >> takeMVar finish)) `finally` putMVar done ())
    await (takeMVar entered)
    pure (finish, done)

queued :: BacktestGate -> IO () -> IO (ThreadId, MVar (Either SomeException (Either BacktestFailure ())))
queued gate action = do
    done <- newEmptyMVar
    tid <- forkIO (try (runBacktestWithGateWait gate action) >>= putMVar done)
    pure (tid, done)

success :: MVar (Either SomeException (Either BacktestFailure ())) -> IO ()
success done = do
    result <- await (takeMVar done)
    case result of
        Right (Right ()) -> pure ()
        _ -> fail "expected waiter success"

testOrder :: IO ()
testOrder = do
    gate <- newBacktestGate 1 10
    (finish, done) <- held gate
    firstEntered <- newEmptyMVar
    firstFinish <- newEmptyMVar
    secondEntered <- newEmptyMVar
    (_, firstDone) <- queued gate (putMVar firstEntered () >> takeMVar firstFinish)
    untilCount gate 1
    (_, secondDone) <- queued gate (putMVar secondEntered ())
    untilCount gate 2
    putMVar finish ()
    await (takeMVar done)
    replicateM_ 16 $ do
        result <- runBacktestWithGate gate (fail "barging callback executed")
        case result of
            Left BacktestBusy -> pure ()
            _ -> fail "immediate caller bypassed waiter"
    await (takeMVar firstEntered)
    expect . (== Nothing) =<< timeout 20000 (takeMVar secondEntered)
    putMVar firstFinish ()
    success firstDone
    success secondDone
    await (takeMVar secondEntered)
    expect . (== 0) =<< backtestWaitingCount gate
    expect . (== 0) =<< backtestRunningCount gate

testCancel :: IO ()
testCancel = forM_ [0, 1] $ \cancelIndex -> do
    gate <- newBacktestGate 1 10
    (finish, ownerDone) <- held gate
    first <- queued gate (pure ())
    untilCount gate 1
    second <- queued gate (pure ())
    untilCount gate 2
    third <- queued gate (pure ())
    untilCount gate 3
    let (victim, survivors) = if cancelIndex == 0 then (first, [second, third]) else (second, [first, third])
    killThread (fst victim)
    result <- await (takeMVar (snd victim))
    case result of
        Left _ -> pure ()
        _ -> fail "queued cancellation swallowed"
    untilCount gate 2
    expect . (== 1) =<< backtestRunningCount gate
    putMVar finish ()
    await (takeMVar ownerDone)
    mapM_ (success . snd) survivors
    untilCount gate 0
    expect . (== 0) =<< backtestRunningCount gate

testDrain :: IO ()
testDrain = do
    drain <- newDrainController
    gate <- newBacktestGateWithDrain drain 1 10
    (finish, ownerDone) <- held gate
    (_, waiterDone) <- queued gate (fail "drained callback ran")
    untilCount gate 1
    void (beginDrain drain)
    result <- await (takeMVar waiterDone)
    case result of
        Right (Left BacktestDraining) -> pure ()
        _ -> fail "waiting drain not observed"
    untilCount gate 0
    expect . (== 1) =<< backtestRunningCount gate
    putMVar finish ()
    await (takeMVar ownerDone)
    expect . (== 0) =<< backtestRunningCount gate

testIsolation :: IO ()
testIsolation = do
    blocked <- newBacktestGate 1 10
    other <- newBacktestGate 1 10
    (finish, ownerDone) <- held blocked
    (_, done) <- queued blocked (pure ())
    untilCount blocked 1
    result <- await (runBacktestWithGateWait other (pure ()))
    case result of
        Right () -> pure ()
        _ -> fail "unrelated gate blocked"
    putMVar finish ()
    await (takeMVar ownerDone)
    success done

testRepeated :: IO ()
testRepeated = forM_ [0 .. 31 :: Int] $ \i -> do
    gate <- newBacktestGate (1 + i `mod` 2) 10
    -- Fixed reproducible alternating failure/success schedule (seed20261006).
    let steps = 1 + (20261006 + i * 1103515245) `mod` 7
    replicateM_ steps $ do
        result <- runBacktestWithGateWait gate (fail "invalid fixture")
        case result of
            Left (BacktestException _) -> pure ()
            _ -> fail "invalid fixture accepted"
    (_, done) <- queued gate (pure ())
    success done
    expect . (== 0) =<< backtestRunningCount gate
    expect . (== 0) =<< backtestWaitingCount gate
