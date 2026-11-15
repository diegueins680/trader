module Trader.Test.DrainPool (drainPoolSuite) where

import Control.Concurrent (forkIO)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, readMVar, takeMVar)
import Control.Concurrent.STM (atomically, newTVarIO, readTVarIO, throwSTM, writeTVar)
import Control.Exception (SomeException, try)
import Control.Monad (forM, forM_, void)
import Data.IORef (atomicModifyIORef', newIORef, readIORef)
import Data.List (sort)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission
import Trader.App.BacktestGate
import Trader.App.GracefulShutdown (beginDrain, isDraining, newDrainController, unlessDraining)

drainPoolSuite :: [(String, IO ())]
drainPoolSuite =
    [ ("drain latch is monotonic and transactions roll back", testLatch)
    , ("stale ingress cannot reserve either drained pool", testStale)
    , ("pre-drain owners release and independent pools remain usable", testOwner)
    , ("generated concurrent drain and pool admissions", testConcurrent)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label expected actual =
    if expected == actual then pure () else ioError (userError (label ++ ": " ++ show actual ++ " /= " ++ show expected))

await :: IO a -> IO a
await action = timeout 5000000 action >>= maybe (ioError (userError "drain pool fixture timeout")) pure

isDrained :: Either BacktestFailure a -> Bool
isDrained (Left BacktestDraining) = True
isDrained _ = False

testLatch :: IO ()
testLatch = do
    drain <- newDrainController
    cell <- newTVarIO False
    result <- try (atomically (unlessDraining drain (writeTVar cell True >> throwSTM (userError "transaction abort")))) :: IO (Either SomeException (Maybe ()))
    expect "transaction abort" True (either (const True) (const False) result)
    expect "transaction rolled back" False =<< readTVarIO cell
    expect "open guard" (Just ()) =<< atomically (unlessDraining drain (writeTVar cell True))
    expect "committed action" True =<< readTVarIO cell
    expect "first drain" True =<< beginDrain drain
    expect "repeated drain" False =<< beginDrain drain
    expect "readiness" True =<< isDraining drain
    expect "closed guard" Nothing =<< atomically (unlessDraining drain (writeTVar cell False))
    expect "closed guard has no effects" True =<< readTVarIO cell

testStale :: IO ()
testStale = do
    drain <- newDrainController
    slots <- newJobSlotsWithDrain drain 1
    gate <- newBacktestGateWithDrain drain 1 2
    expect "ingress snapshot" False =<< isDraining drain
    void (beginDrain drain)
    effects <- newIORef (0 :: Int)
    let effect = atomicModifyIORef' effects (\n -> (n + 1, ()))
    result <- startBoundedJob slots effect (const effect) (\_ _ -> effect)
    expect "async rejected" (Left JobQueueClosed) result
    forM_ [runBacktestWithGate, runBacktestWithGateWait] $ \run ->
        expect "backtest rejected without retry" True . isDrained =<< await (run gate effect)
    expect "no rejected effects" 0 =<< readIORef effects
    expect "async count" 0 =<< runningJobSlots slots
    expect "backtest count" 0 =<< backtestRunningCount gate

testOwner :: IO ()
testOwner = do
    drain <- newDrainController
    slots <- newJobSlotsWithDrain drain 1
    gate <- newBacktestGateWithDrain drain 1 2
    entered <- newEmptyMVar
    finish <- newEmptyMVar
    done <- newEmptyMVar
    _ <- forkIO $ runBacktestWithGate gate (putMVar entered () >> readMVar finish) >>= putMVar done
    await (takeMVar entered)
    result <- startBoundedJob slots (pure ()) (const (readMVar finish)) (\_ _ -> pure ())
    expect "pre-drain async accepted" (Right ()) result
    queued <- newEmptyMVar
    _ <- forkIO (runBacktestWithGateWait gate (pure ()) >>= putMVar queued)
    expect "waiter blocked on capacity" Nothing =<< timeout 20000 (void (readMVar queued))
    void (beginDrain drain)
    expect "existing waiter exits while owner still holds capacity" True . isDrained =<< await (takeMVar queued)
    expect "existing backtest owner" 1 =<< backtestRunningCount gate
    expect "existing async owner" 1 =<< runningJobSlots slots
    closeJobSlots slots
    expect "unfinished ownership cannot acknowledge" Nothing =<< timeout 20000 (waitJobSlots slots)
    putMVar finish ()
    value <- await (takeMVar done)
    expect "owner can finish" True (either (const False) (const True) value)
    await (waitJobSlots slots)
    expect "all backtest reservations released" 0 =<< backtestRunningCount gate
    expect "all async reservations released" 0 =<< runningJobSlots slots
    independent <- newJobSlots 1
    expect "unrelated controller usable" (Right ()) =<< startBoundedJob independent (pure ()) pure (\_ _ -> pure ())
    closeJobSlots independent
    await (waitJobSlots independent)

testConcurrent :: IO ()
testConcurrent = forM_ (take 32 (iterate (\x -> (1103515245 * x + 12345) `mod` 2147483648) (20261004 :: Integer))) $ \seed -> do
    drain <- newDrainController
    slots <- newJobSlotsWithDrain drain (1 + fromInteger (seed `mod` 2))
    gate <- newBacktestGateWithDrain drain 2 2
    start <- newEmptyMVar
    dones <- forM [0 :: Int .. 3] $ \i -> do
        done <- newEmptyMVar
        _ <- forkIO $ do
            readMVar start
            if even (seed + toInteger i)
                then void (startBoundedJob slots (pure ()) pure (\_ _ -> pure ()))
                else void (runBacktestWithGate gate (pure ()))
            putMVar done ()
        pure done
    closers <- forM [0 :: Int, 1] $ \_ -> do
        closed <- newEmptyMVar
        _ <- forkIO (readMVar start >> beginDrain drain >>= putMVar closed)
        pure closed
    putMVar start ()
    outcomes <- mapM (await . takeMVar) closers
    expect "exactly one drain caller wins" [False, True] (sort outcomes)
    mapM_ (await . takeMVar) dones
    expect "late async rejected" (Left JobQueueClosed) =<< startBoundedJob slots (pure ()) pure (\_ _ -> pure ())
    expect "late backtest rejected" True . isDrained =<< runBacktestWithGate gate (pure ())
    closeJobSlots slots
    await (waitJobSlots slots)
    expect "async zero after race" 0 =<< runningJobSlots slots
    expect "backtest zero after race" 0 =<< backtestRunningCount gate
