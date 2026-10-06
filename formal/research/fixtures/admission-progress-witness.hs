{-# LANGUAGE CPP #-}

module Main (main) where

import Control.Concurrent (forkIO, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (finally)
import Control.Monad (void)
#ifdef LEGACY
import GHC.Conc (ThreadStatus (ThreadBlocked), threadStatus)
#endif
import System.Timeout (timeout)
import Trader.App.BacktestGate

await :: IO a -> IO a
await action = timeout 5000000 action >>= maybe (fail "witness timeout") pure

main :: IO ()
main = do
    gate <- newBacktestGate 1 10
    entered <- newEmptyMVar
    finish <- newEmptyMVar
    done <- newEmptyMVar
    _ <- forkIO (void (runBacktestWithGate gate (putMVar entered () >> takeMVar finish)) `finally` putMVar done ())
    await (takeMVar entered)
    waiterFinish <- newEmptyMVar
    waiterDone <- newEmptyMVar
    waiter <- forkIO (void (runBacktestWithGateWait gate (takeMVar waiterFinish)) `finally` putMVar waiterDone ())
#ifdef LEGACY
    let registered = do
            status <- threadStatus waiter
            case status of
                ThreadBlocked _ -> pure ()
                _ -> threadDelay 100 >> registered
#else
    let registered = do
            count <- backtestWaitingCount gate
            if count == 1 then pure () else threadDelay 100 >> registered
    void (pure waiter)
#endif
    await registered
    putMVar finish ()
    await (takeMVar done)
    result <- runBacktestWithGate gate (pure ())
    putStrLn $ case result of
        Right () -> "barged=True"
        Left BacktestBusy -> "barged=False"
        _ -> "unexpected result"
    putMVar waiterFinish ()
    await (takeMVar waiterDone)
