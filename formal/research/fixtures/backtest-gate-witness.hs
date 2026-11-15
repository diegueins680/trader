{-# LANGUAGE CPP #-}
module Main (main) where
import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar
import Control.Exception (AsyncException (ThreadKilled), SomeException, finally, fromException, try)
import Control.Monad (void)
import System.Timeout (timeout)
import Trader.App.BacktestGate
code :: Either BacktestFailure a -> String
code value = case value of
    Right _ -> "success"
    Left BacktestBusy -> "busy"
    Left BacktestTimedOut -> "timeout"
#ifndef LEGACY
    Left BacktestDraining -> "draining"
#endif
    Left (BacktestException _) -> "error"
main :: IO ()
main = do
    gate <- newBacktestGate 1 1
    own <- runBacktestWithGate gate (threadDelay 10000000)
    print ("own timer",code own)
    entered <- newEmptyMVar
    hold <- newEmptyMVar
    done <- newEmptyMVar
    tid <- forkIO $ do
        result <- try (runBacktestWithGate gate (putMVar entered () >> takeMVar hold)) :: IO (Either SomeException (Either BacktestFailure ()))
        putMVar done result
    takeMVar entered
    killThread tid
    cancelled <- takeMVar done
    print ("external cancellation",case cancelled of Left ex | Just ThreadKilled <- fromException ex -> "propagated"; Left _ -> "other"; Right value -> code value)
    outer <- timeout 20000 (runBacktestWithGate gate (threadDelay 10000000))
    print ("outer timer", fmap code outer)
    print ("duration remains positive",timeoutMicroseconds maxBound > 0)
    count <- backtestRunningCount gate
    print ("all slots released",count)
    leaked <- reservationGap
    print ("reservation gap",leaked)

-- Schedule adapter: cancellation at the next interruptible point after reserve.
-- Legacy injects a barrier in Main's unmasked reserve/finalizer gap; the repair
-- reaches that barrier only inside the protected callback. Not full HTTP execution.
reservationGap :: IO Int
reservationGap = do
    entered <- newEmptyMVar
    hold <- newEmptyMVar
    done <- newEmptyMVar
#ifdef LEGACY
    count <- newMVar (0 :: Int)
    let release = modifyMVar_ count (pure . subtract 1)
        action = do
            modifyMVar_ count (pure . (+ 1))
            putMVar entered ()
            takeMVar hold
            pure () `finally` release
    tid <- forkIO (action `finally` putMVar done ())
#else
    gate <- newBacktestGate 1 10
    tid <- forkIO (void (runBacktestWithGate gate (putMVar entered () >> takeMVar hold)) `finally` putMVar done ())
#endif
    takeMVar entered
    killThread tid
    takeMVar done
#ifdef LEGACY
    readMVar count
#else
    backtestRunningCount gate
#endif
