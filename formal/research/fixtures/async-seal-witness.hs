{-# LANGUAGE CPP #-}
module Main (main) where
import Control.Concurrent (forkIO)
import Control.Concurrent.MVar
import Control.Exception (finally)
import Control.Monad (void)
import Data.Maybe (isJust)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission
-- Legacy branch reproduces Main's snapshot/result-cell acknowledgement only;
-- this is a source-bound control slice, not a running HTTP server.
seal :: JobSlots -> IO ()
#ifdef LEGACY
seal _ = pure ()
wait :: JobSlots -> [MVar ()] -> IO ()
wait _ = mapM_ readMVar
#else
seal = closeJobSlots
wait :: JobSlots -> [MVar ()] -> IO ()
wait slots _ = waitJobSlots slots
#endif
main :: IO ()
main = do
    slots <- newJobSlots 1
    preparing <- newEmptyMVar
    resume <- newEmptyMVar
    done <- newEmptyMVar
    _ <- forkIO (void (startBoundedJob slots (putMVar preparing () >> takeMVar resume) (const (pure ())) (\_ _ -> pure ())) `finally` putMVar done ())
    takeMVar preparing
    seal slots
    early <- isJust <$> timeout 20000 (wait slots [])
    count <- runningJobSlots slots
    print ("preparing outside snapshot", early, count)
    putMVar resume ()
    takeMVar done
    other <- newJobSlots 1
    result <- newEmptyMVar
    finalizing <- newEmptyMVar
    finish <- newEmptyMVar
    _ <- startBoundedJob other (pure ()) (\() -> putMVar result () `finally` (putMVar finalizing () >> takeMVar finish)) (\_ _ -> pure ())
    readMVar result
    takeMVar finalizing
    seal other
    acknowledged <- isJust <$> timeout 20000 (wait other [result])
    remaining <- runningJobSlots other
    print ("result before cleanup", acknowledged, remaining)
    putMVar finish ()
