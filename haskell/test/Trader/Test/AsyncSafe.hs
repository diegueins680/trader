module Trader.Test.AsyncSafe (asyncSafeSuite) where

import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (ErrorCall (..), SomeException, finally, throwIO, try)
import Control.Monad (void)
import Data.Either (fromRight)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Maybe (isNothing)
import System.Timeout (timeout)
import Trader.App.AsyncSafe (isAsyncException, trySync)

asyncSafeSuite :: [(String, IO ())]
asyncSafeSuite =
    [ ("legacy try swallows killThread and keeps acting", legacySwallowsKill)
    , ("trySync rethrows killThread so no later step runs", killStopsLaterSteps)
    , ("trySync still captures synchronous failures", synchronousCaptured)
    , ("trySync does not defeat System.Timeout.timeout", timeoutPropagates)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

-- An order routine shaped like the worker: step 1 is an exchange call, step 2 a follow-up order.
routine :: (IO String -> IO (Either SomeException String)) -> IO [String]
routine capture = do
    steps <- newIORef []
    done <- newEmptyMVar
    tid <-
        forkIO $
            ( do
                r <- capture (threadDelay 500000 >> pure "entry")
                modifyIORef' steps (fromRight "entry failed" r :)
                modifyIORef' steps ("follow-up order" :)
            )
                `finally` putMVar done ()
    threadDelay 50000
    killThread tid
    takeMVar done
    reverse <$> readIORef steps

-- Preserved counterexample (CE-LIVE-001): the legacy capture keeps acting after the kill.
legacySwallowsKill :: IO ()
legacySwallowsKill = do
    steps <- routine try
    expect "legacy steps after kill" ["entry failed", "follow-up order"] steps

killStopsLaterSteps :: IO ()
killStopsLaterSteps = do
    steps <- routine trySync
    expect "trySync steps after kill" [] steps

synchronousCaptured :: IO ()
synchronousCaptured = do
    r <- trySync (void (throwIO (ErrorCall "venue rejected") :: IO ()))
    case r of
        Left ex -> expect "synchronous is not async" False (isAsyncException ex)
        Right () -> ioError (userError "synchronous failure was not captured")
    ok <- trySync (pure (7 :: Int))
    expect "success passes through" (Right 7) (either (const (Left ())) Right ok)

timeoutPropagates :: IO ()
timeoutPropagates = do
    r <- timeout 50000 (trySync (threadDelay 2000000))
    expect "timeout fires through trySync" True (isNothing r)
