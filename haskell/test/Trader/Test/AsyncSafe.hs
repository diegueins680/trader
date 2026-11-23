module Trader.Test.AsyncSafe (asyncSafeSuite) where

import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (ErrorCall (..), SomeException, finally, throwIO, try)
import Control.Monad (void)
import Data.Either (fromRight)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.Maybe (isNothing)
import System.Timeout (timeout)
import Trader.App.AsyncSafe (catchSync, isAsyncException, trySync)
import Trader.Http (withTotalDeadline)

asyncSafeSuite :: [(String, IO ())]
asyncSafeSuite =
    [ ("legacy try swallows killThread and keeps acting", legacySwallowsKill)
    , ("trySync rethrows killThread so no later step runs", killStopsLaterSteps)
    , ("trySync still captures synchronous failures", synchronousCaptured)
    , ("trySync does not defeat System.Timeout.timeout", timeoutPropagates)
    , ("catchSync rethrows killThread and still handles synchronous failures", catchSyncBehaves)
    , ("withTotalDeadline turns a stall inside trySync into a deadline error", totalDeadlineBehaves)
    ]

{- | A stalled request must come back as an error at its deadline. The stall runs
inside 'trySync', as an S3 request does, so the timeout's asynchronous exception
must not be captured as an ordinary failure.
-}
totalDeadlineBehaves :: IO ()
totalDeadlineBehaves = do
    fast <- withTotalDeadline "test" 1000000 (pure (Right (42 :: Int)))
    expect "a call inside the deadline passes through" (Right 42) fast
    failed <- withTotalDeadline "test" 1000000 (pure (Left "status 500") :: IO (Either String Int))
    expect "an error inside the deadline passes through" (Left "status 500") failed
    stalled <-
        timeout 5000000 $
            withTotalDeadline "test" 200000 $ do
                captured <- trySync (threadDelay 30000000) :: IO (Either SomeException ())
                pure (either (const (Left "captured")) (const (Right ())) captured)
    expect "a stall returns as a deadline error" (Just (Left "test exceeded its 0 s deadline")) stalled

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

-- An order routine shaped like the worker: step 1 is an exchange call, step 2 a follow-up order.
routine :: (IO String -> IO (Either SomeException String)) -> IO [String]
routine capture = do
    steps <- newIORef []
    done <- newEmptyMVar
    ready <- newEmptyMVar
    tid <-
        forkIO $
            ( do
                r <- capture (putMVar ready () >> threadDelay 5000000 >> pure "entry")
                modifyIORef' steps (fromRight "entry failed" r :)
                modifyIORef' steps ("follow-up order" :)
            )
                `finally` putMVar done ()
    -- Kill only once the worker is inside its exchange call, never before it.
    takeMVar ready
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
    ok <- trySync (pure (7 :: Int)) :: IO (Either SomeException Int)
    expect "success passes through" (Right 7) (either (const (Left ())) Right ok)

timeoutPropagates :: IO ()
timeoutPropagates = do
    r <- timeout 50000 (trySync (threadDelay 2000000) :: IO (Either SomeException ()))
    expect "timeout fires through trySync" True (isNothing r)

catchSyncBehaves :: IO ()
catchSyncBehaves = do
    steps <- routine (\act -> (Right <$> act) `catchSync` (pure . Left))
    expect "catchSync steps after kill" [] steps
    handled <- throwIO (ErrorCall "venue rejected") `catchSync` \(ErrorCall msg) -> pure msg
    expect "catchSync handles synchronous" "venue rejected" handled
