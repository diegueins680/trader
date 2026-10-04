module Trader.App.BacktestGate (
    BacktestGate,
    BacktestFailure (..),
    newBacktestGate,
    runBacktestWithGate,
    runBacktestWithGateWait,
    btTimeoutSec,
    backtestRunningCount,
    timeoutMicroseconds,
) where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeAsyncException, SomeException, finally, fromException, mask, tryJust)
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission (admitSlot, releaseSlot)

data BacktestGate = BacktestGate
    { btRunning :: !(IORef Int)
    , btMaxRunning :: !Int
    , btTimeoutSec :: !Int
    }

data BacktestFailure
    = BacktestBusy
    | BacktestTimedOut
    | BacktestException SomeException
    deriving (Show)

newBacktestGate :: Int -> Int -> IO BacktestGate
newBacktestGate maxRunning timeoutSec = do
    running <- newIORef 0
    pure BacktestGate{btRunning = running, btMaxRunning = max 1 maxRunning, btTimeoutSec = max 1 timeoutSec}

backtestRunningCount :: BacktestGate -> IO Int
backtestRunningCount = readIORef . btRunning

-- System.Timeout treats negative microseconds as unlimited, so multiply exactly
-- and saturate before converting back to the machine representation.
timeoutMicroseconds :: Int -> Int
timeoutMicroseconds seconds = fromInteger (min (toInteger (maxBound :: Int)) (max 1 (toInteger seconds) * 1000000))

-- Keep timer/cancellation exceptions outside the structured domain-error path.
synchronousFailure :: SomeException -> Maybe SomeException
synchronousFailure ex =
    case fromException ex :: Maybe SomeAsyncException of
        Just _ -> Nothing
        Nothing -> Just ex

runBacktestWithGate :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGate gate action = mask $ \restore -> do
    acquired <- atomicModifyIORef' (btRunning gate) (admitSlot (btMaxRunning gate))
    if not acquired
        then pure (Left BacktestBusy)
        else restore runTimed `finally` release
  where
    release = atomicModifyIORef' (btRunning gate) (\count -> (releaseSlot count, ()))
    runTimed = do
        result <- tryJust synchronousFailure (timeout (timeoutMicroseconds (btTimeoutSec gate)) action)
        pure $ case result of
            Left ex -> Left (BacktestException ex)
            Right Nothing -> Left BacktestTimedOut
            Right (Just value) -> Right value

-- Waiting owns no slot. Preserve the existing one-second retry and execution-only
-- timer; cancellation propagates, and only queue contention is retried.
runBacktestWithGateWait :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGateWait gate action = do
    result <- runBacktestWithGate gate action
    case result of
        Left BacktestBusy -> threadDelay 1000000 >> runBacktestWithGateWait gate action
        _ -> pure result
