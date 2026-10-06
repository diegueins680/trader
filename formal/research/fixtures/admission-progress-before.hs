module Trader.App.BacktestGate (
    BacktestGate,
    BacktestFailure (..),
    newBacktestGate,
    newBacktestGateWithDrain,
    runBacktestWithGate,
    runBacktestWithGateWait,
    btTimeoutSec,
    backtestRunningCount,
    timeoutMicroseconds,
) where

import Control.Concurrent (threadDelay)
import Control.Concurrent.STM (TVar, atomically, modifyTVar', newTVarIO, readTVar, readTVarIO, writeTVar)
import Control.Exception (SomeAsyncException, SomeException, finally, fromException, mask, tryJust)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission (admitSlot, releaseSlot)
import Trader.App.GracefulShutdown (DrainController, newDrainController, unlessDraining)

data BacktestGate = BacktestGate
    { btRunning :: !(TVar Int)
    , btMaxRunning :: !Int
    , btTimeoutSec :: !Int
    , btDrain :: !DrainController
    }

data BacktestFailure
    = BacktestBusy
    | BacktestDraining
    | BacktestTimedOut
    | BacktestException SomeException
    deriving (Show)

newBacktestGate :: Int -> Int -> IO BacktestGate
newBacktestGate maxRunning timeoutSec = newDrainController >>= \drain -> newBacktestGateWithDrain drain maxRunning timeoutSec

newBacktestGateWithDrain :: DrainController -> Int -> Int -> IO BacktestGate
newBacktestGateWithDrain drain maxRunning timeoutSec = do
    running <- newTVarIO 0
    pure BacktestGate{btRunning = running, btMaxRunning = max 1 maxRunning, btTimeoutSec = max 1 timeoutSec, btDrain = drain}

backtestRunningCount :: BacktestGate -> IO Int
backtestRunningCount = readTVarIO . btRunning

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
    acquired <- atomically $ unlessDraining (btDrain gate) $ do
        current <- readTVar (btRunning gate)
        let (next, accepted) = admitSlot (btMaxRunning gate) current
        writeTVar (btRunning gate) next
        pure accepted
    case acquired of
        Nothing -> pure (Left BacktestDraining)
        Just False -> pure (Left BacktestBusy)
        Just True -> restore runTimed `finally` release
  where
    release = atomically (modifyTVar' (btRunning gate) releaseSlot)
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
