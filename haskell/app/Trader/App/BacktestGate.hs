module Trader.App.BacktestGate (
    BacktestGate,
    BacktestFailure (..),
    newBacktestGate,
    newBacktestGateWithDrain,
    runBacktestWithGate,
    runBacktestWithGateWait,
    btTimeoutSec,
    backtestRunningCount,
    backtestWaitingCount,
    timeoutMicroseconds,
) where

import Control.Concurrent.STM (TVar, atomically, modifyTVar', newTVarIO, readTVar, readTVarIO, retry, writeTVar)
import Control.Exception (SomeAsyncException, SomeException, finally, fromException, mask, onException, tryJust)
import Control.Monad (when)
import Data.Bifunctor (second)
import qualified Data.Sequence as Seq
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission (admitSlot, releaseSlot)
import Trader.App.GracefulShutdown (DrainController, newDrainController, unlessDraining)

data BacktestGate = BacktestGate
    { btRunning :: !(TVar Int)
    , btMaxRunning :: !Int
    , btTimeoutSec :: !Int
    , btWaiters :: !(TVar (Integer, Seq.Seq Integer))
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
    waiters <- newTVarIO (0, Seq.empty)
    pure BacktestGate{btRunning = running, btWaiters = waiters, btMaxRunning = max 1 maxRunning, btTimeoutSec = max 1 timeoutSec, btDrain = drain}

backtestRunningCount :: BacktestGate -> IO Int
backtestRunningCount = readTVarIO . btRunning

backtestWaitingCount :: BacktestGate -> IO Int
backtestWaitingCount gate = Seq.length . snd <$> readTVarIO (btWaiters gate)

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

-- Immediate callers cannot take a vacancy already owed to a queued waiter.
runBacktestWithGate :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGate gate action = mask $ \restore -> do
    acquired <- atomically $ unlessDraining (btDrain gate) $ do
        (_, queue) <- readTVar (btWaiters gate)
        current <- readTVar (btRunning gate)
        let (next, accepted) = if Seq.null queue then admitSlot (btMaxRunning gate) current else (current, False)
        when accepted (writeTVar (btRunning gate) next)
        pure accepted
    case acquired of
        Nothing -> pure (Left BacktestDraining)
        Just False -> pure (Left BacktestBusy)
        Just True -> restore (runTimedBacktest gate action) `finally` releaseBacktest gate

releaseBacktest :: BacktestGate -> IO ()
releaseBacktest gate = atomically (modifyTVar' (btRunning gate) releaseSlot)

runTimedBacktest :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runTimedBacktest gate action = do
    result <- tryJust synchronousFailure (timeout (timeoutMicroseconds (btTimeoutSec gate)) action)
    pure $ case result of
        Left ex -> Left (BacktestException ex)
        Right Nothing -> Left BacktestTimedOut
        Right (Just value) -> Right value

-- Registration and admission use separate masked transactions: a waiting caller
-- owns only its ticket. Cancellation removes that ticket; successful admission
-- atomically exchanges it for a slot, whose finalizer is installed before unmasking.
runBacktestWithGateWait :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGateWait gate action = mask $ \restore -> do
    registered <- atomically $ unlessDraining (btDrain gate) $ do
        (ticket, queue) <- readTVar (btWaiters gate)
        let next = ticket + 1
        next `seq` writeTVar (btWaiters gate) (next, queue Seq.|> ticket)
        pure ticket
    case registered of
        Nothing -> pure (Left BacktestDraining)
        Just ticket -> do
            let removeTicket = modifyTVar' (btWaiters gate) (second (Seq.filter (/= ticket)))
                acquire = do
                    guarded <- unlessDraining (btDrain gate) $ do
                        (nextTicket, queue) <- readTVar (btWaiters gate)
                        current <- readTVar (btRunning gate)
                        case Seq.viewl queue of
                            first Seq.:< rest | first == ticket -> do
                                let (next, accepted) = admitSlot (btMaxRunning gate) current
                                if accepted
                                    then do
                                        writeTVar (btRunning gate) next
                                        writeTVar (btWaiters gate) (nextTicket, rest)
                                    else retry
                            _ -> retry
                    case guarded of
                        Nothing -> removeTicket >> pure False
                        Just () -> pure True
            acquired <- atomically acquire `onException` atomically removeTicket
            if acquired
                then restore (runTimedBacktest gate action) `finally` releaseBacktest gate
                else pure (Left BacktestDraining)
