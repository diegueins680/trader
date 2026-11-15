module Trader.App.AsyncJobAdmission (
    JobSlots,
    JobAdmissionFailure (..),
    closeJobSlots,
    waitJobSlots,
    reserveOpenSlot,
    newJobSlots,
    newJobSlotsWithDrain,
    runningJobSlots,
    startBoundedJob,
    admitSlot,
    releaseSlot,
) where

import Control.Concurrent (ThreadId, forkIOWithUnmask)
import Control.Concurrent.MVar (MVar, newEmptyMVar, putMVar, readMVar, tryPutMVar)
import Control.Concurrent.STM (TVar, atomically, newTVarIO, readTVar, readTVarIO, writeTVar)
import Control.Exception (finally, mask, mask_, onException)
import Control.Monad (void, when)
import Data.Maybe (fromMaybe)
import Trader.App.GracefulShutdown (DrainController, newDrainController, unlessDraining)

-- Closure and count share one atomic boundary. The private completion cell is
-- monotonic and is signalled only after sealing and releasing every reservation.
data JobSlots = JobSlots !Int !(TVar (Int, Bool)) !(MVar ()) !DrainController

data JobAdmissionFailure = JobQueueFull !Int | JobQueueClosed
    deriving (Eq, Show)

newJobSlots :: Int -> IO JobSlots
newJobSlots limit = newDrainController >>= \drain -> newJobSlotsWithDrain drain limit

newJobSlotsWithDrain :: DrainController -> Int -> IO JobSlots
newJobSlotsWithDrain drain limit = JobSlots (max 1 limit) <$> newTVarIO (0, False) <*> newEmptyMVar <*> pure drain

runningJobSlots :: JobSlots -> IO Int
runningJobSlots (JobSlots _ state _ _) = fst <$> readTVarIO state

closeJobSlots :: JobSlots -> IO ()
closeJobSlots (JobSlots _ state completed _) = mask_ $ do
    idle <- atomically $ do
        (count, _) <- readTVar state
        writeTVar state (count, True)
        pure (count == 0)
    when idle (void (tryPutMVar completed ()))

waitJobSlots :: JobSlots -> IO ()
waitJobSlots (JobSlots _ _ completed _) = readMVar completed

reserveOpenSlot :: Int -> (Int, Bool) -> ((Int, Bool), Either JobAdmissionFailure ())
reserveOpenSlot limit (current, closed)
    | closed = ((current, True), Left JobQueueClosed)
    | otherwise =
        let (next, accepted) = admitSlot limit current
         in ((next, False), if accepted then Right () else Left (JobQueueFull current))

admitSlot :: Int -> Int -> (Int, Bool)
admitSlot limit current
    | current < 0 || limit <= 0 || current >= limit = (current, False)
    | otherwise = (current + 1, True)

releaseSlot :: Int -> Int
releaseSlot current
    | current > 0 = current - 1
    | otherwise = 0

-- The parent owns the reservation until fork succeeds; thereafter only the child
-- releases it. Publication must be masked and exception-safe at its mutation
-- boundary. Failure sends False to the private gate instead of killing a callback.
startBoundedJob :: JobSlots -> IO a -> (a -> IO ()) -> (ThreadId -> a -> IO ()) -> IO (Either JobAdmissionFailure a)
startBoundedJob (JobSlots limit state completed drain) prepare execute publish = mask $ \restore -> do
    admission <- atomically $ do
        guarded <- unlessDraining drain $ do
            current <- readTVar state
            let (next, result) = reserveOpenSlot limit current
            writeTVar state next
            pure result
        pure (fromMaybe (Left JobQueueClosed) guarded)
    case admission of
        Left current -> pure (Left current)
        Right () -> do
            (payload, gate) <-
                ( do
                        payload <- restore prepare
                        gate <- newEmptyMVar
                        pure (payload, gate)
                    )
                    `onException` release
            tid <-
                forkIOWithUnmask
                    ( \unmask ->
                        (readMVar gate >>= \accepted -> when accepted (unmask (execute payload)))
                            `finally` release
                    )
                    `onException` release
            (publish tid payload >> putMVar gate True)
                `onException` putMVar gate False
            pure (Right payload)
  where
    release = mask_ $ do
        idle <- atomically $ do
            (current, closed) <- readTVar state
            let next = releaseSlot current
            writeTVar state (next, closed)
            pure (closed && next == 0)
        when idle (void (tryPutMVar completed ()))
