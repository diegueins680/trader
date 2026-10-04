module Trader.App.AsyncJobAdmission (
    JobSlots,
    newJobSlots,
    runningJobSlots,
    startBoundedJob,
    admitSlot,
    releaseSlot,
) where

import Control.Concurrent (ThreadId, forkIOWithUnmask)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, readMVar)
import Control.Exception (finally, mask, onException)
import Control.Monad (when)
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)

-- Only this module can mutate the counter; release has no blocking lock wait.
data JobSlots = JobSlots !Int !(IORef Int)

newJobSlots :: Int -> IO JobSlots
newJobSlots limit = JobSlots (max 1 limit) <$> newIORef 0

runningJobSlots :: JobSlots -> IO Int
runningJobSlots (JobSlots _ running) = readIORef running

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
startBoundedJob :: JobSlots -> IO a -> (a -> IO ()) -> (ThreadId -> a -> IO ()) -> IO (Either Int a)
startBoundedJob (JobSlots limit running) prepare execute publish = mask $ \restore -> do
    admission <- atomicModifyIORef' running $ \current ->
        let (next, accepted) = admitSlot limit current
         in (next, if accepted then Right () else Left current)
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
    release = atomicModifyIORef' running (\current -> (releaseSlot current, ()))
