{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.GracefulShutdown (
    gracefulShutdownSuite,
    workerRegistrySuite,
) where

import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, readMVar, takeMVar)
import Control.Exception (MaskingState (Unmasked), finally, getMaskingState, mask_, uninterruptibleMask_)
import Control.Monad (forM, forM_, void)
import Data.IORef (newIORef, readIORef, writeIORef)
import Data.Maybe (isJust, isNothing)
import GHC.Clock (getMonotonicTimeNSec)
import System.Timeout (timeout)

import Trader.App.GracefulShutdown (
    ShutdownPhase (..),
    beginDrain,
    forkSupervisedWorker,
    isDraining,
    newDrainController,
    newWorkerRegistry,
    runCleanupStepBounded,
    runShutdownStep,
    shouldRejectDuringDrain,
    shutdownBudget,
    shutdownRemainingUs,
    stopSupervisedWorkersBounded,
    supervisedWorkerCount,
 )

gracefulShutdownSuite :: [(String, IO ())]
gracefulShutdownSuite =
    [ ("drain transition is idempotent", testDrainTransition)
    , ("drain rejects new work but preserves polling and cancellation", testDrainRequestPolicy)
    , ("supervised workers are tracked and stopped", testSupervisedWorkersStop)
    , ("cleanup deadline survives an uninterruptible action", testCleanupDeadline)
    , ("shutdown budgets use bounded exact arithmetic", testShutdownBudget)
    , ("expired shutdown does not dispatch cleanup", testExpiredShutdown)
    , ("shutdown exceptions remain failures", testShutdownException)
    , ("shutdown timeout does not claim acknowledgement", testShutdownTimeout)
    ]

testShutdownBudget :: IO ()
testShutdownBudget = do
    let budget = shutdownBudget 1000000000 20
    expectEq "wall clock cannot enter budget" 20000000 (shutdownRemainingUs budget FinalCleanup 1000000000)
    expectEq "reserve" 18000000 (shutdownRemainingUs budget WorkCleanup 1000000000)
    expectEq "clock regression fails closed" 0 (shutdownRemainingUs budget FinalCleanup 999999999)
    expectEq "expired fails closed" 0 (shutdownRemainingUs budget FinalCleanup 21000000000)
    expectEq "fractional microsecond fails closed" 0 (shutdownRemainingUs budget FinalCleanup 20999999999)
    expectEq "huge timeout saturates before conversion" maxBound (shutdownRemainingUs (shutdownBudget 0 maxBound) FinalCleanup 0)
    expectEq "negative timeout fails closed" 0 (shutdownRemainingUs (shutdownBudget 0 minBound) FinalCleanup 0)

testExpiredShutdown :: IO ()
testExpiredShutdown = do
    ran <- newIORef False
    now <- toInteger <$> getMonotonicTimeNSec
    ok <- runShutdownStep (shutdownBudget now 0) FinalCleanup (\_ -> writeIORef ran True)
    expectEq "expired outcome" False ok
    expectEq "no expired action" False =<< readIORef ran

testShutdownException :: IO ()
testShutdownException = do
    now <- toInteger <$> getMonotonicTimeNSec
    ok <- runShutdownStep (shutdownBudget now 1) FinalCleanup (\_ -> ioError (userError "fixture failure"))
    expectEq "exception outcome" False ok

testShutdownTimeout :: IO ()
testShutdownTimeout = do
    now <- toInteger <$> getMonotonicTimeNSec
    ok <- runShutdownStep (shutdownBudget (now - 800000000) 1) FinalCleanup (\_ -> uninterruptibleMask_ (threadDelay 500000))
    expectEq "uninterruptible action is unacknowledged" False ok

testDrainTransition :: IO ()
testDrainTransition = do
    drain <- newDrainController
    expectEq "starts ready" False =<< isDraining drain
    expectEq "first transition owns drain" True =<< beginDrain drain
    expectEq "reports draining" True =<< isDraining drain
    expectEq "second transition is idempotent" False =<< beginDrain drain

testDrainRequestPolicy :: IO ()
testDrainRequestPolicy = do
    expectEq "reject direct trade" True (shouldRejectDuringDrain "POST" ["trade"])
    expectEq "reject async trade" True (shouldRejectDuringDrain "POST" ["api", "trade", "async"])
    expectEq "reject bot start" True (shouldRejectDuringDrain "POST" ["bot", "start"])
    expectEq "preserve async poll" False (shouldRejectDuringDrain "GET" ["trade", "async", "job-1"])
    expectEq "preserve async cancel" False (shouldRejectDuringDrain "POST" ["trade", "async", "job-1", "cancel"])
    expectEq "preserve bot stop" False (shouldRejectDuringDrain "POST" ["bot", "stop"])

testSupervisedWorkersStop :: IO ()
testSupervisedWorkersStop = do
    workers <- newWorkerRegistry
    _ <- forkSupervisedWorker workers "test-worker" (threadDelay 10000000)
    expectEq "worker registered" 1 =<< supervisedWorkerCount workers
    expectEq "worker stopped before deadline" True =<< stopSupervisedWorkersBounded 500000 workers
    expectEq "registry emptied" 0 =<< supervisedWorkerCount workers

testCleanupDeadline :: IO ()
testCleanupDeadline = do
    completed <-
        runCleanupStepBounded
            20000
            (uninterruptibleMask_ (threadDelay 200000))
    expectEq "uninterruptible cleanup times out" False completed

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)

workerRegistrySuite :: [(String, IO ())]
workerRegistrySuite =
    [ ("unfinished worker remains visible across retries", testRetainedWorker)
    , ("closed registry rejects new workers", testClosedRegistry)
    , ("stop requires finalizer completion and survives caller cancellation", testFinalizerAcknowledgement)
    , ("supervised action explicitly unmasks exceptions", testWorkerUnmask)
    , ("concurrent start and stop preserve registry invariants", testWorkerRaces)
    ]

awaitWorkerFixture :: String -> IO a -> IO a
awaitWorkerFixture label action = do
    result <- timeout 3000000 action
    case result of
        Just value -> pure value
        Nothing -> ioError (userError ("worker fixture timed out: " ++ label))

testRetainedWorker :: IO ()
testRetainedWorker = do
    workers <- newWorkerRegistry
    entered <- newEmptyMVar
    release <- newEmptyMVar
    accepted <- forkSupervisedWorker workers "blocked-fixture" (uninterruptibleMask_ (putMVar entered () >> takeMVar release))
    expectEq "initial start accepted" True (isJust accepted)
    awaitWorkerFixture "worker entered" (takeMVar entered)
    ( do
            expectEq "first stop is incomplete" False =<< stopSupervisedWorkersBounded 20000 workers
            expectEq "unfinished worker retained" 1 =<< supervisedWorkerCount workers
            expectEq "retry cannot erase incomplete stop" False =<< stopSupervisedWorkersBounded 20000 workers
            expectEq "retry retains identity" 1 =<< supervisedWorkerCount workers
        )
        `finally` putMVar release ()
    expectEq "completion acknowledged after release" True =<< stopSupervisedWorkersBounded 1000000 workers
    expectEq "completed count" 0 =<< supervisedWorkerCount workers
    expectEq "completed retry is idempotent" True =<< stopSupervisedWorkersBounded 1000000 workers

testClosedRegistry :: IO ()
testClosedRegistry = do
    workers <- newWorkerRegistry
    ran <- newIORef False
    expectEq "empty stop" True =<< stopSupervisedWorkersBounded 1000000 workers
    accepted <- forkSupervisedWorker workers "forbidden" (writeIORef ran True)
    expectEq "closed admission absent" True (isNothing accepted)
    expectEq "closed action not executed" False =<< readIORef ran
    expectEq "closed registry remains empty" 0 =<< supervisedWorkerCount workers

testFinalizerAcknowledgement :: IO ()
testFinalizerAcknowledgement = do
    workers <- newWorkerRegistry
    entered <- newEmptyMVar
    finalizing <- newEmptyMVar
    release <- newEmptyMVar
    let action =
            (putMVar entered () >> threadDelay 10000000)
                `finally` uninterruptibleMask_ (putMVar finalizing () >> takeMVar release)
    _ <- forkSupervisedWorker workers "finalizing-fixture" action
    awaitWorkerFixture "action entered" (takeMVar entered)
    callerDone <- newEmptyMVar
    caller <- forkIO (void (stopSupervisedWorkersBounded 2000000 workers) `finally` putMVar callerDone ())
    awaitWorkerFixture "cancellation reached finalizer" (takeMVar finalizing)
    ( do
            expectEq "delivery leaves finalizer outstanding" 1 =<< supervisedWorkerCount workers
            killThread caller
            awaitWorkerFixture "stop caller interrupted" (takeMVar callerDone)
            expectEq "retry does not acknowledge a blocked finalizer" False =<< stopSupervisedWorkersBounded 20000 workers
        )
        `finally` putMVar release ()
    expectEq "retry recovers after caller cancellation" True =<< stopSupervisedWorkersBounded 1000000 workers
    expectEq "finalized count" 0 =<< supervisedWorkerCount workers

testWorkerUnmask :: IO ()
testWorkerUnmask = do
    workers <- newWorkerRegistry
    observed <- newEmptyMVar
    _ <- mask_ (forkSupervisedWorker workers "mask-fixture" (getMaskingState >>= putMVar observed >> threadDelay 10000000))
    state <- awaitWorkerFixture "masking state" (takeMVar observed)
    expectEq "callback runs unmasked" Unmasked state
    expectEq "unmasked worker stops" True =<< stopSupervisedWorkersBounded 1000000 workers

testWorkerRaces :: IO ()
testWorkerRaces =
    forM_ seeds $ \seed -> do
        workers <- newWorkerRegistry
        gate <- newEmptyMVar
        let launch delay action = do
                done <- newEmptyMVar
                _ <- forkIO (readMVar gate >> threadDelay delay >> action >>= putMVar done)
                pure done
            delays = [fromInteger ((seed `div` divisor) `mod` 3) * 1000 | divisor <- [1, 3, 9, 27]]
        starts <- forM (take 2 delays) $ \delay -> launch delay (forkSupervisedWorker workers "race-fixture" (threadDelay 10000000))
        stops <- forM (drop 2 delays) $ \delay -> launch delay (stopSupervisedWorkersBounded 1000000 workers)
        putMVar gate ()
        _ <- awaitWorkerFixture "start racers" (mapM takeMVar starts)
        outcomes <- awaitWorkerFixture "stop racers" (mapM takeMVar stops)
        expectEq "both stops acknowledged" [True, True] outcomes
        expectEq "race leaves no unfinished worker" 0 =<< supervisedWorkerCount workers
        rejected <- forkSupervisedWorker workers "late-race" (pure ())
        expectEq "race cannot reopen" True (isNothing rejected)
  where
    seeds :: [Integer]
    seeds = take 32 (tail (iterate (\value -> (1664525 * value + 1013904223) `mod` 4294967296) 20261004))
