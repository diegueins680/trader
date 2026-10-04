{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.GracefulShutdown (
    gracefulShutdownSuite,
) where

import Control.Concurrent (threadDelay)
import Control.Exception (uninterruptibleMask_)
import Data.IORef (newIORef, readIORef, writeIORef)
import GHC.Clock (getMonotonicTimeNSec)

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
