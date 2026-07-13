{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.GracefulShutdown (
    gracefulShutdownSuite,
) where

import Control.Concurrent (threadDelay)
import Control.Exception (uninterruptibleMask_)

import Trader.App.GracefulShutdown (
    beginDrain,
    forkSupervisedWorker,
    isDraining,
    newDrainController,
    newWorkerRegistry,
    runCleanupStepBounded,
    shouldRejectDuringDrain,
    stopSupervisedWorkersBounded,
    supervisedWorkerCount,
 )

gracefulShutdownSuite :: [(String, IO ())]
gracefulShutdownSuite =
    [ ("drain transition is idempotent", testDrainTransition)
    , ("drain rejects new work but preserves polling and cancellation", testDrainRequestPolicy)
    , ("supervised workers are tracked and stopped", testSupervisedWorkersStop)
    , ("cleanup deadline survives an uninterruptible action", testCleanupDeadline)
    ]

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
