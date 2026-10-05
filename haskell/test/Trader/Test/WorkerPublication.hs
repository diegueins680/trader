{-# LANGUAGE DeriveDataTypeable #-}

module Trader.Test.WorkerPublication (workerPublicationSuite) where

import Control.Concurrent (ThreadId, forkIO, myThreadId, threadDelay)
import Control.Concurrent.MVar (modifyMVar, newEmptyMVar, newMVar, putMVar, readMVar, takeMVar)
import Control.Exception (AsyncException (ThreadKilled), Exception, MaskingState (Unmasked), SomeException, getMaskingState, throw, throwIO, throwTo, try)
import Control.Monad (forM, forM_)
import Data.IORef (newIORef, readIORef, writeIORef)
import Data.Typeable (Typeable)
import GHC.Conc (ThreadStatus (..), threadStatus)
import System.Timeout (timeout)
import Trader.App.WorkerPublication (WorkerPlan (..), publishWorker)

workerPublicationSuite :: [(String, IO ())]
workerPublicationSuite =
    [ ("legacy fork can execute before publication rollback", legacyCounterexample)
    , ("worker sees its committed identity and runs unmasked", publicationPrecedesExecution)
    , ("retained state never forks a worker", retainState)
    , ("failed publication releases a rejecting gate", failedPublication)
    , ("cancelled preparation restores the old map", cancelledPreparation)
    , ("concurrent duplicate starts preserve a single registration", concurrentStarts)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

await :: IO a -> IO a
await action = timeout 3000000 action >>= maybe (ioError (userError "worker publication fixture timed out")) pure

awaitFinished :: ThreadId -> IO ()
awaitFinished tid = await loop
  where
    loop = do
        status <- threadStatus tid
        case status of
            ThreadFinished -> pure ()
            ThreadDied -> pure ()
            _ -> threadDelay 1000 >> loop

data PublicationFailure = PublicationFailure ThreadId
    deriving (Show, Typeable)

instance Exception PublicationFailure

legacyCounterexample :: IO ()
legacyCounterexample = do
    cell <- newMVar (Nothing :: Maybe ThreadId)
    executed <- newEmptyMVar
    finish <- newEmptyMVar
    result <- try $ modifyMVar cell $ \_ -> do
        tid <- forkIO (putMVar executed () >> takeMVar finish)
        -- Deterministically schedule the legal fork-before-commit interleaving.
        await (takeMVar executed)
        throwIO (PublicationFailure tid)
    case result :: Either PublicationFailure () of
        Left (PublicationFailure tid) -> do
            expect "old map restored despite executed child" Nothing =<< readMVar cell
            putMVar finish ()
            awaitFinished tid
        Right () -> ioError (userError "legacy failure absent")

publicationPrecedesExecution :: IO ()
publicationPrecedesExecution = do
    cell <- newMVar Nothing
    observed <- newEmptyMVar
    let action = do
            tid <- myThreadId
            owner <- readMVar cell
            masking <- getMaskingState
            putMVar observed (owner == Just tid, masking)
    tid <- publishWorker cell $ \_ -> pure (Launch action (\child -> (Just child, child)))
    expect "committed worker" (Just tid) =<< readMVar cell
    expect "publication and masking" (True, Unmasked) =<< await (takeMVar observed)
    awaitFinished tid

retainState :: IO ()
retainState = do
    cell <- newMVar (7 :: Int)
    result <- publishWorker cell (\_ -> pure (Retain False))
    expect "retained result" False result
    expect "retained map" 7 =<< readMVar cell

failedPublication :: IO ()
failedPublication = forM_ [0 :: Int, 1, 2] $ \cut -> do
    cell <- newMVar (Nothing :: Maybe ThreadId)
    ran <- newIORef False
    let publication tid =
            case cut of
                0 -> throw (PublicationFailure tid)
                1 -> (throw (PublicationFailure tid), ())
                _ -> (Just tid, throw (PublicationFailure tid))
    result <- try (publishWorker cell (\_ -> pure (Launch (writeIORef ran True) publication)))
    case result :: Either PublicationFailure () of
        Left (PublicationFailure tid) -> do
            awaitFinished tid
            expect "rollback" Nothing =<< readMVar cell
            expect "rejected child never executes" False =<< readIORef ran
        Right () -> ioError (userError "publication failure absent")

cancelledPreparation :: IO ()
cancelledPreparation = do
    cell <- newMVar (9 :: Int)
    entered <- newEmptyMVar
    block <- newEmptyMVar
    completed <- newEmptyMVar
    parent <- forkIO $ do
        result <- try (publishWorker cell (\_ -> putMVar entered () >> takeMVar block))
        putMVar completed (result :: Either SomeException ())
    await (takeMVar entered)
    throwTo parent ThreadKilled
    outcome <- await (takeMVar completed)
    case outcome of
        Left _ -> pure ()
        Right () -> ioError (userError "preparation cancellation swallowed")
    expect "cancelled map restored" 9 =<< readMVar cell
    awaitFinished parent

concurrentStarts :: IO ()
concurrentStarts = do
    cell <- newMVar (Nothing :: Maybe ThreadId)
    observed <- newEmptyMVar
    ready <- newEmptyMVar
    go <- newEmptyMVar
    results <- forM [1 :: Int, 2] $ \_ -> do
        completed <- newEmptyMVar
        _ <- forkIO $ do
            putMVar ready ()
            readMVar go
            result <- try $ publishWorker cell $ \state ->
                case state of
                    Just _ -> pure (Retain False)
                    Nothing -> pure (Launch (myThreadId >>= putMVar observed) (\tid -> (Just tid, True)))
            putMVar completed (result :: Either SomeException Bool)
        pure completed
    await (takeMVar ready)
    await (takeMVar ready)
    putMVar go ()
    outcomes <- mapM (await . takeMVar) results
    let accepted = [value | Right value <- outcomes]
    expect "both callers completed" 2 (length accepted)
    expect "one launch" 1 (length (filter id accepted))
    tid <- await (takeMVar observed)
    expect "single registered identity" (Just tid) =<< readMVar cell
    awaitFinished tid
