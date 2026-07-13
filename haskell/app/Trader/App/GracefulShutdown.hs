{-# LANGUAGE OverloadedStrings #-}

module Trader.App.GracefulShutdown (
    DrainController,
    WorkerRegistry,
    beginDrain,
    forkSupervisedWorker,
    isDraining,
    newDrainController,
    newWorkerRegistry,
    runCleanupStepBounded,
    shouldRejectDuringDrain,
    stopSupervisedWorkersBounded,
    stopThreadIdsBounded,
    supervisedWorkerCount,
) where

import Control.Concurrent (ThreadId, forkIO, killThread, myThreadId, threadDelay)
import Control.Concurrent.MVar (MVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, swapMVar, takeMVar)
import Control.Exception (AsyncException, SomeException, displayException, finally, fromException, throwIO, try)
import Control.Monad (forM, void)
import Data.ByteString (ByteString)
import Data.IORef (IORef, atomicModifyIORef', newIORef, readIORef)
import Data.Text (Text)
import System.IO (hPutStrLn, stderr)
import System.Timeout (timeout)

newtype DrainController = DrainController (IORef Bool)

newtype WorkerRegistry = WorkerRegistry (MVar [(String, ThreadId)])

newDrainController :: IO DrainController
newDrainController = DrainController <$> newIORef False

beginDrain :: DrainController -> IO Bool
beginDrain (DrainController ref) =
    atomicModifyIORef' ref $ \draining ->
        if draining
            then (True, False)
            else (True, True)

isDraining :: DrainController -> IO Bool
isDraining (DrainController ref) = readIORef ref

-- Keep polling and cancellation available while draining, but reject endpoints
-- that can launch expensive compute, orders, bots, or optimizer processes.
shouldRejectDuringDrain :: ByteString -> [Text] -> Bool
shouldRejectDuringDrain method path =
    method == "POST" && normalizedPath `elem` workStartingPaths
  where
    normalizedPath =
        case path of
            "api" : rest -> rest
            _ -> path
    workStartingPaths =
        [ ["signal"]
        , ["signal", "async"]
        , ["trade"]
        , ["trade", "async"]
        , ["backtest"]
        , ["backtest", "async"]
        , ["binance", "positions", "close"]
        , ["bot", "start"]
        , ["optimizer", "run"]
        ]

newWorkerRegistry :: IO WorkerRegistry
newWorkerRegistry = WorkerRegistry <$> newMVar []

supervisedWorkerCount :: WorkerRegistry -> IO Int
supervisedWorkerCount (WorkerRegistry workers) = length <$> readMVar workers

forkSupervisedWorker :: WorkerRegistry -> String -> IO () -> IO ThreadId
forkSupervisedWorker (WorkerRegistry workers) name action = do
    ready <- newEmptyMVar
    tid <-
        forkIO $ do
            takeMVar ready
            current <- myThreadId
            loop 0 `finally` modifyMVar_ workers (pure . filter ((/= current) . snd))
    modifyMVar_ workers (pure . ((name, tid) :))
    putMVar ready ()
    pure tid
  where
    restartDelayUs = 1000000

    loop :: Int -> IO ()
    loop restartCount = do
        result <- try action
        case result of
            Right () -> do
                hPutStrLn stderr (workerMessage "exited" restartCount Nothing)
                threadDelay restartDelayUs
                loop (restartCount + 1)
            Left ex ->
                case fromException ex :: Maybe AsyncException of
                    Just asyncEx -> throwIO asyncEx
                    Nothing -> do
                        hPutStrLn stderr (workerMessage "crashed" restartCount (Just ex))
                        threadDelay restartDelayUs
                        loop (restartCount + 1)

    workerMessage :: String -> Int -> Maybe SomeException -> String
    workerMessage outcome restartCount mException =
        "Background worker '"
            ++ name
            ++ "' "
            ++ outcome
            ++ maybe "" ((": " ++) . displayException) mException
            ++ "; restarting (count="
            ++ show (restartCount + 1)
            ++ ")."

-- Run cleanup in a separate thread so an uninterruptible foreign call cannot
-- make the caller exceed its deadline. A timed-out cleanup thread receives a
-- best-effort asynchronous cancellation without waiting for delivery.
runCleanupStepBounded :: Int -> IO () -> IO Bool
runCleanupStepBounded timeoutUs action = do
    done <- newEmptyMVar
    tid <-
        forkIO $ do
            result <- try action :: IO (Either SomeException ())
            putMVar done result
    result <- timeout (max 1 timeoutUs) (takeMVar done)
    case result of
        Just (Right ()) -> pure True
        Just (Left _) -> pure False
        Nothing -> do
            void (forkIO (killThread tid))
            pure False

stopThreadIdsBounded :: Int -> [ThreadId] -> IO Bool
stopThreadIdsBounded timeoutUs tids = do
    completions <-
        forM tids $ \tid -> do
            done <- newEmptyMVar
            void $ forkIO (killThread tid `finally` putMVar done ())
            pure done
    isJustResult <- timeout (max 1 timeoutUs) (mapM_ takeMVar completions)
    pure $ case isJustResult of
        Just () -> True
        Nothing -> False

stopSupervisedWorkersBounded :: Int -> WorkerRegistry -> IO Bool
stopSupervisedWorkersBounded timeoutUs (WorkerRegistry workers) = do
    registered <- swapMVar workers []
    stopThreadIdsBounded timeoutUs (map snd registered)
